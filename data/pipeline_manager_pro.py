"""
Data Pipeline Manager Pro

Enterprise-grade data ingestion with:
1. Multi-source ingestion with priority
2. Quality control and validation
3. Conflict reconciliation with audit
4. Multi-format storage (TimescaleDB + Parquet)
5. Monitoring with Telegram alerts
"""

import os
import json
import hashlib
from datetime import datetime, timezone, timedelta, time as dt_time
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict
import threading
import queue

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text

import structlog

logger = structlog.get_logger()


# =============================================================================
# 1. INGESTION CONFIGURATION
# =============================================================================

class SourceRole(str, Enum):
    """Source roles in the pipeline."""
    PRIMARY = "primary"
    BACKUP = "backup"
    REALTIME = "realtime"


class AssetType(str, Enum):
    """Asset types."""
    STOCK = "stock"
    FUTURE = "future"
    CRYPTO = "crypto"
    FOREX = "forex"
    INDEX = "index"


@dataclass
class SourceConfig:
    """Data source configuration."""
    name: str
    role: SourceRole
    asset_types: List[AssetType]
    confidence: float  # 0.0 - 1.0
    adapter_module: str
    adapter_class: str
    api_key_env: Optional[str] = None
    rate_limit_per_min: int = 60
    enabled: bool = True

    def is_available(self) -> bool:
        """Check if source is available."""
        if not self.enabled:
            return False
        if self.api_key_env:
            return bool(os.getenv(self.api_key_env))
        return True


# Source configurations
SOURCES = {
    # Primary: Databento for stocks/futures
    "databento": SourceConfig(
        name="Databento",
        role=SourceRole.PRIMARY,
        asset_types=[AssetType.STOCK, AssetType.FUTURE, AssetType.INDEX],
        confidence=1.0,
        adapter_module="data.adapters.databento_adapter",
        adapter_class="DatabentoAdapter",
        api_key_env="DATABENTO_API_KEY",
    ),

    # Primary: Binance for crypto
    "binance": SourceConfig(
        name="Binance",
        role=SourceRole.PRIMARY,
        asset_types=[AssetType.CRYPTO],
        confidence=1.0,
        adapter_module="data.adapters.binance_adapter",
        adapter_class="BinanceHistoricalAdapter",
    ),

    # Backup: Polygon for USA markets
    "polygon": SourceConfig(
        name="Polygon",
        role=SourceRole.BACKUP,
        asset_types=[AssetType.STOCK, AssetType.CRYPTO, AssetType.FOREX],
        confidence=0.9,
        adapter_module="data.adapters.polygon_adapter",
        adapter_class="PolygonAdapter",
        api_key_env="POLYGON_API_KEY",
    ),

    # Backup: Yahoo (FREE)
    "yahoo": SourceConfig(
        name="Yahoo Finance",
        role=SourceRole.BACKUP,
        asset_types=[AssetType.STOCK, AssetType.FUTURE, AssetType.INDEX, AssetType.CRYPTO],
        confidence=0.7,
        adapter_module="data.adapters.yahoo_adapter",
        adapter_class="YahooFinanceAdapter",
    ),

    # Realtime: Interactive Brokers
    "ibkr": SourceConfig(
        name="Interactive Brokers",
        role=SourceRole.REALTIME,
        asset_types=[AssetType.STOCK, AssetType.FUTURE, AssetType.FOREX],
        confidence=1.0,
        adapter_module="nautilus_trader.adapters.interactive_brokers",
        adapter_class="InteractiveBrokersDataClient",
        api_key_env="IBKR_ACCOUNT",
    ),

    # Backup: CryptoCompare (FREE)
    "cryptocompare": SourceConfig(
        name="CryptoCompare",
        role=SourceRole.BACKUP,
        asset_types=[AssetType.CRYPTO],
        confidence=0.85,
        adapter_module="data.adapters.cryptocompare_adapter",
        adapter_class="CryptoCompareAdapter",
    ),
}


# =============================================================================
# 2. QUALITY CONTROL
# =============================================================================

@dataclass
class QualityFlags:
    """Quality check flags for a data point."""
    valid: bool = True
    timestamp_valid: bool = True
    price_valid: bool = True
    volume_valid: bool = True
    outlier_detected: bool = False
    reasons: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Quality report for a dataset."""
    total_rows: int = 0
    valid_rows: int = 0
    invalid_timestamp: int = 0
    zero_volume: int = 0
    outliers: int = 0
    issues: List[Dict[str, Any]] = field(default_factory=list)


class QualityController:
    """
    Data quality validation and control.

    Checks:
    - Timestamps: Market hours, chronological order
    - Prices: Outliers (>3 std dev)
    - Volume: Zero/negative volume
    - Gaps: Missing data points
    """

    # Market hours (UTC)
    MARKET_HOURS = {
        AssetType.STOCK: {
            "open": dt_time(14, 30),   # 9:30 AM ET
            "close": dt_time(21, 0),   # 4:00 PM ET
            "days": [0, 1, 2, 3, 4],   # Mon-Fri
        },
        AssetType.FUTURE: {
            "open": dt_time(23, 0),    # 6:00 PM ET (Sun)
            "close": dt_time(22, 0),   # 5:00 PM ET (Fri)
            "days": [0, 1, 2, 3, 4, 6],  # Sun-Fri
        },
        AssetType.CRYPTO: {
            "open": dt_time(0, 0),
            "close": dt_time(23, 59),
            "days": [0, 1, 2, 3, 4, 5, 6],  # 24/7
        },
    }

    def __init__(
        self,
        outlier_std_threshold: float = 3.0,
        max_price_change_pct: float = 20.0,
    ):
        """
        Initialize quality controller.

        Args:
            outlier_std_threshold: Standard deviations for outlier detection.
            max_price_change_pct: Max percentage change between bars.
        """
        self.outlier_std_threshold = outlier_std_threshold
        self.max_price_change_pct = max_price_change_pct

    def validate_timestamps(
        self,
        df: pd.DataFrame,
        asset_type: AssetType,
        strict: bool = False,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Validate timestamps are within market hours.

        Args:
            df: DataFrame with timestamp column.
            asset_type: Asset type for market hours.
            strict: If True, reject out-of-hours data.

        Returns:
            Tuple of (validated DataFrame, rejected count).
        """
        if df.empty or "timestamp" not in df.columns:
            return df, 0

        hours = self.MARKET_HOURS.get(asset_type, self.MARKET_HOURS[AssetType.CRYPTO])

        df = df.copy()
        df["_day"] = df["timestamp"].dt.dayofweek
        df["_time"] = df["timestamp"].dt.time

        # Check if within trading hours
        if asset_type == AssetType.CRYPTO:
            # 24/7 - all timestamps valid
            mask = pd.Series(True, index=df.index)
        else:
            mask = (
                df["_day"].isin(hours["days"]) &
                (df["_time"] >= hours["open"]) &
                (df["_time"] <= hours["close"])
            )

        rejected = (~mask).sum()

        if strict:
            df = df[mask]

        # Cleanup
        df = df.drop(columns=["_day", "_time"], errors="ignore")

        return df, rejected

    def detect_outliers(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        lookback: int = 20,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Detect price outliers using rolling statistics.

        Args:
            df: DataFrame with price data.
            price_col: Column to check for outliers.
            lookback: Lookback period for rolling stats.

        Returns:
            Tuple of (DataFrame with outlier flag, list of outliers).
        """
        if df.empty or price_col not in df.columns:
            return df, []

        df = df.copy()

        # Calculate rolling mean and std
        df["_rolling_mean"] = df[price_col].rolling(window=lookback, min_periods=1).mean()
        df["_rolling_std"] = df[price_col].rolling(window=lookback, min_periods=1).std()

        # Detect outliers
        df["_zscore"] = abs(df[price_col] - df["_rolling_mean"]) / (df["_rolling_std"] + 1e-10)
        df["is_outlier"] = df["_zscore"] > self.outlier_std_threshold

        outliers = []
        for _, row in df[df["is_outlier"]].iterrows():
            outliers.append({
                "timestamp": row["timestamp"].isoformat(),
                "price": row[price_col],
                "zscore": row["_zscore"],
                "expected": row["_rolling_mean"],
            })

        # Cleanup
        df = df.drop(columns=["_rolling_mean", "_rolling_std", "_zscore"], errors="ignore")

        return df, outliers

    def validate_volume(
        self,
        df: pd.DataFrame,
        allow_zero: bool = False,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Validate volume data.

        Args:
            df: DataFrame with volume column.
            allow_zero: Allow zero volume (some markets/timeframes).

        Returns:
            Tuple of (validated DataFrame, suspicious count).
        """
        if df.empty or "volume" not in df.columns:
            return df, 0

        df = df.copy()

        if allow_zero:
            suspicious = (df["volume"] < 0).sum()
            df = df[df["volume"] >= 0]
        else:
            suspicious = (df["volume"] <= 0).sum()
            df["volume_suspicious"] = df["volume"] <= 0

        return df, suspicious

    def check_price_sanity(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Check for impossible price relationships.

        Args:
            df: DataFrame with OHLC data.

        Returns:
            Tuple of (DataFrame, list of issues).
        """
        if df.empty:
            return df, []

        issues = []

        # High should be >= Low
        invalid_hl = df[df["high"] < df["low"]]
        for _, row in invalid_hl.iterrows():
            issues.append({
                "timestamp": row["timestamp"].isoformat(),
                "issue": "high < low",
                "high": row["high"],
                "low": row["low"],
            })

        # Close should be between high and low
        invalid_close = df[(df["close"] > df["high"]) | (df["close"] < df["low"])]
        for _, row in invalid_close.iterrows():
            issues.append({
                "timestamp": row["timestamp"].isoformat(),
                "issue": "close outside high-low range",
                "close": row["close"],
                "high": row["high"],
                "low": row["low"],
            })

        return df, issues

    def run_quality_checks(
        self,
        df: pd.DataFrame,
        asset_type: AssetType,
        symbol: str,
    ) -> Tuple[pd.DataFrame, QualityReport]:
        """
        Run all quality checks on a DataFrame.

        Args:
            df: DataFrame to validate.
            asset_type: Asset type.
            symbol: Symbol for logging.

        Returns:
            Tuple of (cleaned DataFrame, quality report).
        """
        report = QualityReport(total_rows=len(df))

        if df.empty:
            return df, report

        # 1. Validate timestamps
        df, rejected_ts = self.validate_timestamps(df, asset_type, strict=False)
        report.invalid_timestamp = rejected_ts

        # 2. Detect outliers
        df, outliers = self.detect_outliers(df)
        report.outliers = len(outliers)
        if outliers:
            report.issues.extend([{"type": "outlier", **o} for o in outliers])

        # 3. Validate volume
        df, suspicious_vol = self.validate_volume(df, allow_zero=(asset_type == AssetType.FOREX))
        report.zero_volume = suspicious_vol

        # 4. Check price sanity
        df, price_issues = self.check_price_sanity(df)
        if price_issues:
            report.issues.extend([{"type": "price_sanity", **p} for p in price_issues])

        # Calculate valid rows
        report.valid_rows = len(df)

        logger.info(
            f"Quality check for {symbol}",
            total=report.total_rows,
            valid=report.valid_rows,
            outliers=report.outliers,
            issues=len(report.issues),
        )

        return df, report


# =============================================================================
# 3. RECONCILIATION
# =============================================================================

@dataclass
class Discrepancy:
    """Data discrepancy between sources."""
    timestamp: datetime
    field: str
    source_a: str
    source_b: str
    value_a: float
    value_b: float
    pct_diff: float
    resolved_value: float
    resolved_source: str


class DataReconciler:
    """
    Reconciles data from multiple sources.

    Rules:
    - Use source with higher confidence
    - Log all discrepancies for audit
    - Alert if discrepancy > threshold
    """

    def __init__(
        self,
        alert_threshold_pct: float = 0.1,
        telegram_bot_token: str = "",
        telegram_chat_id: str = "",
    ):
        """
        Initialize reconciler.

        Args:
            alert_threshold_pct: Alert if discrepancy exceeds this percentage.
            telegram_bot_token: Telegram bot token for alerts.
            telegram_chat_id: Telegram chat ID for alerts.
        """
        self.alert_threshold_pct = alert_threshold_pct
        self.telegram_bot_token = telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID", "")

        self.discrepancies: List[Discrepancy] = []
        self.audit_log: List[Dict[str, Any]] = []

    def reconcile_dataframes(
        self,
        primary_df: pd.DataFrame,
        primary_source: str,
        primary_confidence: float,
        secondary_df: pd.DataFrame,
        secondary_source: str,
        secondary_confidence: float,
        price_cols: List[str] = None,
    ) -> Tuple[pd.DataFrame, List[Discrepancy]]:
        """
        Reconcile two DataFrames from different sources.

        Args:
            primary_df: Primary source data.
            primary_source: Primary source name.
            primary_confidence: Primary source confidence.
            secondary_df: Secondary source data.
            secondary_source: Secondary source name.
            secondary_confidence: Secondary source confidence.
            price_cols: Columns to compare.

        Returns:
            Tuple of (reconciled DataFrame, list of discrepancies).
        """
        price_cols = price_cols or ["open", "high", "low", "close"]
        discrepancies = []

        if primary_df.empty:
            return secondary_df, discrepancies

        if secondary_df.empty:
            return primary_df, discrepancies

        # Merge on timestamp
        merged = primary_df.merge(
            secondary_df,
            on="timestamp",
            how="outer",
            suffixes=("_primary", "_secondary"),
        )

        # Compare price columns
        result_rows = []

        for _, row in merged.iterrows():
            result_row = {"timestamp": row["timestamp"]}

            for col in price_cols:
                primary_col = f"{col}_primary"
                secondary_col = f"{col}_secondary"

                val_a = row.get(primary_col)
                val_b = row.get(secondary_col)

                # Handle missing values
                if pd.isna(val_a) and pd.isna(val_b):
                    result_row[col] = np.nan
                    continue
                elif pd.isna(val_a):
                    result_row[col] = val_b
                    continue
                elif pd.isna(val_b):
                    result_row[col] = val_a
                    continue

                # Calculate discrepancy
                if val_a != 0:
                    pct_diff = abs(val_a - val_b) / abs(val_a) * 100
                else:
                    pct_diff = 0 if val_b == 0 else 100

                # Choose value based on confidence
                if primary_confidence >= secondary_confidence:
                    resolved_value = val_a
                    resolved_source = primary_source
                else:
                    resolved_value = val_b
                    resolved_source = secondary_source

                result_row[col] = resolved_value

                # Log discrepancy if significant
                if pct_diff > 0.001:  # > 0.001%
                    disc = Discrepancy(
                        timestamp=row["timestamp"],
                        field=col,
                        source_a=primary_source,
                        source_b=secondary_source,
                        value_a=val_a,
                        value_b=val_b,
                        pct_diff=pct_diff,
                        resolved_value=resolved_value,
                        resolved_source=resolved_source,
                    )
                    discrepancies.append(disc)

                    # Alert if exceeds threshold
                    if pct_diff > self.alert_threshold_pct:
                        self._send_alert(disc)

            # Copy volume from primary if available
            vol_col = "volume_primary" if "volume_primary" in row else "volume"
            if vol_col in row:
                result_row["volume"] = row.get(vol_col, row.get("volume_secondary", 0))

            result_rows.append(result_row)

        result_df = pd.DataFrame(result_rows)

        # Log to audit
        self.audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "primary_source": primary_source,
            "secondary_source": secondary_source,
            "rows_compared": len(merged),
            "discrepancies": len(discrepancies),
            "major_discrepancies": sum(1 for d in discrepancies if d.pct_diff > self.alert_threshold_pct),
        })

        self.discrepancies.extend(discrepancies)

        return result_df, discrepancies

    def _send_alert(self, discrepancy: Discrepancy) -> None:
        """Send Telegram alert for major discrepancy."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return

        try:
            import requests

            message = (
                f"⚠️ Data Discrepancy Alert\n\n"
                f"Timestamp: {discrepancy.timestamp}\n"
                f"Field: {discrepancy.field}\n"
                f"{discrepancy.source_a}: {discrepancy.value_a:.4f}\n"
                f"{discrepancy.source_b}: {discrepancy.value_b:.4f}\n"
                f"Difference: {discrepancy.pct_diff:.2f}%\n"
                f"Using: {discrepancy.resolved_source}"
            )

            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            requests.post(url, json={
                "chat_id": self.telegram_chat_id,
                "text": message,
            }, timeout=10)

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get audit log."""
        return self.audit_log

    def save_audit_log(self, path: str) -> None:
        """Save audit log to file."""
        with open(path, "w") as f:
            json.dump(self.audit_log, f, indent=2, default=str)


# =============================================================================
# 4. STORAGE
# =============================================================================

class DataStorage:
    """
    Multi-format data storage.

    Formats:
    - TimescaleDB: Raw + clean data
    - Parquet: Optimized for backtest
    - Retention: Raw 30 days, clean indefinite
    """

    def __init__(
        self,
        timescale_host: str = "",
        timescale_port: int = 5432,
        timescale_db: str = "deskgrade",
        timescale_user: str = "postgres",
        timescale_password: str = "",
        parquet_path: str = "/app/data/catalog",
        raw_retention_days: int = 30,
    ):
        """Initialize storage."""
        self.db_config = {
            "host": timescale_host or os.getenv("TIMESCALE_HOST", "localhost"),
            "port": timescale_port or int(os.getenv("TIMESCALE_PORT", "5432")),
            "database": timescale_db or os.getenv("TIMESCALE_DB", "deskgrade"),
            "user": timescale_user or os.getenv("TIMESCALE_USER", "postgres"),
            "password": timescale_password or os.getenv("TIMESCALE_PASSWORD", ""),
        }

        self.parquet_path = Path(parquet_path)
        self.parquet_path.mkdir(parents=True, exist_ok=True)
        self.raw_retention_days = raw_retention_days

        self._engine = None

    @property
    def engine(self):
        """Get database engine."""
        if self._engine is None:
            conn_str = (
                f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
            self._engine = create_engine(conn_str, pool_size=5, max_overflow=10)
        return self._engine

    def save_raw(
        self,
        df: pd.DataFrame,
        symbol: str,
        source: str,
        timeframe: str = "1h",
    ) -> int:
        """
        Save raw data to TimescaleDB.

        Args:
            df: Raw DataFrame.
            symbol: Symbol.
            source: Data source name.
            timeframe: Timeframe.

        Returns:
            Rows saved.
        """
        if df.empty:
            return 0

        table = f"raw_ohlcv_{timeframe.replace('m', 'min').replace('h', 'hour').replace('d', 'day')}"

        df = df.copy()
        df["symbol"] = symbol
        df["source"] = source
        df["ingested_at"] = datetime.now(timezone.utc)

        try:
            with self.engine.begin() as conn:
                # Create table if not exists
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(50) NOT NULL,
                        source VARCHAR(50) NOT NULL,
                        open DECIMAL(20, 8),
                        high DECIMAL(20, 8),
                        low DECIMAL(20, 8),
                        close DECIMAL(20, 8),
                        volume DECIMAL(30, 8),
                        ingested_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (timestamp, symbol, source)
                    )
                """))

                # Batch insert
                df.to_sql(table, conn, if_exists="append", index=False, method="multi")

            logger.info(f"Saved {len(df)} raw rows to {table}")
            return len(df)

        except Exception as e:
            logger.error(f"Failed to save raw data: {e}")
            return 0

    def save_clean(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "1h",
        quality_score: float = 1.0,
    ) -> int:
        """
        Save clean data to TimescaleDB.

        Args:
            df: Cleaned DataFrame.
            symbol: Symbol.
            timeframe: Timeframe.
            quality_score: Data quality score.

        Returns:
            Rows saved.
        """
        if df.empty:
            return 0

        table = f"clean_ohlcv_{timeframe.replace('m', 'min').replace('h', 'hour').replace('d', 'day')}"

        df = df.copy()
        df["symbol"] = symbol
        df["quality_score"] = quality_score

        try:
            with self.engine.begin() as conn:
                # Create table if not exists
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(50) NOT NULL,
                        open DECIMAL(20, 8),
                        high DECIMAL(20, 8),
                        low DECIMAL(20, 8),
                        close DECIMAL(20, 8),
                        volume DECIMAL(30, 8),
                        quality_score DECIMAL(3, 2),
                        PRIMARY KEY (timestamp, symbol)
                    )
                """))

                # Upsert
                for _, row in df.iterrows():
                    conn.execute(text(f"""
                        INSERT INTO {table} (timestamp, symbol, open, high, low, close, volume, quality_score)
                        VALUES (:timestamp, :symbol, :open, :high, :low, :close, :volume, :quality_score)
                        ON CONFLICT (timestamp, symbol) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            quality_score = EXCLUDED.quality_score
                    """), dict(row))

            logger.info(f"Saved {len(df)} clean rows to {table}")
            return len(df)

        except Exception as e:
            logger.error(f"Failed to save clean data: {e}")
            return 0

    def save_parquet(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "1h",
    ) -> Path:
        """
        Save to Parquet for backtest.

        Args:
            df: DataFrame.
            symbol: Symbol.
            timeframe: Timeframe.

        Returns:
            Path to Parquet file.
        """
        if df.empty:
            return None

        # Create directory structure
        symbol_dir = self.parquet_path / symbol.replace("/", "_")
        symbol_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        filepath = symbol_dir / filename

        # Convert to Parquet
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filepath, compression="snappy")

        logger.info(f"Saved Parquet: {filepath}")
        return filepath

    def cleanup_raw_data(self) -> int:
        """
        Delete raw data older than retention period.

        Returns:
            Rows deleted.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.raw_retention_days)
        deleted = 0

        try:
            with self.engine.begin() as conn:
                # Get all raw tables
                result = conn.execute(text("""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public' AND tablename LIKE 'raw_ohlcv_%'
                """))

                for row in result:
                    table = row[0]
                    result = conn.execute(text(f"""
                        DELETE FROM {table} WHERE ingested_at < :cutoff
                    """), {"cutoff": cutoff})
                    deleted += result.rowcount

            logger.info(f"Cleaned up {deleted} old raw rows")
            return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup raw data: {e}")
            return 0

    def close(self):
        """Close connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None


# =============================================================================
# 5. MONITORING
# =============================================================================

@dataclass
class SourceStatus:
    """Data source status."""
    name: str
    last_success: Optional[datetime] = None
    last_error: Optional[datetime] = None
    consecutive_failures: int = 0
    total_fetches: int = 0
    total_errors: int = 0
    is_healthy: bool = True


class PipelineMonitor:
    """
    Pipeline monitoring with alerts.

    Monitors:
    - Source health (failures > 5 min)
    - Data gaps detected
    - Quality issues
    """

    def __init__(
        self,
        telegram_bot_token: str = "",
        telegram_chat_id: str = "",
        failure_threshold_minutes: int = 5,
    ):
        """Initialize monitor."""
        self.telegram_bot_token = telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.failure_threshold = timedelta(minutes=failure_threshold_minutes)

        self.source_status: Dict[str, SourceStatus] = {}
        self.gaps_detected: List[Dict[str, Any]] = []
        self.alerts_sent: List[Dict[str, Any]] = []

        self._check_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def record_success(self, source: str) -> None:
        """Record successful fetch."""
        if source not in self.source_status:
            self.source_status[source] = SourceStatus(name=source)

        status = self.source_status[source]
        status.last_success = datetime.now(timezone.utc)
        status.consecutive_failures = 0
        status.total_fetches += 1
        status.is_healthy = True

    def record_failure(self, source: str, error: str) -> None:
        """Record fetch failure."""
        if source not in self.source_status:
            self.source_status[source] = SourceStatus(name=source)

        status = self.source_status[source]
        status.last_error = datetime.now(timezone.utc)
        status.consecutive_failures += 1
        status.total_errors += 1

        # Check if should alert
        if status.last_success:
            time_since_success = datetime.now(timezone.utc) - status.last_success
            if time_since_success > self.failure_threshold:
                status.is_healthy = False
                self._send_source_alert(source, error, time_since_success)
        elif status.consecutive_failures >= 3:
            status.is_healthy = False
            self._send_source_alert(source, error, None)

    def record_gap(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bars_missing: int,
    ) -> None:
        """Record detected data gap."""
        gap = {
            "symbol": symbol,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "bars_missing": bars_missing,
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }
        self.gaps_detected.append(gap)

        # Alert if significant gap
        if bars_missing > 10:
            self._send_gap_alert(symbol, start, end, bars_missing)

    def _send_source_alert(
        self,
        source: str,
        error: str,
        duration: Optional[timedelta],
    ) -> None:
        """Send source failure alert."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return

        try:
            import requests

            duration_str = f" for {duration}" if duration else ""
            message = (
                f"🔴 Data Source Down{duration_str}\n\n"
                f"Source: {source}\n"
                f"Error: {error[:200]}"
            )

            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            requests.post(url, json={
                "chat_id": self.telegram_chat_id,
                "text": message,
            }, timeout=10)

            self.alerts_sent.append({
                "type": "source_down",
                "source": source,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def _send_gap_alert(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bars_missing: int,
    ) -> None:
        """Send data gap alert."""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return

        try:
            import requests

            message = (
                f"📊 Data Gap Detected\n\n"
                f"Symbol: {symbol}\n"
                f"Gap: {start.isoformat()} to {end.isoformat()}\n"
                f"Missing bars: {bars_missing}"
            )

            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            requests.post(url, json={
                "chat_id": self.telegram_chat_id,
                "text": message,
            }, timeout=10)

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            "sources": {
                name: asdict(status)
                for name, status in self.source_status.items()
            },
            "gaps_detected": len(self.gaps_detected),
            "recent_gaps": self.gaps_detected[-10:],
            "alerts_sent": len(self.alerts_sent),
            "recent_alerts": self.alerts_sent[-10:],
        }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class DataPipelineManagerPro:
    """
    Enterprise-grade data pipeline manager.

    Combines all components:
    - Multi-source ingestion
    - Quality control
    - Reconciliation
    - Storage
    - Monitoring
    """

    def __init__(self):
        """Initialize pipeline manager."""
        self.quality = QualityController()
        self.reconciler = DataReconciler()
        self.storage = DataStorage()
        self.monitor = PipelineMonitor()

        self._adapters: Dict[str, Any] = {}

    def get_adapter(self, source_name: str) -> Any:
        """Get or create adapter."""
        if source_name not in self._adapters:
            source = SOURCES.get(source_name)
            if not source or not source.is_available():
                return None

            try:
                module = __import__(source.adapter_module, fromlist=[source.adapter_class])
                adapter_class = getattr(module, source.adapter_class)
                self._adapters[source_name] = adapter_class()
            except Exception as e:
                logger.error(f"Failed to load {source_name}: {e}")
                return None

        return self._adapters.get(source_name)

    def infer_asset_type(self, symbol: str) -> AssetType:
        """Infer asset type from symbol."""
        symbol_upper = symbol.upper()

        if symbol_upper.endswith("USDT") or symbol_upper.endswith("USD"):
            if len(symbol_upper) > 6 or symbol_upper.startswith("BTC") or symbol_upper.startswith("ETH"):
                return AssetType.CRYPTO

        if "/" in symbol or symbol_upper in ["EURUSD", "GBPUSD", "USDJPY"]:
            return AssetType.FOREX

        if "=" in symbol or symbol.endswith(".FUT"):
            return AssetType.FUTURE

        if symbol.startswith("^"):
            return AssetType.INDEX

        return AssetType.STOCK

    def get_sources_for_asset(self, asset_type: AssetType) -> List[Tuple[str, SourceConfig]]:
        """Get ordered sources for asset type."""
        sources = []
        for name, config in SOURCES.items():
            if asset_type in config.asset_types and config.is_available():
                sources.append((name, config))

        # Sort: primary first, then by confidence
        sources.sort(key=lambda x: (
            0 if x[1].role == SourceRole.PRIMARY else 1,
            -x[1].confidence,
        ))

        return sources

    def process_symbol(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a single symbol through the full pipeline.

        Args:
            symbol: Symbol to process.
            timeframe: Timeframe.
            start_date: Start date.
            end_date: End date.

        Returns:
            Processing results.
        """
        results = {
            "symbol": symbol,
            "timeframe": timeframe,
            "sources_tried": [],
            "sources_successful": [],
            "rows_raw": 0,
            "rows_clean": 0,
            "quality_issues": 0,
            "discrepancies": 0,
            "errors": [],
        }

        asset_type = self.infer_asset_type(symbol)
        sources = self.get_sources_for_asset(asset_type)

        if not sources:
            results["errors"].append(f"No sources available for {asset_type}")
            return results

        primary_df = pd.DataFrame()
        secondary_df = pd.DataFrame()
        primary_source = None
        secondary_source = None

        # Fetch from sources
        for source_name, source_config in sources:
            results["sources_tried"].append(source_name)

            try:
                adapter = self.get_adapter(source_name)
                if adapter is None:
                    continue

                # Fetch data based on adapter type
                if hasattr(adapter, "fetch_all_klines"):
                    df = adapter.fetch_all_klines(symbol, timeframe, start_date, end_date)
                elif hasattr(adapter, "fetch_ohlcv"):
                    df = adapter.fetch_ohlcv(symbol, timeframe, start_date, end_date)
                elif hasattr(adapter, "fetch_all_history"):
                    base = symbol.replace("USDT", "").replace("USD", "")
                    df = adapter.fetch_all_history(base, "USD", timeframe, start_date, end_date)
                else:
                    continue

                if not df.empty:
                    results["sources_successful"].append(source_name)
                    self.monitor.record_success(source_name)

                    if primary_df.empty:
                        primary_df = df
                        primary_source = source_name
                        results["rows_raw"] = len(df)
                    else:
                        secondary_df = df
                        secondary_source = source_name
                        break  # Got primary + secondary

            except Exception as e:
                error = f"{source_name}: {str(e)}"
                results["errors"].append(error)
                self.monitor.record_failure(source_name, str(e))

        if primary_df.empty:
            return results

        # Save raw data
        self.storage.save_raw(primary_df, symbol, primary_source, timeframe)

        # Quality control
        clean_df, quality_report = self.quality.run_quality_checks(
            primary_df, asset_type, symbol
        )
        results["quality_issues"] = len(quality_report.issues)

        # Reconciliation if secondary available
        if not secondary_df.empty and secondary_source:
            primary_conf = SOURCES[primary_source].confidence
            secondary_conf = SOURCES[secondary_source].confidence

            clean_df, discrepancies = self.reconciler.reconcile_dataframes(
                clean_df, primary_source, primary_conf,
                secondary_df, secondary_source, secondary_conf,
            )
            results["discrepancies"] = len(discrepancies)

        # Save clean data
        results["rows_clean"] = len(clean_df)
        quality_score = SOURCES[primary_source].confidence

        self.storage.save_clean(clean_df, symbol, timeframe, quality_score)
        self.storage.save_parquet(clean_df, symbol, timeframe)

        logger.info(
            f"Processed {symbol}",
            raw=results["rows_raw"],
            clean=results["rows_clean"],
            quality_issues=results["quality_issues"],
            discrepancies=results["discrepancies"],
        )

        return results

    def run_batch(
        self,
        symbols: List[str],
        timeframes: List[str] = None,
        start_date: str = "2020-01-01",
    ) -> Dict[str, Any]:
        """
        Run pipeline for multiple symbols.

        Args:
            symbols: List of symbols.
            timeframes: List of timeframes.
            start_date: Start date.

        Returns:
            Batch results.
        """
        timeframes = timeframes or ["1h", "4h", "1d"]
        all_results = {}

        for timeframe in timeframes:
            logger.info(f"Processing timeframe: {timeframe}")
            timeframe_results = {}

            for symbol in symbols:
                result = self.process_symbol(symbol, timeframe, start_date)
                timeframe_results[symbol] = result

            all_results[timeframe] = timeframe_results

        # Cleanup old raw data
        self.storage.cleanup_raw_data()

        # Save audit log
        self.reconciler.save_audit_log("/app/logs/reconciliation_audit.json")

        return all_results

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get monitoring dashboard data."""
        return self.monitor.get_dashboard_data()

    def close(self):
        """Close all connections."""
        self.storage.close()


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Pipeline Manager Pro")
    parser.add_argument("--stocks", nargs="+", default=["SPY", "QQQ", "AAPL"])
    parser.add_argument("--crypto", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--timeframes", nargs="+", default=["1h", "1d"])
    parser.add_argument("--start-date", default="2020-01-01")

    args = parser.parse_args()

    manager = DataPipelineManagerPro()

    try:
        symbols = args.stocks + args.crypto
        results = manager.run_batch(symbols, args.timeframes, args.start_date)

        # Print summary
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)

        for tf, tf_results in results.items():
            print(f"\n{tf}:")
            total_raw = sum(r["rows_raw"] for r in tf_results.values())
            total_clean = sum(r["rows_clean"] for r in tf_results.values())
            total_issues = sum(r["quality_issues"] for r in tf_results.values())

            print(f"  Symbols: {len(tf_results)}")
            print(f"  Raw rows: {total_raw}")
            print(f"  Clean rows: {total_clean}")
            print(f"  Quality issues: {total_issues}")

        # Print monitoring status
        dashboard = manager.get_monitoring_dashboard()
        print("\nSource Status:")
        for name, status in dashboard["sources"].items():
            health = "✓" if status["is_healthy"] else "✗"
            print(f"  {health} {name}: {status['total_fetches']} fetches, {status['total_errors']} errors")

    finally:
        manager.close()
