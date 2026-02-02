"""
Data Storage

Multi-format storage for market data:
1. TimescaleDB: Raw data (30-day retention) + Clean data (indefinite)
2. Parquet: Optimized files for NautilusTrader backtest
"""

import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

import structlog

from data.pipeline.config import StorageConfig

logger = structlog.get_logger()


class DataStorage:
    """
    Multi-format data storage manager.

    Supports:
    - TimescaleDB for raw and clean OHLCV data
    - Parquet files for backtest optimization
    - Automatic retention management
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize storage manager.

        Args:
            config: Storage configuration.
        """
        self.config = config or StorageConfig.from_env()
        self._engine: Optional[Engine] = None

        # Ensure parquet directory exists
        self.parquet_path = Path(self.config.parquet_path)
        self.parquet_path.mkdir(parents=True, exist_ok=True)

    @property
    def engine(self) -> Engine:
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(
                self.config.connection_string,
                pool_size=5,
                max_overflow=10,
                pool_recycle=3600,
            )
        return self._engine

    # =========================================================================
    # TIMESCALEDB OPERATIONS
    # =========================================================================

    def initialize_tables(self) -> None:
        """Create required tables if they don't exist."""
        timeframes = ["1min", "5min", "15min", "30min", "1hour", "4hour", "1day", "1week"]

        with self.engine.begin() as conn:
            for tf in timeframes:
                # Raw table
                raw_table = f"{self.config.raw_table_prefix}_{tf}"
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {raw_table} (
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

                # Clean table
                clean_table = f"{self.config.clean_table_prefix}_{tf}"
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {clean_table} (
                        timestamp TIMESTAMPTZ NOT NULL,
                        symbol VARCHAR(50) NOT NULL,
                        open DECIMAL(20, 8),
                        high DECIMAL(20, 8),
                        low DECIMAL(20, 8),
                        close DECIMAL(20, 8),
                        volume DECIMAL(30, 8),
                        quality_score DECIMAL(3, 2) DEFAULT 1.0,
                        sources TEXT,
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (timestamp, symbol)
                    )
                """))

            # Metadata table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS data_pipeline_metadata (
                    symbol VARCHAR(50) NOT NULL,
                    timeframe VARCHAR(20) NOT NULL,
                    source VARCHAR(50) NOT NULL,
                    first_bar TIMESTAMPTZ,
                    last_bar TIMESTAMPTZ,
                    total_bars BIGINT DEFAULT 0,
                    quality_score DECIMAL(3, 2),
                    last_updated TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (symbol, timeframe, source)
                )
            """))

        logger.info("Database tables initialized")

    def _get_table_name(self, prefix: str, timeframe: str) -> str:
        """Convert timeframe to table name."""
        tf_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1hour",
            "4h": "4hour",
            "1d": "1day",
            "1w": "1week",
        }
        tf = tf_map.get(timeframe, timeframe)
        return f"{prefix}_{tf}"

    def save_raw(
        self,
        df: pd.DataFrame,
        symbol: str,
        source: str,
        timeframe: str,
    ) -> int:
        """
        Save raw data to TimescaleDB.

        Args:
            df: DataFrame with OHLCV data.
            symbol: Symbol.
            source: Data source name.
            timeframe: Timeframe.

        Returns:
            Number of rows saved.
        """
        if df.empty:
            return 0

        table = self._get_table_name(self.config.raw_table_prefix, timeframe)

        # Prepare data
        df = df.copy()
        df["symbol"] = symbol
        df["source"] = source
        df["ingested_at"] = datetime.now(timezone.utc)

        # Select and order columns
        columns = ["timestamp", "symbol", "source", "open", "high", "low", "close", "volume", "ingested_at"]
        df = df[[c for c in columns if c in df.columns]]

        try:
            with self.engine.begin() as conn:
                # Use ON CONFLICT DO NOTHING for raw data (keep first)
                for _, row in df.iterrows():
                    conn.execute(text(f"""
                        INSERT INTO {table} (timestamp, symbol, source, open, high, low, close, volume, ingested_at)
                        VALUES (:timestamp, :symbol, :source, :open, :high, :low, :close, :volume, :ingested_at)
                        ON CONFLICT (timestamp, symbol, source) DO NOTHING
                    """), dict(row))

            logger.debug(f"Saved {len(df)} raw rows to {table}", symbol=symbol, source=source)
            return len(df)

        except Exception as e:
            logger.error(f"Failed to save raw data: {e}", symbol=symbol, table=table)
            return 0

    def save_clean(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        quality_score: float = 1.0,
        sources: Optional[List[str]] = None,
    ) -> int:
        """
        Save clean data to TimescaleDB.

        Args:
            df: DataFrame with clean OHLCV data.
            symbol: Symbol.
            timeframe: Timeframe.
            quality_score: Overall quality score.
            sources: List of sources used.

        Returns:
            Number of rows saved.
        """
        if df.empty:
            return 0

        table = self._get_table_name(self.config.clean_table_prefix, timeframe)

        # Prepare data
        df = df.copy()
        df["symbol"] = symbol
        df["quality_score"] = quality_score
        df["sources"] = ",".join(sources) if sources else ""
        df["updated_at"] = datetime.now(timezone.utc)

        # Select columns
        columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "quality_score", "sources", "updated_at"]
        df = df[[c for c in columns if c in df.columns]]

        try:
            with self.engine.begin() as conn:
                # Use ON CONFLICT DO UPDATE for clean data (keep latest)
                for _, row in df.iterrows():
                    conn.execute(text(f"""
                        INSERT INTO {table} (timestamp, symbol, open, high, low, close, volume, quality_score, sources, updated_at)
                        VALUES (:timestamp, :symbol, :open, :high, :low, :close, :volume, :quality_score, :sources, :updated_at)
                        ON CONFLICT (timestamp, symbol) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            quality_score = EXCLUDED.quality_score,
                            sources = EXCLUDED.sources,
                            updated_at = EXCLUDED.updated_at
                    """), dict(row))

            # Update metadata
            self._update_metadata(symbol, timeframe, sources[0] if sources else "", df, quality_score)

            logger.debug(f"Saved {len(df)} clean rows to {table}", symbol=symbol)
            return len(df)

        except Exception as e:
            logger.error(f"Failed to save clean data: {e}", symbol=symbol, table=table)
            return 0

    def _update_metadata(
        self,
        symbol: str,
        timeframe: str,
        source: str,
        df: pd.DataFrame,
        quality_score: float,
    ) -> None:
        """Update pipeline metadata."""
        if df.empty:
            return

        try:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO data_pipeline_metadata (symbol, timeframe, source, first_bar, last_bar, total_bars, quality_score, last_updated)
                    VALUES (:symbol, :timeframe, :source, :first_bar, :last_bar, :total_bars, :quality_score, :last_updated)
                    ON CONFLICT (symbol, timeframe, source) DO UPDATE SET
                        first_bar = LEAST(data_pipeline_metadata.first_bar, EXCLUDED.first_bar),
                        last_bar = GREATEST(data_pipeline_metadata.last_bar, EXCLUDED.last_bar),
                        total_bars = EXCLUDED.total_bars,
                        quality_score = EXCLUDED.quality_score,
                        last_updated = EXCLUDED.last_updated
                """), {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "source": source,
                    "first_bar": df["timestamp"].min(),
                    "last_bar": df["timestamp"].max(),
                    "total_bars": len(df),
                    "quality_score": quality_score,
                    "last_updated": datetime.now(timezone.utc),
                })
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")

    def read_clean(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Read clean data from TimescaleDB.

        Args:
            symbol: Symbol.
            timeframe: Timeframe.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            limit: Maximum rows.

        Returns:
            DataFrame with OHLCV data.
        """
        table = self._get_table_name(self.config.clean_table_prefix, timeframe)

        query = f"""
            SELECT timestamp, open, high, low, close, volume, quality_score
            FROM {table}
            WHERE symbol = :symbol
        """

        params: Dict[str, Any] = {"symbol": symbol}

        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)

            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            return df

        except Exception as e:
            logger.error(f"Failed to read clean data: {e}", symbol=symbol)
            return pd.DataFrame()

    def cleanup_raw_data(self) -> int:
        """
        Delete raw data older than retention period.

        Returns:
            Number of rows deleted.
        """
        if self.config.raw_retention_days <= 0:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.raw_retention_days)
        deleted = 0

        try:
            with self.engine.begin() as conn:
                # Find all raw tables
                result = conn.execute(text("""
                    SELECT tablename FROM pg_tables
                    WHERE schemaname = 'public' AND tablename LIKE :prefix
                """), {"prefix": f"{self.config.raw_table_prefix}_%"})

                for row in result:
                    table = row[0]
                    del_result = conn.execute(text(f"""
                        DELETE FROM {table} WHERE ingested_at < :cutoff
                    """), {"cutoff": cutoff})
                    deleted += del_result.rowcount

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old raw rows")

            return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup raw data: {e}")
            return 0

    # =========================================================================
    # PARQUET OPERATIONS
    # =========================================================================

    def save_parquet(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        partition_by_date: bool = True,
    ) -> Path:
        """
        Save data to Parquet file for backtest.

        Args:
            df: DataFrame with OHLCV data.
            symbol: Symbol.
            timeframe: Timeframe.
            partition_by_date: Partition by year-month.

        Returns:
            Path to Parquet file.
        """
        if df.empty:
            return None

        # Create symbol directory
        symbol_clean = symbol.replace("/", "_").replace(".", "_")
        symbol_dir = self.parquet_path / symbol_clean
        symbol_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data
        df = df.copy()

        # Ensure timestamp is the right type
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if partition_by_date and not df.empty:
            # Partition by year-month
            df["_year_month"] = df["timestamp"].dt.to_period("M").astype(str)

            for ym, group in df.groupby("_year_month"):
                filename = f"{symbol_clean}_{timeframe}_{ym}.parquet"
                filepath = symbol_dir / filename

                group = group.drop(columns=["_year_month"])
                table = pa.Table.from_pandas(group, preserve_index=False)
                pq.write_table(table, filepath, compression="snappy")

            return symbol_dir

        else:
            filename = f"{symbol_clean}_{timeframe}.parquet"
            filepath = symbol_dir / filename

            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, filepath, compression="snappy")

            logger.debug(f"Saved Parquet: {filepath}", rows=len(df))
            return filepath

    def read_parquet(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Read data from Parquet files.

        Args:
            symbol: Symbol.
            timeframe: Timeframe.
            start_date: Start date filter.
            end_date: End date filter.

        Returns:
            DataFrame with OHLCV data.
        """
        symbol_clean = symbol.replace("/", "_").replace(".", "_")
        symbol_dir = self.parquet_path / symbol_clean

        if not symbol_dir.exists():
            return pd.DataFrame()

        # Find matching files
        pattern = f"{symbol_clean}_{timeframe}*.parquet"
        files = list(symbol_dir.glob(pattern))

        if not files:
            return pd.DataFrame()

        # Read all files
        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        combined = combined.sort_values("timestamp")
        combined = combined.drop_duplicates(subset=["timestamp"])

        # Apply date filters
        if start_date:
            combined = combined[combined["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]

        if end_date:
            combined = combined[combined["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]

        return combined.reset_index(drop=True)

    def get_parquet_stats(self) -> Dict[str, Any]:
        """Get statistics about Parquet storage."""
        stats = {
            "total_files": 0,
            "total_size_mb": 0,
            "symbols": [],
        }

        for symbol_dir in self.parquet_path.iterdir():
            if symbol_dir.is_dir():
                files = list(symbol_dir.glob("*.parquet"))
                total_size = sum(f.stat().st_size for f in files)

                stats["total_files"] += len(files)
                stats["total_size_mb"] += total_size / (1024 * 1024)
                stats["symbols"].append({
                    "symbol": symbol_dir.name,
                    "files": len(files),
                    "size_mb": total_size / (1024 * 1024),
                })

        return stats

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def close(self):
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
