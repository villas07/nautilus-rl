"""
Data Pipeline Manager

Centralizes data ingestion from multiple sources:
1. Prioritizes sources by quality (Databento > Polygon > Yahoo for stocks)
2. Eliminates duplicates
3. Fills gaps with secondary sources
4. Saves clean data to TimescaleDB

Source Priority:
- Stocks/Futures: Databento (primary) > Polygon (secondary) > Yahoo (fallback)
- Crypto: Binance (primary) > CryptoCompare (secondary) > CoinGecko (fallback)
- Macro: FRED (primary)
- Forex: IBKR (primary) > Alpha Vantage (secondary)
"""

import os
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

import structlog

logger = structlog.get_logger()


class AssetClass(str, Enum):
    """Asset classes."""

    STOCK = "stock"
    ETF = "etf"
    FUTURE = "future"
    CRYPTO = "crypto"
    FOREX = "forex"
    INDEX = "index"
    MACRO = "macro"


class DataQuality(str, Enum):
    """Data quality levels."""

    EXCELLENT = "excellent"  # Databento, direct exchange
    GOOD = "good"           # Polygon, premium APIs
    FAIR = "fair"           # Yahoo, free APIs
    POOR = "poor"           # Aggregated, delayed


@dataclass
class DataSource:
    """Data source configuration."""

    name: str
    quality: DataQuality
    asset_classes: List[AssetClass]
    priority: int  # Lower = higher priority
    adapter_class: str
    enabled: bool = True
    api_key_env: Optional[str] = None

    def is_available(self) -> bool:
        """Check if source is available (API key present)."""
        if not self.enabled:
            return False
        if self.api_key_env:
            return bool(os.getenv(self.api_key_env))
        return True


# Source configurations with priority
DATA_SOURCES = {
    # Stocks & Futures - Primary: Databento
    "databento": DataSource(
        name="Databento",
        quality=DataQuality.EXCELLENT,
        asset_classes=[AssetClass.STOCK, AssetClass.ETF, AssetClass.FUTURE, AssetClass.INDEX],
        priority=1,
        adapter_class="data.adapters.databento_adapter.DatabentoAdapter",
        api_key_env="DATABENTO_API_KEY",
    ),

    # Stocks backup: Polygon
    "polygon": DataSource(
        name="Polygon",
        quality=DataQuality.GOOD,
        asset_classes=[AssetClass.STOCK, AssetClass.ETF, AssetClass.CRYPTO, AssetClass.FOREX],
        priority=2,
        adapter_class="data.adapters.polygon_adapter.PolygonAdapter",
        api_key_env="POLYGON_API_KEY",
    ),

    # Stocks fallback: Yahoo (FREE)
    "yahoo": DataSource(
        name="Yahoo Finance",
        quality=DataQuality.FAIR,
        asset_classes=[AssetClass.STOCK, AssetClass.ETF, AssetClass.FUTURE, AssetClass.INDEX, AssetClass.CRYPTO, AssetClass.FOREX],
        priority=3,
        adapter_class="data.adapters.yahoo_adapter.YahooFinanceAdapter",
    ),

    # Crypto - Primary: Binance (FREE)
    "binance": DataSource(
        name="Binance",
        quality=DataQuality.EXCELLENT,
        asset_classes=[AssetClass.CRYPTO],
        priority=1,
        adapter_class="data.adapters.binance_adapter.BinanceHistoricalAdapter",
    ),

    # Crypto backup: CryptoCompare (FREE)
    "cryptocompare": DataSource(
        name="CryptoCompare",
        quality=DataQuality.GOOD,
        asset_classes=[AssetClass.CRYPTO],
        priority=2,
        adapter_class="data.adapters.cryptocompare_adapter.CryptoCompareAdapter",
    ),

    # Crypto fallback: CoinGecko (FREE)
    "coingecko": DataSource(
        name="CoinGecko",
        quality=DataQuality.FAIR,
        asset_classes=[AssetClass.CRYPTO],
        priority=3,
        adapter_class="data.adapters.coingecko_adapter.CoinGeckoAdapter",
    ),

    # Macro - Primary: FRED (FREE)
    "fred": DataSource(
        name="FRED",
        quality=DataQuality.EXCELLENT,
        asset_classes=[AssetClass.MACRO],
        priority=1,
        adapter_class="data.adapters.fred_adapter.FREDAdapter",
    ),

    # Forex backup: Alpha Vantage (FREE, limited)
    "alphavantage": DataSource(
        name="Alpha Vantage",
        quality=DataQuality.FAIR,
        asset_classes=[AssetClass.STOCK, AssetClass.FOREX, AssetClass.CRYPTO],
        priority=4,
        adapter_class="data.adapters.alphavantage_adapter.AlphaVantageAdapter",
        api_key_env="ALPHAVANTAGE_API_KEY",
    ),
}


@dataclass
class GapInfo:
    """Information about a data gap."""

    start: datetime
    end: datetime
    bars_missing: int
    filled: bool = False
    filled_by: Optional[str] = None


@dataclass
class PipelineStats:
    """Pipeline execution statistics."""

    symbol: str
    asset_class: AssetClass
    primary_source: str
    bars_fetched: int = 0
    gaps_found: int = 0
    gaps_filled: int = 0
    duplicates_removed: int = 0
    final_bar_count: int = 0
    sources_used: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DataPipelineManager:
    """
    Manages data pipeline from multiple sources to TimescaleDB.

    Features:
    - Automatic source selection based on asset class and priority
    - Deduplication across sources
    - Gap detection and filling from secondary sources
    - Quality scoring and metadata
    """

    def __init__(
        self,
        timescale_host: str = "localhost",
        timescale_port: int = 5432,
        timescale_db: str = "deskgrade",
        timescale_user: str = "postgres",
        timescale_password: str = "",
    ):
        """Initialize pipeline manager."""
        self.db_config = {
            "host": timescale_host or os.getenv("TIMESCALE_HOST", "localhost"),
            "port": timescale_port or int(os.getenv("TIMESCALE_PORT", "5432")),
            "database": timescale_db or os.getenv("TIMESCALE_DB", "deskgrade"),
            "user": timescale_user or os.getenv("TIMESCALE_USER", "postgres"),
            "password": timescale_password or os.getenv("TIMESCALE_PASSWORD", ""),
        }

        self._engine = None
        self._adapters: Dict[str, Any] = {}

    @property
    def engine(self):
        """Get or create database engine."""
        if self._engine is None:
            conn_str = (
                f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
            self._engine = create_engine(conn_str, pool_size=5, max_overflow=10)
        return self._engine

    def get_adapter(self, source_name: str) -> Any:
        """Get or create adapter instance."""
        if source_name not in self._adapters:
            source = DATA_SOURCES.get(source_name)
            if not source or not source.is_available():
                return None

            # Dynamic import
            try:
                module_path, class_name = source.adapter_class.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                adapter_class = getattr(module, class_name)
                self._adapters[source_name] = adapter_class()
            except Exception as e:
                logger.error(f"Failed to load adapter {source_name}: {e}")
                return None

        return self._adapters.get(source_name)

    def infer_asset_class(self, symbol: str) -> AssetClass:
        """Infer asset class from symbol."""
        symbol_upper = symbol.upper()

        # Crypto
        if symbol_upper.endswith("USDT") or symbol_upper.endswith("USD"):
            if symbol_upper in ["BTCUSDT", "ETHUSDT", "BTCUSD", "ETHUSD"] or len(symbol_upper) > 6:
                return AssetClass.CRYPTO

        # Forex
        if "/" in symbol or symbol_upper in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]:
            return AssetClass.FOREX

        # Futures
        if "=" in symbol or symbol.endswith(".FUT"):
            return AssetClass.FUTURE

        # Indices
        if symbol.startswith("^") or symbol in ["SPX", "NDX", "DJI", "VIX"]:
            return AssetClass.INDEX

        # ETFs
        etfs = {"SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "GLD", "TLT"}
        if symbol_upper in etfs:
            return AssetClass.ETF

        # Default: Stock
        return AssetClass.STOCK

    def get_sources_for_asset(self, asset_class: AssetClass) -> List[Tuple[str, DataSource]]:
        """Get ordered list of sources for an asset class."""
        sources = []

        for name, source in DATA_SOURCES.items():
            if asset_class in source.asset_classes and source.is_available():
                sources.append((name, source))

        # Sort by priority
        sources.sort(key=lambda x: x[1].priority)

        return sources

    def detect_gaps(
        self,
        df: pd.DataFrame,
        timeframe: str = "1h",
        max_gap_bars: int = 5,
    ) -> List[GapInfo]:
        """
        Detect gaps in data.

        Args:
            df: DataFrame with timestamp column.
            timeframe: Expected timeframe.
            max_gap_bars: Consider it a gap if more than this many bars missing.

        Returns:
            List of detected gaps.
        """
        if df.empty or len(df) < 2:
            return []

        # Calculate expected interval
        interval_map = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
        }
        expected_interval = interval_map.get(timeframe, timedelta(hours=1))

        df = df.sort_values("timestamp")
        gaps = []

        for i in range(1, len(df)):
            prev_ts = df.iloc[i - 1]["timestamp"]
            curr_ts = df.iloc[i]["timestamp"]
            diff = curr_ts - prev_ts

            expected_bars = diff / expected_interval

            if expected_bars > max_gap_bars:
                gaps.append(GapInfo(
                    start=prev_ts,
                    end=curr_ts,
                    bars_missing=int(expected_bars) - 1,
                ))

        return gaps

    def remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Remove duplicate timestamps, keeping highest quality data.

        Returns:
            Tuple of (cleaned DataFrame, number of duplicates removed).
        """
        if df.empty:
            return df, 0

        initial_count = len(df)

        # Drop exact duplicates
        df = df.drop_duplicates(subset=["timestamp"], keep="first")

        removed = initial_count - len(df)

        return df, removed

    def merge_with_secondary(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        gaps: List[GapInfo],
    ) -> Tuple[pd.DataFrame, int]:
        """
        Fill gaps in primary data using secondary source.

        Args:
            primary_df: Primary source data.
            secondary_df: Secondary source data.
            gaps: List of detected gaps.

        Returns:
            Tuple of (merged DataFrame, number of gaps filled).
        """
        if secondary_df.empty or not gaps:
            return primary_df, 0

        gaps_filled = 0
        filled_data = []

        for gap in gaps:
            # Get data from secondary source for this gap
            gap_data = secondary_df[
                (secondary_df["timestamp"] > gap.start) &
                (secondary_df["timestamp"] < gap.end)
            ]

            if not gap_data.empty:
                filled_data.append(gap_data)
                gap.filled = True
                gap.filled_by = "secondary"
                gaps_filled += 1

        if filled_data:
            # Combine primary with gap fills
            combined = pd.concat([primary_df] + filled_data, ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            return combined, gaps_filled

        return primary_df, 0

    def fetch_from_sources(
        self,
        symbol: str,
        asset_class: AssetClass,
        timeframe: str = "1h",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, PipelineStats]:
        """
        Fetch data from multiple sources with priority.

        Args:
            symbol: Symbol to fetch.
            asset_class: Asset class.
            timeframe: Timeframe.
            start_date: Start date.
            end_date: End date.

        Returns:
            Tuple of (DataFrame, statistics).
        """
        sources = self.get_sources_for_asset(asset_class)
        stats = PipelineStats(
            symbol=symbol,
            asset_class=asset_class,
            primary_source=sources[0][0] if sources else "none",
        )

        if not sources:
            stats.errors.append(f"No sources available for {asset_class}")
            return pd.DataFrame(), stats

        primary_df = pd.DataFrame()
        secondary_dfs = []

        # Fetch from all available sources
        for source_name, source in sources:
            try:
                adapter = self.get_adapter(source_name)
                if adapter is None:
                    continue

                logger.info(f"Fetching {symbol} from {source_name}")

                # Call appropriate fetch method based on adapter type
                if hasattr(adapter, "fetch_all_klines"):  # Binance
                    df = adapter.fetch_all_klines(
                        symbol=symbol,
                        interval=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                    )
                elif hasattr(adapter, "fetch_ohlcv"):  # Yahoo
                    df = adapter.fetch_ohlcv(
                        symbol=symbol,
                        interval=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                    )
                elif hasattr(adapter, "fetch_all_history"):  # CryptoCompare
                    df = adapter.fetch_all_history(
                        symbol=symbol.replace("USDT", "").replace("USD", ""),
                        quote="USD",
                        interval=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                    )
                else:
                    logger.warning(f"Unknown adapter type for {source_name}")
                    continue

                if not df.empty:
                    stats.sources_used.append(source_name)

                    if primary_df.empty:
                        primary_df = df
                        stats.bars_fetched = len(df)
                    else:
                        secondary_dfs.append(df)

                    # If primary is excellent quality, stop fetching
                    if source.quality == DataQuality.EXCELLENT and not primary_df.empty:
                        break

            except Exception as e:
                error = f"Error fetching from {source_name}: {str(e)}"
                stats.errors.append(error)
                logger.error(error)

        if primary_df.empty:
            return pd.DataFrame(), stats

        # Remove duplicates
        primary_df, dups_removed = self.remove_duplicates(primary_df)
        stats.duplicates_removed = dups_removed

        # Detect and fill gaps
        gaps = self.detect_gaps(primary_df, timeframe)
        stats.gaps_found = len(gaps)

        for secondary_df in secondary_dfs:
            if gaps:
                primary_df, filled = self.merge_with_secondary(primary_df, secondary_df, gaps)
                stats.gaps_filled += filled
                # Update remaining gaps
                gaps = [g for g in gaps if not g.filled]

        stats.final_bar_count = len(primary_df)

        return primary_df, stats

    def save_to_timescale(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = "1h",
        table_prefix: str = "ohlcv",
    ) -> int:
        """
        Save data to TimescaleDB.

        Args:
            df: DataFrame with OHLCV data.
            symbol: Symbol name.
            timeframe: Timeframe.
            table_prefix: Table name prefix.

        Returns:
            Number of rows inserted.
        """
        if df.empty:
            return 0

        # Map timeframe to table
        table_map = {
            "1m": f"{table_prefix}_1m",
            "5m": f"{table_prefix}_5m",
            "15m": f"{table_prefix}_15m",
            "1h": f"{table_prefix}_1h",
            "4h": f"{table_prefix}_4h",
            "1d": f"{table_prefix}_1d",
        }

        table = table_map.get(timeframe, f"{table_prefix}_1h")

        # Add symbol column
        df = df.copy()
        df["symbol"] = symbol

        # Ensure correct column order
        columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        df = df[[c for c in columns if c in df.columns]]

        try:
            # Use upsert logic (ON CONFLICT)
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
                        PRIMARY KEY (timestamp, symbol)
                    )
                """))

                # Insert with conflict handling
                for _, row in df.iterrows():
                    conn.execute(text(f"""
                        INSERT INTO {table} (timestamp, symbol, open, high, low, close, volume)
                        VALUES (:timestamp, :symbol, :open, :high, :low, :close, :volume)
                        ON CONFLICT (timestamp, symbol) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                    """), dict(row))

            logger.info(f"Saved {len(df)} bars to {table}")
            return len(df)

        except Exception as e:
            logger.error(f"Failed to save to TimescaleDB: {e}")
            return 0

    def run_pipeline(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        save_to_db: bool = True,
    ) -> Dict[str, PipelineStats]:
        """
        Run full pipeline for multiple symbols.

        Args:
            symbols: List of symbols to process.
            timeframe: Timeframe.
            start_date: Start date.
            end_date: End date.
            save_to_db: Save to TimescaleDB.

        Returns:
            Dict of statistics per symbol.
        """
        all_stats = {}

        for symbol in symbols:
            logger.info(f"Processing {symbol}")

            asset_class = self.infer_asset_class(symbol)

            df, stats = self.fetch_from_sources(
                symbol=symbol,
                asset_class=asset_class,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

            if save_to_db and not df.empty:
                rows_saved = self.save_to_timescale(df, symbol, timeframe)
                logger.info(f"Saved {rows_saved} rows for {symbol}")

            all_stats[symbol] = stats

            logger.info(
                f"Completed {symbol}",
                bars=stats.final_bar_count,
                gaps_found=stats.gaps_found,
                gaps_filled=stats.gaps_filled,
                sources=stats.sources_used,
            )

        return all_stats

    def close(self):
        """Close connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None


def run_full_pipeline(
    stocks: Optional[List[str]] = None,
    crypto: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    start_date: str = "2020-01-01",
) -> Dict[str, Any]:
    """
    Run full data pipeline.

    Args:
        stocks: Stock/ETF symbols.
        crypto: Crypto symbols.
        timeframes: Timeframes to fetch.
        start_date: Start date.

    Returns:
        Pipeline results.
    """
    manager = DataPipelineManager()

    stocks = stocks or [
        "SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
    ]

    crypto = crypto or [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    ]

    timeframes = timeframes or ["1h", "4h", "1d"]

    all_results = {}

    try:
        for timeframe in timeframes:
            logger.info(f"Processing timeframe: {timeframe}")

            # Process stocks
            stock_stats = manager.run_pipeline(
                symbols=stocks,
                timeframe=timeframe,
                start_date=start_date,
            )

            # Process crypto
            crypto_stats = manager.run_pipeline(
                symbols=crypto,
                timeframe=timeframe,
                start_date=start_date,
            )

            all_results[timeframe] = {
                "stocks": stock_stats,
                "crypto": crypto_stats,
            }

    finally:
        manager.close()

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run data pipeline")
    parser.add_argument("--stocks", nargs="+", help="Stock symbols")
    parser.add_argument("--crypto", nargs="+", help="Crypto symbols")
    parser.add_argument("--timeframes", nargs="+", default=["1h", "1d"])
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--no-save", action="store_true", help="Don't save to DB")

    args = parser.parse_args()

    results = run_full_pipeline(
        stocks=args.stocks,
        crypto=args.crypto,
        timeframes=args.timeframes,
        start_date=args.start_date,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    for timeframe, data in results.items():
        print(f"\n{timeframe}:")
        for category, stats in data.items():
            total_bars = sum(s.final_bar_count for s in stats.values())
            total_gaps = sum(s.gaps_found for s in stats.values())
            filled = sum(s.gaps_filled for s in stats.values())
            print(f"  {category}: {len(stats)} symbols, {total_bars} bars, {total_gaps} gaps ({filled} filled)")
