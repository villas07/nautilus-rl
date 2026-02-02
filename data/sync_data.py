#!/usr/bin/env python3
"""
Data Synchronization Script

Syncs data from TimescaleDB to Nautilus DataCatalog (Parquet format).
Designed to run daily via cron or scheduler.

Usage:
    python sync_data.py                    # Sync all instruments
    python sync_data.py --symbols SPY QQQ  # Sync specific symbols
    python sync_data.py --incremental      # Only sync new data
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
import structlog

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.adapters.timescale_adapter import TimescaleAdapter, TimescaleConfig
from data.adapters.bar_converter import BarConverter, convert_symbol_to_nautilus, infer_venue

load_dotenv()

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)

logger = structlog.get_logger()


class DataSyncer:
    """Synchronizes data from TimescaleDB to Parquet catalog."""

    def __init__(
        self,
        catalog_path: str = "/app/data/catalog",
        timescale_config: Optional[TimescaleConfig] = None,
    ):
        """
        Initialize syncer.

        Args:
            catalog_path: Path to Nautilus DataCatalog.
            timescale_config: TimescaleDB configuration.
        """
        self.catalog_path = Path(catalog_path)
        self.catalog_path.mkdir(parents=True, exist_ok=True)

        self.adapter = TimescaleAdapter(timescale_config)
        self.converter = BarConverter()

        # Track sync metadata
        self.metadata_file = self.catalog_path / ".sync_metadata.json"

    def sync_symbol(
        self,
        symbol: str,
        venue: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        incremental: bool = False,
    ) -> int:
        """
        Sync a single symbol.

        Args:
            symbol: Symbol to sync.
            venue: Venue (IBKR, BINANCE).
            timeframe: Timeframe to sync.
            start_date: Start date (optional).
            end_date: End date (optional).
            incremental: Only sync new data.

        Returns:
            Number of bars synced.
        """
        instrument_id = f"{symbol}.{venue}"

        logger.info(
            "Syncing symbol",
            symbol=symbol,
            venue=venue,
            timeframe=timeframe,
        )

        # Get last sync date if incremental
        if incremental:
            last_sync = self._get_last_sync(instrument_id, timeframe)
            if last_sync:
                start_date = (last_sync + timedelta(days=1)).strftime("%Y-%m-%d")
                logger.info(f"Incremental sync from {start_date}")

        # Fetch data from TimescaleDB
        df = self.adapter.fetch_bars(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )

        if df.empty:
            logger.warning(f"No data for {symbol}")
            return 0

        # Write to catalog
        self.converter.write_to_catalog(
            df=df,
            instrument_id=instrument_id,
            timeframe=timeframe,
            catalog_path=str(self.catalog_path),
        )

        # Update sync metadata
        self._update_sync_metadata(instrument_id, timeframe, df["timestamp"].max())

        return len(df)

    def sync_all(
        self,
        timeframes: List[str] = ["1h", "1d"],
        incremental: bool = False,
    ) -> dict:
        """
        Sync all available symbols.

        Args:
            timeframes: Timeframes to sync.
            incremental: Only sync new data.

        Returns:
            Dict with sync statistics.
        """
        stats = {
            "total_symbols": 0,
            "total_bars": 0,
            "errors": [],
        }

        for timeframe in timeframes:
            symbols = self.adapter.get_available_symbols(timeframe)
            logger.info(f"Found {len(symbols)} symbols for {timeframe}")

            for symbol in symbols:
                try:
                    venue = infer_venue(symbol)
                    bars_synced = self.sync_symbol(
                        symbol=symbol,
                        venue=venue,
                        timeframe=timeframe,
                        incremental=incremental,
                    )
                    stats["total_bars"] += bars_synced
                    stats["total_symbols"] += 1
                except Exception as e:
                    logger.error(f"Error syncing {symbol}: {e}")
                    stats["errors"].append({"symbol": symbol, "error": str(e)})

        return stats

    def sync_specific(
        self,
        symbols: List[str],
        timeframes: List[str] = ["1h", "1d"],
        venues: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        incremental: bool = False,
    ) -> dict:
        """
        Sync specific symbols.

        Args:
            symbols: List of symbols to sync.
            timeframes: Timeframes to sync.
            venues: Venues for each symbol (or auto-infer).
            start_date: Start date.
            end_date: End date.
            incremental: Only sync new data.

        Returns:
            Dict with sync statistics.
        """
        stats = {
            "total_symbols": 0,
            "total_bars": 0,
            "errors": [],
        }

        for i, symbol in enumerate(symbols):
            venue = venues[i] if venues and i < len(venues) else infer_venue(symbol)

            for timeframe in timeframes:
                try:
                    bars_synced = self.sync_symbol(
                        symbol=symbol,
                        venue=venue,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        incremental=incremental,
                    )
                    stats["total_bars"] += bars_synced
                except Exception as e:
                    logger.error(f"Error syncing {symbol}: {e}")
                    stats["errors"].append({"symbol": symbol, "error": str(e)})

            stats["total_symbols"] += 1

        return stats

    def _get_last_sync(
        self,
        instrument_id: str,
        timeframe: str,
    ) -> Optional[datetime]:
        """Get last sync timestamp for an instrument."""
        import json

        if not self.metadata_file.exists():
            return None

        try:
            with open(self.metadata_file) as f:
                metadata = json.load(f)

            key = f"{instrument_id}_{timeframe}"
            if key in metadata:
                return datetime.fromisoformat(metadata[key]["last_sync"])
        except Exception:
            pass

        return None

    def _update_sync_metadata(
        self,
        instrument_id: str,
        timeframe: str,
        last_timestamp: datetime,
    ) -> None:
        """Update sync metadata."""
        import json

        metadata = {}
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    metadata = json.load(f)
            except Exception:
                pass

        key = f"{instrument_id}_{timeframe}"
        metadata[key] = {
            "last_sync": last_timestamp.isoformat(),
            "synced_at": datetime.now(tz=None).isoformat(),
        }

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sync data to Nautilus catalog")

    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Specific symbols to sync",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1h", "1d"],
        help="Timeframes to sync (default: 1h 1d)",
    )
    parser.add_argument(
        "--catalog",
        default=os.getenv("CATALOG_PATH", "/app/data/catalog"),
        help="Path to catalog directory",
    )
    parser.add_argument(
        "--start-date",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only sync new data since last sync",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sync all available symbols",
    )

    args = parser.parse_args()

    # Initialize syncer
    syncer = DataSyncer(catalog_path=args.catalog)

    # Run sync
    if args.symbols:
        stats = syncer.sync_specific(
            symbols=args.symbols,
            timeframes=args.timeframes,
            start_date=args.start_date,
            end_date=args.end_date,
            incremental=args.incremental,
        )
    elif args.all:
        stats = syncer.sync_all(
            timeframes=args.timeframes,
            incremental=args.incremental,
        )
    else:
        # Default: sync common symbols
        default_symbols = [
            "SPY", "QQQ", "IWM",  # ETFs
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",  # Tech
            "BTCUSDT", "ETHUSDT",  # Crypto
        ]
        stats = syncer.sync_specific(
            symbols=default_symbols,
            timeframes=args.timeframes,
            start_date=args.start_date,
            end_date=args.end_date,
            incremental=args.incremental,
        )

    # Print summary
    print("\n" + "=" * 50)
    print("SYNC COMPLETE")
    print("=" * 50)
    print(f"Symbols synced: {stats['total_symbols']}")
    print(f"Total bars: {stats['total_bars']}")
    if stats["errors"]:
        print(f"Errors: {len(stats['errors'])}")
        for err in stats["errors"]:
            print(f"  - {err['symbol']}: {err['error']}")

    # Close connections
    syncer.adapter.close()


if __name__ == "__main__":
    main()
