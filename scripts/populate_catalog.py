#!/usr/bin/env python3
"""
Populate ParquetDataCatalog with Historical Data

Downloads data from free sources and writes to NautilusTrader's
ParquetDataCatalog format for backtesting and training.

Usage:
    python scripts/populate_catalog.py --symbols BTCUSDT ETHUSDT --timeframe 1h
    python scripts/populate_catalog.py --preset crypto --start 2022-01-01
    python scripts/populate_catalog.py --preset stocks --timeframe 1d
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import argparse

import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

logger = structlog.get_logger()


# Symbol presets
PRESETS = {
    "crypto_major": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"],
    "crypto_all": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    ],
    "stocks_major": ["SPY", "QQQ", "IWM", "DIA"],
    "stocks_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "stocks_sectors": ["XLF", "XLK", "XLE", "XLV", "XLI"],
    "test": ["BTCUSDT", "SPY"],  # Quick test
}


def infer_asset_type(symbol: str) -> str:
    """Infer asset type from symbol."""
    symbol_upper = symbol.upper()

    if symbol_upper.endswith("USDT") or symbol_upper.endswith("BUSD"):
        return "crypto"
    if symbol_upper.endswith("USD") and not symbol_upper.startswith("^"):
        if len(symbol_upper) <= 7:  # BTCUSD, ETHUSD
            return "crypto"

    return "equity"


def fetch_crypto_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch crypto data from Binance."""
    try:
        from data.adapters.binance_adapter import BinanceHistoricalAdapter

        adapter = BinanceHistoricalAdapter()
        result = adapter.fetch(symbol, timeframe, start_date, end_date)

        if result.success and not result.data.empty:
            logger.info(
                f"Fetched {len(result.data)} bars from Binance",
                symbol=symbol,
                timeframe=timeframe,
            )
            return result.data
        else:
            logger.warning(f"Binance fetch failed: {result.error}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Binance fetch error: {e}")
        return pd.DataFrame()


def fetch_stock_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch stock data from Yahoo Finance."""
    try:
        from data.adapters.yahoo_adapter import YahooFinanceAdapter

        adapter = YahooFinanceAdapter()
        result = adapter.fetch(symbol, timeframe, start_date, end_date)

        if result.success and not result.data.empty:
            logger.info(
                f"Fetched {len(result.data)} bars from Yahoo",
                symbol=symbol,
                timeframe=timeframe,
            )
            return result.data
        else:
            logger.warning(f"Yahoo fetch failed: {result.error}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Yahoo fetch error: {e}")
        return pd.DataFrame()


def fetch_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch data from appropriate source."""
    asset_type = infer_asset_type(symbol)

    if asset_type == "crypto":
        return fetch_crypto_data(symbol, timeframe, start_date, end_date)
    else:
        return fetch_stock_data(symbol, timeframe, start_date, end_date)


def write_to_catalog(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    catalog_path: str,
) -> int:
    """Write DataFrame to NautilusTrader catalog."""
    try:
        from data.adapters.nautilus_catalog_writer import NautilusCatalogWriter

        writer = NautilusCatalogWriter(catalog_path)

        # Determine asset type and venue
        asset_type = infer_asset_type(symbol)
        if asset_type == "crypto":
            venue = "BINANCE"
        else:
            venue = "NASDAQ"  # Default for stocks

        # Write to catalog
        bars_written = writer.write_from_pipeline(
            symbol=symbol,
            venue=venue,
            timeframe=timeframe,
            df=df,
            asset_type=asset_type,
        )

        writer.close()
        return bars_written

    except Exception as e:
        logger.error(f"Catalog write error: {e}")
        import traceback
        traceback.print_exc()
        return 0


def populate_symbol(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: Optional[str],
    catalog_path: str,
) -> Dict[str, Any]:
    """Populate catalog for a single symbol."""
    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "success": False,
        "rows_fetched": 0,
        "rows_written": 0,
        "error": None,
    }

    try:
        # Fetch data
        df = fetch_data(symbol, timeframe, start_date, end_date)

        if df.empty:
            result["error"] = "No data fetched"
            return result

        result["rows_fetched"] = len(df)

        # Write to catalog
        rows_written = write_to_catalog(df, symbol, timeframe, catalog_path)
        result["rows_written"] = rows_written
        result["success"] = rows_written > 0

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error processing {symbol}: {e}")

    return result


def populate_catalog(
    symbols: List[str],
    timeframes: List[str],
    start_date: str,
    end_date: Optional[str],
    catalog_path: str,
) -> List[Dict[str, Any]]:
    """Populate catalog for multiple symbols and timeframes."""
    results = []

    total = len(symbols) * len(timeframes)
    current = 0

    for timeframe in timeframes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing timeframe: {timeframe}")
        logger.info(f"{'='*60}")

        for symbol in symbols:
            current += 1
            logger.info(f"[{current}/{total}] {symbol} {timeframe}")

            result = populate_symbol(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                catalog_path=catalog_path,
            )
            results.append(result)

            # Log result
            if result["success"]:
                logger.info(
                    f"  OK: {result['rows_written']} bars written"
                )
            else:
                logger.warning(f"  FAILED: {result['error']}")

    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("CATALOG POPULATION SUMMARY")
    print("=" * 70)

    success = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\nTotal: {len(results)}")
    print(f"Success: {len(success)}")
    print(f"Failed: {len(failed)}")

    total_rows = sum(r["rows_written"] for r in success)
    print(f"\nTotal bars written: {total_rows:,}")

    if success:
        print("\nSuccessful:")
        for r in success:
            print(f"  {r['symbol']:12s} {r['timeframe']:4s} - {r['rows_written']:,} bars")

    if failed:
        print("\nFailed:")
        for r in failed:
            print(f"  {r['symbol']:12s} {r['timeframe']:4s} - {r['error']}")


def verify_catalog(catalog_path: str) -> None:
    """Verify catalog contents."""
    try:
        from nautilus_trader.persistence.catalog import ParquetDataCatalog

        catalog = ParquetDataCatalog(catalog_path)
        instruments = catalog.instruments()

        print(f"\n{'='*70}")
        print("CATALOG VERIFICATION")
        print(f"{'='*70}")
        print(f"\nPath: {catalog_path}")
        print(f"Instruments: {len(instruments)}")

        for inst in instruments:
            bars = catalog.bars(instrument_ids=[str(inst.id)])
            if bars:
                print(f"\n  {inst.id}:")
                print(f"    Bars: {len(bars):,}")
                print(f"    First: {bars[0].ts_event}")
                print(f"    Last: {bars[-1].ts_event}")

    except Exception as e:
        print(f"\nVerification error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Populate ParquetDataCatalog with historical data"
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Symbols to download (e.g., BTCUSDT SPY AAPL)",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        help="Use a preset symbol list",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1h"],
        help="Timeframes (default: 1h)",
    )
    parser.add_argument(
        "--start",
        default="2023-01-01",
        help="Start date (YYYY-MM-DD, default: 2023-01-01)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--catalog-path",
        default=str(Path(__file__).parent.parent / "data" / "catalog"),
        help="Path to catalog directory",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify catalog after population",
    )

    args = parser.parse_args()

    # Get symbols
    if args.symbols:
        symbols = args.symbols
    elif args.preset:
        symbols = PRESETS[args.preset]
    else:
        symbols = PRESETS["test"]

    # Create catalog directory
    catalog_path = Path(args.catalog_path)
    catalog_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CATALOG POPULATION")
    print("=" * 70)
    print(f"\nSymbols: {symbols}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Start: {args.start}")
    print(f"End: {args.end or 'today'}")
    print(f"Catalog: {catalog_path}")
    print()

    # Run population
    results = populate_catalog(
        symbols=symbols,
        timeframes=args.timeframes,
        start_date=args.start,
        end_date=args.end,
        catalog_path=str(catalog_path),
    )

    # Print summary
    print_summary(results)

    # Verify if requested
    if args.verify or any(r["success"] for r in results):
        verify_catalog(str(catalog_path))

    # Return exit code
    success_count = sum(1 for r in results if r["success"])
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
