#!/usr/bin/env python3
"""
Load EOD Historical Data to NautilusTrader ParquetDataCatalog

Reads CSV files from data/eod/ and writes them to the catalog.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import argparse

import pandas as pd

# NautilusTrader imports
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.enums import BarAggregation, PriceType, AggregationSource
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.currencies import USD, EUR, GBP, JPY, HKD, KRW, CHF

import structlog

logger = structlog.get_logger()

# Currency mapping by exchange
EXCHANGE_CURRENCIES = {
    "LSE": GBP,      # London - British Pound (prices in pence, need conversion)
    "XETRA": EUR,    # Germany - Euro
    "PA": EUR,       # Paris - Euro
    "AS": EUR,       # Amsterdam - Euro
    "MI": EUR,       # Milan - Euro
    "MC": EUR,       # Madrid - Euro
    "SW": CHF,       # Switzerland - Swiss Franc
    "HK": HKD,       # Hong Kong - HK Dollar
    "TSE": JPY,      # Tokyo - Yen
    "KO": KRW,       # Korea - Won
    "SS": USD,       # Shanghai - treat as USD for simplicity
    "SZ": USD,       # Shenzhen - treat as USD
    "NSE": USD,      # India - treat as USD
    "BSE": USD,      # India - treat as USD
    "SG": USD,       # Singapore - treat as USD
    "INDX": USD,     # Indices - USD
}

# Venue mapping
VENUE_MAPPING = {
    "LSE": "LSE",
    "XETRA": "XETRA",
    "PA": "EURONEXT",
    "AS": "EURONEXT",
    "MI": "BORSA",
    "MC": "BME",
    "SW": "SIX",
    "HK": "HKEX",
    "TSE": "TSE",
    "KO": "KRX",
    "SS": "SSE",
    "SZ": "SZSE",
    "NSE": "NSE",
    "BSE": "BSE",
    "SG": "SGX",
    "INDX": "INDEX",
}


class EODCatalogLoader:
    """Load EOD data into NautilusTrader catalog."""

    BAR_SPEC_1D = BarSpecification(
        step=1,
        aggregation=BarAggregation.DAY,
        price_type=PriceType.LAST,
    )

    def __init__(self, catalog_path: str):
        """Initialize loader."""
        self.catalog_path = Path(catalog_path)
        self.catalog_path.mkdir(parents=True, exist_ok=True)
        self.catalog = ParquetDataCatalog(str(self.catalog_path))

    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse EOD filename to extract symbol and exchange.

        Format: SYMBOL_EXCHANGE_TIMEFRAME.csv
        Examples:
            SHEL_LSE_1d.csv -> symbol=SHEL, exchange=LSE
            005930_KO_1d.csv -> symbol=005930, exchange=KO
            GDAXI_INDX_1d.csv -> symbol=GDAXI, exchange=INDX
        """
        name = filename.replace(".csv", "")
        parts = name.rsplit("_", 2)

        if len(parts) >= 3:
            symbol = parts[0]
            exchange = parts[1]
            timeframe = parts[2]
        elif len(parts) == 2:
            symbol = parts[0]
            exchange = parts[1]
            timeframe = "1d"
        else:
            symbol = name
            exchange = "UNKNOWN"
            timeframe = "1d"

        return {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
        }

    def create_equity_instrument(self, symbol: str, exchange: str) -> Equity:
        """Create an equity instrument."""
        venue_str = VENUE_MAPPING.get(exchange, exchange)
        currency = EXCHANGE_CURRENCIES.get(exchange, USD)

        instrument_id = InstrumentId(Symbol(symbol), Venue(venue_str))

        # Determine price precision based on currency/exchange
        # LSE prices are in pence (2 decimals)
        # Asian markets often use 0 decimals for large numbers
        if exchange in ["KO", "TSE"]:
            price_precision = 0
            price_increment = "1"
        elif exchange in ["HK"]:
            price_precision = 2
            price_increment = "0.01"
        else:
            price_precision = 2
            price_increment = "0.01"

        return Equity(
            instrument_id=instrument_id,
            raw_symbol=Symbol(symbol),
            currency=currency,
            price_precision=price_precision,
            price_increment=Price.from_str(price_increment),
            lot_size=Quantity.from_str("1"),
            max_quantity=Quantity.from_str("1000000"),
            min_quantity=Quantity.from_str("1"),
            ts_event=0,
            ts_init=0,
        )

    def create_index_instrument(self, symbol: str) -> Equity:
        """Create an index instrument (using Equity as base)."""
        instrument_id = InstrumentId(Symbol(symbol), Venue("INDEX"))

        return Equity(
            instrument_id=instrument_id,
            raw_symbol=Symbol(symbol),
            currency=USD,
            price_precision=2,
            price_increment=Price.from_str("0.01"),
            lot_size=Quantity.from_str("1"),
            max_quantity=Quantity.from_str("1000000"),
            min_quantity=Quantity.from_str("1"),
            ts_event=0,
            ts_init=0,
        )

    def load_csv(self, filepath: Path) -> pd.DataFrame:
        """Load and clean CSV file."""
        df = pd.read_csv(filepath)

        # Standardize column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Parse timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        elif "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"], utc=True)

        # Ensure numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with NaN
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def write_bars(self, df: pd.DataFrame, instrument_id: InstrumentId) -> int:
        """Write bars to catalog."""
        if df.empty:
            return 0

        bar_type = BarType(
            instrument_id=instrument_id,
            bar_spec=self.BAR_SPEC_1D,
            aggregation_source=AggregationSource.EXTERNAL,
        )

        # Get precision from symbol/exchange
        symbol = str(instrument_id.symbol)
        venue = str(instrument_id.venue)

        # Determine precision
        if venue in ["KRX", "TSE"]:
            precision = 0
        else:
            precision = 2

        bars = []
        for _, row in df.iterrows():
            ts = row["timestamp"]
            if isinstance(ts, pd.Timestamp):
                ts_ns = int(ts.value)
            else:
                ts_ns = int(pd.Timestamp(ts).value)

            try:
                bar = Bar(
                    bar_type=bar_type,
                    open=Price.from_str(f"{row['open']:.{precision}f}"),
                    high=Price.from_str(f"{row['high']:.{precision}f}"),
                    low=Price.from_str(f"{row['low']:.{precision}f}"),
                    close=Price.from_str(f"{row['close']:.{precision}f}"),
                    volume=Quantity.from_str(f"{max(0, row['volume']):.0f}"),
                    ts_event=ts_ns,
                    ts_init=ts_ns,
                )
                bars.append(bar)
            except Exception as e:
                logger.warning(f"Error creating bar: {e}", row=row.to_dict())
                continue

        if bars:
            self.catalog.write_data(bars)

        return len(bars)

    def load_file(self, filepath: Path) -> Dict[str, Any]:
        """Load a single CSV file into catalog."""
        info = self.parse_filename(filepath.name)
        symbol = info["symbol"]
        exchange = info["exchange"]

        # Load data
        df = self.load_csv(filepath)
        if df.empty:
            return {"status": "empty", "bars": 0}

        # Create instrument
        if exchange == "INDX":
            instrument = self.create_index_instrument(symbol)
        else:
            instrument = self.create_equity_instrument(symbol, exchange)

        instrument_id = instrument.id

        # Check if instrument exists
        existing = self.catalog.instruments(instrument_ids=[str(instrument_id)])
        if not existing:
            self.catalog.write_data([instrument])
            logger.info(f"Created instrument: {instrument_id}")

        # Write bars
        bars_written = self.write_bars(df, instrument_id)

        return {
            "status": "success",
            "instrument_id": str(instrument_id),
            "bars": bars_written,
            "start": df["timestamp"].min().isoformat() if not df.empty else None,
            "end": df["timestamp"].max().isoformat() if not df.empty else None,
        }

    def load_directory(self, data_dir: Path) -> Dict[str, Any]:
        """Load all CSV files from directory."""
        csv_files = list(data_dir.glob("*.csv"))

        if not csv_files:
            logger.warning(f"No CSV files found in {data_dir}")
            return {"total": 0, "success": 0, "failed": 0}

        logger.info(f"Found {len(csv_files)} CSV files to load")

        results = {
            "total": len(csv_files),
            "success": 0,
            "failed": 0,
            "total_bars": 0,
            "details": [],
        }

        for i, filepath in enumerate(sorted(csv_files)):
            print(f"[{i+1}/{len(csv_files)}] Loading {filepath.name}...", end=" ")

            try:
                result = self.load_file(filepath)

                if result["status"] == "success":
                    results["success"] += 1
                    results["total_bars"] += result["bars"]
                    print(f"OK ({result['bars']} bars)")
                else:
                    results["failed"] += 1
                    print(f"EMPTY")

                results["details"].append({
                    "file": filepath.name,
                    **result,
                })

            except Exception as e:
                results["failed"] += 1
                print(f"ERROR: {e}")
                results["details"].append({
                    "file": filepath.name,
                    "status": "error",
                    "error": str(e),
                })

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load EOD data to Nautilus catalog")
    parser.add_argument(
        "--data-dir",
        default="data/eod",
        help="Directory containing EOD CSV files",
    )
    parser.add_argument(
        "--catalog-path",
        default="data/catalog",
        help="Path to NautilusTrader catalog",
    )
    parser.add_argument(
        "--file",
        help="Load specific file only",
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    catalog_path = project_root / args.catalog_path

    print(f"Data directory: {data_dir}")
    print(f"Catalog path: {catalog_path}")
    print()

    loader = EODCatalogLoader(str(catalog_path))

    if args.file:
        # Load single file
        filepath = data_dir / args.file
        result = loader.load_file(filepath)
        print(f"\nResult: {result}")
    else:
        # Load all files
        results = loader.load_directory(data_dir)

        print("\n" + "=" * 60)
        print("LOAD COMPLETE")
        print("=" * 60)
        print(f"Total files:  {results['total']}")
        print(f"Success:      {results['success']}")
        print(f"Failed:       {results['failed']}")
        print(f"Total bars:   {results['total_bars']}")
        print("=" * 60)

        # Show catalog summary
        print("\nCatalog contents:")
        instruments = loader.catalog.instruments()
        print(f"  Instruments: {len(instruments)}")

        # Group by venue
        venues = {}
        for inst in instruments:
            venue = str(inst.id.venue)
            venues[venue] = venues.get(venue, 0) + 1

        for venue, count in sorted(venues.items()):
            print(f"    {venue}: {count}")


if __name__ == "__main__":
    main()
