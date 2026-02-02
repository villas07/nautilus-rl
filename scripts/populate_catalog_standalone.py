#!/usr/bin/env python3
"""
Standalone Catalog Population Script

Downloads data directly from APIs and writes to ParquetDataCatalog.
No database dependencies required.

Usage:
    python scripts/populate_catalog_standalone.py --symbols BTCUSDT SPY
    python scripts/populate_catalog_standalone.py --preset crypto --start 2024-01-01
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import argparse
import time

import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# NautilusTrader imports
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity, CryptoPerpetual
from nautilus_trader.model.enums import BarAggregation, PriceType, AggregationSource
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.model.currencies import USD, USDT


# Presets
PRESETS = {
    "crypto": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
    "stocks": ["SPY", "QQQ", "AAPL", "MSFT"],
    "test": ["BTCUSDT", "SPY"],
}


class BinanceFetcher:
    """Simple Binance data fetcher."""

    BASE_URL = "https://api.binance.com/api/v3/klines"
    MAX_BARS = 1000

    INTERVAL_MS = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000,
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
    }

    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def fetch(self, symbol: str, interval: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Fetch OHLCV data from Binance."""
        start_ts = int(datetime.fromisoformat(start_date).timestamp() * 1000)
        end_ts = int(datetime.now().timestamp() * 1000) if not end_date else int(datetime.fromisoformat(end_date).timestamp() * 1000)

        all_data = []
        current_start = start_ts
        interval_ms = self.INTERVAL_MS.get(interval, 3_600_000)

        while current_start < end_ts:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": self.MAX_BARS,
            }

            try:
                resp = self.session.get(self.BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                if not data:
                    break

                all_data.extend(data)
                last_ts = data[-1][0]
                current_start = last_ts + interval_ms

                if len(data) < self.MAX_BARS:
                    break

                time.sleep(0.1)  # Rate limiting

            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        return df[["timestamp", "open", "high", "low", "close", "volume"]]


class YahooFetcher:
    """Simple Yahoo Finance data fetcher."""

    INTERVAL_MAP = {"1h": "1h", "4h": "1h", "1d": "1d", "1m": "1m", "5m": "5m", "15m": "15m"}

    def fetch(self, symbol: str, interval: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        try:
            import yfinance as yf

            yf_interval = self.INTERVAL_MAP.get(interval, "1h")
            ticker = yf.Ticker(symbol)

            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.now() if not end_date else datetime.fromisoformat(end_date)

            df = ticker.history(start=start_dt, end=end_dt, interval=yf_interval)

            if df.empty:
                return pd.DataFrame()

            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]

            # Handle datetime/date column name
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "timestamp"})
            elif "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})

            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Select columns
            cols = ["timestamp", "open", "high", "low", "close", "volume"]
            df = df[[c for c in cols if c in df.columns]]

            return df

        except Exception as e:
            print(f"Error fetching {symbol} from Yahoo: {e}")
            return pd.DataFrame()


class CatalogWriter:
    """Writes data to NautilusTrader ParquetDataCatalog."""

    BAR_SPECS = {
        "1m": BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
        "5m": BarSpecification(5, BarAggregation.MINUTE, PriceType.LAST),
        "15m": BarSpecification(15, BarAggregation.MINUTE, PriceType.LAST),
        "1h": BarSpecification(1, BarAggregation.HOUR, PriceType.LAST),
        "4h": BarSpecification(4, BarAggregation.HOUR, PriceType.LAST),
        "1d": BarSpecification(1, BarAggregation.DAY, PriceType.LAST),
    }

    def __init__(self, catalog_path: str):
        self.catalog_path = Path(catalog_path)
        self.catalog_path.mkdir(parents=True, exist_ok=True)
        self._catalog = None

    @property
    def catalog(self) -> ParquetDataCatalog:
        if self._catalog is None:
            self._catalog = ParquetDataCatalog(str(self.catalog_path))
        return self._catalog

    def _infer_precision(self, value: float) -> int:
        """Infer decimal precision from value."""
        s = f"{value:.10f}".rstrip("0")
        return len(s.split(".")[1]) if "." in s else 0

    def create_crypto_instrument(self, symbol: str, venue: str = "BINANCE") -> CryptoPerpetual:
        """Create a crypto instrument."""
        instrument_id = InstrumentId(Symbol(symbol), Venue(venue))

        # Parse base/quote
        if symbol.endswith("USDT"):
            base, quote = symbol[:-4], "USDT"
        elif symbol.endswith("USD"):
            base, quote = symbol[:-3], "USD"
        else:
            base, quote = symbol[:3], symbol[3:]

        return CryptoPerpetual(
            instrument_id=instrument_id,
            raw_symbol=Symbol(symbol),
            base_currency=USDT,  # Simplified
            quote_currency=USDT,
            settlement_currency=USDT,
            is_inverse=False,
            price_precision=2,
            size_precision=3,
            price_increment=Price.from_str("0.01"),
            size_increment=Quantity.from_str("0.001"),
            max_quantity=Quantity.from_str("10000"),
            min_quantity=Quantity.from_str("0.001"),
            max_notional=None,
            min_notional=Money.from_str("10 USDT"),
            max_price=Price.from_str("1000000"),
            min_price=Price.from_str("0.01"),
            margin_init=0.01,
            margin_maint=0.005,
            maker_fee=0.0002,
            taker_fee=0.0004,
            ts_event=0,
            ts_init=0,
        )

    def create_equity_instrument(self, symbol: str, venue: str = "NASDAQ") -> Equity:
        """Create an equity instrument."""
        instrument_id = InstrumentId(Symbol(symbol), Venue(venue))

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

    def write_bars(self, df: pd.DataFrame, instrument_id: str, timeframe: str) -> int:
        """Write bars to catalog."""
        if df.empty:
            return 0

        inst_id = InstrumentId.from_str(instrument_id)
        bar_spec = self.BAR_SPECS.get(timeframe)
        if not bar_spec:
            print(f"Unknown timeframe: {timeframe}")
            return 0

        bar_type = BarType(
            instrument_id=inst_id,
            bar_spec=bar_spec,
            aggregation_source=AggregationSource.EXTERNAL,
        )

        # Get precision from instrument
        # Equity = 2, Crypto = 2 (we set this in instrument creation)
        precision = 2  # Match instrument price_precision

        bars = []
        df = df.sort_values("timestamp").reset_index(drop=True)

        for _, row in df.iterrows():
            ts = row["timestamp"]
            if isinstance(ts, pd.Timestamp):
                ts_ns = int(ts.value)
            else:
                ts_ns = int(pd.Timestamp(ts).value)

            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{row['open']:.{precision}f}"),
                high=Price.from_str(f"{row['high']:.{precision}f}"),
                low=Price.from_str(f"{row['low']:.{precision}f}"),
                close=Price.from_str(f"{row['close']:.{precision}f}"),
                volume=Quantity.from_str(f"{row['volume']:.0f}"),
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
            bars.append(bar)

        if bars:
            self.catalog.write_data(bars)

        return len(bars)

    def write_instrument(self, instrument) -> None:
        """Write instrument to catalog."""
        self.catalog.write_data([instrument])


def is_crypto(symbol: str) -> bool:
    """Check if symbol is crypto."""
    return symbol.upper().endswith("USDT") or symbol.upper().endswith("BUSD")


def populate(
    symbols: List[str],
    timeframe: str,
    start_date: str,
    end_date: Optional[str],
    catalog_path: str,
) -> Dict[str, Any]:
    """Populate catalog with data."""

    binance = BinanceFetcher()
    yahoo = YahooFetcher()
    writer = CatalogWriter(catalog_path)

    results = {"success": 0, "failed": 0, "total_bars": 0, "details": []}

    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] {symbol} {timeframe}")

        # Fetch data
        if is_crypto(symbol):
            venue = "BINANCE"
            df = binance.fetch(symbol, timeframe, start_date, end_date)
        else:
            venue = "NASDAQ"
            df = yahoo.fetch(symbol, timeframe, start_date, end_date)

        if df.empty:
            print(f"  FAILED: No data")
            results["failed"] += 1
            results["details"].append({"symbol": symbol, "status": "failed", "bars": 0})
            continue

        print(f"  Fetched {len(df)} bars")

        # Create instrument
        instrument_id = f"{symbol}.{venue}"
        try:
            if is_crypto(symbol):
                instrument = writer.create_crypto_instrument(symbol, venue)
            else:
                instrument = writer.create_equity_instrument(symbol, venue)

            # Check if instrument exists
            existing = writer.catalog.instruments(instrument_ids=[instrument_id])
            if not existing:
                writer.write_instrument(instrument)
                print(f"  Created instrument: {instrument_id}")
        except Exception as e:
            print(f"  Warning creating instrument: {e}")

        # Write bars
        try:
            bars_written = writer.write_bars(df, instrument_id, timeframe)
            print(f"  Wrote {bars_written} bars")
            results["success"] += 1
            results["total_bars"] += bars_written
            results["details"].append({"symbol": symbol, "status": "success", "bars": bars_written})
        except Exception as e:
            print(f"  FAILED writing bars: {e}")
            results["failed"] += 1
            results["details"].append({"symbol": symbol, "status": "failed", "bars": 0})

    return results


def verify_catalog(catalog_path: str) -> None:
    """Verify catalog contents."""
    print(f"\n{'='*60}")
    print("CATALOG VERIFICATION")
    print(f"{'='*60}")

    try:
        catalog = ParquetDataCatalog(catalog_path)
        instruments = catalog.instruments()

        print(f"\nPath: {catalog_path}")
        print(f"Instruments: {len(instruments)}")

        for inst in instruments[:10]:  # Limit to 10
            bars = catalog.bars(instrument_ids=[str(inst.id)])
            bar_count = len(bars) if bars else 0
            print(f"  {inst.id}: {bar_count:,} bars")

    except Exception as e:
        print(f"Verification error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Populate catalog with market data")
    parser.add_argument("--symbols", nargs="+", help="Symbols to fetch")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Use preset")
    parser.add_argument("--timeframe", default="1h", help="Timeframe (1h, 4h, 1d)")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--catalog", default="./data/catalog", help="Catalog path")
    parser.add_argument("--verify", action="store_true", help="Verify after population")

    args = parser.parse_args()

    symbols = args.symbols or PRESETS.get(args.preset, PRESETS["test"])

    print("=" * 60)
    print("CATALOG POPULATION")
    print("=" * 60)
    print(f"Symbols: {symbols}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Start: {args.start}")
    print(f"End: {args.end or 'today'}")
    print(f"Catalog: {args.catalog}")

    results = populate(
        symbols=symbols,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        catalog_path=args.catalog,
    )

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Total bars: {results['total_bars']:,}")

    if args.verify or results['success'] > 0:
        verify_catalog(args.catalog)

    return 0 if results['success'] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
