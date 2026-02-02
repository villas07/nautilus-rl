"""
Yahoo Finance Historical Data Adapter

FREE unlimited historical data for:
- US Stocks (all NYSE, NASDAQ)
- ETFs (SPY, QQQ, IWM, etc.)
- Indices (^GSPC, ^IXIC, ^DJI)
- Forex (EURUSD=X, GBPUSD=X)
- Crypto (BTC-USD, ETH-USD)
- Commodities (GC=F, CL=F)

Uses yfinance library - no API key required!
"""

import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

import structlog

logger = structlog.get_logger()


class YahooFinanceAdapter:
    """
    Adapter for downloading free historical data from Yahoo Finance.

    Coverage:
    - US Stocks: All NYSE, NASDAQ tickers
    - ETFs: All US-listed ETFs
    - Indices: Major world indices
    - Forex: Currency pairs (EURUSD=X format)
    - Crypto: Major cryptocurrencies (BTC-USD format)
    - Futures: Front-month contracts (GC=F format)

    Timeframes: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
    History: varies by timeframe (1m: 7 days, 1d: full history)
    """

    # Timeframe limits
    INTERVAL_LIMITS = {
        "1m": 7,       # 7 days max
        "2m": 60,      # 60 days
        "5m": 60,
        "15m": 60,
        "30m": 60,
        "1h": 730,     # ~2 years
        "1d": 36500,   # ~100 years
        "5d": 36500,
        "1wk": 36500,
        "1mo": 36500,
    }

    # Common symbol lists
    TOP_US_STOCKS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "UNH", "JPM", "V", "JNJ", "XOM", "PG", "MA", "HD", "CVX", "LLY",
        "MRK", "ABBV", "KO", "PEP", "COST", "TMO", "AVGO", "WMT", "MCD",
    ]

    TOP_ETFS = [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO",
        "EFA", "AGG", "BND", "TLT", "GLD", "SLV", "USO", "XLF",
        "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "ARKK",
    ]

    INDICES = [
        "^GSPC",    # S&P 500
        "^IXIC",    # NASDAQ Composite
        "^DJI",     # Dow Jones
        "^RUT",     # Russell 2000
        "^VIX",     # VIX
        "^FTSE",    # FTSE 100
        "^GDAXI",   # DAX
        "^N225",    # Nikkei 225
    ]

    FOREX = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X",
        "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURGBP=X",
    ]

    CRYPTO = [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
        "ADA-USD", "AVAX-USD", "DOT-USD", "MATIC-USD", "LINK-USD",
    ]

    FUTURES = [
        "ES=F",     # E-mini S&P 500
        "NQ=F",     # E-mini NASDAQ
        "YM=F",     # E-mini Dow
        "RTY=F",    # E-mini Russell 2000
        "GC=F",     # Gold
        "SI=F",     # Silver
        "CL=F",     # Crude Oil
        "NG=F",     # Natural Gas
        "ZB=F",     # 30Y Treasury
        "ZN=F",     # 10Y Treasury
    ]

    def __init__(self):
        """Initialize adapter."""
        if yf is None:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Yahoo Finance symbol.
            interval: Timeframe (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo).
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            period: Alternative to dates (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max).

        Returns:
            DataFrame with timestamp, open, high, low, close, volume.
        """
        try:
            ticker = yf.Ticker(symbol)

            if period:
                df = ticker.history(period=period, interval=interval)
            elif start_date:
                df = ticker.history(
                    start=start_date,
                    end=end_date or datetime.now().strftime("%Y-%m-%d"),
                    interval=interval,
                )
            else:
                df = ticker.history(period="max", interval=interval)

            if df.empty:
                logger.warning(f"No data for {symbol}")
                return pd.DataFrame()

            # Standardize columns
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            # Rename date/datetime column to timestamp
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            elif "datetime" in df.columns:
                df = df.rename(columns={"datetime": "timestamp"})

            # Ensure UTC timezone
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
            else:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

            # Keep only OHLCV columns
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            logger.info(
                f"Fetched {symbol}",
                bars=len(df),
                start=df["timestamp"].min().isoformat(),
                end=df["timestamp"].max().isoformat(),
            )

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def fetch_multiple(
        self,
        symbols: List[str],
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of symbols.
            interval: Timeframe.
            start_date: Start date.
            end_date: End date.
            period: Period alternative.

        Returns:
            Dict mapping symbol to DataFrame.
        """
        results = {}

        for symbol in symbols:
            df = self.fetch_ohlcv(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                period=period,
            )
            results[symbol] = df

        return results

    def fetch_all_asset_classes(
        self,
        interval: str = "1d",
        start_date: str = "2015-01-01",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all major asset classes.

        Returns dict with all available data.
        """
        all_symbols = (
            self.TOP_US_STOCKS +
            self.TOP_ETFS +
            self.INDICES +
            self.FOREX +
            self.CRYPTO +
            self.FUTURES
        )

        return self.fetch_multiple(
            symbols=all_symbols,
            interval=interval,
            start_date=start_date,
        )

    def get_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """Get ticker information."""
        ticker = yf.Ticker(symbol)
        return ticker.info

    def search_tickers(self, query: str) -> List[Dict[str, Any]]:
        """Search for tickers by name or symbol."""
        # yfinance doesn't have built-in search, return empty
        # Could integrate with Yahoo search API
        return []


def download_yahoo_data(
    symbols: Optional[List[str]] = None,
    intervals: Optional[List[str]] = None,
    start_date: str = "2015-01-01",
    output_dir: str = "/app/data/yahoo",
) -> Dict[str, int]:
    """
    Download Yahoo Finance data and save to CSV.

    Args:
        symbols: Symbols to download (default: all asset classes).
        intervals: Timeframes (default: 1h, 1d).
        start_date: Start date.
        output_dir: Output directory.

    Returns:
        Download statistics.
    """
    adapter = YahooFinanceAdapter()

    if symbols is None:
        symbols = (
            adapter.TOP_US_STOCKS[:10] +
            adapter.TOP_ETFS[:10] +
            adapter.INDICES +
            adapter.CRYPTO
        )

    intervals = intervals or ["1d"]  # 1h limited to 730 days

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"total_bars": 0, "files_created": 0, "errors": []}

    for symbol in symbols:
        for interval in intervals:
            try:
                df = adapter.fetch_ohlcv(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                )

                if not df.empty:
                    # Clean symbol for filename
                    clean_symbol = symbol.replace("=", "_").replace("^", "").replace("-", "_")
                    filename = f"{clean_symbol}_{interval}.csv"
                    filepath = output_path / filename

                    df.to_csv(filepath, index=False)

                    stats["total_bars"] += len(df)
                    stats["files_created"] += 1

                    logger.info(f"Saved {filename}: {len(df)} bars")

            except Exception as e:
                error = f"{symbol}_{interval}: {str(e)}"
                stats["errors"].append(error)
                logger.error(error)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Yahoo Finance data")
    parser.add_argument("--symbols", nargs="+", help="Symbols to download")
    parser.add_argument("--intervals", nargs="+", default=["1d"])
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--output-dir", default="./data/yahoo")

    args = parser.parse_args()

    stats = download_yahoo_data(
        symbols=args.symbols,
        intervals=args.intervals,
        start_date=args.start_date,
        output_dir=args.output_dir,
    )

    print(f"\nDownload complete:")
    print(f"  Files: {stats['files_created']}")
    print(f"  Bars: {stats['total_bars']}")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
