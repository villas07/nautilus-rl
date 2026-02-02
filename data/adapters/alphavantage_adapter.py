"""
Alpha Vantage Historical Data Adapter

FREE tier:
- 25 API requests per day
- Full historical data access

Features:
- US Stocks (full history)
- Forex pairs (intraday and daily)
- Crypto pairs
- Technical indicators
- Fundamental data
"""

import os
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path
from io import StringIO

import pandas as pd
import requests

import structlog

logger = structlog.get_logger()


class AlphaVantageAdapter:
    """
    Adapter for Alpha Vantage free API.

    Rate limits:
    - Free: 25 requests/day
    - Premium: Higher limits

    Data available:
    - TIME_SERIES_INTRADAY (1min, 5min, 15min, 30min, 60min)
    - TIME_SERIES_DAILY (full history since 2000)
    - TIME_SERIES_DAILY_ADJUSTED (with splits/dividends)
    - FX_INTRADAY, FX_DAILY
    - CRYPTO_INTRADAY, DIGITAL_CURRENCY_DAILY
    """

    BASE_URL = "https://www.alphavantage.co/query"

    # Output size options
    OUTPUT_COMPACT = "compact"  # Last 100 data points
    OUTPUT_FULL = "full"       # Full history (20+ years)

    def __init__(self, api_key: str = ""):
        """
        Initialize adapter.

        Args:
            api_key: Alpha Vantage API key (free from alphavantage.co).
        """
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY", "")

        if not self.api_key:
            logger.warning(
                "No Alpha Vantage API key. Get free key at: "
                "https://www.alphavantage.co/support/#api-key"
            )

        self.session = requests.Session()
        self._request_count = 0
        self._last_request_time = 0
        self._min_interval = 12  # ~5 requests/minute with free tier

    def _make_request(self, params: Dict[str, Any]) -> pd.DataFrame:
        """Make API request with rate limiting."""
        if not self.api_key:
            raise ValueError("API key required. Get free key at alphavantage.co")

        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        params["apikey"] = self.api_key
        params["datatype"] = "csv"

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=60)
            self._last_request_time = time.time()
            self._request_count += 1

            response.raise_for_status()

            # Check for error messages
            if "Error Message" in response.text:
                raise ValueError(f"API Error: {response.text[:200]}")

            if "Note" in response.text and "API call frequency" in response.text:
                logger.warning("Rate limit warning received")

            # Parse CSV
            df = pd.read_csv(StringIO(response.text))

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def fetch_daily_stock(
        self,
        symbol: str,
        adjusted: bool = True,
        full_history: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch daily stock data.

        Args:
            symbol: Stock symbol (e.g., "AAPL").
            adjusted: Include splits/dividend adjustments.
            full_history: Get full 20+ year history.

        Returns:
            DataFrame with OHLCV data.
        """
        function = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"

        params = {
            "function": function,
            "symbol": symbol.upper(),
            "outputsize": self.OUTPUT_FULL if full_history else self.OUTPUT_COMPACT,
        }

        df = self._make_request(params)

        if df.empty:
            return df

        # Standardize columns
        df = df.rename(columns={
            "timestamp": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adjusted_close": "adj_close",
            "volume": "volume",
            "dividend_amount": "dividend",
            "split_coefficient": "split",
        })

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
            # First column is usually timestamp
            df = df.rename(columns={df.columns[0]: "timestamp"})

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")

        # Keep only essential columns
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        if "adj_close" in df.columns:
            cols.append("adj_close")
        df = df[[c for c in cols if c in df.columns]]

        logger.info(
            f"Fetched {symbol}",
            bars=len(df),
            start=df["timestamp"].min().isoformat() if len(df) > 0 else None,
            end=df["timestamp"].max().isoformat() if len(df) > 0 else None,
        )

        return df

    def fetch_intraday_stock(
        self,
        symbol: str,
        interval: str = "60min",
        full_history: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch intraday stock data.

        Note: Full intraday history requires premium subscription.

        Args:
            symbol: Stock symbol.
            interval: 1min, 5min, 15min, 30min, 60min.
            full_history: Get extended history (premium only).

        Returns:
            DataFrame with OHLCV data.
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol.upper(),
            "interval": interval,
            "outputsize": self.OUTPUT_FULL if full_history else self.OUTPUT_COMPACT,
        }

        df = self._make_request(params)

        if df.empty:
            return df

        # Standardize
        df = df.rename(columns={df.columns[0]: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")

        return df

    def fetch_forex_daily(
        self,
        from_symbol: str,
        to_symbol: str,
        full_history: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch daily forex data.

        Args:
            from_symbol: Base currency (e.g., "EUR").
            to_symbol: Quote currency (e.g., "USD").
            full_history: Get full history.

        Returns:
            DataFrame with OHLC data.
        """
        params = {
            "function": "FX_DAILY",
            "from_symbol": from_symbol.upper(),
            "to_symbol": to_symbol.upper(),
            "outputsize": self.OUTPUT_FULL if full_history else self.OUTPUT_COMPACT,
        }

        df = self._make_request(params)

        if df.empty:
            return df

        df = df.rename(columns={df.columns[0]: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["volume"] = 0  # Forex doesn't have volume
        df = df.sort_values("timestamp")

        return df

    def fetch_crypto_daily(
        self,
        symbol: str,
        market: str = "USD",
    ) -> pd.DataFrame:
        """
        Fetch daily crypto data.

        Args:
            symbol: Crypto symbol (e.g., "BTC").
            market: Quote currency.

        Returns:
            DataFrame with OHLCV data.
        """
        params = {
            "function": "DIGITAL_CURRENCY_DAILY",
            "symbol": symbol.upper(),
            "market": market.upper(),
        }

        df = self._make_request(params)

        if df.empty:
            return df

        # Crypto has multiple columns for different markets
        df = df.rename(columns={df.columns[0]: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Extract USD columns
        for col in ["open", "high", "low", "close"]:
            usd_col = f"{col} (USD)"
            if usd_col in df.columns:
                df[col] = df[usd_col]

        df = df.sort_values("timestamp")

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def fetch_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close",
    ) -> pd.DataFrame:
        """
        Fetch technical indicator data.

        Indicators: SMA, EMA, RSI, MACD, BBANDS, ADX, ATR, etc.

        Args:
            symbol: Stock symbol.
            indicator: Indicator name.
            interval: Time interval.
            time_period: Lookback period.
            series_type: Price type (close, open, high, low).

        Returns:
            DataFrame with indicator values.
        """
        params = {
            "function": indicator.upper(),
            "symbol": symbol.upper(),
            "interval": interval,
            "time_period": time_period,
            "series_type": series_type,
        }

        return self._make_request(params)

    def get_request_count(self) -> int:
        """Get number of requests made today."""
        return self._request_count


def download_alphavantage_data(
    symbols: Optional[List[str]] = None,
    data_type: str = "stock",
    output_dir: str = "/app/data/alphavantage",
) -> Dict[str, int]:
    """
    Download Alpha Vantage data and save to CSV.

    Note: Limited to 25 requests/day with free tier.

    Args:
        symbols: Symbols to download.
        data_type: "stock", "forex", or "crypto".
        output_dir: Output directory.

    Returns:
        Download statistics.
    """
    adapter = AlphaVantageAdapter()

    if data_type == "stock":
        symbols = symbols or ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]
    elif data_type == "forex":
        symbols = symbols or ["EURUSD", "GBPUSD", "USDJPY"]
    elif data_type == "crypto":
        symbols = symbols or ["BTC", "ETH"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"total_bars": 0, "files_created": 0, "errors": [], "requests": 0}

    for symbol in symbols:
        try:
            if data_type == "stock":
                df = adapter.fetch_daily_stock(symbol)
            elif data_type == "forex":
                # Parse forex pairs
                if len(symbol) == 6:
                    from_sym = symbol[:3]
                    to_sym = symbol[3:]
                else:
                    from_sym = symbol.split("/")[0]
                    to_sym = symbol.split("/")[1]
                df = adapter.fetch_forex_daily(from_sym, to_sym)
            elif data_type == "crypto":
                df = adapter.fetch_crypto_daily(symbol)
            else:
                continue

            if not df.empty:
                filename = f"{symbol}_daily.csv"
                filepath = output_path / filename

                df.to_csv(filepath, index=False)

                stats["total_bars"] += len(df)
                stats["files_created"] += 1
                stats["requests"] = adapter.get_request_count()

                logger.info(f"Saved {filename}: {len(df)} bars")

        except Exception as e:
            error = f"{symbol}: {str(e)}"
            stats["errors"].append(error)
            logger.error(error)

        # Check if hitting daily limit
        if adapter.get_request_count() >= 25:
            logger.warning("Daily request limit reached (25)")
            break

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Alpha Vantage data")
    parser.add_argument("--symbols", nargs="+", help="Symbols to download")
    parser.add_argument("--type", default="stock", choices=["stock", "forex", "crypto"])
    parser.add_argument("--output-dir", default="./data/alphavantage")

    args = parser.parse_args()

    stats = download_alphavantage_data(
        symbols=args.symbols,
        data_type=args.type,
        output_dir=args.output_dir,
    )

    print(f"\nDownload complete:")
    print(f"  Files: {stats['files_created']}")
    print(f"  Bars: {stats['total_bars']}")
    print(f"  Requests used: {stats['requests']}/25")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
