"""
CryptoCompare Historical Data Adapter

FREE tier: 100,000 API calls per month
No API key required for basic endpoints!

Features:
- OHLCV data for 5000+ cryptocurrencies
- Multiple exchanges aggregated
- Data from 2010 onwards for BTC
- Full historical data available
"""

import os
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

import pandas as pd
import requests

import structlog

logger = structlog.get_logger()


class CryptoCompareAdapter:
    """
    Adapter for CryptoCompare free API.

    Rate limits:
    - No API key: 100,000 calls/month
    - Free API key: 100,000 calls/month + higher limits
    - Paid: Unlimited
    """

    BASE_URL = "https://min-api.cryptocompare.com/data/v2"

    # Available endpoints
    ENDPOINTS = {
        "histominute": "/histominute",
        "histohour": "/histohour",
        "histoday": "/histoday",
    }

    # Limits per request
    MAX_LIMIT = 2000

    # Top coins by market cap
    TOP_COINS = [
        "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "AVAX", "DOT",
        "MATIC", "LINK", "ATOM", "UNI", "LTC", "FIL", "APT", "ARB",
        "OP", "NEAR", "INJ", "FTM", "AAVE", "MKR", "SNX", "CRV",
        "DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI",
    ]

    # Stable coins for pairs
    QUOTE_CURRENCIES = ["USD", "USDT", "BTC", "ETH"]

    def __init__(self, api_key: str = ""):
        """
        Initialize adapter.

        Args:
            api_key: Optional API key for higher limits.
        """
        self.api_key = api_key or os.getenv("CRYPTOCOMPARE_API_KEY", "")
        self.session = requests.Session()

        if self.api_key:
            self.session.headers["authorization"] = f"Apikey {self.api_key}"

        self._last_request_time = 0
        self._min_interval = 0.1  # 100ms between requests

    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Make API request with rate limiting."""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            self._last_request_time = time.time()

            response.raise_for_status()
            data = response.json()

            if data.get("Response") == "Error":
                raise ValueError(data.get("Message", "Unknown error"))

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def fetch_ohlcv(
        self,
        symbol: str,
        quote: str = "USD",
        interval: str = "1h",
        limit: int = 2000,
        to_timestamp: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data.

        Args:
            symbol: Crypto symbol (e.g., "BTC").
            quote: Quote currency (USD, USDT, BTC).
            interval: Timeframe (1m, 1h, 1d).
            limit: Number of bars (max 2000).
            to_timestamp: End timestamp (Unix seconds).

        Returns:
            DataFrame with OHLCV data.
        """
        # Map interval to endpoint
        if interval in ["1m", "1min"]:
            endpoint = self.ENDPOINTS["histominute"]
        elif interval in ["1h", "1H", "1hour"]:
            endpoint = self.ENDPOINTS["histohour"]
        elif interval in ["1d", "1D", "daily"]:
            endpoint = self.ENDPOINTS["histoday"]
        else:
            raise ValueError(f"Unsupported interval: {interval}")

        params = {
            "fsym": symbol.upper(),
            "tsym": quote.upper(),
            "limit": min(limit, self.MAX_LIMIT),
        }

        if to_timestamp:
            params["toTs"] = to_timestamp

        data = self._make_request(endpoint, params)

        if not data.get("Data", {}).get("Data"):
            return pd.DataFrame()

        # Parse response
        df = pd.DataFrame(data["Data"]["Data"])

        # Rename columns
        df = df.rename(columns={
            "time": "timestamp",
            "volumefrom": "volume",
            "volumeto": "quote_volume",
        })

        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

        # Keep only essential columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        # Remove zero-volume bars (no trading)
        df = df[df["volume"] > 0]

        return df

    def fetch_all_history(
        self,
        symbol: str,
        quote: str = "USD",
        interval: str = "1h",
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch complete historical data with pagination.

        Args:
            symbol: Crypto symbol.
            quote: Quote currency.
            interval: Timeframe.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            Complete DataFrame.
        """
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        end_ts = int(
            datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            if end_date
            else datetime.now(timezone.utc).timestamp()
        )

        logger.info(f"Fetching {symbol}/{quote} {interval} from {start_date}")

        all_data = []
        current_end = end_ts

        # Calculate interval in seconds
        if interval in ["1m", "1min"]:
            interval_sec = 60
        elif interval in ["1h", "1H"]:
            interval_sec = 3600
        else:
            interval_sec = 86400

        while current_end > start_ts:
            df = self.fetch_ohlcv(
                symbol=symbol,
                quote=quote,
                interval=interval,
                limit=2000,
                to_timestamp=current_end,
            )

            if df.empty:
                break

            all_data.append(df)

            # Get earliest timestamp
            earliest = df["timestamp"].min()
            earliest_ts = int(earliest.timestamp())

            if earliest_ts <= start_ts:
                break

            # Move to earlier data
            current_end = earliest_ts - interval_sec

            logger.debug(
                f"Fetched {len(df)} bars, earliest: {earliest}",
                total=sum(len(d) for d in all_data),
            )

            time.sleep(0.1)

        if not all_data:
            return pd.DataFrame()

        # Combine and deduplicate
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        # Filter to date range
        result = result[
            (result["timestamp"] >= pd.Timestamp(start_date, tz="UTC")) &
            (result["timestamp"] <= pd.Timestamp(end_date or datetime.now(), tz="UTC"))
        ]

        logger.info(
            f"Downloaded {len(result)} bars for {symbol}/{quote}",
            start=result["timestamp"].min().isoformat() if len(result) > 0 else None,
            end=result["timestamp"].max().isoformat() if len(result) > 0 else None,
        )

        return result

    def fetch_multiple_pairs(
        self,
        symbols: List[str],
        quote: str = "USD",
        interval: str = "1h",
        start_date: str = "2015-01-01",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple pairs."""
        results = {}

        for symbol in symbols:
            try:
                df = self.fetch_all_history(
                    symbol=symbol,
                    quote=quote,
                    interval=interval,
                    start_date=start_date,
                )
                pair = f"{symbol}{quote}"
                results[pair] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                results[f"{symbol}{quote}"] = pd.DataFrame()

        return results

    def get_available_coins(self) -> List[str]:
        """Get list of all available coins."""
        url = "https://min-api.cryptocompare.com/data/all/coinlist"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("Response") == "Success":
                return list(data.get("Data", {}).keys())

        except Exception as e:
            logger.error(f"Failed to get coin list: {e}")

        return self.TOP_COINS

    def get_exchanges(self) -> List[str]:
        """Get list of supported exchanges."""
        url = "https://min-api.cryptocompare.com/data/v4/all/exchanges"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("Response") == "Success":
                return list(data.get("Data", {}).get("exchanges", {}).keys())

        except Exception as e:
            logger.error(f"Failed to get exchanges: {e}")

        return []


def download_cryptocompare_data(
    symbols: Optional[List[str]] = None,
    quote: str = "USD",
    intervals: Optional[List[str]] = None,
    start_date: str = "2017-01-01",
    output_dir: str = "/app/data/cryptocompare",
) -> Dict[str, int]:
    """
    Download CryptoCompare data and save to CSV.

    Args:
        symbols: Coins to download.
        quote: Quote currency.
        intervals: Timeframes.
        start_date: Start date.
        output_dir: Output directory.

    Returns:
        Download statistics.
    """
    adapter = CryptoCompareAdapter()

    symbols = symbols or adapter.TOP_COINS[:20]
    intervals = intervals or ["1h", "1d"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"total_bars": 0, "files_created": 0, "errors": []}

    for symbol in symbols:
        for interval in intervals:
            try:
                df = adapter.fetch_all_history(
                    symbol=symbol,
                    quote=quote,
                    interval=interval,
                    start_date=start_date,
                )

                if not df.empty:
                    filename = f"{symbol}{quote}_{interval}.csv"
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

    parser = argparse.ArgumentParser(description="Download CryptoCompare data")
    parser.add_argument("--symbols", nargs="+", help="Coins to download")
    parser.add_argument("--quote", default="USD", help="Quote currency")
    parser.add_argument("--intervals", nargs="+", default=["1h", "1d"])
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument("--output-dir", default="./data/cryptocompare")

    args = parser.parse_args()

    stats = download_cryptocompare_data(
        symbols=args.symbols,
        quote=args.quote,
        intervals=args.intervals,
        start_date=args.start_date,
        output_dir=args.output_dir,
    )

    print(f"\nDownload complete:")
    print(f"  Files: {stats['files_created']}")
    print(f"  Bars: {stats['total_bars']}")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
