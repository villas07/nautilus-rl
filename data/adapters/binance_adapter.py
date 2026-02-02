"""
Binance Historical Data Adapter

Downloads FREE historical OHLCV data from Binance API.
No API key required for historical klines!

Features:
- Spot and Futures data
- All timeframes: 1m, 5m, 15m, 1h, 4h, 1d
- Data from 2017 onwards
- Automatic pagination (1000 bars per request)
- Rate limiting handled
"""

import os
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import structlog

logger = structlog.get_logger()


class BinanceHistoricalAdapter:
    """
    Adapter for downloading free historical data from Binance.

    No API key required for historical klines!
    """

    # API endpoints
    SPOT_BASE_URL = "https://api.binance.com"
    FUTURES_BASE_URL = "https://fapi.binance.com"

    # Klines endpoints
    SPOT_KLINES = "/api/v3/klines"
    FUTURES_KLINES = "/fapi/v1/klines"

    # Rate limits
    MAX_BARS_PER_REQUEST = 1000
    REQUESTS_PER_MINUTE = 1200
    REQUEST_WEIGHT = 1  # Klines endpoint weight

    # Timeframe to milliseconds
    INTERVAL_MS = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "8h": 8 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "3d": 3 * 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
        "1M": 30 * 24 * 60 * 60 * 1000,
    }

    def __init__(self, api_key: str = "", api_secret: str = ""):
        """
        Initialize adapter.

        Args:
            api_key: Optional API key (not required for historical data).
            api_secret: Optional API secret.
        """
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")

        # Setup session with retries
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.05  # 50ms between requests

    def _make_request(
        self,
        base_url: str,
        endpoint: str,
        params: Dict[str, Any],
    ) -> List[Any]:
        """Make API request with rate limiting."""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)

        url = f"{base_url}{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            self._last_request_time = time.time()

            # Check rate limits
            if "X-MBX-USED-WEIGHT-1M" in response.headers:
                used_weight = int(response.headers["X-MBX-USED-WEIGHT-1M"])
                if used_weight > 1000:
                    logger.warning(f"High API weight usage: {used_weight}/1200")
                    time.sleep(1)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        futures: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch klines (OHLCV) data.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT").
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d).
            start_time: Start datetime (UTC).
            end_time: End datetime (UTC).
            limit: Max bars per request (max 1000).
            futures: Use futures API instead of spot.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        base_url = self.FUTURES_BASE_URL if futures else self.SPOT_BASE_URL
        endpoint = self.FUTURES_KLINES if futures else self.SPOT_KLINES

        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, self.MAX_BARS_PER_REQUEST),
        }

        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)

        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        data = self._make_request(base_url, endpoint, params)

        if not data:
            return pd.DataFrame()

        # Parse response
        # [open_time, open, high, low, close, volume, close_time, quote_volume,
        #  trades, taker_buy_base, taker_buy_quote, ignore]
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Keep only essential columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]

        return df

    def fetch_all_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        futures: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch all historical klines with automatic pagination.

        Args:
            symbol: Trading pair.
            interval: Timeframe.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD), defaults to now.
            futures: Use futures API.

        Returns:
            Complete DataFrame with all bars.
        """
        start_time = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_time = (
            datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if end_date
            else datetime.now(timezone.utc)
        )

        logger.info(
            f"Fetching {symbol} {interval} from {start_date} to {end_date or 'now'}",
            futures=futures,
        )

        all_data = []
        current_start = start_time
        interval_ms = self.INTERVAL_MS.get(interval, 3600000)

        while current_start < end_time:
            df = self.fetch_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=end_time,
                limit=1000,
                futures=futures,
            )

            if df.empty:
                break

            all_data.append(df)

            # Move to next batch
            last_timestamp = df["timestamp"].max()
            current_start = last_timestamp + timedelta(milliseconds=interval_ms)

            logger.debug(
                f"Fetched {len(df)} bars, last: {last_timestamp}",
                total_bars=sum(len(d) for d in all_data),
            )

            # Small delay to respect rate limits
            time.sleep(0.1)

        if not all_data:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()

        # Combine all data
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        logger.info(
            f"Downloaded {len(result)} bars for {symbol}",
            start=result["timestamp"].min().isoformat(),
            end=result["timestamp"].max().isoformat(),
        )

        return result

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = "1h",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        futures: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of trading pairs.
            interval: Timeframe.
            start_date: Start date.
            end_date: End date.
            futures: Use futures API.

        Returns:
            Dict mapping symbol to DataFrame.
        """
        results = {}

        for symbol in symbols:
            try:
                df = self.fetch_all_klines(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    futures=futures,
                )
                results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                results[symbol] = pd.DataFrame()

        return results

    def get_available_symbols(self, futures: bool = False) -> List[str]:
        """Get list of available trading pairs."""
        if futures:
            url = f"{self.FUTURES_BASE_URL}/fapi/v1/exchangeInfo"
        else:
            url = f"{self.SPOT_BASE_URL}/api/v3/exchangeInfo"

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            symbols = [
                s["symbol"]
                for s in data["symbols"]
                if s.get("status") == "TRADING" or s.get("contractStatus") == "TRADING"
            ]

            # Filter USDT pairs (most liquid)
            usdt_symbols = [s for s in symbols if s.endswith("USDT")]

            return sorted(usdt_symbols)

        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []

    def get_earliest_timestamp(
        self,
        symbol: str,
        interval: str = "1h",
        futures: bool = False,
    ) -> Optional[datetime]:
        """Get earliest available data timestamp for a symbol."""
        base_url = self.FUTURES_BASE_URL if futures else self.SPOT_BASE_URL
        endpoint = self.FUTURES_KLINES if futures else self.SPOT_KLINES

        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": 0,
            "limit": 1,
        }

        try:
            data = self._make_request(base_url, endpoint, params)
            if data:
                return datetime.fromtimestamp(data[0][0] / 1000, tz=timezone.utc)
        except Exception as e:
            logger.error(f"Failed to get earliest timestamp: {e}")

        return None


def download_binance_data(
    symbols: Optional[List[str]] = None,
    intervals: Optional[List[str]] = None,
    start_date: str = "2020-01-01",
    output_dir: str = "/app/data/binance",
    futures: bool = False,
) -> Dict[str, int]:
    """
    Download Binance historical data and save to CSV.

    Args:
        symbols: Symbols to download (default: top 10 by volume).
        intervals: Timeframes to download.
        start_date: Start date.
        output_dir: Output directory.
        futures: Download futures data.

    Returns:
        Dict with download statistics.
    """
    adapter = BinanceHistoricalAdapter()

    symbols = symbols or [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT",
    ]

    intervals = intervals or ["1h", "4h", "1d"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"total_bars": 0, "files_created": 0, "errors": []}

    for symbol in symbols:
        for interval in intervals:
            try:
                df = adapter.fetch_all_klines(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    futures=futures,
                )

                if not df.empty:
                    # Save to CSV
                    market = "futures" if futures else "spot"
                    filename = f"{symbol}_{interval}_{market}.csv"
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
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Download Binance historical data")
    parser.add_argument("--symbols", nargs="+", help="Symbols to download")
    parser.add_argument("--intervals", nargs="+", default=["1h", "4h", "1d"])
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--output-dir", default="./data/binance")
    parser.add_argument("--futures", action="store_true")

    args = parser.parse_args()

    stats = download_binance_data(
        symbols=args.symbols,
        intervals=args.intervals,
        start_date=args.start_date,
        output_dir=args.output_dir,
        futures=args.futures,
    )

    print(f"\nDownload complete:")
    print(f"  Files: {stats['files_created']}")
    print(f"  Bars: {stats['total_bars']}")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
