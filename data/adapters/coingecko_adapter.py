"""
CoinGecko Historical Data Adapter

FREE tier (no API key):
- 10-30 calls/minute
- OHLC data (limited timeframes)
- Market data, prices, volumes
- 5000+ cryptocurrencies

Features:
- Full OHLC history for top coins
- Market cap, volume rankings
- Exchange data
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


class CoinGeckoAdapter:
    """
    Adapter for CoinGecko free API.

    Rate limits (no API key):
    - 10-30 calls/minute
    - Reduces during high load

    Data available:
    - OHLC: 1/7/14/30/90/180/365 days, max (all history)
    - Granularity depends on range
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    # Coin ID mappings (CoinGecko uses IDs, not symbols)
    COIN_IDS = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binancecoin",
        "SOL": "solana",
        "XRP": "ripple",
        "ADA": "cardano",
        "AVAX": "avalanche-2",
        "DOT": "polkadot",
        "MATIC": "matic-network",
        "LINK": "chainlink",
        "ATOM": "cosmos",
        "UNI": "uniswap",
        "LTC": "litecoin",
        "FIL": "filecoin",
        "APT": "aptos",
        "ARB": "arbitrum",
        "OP": "optimism",
        "NEAR": "near",
        "INJ": "injective-protocol",
        "FTM": "fantom",
        "AAVE": "aave",
        "MKR": "maker",
        "SNX": "synthetix-network-token",
        "CRV": "curve-dao-token",
        "DOGE": "dogecoin",
        "SHIB": "shiba-inu",
        "PEPE": "pepe",
    }

    # OHLC ranges (days)
    OHLC_RANGES = [1, 7, 14, 30, 90, 180, 365, "max"]

    def __init__(self, api_key: str = ""):
        """
        Initialize adapter.

        Args:
            api_key: Optional Pro API key.
        """
        self.api_key = api_key or os.getenv("COINGECKO_API_KEY", "")
        self.session = requests.Session()

        if self.api_key:
            self.session.headers["x-cg-pro-api-key"] = self.api_key
            self.base_url = "https://pro-api.coingecko.com/api/v3"
        else:
            self.base_url = self.BASE_URL

        self._last_request_time = 0
        self._min_interval = 2.0  # 2 seconds between requests (conservative)

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make API request with rate limiting."""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            self._last_request_time = time.time()

            # Handle rate limits
            if response.status_code == 429:
                logger.warning("Rate limited, waiting 60 seconds")
                time.sleep(60)
                return self._make_request(endpoint, params)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get_coin_id(self, symbol: str) -> str:
        """Get CoinGecko ID from symbol."""
        return self.COIN_IDS.get(symbol.upper(), symbol.lower())

    def fetch_ohlc(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: Any = "max",
    ) -> pd.DataFrame:
        """
        Fetch OHLC data.

        Granularity:
        - 1-2 days: 30-minute candles
        - 3-30 days: 4-hour candles
        - 31+ days: daily candles

        Args:
            coin_id: CoinGecko coin ID.
            vs_currency: Quote currency (usd, eur, btc).
            days: Number of days (1, 7, 14, 30, 90, 180, 365, "max").

        Returns:
            DataFrame with OHLC data.
        """
        endpoint = f"/coins/{coin_id}/ohlc"

        params = {
            "vs_currency": vs_currency.lower(),
            "days": str(days),
        }

        data = self._make_request(endpoint, params)

        if not data:
            return pd.DataFrame()

        # Parse response [timestamp, open, high, low, close]
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])

        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Add volume placeholder (OHLC endpoint doesn't include volume)
        df["volume"] = 0.0

        return df

    def fetch_market_chart(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: Any = "max",
    ) -> pd.DataFrame:
        """
        Fetch market chart data (prices, volumes, market caps).

        Granularity:
        - 1 day: 5-minute
        - 2-90 days: hourly
        - 91+ days: daily

        Args:
            coin_id: CoinGecko coin ID.
            vs_currency: Quote currency.
            days: Number of days.

        Returns:
            DataFrame with price and volume data.
        """
        endpoint = f"/coins/{coin_id}/market_chart"

        params = {
            "vs_currency": vs_currency.lower(),
            "days": str(days),
        }

        data = self._make_request(endpoint, params)

        if not data:
            return pd.DataFrame()

        # Parse prices
        prices_df = pd.DataFrame(data.get("prices", []), columns=["timestamp", "close"])
        volumes_df = pd.DataFrame(data.get("total_volumes", []), columns=["timestamp", "volume"])

        # Merge
        df = prices_df.merge(volumes_df, on="timestamp", how="outer")

        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        # Add OHLC placeholders (market_chart only gives close price)
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def fetch_historical_range(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical data for a date range.

        Args:
            coin_id: CoinGecko coin ID.
            vs_currency: Quote currency.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            DataFrame with historical data.
        """
        endpoint = f"/coins/{coin_id}/market_chart/range"

        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
        end_ts = int(
            datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
            if end_date
            else datetime.now(timezone.utc).timestamp()
        )

        params = {
            "vs_currency": vs_currency.lower(),
            "from": start_ts,
            "to": end_ts,
        }

        data = self._make_request(endpoint, params)

        if not data:
            return pd.DataFrame()

        # Parse prices and volumes
        prices_df = pd.DataFrame(data.get("prices", []), columns=["timestamp", "close"])
        volumes_df = pd.DataFrame(data.get("total_volumes", []), columns=["timestamp", "volume"])

        df = prices_df.merge(volumes_df, on="timestamp", how="outer")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]

        logger.info(
            f"Fetched {coin_id}",
            bars=len(df),
            start=df["timestamp"].min().isoformat() if len(df) > 0 else None,
            end=df["timestamp"].max().isoformat() if len(df) > 0 else None,
        )

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def get_coin_list(self) -> List[Dict[str, str]]:
        """Get list of all supported coins."""
        endpoint = "/coins/list"
        return self._make_request(endpoint)

    def get_top_coins(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get top coins by market cap."""
        endpoint = "/coins/markets"

        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": limit,
            "page": 1,
        }

        return self._make_request(endpoint, params)


def download_coingecko_data(
    symbols: Optional[List[str]] = None,
    vs_currency: str = "usd",
    start_date: str = "2017-01-01",
    output_dir: str = "/app/data/coingecko",
) -> Dict[str, int]:
    """
    Download CoinGecko data and save to CSV.

    Args:
        symbols: Coin symbols to download.
        vs_currency: Quote currency.
        start_date: Start date.
        output_dir: Output directory.

    Returns:
        Download statistics.
    """
    adapter = CoinGeckoAdapter()

    symbols = symbols or list(adapter.COIN_IDS.keys())[:20]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"total_bars": 0, "files_created": 0, "errors": []}

    for symbol in symbols:
        try:
            coin_id = adapter.get_coin_id(symbol)

            df = adapter.fetch_historical_range(
                coin_id=coin_id,
                vs_currency=vs_currency,
                start_date=start_date,
            )

            if not df.empty:
                filename = f"{symbol}{vs_currency.upper()}_1d.csv"
                filepath = output_path / filename

                df.to_csv(filepath, index=False)

                stats["total_bars"] += len(df)
                stats["files_created"] += 1

                logger.info(f"Saved {filename}: {len(df)} bars")

        except Exception as e:
            error = f"{symbol}: {str(e)}"
            stats["errors"].append(error)
            logger.error(error)

        # Respect rate limits
        time.sleep(2)

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download CoinGecko data")
    parser.add_argument("--symbols", nargs="+", help="Coins to download")
    parser.add_argument("--currency", default="usd", help="Quote currency")
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument("--output-dir", default="./data/coingecko")

    args = parser.parse_args()

    stats = download_coingecko_data(
        symbols=args.symbols,
        vs_currency=args.currency,
        start_date=args.start_date,
        output_dir=args.output_dir,
    )

    print(f"\nDownload complete:")
    print(f"  Files: {stats['files_created']}")
    print(f"  Bars: {stats['total_bars']}")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
