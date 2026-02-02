"""
EOD Historical Data Adapter

Premium data source for global markets:
- Europe: LSE, XETRA, Euronext, SIX, BME, BIT, etc.
- Asia: TSE (Tokyo), HKEX, SSE, SZSE, NSE, BSE, KRX, TWSE, SGX, etc.
- Americas: NYSE, NASDAQ, TSX, BOVESPA, BMV, etc.
- Indices, ETFs, Forex, Crypto, Bonds, Commodities

API Documentation: https://eodhistoricaldata.com/financial-apis/
"""

import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import time

import pandas as pd
import requests

import structlog

logger = structlog.get_logger()


class EODHistoricalDataAdapter:
    """
    Adapter for EOD Historical Data API.

    Provides access to:
    - 70+ global exchanges
    - 150,000+ tickers
    - 30+ years of historical data
    - Real-time and delayed quotes
    - Fundamental data
    - Economic indicators

    Rate limits: 100,000 API calls/day on standard plan
    """

    BASE_URL = "https://eodhistoricaldata.com/api"

    # European Exchanges
    EUROPEAN_EXCHANGES = {
        "LSE": "London Stock Exchange (UK)",
        "XETRA": "Deutsche Borse (Germany)",
        "PA": "Euronext Paris (France)",
        "AS": "Euronext Amsterdam (Netherlands)",
        "BR": "Euronext Brussels (Belgium)",
        "LS": "Euronext Lisbon (Portugal)",
        "MI": "Borsa Italiana (Italy)",
        "MC": "Bolsa de Madrid (Spain)",
        "SW": "SIX Swiss Exchange",
        "VI": "Vienna Stock Exchange (Austria)",
        "ST": "Nasdaq Stockholm (Sweden)",
        "HE": "Nasdaq Helsinki (Finland)",
        "CO": "Nasdaq Copenhagen (Denmark)",
        "OL": "Oslo Stock Exchange (Norway)",
        "IR": "Euronext Dublin (Ireland)",
        "WAR": "Warsaw Stock Exchange (Poland)",
        "PR": "Prague Stock Exchange (Czech)",
        "BUD": "Budapest Stock Exchange (Hungary)",
        "AT": "Athens Stock Exchange (Greece)",
        "IS": "Istanbul Stock Exchange (Turkey)",
    }

    # Asian Exchanges
    ASIAN_EXCHANGES = {
        "TSE": "Tokyo Stock Exchange (Japan)",
        "HK": "Hong Kong Stock Exchange",
        "SS": "Shanghai Stock Exchange (China)",
        "SZ": "Shenzhen Stock Exchange (China)",
        "NSE": "National Stock Exchange (India)",
        "BSE": "Bombay Stock Exchange (India)",
        "KO": "Korea Stock Exchange (KRX)",
        "TW": "Taiwan Stock Exchange",
        "SG": "Singapore Exchange",
        "KL": "Bursa Malaysia",
        "BK": "Stock Exchange of Thailand",
        "JK": "Indonesia Stock Exchange",
        "PH": "Philippine Stock Exchange",
        "VN": "Ho Chi Minh Stock Exchange (Vietnam)",
        "AU": "Australian Securities Exchange",
        "NZ": "New Zealand Exchange",
    }

    # Major European Stocks
    TOP_EUROPEAN_STOCKS = {
        "LSE": ["SHEL", "AZN", "HSBA", "ULVR", "BP", "GSK", "RIO", "DGE", "BATS", "LSEG"],
        "XETRA": ["SAP", "SIE", "ALV", "DTE", "BAS", "BAYN", "MRK", "BMW", "VOW3", "ADS"],
        "PA": ["MC", "OR", "SAN", "AI", "TTE", "SU", "AIR", "BNP", "CS", "DG"],
        "AS": ["ASML", "ADYEN", "INGA", "PHIA", "UNA", "RAND", "KPN", "AKZA", "WKL", "HEIA"],
        "MI": ["ENEL", "ENI", "ISP", "UCG", "STM", "G", "RACE", "LDO", "CNHI", "PRY"],
        "MC": ["ITX", "IBE", "SAN", "BBVA", "TEF", "REP", "FER", "AMS", "GRF", "IAG"],
        "SW": ["NESN", "ROG", "NOVN", "UBSG", "ABBN", "CSGN", "ZURN", "SREN", "GIVN", "LONN"],
    }

    # Major Asian Stocks
    TOP_ASIAN_STOCKS = {
        "TSE": ["7203", "6758", "9984", "8306", "6861", "9432", "6501", "7267", "8035", "4502"],  # Toyota, Sony, SoftBank, etc.
        "HK": ["0700", "9988", "1299", "0005", "2318", "0941", "0388", "0883", "1398", "2628"],  # Tencent, Alibaba, etc.
        "SS": ["600519", "601318", "600036", "601166", "600276", "603259", "600900", "601012", "600030", "600887"],
        "NSE": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK"],
        "KO": ["005930", "000660", "035420", "005380", "051910", "035720", "006400", "068270", "028260", "003550"],  # Samsung, etc.
        "SG": ["D05", "O39", "U11", "Z74", "BN4", "C52", "G13", "S58", "C38U", "A17U"],
    }

    # European Indices
    EUROPEAN_INDICES = [
        "FTSE.INDX",       # FTSE 100
        "GDAXI.INDX",      # DAX
        "FCHI.INDX",       # CAC 40
        "AEX.INDX",        # AEX (Amsterdam)
        "IBEX.INDX",       # IBEX 35
        "FTSEMIB.INDX",    # FTSE MIB
        "SMI.INDX",        # Swiss Market Index
        "STOXX50E.INDX",   # Euro Stoxx 50
    ]

    # Asian Indices
    ASIAN_INDICES = [
        "N225.INDX",       # Nikkei 225
        "HSI.INDX",        # Hang Seng
        "SSEC.INDX",       # Shanghai Composite
        "SZCOMP.INDX",     # Shenzhen Composite
        "NSEI.INDX",       # Nifty 50
        "BSESN.INDX",      # BSE Sensex
        "KS11.INDX",       # KOSPI
        "TWII.INDX",       # TAIEX
        "STI.INDX",        # Straits Times Index
        "AXJO.INDX",       # ASX 200
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize adapter.

        Args:
            api_key: EOD Historical Data API key.
                     If not provided, reads from EOD_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("EOD_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EOD API key required. Set EOD_API_KEY env var or pass api_key parameter."
            )
        self._request_count = 0
        self._last_request = 0

    def _rate_limit(self):
        """Apply rate limiting (max 20 requests/second for safety)."""
        elapsed = time.time() - self._last_request
        if elapsed < 0.05:  # 50ms minimum between requests
            time.sleep(0.05 - elapsed)
        self._last_request = time.time()
        self._request_count += 1

    def _request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict:
        """Make API request."""
        self._rate_limit()

        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        params["api_token"] = self.api_key
        params["fmt"] = "json"

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"EOD API error: {e}")
            raise

    def fetch_ohlcv(
        self,
        symbol: str,
        exchange: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "d",  # d=daily, w=weekly, m=monthly
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Ticker symbol (e.g., "SHEL" for Shell on LSE).
            exchange: Exchange code (e.g., "LSE", "XETRA", "TSE").
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            period: d=daily, w=weekly, m=monthly.

        Returns:
            DataFrame with timestamp, open, high, low, close, volume, adjusted_close.
        """
        ticker = f"{symbol}.{exchange}"

        params = {"period": period}
        if start_date:
            params["from"] = start_date
        if end_date:
            params["to"] = end_date

        try:
            data = self._request(f"eod/{ticker}", params)

            if not data:
                logger.warning(f"No data for {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # Standardize columns
            df = df.rename(columns={
                "date": "timestamp",
                "adjusted_close": "adj_close",
            })

            # Parse timestamp
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

            # Ensure numeric columns
            for col in ["open", "high", "low", "close", "volume", "adj_close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)

            logger.info(
                f"Fetched {ticker}",
                bars=len(df),
                start=df["timestamp"].min().isoformat() if not df.empty else "N/A",
                end=df["timestamp"].max().isoformat() if not df.empty else "N/A",
            )

            return df

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def fetch_intraday(
        self,
        symbol: str,
        exchange: str,
        interval: str = "1h",  # 1m, 5m, 1h
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch intraday data (requires subscription upgrade).

        Args:
            symbol: Ticker symbol.
            exchange: Exchange code.
            interval: 1m, 5m, or 1h.
            start_timestamp: Unix timestamp start.
            end_timestamp: Unix timestamp end.

        Returns:
            DataFrame with intraday data.
        """
        ticker = f"{symbol}.{exchange}"

        params = {"interval": interval}
        if start_timestamp:
            params["from"] = start_timestamp
        if end_timestamp:
            params["to"] = end_timestamp

        try:
            data = self._request(f"intraday/{ticker}", params)

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching intraday {ticker}: {e}")
            return pd.DataFrame()

    def fetch_fundamentals(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Fetch fundamental data for a symbol."""
        ticker = f"{symbol}.{exchange}"
        return self._request(f"fundamentals/{ticker}")

    def fetch_exchange_symbols(self, exchange: str) -> List[Dict[str, Any]]:
        """Get all symbols for an exchange."""
        return self._request(f"exchange-symbol-list/{exchange}")

    def fetch_european_stocks(
        self,
        exchanges: Optional[List[str]] = None,
        start_date: str = "2015-01-01",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for major European stocks.

        Args:
            exchanges: List of exchange codes (default: all European).
            start_date: Start date.

        Returns:
            Dict mapping ticker to DataFrame.
        """
        exchanges = exchanges or list(self.TOP_EUROPEAN_STOCKS.keys())
        results = {}

        for exchange in exchanges:
            symbols = self.TOP_EUROPEAN_STOCKS.get(exchange, [])

            for symbol in symbols:
                ticker = f"{symbol}.{exchange}"
                df = self.fetch_ohlcv(
                    symbol=symbol,
                    exchange=exchange,
                    start_date=start_date,
                )
                if not df.empty:
                    results[ticker] = df

        return results

    def fetch_asian_stocks(
        self,
        exchanges: Optional[List[str]] = None,
        start_date: str = "2015-01-01",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for major Asian stocks.

        Args:
            exchanges: List of exchange codes (default: all Asian).
            start_date: Start date.

        Returns:
            Dict mapping ticker to DataFrame.
        """
        exchanges = exchanges or list(self.TOP_ASIAN_STOCKS.keys())
        results = {}

        for exchange in exchanges:
            symbols = self.TOP_ASIAN_STOCKS.get(exchange, [])

            for symbol in symbols:
                ticker = f"{symbol}.{exchange}"
                df = self.fetch_ohlcv(
                    symbol=symbol,
                    exchange=exchange,
                    start_date=start_date,
                )
                if not df.empty:
                    results[ticker] = df

        return results

    def fetch_indices(
        self,
        region: str = "all",  # "europe", "asia", "all"
        start_date: str = "2015-01-01",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch index data.

        Args:
            region: "europe", "asia", or "all".
            start_date: Start date.

        Returns:
            Dict mapping index to DataFrame.
        """
        if region == "europe":
            indices = self.EUROPEAN_INDICES
        elif region == "asia":
            indices = self.ASIAN_INDICES
        else:
            indices = self.EUROPEAN_INDICES + self.ASIAN_INDICES

        results = {}

        for index in indices:
            # Index format: NAME.INDX
            symbol = index.replace(".INDX", "")
            try:
                params = {"from": start_date}
                data = self._request(f"eod/{index}", params)

                if data:
                    df = pd.DataFrame(data)
                    df["timestamp"] = pd.to_datetime(df["date"])
                    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
                    df = df.sort_values("timestamp").reset_index(drop=True)
                    results[index] = df

                    logger.info(f"Fetched index {index}", bars=len(df))

            except Exception as e:
                logger.error(f"Error fetching index {index}: {e}")

        return results

    def get_exchange_list(self) -> List[Dict[str, Any]]:
        """Get list of all available exchanges."""
        return self._request("exchanges-list")


def download_eod_data(
    api_key: Optional[str] = None,
    regions: List[str] = None,
    start_date: str = "2015-01-01",
    output_dir: str = "./data/eod",
) -> Dict[str, int]:
    """
    Download EOD Historical data and save to CSV.

    Args:
        api_key: API key (or set EOD_API_KEY env var).
        regions: Regions to download ("europe", "asia").
        start_date: Start date.
        output_dir: Output directory.

    Returns:
        Download statistics.
    """
    adapter = EODHistoricalDataAdapter(api_key)
    regions = regions or ["europe", "asia"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"total_bars": 0, "files_created": 0, "errors": []}

    # Download stocks
    for region in regions:
        logger.info(f"Downloading {region} stocks...")

        if region == "europe":
            data = adapter.fetch_european_stocks(start_date=start_date)
        elif region == "asia":
            data = adapter.fetch_asian_stocks(start_date=start_date)
        else:
            continue

        for ticker, df in data.items():
            if not df.empty:
                clean_ticker = ticker.replace(".", "_")
                filename = f"{clean_ticker}_1d.csv"
                filepath = output_path / filename

                df.to_csv(filepath, index=False)
                stats["total_bars"] += len(df)
                stats["files_created"] += 1

    # Download indices
    logger.info("Downloading indices...")
    indices_data = adapter.fetch_indices(region="all", start_date=start_date)

    for index, df in indices_data.items():
        if not df.empty:
            clean_name = index.replace(".", "_")
            filename = f"{clean_name}_1d.csv"
            filepath = output_path / filename

            df.to_csv(filepath, index=False)
            stats["total_bars"] += len(df)
            stats["files_created"] += 1

    logger.info(
        "Download complete",
        files=stats["files_created"],
        bars=stats["total_bars"],
    )

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download EOD Historical Data")
    parser.add_argument("--api-key", help="EOD API key")
    parser.add_argument("--regions", nargs="+", default=["europe", "asia"])
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--output-dir", default="./data/eod")
    parser.add_argument("--test", action="store_true", help="Test connection only")

    args = parser.parse_args()

    if args.test:
        # Test connection
        adapter = EODHistoricalDataAdapter(args.api_key)
        exchanges = adapter.get_exchange_list()
        print(f"Connected! {len(exchanges)} exchanges available.")

        # Test fetch
        df = adapter.fetch_ohlcv("SHEL", "LSE", start_date="2024-01-01")
        print(f"Test fetch Shell (LSE): {len(df)} bars")
    else:
        stats = download_eod_data(
            api_key=args.api_key,
            regions=args.regions,
            start_date=args.start_date,
            output_dir=args.output_dir,
        )

        print(f"\nDownload complete:")
        print(f"  Files: {stats['files_created']}")
        print(f"  Bars: {stats['total_bars']}")
