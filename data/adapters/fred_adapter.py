"""
FRED (Federal Reserve Economic Data) Adapter

100% FREE - No API key required for basic access
API key available for higher limits: https://fred.stlouisfed.org/docs/api/api_key.html

Features:
- 800,000+ economic time series
- GDP, unemployment, inflation, interest rates
- Housing, manufacturing, trade data
- International economic indicators

Essential for macro regime detection in RL models.
"""

import os
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path

import pandas as pd
import requests

import structlog

logger = structlog.get_logger()


class FREDAdapter:
    """
    Adapter for FRED (Federal Reserve Economic Data).

    Free access, API key optional for higher limits.
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    # Key economic indicators for trading
    MACRO_INDICATORS = {
        # GDP and Growth
        "GDP": "Gross Domestic Product",
        "GDPC1": "Real GDP",
        "A191RL1Q225SBEA": "Real GDP Growth Rate",

        # Employment
        "UNRATE": "Unemployment Rate",
        "PAYEMS": "Total Nonfarm Payrolls",
        "ICSA": "Initial Jobless Claims",
        "CIVPART": "Labor Force Participation Rate",

        # Inflation
        "CPIAUCSL": "Consumer Price Index",
        "CPILFESL": "Core CPI (Ex Food & Energy)",
        "PCEPI": "PCE Price Index",
        "PCEPILFE": "Core PCE",
        "T10YIE": "10-Year Breakeven Inflation",

        # Interest Rates
        "FEDFUNDS": "Federal Funds Rate",
        "DFF": "Federal Funds Effective Rate (Daily)",
        "DGS2": "2-Year Treasury Yield",
        "DGS10": "10-Year Treasury Yield",
        "DGS30": "30-Year Treasury Yield",
        "T10Y2Y": "10Y-2Y Treasury Spread (Yield Curve)",
        "T10Y3M": "10Y-3M Treasury Spread",

        # Financial Conditions
        "VIXCLS": "VIX (CBOE Volatility Index)",
        "BAMLH0A0HYM2": "High Yield Bond Spread",
        "TEDRATE": "TED Spread",
        "DCOILWTICO": "WTI Crude Oil Price",

        # Money Supply
        "M2SL": "M2 Money Stock",
        "BOGMBASE": "Monetary Base",

        # Housing
        "CSUSHPINSA": "Case-Shiller Home Price Index",
        "HOUST": "Housing Starts",
        "PERMIT": "Building Permits",
        "MORTGAGE30US": "30-Year Mortgage Rate",

        # Manufacturing & Business
        "INDPRO": "Industrial Production Index",
        "UMCSENT": "Consumer Sentiment",
        "RSXFS": "Retail Sales",
        "DGORDER": "Durable Goods Orders",

        # International
        "DEXUSEU": "USD/EUR Exchange Rate",
        "DTWEXBGS": "Trade Weighted Dollar Index",
    }

    def __init__(self, api_key: str = ""):
        """
        Initialize adapter.

        Args:
            api_key: FRED API key (optional, get free at fred.stlouisfed.org).
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY", "")
        self.session = requests.Session()

        self._last_request_time = 0
        self._min_interval = 0.5  # 500ms between requests

    def _make_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Make API request."""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        if self.api_key:
            params["api_key"] = self.api_key

        params["file_type"] = "json"

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            self._last_request_time = time.time()

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch a FRED series.

        Args:
            series_id: FRED series ID (e.g., "GDP", "UNRATE").
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            frequency: Frequency (d, w, bw, m, q, sa, a).

        Returns:
            DataFrame with date and value.
        """
        params = {
            "series_id": series_id,
        }

        if start_date:
            params["observation_start"] = start_date

        if end_date:
            params["observation_end"] = end_date

        if frequency:
            params["frequency"] = frequency

        data = self._make_request("series/observations", params)

        if not data.get("observations"):
            return pd.DataFrame()

        df = pd.DataFrame(data["observations"])

        # Parse data
        df["timestamp"] = pd.to_datetime(df["date"], utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Drop missing values
        df = df.dropna(subset=["value"])

        df = df[["timestamp", "value"]].sort_values("timestamp")

        logger.info(
            f"Fetched {series_id}",
            observations=len(df),
            start=df["timestamp"].min().isoformat() if len(df) > 0 else None,
            end=df["timestamp"].max().isoformat() if len(df) > 0 else None,
        )

        return df

    def fetch_multiple_series(
        self,
        series_ids: List[str],
        start_date: str = "2000-01-01",
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch multiple series.

        Args:
            series_ids: List of series IDs.
            start_date: Start date.
            end_date: End date.

        Returns:
            Dict mapping series ID to DataFrame.
        """
        results = {}

        for series_id in series_ids:
            try:
                df = self.fetch_series(
                    series_id=series_id,
                    start_date=start_date,
                    end_date=end_date,
                )
                results[series_id] = df
            except Exception as e:
                logger.error(f"Failed to fetch {series_id}: {e}")
                results[series_id] = pd.DataFrame()

        return results

    def fetch_all_macro_indicators(
        self,
        start_date: str = "2000-01-01",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all key macro indicators.

        Returns:
            Dict of all macro indicator DataFrames.
        """
        return self.fetch_multiple_series(
            series_ids=list(self.MACRO_INDICATORS.keys()),
            start_date=start_date,
        )

    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """Get metadata about a series."""
        params = {"series_id": series_id}
        data = self._make_request("series", params)
        return data.get("seriess", [{}])[0]

    def search_series(
        self,
        search_text: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search for series by text."""
        params = {
            "search_text": search_text,
            "limit": limit,
        }
        data = self._make_request("series/search", params)
        return data.get("seriess", [])

    def create_macro_features(
        self,
        start_date: str = "2000-01-01",
        resample_freq: str = "D",
    ) -> pd.DataFrame:
        """
        Create a DataFrame of macro features for RL.

        Resamples all indicators to daily and forward-fills.

        Args:
            start_date: Start date.
            resample_freq: Resampling frequency (D, W, M).

        Returns:
            DataFrame with all macro features.
        """
        data = self.fetch_all_macro_indicators(start_date)

        # Merge all series
        merged = None

        for series_id, df in data.items():
            if df.empty:
                continue

            df = df.rename(columns={"value": series_id})
            df = df.set_index("timestamp")

            # Resample to target frequency
            df = df.resample(resample_freq).last().ffill()

            if merged is None:
                merged = df
            else:
                merged = merged.join(df, how="outer")

        if merged is None:
            return pd.DataFrame()

        # Forward fill all
        merged = merged.ffill()

        # Reset index
        merged = merged.reset_index()

        logger.info(
            f"Created macro features",
            features=len(merged.columns) - 1,
            rows=len(merged),
        )

        return merged


def download_fred_data(
    series_ids: Optional[List[str]] = None,
    start_date: str = "2000-01-01",
    output_dir: str = "/app/data/fred",
) -> Dict[str, int]:
    """
    Download FRED data and save to CSV.

    Args:
        series_ids: Series to download (default: all macro indicators).
        start_date: Start date.
        output_dir: Output directory.

    Returns:
        Download statistics.
    """
    adapter = FREDAdapter()

    series_ids = series_ids or list(adapter.MACRO_INDICATORS.keys())

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {"total_observations": 0, "files_created": 0, "errors": []}

    for series_id in series_ids:
        try:
            df = adapter.fetch_series(
                series_id=series_id,
                start_date=start_date,
            )

            if not df.empty:
                filename = f"{series_id}.csv"
                filepath = output_path / filename

                df.to_csv(filepath, index=False)

                stats["total_observations"] += len(df)
                stats["files_created"] += 1

                logger.info(f"Saved {filename}: {len(df)} observations")

        except Exception as e:
            error = f"{series_id}: {str(e)}"
            stats["errors"].append(error)
            logger.error(error)

    # Also save combined macro features
    try:
        macro_df = adapter.create_macro_features(start_date)
        if not macro_df.empty:
            macro_df.to_csv(output_path / "macro_features.csv", index=False)
            logger.info(f"Saved combined macro features: {len(macro_df)} rows")
    except Exception as e:
        logger.error(f"Failed to create macro features: {e}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download FRED data")
    parser.add_argument("--series", nargs="+", help="Series IDs to download")
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--output-dir", default="./data/fred")

    args = parser.parse_args()

    stats = download_fred_data(
        series_ids=args.series,
        start_date=args.start_date,
        output_dir=args.output_dir,
    )

    print(f"\nDownload complete:")
    print(f"  Files: {stats['files_created']}")
    print(f"  Observations: {stats['total_observations']}")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
