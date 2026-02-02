"""
Base Adapter Interface

Defines the contract that all data source adapters must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FetchResult:
    """Result of a data fetch operation."""

    success: bool
    data: pd.DataFrame
    source: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    bars_fetched: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(
        cls,
        source: str,
        symbol: str,
        timeframe: str,
        error: Optional[str] = None,
    ) -> "FetchResult":
        """Create empty result (failed fetch)."""
        return cls(
            success=False,
            data=pd.DataFrame(),
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            start_date=datetime.min,
            end_date=datetime.min,
            bars_fetched=0,
            error=error,
        )


class BaseAdapter(ABC):
    """
    Abstract base class for data source adapters.

    All adapters must implement:
    - fetch(): Synchronous data fetch
    - get_available_symbols(): List of available symbols

    Optional methods:
    - fetch_async(): Asynchronous data fetch
    - get_earliest_date(): Earliest available data
    - validate_symbol(): Check if symbol is valid
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return adapter name."""
        pass

    @property
    @abstractmethod
    def supported_timeframes(self) -> List[str]:
        """Return list of supported timeframes."""
        pass

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> FetchResult:
        """
        Fetch historical OHLCV data.

        Args:
            symbol: Instrument symbol (e.g., "AAPL", "BTCUSDT").
            timeframe: Bar timeframe (e.g., "1m", "1h", "1d").
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format (optional).

        Returns:
            FetchResult containing DataFrame with columns:
            - timestamp: datetime64[ns, UTC]
            - open: float64
            - high: float64
            - low: float64
            - close: float64
            - volume: float64
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of symbols available from this source."""
        pass

    def fetch_async(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> FetchResult:
        """
        Asynchronous fetch (default: delegates to sync fetch).

        Override in adapters that support async operations.
        """
        return self.fetch(symbol, timeframe, start_date, end_date)

    def get_earliest_date(self, symbol: str) -> Optional[str]:
        """
        Get earliest available data date for a symbol.

        Returns:
            Date string in YYYY-MM-DD format, or None if unknown.
        """
        return None

    def validate_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is valid for this source.

        Default implementation: return True.
        Override for sources with specific symbol formats.
        """
        return True

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to source-specific format.

        Override in adapters that use different symbol conventions.
        """
        return symbol

    def normalize_timeframe(self, timeframe: str) -> str:
        """
        Normalize timeframe to source-specific format.

        Override in adapters that use different timeframe conventions.
        """
        return timeframe

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame to standard format.

        Ensures columns are:
        - timestamp: datetime64[ns, UTC]
        - open, high, low, close: float64
        - volume: float64

        Override to handle source-specific column names/formats.
        """
        if df.empty:
            return df

        # Standard column mapping
        column_map = {
            # Timestamp variations
            "time": "timestamp",
            "date": "timestamp",
            "datetime": "timestamp",
            "Date": "timestamp",
            "Time": "timestamp",
            "t": "timestamp",

            # OHLC variations
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",

            # Volume variations
            "Volume": "volume",
            "vol": "volume",
            "v": "volume",
        }

        df = df.rename(columns=column_map)

        # Ensure timestamp is UTC datetime
        if "timestamp" in df.columns:
            if df["timestamp"].dtype == "object" or df["timestamp"].dtype.name == "datetime64[ns]":
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            elif df["timestamp"].dtype.name.startswith("datetime64"):
                if df["timestamp"].dt.tz is None:
                    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
                else:
                    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

        # Ensure numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort by timestamp
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")

        # Select only standard columns
        standard_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        existing_cols = [c for c in standard_cols if c in df.columns]
        df = df[existing_cols]

        return df.reset_index(drop=True)


class RateLimiter:
    """
    Rate limiter for API requests.

    Implements token bucket algorithm with configurable:
    - requests_per_minute: Maximum requests per minute
    - burst_size: Maximum burst requests

    Usage:
        limiter = RateLimiter(requests_per_minute=60)
        await limiter.acquire()  # Blocks until request allowed
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute.
            burst_size: Max burst requests.
        """
        import time
        import threading

        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_refill = time.time()
        self.lock = threading.Lock()
        self.refill_rate = requests_per_minute / 60.0  # Tokens per second

    def acquire(self, timeout: float = 60.0) -> bool:
        """
        Acquire a token (blocking).

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if token acquired, False if timeout.
        """
        import time

        start = time.time()

        while True:
            with self.lock:
                self._refill()

                if self.tokens >= 1:
                    self.tokens -= 1
                    return True

            if time.time() - start > timeout:
                return False

            # Wait for tokens to refill
            time.sleep(1.0 / self.refill_rate)

    def _refill(self):
        """Refill tokens based on elapsed time."""
        import time

        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate

        if new_tokens > 0:
            self.tokens = min(self.burst_size, self.tokens + new_tokens)
            self.last_refill = now


class RetryPolicy:
    """
    Retry policy for failed requests.

    Implements exponential backoff with jitter.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """
        Initialize retry policy.

        Args:
            max_attempts: Maximum retry attempts.
            initial_delay: Initial delay in seconds.
            max_delay: Maximum delay in seconds.
            exponential_base: Exponential backoff base.
            jitter: Add random jitter to delays.
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """
        Get delay for a retry attempt.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in seconds.
        """
        import random

        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            delay = delay * (0.5 + random.random())

        return delay

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """
        Check if should retry after an exception.

        Args:
            attempt: Current attempt number.
            exception: The exception that occurred.

        Returns:
            True if should retry.
        """
        if attempt >= self.max_attempts:
            return False

        # Always retry on network errors
        import requests

        if isinstance(exception, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return True

        # Retry on rate limits (429)
        if isinstance(exception, requests.exceptions.HTTPError):
            if exception.response is not None and exception.response.status_code == 429:
                return True

        # Retry on server errors (5xx)
        if isinstance(exception, requests.exceptions.HTTPError):
            if exception.response is not None and 500 <= exception.response.status_code < 600:
                return True

        return False
