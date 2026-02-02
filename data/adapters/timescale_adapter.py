"""TimescaleDB adapter for reading historical data."""

import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import asyncpg

import structlog

logger = structlog.get_logger()


@dataclass
class TimescaleConfig:
    """TimescaleDB connection configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "deskgrade"
    user: str = "postgres"
    password: str = ""

    @classmethod
    def from_env(cls) -> "TimescaleConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("TIMESCALE_HOST", "localhost"),
            port=int(os.getenv("TIMESCALE_PORT", "5432")),
            database=os.getenv("TIMESCALE_DB", "deskgrade"),
            user=os.getenv("TIMESCALE_USER", "postgres"),
            password=os.getenv("TIMESCALE_PASSWORD", ""),
        )

    @property
    def connection_string(self) -> str:
        """Get SQLAlchemy connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_connection_string(self) -> str:
        """Get asyncpg connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class TimescaleAdapter:
    """
    Adapter for reading data from TimescaleDB.

    Integrates with existing DeskGrade data pipeline tables:
    - ohlcv_1m: 1-minute bars
    - ohlcv_1h: 1-hour bars
    - ohlcv_1d: daily bars
    """

    # Table mappings for different timeframes
    TIMEFRAME_TABLES = {
        "1m": "ohlcv_1m",
        "1min": "ohlcv_1m",
        "1T": "ohlcv_1m",
        "5m": "ohlcv_5m",
        "5min": "ohlcv_5m",
        "5T": "ohlcv_5m",
        "15m": "ohlcv_15m",
        "15min": "ohlcv_15m",
        "15T": "ohlcv_15m",
        "1h": "ohlcv_1h",
        "1H": "ohlcv_1h",
        "1hour": "ohlcv_1h",
        "4h": "ohlcv_4h",
        "4H": "ohlcv_4h",
        "1d": "ohlcv_1d",
        "1D": "ohlcv_1d",
        "daily": "ohlcv_1d",
    }

    def __init__(self, config: Optional[TimescaleConfig] = None):
        """Initialize the adapter."""
        self.config = config or TimescaleConfig.from_env()
        self._engine: Optional[Engine] = None
        self._async_pool: Optional[asyncpg.Pool] = None

    @property
    def engine(self) -> Engine:
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(
                self.config.connection_string,
                pool_size=5,
                max_overflow=10,
                pool_recycle=3600,
            )
        return self._engine

    async def get_async_pool(self) -> asyncpg.Pool:
        """Get or create async connection pool."""
        if self._async_pool is None:
            self._async_pool = await asyncpg.create_pool(
                self.config.async_connection_string,
                min_size=2,
                max_size=10,
            )
        return self._async_pool

    def get_table_name(self, timeframe: str) -> str:
        """Get table name for timeframe."""
        table = self.TIMEFRAME_TABLES.get(timeframe)
        if table is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        return table

    def fetch_bars(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars from TimescaleDB.

        Args:
            symbol: Instrument symbol (e.g., "SPY", "BTCUSDT").
            timeframe: Bar timeframe (1m, 5m, 15m, 1h, 4h, 1d).
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            limit: Maximum number of bars.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        table = self.get_table_name(timeframe)

        # Build query
        query = f"""
            SELECT
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM {table}
            WHERE symbol = :symbol
        """

        params: Dict[str, Any] = {"symbol": symbol}

        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = end_date

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        # Execute query
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)

        if df.empty:
            logger.warning(
                "No data found",
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )
            return pd.DataFrame()

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Convert numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(
            "Fetched bars",
            symbol=symbol,
            timeframe=timeframe,
            count=len(df),
            start=df["timestamp"].min().isoformat() if not df.empty else None,
            end=df["timestamp"].max().isoformat() if not df.empty else None,
        )

        return df

    async def fetch_bars_async(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Async version of fetch_bars."""
        table = self.get_table_name(timeframe)
        pool = await self.get_async_pool()

        # Build query
        query = f"""
            SELECT
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM {table}
            WHERE symbol = $1
        """

        args = [symbol]
        arg_idx = 2

        if start_date:
            query += f" AND timestamp >= ${arg_idx}"
            args.append(start_date)
            arg_idx += 1

        if end_date:
            query += f" AND timestamp <= ${arg_idx}"
            args.append(end_date)
            arg_idx += 1

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *args)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        return df

    def get_available_symbols(self, timeframe: str = "1h") -> List[str]:
        """Get list of available symbols in the database."""
        table = self.get_table_name(timeframe)

        query = f"SELECT DISTINCT symbol FROM {table} ORDER BY symbol"

        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            symbols = [row[0] for row in result]

        return symbols

    def get_date_range(
        self,
        symbol: str,
        timeframe: str = "1h",
    ) -> tuple:
        """Get available date range for a symbol."""
        table = self.get_table_name(timeframe)

        query = f"""
            SELECT
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date,
                COUNT(*) as bar_count
            FROM {table}
            WHERE symbol = :symbol
        """

        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"symbol": symbol}).fetchone()

        if result and result[0]:
            return (result[0], result[1], result[2])
        return (None, None, 0)

    def close(self):
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None

    async def close_async(self):
        """Close async connections."""
        if self._async_pool:
            await self._async_pool.close()
            self._async_pool = None


# Convenience function
def create_adapter(config: Optional[TimescaleConfig] = None) -> TimescaleAdapter:
    """Create a TimescaleAdapter instance."""
    return TimescaleAdapter(config)
