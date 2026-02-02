"""Convert data between formats for NautilusTrader."""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.enums import AggregationSource, BarAggregation, PriceType
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog

import structlog

logger = structlog.get_logger()


class BarConverter:
    """
    Converts bar data between formats.

    Supports:
    - DataFrame → Nautilus Bar objects
    - DataFrame → Parquet (for DataCatalog)
    - TimescaleDB → Parquet
    """

    # Timeframe mappings to Nautilus BarAggregation
    TIMEFRAME_MAP = {
        "1m": (1, BarAggregation.MINUTE),
        "1min": (1, BarAggregation.MINUTE),
        "5m": (5, BarAggregation.MINUTE),
        "5min": (5, BarAggregation.MINUTE),
        "15m": (15, BarAggregation.MINUTE),
        "15min": (15, BarAggregation.MINUTE),
        "30m": (30, BarAggregation.MINUTE),
        "30min": (30, BarAggregation.MINUTE),
        "1h": (1, BarAggregation.HOUR),
        "1H": (1, BarAggregation.HOUR),
        "1hour": (1, BarAggregation.HOUR),
        "4h": (4, BarAggregation.HOUR),
        "4H": (4, BarAggregation.HOUR),
        "1d": (1, BarAggregation.DAY),
        "1D": (1, BarAggregation.DAY),
        "daily": (1, BarAggregation.DAY),
    }

    def __init__(
        self,
        price_precision: int = 2,
        size_precision: int = 2,
    ):
        """
        Initialize converter.

        Args:
            price_precision: Decimal places for prices.
            size_precision: Decimal places for quantities.
        """
        self.price_precision = price_precision
        self.size_precision = size_precision

    def get_bar_specification(self, timeframe: str) -> BarSpecification:
        """Get BarSpecification for a timeframe."""
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        step, aggregation = self.TIMEFRAME_MAP[timeframe]

        return BarSpecification(
            step=step,
            aggregation=aggregation,
            price_type=PriceType.LAST,
        )

    def get_bar_type(
        self,
        instrument_id: str,
        timeframe: str,
    ) -> BarType:
        """Get BarType for instrument and timeframe."""
        spec = self.get_bar_specification(timeframe)
        return BarType(
            instrument_id=InstrumentId.from_str(instrument_id),
            bar_spec=spec,
            aggregation_source=AggregationSource.EXTERNAL,
        )

    def dataframe_to_bars(
        self,
        df: pd.DataFrame,
        instrument_id: str,
        timeframe: str = "1h",
    ) -> List[Bar]:
        """
        Convert DataFrame to list of Nautilus Bar objects.

        Args:
            df: DataFrame with timestamp, open, high, low, close, volume.
            instrument_id: Full instrument ID (e.g., "SPY.IBKR").
            timeframe: Bar timeframe.

        Returns:
            List of Bar objects.
        """
        if df.empty:
            return []

        bar_type = self.get_bar_type(instrument_id, timeframe)
        bars = []

        for _, row in df.iterrows():
            # Convert timestamp to nanoseconds
            ts = row["timestamp"]
            if isinstance(ts, pd.Timestamp):
                ts_ns = int(ts.value)
            elif isinstance(ts, datetime):
                ts_ns = int(ts.timestamp() * 1e9)
            else:
                ts_ns = int(pd.Timestamp(ts).value)

            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{row['open']:.{self.price_precision}f}"),
                high=Price.from_str(f"{row['high']:.{self.price_precision}f}"),
                low=Price.from_str(f"{row['low']:.{self.price_precision}f}"),
                close=Price.from_str(f"{row['close']:.{self.price_precision}f}"),
                volume=Quantity.from_str(f"{row['volume']:.{self.size_precision}f}"),
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
            bars.append(bar)

        logger.info(
            "Converted bars",
            instrument_id=instrument_id,
            count=len(bars),
        )

        return bars

    def dataframe_to_parquet(
        self,
        df: pd.DataFrame,
        instrument_id: str,
        timeframe: str,
        output_dir: str,
    ) -> Path:
        """
        Convert DataFrame to Parquet file for DataCatalog.

        Args:
            df: DataFrame with OHLCV data.
            instrument_id: Full instrument ID.
            timeframe: Bar timeframe.
            output_dir: Output directory for Parquet file.

        Returns:
            Path to created Parquet file.
        """
        if df.empty:
            raise ValueError("Cannot convert empty DataFrame")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare data in Nautilus format
        bar_type = self.get_bar_type(instrument_id, timeframe)

        # Create Nautilus-compatible DataFrame
        nautilus_df = pd.DataFrame({
            "bar_type": str(bar_type),
            "instrument_id": instrument_id,
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["volume"].astype(float),
            "ts_event": df["timestamp"].apply(
                lambda x: int(pd.Timestamp(x).value)
            ),
            "ts_init": df["timestamp"].apply(
                lambda x: int(pd.Timestamp(x).value)
            ),
        })

        # Create filename
        symbol = instrument_id.replace(".", "_")
        filename = f"bars_{symbol}_{timeframe}.parquet"
        file_path = output_path / filename

        # Write Parquet
        table = pa.Table.from_pandas(nautilus_df)
        pq.write_table(table, file_path, compression="snappy")

        logger.info(
            "Wrote Parquet file",
            path=str(file_path),
            rows=len(nautilus_df),
        )

        return file_path

    def write_to_catalog(
        self,
        df: pd.DataFrame,
        instrument_id: str,
        timeframe: str,
        catalog_path: str,
    ) -> None:
        """
        Write bars directly to Nautilus DataCatalog.

        Args:
            df: DataFrame with OHLCV data.
            instrument_id: Full instrument ID.
            timeframe: Bar timeframe.
            catalog_path: Path to DataCatalog.
        """
        # Convert to Bar objects
        bars = self.dataframe_to_bars(df, instrument_id, timeframe)

        if not bars:
            logger.warning("No bars to write to catalog")
            return

        # Initialize catalog
        catalog = ParquetDataCatalog(catalog_path)

        # Write bars
        catalog.write_data(bars)

        logger.info(
            "Wrote to catalog",
            catalog_path=catalog_path,
            instrument_id=instrument_id,
            bar_count=len(bars),
        )


def convert_symbol_to_nautilus(
    symbol: str,
    venue: str,
) -> str:
    """Convert a symbol to Nautilus InstrumentId format."""
    # Remove any existing venue suffix
    symbol = symbol.split(".")[0]

    # Handle crypto pairs
    if symbol.endswith("USDT") or symbol.endswith("USD"):
        # Already in correct format
        pass
    elif "/" in symbol:
        # Forex: EUR/USD -> EURUSD
        symbol = symbol.replace("/", "")

    return f"{symbol}.{venue}"


def infer_venue(symbol: str) -> str:
    """Infer venue from symbol."""
    symbol_upper = symbol.upper()

    # Crypto symbols
    if symbol_upper.endswith("USDT") or symbol_upper.endswith("BUSD"):
        return "BINANCE"

    # Forex symbols
    if "/" in symbol or symbol_upper in ["EURUSD", "GBPUSD", "USDJPY"]:
        return "IBKR"

    # Default to IBKR for equities
    return "IBKR"
