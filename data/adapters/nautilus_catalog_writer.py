"""
NautilusTrader Catalog Writer

Writes data in NautilusTrader's native ParquetDataCatalog format.

This ensures data is properly formatted for:
- BacktestNode consumption
- ParquetDataCatalog.bars() queries
- Instrument metadata

IMPORTANT: NautilusTrader expects specific schemas and formats.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import pandas as pd
import numpy as np

from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import (
    Equity,
    CurrencyPair,
    CryptoPerpetual,
)
from nautilus_trader.model.enums import (
    AssetClass,
    InstrumentClass,
    BarAggregation,
    PriceType,
    AggregationSource,
)
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.model.currencies import USD, EUR, BTC, USDT
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.persistence.wranglers import BarDataWrangler

import structlog

logger = structlog.get_logger()


# Currency mappings
CURRENCY_MAP = {
    "USD": USD,
    "EUR": EUR,
    "BTC": BTC,
    "USDT": USDT,
}


class NautilusCatalogWriter:
    """
    Writes market data to NautilusTrader's ParquetDataCatalog format.

    This is the CORRECT way to store data for NautilusTrader backtest.

    Usage:
        writer = NautilusCatalogWriter("/app/data/catalog")

        # Create instrument
        instrument = writer.create_equity("SPY", "NASDAQ")
        writer.write_instrument(instrument)

        # Write bars
        df = pd.DataFrame(...)  # Your OHLCV data
        writer.write_bars(df, "SPY.NASDAQ", "1-HOUR-LAST")
    """

    # Timeframe mappings
    BAR_SPECS = {
        "1m": BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
        "5m": BarSpecification(5, BarAggregation.MINUTE, PriceType.LAST),
        "15m": BarSpecification(15, BarAggregation.MINUTE, PriceType.LAST),
        "30m": BarSpecification(30, BarAggregation.MINUTE, PriceType.LAST),
        "1h": BarSpecification(1, BarAggregation.HOUR, PriceType.LAST),
        "4h": BarSpecification(4, BarAggregation.HOUR, PriceType.LAST),
        "1d": BarSpecification(1, BarAggregation.DAY, PriceType.LAST),
    }

    def __init__(self, catalog_path: str):
        """
        Initialize catalog writer.

        Args:
            catalog_path: Path to ParquetDataCatalog directory.
        """
        self.catalog_path = Path(catalog_path)
        self.catalog_path.mkdir(parents=True, exist_ok=True)

        self._catalog: Optional[ParquetDataCatalog] = None

    @property
    def catalog(self) -> ParquetDataCatalog:
        """Get or create catalog."""
        if self._catalog is None:
            self._catalog = ParquetDataCatalog(str(self.catalog_path))
        return self._catalog

    def create_equity(
        self,
        symbol: str,
        venue: str,
        currency: str = "USD",
        price_precision: int = 2,
        size_precision: int = 0,
        lot_size: float = 1.0,
        isin: Optional[str] = None,
    ) -> Equity:
        """
        Create an Equity instrument.

        Args:
            symbol: Stock symbol (e.g., "SPY").
            venue: Exchange venue (e.g., "NASDAQ", "NYSE").
            currency: Quote currency.
            price_precision: Price decimal places.
            size_precision: Quantity decimal places.
            lot_size: Minimum lot size.
            isin: ISIN code (optional).

        Returns:
            Equity instrument.
        """
        instrument_id = InstrumentId(
            symbol=Symbol(symbol),
            venue=Venue(venue),
        )

        currency_obj = CURRENCY_MAP.get(currency, USD)

        return Equity(
            instrument_id=instrument_id,
            raw_symbol=Symbol(symbol),
            currency=currency_obj,
            price_precision=price_precision,
            price_increment=Price.from_str(f"0.{'0' * (price_precision - 1)}1"),
            lot_size=Quantity.from_str(str(lot_size)),
            max_quantity=Quantity.from_str("1000000"),
            min_quantity=Quantity.from_str(str(lot_size)),
            ts_event=0,
            ts_init=0,
            isin=isin,
        )

    def create_crypto_perpetual(
        self,
        symbol: str,
        venue: str = "BINANCE",
        base_currency: str = "BTC",
        quote_currency: str = "USDT",
        price_precision: int = 2,
        size_precision: int = 3,
    ) -> CryptoPerpetual:
        """
        Create a CryptoPerpetual instrument.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT").
            venue: Exchange venue.
            base_currency: Base currency.
            quote_currency: Quote currency.
            price_precision: Price decimal places.
            size_precision: Quantity decimal places.

        Returns:
            CryptoPerpetual instrument.
        """
        instrument_id = InstrumentId(
            symbol=Symbol(symbol),
            venue=Venue(venue),
        )

        base = CURRENCY_MAP.get(base_currency, BTC)
        quote = CURRENCY_MAP.get(quote_currency, USDT)

        return CryptoPerpetual(
            instrument_id=instrument_id,
            raw_symbol=Symbol(symbol),
            base_currency=base,
            quote_currency=quote,
            settlement_currency=quote,
            is_inverse=False,
            price_precision=price_precision,
            size_precision=size_precision,
            price_increment=Price.from_str(f"0.{'0' * (price_precision - 1)}1"),
            size_increment=Quantity.from_str(f"0.{'0' * (size_precision - 1)}1"),
            max_quantity=Quantity.from_str("10000"),
            min_quantity=Quantity.from_str("0.001"),
            max_notional=None,
            min_notional=Money.from_str("10 USDT"),
            max_price=Price.from_str("1000000"),
            min_price=Price.from_str("0.01"),
            margin_init=0.01,  # 1%
            margin_maint=0.005,  # 0.5%
            maker_fee=0.0002,
            taker_fee=0.0004,
            ts_event=0,
            ts_init=0,
        )

    def write_instrument(self, instrument: Union[Equity, CryptoPerpetual]) -> None:
        """
        Write instrument to catalog.

        Args:
            instrument: Instrument to write.
        """
        self.catalog.write_data([instrument])
        logger.info(f"Wrote instrument: {instrument.id}")

    def write_bars(
        self,
        df: pd.DataFrame,
        instrument_id: str,
        timeframe: str = "1h",
    ) -> int:
        """
        Write OHLCV bars to catalog.

        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume.
            instrument_id: Full instrument ID (e.g., "SPY.NASDAQ").
            timeframe: Timeframe string (1m, 5m, 15m, 30m, 1h, 4h, 1d).

        Returns:
            Number of bars written.
        """
        if df.empty:
            return 0

        # Parse instrument ID
        inst_id = InstrumentId.from_str(instrument_id)

        # Get bar specification
        bar_spec = self.BAR_SPECS.get(timeframe)
        if bar_spec is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        # Create bar type
        bar_type = BarType(
            instrument_id=inst_id,
            bar_spec=bar_spec,
            aggregation_source=AggregationSource.EXTERNAL,
        )

        # Convert DataFrame to Bar objects
        bars = self._dataframe_to_bars(df, bar_type)

        if not bars:
            return 0

        # Write to catalog
        self.catalog.write_data(bars)

        logger.info(
            f"Wrote {len(bars)} bars",
            instrument_id=instrument_id,
            timeframe=timeframe,
            start=bars[0].ts_event if bars else None,
            end=bars[-1].ts_event if bars else None,
        )

        return len(bars)

    def _dataframe_to_bars(
        self,
        df: pd.DataFrame,
        bar_type: BarType,
    ) -> List[Bar]:
        """
        Convert DataFrame to NautilusTrader Bar objects.

        Args:
            df: DataFrame with OHLCV data.
            bar_type: Bar type specification.

        Returns:
            List of Bar objects.
        """
        bars = []

        # Ensure sorted by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Determine precision from data
        sample_price = df["close"].iloc[0]
        price_precision = self._infer_precision(sample_price)

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
                open=Price.from_str(f"{row['open']:.{price_precision}f}"),
                high=Price.from_str(f"{row['high']:.{price_precision}f}"),
                low=Price.from_str(f"{row['low']:.{price_precision}f}"),
                close=Price.from_str(f"{row['close']:.{price_precision}f}"),
                volume=Quantity.from_str(f"{row['volume']:.0f}"),
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
            bars.append(bar)

        return bars

    def _infer_precision(self, value: float) -> int:
        """Infer decimal precision from a value."""
        str_val = f"{value:.10f}".rstrip("0")
        if "." in str_val:
            return len(str_val.split(".")[1])
        return 0

    def write_from_pipeline(
        self,
        symbol: str,
        venue: str,
        timeframe: str,
        df: pd.DataFrame,
        asset_type: str = "equity",
    ) -> int:
        """
        Write data from the pipeline to catalog.

        This is a convenience method that:
        1. Creates the instrument if it doesn't exist
        2. Writes the bar data

        Args:
            symbol: Symbol (e.g., "SPY", "BTCUSDT").
            venue: Venue (e.g., "NASDAQ", "BINANCE").
            timeframe: Timeframe string.
            df: DataFrame with OHLCV data.
            asset_type: "equity", "crypto", "forex".

        Returns:
            Number of bars written.
        """
        instrument_id = f"{symbol}.{venue}"

        # Check if instrument exists
        existing = self.catalog.instruments(instrument_ids=[instrument_id])

        if not existing:
            # Create instrument
            if asset_type == "crypto":
                # Parse crypto pair
                if symbol.endswith("USDT"):
                    base = symbol[:-4]
                    quote = "USDT"
                elif symbol.endswith("USD"):
                    base = symbol[:-3]
                    quote = "USD"
                else:
                    base = symbol[:3]
                    quote = symbol[3:]

                instrument = self.create_crypto_perpetual(
                    symbol=symbol,
                    venue=venue,
                    base_currency=base,
                    quote_currency=quote,
                )
            else:
                instrument = self.create_equity(
                    symbol=symbol,
                    venue=venue,
                )

            self.write_instrument(instrument)

        # Write bars
        return self.write_bars(df, instrument_id, timeframe)

    def get_catalog_stats(self) -> Dict[str, Any]:
        """Get statistics about the catalog."""
        instruments = self.catalog.instruments()

        stats = {
            "catalog_path": str(self.catalog_path),
            "instruments_count": len(instruments),
            "instruments": [],
        }

        for inst in instruments:
            # Get bar counts per timeframe
            bar_counts = {}
            for tf in self.BAR_SPECS.keys():
                try:
                    bars = self.catalog.bars(
                        instrument_ids=[str(inst.id)],
                        bar_type=f"{inst.id}-{self.BAR_SPECS[tf]}",
                    )
                    bar_counts[tf] = len(bars) if bars else 0
                except Exception:
                    bar_counts[tf] = 0

            stats["instruments"].append({
                "id": str(inst.id),
                "type": type(inst).__name__,
                "bars": bar_counts,
            })

        return stats

    def close(self) -> None:
        """Close catalog connection."""
        self._catalog = None


def sync_to_nautilus_catalog(
    symbols: List[str],
    timeframes: List[str],
    catalog_path: str = "/app/data/catalog",
    start_date: str = "2020-01-01",
) -> Dict[str, int]:
    """
    Sync data from pipeline to NautilusTrader catalog.

    Args:
        symbols: List of symbols.
        timeframes: List of timeframes.
        catalog_path: Catalog path.
        start_date: Start date.

    Returns:
        Dict with sync statistics.
    """
    from data.pipeline import DataPipeline

    writer = NautilusCatalogWriter(catalog_path)
    pipeline = DataPipeline()

    stats = {"total_bars": 0, "instruments": 0, "errors": []}

    for symbol in symbols:
        # Determine asset type
        if symbol.endswith("USDT") or symbol.endswith("USD"):
            asset_type = "crypto"
            venue = "BINANCE"
        else:
            asset_type = "equity"
            venue = "NASDAQ"

        for timeframe in timeframes:
            try:
                # Fetch data through pipeline
                result = pipeline.process(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    save_to_db=False,
                    save_to_parquet=False,
                )

                if result.success:
                    # Get clean data from storage
                    df = pipeline.storage.read_clean(symbol, timeframe, start_date)

                    if not df.empty:
                        # Write to Nautilus catalog
                        bars_written = writer.write_from_pipeline(
                            symbol=symbol,
                            venue=venue,
                            timeframe=timeframe,
                            df=df,
                            asset_type=asset_type,
                        )
                        stats["total_bars"] += bars_written

            except Exception as e:
                stats["errors"].append(f"{symbol}_{timeframe}: {str(e)}")
                logger.error(f"Failed to sync {symbol}: {e}")

        stats["instruments"] += 1

    pipeline.close()
    writer.close()

    return stats


if __name__ == "__main__":
    # Example usage
    writer = NautilusCatalogWriter("/app/data/catalog")

    # Create equity instrument
    spy = writer.create_equity("SPY", "NASDAQ")
    writer.write_instrument(spy)

    # Create crypto instrument
    btc = writer.create_crypto_perpetual("BTCUSDT", "BINANCE")
    writer.write_instrument(btc)

    # Print catalog stats
    stats = writer.get_catalog_stats()
    print(f"Catalog: {stats['instruments_count']} instruments")

    writer.close()
