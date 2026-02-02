"""Instrument definitions for NautilusTrader."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import (
    CurrencyPair,
    Equity,
    FuturesContract,
)
from nautilus_trader.model.enums import AssetClass, InstrumentClass
from nautilus_trader.model.currencies import USD, EUR, GBP, JPY, USDT, BTC, ETH
from nautilus_trader.model.objects import Price, Quantity


class Market(str, Enum):
    """Supported markets."""
    IBKR = "IBKR"
    BINANCE = "BINANCE"


@dataclass
class InstrumentConfig:
    """Configuration for an instrument."""

    symbol: str
    venue: str
    asset_class: str
    price_precision: int = 2
    size_precision: int = 2
    min_quantity: float = 1.0
    max_quantity: float = 1000000.0
    lot_size: float = 1.0
    tick_size: float = 0.01
    margin_init: float = 0.1
    margin_maint: float = 0.05
    maker_fee: float = 0.0001
    taker_fee: float = 0.0002

    @property
    def instrument_id(self) -> str:
        """Get full instrument ID."""
        return f"{self.symbol}.{self.venue}"


# ============================================================================
# IBKR Instruments
# ============================================================================

IBKR_EQUITIES: List[InstrumentConfig] = [
    InstrumentConfig(
        symbol="SPY",
        venue="IBKR",
        asset_class="EQUITY",
        price_precision=2,
        size_precision=0,
        min_quantity=1,
        tick_size=0.01,
        margin_init=0.25,
        margin_maint=0.25,
    ),
    InstrumentConfig(
        symbol="QQQ",
        venue="IBKR",
        asset_class="EQUITY",
        price_precision=2,
        size_precision=0,
        min_quantity=1,
        tick_size=0.01,
        margin_init=0.25,
        margin_maint=0.25,
    ),
    InstrumentConfig(
        symbol="IWM",
        venue="IBKR",
        asset_class="EQUITY",
        price_precision=2,
        size_precision=0,
        min_quantity=1,
        tick_size=0.01,
        margin_init=0.25,
        margin_maint=0.25,
    ),
    InstrumentConfig(
        symbol="AAPL",
        venue="IBKR",
        asset_class="EQUITY",
        price_precision=2,
        size_precision=0,
        min_quantity=1,
        tick_size=0.01,
    ),
    InstrumentConfig(
        symbol="MSFT",
        venue="IBKR",
        asset_class="EQUITY",
        price_precision=2,
        size_precision=0,
        min_quantity=1,
        tick_size=0.01,
    ),
    InstrumentConfig(
        symbol="GOOGL",
        venue="IBKR",
        asset_class="EQUITY",
        price_precision=2,
        size_precision=0,
        min_quantity=1,
        tick_size=0.01,
    ),
    InstrumentConfig(
        symbol="AMZN",
        venue="IBKR",
        asset_class="EQUITY",
        price_precision=2,
        size_precision=0,
        min_quantity=1,
        tick_size=0.01,
    ),
    InstrumentConfig(
        symbol="NVDA",
        venue="IBKR",
        asset_class="EQUITY",
        price_precision=2,
        size_precision=0,
        min_quantity=1,
        tick_size=0.01,
    ),
    InstrumentConfig(
        symbol="META",
        venue="IBKR",
        asset_class="EQUITY",
        price_precision=2,
        size_precision=0,
        min_quantity=1,
        tick_size=0.01,
    ),
    InstrumentConfig(
        symbol="TSLA",
        venue="IBKR",
        asset_class="EQUITY",
        price_precision=2,
        size_precision=0,
        min_quantity=1,
        tick_size=0.01,
    ),
]

IBKR_FOREX: List[InstrumentConfig] = [
    InstrumentConfig(
        symbol="EUR/USD",
        venue="IBKR",
        asset_class="FX",
        price_precision=5,
        size_precision=0,
        min_quantity=1000,
        lot_size=1000,
        tick_size=0.00001,
        margin_init=0.02,
        margin_maint=0.02,
    ),
    InstrumentConfig(
        symbol="GBP/USD",
        venue="IBKR",
        asset_class="FX",
        price_precision=5,
        size_precision=0,
        min_quantity=1000,
        lot_size=1000,
        tick_size=0.00001,
        margin_init=0.02,
        margin_maint=0.02,
    ),
    InstrumentConfig(
        symbol="USD/JPY",
        venue="IBKR",
        asset_class="FX",
        price_precision=3,
        size_precision=0,
        min_quantity=1000,
        lot_size=1000,
        tick_size=0.001,
        margin_init=0.02,
        margin_maint=0.02,
    ),
]

# ============================================================================
# Binance Instruments
# ============================================================================

BINANCE_FUTURES: List[InstrumentConfig] = [
    InstrumentConfig(
        symbol="BTCUSDT",
        venue="BINANCE",
        asset_class="CRYPTO",
        price_precision=2,
        size_precision=3,
        min_quantity=0.001,
        lot_size=0.001,
        tick_size=0.01,
        margin_init=0.05,
        margin_maint=0.025,
        maker_fee=0.0002,
        taker_fee=0.0004,
    ),
    InstrumentConfig(
        symbol="ETHUSDT",
        venue="BINANCE",
        asset_class="CRYPTO",
        price_precision=2,
        size_precision=3,
        min_quantity=0.001,
        lot_size=0.001,
        tick_size=0.01,
        margin_init=0.05,
        margin_maint=0.025,
        maker_fee=0.0002,
        taker_fee=0.0004,
    ),
    InstrumentConfig(
        symbol="SOLUSDT",
        venue="BINANCE",
        asset_class="CRYPTO",
        price_precision=3,
        size_precision=1,
        min_quantity=0.1,
        lot_size=0.1,
        tick_size=0.001,
        margin_init=0.05,
        margin_maint=0.025,
        maker_fee=0.0002,
        taker_fee=0.0004,
    ),
    InstrumentConfig(
        symbol="BNBUSDT",
        venue="BINANCE",
        asset_class="CRYPTO",
        price_precision=2,
        size_precision=2,
        min_quantity=0.01,
        lot_size=0.01,
        tick_size=0.01,
        margin_init=0.05,
        margin_maint=0.025,
        maker_fee=0.0002,
        taker_fee=0.0004,
    ),
    InstrumentConfig(
        symbol="XRPUSDT",
        venue="BINANCE",
        asset_class="CRYPTO",
        price_precision=4,
        size_precision=1,
        min_quantity=0.1,
        lot_size=0.1,
        tick_size=0.0001,
        margin_init=0.05,
        margin_maint=0.025,
        maker_fee=0.0002,
        taker_fee=0.0004,
    ),
]


def get_instruments(
    market: Optional[Market] = None,
    asset_class: Optional[str] = None,
) -> List[InstrumentConfig]:
    """
    Get list of instrument configurations.

    Args:
        market: Filter by market (IBKR, BINANCE).
        asset_class: Filter by asset class (EQUITY, FX, CRYPTO).

    Returns:
        List of matching InstrumentConfig.
    """
    all_instruments: List[InstrumentConfig] = []

    if market is None or market == Market.IBKR:
        all_instruments.extend(IBKR_EQUITIES)
        all_instruments.extend(IBKR_FOREX)

    if market is None or market == Market.BINANCE:
        all_instruments.extend(BINANCE_FUTURES)

    if asset_class:
        all_instruments = [
            i for i in all_instruments
            if i.asset_class.upper() == asset_class.upper()
        ]

    return all_instruments


def get_instrument_ids(
    market: Optional[Market] = None,
    asset_class: Optional[str] = None,
) -> List[str]:
    """Get list of instrument ID strings."""
    return [i.instrument_id for i in get_instruments(market, asset_class)]


def get_instrument_by_id(instrument_id: str) -> Optional[InstrumentConfig]:
    """Get instrument config by ID."""
    for instrument in get_instruments():
        if instrument.instrument_id == instrument_id:
            return instrument
    return None


# All instrument IDs for convenience
ALL_INSTRUMENT_IDS = get_instrument_ids()
IBKR_INSTRUMENT_IDS = get_instrument_ids(Market.IBKR)
BINANCE_INSTRUMENT_IDS = get_instrument_ids(Market.BINANCE)
