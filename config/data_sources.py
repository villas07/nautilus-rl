"""
Data Sources Configuration

Centralizes configuration for all data providers:
- Databento (professional market data)
- Tardis (crypto tick data)
- Binance (free crypto data)
- IBKR (live data via NautilusTrader)
- Polygon (via existing DeskGrade integration)

Each provider has different data types, coverage, and costs.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class DataSource(str, Enum):
    """Available data sources."""

    DATABENTO = "databento"
    TARDIS = "tardis"
    BINANCE = "binance"
    IBKR = "ibkr"
    POLYGON = "polygon"
    TIMESCALE = "timescale"  # Local TimescaleDB cache


class DataType(str, Enum):
    """Types of market data."""

    TRADES = "trades"
    QUOTES = "quotes"
    OHLCV = "ohlcv"
    ORDER_BOOK = "order_book"
    TICK = "tick"


@dataclass
class DatabentoConfig:
    """Databento configuration.

    Databento provides high-quality historical and real-time data.
    Free tier: Limited historical data access
    Paid: Full historical archive

    Supported venues: CME, NASDAQ, NYSE, etc.
    """

    api_key: str = ""
    dataset: str = "GLBX.MDP3"  # CME Globex
    enabled: bool = False

    # Data types available
    schemas: List[str] = field(default_factory=lambda: [
        "trades",
        "ohlcv-1m",
        "ohlcv-1h",
        "ohlcv-1d",
        "mbp-1",  # Market by price (L2)
    ])

    # Instruments to fetch
    symbols: List[str] = field(default_factory=lambda: [
        "ES.FUT",   # E-mini S&P 500
        "NQ.FUT",   # E-mini NASDAQ
        "CL.FUT",   # Crude Oil
        "GC.FUT",   # Gold
    ])

    @classmethod
    def from_env(cls) -> "DatabentoConfig":
        """Create config from environment variables."""
        api_key = os.getenv("DATABENTO_API_KEY", "")
        return cls(
            api_key=api_key,
            enabled=bool(api_key),
        )


@dataclass
class TardisConfig:
    """Tardis.dev configuration.

    Tardis provides historical crypto exchange data.
    Free tier: 10 API calls/day, last 3 days of data
    Paid: Full historical archive from 2017+

    Supported exchanges: Binance, FTX, Deribit, BitMEX, etc.
    """

    api_key: str = ""
    enabled: bool = False

    # Exchanges to fetch from
    exchanges: List[str] = field(default_factory=lambda: [
        "binance",
        "binance-futures",
        "deribit",
        "bitmex",
    ])

    # Data types
    data_types: List[str] = field(default_factory=lambda: [
        "trades",
        "book_snapshot_25",
        "incremental_book_L2",
    ])

    # Symbols per exchange
    symbols: Dict[str, List[str]] = field(default_factory=lambda: {
        "binance": ["BTCUSDT", "ETHUSDT"],
        "binance-futures": ["BTCUSDT", "ETHUSDT"],
        "deribit": ["BTC-PERPETUAL", "ETH-PERPETUAL"],
    })

    @classmethod
    def from_env(cls) -> "TardisConfig":
        """Create config from environment variables."""
        api_key = os.getenv("TARDIS_API_KEY", "")
        return cls(
            api_key=api_key,
            enabled=bool(api_key),
        )


@dataclass
class BinanceDataConfig:
    """Binance data configuration.

    Binance provides FREE historical OHLCV data via their API.
    No API key required for historical klines!

    Limitations:
    - Max 1000 bars per request
    - Rate limits apply
    - Data from 2017 onwards
    """

    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    enabled: bool = True  # Always enabled (free data)

    # Base URLs
    spot_base_url: str = "https://api.binance.com"
    futures_base_url: str = "https://fapi.binance.com"
    testnet_url: str = "https://testnet.binance.vision"

    # Symbols to fetch
    spot_symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT",
    ])

    futures_symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    ])

    # Timeframes available
    intervals: List[str] = field(default_factory=lambda: [
        "1m", "5m", "15m", "1h", "4h", "1d",
    ])

    @classmethod
    def from_env(cls) -> "BinanceDataConfig":
        """Create config from environment variables."""
        return cls(
            api_key=os.getenv("BINANCE_API_KEY", ""),
            api_secret=os.getenv("BINANCE_API_SECRET", ""),
            testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true",
        )


@dataclass
class PolygonConfig:
    """Polygon.io configuration.

    Already configured in DeskGrade - this is for reference.
    User has paid subscription.
    """

    api_key: str = ""
    enabled: bool = False

    @classmethod
    def from_env(cls) -> "PolygonConfig":
        """Create config from environment variables."""
        api_key = os.getenv("POLYGON_API_KEY", "")
        return cls(
            api_key=api_key,
            enabled=bool(api_key),
        )


@dataclass
class DataSourcesConfig:
    """Combined data sources configuration."""

    databento: DatabentoConfig = field(default_factory=DatabentoConfig.from_env)
    tardis: TardisConfig = field(default_factory=TardisConfig.from_env)
    binance: BinanceDataConfig = field(default_factory=BinanceDataConfig.from_env)
    polygon: PolygonConfig = field(default_factory=PolygonConfig.from_env)

    # TimescaleDB for centralized storage
    timescale_host: str = ""
    timescale_port: int = 5432
    timescale_db: str = "deskgrade"
    timescale_user: str = "postgres"
    timescale_password: str = ""

    # Catalog path for NautilusTrader
    catalog_path: str = "/app/data/catalog"

    @classmethod
    def from_env(cls) -> "DataSourcesConfig":
        """Create full config from environment."""
        return cls(
            databento=DatabentoConfig.from_env(),
            tardis=TardisConfig.from_env(),
            binance=BinanceDataConfig.from_env(),
            polygon=PolygonConfig.from_env(),
            timescale_host=os.getenv("TIMESCALE_HOST", "localhost"),
            timescale_port=int(os.getenv("TIMESCALE_PORT", "5432")),
            timescale_db=os.getenv("TIMESCALE_DB", "deskgrade"),
            timescale_user=os.getenv("TIMESCALE_USER", "postgres"),
            timescale_password=os.getenv("TIMESCALE_PASSWORD", ""),
            catalog_path=os.getenv("CATALOG_PATH", "/app/data/catalog"),
        )

    def get_enabled_sources(self) -> List[DataSource]:
        """Get list of enabled data sources."""
        enabled = []

        if self.databento.enabled:
            enabled.append(DataSource.DATABENTO)
        if self.tardis.enabled:
            enabled.append(DataSource.TARDIS)
        if self.binance.enabled:
            enabled.append(DataSource.BINANCE)
        if self.polygon.enabled:
            enabled.append(DataSource.POLYGON)

        return enabled


# Symbol mappings between sources
SYMBOL_MAPPINGS = {
    # Crypto: Binance symbol -> Other sources
    "BTCUSDT": {
        "binance": "BTCUSDT",
        "tardis": "BTCUSDT",
        "polygon": "X:BTCUSD",
    },
    "ETHUSDT": {
        "binance": "ETHUSDT",
        "tardis": "ETHUSDT",
        "polygon": "X:ETHUSD",
    },
    # Futures: NautilusTrader symbol -> Databento
    "ES.CME": {
        "databento": "ES.FUT",
        "ibkr": "ES",
    },
    "NQ.CME": {
        "databento": "NQ.FUT",
        "ibkr": "NQ",
    },
}


def get_symbol_for_source(
    nautilus_symbol: str,
    source: DataSource,
) -> str:
    """Convert NautilusTrader symbol to source-specific format."""
    base_symbol = nautilus_symbol.split(".")[0]

    if base_symbol in SYMBOL_MAPPINGS:
        mappings = SYMBOL_MAPPINGS[base_symbol]
        if source.value in mappings:
            return mappings[source.value]

    # Default: return as-is
    return base_symbol
