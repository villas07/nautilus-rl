"""
Pipeline Configuration

Defines all configuration for the data pipeline including:
- Data source configurations
- Quality control parameters
- Storage settings
- Monitoring thresholds
"""

import os
from dataclasses import dataclass, field
from datetime import time as dt_time, timedelta
from enum import Enum
from typing import List, Dict, Optional, Set
from pathlib import Path


class AssetType(str, Enum):
    """Market asset types."""

    STOCK = "stock"
    ETF = "etf"
    FUTURE = "future"
    OPTION = "option"
    CRYPTO = "crypto"
    FOREX = "forex"
    INDEX = "index"
    BOND = "bond"


class SourceRole(str, Enum):
    """Role of a data source in the pipeline."""

    PRIMARY = "primary"      # Main source, highest priority
    SECONDARY = "secondary"  # Backup for gap filling
    TERTIARY = "tertiary"    # Last resort fallback
    REALTIME = "realtime"    # Live data only


class DataQuality(str, Enum):
    """Data quality tier."""

    EXCHANGE = "exchange"      # Direct from exchange
    PREMIUM = "premium"        # Premium vendor (Databento, Bloomberg)
    STANDARD = "standard"      # Standard vendor (Polygon, IEX)
    FREE = "free"              # Free APIs (Yahoo, CryptoCompare)


@dataclass
class MarketHours:
    """Trading session hours (UTC)."""

    open_time: dt_time
    close_time: dt_time
    trading_days: Set[int]  # 0=Monday, 6=Sunday
    timezone: str = "UTC"

    @classmethod
    def stock_us(cls) -> "MarketHours":
        """US Stock market hours (9:30 AM - 4:00 PM ET)."""
        return cls(
            open_time=dt_time(14, 30),  # 9:30 AM ET in UTC
            close_time=dt_time(21, 0),   # 4:00 PM ET in UTC
            trading_days={0, 1, 2, 3, 4},  # Mon-Fri
            timezone="America/New_York",
        )

    @classmethod
    def futures_cme(cls) -> "MarketHours":
        """CME Futures hours (Sun 6PM - Fri 5PM ET)."""
        return cls(
            open_time=dt_time(23, 0),   # Sun 6PM ET
            close_time=dt_time(22, 0),  # Fri 5PM ET
            trading_days={0, 1, 2, 3, 4, 6},  # Sun-Fri
            timezone="America/New_York",
        )

    @classmethod
    def crypto_24_7(cls) -> "MarketHours":
        """Crypto markets - 24/7."""
        return cls(
            open_time=dt_time(0, 0),
            close_time=dt_time(23, 59, 59),
            trading_days={0, 1, 2, 3, 4, 5, 6},
            timezone="UTC",
        )

    @classmethod
    def forex(cls) -> "MarketHours":
        """Forex markets - 24/5."""
        return cls(
            open_time=dt_time(22, 0),   # Sun 5PM ET
            close_time=dt_time(22, 0),  # Fri 5PM ET
            trading_days={0, 1, 2, 3, 4, 6},
            timezone="America/New_York",
        )


@dataclass
class SourceConfig:
    """Configuration for a data source."""

    name: str
    role: SourceRole
    asset_types: List[AssetType]
    confidence: float  # 0.0 to 1.0
    quality: DataQuality

    # Adapter configuration
    adapter_module: str
    adapter_class: str

    # Authentication
    api_key_env: Optional[str] = None
    api_secret_env: Optional[str] = None

    # Rate limits
    requests_per_minute: int = 60
    max_concurrent: int = 5

    # Capabilities
    supports_historical: bool = True
    supports_realtime: bool = False
    max_bars_per_request: int = 1000
    earliest_data: str = "2015-01-01"  # YYYY-MM-DD

    # Status
    enabled: bool = True

    def is_available(self) -> bool:
        """Check if source is available (enabled and has required credentials)."""
        if not self.enabled:
            return False
        if self.api_key_env and not os.getenv(self.api_key_env):
            return False
        return True

    def get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None

    def get_api_secret(self) -> Optional[str]:
        """Get API secret from environment."""
        if self.api_secret_env:
            return os.getenv(self.api_secret_env)
        return None


# Default source configurations
DEFAULT_SOURCES: Dict[str, SourceConfig] = {
    # PRIMARY: Databento for stocks/futures (highest quality)
    "databento": SourceConfig(
        name="Databento",
        role=SourceRole.PRIMARY,
        asset_types=[AssetType.STOCK, AssetType.ETF, AssetType.FUTURE, AssetType.INDEX],
        confidence=1.0,
        quality=DataQuality.PREMIUM,
        adapter_module="data.adapters.databento_adapter",
        adapter_class="DatabentoAdapter",
        api_key_env="DATABENTO_API_KEY",
        requests_per_minute=100,
        max_bars_per_request=10000,
        earliest_data="2015-01-01",
    ),

    # PRIMARY: Binance for crypto (direct exchange data)
    "binance": SourceConfig(
        name="Binance",
        role=SourceRole.PRIMARY,
        asset_types=[AssetType.CRYPTO],
        confidence=1.0,
        quality=DataQuality.EXCHANGE,
        adapter_module="data.adapters.binance_adapter",
        adapter_class="BinanceHistoricalAdapter",
        requests_per_minute=1200,
        max_bars_per_request=1000,
        earliest_data="2017-08-01",
    ),

    # SECONDARY: Polygon for US markets backup
    "polygon": SourceConfig(
        name="Polygon",
        role=SourceRole.SECONDARY,
        asset_types=[AssetType.STOCK, AssetType.ETF, AssetType.CRYPTO, AssetType.FOREX],
        confidence=0.95,
        quality=DataQuality.STANDARD,
        adapter_module="data.adapters.polygon_adapter",
        adapter_class="PolygonAdapter",
        api_key_env="POLYGON_API_KEY",
        requests_per_minute=5,  # Free tier
        max_bars_per_request=50000,
        earliest_data="2004-01-01",
    ),

    # TERTIARY: Yahoo Finance (free fallback)
    "yahoo": SourceConfig(
        name="Yahoo Finance",
        role=SourceRole.TERTIARY,
        asset_types=[AssetType.STOCK, AssetType.ETF, AssetType.INDEX, AssetType.CRYPTO, AssetType.FOREX],
        confidence=0.75,
        quality=DataQuality.FREE,
        adapter_module="data.adapters.yahoo_adapter",
        adapter_class="YahooFinanceAdapter",
        requests_per_minute=60,
        max_bars_per_request=10000,
        earliest_data="1970-01-01",
    ),

    # SECONDARY: CryptoCompare for crypto backup
    "cryptocompare": SourceConfig(
        name="CryptoCompare",
        role=SourceRole.SECONDARY,
        asset_types=[AssetType.CRYPTO],
        confidence=0.90,
        quality=DataQuality.STANDARD,
        adapter_module="data.adapters.cryptocompare_adapter",
        adapter_class="CryptoCompareAdapter",
        api_key_env="CRYPTOCOMPARE_API_KEY",
        requests_per_minute=80,
        max_bars_per_request=2000,
        earliest_data="2010-01-01",
    ),

    # TERTIARY: CoinGecko for crypto fallback
    "coingecko": SourceConfig(
        name="CoinGecko",
        role=SourceRole.TERTIARY,
        asset_types=[AssetType.CRYPTO],
        confidence=0.70,
        quality=DataQuality.FREE,
        adapter_module="data.adapters.coingecko_adapter",
        adapter_class="CoinGeckoAdapter",
        requests_per_minute=10,
        max_bars_per_request=365,
        earliest_data="2013-01-01",
    ),

    # REALTIME: Interactive Brokers for live data
    "ibkr": SourceConfig(
        name="Interactive Brokers",
        role=SourceRole.REALTIME,
        asset_types=[AssetType.STOCK, AssetType.ETF, AssetType.FUTURE, AssetType.FOREX, AssetType.OPTION],
        confidence=1.0,
        quality=DataQuality.EXCHANGE,
        adapter_module="data.adapters.ibkr_adapter",
        adapter_class="IBKRAdapter",
        api_key_env="IBKR_ACCOUNT",
        supports_historical=True,
        supports_realtime=True,
        requests_per_minute=50,
        max_bars_per_request=5000,
        earliest_data="2005-01-01",
    ),

    # Secondary: Alpha Vantage
    "alphavantage": SourceConfig(
        name="Alpha Vantage",
        role=SourceRole.TERTIARY,
        asset_types=[AssetType.STOCK, AssetType.FOREX, AssetType.CRYPTO],
        confidence=0.70,
        quality=DataQuality.FREE,
        adapter_module="data.adapters.alphavantage_adapter",
        adapter_class="AlphaVantageAdapter",
        api_key_env="ALPHAVANTAGE_API_KEY",
        requests_per_minute=5,  # Free: 5/min, 500/day
        max_bars_per_request=10000,
        earliest_data="2000-01-01",
    ),

    # MACRO: FRED for economic data
    "fred": SourceConfig(
        name="FRED",
        role=SourceRole.PRIMARY,
        asset_types=[],  # Special case: macro data
        confidence=1.0,
        quality=DataQuality.PREMIUM,
        adapter_module="data.adapters.fred_adapter",
        adapter_class="FREDAdapter",
        api_key_env="FRED_API_KEY",
        requests_per_minute=120,
        max_bars_per_request=100000,
        earliest_data="1947-01-01",
    ),
}


@dataclass
class QualityConfig:
    """Quality control configuration."""

    # Outlier detection
    outlier_std_threshold: float = 3.0
    max_price_change_pct: float = 20.0

    # Volume validation
    allow_zero_volume: bool = False
    min_volume_threshold: float = 0.0

    # Timestamp validation
    strict_market_hours: bool = False  # Reject out-of-hours data
    max_future_timestamp_seconds: int = 60  # Allow small clock drift

    # Data completeness
    max_gap_bars: int = 5  # Consecutive missing bars to flag as gap
    min_coverage_pct: float = 95.0  # Minimum % of expected bars


@dataclass
class ReconciliationConfig:
    """Reconciliation configuration."""

    # Discrepancy thresholds
    alert_threshold_pct: float = 0.1  # Alert if price differs > 0.1%
    reject_threshold_pct: float = 1.0  # Reject if price differs > 1.0%

    # Conflict resolution
    prefer_higher_confidence: bool = True
    prefer_newer_data: bool = False

    # Audit
    log_all_discrepancies: bool = True
    audit_log_path: str = "/app/logs/reconciliation"


@dataclass
class StorageConfig:
    """Storage configuration."""

    # TimescaleDB
    timescale_host: str = ""
    timescale_port: int = 5432
    timescale_db: str = ""
    timescale_user: str = ""
    timescale_password: str = ""

    # Parquet catalog
    parquet_path: str = ""

    # Retention
    raw_retention_days: int = 30
    clean_retention_days: int = 0  # 0 = indefinite

    # Table prefixes
    raw_table_prefix: str = "raw_ohlcv"
    clean_table_prefix: str = "clean_ohlcv"

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create from environment variables."""
        return cls(
            timescale_host=os.getenv("TIMESCALE_HOST", "localhost"),
            timescale_port=int(os.getenv("TIMESCALE_PORT", "5432")),
            timescale_db=os.getenv("TIMESCALE_DB", "deskgrade"),
            timescale_user=os.getenv("TIMESCALE_USER", "postgres"),
            timescale_password=os.getenv("TIMESCALE_PASSWORD", ""),
            parquet_path=os.getenv("CATALOG_PATH", "/app/data/catalog"),
        )

    @property
    def connection_string(self) -> str:
        """Get SQLAlchemy connection string."""
        return (
            f"postgresql://{self.timescale_user}:{self.timescale_password}"
            f"@{self.timescale_host}:{self.timescale_port}/{self.timescale_db}"
        )


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""

    # Telegram alerts
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Thresholds
    source_failure_alert_minutes: int = 5
    gap_alert_min_bars: int = 10
    quality_alert_threshold: float = 0.9

    # Health checks
    health_check_interval_seconds: int = 60

    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        """Create from environment variables."""
        return cls(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        )


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    # Sub-configs
    sources: Dict[str, SourceConfig] = field(default_factory=lambda: DEFAULT_SOURCES.copy())
    quality: QualityConfig = field(default_factory=QualityConfig)
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig.from_env)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig.from_env)

    # General settings
    default_timeframes: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    default_start_date: str = "2020-01-01"
    max_concurrent_downloads: int = 5
    retry_attempts: int = 3
    retry_delay_seconds: int = 5

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create configuration from environment variables."""
        return cls(
            storage=StorageConfig.from_env(),
            monitoring=MonitoringConfig.from_env(),
        )

    def get_sources_for_asset(self, asset_type: AssetType) -> List[SourceConfig]:
        """Get available sources for an asset type, ordered by priority."""
        sources = [
            s for s in self.sources.values()
            if asset_type in s.asset_types and s.is_available()
        ]

        # Sort: PRIMARY > SECONDARY > TERTIARY, then by confidence
        role_order = {
            SourceRole.PRIMARY: 0,
            SourceRole.SECONDARY: 1,
            SourceRole.TERTIARY: 2,
            SourceRole.REALTIME: 3,
        }

        return sorted(sources, key=lambda s: (role_order.get(s.role, 9), -s.confidence))

    def validate(self) -> List[str]:
        """Validate configuration. Returns list of issues."""
        issues = []

        # Check storage
        if not self.storage.timescale_host:
            issues.append("TimescaleDB host not configured")

        if not self.storage.parquet_path:
            issues.append("Parquet path not configured")

        # Check at least one source available
        available_sources = [s for s in self.sources.values() if s.is_available()]
        if not available_sources:
            issues.append("No data sources available")

        return issues
