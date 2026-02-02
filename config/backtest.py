"""Backtest configuration for NautilusTrader."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from nautilus_trader.config import (
    BacktestRunConfig,
    BacktestDataConfig,
    BacktestVenueConfig,
    BacktestEngineConfig,
    LoggingConfig,
    RiskEngineConfig,
)
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.enums import AccountType, OmsType


@dataclass
class BacktestConfig:
    """Configuration holder for backtesting."""

    # Data settings
    catalog_path: str = "/app/data/catalog"
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"

    # Venue settings
    venues: List[str] = field(default_factory=lambda: ["IBKR", "BINANCE"])
    starting_capital: float = 100000.0

    # Execution settings
    latency_ms: int = 50
    frozen_account: bool = False

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "BacktestConfig":
        """Create config from environment variables."""
        return cls(
            catalog_path=os.getenv("CATALOG_PATH", "/app/data/catalog"),
            start_date=os.getenv("BACKTEST_START", "2015-01-01"),
            end_date=os.getenv("BACKTEST_END", "2024-12-31"),
            starting_capital=float(os.getenv("STARTING_CAPITAL", "100000")),
        )


def get_venue_config(
    venue: str,
    starting_capital: float = 100000.0,
    latency_ms: int = 50,
) -> BacktestVenueConfig:
    """
    Create venue configuration for backtesting.

    Args:
        venue: Venue name (IBKR, BINANCE).
        starting_capital: Initial account balance.
        latency_ms: Simulated latency in milliseconds.

    Returns:
        BacktestVenueConfig for the venue.
    """
    if venue.upper() == "BINANCE":
        return BacktestVenueConfig(
            name="BINANCE",
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency="USDT",
            starting_balances=["100000 USDT"],
            default_leverage=10.0,
            leverages={"BTCUSDT.BINANCE": 20.0, "ETHUSDT.BINANCE": 15.0},
            fill_model=None,  # Use default
            latency_model=None,  # Use default
        )
    else:  # IBKR or default
        return BacktestVenueConfig(
            name="IBKR",
            oms_type=OmsType.HEDGING,
            account_type=AccountType.MARGIN,
            base_currency="USD",
            starting_balances=["100000 USD"],
            default_leverage=4.0,
            fill_model=None,
            latency_model=None,
        )


def get_data_config(
    catalog_path: str,
    instrument_ids: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> BacktestDataConfig:
    """
    Create data configuration for backtesting.

    Args:
        catalog_path: Path to Parquet data catalog.
        instrument_ids: List of instrument IDs to load.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).

    Returns:
        BacktestDataConfig.
    """
    return BacktestDataConfig(
        catalog_path=catalog_path,
        data_cls_strs=["Bar", "QuoteTick", "TradeTick"],
        instrument_ids=instrument_ids,
        start_time=start_date,
        end_time=end_date,
    )


def get_backtest_config(
    config: Optional[BacktestConfig] = None,
    instrument_ids: Optional[List[str]] = None,
    strategies: Optional[List] = None,
) -> BacktestRunConfig:
    """
    Create BacktestRunConfig for running backtests.

    Args:
        config: Backtest configuration.
        instrument_ids: List of instrument IDs.
        strategies: List of strategy configurations.

    Returns:
        BacktestRunConfig ready to run.
    """
    if config is None:
        config = BacktestConfig.from_env()

    # Create venue configs
    venue_configs = [
        get_venue_config(venue, config.starting_capital, config.latency_ms)
        for venue in config.venues
    ]

    # Create data config
    data_config = get_data_config(
        catalog_path=config.catalog_path,
        instrument_ids=instrument_ids,
        start_date=config.start_date,
        end_date=config.end_date,
    )

    return BacktestRunConfig(
        engine=BacktestEngineConfig(
            strategies=strategies or [],
            logging=LoggingConfig(
                log_level=config.log_level,
            ),
            risk_engine=RiskEngineConfig(
                bypass=False,
            ),
        ),
        data=[data_config],
        venues=venue_configs,
    )


def create_walkforward_configs(
    base_config: Optional[BacktestConfig] = None,
    train_years: int = 7,  # 2015-2022
    val_years: int = 1,    # 2023
    test_years: int = 1,   # 2024
) -> dict:
    """
    Create walk-forward backtest configurations.

    Returns dict with 'train', 'validation', 'test' configs.
    """
    if base_config is None:
        base_config = BacktestConfig()

    # Define periods
    configs = {}

    # Training: 2015-2022
    configs["train"] = BacktestConfig(
        catalog_path=base_config.catalog_path,
        start_date="2015-01-01",
        end_date="2022-12-31",
        venues=base_config.venues,
        starting_capital=base_config.starting_capital,
    )

    # Validation: 2023
    configs["validation"] = BacktestConfig(
        catalog_path=base_config.catalog_path,
        start_date="2023-01-01",
        end_date="2023-12-31",
        venues=base_config.venues,
        starting_capital=base_config.starting_capital,
    )

    # Test: 2024
    configs["test"] = BacktestConfig(
        catalog_path=base_config.catalog_path,
        start_date="2024-01-01",
        end_date="2024-12-31",
        venues=base_config.venues,
        starting_capital=base_config.starting_capital,
    )

    return configs
