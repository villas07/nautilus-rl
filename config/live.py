"""Live trading configuration for NautilusTrader."""

import os
from dataclasses import dataclass
from typing import Optional

from nautilus_trader.config import (
    TradingNodeConfig,
    LiveDataClientConfig,
    LiveExecClientConfig,
)
from nautilus_trader.live.node import TradingNode


@dataclass
class LiveConfig:
    """Configuration holder for live trading."""

    # IBKR settings
    ibkr_host: str = "host.docker.internal"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1
    ibkr_account: str = ""

    # Binance settings
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True

    # Trading settings
    trading_mode: str = "paper"
    max_position_size: float = 10000.0
    max_daily_loss: float = 500.0
    risk_per_trade: float = 0.02

    @classmethod
    def from_env(cls) -> "LiveConfig":
        """Create config from environment variables."""
        return cls(
            ibkr_host=os.getenv("IBKR_HOST", "host.docker.internal"),
            ibkr_port=int(os.getenv("IBKR_PORT", "7497")),
            ibkr_client_id=int(os.getenv("IBKR_CLIENT_ID", "1")),
            ibkr_account=os.getenv("IBKR_ACCOUNT", ""),
            binance_api_key=os.getenv("BINANCE_API_KEY", ""),
            binance_api_secret=os.getenv("BINANCE_API_SECRET", ""),
            binance_testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true",
            trading_mode=os.getenv("TRADING_MODE", "paper"),
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", "10000")),
            max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "500")),
            risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.02")),
        )


def get_ibkr_data_config(config: LiveConfig) -> LiveDataClientConfig:
    """Create IBKR data client configuration."""
    from nautilus_trader.adapters.interactive_brokers.config import (
        InteractiveBrokersDataClientConfig,
    )

    return InteractiveBrokersDataClientConfig(
        ibg_host=config.ibkr_host,
        ibg_port=config.ibkr_port,
        ibg_client_id=config.ibkr_client_id,
    )


def get_ibkr_exec_config(config: LiveConfig) -> LiveExecClientConfig:
    """Create IBKR execution client configuration."""
    from nautilus_trader.adapters.interactive_brokers.config import (
        InteractiveBrokersExecClientConfig,
    )

    return InteractiveBrokersExecClientConfig(
        ibg_host=config.ibkr_host,
        ibg_port=config.ibkr_port,
        ibg_client_id=config.ibkr_client_id + 100,  # Separate client ID for exec
        account_id=config.ibkr_account,
    )


def get_binance_data_config(config: LiveConfig) -> LiveDataClientConfig:
    """Create Binance data client configuration."""
    from nautilus_trader.adapters.binance.config import BinanceDataClientConfig

    return BinanceDataClientConfig(
        api_key=config.binance_api_key,
        api_secret=config.binance_api_secret,
        account_type="USDT_FUTURE",
        testnet=config.binance_testnet,
    )


def get_binance_exec_config(config: LiveConfig) -> LiveExecClientConfig:
    """Create Binance execution client configuration."""
    from nautilus_trader.adapters.binance.config import BinanceExecClientConfig

    return BinanceExecClientConfig(
        api_key=config.binance_api_key,
        api_secret=config.binance_api_secret,
        account_type="USDT_FUTURE",
        testnet=config.binance_testnet,
    )


def get_live_config(
    config: Optional[LiveConfig] = None,
    enable_ibkr: bool = True,
    enable_binance: bool = True,
) -> TradingNodeConfig:
    """
    Create TradingNodeConfig for live trading.

    Args:
        config: Live configuration (defaults to env vars).
        enable_ibkr: Enable IBKR adapter.
        enable_binance: Enable Binance adapter.

    Returns:
        TradingNodeConfig for live trading.
    """
    if config is None:
        config = LiveConfig.from_env()

    data_clients = {}
    exec_clients = {}

    # Add IBKR clients
    if enable_ibkr and config.ibkr_account:
        data_clients["IB"] = get_ibkr_data_config(config)
        exec_clients["IB"] = get_ibkr_exec_config(config)

    # Add Binance clients
    if enable_binance and config.binance_api_key:
        data_clients["BINANCE"] = get_binance_data_config(config)
        exec_clients["BINANCE"] = get_binance_exec_config(config)

    return TradingNodeConfig(
        trader_id="NAUTILUS-AGENTS-001",
        data_clients=data_clients,
        exec_clients=exec_clients,
        timeout_connection=30.0,
        timeout_reconciliation=10.0,
        timeout_portfolio=10.0,
        timeout_disconnection=10.0,
        timeout_post_stop=5.0,
    )


def create_trading_node(
    config: Optional[LiveConfig] = None,
    enable_ibkr: bool = True,
    enable_binance: bool = True,
) -> TradingNode:
    """
    Create and configure a TradingNode for live trading.

    Args:
        config: Live configuration.
        enable_ibkr: Enable IBKR adapter.
        enable_binance: Enable Binance adapter.

    Returns:
        Configured TradingNode.
    """
    node_config = get_live_config(config, enable_ibkr, enable_binance)
    return TradingNode(config=node_config)
