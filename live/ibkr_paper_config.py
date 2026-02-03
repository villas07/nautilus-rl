"""
IBKR Paper Trading Configuration

Configuration for connecting NautilusTrader to Interactive Brokers paper account.

Reference: Phase 6 Task 6.1
Owner: @quant_developer
"""

from decimal import Decimal
from pathlib import Path
from typing import List, Optional

from nautilus_trader.config import (
    TradingNodeConfig,
    LiveExecEngineConfig,
    LiveDataEngineConfig,
    LoggingConfig,
    StrategyConfig,
)
from nautilus_trader.adapters.interactive_brokers.config import (
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersExecClientConfig,
    InteractiveBrokersInstrumentProviderConfig,
)


def create_ibkr_paper_config(
    account_id: str = "DUO275624",  # Paper account
    host: str = "127.0.0.1",
    port: int = 7497,  # Paper trading port (7496 for live)
    client_id: int = 1,
    models_dir: str = "/app/models",
    instruments: Optional[List[str]] = None,
    min_confidence: float = 0.6,
) -> TradingNodeConfig:
    """
    Create TradingNodeConfig for IBKR paper trading.

    Args:
        account_id: IBKR account ID
        host: IB Gateway host
        port: IB Gateway port (7497=paper, 7496=live)
        client_id: TWS client ID
        models_dir: Directory with validated models
        instruments: List of instruments to trade
        min_confidence: Minimum voting confidence

    Returns:
        TradingNodeConfig ready for paper trading
    """
    # Default instruments
    if instruments is None:
        instruments = [
            "SPY.NASDAQ",
            "QQQ.NASDAQ",
            "AAPL.NASDAQ",
            "MSFT.NASDAQ",
        ]

    # Data client config
    data_client = InteractiveBrokersDataClientConfig(
        ibg_host=host,
        ibg_port=port,
        ibg_client_id=client_id,
        # Request bars on connection
        # bar_types=["1-HOUR-LAST"] for each instrument
    )

    # Execution client config
    exec_client = InteractiveBrokersExecClientConfig(
        ibg_host=host,
        ibg_port=port,
        ibg_client_id=client_id,
        account_id=account_id,
    )

    # Instrument provider config
    instrument_provider = InteractiveBrokersInstrumentProviderConfig(
        # Load instruments on start
        load_all=False,
        load_ids=frozenset(instruments),
    )

    # Complete trading node config
    config = TradingNodeConfig(
        trader_id="NAUTILUS-PAPER-001",
        log_level="INFO",

        # Data engine
        data_engine=LiveDataEngineConfig(
            qsize=100_000,
        ),

        # Execution engine
        exec_engine=LiveExecEngineConfig(
            qsize=100_000,
        ),

        # Logging
        logging=LoggingConfig(
            log_level="INFO",
            log_directory=str(Path.home() / "nautilus_logs"),
        ),

        # Data clients
        data_clients={
            "IB": data_client,
        },

        # Execution clients
        exec_clients={
            "IB": exec_client,
        },

        # Timeout settings (milliseconds)
        timeout_connection=30_000,
        timeout_reconciliation=60_000,
        timeout_portfolio=30_000,
        timeout_disconnection=10_000,
        timeout_post_stop=5_000,
    )

    return config


def create_rl_strategy_configs(
    instruments: List[str],
    models_dir: str = "/app/models",
    min_confidence: float = 0.6,
) -> List[dict]:
    """
    Create strategy configs for each instrument.

    Args:
        instruments: List of instrument IDs
        models_dir: Directory with validated models
        min_confidence: Minimum voting confidence

    Returns:
        List of strategy config dicts
    """
    from strategies.rl_strategy import RLTradingStrategyConfig

    configs = []

    for instrument_id in instruments:
        config = {
            "strategy_class": "strategies.rl_strategy.RLTradingStrategy",
            "config": RLTradingStrategyConfig(
                instrument_id=instrument_id,
                models_dir=models_dir,
                use_voting=True,
                min_confidence=min_confidence,
                bar_type="1-HOUR-LAST",
                position_sizing="fixed",
                fixed_size=1.0,
                max_position_size=10.0,
            ),
        }
        configs.append(config)

    return configs


# Default paper trading configuration
PAPER_CONFIG = {
    "account_id": "DUO275624",
    "host": "127.0.0.1",
    "port": 7497,
    "client_id": 1,

    # Instruments to trade
    "instruments": [
        "SPY.NASDAQ",
        "QQQ.NASDAQ",
    ],

    # Trading parameters
    "min_confidence": 0.6,
    "models_dir": "/app/models",

    # Risk limits (from IMMUTABLE_RULES.md)
    "risk_limits": {
        "max_position_per_symbol_usd": 5000,
        "max_total_exposure_usd": 50000,
        "max_daily_loss_usd": 2000,
        "circuit_breaker_consecutive_losses": 7,
    },
}


def get_paper_trading_config() -> TradingNodeConfig:
    """Get default paper trading configuration."""
    return create_ibkr_paper_config(
        account_id=PAPER_CONFIG["account_id"],
        host=PAPER_CONFIG["host"],
        port=PAPER_CONFIG["port"],
        client_id=PAPER_CONFIG["client_id"],
        models_dir=PAPER_CONFIG["models_dir"],
        instruments=PAPER_CONFIG["instruments"],
        min_confidence=PAPER_CONFIG["min_confidence"],
    )


if __name__ == "__main__":
    # Test config creation
    config = get_paper_trading_config()
    print(f"Trading Node Config:")
    print(f"  Trader ID: {config.trader_id}")
    print(f"  Log Level: {config.log_level}")

    strategies = create_rl_strategy_configs(
        instruments=PAPER_CONFIG["instruments"],
        models_dir=PAPER_CONFIG["models_dir"],
    )
    print(f"\nStrategy Configs: {len(strategies)}")
    for s in strategies:
        print(f"  - {s['config'].instrument_id}")
