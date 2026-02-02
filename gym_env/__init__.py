"""
Gymnasium Environment for RL Training with NautilusTrader.

This module provides a proper integration with NautilusTrader's:
- ParquetDataCatalog for data loading
- BacktestEngine for realistic execution simulation
- Order matching with slippage
- Position and portfolio tracking

Usage:
    from gym_env import NautilusGymEnv, NautilusEnvConfig

    config = NautilusEnvConfig(
        instrument_id="BTCUSDT.BINANCE",
        catalog_path="/app/data/catalog",
    )
    env = NautilusGymEnv(config)
"""

from gym_env.nautilus_env import (
    NautilusBacktestEnv as NautilusGymEnv,
    NautilusEnvConfig,
    create_nautilus_env,
)
from gym_env.observation import ObservationBuilder
from gym_env.rewards import RewardCalculator, RewardType
from gym_env.actions import ActionHandler

# Alias for compatibility
EnvConfig = NautilusEnvConfig

__all__ = [
    "NautilusGymEnv",
    "NautilusEnvConfig",
    "EnvConfig",
    "create_nautilus_env",
    "ObservationBuilder",
    "RewardCalculator",
    "RewardType",
    "ActionHandler",
]
