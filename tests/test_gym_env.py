"""Tests for the Gymnasium environment."""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gym_env.nautilus_env import NautilusGymEnv, EnvConfig
from gym_env.observation import ObservationBuilder
from gym_env.rewards import RewardCalculator, RewardType
from gym_env.actions import ActionHandler, ActionType


class TestNautilusGymEnv:
    """Test cases for NautilusGymEnv."""

    def test_env_creation(self):
        """Test environment can be created."""
        config = EnvConfig(symbol="SPY", venue="IBKR")
        env = NautilusGymEnv(config=config)

        assert env is not None
        assert env.observation_space.shape == (45,)
        assert env.action_space.n == 3

        env.close()

    def test_env_reset(self):
        """Test environment reset."""
        env = NautilusGymEnv()
        obs, info = env.reset()

        assert obs.shape == (45,)
        assert isinstance(info, dict)
        assert "portfolio_value" in info

        env.close()

    def test_env_step(self):
        """Test environment step."""
        env = NautilusGymEnv()
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)  # Hold

        assert obs.shape == (45,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        env.close()

    def test_env_episode(self):
        """Test running a full episode."""
        config = EnvConfig(max_episode_steps=100)
        env = NautilusGymEnv(config=config)
        obs, _ = env.reset()

        total_reward = 0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        assert steps > 0
        assert steps <= config.max_episode_steps
        env.close()


class TestObservationBuilder:
    """Test cases for ObservationBuilder."""

    def test_observation_shape(self):
        """Test observation has correct shape."""
        import pandas as pd

        builder = ObservationBuilder(lookback_period=20, num_features=45)

        # Create dummy data
        n_bars = 30
        data = pd.DataFrame({
            "timestamp": pd.date_range(end="2024-01-01", periods=n_bars, freq="h"),
            "open": np.random.randn(n_bars).cumsum() + 100,
            "high": np.random.randn(n_bars).cumsum() + 101,
            "low": np.random.randn(n_bars).cumsum() + 99,
            "close": np.random.randn(n_bars).cumsum() + 100,
            "volume": np.random.randint(1000000, 10000000, n_bars),
        })

        obs = builder.build(data)

        assert obs.shape == (45,)
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))


class TestRewardCalculator:
    """Test cases for RewardCalculator."""

    def test_pnl_reward(self):
        """Test PnL reward calculation."""
        calculator = RewardCalculator(reward_type=RewardType.PNL)

        reward = calculator.calculate(
            portfolio_value=101000,
            prev_portfolio_value=100000,
        )

        assert reward > 0

    def test_sharpe_reward(self):
        """Test Sharpe reward calculation."""
        calculator = RewardCalculator(reward_type=RewardType.SHARPE)

        # Simulate some returns
        for _ in range(10):
            calculator.calculate(
                portfolio_value=100000 + np.random.randn() * 100,
                prev_portfolio_value=100000,
            )

        reward = calculator.calculate(
            portfolio_value=101000,
            prev_portfolio_value=100000,
        )

        assert isinstance(reward, float)


class TestActionHandler:
    """Test cases for ActionHandler."""

    def test_discrete_actions(self):
        """Test discrete action handling."""
        handler = ActionHandler(action_type=ActionType.DISCRETE)

        # Hold
        assert handler.get_target_position(0, 0.0) == 0.0

        # Buy
        assert handler.get_target_position(1, 0.0) > 0

        # Sell
        assert handler.get_target_position(2, 0.0) < 0

    def test_continuous_actions(self):
        """Test continuous action handling."""
        handler = ActionHandler(action_type=ActionType.CONTINUOUS)

        # Full buy
        assert handler.get_target_position(np.array([1.0])) == 1.0

        # Full sell
        assert handler.get_target_position(np.array([-1.0])) == -1.0

        # Neutral
        assert handler.get_target_position(np.array([0.0])) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
