"""Reward functions for the trading environment."""

from enum import Enum
from typing import List, Optional
import numpy as np


class RewardType(str, Enum):
    """Available reward function types."""

    PNL = "pnl"              # Simple PnL reward
    SHARPE = "sharpe"        # Sharpe ratio based
    SORTINO = "sortino"      # Sortino ratio based
    CALMAR = "calmar"        # Calmar ratio based
    RISK_ADJUSTED = "risk_adjusted"  # Custom risk-adjusted


class RewardCalculator:
    """
    Calculate rewards for RL agent actions.

    Supports multiple reward functions:
    - PnL: Simple profit/loss
    - Sharpe: Risk-adjusted using Sharpe ratio
    - Sortino: Downside risk adjusted
    - Calmar: Drawdown adjusted
    - Risk-adjusted: Custom combination
    """

    def __init__(
        self,
        reward_type: RewardType = RewardType.SHARPE,
        scaling: float = 1.0,
        lookback_window: int = 20,
        risk_free_rate: float = 0.0,
        target_return: float = 0.0,
    ):
        """
        Initialize reward calculator.

        Args:
            reward_type: Type of reward function.
            scaling: Reward scaling factor.
            lookback_window: Window for calculating metrics.
            risk_free_rate: Annual risk-free rate (for Sharpe).
            target_return: Target return (for Sortino).
        """
        self.reward_type = reward_type
        self.scaling = scaling
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return

        # State for tracking
        self._peak_value: float = 0.0
        self._returns_history: List[float] = []

    def reset(self) -> None:
        """Reset calculator state."""
        self._peak_value = 0.0
        self._returns_history = []

    def calculate(
        self,
        portfolio_value: float,
        prev_portfolio_value: float,
        returns: Optional[List[float]] = None,
        position: float = 0.0,
    ) -> float:
        """
        Calculate reward for the current step.

        Args:
            portfolio_value: Current portfolio value.
            prev_portfolio_value: Previous portfolio value.
            returns: List of historical returns.
            position: Current position.

        Returns:
            Reward value.
        """
        # Calculate step return
        if prev_portfolio_value > 0:
            step_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            step_return = 0.0

        # Update peak for drawdown
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

        # Store return
        self._returns_history.append(step_return)
        if len(self._returns_history) > self.lookback_window:
            self._returns_history.pop(0)

        # Use provided returns or internal history
        returns_arr = np.array(returns if returns else self._returns_history)

        # Calculate reward based on type
        if self.reward_type == RewardType.PNL:
            reward = self._pnl_reward(step_return)

        elif self.reward_type == RewardType.SHARPE:
            reward = self._sharpe_reward(returns_arr, step_return)

        elif self.reward_type == RewardType.SORTINO:
            reward = self._sortino_reward(returns_arr, step_return)

        elif self.reward_type == RewardType.CALMAR:
            reward = self._calmar_reward(
                returns_arr, portfolio_value, step_return
            )

        elif self.reward_type == RewardType.RISK_ADJUSTED:
            reward = self._risk_adjusted_reward(
                returns_arr, portfolio_value, position, step_return
            )

        else:
            reward = step_return

        return reward * self.scaling

    def _pnl_reward(self, step_return: float) -> float:
        """Simple PnL reward."""
        return step_return * 100  # Scale for better gradient

    def _sharpe_reward(
        self,
        returns: np.ndarray,
        step_return: float,
    ) -> float:
        """
        Sharpe ratio based reward.

        Uses differential Sharpe for step-by-step training.
        """
        if len(returns) < 2:
            return step_return

        # Calculate rolling Sharpe
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return step_return

        # Annualization factor (assuming hourly bars)
        annualization = np.sqrt(252 * 6)

        sharpe = annualization * mean_return / std_return

        # Differential Sharpe: reward improvement in Sharpe
        n = len(returns)
        delta_mean = (step_return - mean_return) / n
        delta_var = ((step_return - mean_return) ** 2 - std_return ** 2) / n

        if std_return > 0:
            delta_sharpe = (delta_mean * std_return - 0.5 * mean_return * delta_var / std_return) / std_return
        else:
            delta_sharpe = step_return

        return delta_sharpe * annualization

    def _sortino_reward(
        self,
        returns: np.ndarray,
        step_return: float,
    ) -> float:
        """
        Sortino ratio based reward.

        Penalizes downside volatility more than upside.
        """
        if len(returns) < 2:
            return step_return

        # Calculate downside deviation
        excess_returns = returns - self.target_return
        downside_returns = np.minimum(excess_returns, 0)
        downside_std = np.sqrt(np.mean(downside_returns ** 2))

        if downside_std == 0:
            return step_return * 2  # Reward no downside risk

        mean_return = np.mean(returns)
        annualization = np.sqrt(252 * 6)

        # Penalize negative returns more
        if step_return < 0:
            return step_return * 2  # Double penalty for losses
        else:
            sortino = annualization * mean_return / downside_std
            return step_return + 0.01 * sortino

    def _calmar_reward(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        step_return: float,
    ) -> float:
        """
        Calmar ratio based reward.

        Penalizes drawdowns.
        """
        if self._peak_value == 0:
            return step_return

        # Calculate current drawdown
        drawdown = (self._peak_value - portfolio_value) / self._peak_value

        # Penalize based on drawdown
        drawdown_penalty = -drawdown * 0.5

        # Reward for positive returns with drawdown penalty
        return step_return + drawdown_penalty

    def _risk_adjusted_reward(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        position: float,
        step_return: float,
    ) -> float:
        """
        Custom risk-adjusted reward combining multiple factors.

        Components:
        1. Base return (scaled)
        2. Sharpe improvement bonus
        3. Drawdown penalty
        4. Turnover penalty (encourages holding)
        """
        reward = 0.0

        # 1. Base return component (50%)
        reward += step_return * 50

        # 2. Sharpe improvement (25%)
        if len(returns) >= 5:
            sharpe = self._calculate_sharpe(returns)
            reward += sharpe * 0.25

        # 3. Drawdown penalty (15%)
        if self._peak_value > 0:
            drawdown = (self._peak_value - portfolio_value) / self._peak_value
            # Exponential penalty for larger drawdowns
            reward -= (drawdown ** 2) * 100 * 0.15

        # 4. Activity penalty (10%)
        # Small penalty for changing positions to encourage conviction
        # This is handled elsewhere if position changes are tracked

        return reward

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret == 0:
            return 0.0

        return np.sqrt(252 * 6) * mean_ret / std_ret


class RewardShaper:
    """
    Reward shaping utilities for improving training.

    Provides potential-based reward shaping that doesn't
    change optimal policy.
    """

    def __init__(self, gamma: float = 0.99):
        """
        Initialize reward shaper.

        Args:
            gamma: Discount factor.
        """
        self.gamma = gamma
        self._prev_potential: Optional[float] = None

    def reset(self) -> None:
        """Reset shaper state."""
        self._prev_potential = None

    def shape(
        self,
        reward: float,
        state_potential: float,
    ) -> float:
        """
        Apply potential-based reward shaping.

        Args:
            reward: Original reward.
            state_potential: Potential of current state.

        Returns:
            Shaped reward.
        """
        if self._prev_potential is None:
            self._prev_potential = state_potential
            return reward

        # F = gamma * potential(s') - potential(s)
        shaping = self.gamma * state_potential - self._prev_potential
        self._prev_potential = state_potential

        return reward + shaping

    @staticmethod
    def portfolio_potential(
        portfolio_value: float,
        initial_capital: float,
        target_return: float = 0.20,
    ) -> float:
        """
        Calculate portfolio potential for shaping.

        Potential increases as we approach target return.
        """
        current_return = (portfolio_value - initial_capital) / initial_capital
        distance_to_target = target_return - current_return

        # Potential decreases with distance to target
        return -distance_to_target * 10
