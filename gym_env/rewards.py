"""
Reward functions for the trading environment.

Includes Triple Barrier reward (R-005) from ML Institutional:
- Rewards based on TP/SL/Timeout barriers
- Dynamic barriers based on volatility
- Asymmetric risk/reward incentives
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np


class RewardType(str, Enum):
    """Available reward function types."""

    PNL = "pnl"              # Simple PnL reward
    SHARPE = "sharpe"        # Sharpe ratio based
    SORTINO = "sortino"      # Sortino ratio based
    CALMAR = "calmar"        # Calmar ratio based
    RISK_ADJUSTED = "risk_adjusted"  # Custom risk-adjusted
    TRIPLE_BARRIER = "triple_barrier"  # R-005: Institutional triple barrier


@dataclass
class TripleBarrierConfig:
    """Configuration for Triple Barrier reward."""
    pt_mult: float = 2.0      # Take profit multiplier (x volatility)
    sl_mult: float = 1.0      # Stop loss multiplier (x volatility)
    vol_lookback: int = 20    # Lookback for volatility calculation
    max_holding_bars: int = 10  # Vertical barrier (max holding period)
    tp_reward: float = 1.0    # Reward for hitting TP
    sl_penalty: float = -1.0  # Penalty for hitting SL
    timeout_reward: float = 0.0  # Reward for timeout (0 = return-based)
    scale_by_return: bool = True  # Scale reward by actual return magnitude


class RewardCalculator:
    """
    Calculate rewards for RL agent actions.

    Supports multiple reward functions:
    - PnL: Simple profit/loss
    - Sharpe: Risk-adjusted using Sharpe ratio
    - Sortino: Downside risk adjusted
    - Calmar: Drawdown adjusted
    - Risk-adjusted: Custom combination
    - Triple Barrier: Institutional TP/SL/Timeout based (R-005)
    """

    def __init__(
        self,
        reward_type: RewardType = RewardType.SHARPE,
        scaling: float = 1.0,
        lookback_window: int = 20,
        risk_free_rate: float = 0.0,
        target_return: float = 0.0,
        triple_barrier_config: Optional[TripleBarrierConfig] = None,
    ):
        """
        Initialize reward calculator.

        Args:
            reward_type: Type of reward function.
            scaling: Reward scaling factor.
            lookback_window: Window for calculating metrics.
            risk_free_rate: Annual risk-free rate (for Sharpe).
            target_return: Target return (for Sortino).
            triple_barrier_config: Config for Triple Barrier reward (R-005).
        """
        self.reward_type = reward_type
        self.scaling = scaling
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return

        # Triple Barrier config (R-005)
        self.tb_config = triple_barrier_config or TripleBarrierConfig()

        # State for tracking
        self._peak_value: float = 0.0
        self._returns_history: List[float] = []

        # Triple Barrier state
        self._trade_entry_price: float = 0.0
        self._trade_entry_bar: int = 0
        self._trade_direction: int = 0  # 1=long, -1=short, 0=no position
        self._trade_tp_level: float = 0.0
        self._trade_sl_level: float = 0.0
        self._current_bar: int = 0
        self._prices_history: List[float] = []

    def reset(self) -> None:
        """Reset calculator state."""
        self._peak_value = 0.0
        self._returns_history = []

        # Reset Triple Barrier state
        self._trade_entry_price = 0.0
        self._trade_entry_bar = 0
        self._trade_direction = 0
        self._trade_tp_level = 0.0
        self._trade_sl_level = 0.0
        self._current_bar = 0
        self._prices_history = []

    def calculate(
        self,
        portfolio_value: float,
        prev_portfolio_value: float,
        returns: Optional[List[float]] = None,
        position: float = 0.0,
        current_price: float = 0.0,
        entry_price: float = 0.0,
    ) -> float:
        """
        Calculate reward for the current step.

        Args:
            portfolio_value: Current portfolio value.
            prev_portfolio_value: Previous portfolio value.
            returns: List of historical returns.
            position: Current position (positive=long, negative=short).
            current_price: Current bar close price (for Triple Barrier).
            entry_price: Trade entry price (for Triple Barrier).

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

        # Store return and price for history
        self._returns_history.append(step_return)
        if len(self._returns_history) > self.lookback_window:
            self._returns_history.pop(0)

        if current_price > 0:
            self._prices_history.append(current_price)
            if len(self._prices_history) > self.tb_config.vol_lookback * 2:
                self._prices_history.pop(0)

        # Increment bar counter
        self._current_bar += 1

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

        elif self.reward_type == RewardType.TRIPLE_BARRIER:
            reward = self._triple_barrier_reward(
                position, current_price, entry_price, step_return
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

    def _triple_barrier_reward(
        self,
        position: float,
        current_price: float,
        entry_price: float,
        step_return: float,
    ) -> float:
        """
        Triple Barrier reward function (R-005).

        Rewards based on which barrier is hit first:
        - Take Profit (TP): Large positive reward
        - Stop Loss (SL): Negative penalty
        - Timeout: Small reward based on return

        This teaches the agent to:
        1. Hold profitable trades until TP
        2. Cut losses at SL
        3. Not hold too long (timeout penalty)
        """
        # No position = no trade reward
        if position == 0 or current_price == 0 or entry_price == 0:
            # Small penalty for not trading (encourages activity)
            return -0.001

        # Detect new trade
        current_direction = 1 if position > 0 else -1

        if self._trade_direction == 0:
            # New trade opened
            self._trade_entry_price = entry_price
            self._trade_entry_bar = self._current_bar
            self._trade_direction = current_direction

            # Calculate volatility for barriers
            vol = self._calculate_volatility()

            # Set barriers (as price levels)
            if current_direction > 0:  # Long
                self._trade_tp_level = entry_price * (1 + self.tb_config.pt_mult * vol)
                self._trade_sl_level = entry_price * (1 - self.tb_config.sl_mult * vol)
            else:  # Short
                self._trade_tp_level = entry_price * (1 - self.tb_config.pt_mult * vol)
                self._trade_sl_level = entry_price * (1 + self.tb_config.sl_mult * vol)

            # Small reward for entering a trade
            return 0.01

        # Check if direction changed (trade closed)
        if current_direction != self._trade_direction:
            self._trade_direction = 0
            self._trade_entry_price = 0
            return step_return * 100  # Return PnL-based reward on exit

        # Calculate current return from entry
        if self._trade_direction > 0:  # Long
            trade_return = (current_price - self._trade_entry_price) / self._trade_entry_price
        else:  # Short
            trade_return = (self._trade_entry_price - current_price) / self._trade_entry_price

        # Check barriers
        holding_bars = self._current_bar - self._trade_entry_bar

        # Check Take Profit
        if self._trade_direction > 0 and current_price >= self._trade_tp_level:
            reward = self.tb_config.tp_reward
            if self.tb_config.scale_by_return:
                reward *= (1 + abs(trade_return) * 10)
            return reward

        if self._trade_direction < 0 and current_price <= self._trade_tp_level:
            reward = self.tb_config.tp_reward
            if self.tb_config.scale_by_return:
                reward *= (1 + abs(trade_return) * 10)
            return reward

        # Check Stop Loss
        if self._trade_direction > 0 and current_price <= self._trade_sl_level:
            reward = self.tb_config.sl_penalty
            if self.tb_config.scale_by_return:
                reward *= (1 + abs(trade_return) * 5)
            return reward

        if self._trade_direction < 0 and current_price >= self._trade_sl_level:
            reward = self.tb_config.sl_penalty
            if self.tb_config.scale_by_return:
                reward *= (1 + abs(trade_return) * 5)
            return reward

        # Check Timeout (vertical barrier)
        if holding_bars >= self.tb_config.max_holding_bars:
            if self.tb_config.timeout_reward != 0:
                return self.tb_config.timeout_reward
            else:
                # Return-based reward for timeout
                return trade_return * 50

        # Still in trade, no barrier hit
        # Small shaping reward based on progress toward TP
        if trade_return > 0:
            # Profitable: small positive reward proportional to progress
            tp_distance = abs(self._trade_tp_level - self._trade_entry_price)
            if tp_distance > 0:
                progress = (current_price - self._trade_entry_price) / tp_distance
                if self._trade_direction < 0:
                    progress = (self._trade_entry_price - current_price) / tp_distance
                return min(0.1, progress * 0.05)
        else:
            # Losing: small negative shaping
            return trade_return * 10

        return 0.0

    def _calculate_volatility(self) -> float:
        """Calculate volatility from price history."""
        if len(self._prices_history) < 5:
            return 0.02  # Default 2% volatility

        prices = np.array(self._prices_history[-self.tb_config.vol_lookback:])
        returns = np.diff(prices) / prices[:-1]

        if len(returns) < 2:
            return 0.02

        return float(np.std(returns)) if np.std(returns) > 0 else 0.02

    def update_trade_state(
        self,
        position: float,
        entry_price: float,
        current_price: float,
    ) -> Dict[str, Any]:
        """
        Update and return Triple Barrier trade state.

        Useful for debugging and visualization.

        Returns:
            Dict with trade state info.
        """
        return {
            "direction": self._trade_direction,
            "entry_price": self._trade_entry_price,
            "tp_level": self._trade_tp_level,
            "sl_level": self._trade_sl_level,
            "holding_bars": self._current_bar - self._trade_entry_bar,
            "max_holding": self.tb_config.max_holding_bars,
            "current_price": current_price,
            "current_return": (
                (current_price - self._trade_entry_price) / self._trade_entry_price
                if self._trade_entry_price > 0 else 0
            ),
        }


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
