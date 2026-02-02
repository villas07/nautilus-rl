"""Action space handling for the trading environment."""

from enum import Enum
from typing import Optional, Tuple, Union
import numpy as np


class ActionType(str, Enum):
    """Available action space types."""

    DISCRETE = "discrete"      # 0=hold, 1=buy, 2=sell
    CONTINUOUS = "continuous"  # [-1, 1] position target
    MULTI_DISCRETE = "multi_discrete"  # Separate buy/sell amounts


class ActionHandler:
    """
    Handles action interpretation and execution.

    Supports:
    - Discrete actions: hold, buy, sell
    - Continuous actions: position targets
    - Multi-discrete: separate buy/sell with amounts
    """

    def __init__(
        self,
        action_type: ActionType = ActionType.DISCRETE,
        max_position: float = 1.0,
        position_step: float = 0.25,
        allow_shorting: bool = True,
    ):
        """
        Initialize action handler.

        Args:
            action_type: Type of action space.
            max_position: Maximum position size (as fraction of capital).
            position_step: Step size for discrete position changes.
            allow_shorting: Whether to allow short positions.
        """
        self.action_type = action_type
        self.max_position = max_position
        self.position_step = position_step
        self.allow_shorting = allow_shorting

    def get_target_position(
        self,
        action: Union[int, float, np.ndarray],
        current_position: float = 0.0,
    ) -> float:
        """
        Convert action to target position.

        Args:
            action: Action from the policy.
            current_position: Current position.

        Returns:
            Target position (between -max_position and max_position).
        """
        if self.action_type == ActionType.DISCRETE:
            return self._discrete_to_position(action, current_position)

        elif self.action_type == ActionType.CONTINUOUS:
            return self._continuous_to_position(action)

        elif self.action_type == ActionType.MULTI_DISCRETE:
            return self._multi_discrete_to_position(action, current_position)

        else:
            return current_position

    def _discrete_to_position(
        self,
        action: int,
        current_position: float,
    ) -> float:
        """
        Convert discrete action to position.

        Actions:
            0: Hold (no change)
            1: Buy (increase position)
            2: Sell (decrease position)
        """
        action = int(action)

        if action == 0:  # Hold
            return current_position

        elif action == 1:  # Buy
            new_position = current_position + self.position_step
            return min(new_position, self.max_position)

        elif action == 2:  # Sell
            new_position = current_position - self.position_step
            if self.allow_shorting:
                return max(new_position, -self.max_position)
            else:
                return max(new_position, 0.0)

        return current_position

    def _continuous_to_position(
        self,
        action: Union[float, np.ndarray],
    ) -> float:
        """
        Convert continuous action to position.

        Action is in [-1, 1], representing position as fraction of max.
        """
        if isinstance(action, np.ndarray):
            action = float(action[0])

        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)

        # Convert to position
        target = action * self.max_position

        if not self.allow_shorting and target < 0:
            target = 0.0

        return target

    def _multi_discrete_to_position(
        self,
        action: np.ndarray,
        current_position: float,
    ) -> float:
        """
        Convert multi-discrete action to position.

        Action is [direction, amount]:
            direction: 0=hold, 1=buy, 2=sell
            amount: 0=small, 1=medium, 2=large
        """
        direction = int(action[0])
        amount = int(action[1]) if len(action) > 1 else 1

        # Amount multiplier
        amount_map = {0: 0.25, 1: 0.5, 2: 1.0}
        multiplier = amount_map.get(amount, 0.5)

        if direction == 0:  # Hold
            return current_position

        elif direction == 1:  # Buy
            delta = self.position_step * multiplier
            new_position = current_position + delta
            return min(new_position, self.max_position)

        elif direction == 2:  # Sell
            delta = self.position_step * multiplier
            new_position = current_position - delta
            if self.allow_shorting:
                return max(new_position, -self.max_position)
            else:
                return max(new_position, 0.0)

        return current_position


class ActionMasker:
    """
    Masks invalid actions based on current state.

    Useful for action-masked RL (e.g., with invalid action masking).
    """

    def __init__(
        self,
        action_type: ActionType = ActionType.DISCRETE,
        max_position: float = 1.0,
        allow_shorting: bool = True,
    ):
        """
        Initialize action masker.

        Args:
            action_type: Type of action space.
            max_position: Maximum position.
            allow_shorting: Allow short positions.
        """
        self.action_type = action_type
        self.max_position = max_position
        self.allow_shorting = allow_shorting

    def get_action_mask(
        self,
        current_position: float,
        cash_ratio: float = 1.0,
    ) -> np.ndarray:
        """
        Get mask of valid actions.

        Args:
            current_position: Current position.
            cash_ratio: Available cash as fraction of capital.

        Returns:
            Boolean mask (True = valid action).
        """
        if self.action_type == ActionType.DISCRETE:
            return self._discrete_mask(current_position, cash_ratio)
        else:
            # For continuous, no masking
            return np.array([True])

    def _discrete_mask(
        self,
        current_position: float,
        cash_ratio: float,
    ) -> np.ndarray:
        """Get mask for discrete actions."""
        mask = np.array([True, True, True])  # [hold, buy, sell]

        # Can't buy more if at max long
        if current_position >= self.max_position:
            mask[1] = False

        # Can't sell more if at max short (or can't short)
        if not self.allow_shorting and current_position <= 0:
            mask[2] = False
        elif current_position <= -self.max_position:
            mask[2] = False

        # Can't buy without cash
        if cash_ratio <= 0:
            mask[1] = False

        return mask


def discretize_action(
    continuous_action: float,
    thresholds: Tuple[float, float] = (-0.3, 0.3),
) -> int:
    """
    Convert continuous action to discrete.

    Args:
        continuous_action: Action in [-1, 1].
        thresholds: (sell_threshold, buy_threshold).

    Returns:
        0 (hold), 1 (buy), or 2 (sell).
    """
    if continuous_action < thresholds[0]:
        return 2  # Sell
    elif continuous_action > thresholds[1]:
        return 1  # Buy
    else:
        return 0  # Hold


def continuize_action(
    discrete_action: int,
    magnitude: float = 0.5,
) -> float:
    """
    Convert discrete action to continuous.

    Args:
        discrete_action: 0 (hold), 1 (buy), 2 (sell).
        magnitude: Strength of action.

    Returns:
        Action in [-1, 1].
    """
    action_map = {
        0: 0.0,
        1: magnitude,
        2: -magnitude,
    }
    return action_map.get(discrete_action, 0.0)
