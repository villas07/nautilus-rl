"""
Risk Manager for Live Trading (R-012)

Implements risk controls and circuit breakers for live trading:
- Position limits
- Daily loss limits
- Drawdown limits
- Volatility-based sizing
- Pre-trade validation
- Circuit breakers

Reference: EVAL-002, governance/evaluations/EVAL-002_gaps_analysis.md
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum

import numpy as np

import structlog

logger = structlog.get_logger()


class RiskAction(str, Enum):
    """Risk management actions."""

    ALLOW = "allow"
    REDUCE = "reduce"
    BLOCK = "block"
    CLOSE_ALL = "close_all"


@dataclass
class RiskLimits:
    """Risk limits configuration."""

    # Position limits
    max_position_size: float = 10000.0  # Max per position
    max_total_exposure: float = 50000.0  # Total across all positions
    max_positions: int = 10  # Max concurrent positions

    # Loss limits
    max_daily_loss: float = 500.0  # Max loss per day
    max_weekly_loss: float = 2000.0  # Max loss per week
    max_drawdown: float = 0.10  # 10% drawdown limit

    # Per-trade limits
    max_risk_per_trade: float = 0.02  # 2% of capital
    max_trade_size: float = 5000.0  # Max single trade

    # Volatility limits
    max_portfolio_volatility: float = 0.20  # 20% annualized
    reduce_size_above_volatility: float = 0.15  # Start reducing at 15%

    # Circuit breakers
    consecutive_losses_limit: int = 5  # Pause after N losses
    cooldown_minutes: int = 30  # Cooldown after circuit break


@dataclass
class RiskState:
    """Current risk state."""

    current_date: date = field(default_factory=date.today)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_equity: float = 100000.0
    current_equity: float = 100000.0
    current_drawdown: float = 0.0
    consecutive_losses: int = 0
    total_positions: int = 0
    total_exposure: float = 0.0
    last_circuit_break: Optional[datetime] = None
    is_paused: bool = False


class RiskManager:
    """
    Risk management system for live trading.

    Monitors positions, PnL, and enforces risk limits.
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        initial_equity: float = 100000.0,
    ):
        """
        Initialize risk manager.

        Args:
            limits: Risk limits configuration.
            initial_equity: Starting equity.
        """
        self.limits = limits or RiskLimits()
        self.state = RiskState(
            peak_equity=initial_equity,
            current_equity=initial_equity,
        )

        # Track positions
        self._positions: Dict[str, Dict[str, Any]] = {}

    def check_order(
        self,
        instrument_id: str,
        side: str,  # "buy" or "sell"
        quantity: float,
        price: float,
    ) -> tuple:
        """
        Check if an order is allowed.

        Args:
            instrument_id: Instrument being traded.
            side: Order side.
            quantity: Order quantity.
            price: Current/expected price.

        Returns:
            Tuple of (RiskAction, adjusted_quantity, reason).
        """
        order_value = quantity * price

        # Check if trading is paused
        if self.state.is_paused:
            if not self._check_cooldown():
                return RiskAction.BLOCK, 0, "Trading paused (circuit breaker)"

        # Check daily loss limit
        if self.state.daily_pnl < -self.limits.max_daily_loss:
            return RiskAction.BLOCK, 0, f"Daily loss limit reached: ${self.state.daily_pnl:.2f}"

        # Check drawdown limit
        if self.state.current_drawdown > self.limits.max_drawdown:
            return RiskAction.BLOCK, 0, f"Drawdown limit reached: {self.state.current_drawdown:.1%}"

        # Check position count
        if instrument_id not in self._positions:
            if self.state.total_positions >= self.limits.max_positions:
                return RiskAction.BLOCK, 0, f"Max positions reached: {self.state.total_positions}"

        # Check per-trade size
        if order_value > self.limits.max_trade_size:
            adjusted_qty = self.limits.max_trade_size / price
            return RiskAction.REDUCE, adjusted_qty, f"Trade size reduced to ${self.limits.max_trade_size}"

        # Check total exposure
        new_exposure = self.state.total_exposure + order_value
        if new_exposure > self.limits.max_total_exposure:
            available = self.limits.max_total_exposure - self.state.total_exposure
            if available <= 0:
                return RiskAction.BLOCK, 0, "Max exposure reached"

            adjusted_qty = available / price
            return RiskAction.REDUCE, adjusted_qty, f"Exposure limited: ${available:.2f} available"

        # Check position size
        current_position = self._positions.get(instrument_id, {}).get("value", 0)
        new_position = current_position + order_value if side == "buy" else current_position - order_value

        if abs(new_position) > self.limits.max_position_size:
            return RiskAction.REDUCE, 0, f"Position size limit: ${self.limits.max_position_size}"

        return RiskAction.ALLOW, quantity, "Order approved"

    def record_trade(
        self,
        instrument_id: str,
        side: str,
        quantity: float,
        price: float,
        pnl: float = 0.0,
    ) -> None:
        """
        Record an executed trade.

        Args:
            instrument_id: Instrument traded.
            side: Trade side.
            quantity: Trade quantity.
            price: Execution price.
            pnl: Realized PnL (if closing position).
        """
        trade_value = quantity * price

        # Update position
        if instrument_id not in self._positions:
            self._positions[instrument_id] = {
                "quantity": 0,
                "value": 0,
                "entry_price": price,
            }

        position = self._positions[instrument_id]

        if side == "buy":
            position["quantity"] += quantity
            position["value"] += trade_value
        else:
            position["quantity"] -= quantity
            position["value"] -= trade_value

        # Clean up closed positions
        if abs(position["quantity"]) < 0.01:
            del self._positions[instrument_id]

        # Update state
        self._update_exposure()

        # Update PnL
        if pnl != 0:
            self.record_pnl(pnl)

        logger.info(
            f"Trade recorded: {side} {quantity} {instrument_id} @ {price}",
            pnl=pnl,
        )

    def record_pnl(self, pnl: float) -> None:
        """Record realized PnL."""
        # Update daily/weekly PnL
        current_date = date.today()

        if current_date != self.state.current_date:
            # New day, reset daily PnL
            self.state.daily_pnl = 0
            self.state.current_date = current_date

            # Check if new week
            if current_date.weekday() == 0:  # Monday
                self.state.weekly_pnl = 0

        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl

        # Update equity
        self.state.current_equity += pnl

        # Update peak and drawdown
        if self.state.current_equity > self.state.peak_equity:
            self.state.peak_equity = self.state.current_equity
            self.state.current_drawdown = 0
        else:
            self.state.current_drawdown = (
                (self.state.peak_equity - self.state.current_equity)
                / self.state.peak_equity
            )

        # Update consecutive losses
        if pnl < 0:
            self.state.consecutive_losses += 1
            self._check_circuit_breaker()
        else:
            self.state.consecutive_losses = 0

    def _check_circuit_breaker(self) -> None:
        """Check and trigger circuit breaker if needed."""
        if self.state.consecutive_losses >= self.limits.consecutive_losses_limit:
            self.state.is_paused = True
            self.state.last_circuit_break = datetime.now()
            logger.warning(
                f"Circuit breaker triggered: {self.state.consecutive_losses} consecutive losses"
            )

    def _check_cooldown(self) -> bool:
        """Check if cooldown period has passed."""
        if self.state.last_circuit_break is None:
            return True

        elapsed = (datetime.now() - self.state.last_circuit_break).total_seconds() / 60

        if elapsed >= self.limits.cooldown_minutes:
            self.state.is_paused = False
            self.state.consecutive_losses = 0
            logger.info("Cooldown period ended, trading resumed")
            return True

        return False

    def _update_exposure(self) -> None:
        """Update total exposure and position count."""
        self.state.total_positions = len(self._positions)
        self.state.total_exposure = sum(
            abs(p["value"]) for p in self._positions.values()
        )

    def calculate_position_size(
        self,
        price: float,
        stop_loss_price: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Calculate appropriate position size.

        Args:
            price: Current price.
            stop_loss_price: Stop loss level (optional).
            volatility: Current volatility (optional).

        Returns:
            Recommended position size.
        """
        # Base size from risk per trade
        risk_amount = self.state.current_equity * self.limits.max_risk_per_trade

        if stop_loss_price:
            risk_per_unit = abs(price - stop_loss_price)
            if risk_per_unit > 0:
                base_size = risk_amount / risk_per_unit
            else:
                base_size = self.limits.max_trade_size / price
        else:
            # Default: use max trade size
            base_size = self.limits.max_trade_size / price

        # Adjust for volatility
        if volatility and volatility > self.limits.reduce_size_above_volatility:
            vol_ratio = self.limits.reduce_size_above_volatility / volatility
            base_size *= vol_ratio

        # Ensure within limits
        max_size = self.limits.max_trade_size / price
        return min(base_size, max_size)

    def get_status(self) -> Dict[str, Any]:
        """Get current risk status."""
        return {
            "is_paused": self.state.is_paused,
            "daily_pnl": self.state.daily_pnl,
            "weekly_pnl": self.state.weekly_pnl,
            "current_equity": self.state.current_equity,
            "drawdown": self.state.current_drawdown,
            "consecutive_losses": self.state.consecutive_losses,
            "total_positions": self.state.total_positions,
            "total_exposure": self.state.total_exposure,
            "remaining_daily_loss": self.limits.max_daily_loss + self.state.daily_pnl,
            "remaining_exposure": self.limits.max_total_exposure - self.state.total_exposure,
        }

    def close_all_positions(self) -> List[str]:
        """
        Signal to close all positions.

        Returns list of instrument_ids that need closing.
        """
        instruments = list(self._positions.keys())
        logger.warning(f"Close all positions requested: {len(instruments)} positions")
        return instruments

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self.state.daily_pnl = 0
        self.state.current_date = date.today()
        logger.info("Daily risk counters reset")
