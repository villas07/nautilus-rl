"""Tests for the risk manager."""

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from live.risk_manager import RiskManager, RiskLimits, RiskAction


class TestRiskManager:
    """Test cases for RiskManager."""

    def test_allow_order(self):
        """Test that normal orders are allowed."""
        limits = RiskLimits(
            max_position_size=10000,
            max_total_exposure=50000,
        )
        rm = RiskManager(limits=limits)

        action, qty, reason = rm.check_order(
            instrument_id="SPY.IBKR",
            side="buy",
            quantity=10,
            price=100,
        )

        assert action == RiskAction.ALLOW
        assert qty == 10

    def test_block_over_daily_loss(self):
        """Test blocking orders when daily loss exceeded."""
        limits = RiskLimits(max_daily_loss=100)
        rm = RiskManager(limits=limits)

        # Simulate loss
        rm.record_pnl(-150)

        action, qty, reason = rm.check_order(
            instrument_id="SPY.IBKR",
            side="buy",
            quantity=10,
            price=100,
        )

        assert action == RiskAction.BLOCK
        assert "Daily loss limit" in reason

    def test_reduce_over_trade_size(self):
        """Test reducing order when over max trade size."""
        limits = RiskLimits(max_trade_size=500)
        rm = RiskManager(limits=limits)

        action, qty, reason = rm.check_order(
            instrument_id="SPY.IBKR",
            side="buy",
            quantity=10,
            price=100,  # 10 * 100 = 1000 > 500
        )

        assert action == RiskAction.REDUCE
        assert qty == 5  # 500 / 100

    def test_circuit_breaker(self):
        """Test circuit breaker triggers."""
        limits = RiskLimits(consecutive_losses_limit=3)
        rm = RiskManager(limits=limits)

        # Simulate consecutive losses
        for _ in range(3):
            rm.record_pnl(-10)

        assert rm.state.is_paused

        action, _, reason = rm.check_order(
            instrument_id="SPY.IBKR",
            side="buy",
            quantity=10,
            price=100,
        )

        assert action == RiskAction.BLOCK
        assert "circuit breaker" in reason.lower()

    def test_drawdown_limit(self):
        """Test drawdown limit enforcement."""
        limits = RiskLimits(max_drawdown=0.05)
        rm = RiskManager(limits=limits, initial_equity=100000)

        # Simulate 6% drawdown
        rm.record_pnl(-6000)

        action, _, reason = rm.check_order(
            instrument_id="SPY.IBKR",
            side="buy",
            quantity=10,
            price=100,
        )

        assert action == RiskAction.BLOCK
        assert "Drawdown" in reason

    def test_position_tracking(self):
        """Test position tracking."""
        rm = RiskManager()

        # Record a trade
        rm.record_trade(
            instrument_id="SPY.IBKR",
            side="buy",
            quantity=10,
            price=100,
        )

        assert rm.state.total_positions == 1
        assert rm.state.total_exposure == 1000

        # Close position
        rm.record_trade(
            instrument_id="SPY.IBKR",
            side="sell",
            quantity=10,
            price=105,
            pnl=50,
        )

        assert rm.state.total_positions == 0

    def test_calculate_position_size(self):
        """Test position size calculation."""
        limits = RiskLimits(max_risk_per_trade=0.02)
        rm = RiskManager(limits=limits, initial_equity=100000)

        size = rm.calculate_position_size(
            price=100,
            stop_loss_price=95,  # $5 risk per share
        )

        # Risk amount = 100000 * 0.02 = 2000
        # Size = 2000 / 5 = 400 shares
        assert size <= 400

    def test_get_status(self):
        """Test status reporting."""
        rm = RiskManager()
        status = rm.get_status()

        assert "is_paused" in status
        assert "daily_pnl" in status
        assert "drawdown" in status
        assert "total_positions" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
