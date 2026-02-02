"""Base strategy class for NautilusTrader."""

from abc import abstractmethod
from typing import Optional, Dict, Any
from decimal import Decimal

from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.model.data import Bar, QuoteTick, TradeTick
from nautilus_trader.model.identifiers import InstrumentId, TraderId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder, LimitOrder
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.position import Position

import structlog

logger = structlog.get_logger()


class BaseStrategyConfig(StrategyConfig, frozen=True):
    """Base configuration for strategies."""

    instrument_id: str
    max_position_size: float = 100.0
    risk_per_trade: float = 0.02
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04


class BaseStrategy(Strategy):
    """
    Base strategy class with common functionality.

    Provides:
    - Position management
    - Risk controls
    - Order helpers
    - Logging
    """

    def __init__(self, config: BaseStrategyConfig) -> None:
        """Initialize the strategy."""
        super().__init__(config)

        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.max_position_size = config.max_position_size
        self.risk_per_trade = config.risk_per_trade
        self.stop_loss_pct = config.stop_loss_pct
        self.take_profit_pct = config.take_profit_pct

        self.instrument: Optional[Instrument] = None
        self._last_bar: Optional[Bar] = None

    def on_start(self) -> None:
        """Called when strategy starts."""
        self.instrument = self.cache.instrument(self.instrument_id)
        if self.instrument is None:
            self.log.error(f"Instrument {self.instrument_id} not found")
            return

        self.subscribe_bars(self.instrument_id)
        self.log.info(f"Strategy started for {self.instrument_id}")

    def on_stop(self) -> None:
        """Called when strategy stops."""
        self.close_all_positions()
        self.log.info(f"Strategy stopped for {self.instrument_id}")

    def on_bar(self, bar: Bar) -> None:
        """Handle bar updates."""
        self._last_bar = bar

    def on_order_filled(self, event) -> None:
        """Handle order fill events."""
        self.log.info(f"Order filled: {event}")

    def on_position_opened(self, event) -> None:
        """Handle position opened events."""
        self.log.info(f"Position opened: {event}")

    def on_position_closed(self, event) -> None:
        """Handle position closed events."""
        self.log.info(f"Position closed: {event}")

    # ========================================================================
    # Position Management
    # ========================================================================

    def get_position(self) -> Optional[Position]:
        """Get current position for this instrument."""
        positions = self.cache.positions(instrument_id=self.instrument_id)
        return positions[0] if positions else None

    def get_position_size(self) -> float:
        """Get current position size (signed)."""
        position = self.get_position()
        if position is None:
            return 0.0
        return float(position.quantity) if position.is_long else -float(position.quantity)

    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.get_position() is None

    def is_long(self) -> bool:
        """Check if position is long."""
        position = self.get_position()
        return position is not None and position.is_long

    def is_short(self) -> bool:
        """Check if position is short."""
        position = self.get_position()
        return position is not None and position.is_short

    # ========================================================================
    # Order Helpers
    # ========================================================================

    def calculate_position_size(
        self,
        price: float,
        stop_loss: Optional[float] = None,
    ) -> float:
        """
        Calculate position size based on risk.

        Uses risk_per_trade and account equity to size positions.
        """
        account = self.portfolio.account(self.instrument_id.venue)
        if account is None:
            return self.max_position_size

        equity = float(account.balance_total().as_double())
        risk_amount = equity * self.risk_per_trade

        if stop_loss is None:
            stop_loss = price * (1 - self.stop_loss_pct)

        risk_per_unit = abs(price - stop_loss)
        if risk_per_unit <= 0:
            return self.max_position_size

        size = risk_amount / risk_per_unit
        return min(size, self.max_position_size)

    def buy_market(self, quantity: float) -> None:
        """Submit market buy order."""
        if self.instrument is None:
            return

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(quantity),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        self.log.info(f"Submitted BUY market order: {quantity}")

    def sell_market(self, quantity: float) -> None:
        """Submit market sell order."""
        if self.instrument is None:
            return

        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(quantity),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        self.log.info(f"Submitted SELL market order: {quantity}")

    def buy_limit(self, quantity: float, price: float) -> None:
        """Submit limit buy order."""
        if self.instrument is None:
            return

        order = self.order_factory.limit(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(quantity),
            price=self.instrument.make_price(price),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    def sell_limit(self, quantity: float, price: float) -> None:
        """Submit limit sell order."""
        if self.instrument is None:
            return

        order = self.order_factory.limit(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(quantity),
            price=self.instrument.make_price(price),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)

    def close_position(self) -> None:
        """Close current position."""
        position = self.get_position()
        if position is None:
            return

        if position.is_long:
            self.sell_market(float(position.quantity))
        else:
            self.buy_market(float(position.quantity))

    def close_all_positions(self) -> None:
        """Close all positions for this strategy."""
        self.close_position()

    # ========================================================================
    # Abstract Methods
    # ========================================================================

    @abstractmethod
    def generate_signal(self) -> int:
        """
        Generate trading signal.

        Returns:
            1 for buy, -1 for sell, 0 for hold.
        """
        pass
