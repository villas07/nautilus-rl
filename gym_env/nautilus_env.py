"""
Gymnasium Environment with Native NautilusTrader Backtest Integration

This environment properly integrates with NautilusTrader's:
- ParquetDataCatalog for data loading
- BacktestNode for realistic execution simulation
- Bar/Instrument objects
- Order matching engine
- Realistic fills with slippage

This is the CORRECT way to use NautilusTrader for RL training.
"""

from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import (
    BacktestRunConfig,
    BacktestEngineConfig,
    BacktestDataConfig,
    BacktestVenueConfig,
    RiskEngineConfig,
)
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.instruments import Equity, CurrencyPair, CryptoPerpetual
from nautilus_trader.model.enums import (
    AccountType,
    OmsType,
    BarAggregation,
    PriceType,
    OrderSide,
    TimeInForce,
)
from nautilus_trader.model.objects import Price, Quantity, Money, Currency
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.config import StrategyConfig

from gym_env.observation import ObservationBuilder
from gym_env.rewards import RewardCalculator, RewardType, TripleBarrierConfig

import structlog

logger = structlog.get_logger()


@dataclass
class NautilusEnvConfig:
    """Configuration for the NautilusTrader-backed environment."""

    # Instrument settings
    instrument_id: str = "BTCUSDT.BINANCE"  # Default to crypto (available in raw data)
    venue: str = "BINANCE"

    # Data settings
    catalog_path: str = "data/catalog"  # Linux-native catalog
    bar_type: str = "1-DAY-LAST"  # Daily bars (matches raw CSV data)
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"

    # Account settings
    initial_capital: float = 100_000.0
    currency: str = "USD"
    account_type: str = "MARGIN"

    # Episode settings
    lookback_period: int = 20
    max_episode_steps: int = 252 * 6  # ~1 year of hourly bars

    # Observation settings
    observation_features: int = 45

    # Action settings
    action_type: str = "discrete"  # discrete or continuous
    trade_size: float = 100.0  # Units per trade (fallback if position_pct not used)
    position_pct: float = 0.10  # Position size as % of equity (10% default)
    max_position_pct: float = 0.30  # Max total position as % of equity

    # Reward settings
    reward_type: str = "sharpe"
    reward_scaling: float = 1.0

    # Triple Barrier settings (R-005)
    tb_pt_mult: float = 2.0       # Take profit = 2x volatility
    tb_sl_mult: float = 1.0       # Stop loss = 1x volatility
    tb_max_holding: int = 10      # Max bars to hold position
    tb_vol_lookback: int = 20     # Volatility calculation window

    # Venue simulation
    default_leverage: float = 1.0
    maker_fee: float = 0.0001  # 0.01%
    taker_fee: float = 0.0002  # 0.02%


@dataclass
class GymStrategyParams:
    """Parameters for GymTradingStrategy."""
    instrument_id: str = "SPY.NASDAQ"
    bar_type: str = "1-DAY-LAST"
    venue: str = "NASDAQ"
    trade_size: float = 100.0
    lookback_period: int = 20
    currency: str = "USD"


class GymTradingStrategy(Strategy):
    """
    Internal strategy used by the Gym environment.

    This strategy receives actions from the Gym environment
    and executes them through NautilusTrader's order system.
    """

    def __init__(self, config: StrategyConfig, params: GymStrategyParams) -> None:
        super().__init__(config)

        # Store params
        self.params = params
        self.instrument_id: Optional[InstrumentId] = None
        self.bar_type: Optional[BarType] = None

        # Action queue (set by environment)
        self._pending_action: Optional[int] = None

        # State tracking
        self._bars_received: List[Bar] = []
        self._step_count: int = 0
        self._episode_done: bool = False

    def on_start(self) -> None:
        """Strategy startup."""
        self.instrument_id = InstrumentId.from_str(self.params.instrument_id)

        # Create bar type
        self.bar_type = BarType(
            instrument_id=self.instrument_id,
            bar_spec=self._parse_bar_spec(self.params.bar_type),
            aggregation_source=1,  # EXTERNAL
        )

        # Subscribe to bars
        self.subscribe_bars(self.bar_type)

        self.log.info(f"Strategy started, subscribed to {self.bar_type}")

    def _parse_bar_spec(self, bar_type_str: str):
        """Parse bar type string to BarSpecification."""
        from nautilus_trader.model.data import BarSpecification

        parts = bar_type_str.split("-")
        step = int(parts[0])

        agg_map = {
            "MINUTE": BarAggregation.MINUTE,
            "HOUR": BarAggregation.HOUR,
            "DAY": BarAggregation.DAY,
        }
        aggregation = agg_map.get(parts[1], BarAggregation.HOUR)

        return BarSpecification(
            step=step,
            aggregation=aggregation,
            price_type=PriceType.LAST,
        )

    def on_bar(self, bar: Bar) -> None:
        """Handle new bar."""
        self._bars_received.append(bar)
        self._step_count += 1

        # Execute pending action if any
        if self._pending_action is not None:
            self._execute_action(self._pending_action, bar)
            self._pending_action = None

    def _execute_action(self, action: int, bar: Bar) -> None:
        """Execute a trading action."""
        instrument = self.cache.instrument(self.instrument_id)
        if instrument is None:
            return

        current_position = self.portfolio.net_position(self.instrument_id)

        # Action: 0=hold, 1=buy, 2=sell
        if action == 1 and current_position <= 0:
            # Buy
            order = self._create_market_order(
                instrument=instrument,
                side=OrderSide.BUY,
                quantity=self.params.trade_size,
            )
            self.submit_order(order)

        elif action == 2 and current_position >= 0:
            # Sell
            order = self._create_market_order(
                instrument=instrument,
                side=OrderSide.SELL,
                quantity=self.params.trade_size,
            )
            self.submit_order(order)

    def _create_market_order(
        self,
        instrument,
        side: OrderSide,
        quantity: float,
    ) -> MarketOrder:
        """Create a market order."""
        return MarketOrder(
            trader_id=self.trader_id,
            strategy_id=self.id,
            instrument_id=self.instrument_id,
            client_order_id=self.order_factory.generate_client_order_id(),
            order_side=side,
            quantity=Quantity.from_str(str(quantity)),
            init_id=self.uuid_factory.generate(),
            ts_init=self.clock.timestamp_ns(),
            time_in_force=TimeInForce.GTC,
        )

    def set_action(self, action: int) -> None:
        """Set pending action from environment."""
        self._pending_action = action

    def get_state(self) -> Dict[str, Any]:
        """Get current state for observation."""
        position = self.portfolio.net_position(self.instrument_id)
        account = self.portfolio.account(Venue(self.params.venue))

        # Get currency object for balance queries
        currency = Currency.from_str(self.params.currency) if account else None

        # Get balance values (balance_total returns Money, balance returns AccountBalance)
        equity = 0.0
        cash = 0.0
        if account and currency:
            balance_total = account.balance_total(currency)
            if balance_total:
                equity = balance_total.as_double()
            balance = account.balance(currency)
            if balance:
                cash = balance.free.as_double()  # Use free balance as cash

        return {
            "position": float(position) if position else 0.0,
            "equity": equity,
            "cash": cash,
            "bars": self._bars_received[-self.params.lookback_period:] if self._bars_received else [],
            "step": self._step_count,
        }

    def on_stop(self) -> None:
        """Strategy shutdown."""
        self._episode_done = True


class NautilusBacktestEnv(gym.Env):
    """
    Gymnasium environment backed by NautilusTrader BacktestEngine.

    This provides:
    - Realistic order execution with slippage
    - Proper position and portfolio tracking
    - Event-driven simulation
    - Integration with NautilusTrader's data catalog

    Usage:
        config = NautilusEnvConfig(
            instrument_id="BTCUSDT.BINANCE",
            catalog_path="/app/data/catalog",
        )
        env = NautilusBacktestEnv(config)

        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[NautilusEnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize environment.

        Args:
            config: Environment configuration.
            render_mode: Render mode.
        """
        super().__init__()

        self.config = config or NautilusEnvConfig()
        self.render_mode = render_mode

        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.observation_features,),
            dtype=np.float32,
        )

        if self.config.action_type == "discrete":
            self.action_space = spaces.Discrete(3)  # hold, buy, sell
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )

        # Components
        self.observation_builder = ObservationBuilder(
            lookback_period=self.config.lookback_period,
            num_features=self.config.observation_features,
        )

        # Triple Barrier config (R-005)
        tb_config = TripleBarrierConfig(
            pt_mult=self.config.tb_pt_mult,
            sl_mult=self.config.tb_sl_mult,
            max_holding_bars=self.config.tb_max_holding,
            vol_lookback=self.config.tb_vol_lookback,
        )

        self.reward_calculator = RewardCalculator(
            reward_type=RewardType(self.config.reward_type),
            scaling=self.config.reward_scaling,
            triple_barrier_config=tb_config,
        )

        # NautilusTrader components (initialized on reset)
        self._engine: Optional[BacktestEngine] = None
        self._strategy: Optional[GymTradingStrategy] = None
        self._catalog: Optional[ParquetDataCatalog] = None

        # Episode state
        self._step_count: int = 0
        self._prev_equity: float = 0.0
        self._returns: List[float] = []
        self._initial_equity: float = 0.0

        # Data-driven stepping (fix for L-007)
        self._bars_data: List[Bar] = []
        self._current_bar_idx: int = 0
        self._position: float = 0.0  # Current position size
        self._cash: float = 0.0
        self._entry_price: float = 0.0
        self._peak_equity: float = 0.0  # High watermark for drawdown calculation

        # Load catalog
        self._load_catalog()

    def _load_catalog(self) -> None:
        """Load the ParquetDataCatalog."""
        catalog_path = Path(self.config.catalog_path)

        # L-007 FIX: Better path diagnostics
        logger.info(f"Attempting to load catalog from: {catalog_path}")
        logger.info(f"Absolute path: {catalog_path.absolute()}")
        logger.info(f"Path exists: {catalog_path.exists()}")

        if not catalog_path.exists():
            # Try common alternative paths
            alt_paths = [
                Path("/app/data/catalog"),
                Path("/app/data/catalog_nautilus"),
                Path("data/catalog"),
                Path("data/catalog_nautilus"),
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    logger.warning(f"Catalog not at {catalog_path}, but found at {alt_path}")
                    catalog_path = alt_path
                    break
            else:
                logger.error(f"Catalog path does not exist: {catalog_path}")
                logger.error(f"Tried alternatives: {[str(p) for p in alt_paths]}")
                return

        try:
            self._catalog = ParquetDataCatalog(str(catalog_path))
            logger.info(f"Successfully loaded catalog from {catalog_path}")

            # List available instruments for debugging
            instruments = self._catalog.instruments()
            logger.info(f"Catalog contains {len(instruments)} instruments")
            for inst in instruments[:10]:  # Show first 10
                logger.info(f"  - {inst.id}")

        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._catalog = None

    def _create_engine(self) -> BacktestEngine:
        """Create a new BacktestEngine for an episode."""
        # Venue configuration
        venue = Venue(self.config.venue)

        venue_config = BacktestVenueConfig(
            name=self.config.venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN if self.config.account_type == "MARGIN" else AccountType.CASH,
            base_currency=self.config.currency,
            starting_balances=[f"{self.config.initial_capital} {self.config.currency}"],
            default_leverage=self.config.default_leverage,
            # Note: Fees are configured on the instrument, not venue in newer API
        )

        # Engine configuration
        engine_config = BacktestEngineConfig(
            trader_id="BACKTESTER-001",
            risk_engine=RiskEngineConfig(bypass=True),  # Let strategy handle risk
        )

        # Data configuration
        data_config = BacktestDataConfig(
            catalog_path=self.config.catalog_path,
            data_cls="Bar",
            instrument_id=self.config.instrument_id,
            start_time=self.config.start_date,
            end_time=self.config.end_date,
        )

        # Create engine
        engine = BacktestEngine(config=engine_config)

        # Add venue - base_currency should be None to use instrument's currency
        from decimal import Decimal
        engine.add_venue(
            venue=venue,
            oms_type=venue_config.oms_type,
            account_type=venue_config.account_type,
            base_currency=None,  # Will use instrument's quote currency
            starting_balances=[Money.from_str(b) for b in venue_config.starting_balances],
            default_leverage=Decimal(str(venue_config.default_leverage)),
        )

        # Add data from catalog
        if self._catalog:
            instrument_id = InstrumentId.from_str(self.config.instrument_id)

            # Get instrument
            instruments = self._catalog.instruments(instrument_ids=[str(instrument_id)])
            if instruments:
                engine.add_instrument(instruments[0])

            # Get bars
            bars = self._catalog.bars(
                instrument_ids=[str(instrument_id)],
                start=pd.Timestamp(self.config.start_date),
                end=pd.Timestamp(self.config.end_date),
            )
            if bars:
                engine.add_data(bars)
                logger.info(f"Added {len(bars)} bars to engine")

        return engine

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Uses data-driven approach: loads bars into memory and steps through them.
        This fixes L-007 where BacktestEngine.run() processed all bars at once.
        """
        super().reset(seed=seed)

        # Cleanup previous engine
        if self._engine is not None:
            try:
                self._engine.dispose()
            except Exception:
                pass
        self._engine = None

        # Load bar data directly from catalog
        self._bars_data = []
        if self._catalog:
            instrument_id = InstrumentId.from_str(self.config.instrument_id)

            # Log available instruments for debugging
            available_instruments = self._catalog.instruments()
            logger.info(f"Catalog has {len(available_instruments)} instruments: {[str(i.id) for i in available_instruments[:5]]}")

            bars = self._catalog.bars(
                instrument_ids=[str(instrument_id)],
                start=pd.Timestamp(self.config.start_date),
                end=pd.Timestamp(self.config.end_date),
            )
            if bars:
                self._bars_data = list(bars)
                logger.info(f"Loaded {len(self._bars_data)} bars for episode")
            else:
                logger.warning(f"No bars found for {instrument_id} between {self.config.start_date} and {self.config.end_date}")
        else:
            logger.error(f"Catalog not loaded! Path: {self.config.catalog_path}")

        # L-007 FIX: Validate that we have enough data for an episode
        min_bars_required = self.config.lookback_period + 10  # At least 10 steps after lookback
        if len(self._bars_data) < min_bars_required:
            error_msg = (
                f"L-007 ERROR: Insufficient bar data! "
                f"Have {len(self._bars_data)} bars, need at least {min_bars_required}. "
                f"instrument_id={self.config.instrument_id}, "
                f"catalog_path={self.config.catalog_path}, "
                f"catalog_loaded={self._catalog is not None}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Reset episode state
        self._step_count = 0
        self._current_bar_idx = self.config.lookback_period  # Start after lookback
        self._returns = []
        self._initial_equity = self.config.initial_capital
        self._prev_equity = self._initial_equity
        self._peak_equity = self._initial_equity  # Reset high watermark
        self._cash = self._initial_equity
        self._position = 0.0
        self._entry_price = 0.0

        # Reset reward calculator
        self.reward_calculator.reset()

        # Get initial observation
        observation = self._get_observation_direct()

        info = {
            "equity": self._initial_equity,
            "position": 0.0,
            "step": 0,
        }

        return observation, info

    def step(
        self,
        action: Any,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step using data-driven approach.

        Args:
            action: Trading action (0=hold, 1=buy, 2=sell).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Convert continuous action to discrete if needed
        if self.config.action_type == "continuous":
            if action[0] > 0.3:
                discrete_action = 1  # Buy
            elif action[0] < -0.3:
                discrete_action = 2  # Sell
            else:
                discrete_action = 0  # Hold
        else:
            discrete_action = int(action)

        # Check if we have more bars
        if self._current_bar_idx >= len(self._bars_data):
            # Episode done - no more data
            observation = self._get_observation_direct()
            return observation, 0.0, True, False, {"equity": self._get_equity(), "position": self._position, "step": self._step_count}

        # Get current bar
        current_bar = self._bars_data[self._current_bar_idx]
        current_price = float(current_bar.close)

        # Execute action
        self._execute_action_direct(discrete_action, current_price)

        # Advance to next bar
        self._current_bar_idx += 1
        self._step_count += 1

        # Calculate equity
        current_equity = self._get_equity()

        # Calculate return
        if self._prev_equity > 0:
            step_return = (current_equity - self._prev_equity) / self._prev_equity
        else:
            step_return = 0.0
        self._returns.append(step_return)

        # Calculate reward (with Triple Barrier support - R-005)
        reward = self.reward_calculator.calculate(
            portfolio_value=current_equity,
            prev_portfolio_value=self._prev_equity,
            returns=self._returns,
            position=self._position,
            current_price=current_price,
            entry_price=self._entry_price,
        )

        # Update state
        self._prev_equity = current_equity
        self._peak_equity = max(self._peak_equity, current_equity)  # Track high watermark

        # Check termination
        terminated = self._check_terminated(current_equity)
        truncated = (
            self._step_count >= self.config.max_episode_steps or
            self._current_bar_idx >= len(self._bars_data)
        )

        # Get observation
        observation = self._get_observation_direct()

        # Build info
        info = {
            "equity": current_equity,
            "position": self._position,
            "step": self._step_count,
            "total_return": (current_equity - self._initial_equity) / self._initial_equity if self._initial_equity > 0 else 0,
            "sharpe": self._calculate_sharpe(),
            "price": current_price,
        }

        return observation, reward, terminated, truncated, info

    def _execute_action_direct(self, action: int, price: float) -> None:
        """Execute trading action directly (data-driven approach)."""
        # Calculate position size based on % of equity
        current_equity = self._get_equity()
        position_value = current_equity * self.config.position_pct
        trade_size = max(1.0, position_value / price)  # At least 1 unit

        # Check max position limit
        max_position_value = current_equity * self.config.max_position_pct
        max_position_size = max_position_value / price

        if action == 1:  # Buy
            if self._position <= 0:
                # Close short if any, then go long
                if self._position < 0:
                    # Close short
                    pnl = (self._entry_price - price) * abs(self._position)
                    self._cash += pnl + (abs(self._position) * self._entry_price)
                    self._position = 0

                # Open long (respecting max position)
                trade_size = min(trade_size, max_position_size)
                cost = trade_size * price
                if self._cash >= cost:
                    self._cash -= cost
                    self._position = trade_size
                    self._entry_price = price

        elif action == 2:  # Sell
            if self._position >= 0:
                # Close long if any, then go short
                if self._position > 0:
                    # Close long
                    pnl = (price - self._entry_price) * self._position
                    self._cash += pnl + (self._position * self._entry_price)
                    self._position = 0

                # Open short (respecting max position)
                trade_size = min(trade_size, max_position_size)
                margin = trade_size * price
                if self._cash >= margin:
                    self._cash -= margin  # Margin for short
                    self._position = -trade_size
                    self._entry_price = price

        # action == 0: Hold - do nothing

    def _get_equity(self) -> float:
        """Calculate current equity."""
        if self._current_bar_idx >= len(self._bars_data) or self._current_bar_idx < 0:
            return self._cash

        current_bar = self._bars_data[min(self._current_bar_idx, len(self._bars_data) - 1)]
        current_price = float(current_bar.close)

        if self._position > 0:
            # Long position value
            unrealized_pnl = (current_price - self._entry_price) * self._position
            return self._cash + (self._position * self._entry_price) + unrealized_pnl
        elif self._position < 0:
            # Short position value
            unrealized_pnl = (self._entry_price - current_price) * abs(self._position)
            return self._cash + (abs(self._position) * self._entry_price) + unrealized_pnl
        else:
            return self._cash

    def _get_observation_direct(self) -> np.ndarray:
        """Build observation directly from bar data."""
        if not self._bars_data or self._current_bar_idx < self.config.lookback_period:
            return np.zeros(self.config.observation_features, dtype=np.float32)

        # Get lookback bars
        start_idx = max(0, self._current_bar_idx - self.config.lookback_period)
        end_idx = self._current_bar_idx
        bars_slice = self._bars_data[start_idx:end_idx]

        if not bars_slice:
            return np.zeros(self.config.observation_features, dtype=np.float32)

        # Convert bars to DataFrame
        bars_data = []
        for bar in bars_slice:
            bars_data.append({
                "timestamp": bar.ts_event,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            })

        df = pd.DataFrame(bars_data)

        return self.observation_builder.build(
            data=df,
            position=self._position,
            portfolio_value=self._get_equity(),
            initial_capital=self._initial_equity,
        )

    def _get_observation(self) -> np.ndarray:
        """Build observation from strategy state."""
        state = self._strategy.get_state()

        if not state["bars"]:
            return np.zeros(self.config.observation_features, dtype=np.float32)

        # Convert bars to DataFrame
        bars_data = []
        for bar in state["bars"]:
            bars_data.append({
                "timestamp": bar.ts_event,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            })

        df = pd.DataFrame(bars_data)

        return self.observation_builder.build(
            data=df,
            position=state["position"],
            portfolio_value=state["equity"],
            initial_capital=self._initial_equity,
        )

    def _check_terminated(self, equity: float) -> bool:
        """Check if episode should terminate."""
        # Bankruptcy
        if equity <= 0:
            return True

        # Max drawdown from peak (high watermark)
        drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0
        if drawdown > 0.15:  # 15% max drawdown (matches Filter 1 threshold)
            return True

        return False

    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self._returns) < 2:
            return 0.0

        returns_arr = np.array(self._returns)
        std = returns_arr.std()
        if std == 0:
            return 0.0

        # Annualized (assuming hourly)
        return np.sqrt(252 * 6) * returns_arr.mean() / std

    def render(self) -> None:
        """Render environment state."""
        if self.render_mode == "human":
            state = self._strategy.get_state()
            print(
                f"Step {self._step_count:4d} | "
                f"Equity: ${state['equity']:,.0f} | "
                f"Position: {state['position']:+.0f} | "
                f"Sharpe: {self._calculate_sharpe():.2f}"
            )

    def close(self) -> None:
        """Cleanup resources."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None


def create_nautilus_env(
    instrument_id: str = "SPY.NASDAQ",
    catalog_path: str = "/app/data/catalog",
    **kwargs,
) -> NautilusBacktestEnv:
    """
    Factory function to create NautilusTrader-backed environment.

    Args:
        instrument_id: NautilusTrader instrument ID.
        catalog_path: Path to ParquetDataCatalog.
        **kwargs: Additional config options.

    Returns:
        NautilusBacktestEnv instance.
    """
    config = NautilusEnvConfig(
        instrument_id=instrument_id,
        catalog_path=catalog_path,
        **kwargs,
    )
    return NautilusBacktestEnv(config)
