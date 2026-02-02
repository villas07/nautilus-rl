"""RL-based trading strategy for NautilusTrader."""

from typing import Optional, Dict, Any, List
from pathlib import Path
import numpy as np

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId

from strategies.base import BaseStrategy, BaseStrategyConfig

import structlog

logger = structlog.get_logger()


class RLTradingStrategyConfig(BaseStrategyConfig, frozen=True):
    """Configuration for RL trading strategy."""

    # Model settings
    models_dir: str = "/app/models"
    use_voting: bool = True
    min_confidence: float = 0.6  # Minimum voting confidence to trade

    # Bar settings
    bar_type: str = "1-HOUR-LAST"

    # Trading settings
    position_sizing: str = "fixed"  # fixed, risk_based, kelly
    fixed_size: float = 1.0


class RLTradingStrategy(BaseStrategy):
    """
    Trading strategy that uses RL model predictions.

    Features:
    - Loads multiple RL models (ensemble)
    - Uses voting system for signal aggregation
    - Confidence-based position sizing
    - Integrates with NautilusTrader execution
    """

    def __init__(self, config: RLTradingStrategyConfig) -> None:
        """Initialize the RL strategy."""
        super().__init__(config)

        self.models_dir = Path(config.models_dir)
        self.use_voting = config.use_voting
        self.min_confidence = config.min_confidence
        self.bar_type_str = config.bar_type
        self.position_sizing = config.position_sizing
        self.fixed_size = config.fixed_size

        # Models and voting system (loaded on start)
        self.models: List[Any] = []
        self.voting_system: Optional["VotingSystem"] = None

        # State
        self.bar_type: Optional[BarType] = None
        self.observation_buffer: List[Bar] = []
        self.lookback_period: int = 20  # Bars for observation

    def on_start(self) -> None:
        """Initialize strategy on start."""
        super().on_start()

        # Parse bar type
        self.bar_type = BarType.from_str(
            f"{self.instrument_id}-{self.bar_type_str}"
        )

        # Subscribe to bars
        self.subscribe_bars(self.bar_type)

        # Load models
        self._load_models()

        self.log.info(
            f"RLTradingStrategy started with {len(self.models)} models"
        )

    def _load_models(self) -> None:
        """Load RL models from disk."""
        try:
            from live.model_loader import load_validated_models
            from live.voting_system import VotingSystem

            # Load validated models
            self.models = load_validated_models(
                models_dir=self.models_dir,
                instrument_id=str(self.instrument_id),
            )

            if self.use_voting and self.models:
                self.voting_system = VotingSystem(
                    models=self.models,
                    min_confidence=self.min_confidence,
                )

            self.log.info(f"Loaded {len(self.models)} models")

        except ImportError as e:
            self.log.warning(f"Could not load models: {e}")
            self.models = []

    def on_bar(self, bar: Bar) -> None:
        """Process new bar data."""
        super().on_bar(bar)

        # Update observation buffer
        self.observation_buffer.append(bar)
        if len(self.observation_buffer) > self.lookback_period:
            self.observation_buffer.pop(0)

        # Wait for enough data
        if len(self.observation_buffer) < self.lookback_period:
            return

        # Generate signal and execute
        signal, confidence = self.generate_signal_with_confidence()

        if signal != 0 and confidence >= self.min_confidence:
            self._execute_signal(signal, confidence)

    def generate_signal(self) -> int:
        """Generate trading signal (basic version)."""
        signal, _ = self.generate_signal_with_confidence()
        return signal

    def generate_signal_with_confidence(self) -> tuple:
        """
        Generate trading signal with confidence score.

        Returns:
            Tuple of (signal, confidence) where:
            - signal: 1 (buy), -1 (sell), 0 (hold)
            - confidence: 0.0 to 1.0
        """
        if not self.models:
            return 0, 0.0

        # Build observation
        observation = self._build_observation()

        if self.voting_system is not None:
            # Use voting system for ensemble
            return self.voting_system.get_signal(observation)
        else:
            # Single model prediction
            model = self.models[0]
            action, _ = model.predict(observation, deterministic=True)
            # Map action to signal: 0=hold, 1=buy, 2=sell
            signal_map = {0: 0, 1: 1, 2: -1}
            return signal_map.get(int(action), 0), 1.0

    def _build_observation(self) -> np.ndarray:
        """Build observation vector from bar buffer."""
        if len(self.observation_buffer) < self.lookback_period:
            return np.zeros(45)  # Empty observation

        features = []

        # Price features (OHLC normalized)
        closes = [bar.close.as_double() for bar in self.observation_buffer]
        current_close = closes[-1]

        # Returns
        returns = np.diff(closes) / closes[:-1]
        features.extend([
            returns[-1],  # Last return
            np.mean(returns),  # Mean return
            np.std(returns),  # Volatility
        ])

        # Moving averages (normalized)
        ma5 = np.mean(closes[-5:])
        ma10 = np.mean(closes[-10:])
        ma20 = np.mean(closes[-20:])
        features.extend([
            (current_close - ma5) / ma5,
            (current_close - ma10) / ma10,
            (current_close - ma20) / ma20,
            (ma5 - ma10) / ma10,
            (ma10 - ma20) / ma20,
        ])

        # High/Low features
        highs = [bar.high.as_double() for bar in self.observation_buffer]
        lows = [bar.low.as_double() for bar in self.observation_buffer]

        high_20 = max(highs)
        low_20 = min(lows)
        range_20 = high_20 - low_20 if high_20 > low_20 else 1.0

        features.extend([
            (current_close - low_20) / range_20,  # Position in range
            (high_20 - current_close) / range_20,  # Distance from high
        ])

        # Volume features
        volumes = [bar.volume.as_double() for bar in self.observation_buffer]
        avg_volume = np.mean(volumes)
        features.extend([
            volumes[-1] / avg_volume if avg_volume > 0 else 1.0,
            np.std(volumes) / avg_volume if avg_volume > 0 else 0.0,
        ])

        # RSI-like features
        gains = [max(r, 0) for r in returns]
        losses = [-min(r, 0) for r in returns]
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        features.append((rsi - 50) / 50)  # Normalized RSI

        # Momentum features
        features.extend([
            (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0,
            (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0,
        ])

        # Bollinger Band features
        std_20 = np.std(closes[-20:])
        upper_bb = ma20 + 2 * std_20
        lower_bb = ma20 - 2 * std_20
        bb_width = upper_bb - lower_bb
        features.extend([
            (current_close - lower_bb) / bb_width if bb_width > 0 else 0.5,
            bb_width / ma20,  # BB width normalized
        ])

        # Position features
        position_size = self.get_position_size()
        features.extend([
            1 if position_size > 0 else 0,  # Is long
            1 if position_size < 0 else 0,  # Is short
            min(abs(position_size) / self.max_position_size, 1.0),  # Position ratio
        ])

        # Time features (if available)
        last_bar = self.observation_buffer[-1]
        hour = last_bar.ts_event // (10**9 * 3600) % 24
        day_of_week = (last_bar.ts_event // (10**9 * 86400)) % 7
        features.extend([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
        ])

        # Pad to 45 features
        while len(features) < 45:
            features.append(0.0)

        return np.array(features[:45], dtype=np.float32)

    def _execute_signal(self, signal: int, confidence: float) -> None:
        """Execute trading signal."""
        current_position = self.get_position_size()

        # Calculate position size
        if self.position_sizing == "fixed":
            target_size = self.fixed_size * signal
        else:
            # Risk-based sizing with confidence
            base_size = self.calculate_position_size(
                self._last_bar.close.as_double() if self._last_bar else 100.0
            )
            target_size = base_size * signal * confidence

        # Clamp to max
        target_size = max(-self.max_position_size, min(self.max_position_size, target_size))

        # Calculate order size
        order_size = target_size - current_position

        if abs(order_size) < 0.01:
            return  # Too small to trade

        # Execute order
        if order_size > 0:
            self.buy_market(abs(order_size))
            self.log.info(
                f"BUY signal: size={order_size:.2f}, confidence={confidence:.2f}"
            )
        else:
            self.sell_market(abs(order_size))
            self.log.info(
                f"SELL signal: size={order_size:.2f}, confidence={confidence:.2f}"
            )

    def on_reset(self) -> None:
        """Reset strategy state."""
        self.observation_buffer.clear()
