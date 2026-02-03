"""
Institutional Observation Builder with advanced features.

Integrates:
- Standard technical features (45)
- Microstructure features (8) - R-008
- Entropy features (4) - R-008
- Regime features (6) - R-018
- Fractional differentiation features (2) - Optional

Total: 45 + 8 + 4 = 57 features (base)
       57 + 6 = 63 features (with regime)
       63 + 2 = 65 features (with frac diff)
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from dataclasses import dataclass

from gym_env.observation import ObservationBuilder
from ml_institutional.microstructure_features import (
    MicrostructureFeatures,
    MicrostructureConfig,
    get_microstructure_features,
)
from ml_institutional.entropy_features import (
    EntropyFeatures,
    EntropyConfig,
    get_entropy_features,
)
from ml_institutional.regime_detector import (
    RegimeDetector,
    RegimeDetectorConfig,
    get_regime_features,
)


@dataclass
class InstitutionalObservationConfig:
    """Configuration for institutional observation builder."""
    # Base observation config
    lookback_period: int = 20
    include_position: bool = True
    include_time: bool = True

    # Microstructure features (R-008)
    include_microstructure: bool = True
    microstructure_config: Optional[MicrostructureConfig] = None

    # Entropy features (R-008)
    include_entropy: bool = True
    entropy_config: Optional[EntropyConfig] = None

    # Regime features (R-018)
    include_regime: bool = True
    regime_config: Optional[RegimeDetectorConfig] = None

    # Fractional differentiation
    include_frac_diff: bool = False
    frac_diff_d: float = 0.4  # Differentiation order


class InstitutionalObservationBuilder:
    """
    Builds observation vectors with institutional-grade features.

    Features (57-65 total):
        Base Features (45):
            - Price features (15)
            - Volume features (5)
            - Technical indicators (10)
            - Position features (5)
            - Time features (4)
            - Portfolio features (6)

        Microstructure Features (8) - R-008:
            - Kyle Lambda (market impact)
            - Amihud Illiquidity
            - VPIN (informed trading probability)
            - Roll Spread (bid-ask estimator)
            - Corwin-Schultz Spread
            - Flow Toxicity
            - Volume Clock
            - Price Efficiency (variance ratio)

        Entropy Features (4) - R-008:
            - Shannon Entropy
            - Approximate Entropy
            - Sample Entropy
            - Lempel-Ziv Complexity

        Regime Features (6) - R-018:
            - trend_bull (one-hot)
            - trend_bear (one-hot)
            - trend_sideways (one-hot)
            - vol_low (one-hot)
            - vol_high (one-hot)
            - regime_confidence

        Fractional Diff Features (2, optional):
            - Frac diff close price
            - Frac diff volume
    """

    def __init__(self, config: Optional[InstitutionalObservationConfig] = None):
        """Initialize the institutional observation builder."""
        self.config = config or InstitutionalObservationConfig()

        # Calculate total features
        self.num_base_features = 45
        self.num_microstructure = 8 if self.config.include_microstructure else 0
        self.num_entropy = 4 if self.config.include_entropy else 0
        self.num_regime = 6 if self.config.include_regime else 0
        self.num_frac_diff = 2 if self.config.include_frac_diff else 0

        self.num_features = (
            self.num_base_features +
            self.num_microstructure +
            self.num_entropy +
            self.num_regime +
            self.num_frac_diff
        )

        # Initialize base builder
        self.base_builder = ObservationBuilder(
            lookback_period=self.config.lookback_period,
            num_features=self.num_base_features,
            include_position=self.config.include_position,
            include_time=self.config.include_time,
        )

        # Initialize microstructure calculator
        if self.config.include_microstructure:
            self.micro_config = self.config.microstructure_config or MicrostructureConfig()
            self.micro_calc = MicrostructureFeatures(self.micro_config)

        # Initialize entropy calculator
        if self.config.include_entropy:
            self.entropy_config = self.config.entropy_config or EntropyConfig()
            self.entropy_calc = EntropyFeatures(self.entropy_config)

        # Initialize regime detector
        if self.config.include_regime:
            self.regime_config = self.config.regime_config or RegimeDetectorConfig()
            self.regime_detector = RegimeDetector(self.regime_config)

    def build(
        self,
        data: pd.DataFrame,
        position: float = 0.0,
        portfolio_value: float = 100000.0,
        initial_capital: float = 100000.0,
    ) -> np.ndarray:
        """
        Build observation vector from market data.

        Args:
            data: DataFrame with OHLCV data (needs at least lookback_period rows)
            position: Current position (-1 to 1)
            portfolio_value: Current portfolio value
            initial_capital: Starting capital

        Returns:
            Observation vector of shape (num_features,)
        """
        features_list = []

        # Extract price data
        closes = data["close"].values.astype(float)
        highs = data["high"].values.astype(float)
        lows = data["low"].values.astype(float)
        volumes = data["volume"].values.astype(float)

        # 1. Base features (45)
        base_features = self.base_builder.build(
            data, position, portfolio_value, initial_capital
        )
        features_list.append(base_features)

        # 2. Microstructure features (8)
        if self.config.include_microstructure:
            micro_features = get_microstructure_features(
                closes, highs, lows, volumes, self.micro_config
            )
            features_list.append(micro_features)

        # 3. Entropy features (4)
        if self.config.include_entropy:
            entropy_features = get_entropy_features(closes, self.entropy_config)
            features_list.append(entropy_features)

        # 4. Regime features (6)
        if self.config.include_regime:
            regime_features = get_regime_features(closes, self.regime_config)
            features_list.append(regime_features)

        # 5. Fractional differentiation features (2, optional)
        if self.config.include_frac_diff:
            frac_features = self._frac_diff_features(closes, volumes)
            features_list.append(frac_features)

        # Combine all features
        observation = np.concatenate(features_list)

        # Ensure correct size
        if len(observation) < self.num_features:
            observation = np.pad(observation, (0, self.num_features - len(observation)))
        elif len(observation) > self.num_features:
            observation = observation[:self.num_features]

        # Handle NaN/Inf and clip
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        observation = np.clip(observation, -10.0, 10.0)

        return observation.astype(np.float32)

    def _frac_diff_features(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate fractionally differentiated features.

        Fractional differentiation preserves memory while ensuring stationarity.
        Uses the fixed-width window FFD method.

        Args:
            closes: Close prices
            volumes: Volumes

        Returns:
            Array of 2 frac diff features
        """
        d = self.config.frac_diff_d

        # FFD weights
        def get_weights(d_val: float, size: int) -> np.ndarray:
            """Get FFD weights."""
            w = [1.0]
            for k in range(1, size):
                w.append(-w[-1] * (d_val - k + 1) / k)
            return np.array(w[::-1])

        window = min(20, len(closes))
        weights = get_weights(d, window)

        # Apply to close prices
        if len(closes) >= window:
            frac_close = np.dot(weights, closes[-window:])
            # Normalize
            frac_close = frac_close / closes[-1] if closes[-1] != 0 else 0
        else:
            frac_close = 0.0

        # Apply to volume
        if len(volumes) >= window:
            frac_vol = np.dot(weights, volumes[-window:])
            avg_vol = np.mean(volumes[-window:])
            frac_vol = frac_vol / avg_vol if avg_vol != 0 else 0
        else:
            frac_vol = 0.0

        return np.array([frac_close, frac_vol], dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Get names of all features in order."""
        names = []

        # Base feature names
        names.extend([
            # Price features (15)
            "return_t1", "return_t2", "return_t3", "return_t4", "return_t5",
            "mean_return", "return_volatility",
            "ma_cross_5_10", "ma_cross_10_20",
            "price_vs_ma5", "price_vs_ma20",
            "bollinger_position", "range_position",
            # Volume features (5)
            "volume_ratio", "volume_trend", "volume_volatility",
            "vwap_distance", "obv_direction",
            # Technical indicators (10)
            "rsi", "macd_signal", "adx", "cci",
            "stoch_k", "stoch_d", "atr_ratio",
            "momentum_5", "momentum_10",
            # Position features (5)
            "position", "is_long", "is_short", "pnl", "position_size",
            # Time features (4)
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            # Portfolio features (6)
            "portfolio_return", "drawdown",
            "win_rate", "profit_factor", "trades_count", "cash_ratio",
        ])

        # Microstructure feature names
        if self.config.include_microstructure:
            names.extend([
                "kyle_lambda", "amihud_illiquidity", "vpin",
                "roll_spread", "cs_spread", "flow_toxicity",
                "volume_clock", "price_efficiency",
            ])

        # Entropy feature names
        if self.config.include_entropy:
            names.extend([
                "shannon_entropy", "approx_entropy",
                "sample_entropy", "lz_complexity",
            ])

        # Regime feature names (R-018)
        if self.config.include_regime:
            names.extend([
                "regime_trend_bull", "regime_trend_bear", "regime_trend_sideways",
                "regime_vol_low", "regime_vol_high", "regime_confidence",
            ])

        # Frac diff feature names
        if self.config.include_frac_diff:
            names.extend(["frac_diff_close", "frac_diff_volume"])

        return names

    def get_feature_dict(
        self,
        data: pd.DataFrame,
        position: float = 0.0,
        portfolio_value: float = 100000.0,
        initial_capital: float = 100000.0,
    ) -> Dict[str, float]:
        """
        Get observation as dictionary with feature names.

        Useful for debugging and feature importance analysis.
        """
        observation = self.build(data, position, portfolio_value, initial_capital)
        names = self.get_feature_names()

        return {name: float(val) for name, val in zip(names, observation)}


def create_institutional_observation_space(
    include_microstructure: bool = True,
    include_entropy: bool = True,
    include_regime: bool = True,
    include_frac_diff: bool = False,
) -> int:
    """
    Calculate the size of the institutional observation space.

    Args:
        include_microstructure: Include microstructure features (8)
        include_entropy: Include entropy features (4)
        include_regime: Include regime features (6)
        include_frac_diff: Include fractional diff features (2)

    Returns:
        Total number of features
    """
    total = 45  # Base features

    if include_microstructure:
        total += 8
    if include_entropy:
        total += 4
    if include_regime:
        total += 6
    if include_frac_diff:
        total += 2

    return total
