"""
Regime Detection System for Institutional ML.

Based on EVAL-003 evaluation of sistema_regimen_agentes.md

Detects market regimes using ensemble of methods:
1. Rule-based (SMA crossovers, volatility ratios)
2. HMM (Hidden Markov Model) - optional
3. Momentum-based (multi-timeframe)

Reference: Marcos López de Prado - Advances in Financial Machine Learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

# HMM is optional
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn("hmmlearn not installed. HMM detector disabled. Install with: pip install hmmlearn")


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"


@dataclass
class RegimeState:
    """Current regime state with metadata."""
    regime: MarketRegime
    confidence: float
    trend: str  # 'bull', 'bear', 'sideways'
    volatility: str  # 'low', 'high'
    details: Dict = field(default_factory=dict)
    timestamp: Optional[pd.Timestamp] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'regime': self.regime.value,
            'confidence': self.confidence,
            'trend': self.trend,
            'volatility': self.volatility,
            'details': self.details,
            'timestamp': str(self.timestamp) if self.timestamp else None,
        }

    def to_features(self) -> np.ndarray:
        """
        Convert regime state to feature vector for RL observation.

        Returns 6 features:
        - trend_bull, trend_bear, trend_sideways (one-hot)
        - vol_low, vol_high (one-hot)
        - confidence
        """
        trend_encoding = [0.0, 0.0, 0.0]
        if self.trend == 'bull':
            trend_encoding[0] = 1.0
        elif self.trend == 'bear':
            trend_encoding[1] = 1.0
        else:
            trend_encoding[2] = 1.0

        vol_encoding = [0.0, 0.0]
        if self.volatility == 'low':
            vol_encoding[0] = 1.0
        else:
            vol_encoding[1] = 1.0

        return np.array(trend_encoding + vol_encoding + [self.confidence], dtype=np.float32)


@dataclass
class RegimeDetectorConfig:
    """Configuration for regime detection."""
    # Trend detection
    sma_short: int = 50
    sma_long: int = 200
    trend_threshold: float = 0.02  # 2% above/below for trend

    # Volatility detection
    vol_short: int = 20
    vol_long: int = 60
    vol_threshold: float = 1.5  # Ratio for high vol

    # HMM settings
    hmm_n_regimes: int = 3
    hmm_lookback: int = 252

    # Smoothing
    regime_min_duration: int = 5  # Minimum bars in a regime before switching

    # Ensemble weights
    weight_rule_based: float = 0.5
    weight_hmm: float = 0.3
    weight_momentum: float = 0.2


class RegimeDetector:
    """
    Market regime detector using ensemble of methods.

    Methods:
    1. Rule-based: SMA crossovers and volatility ratios
    2. HMM: Hidden Markov Model (if hmmlearn installed)
    3. Momentum: Multi-timeframe momentum analysis

    Usage:
        detector = RegimeDetector()
        regime = detector.detect(df)  # df with 'close' column
        print(f"Current regime: {regime.regime.value}")
    """

    def __init__(self, config: Optional[RegimeDetectorConfig] = None):
        """Initialize regime detector."""
        self.config = config or RegimeDetectorConfig()
        self.hmm_model = None
        self.hmm_state_order = None
        self.hmm_state_characteristics = {}
        self.history: List[RegimeState] = []

    # =========================================================================
    # METHOD 1: RULE-BASED (Simple and robust)
    # =========================================================================

    def detect_rule_based(self, df: pd.DataFrame) -> RegimeState:
        """
        Detect regime using simple rules.

        More robust, less prone to overfitting.

        Args:
            df: DataFrame with 'close' column (and index as datetime)

        Returns:
            RegimeState with detected regime
        """
        close = df['close'].values if isinstance(df['close'], pd.Series) else df['close']

        # Ensure we have enough data
        min_required = max(self.config.sma_long, self.config.vol_long) + 1
        if len(close) < min_required:
            return RegimeState(
                regime=MarketRegime.SIDEWAYS_LOW_VOL,
                confidence=0.0,
                trend='sideways',
                volatility='low',
                details={'error': 'insufficient_data'},
                timestamp=df.index[-1] if hasattr(df, 'index') else None,
            )

        # ─────────────────────────────────────────────────────────────────────
        # 1. TREND DETECTION
        # ─────────────────────────────────────────────────────────────────────
        sma_short = self._sma(close, self.config.sma_short)
        sma_long = self._sma(close, self.config.sma_long)

        current_price = close[-1]
        current_sma_short = sma_short[-1]
        current_sma_long = sma_long[-1]

        # Trend score: -1 (bear) to +1 (bull)
        price_vs_sma_short = (current_price / current_sma_short - 1) if current_sma_short > 0 else 0
        price_vs_sma_long = (current_price / current_sma_long - 1) if current_sma_long > 0 else 0
        sma_cross = (current_sma_short / current_sma_long - 1) if current_sma_long > 0 else 0

        trend_score = (price_vs_sma_short + price_vs_sma_long + sma_cross) / 3

        if trend_score > self.config.trend_threshold:
            trend = 'bull'
        elif trend_score < -self.config.trend_threshold:
            trend = 'bear'
        else:
            trend = 'sideways'

        # ─────────────────────────────────────────────────────────────────────
        # 2. VOLATILITY DETECTION
        # ─────────────────────────────────────────────────────────────────────
        returns = np.diff(close) / close[:-1]

        vol_short = np.std(returns[-self.config.vol_short:]) if len(returns) >= self.config.vol_short else np.std(returns)
        vol_long = np.std(returns[-self.config.vol_long:]) if len(returns) >= self.config.vol_long else np.std(returns)

        vol_ratio = vol_short / vol_long if vol_long > 0 else 1.0

        volatility = 'high' if vol_ratio > self.config.vol_threshold else 'low'

        # ─────────────────────────────────────────────────────────────────────
        # 3. COMBINE
        # ─────────────────────────────────────────────────────────────────────
        regime = MarketRegime(f"{trend}_{volatility}_vol")

        # Confidence based on signal clarity
        trend_confidence = min(abs(trend_score) / 0.05, 1.0)
        vol_confidence = min(abs(vol_ratio - 1) / 0.5, 1.0)
        confidence = (trend_confidence + vol_confidence) / 2

        return RegimeState(
            regime=regime,
            confidence=confidence,
            trend=trend,
            volatility=volatility,
            details={
                'method': 'rule_based',
                'trend_score': float(trend_score),
                'vol_ratio': float(vol_ratio),
                'price_vs_sma_short': float(price_vs_sma_short),
                'price_vs_sma_long': float(price_vs_sma_long),
                'sma_cross': float(sma_cross),
                'vol_short_ann': float(vol_short * np.sqrt(252)),
                'vol_long_ann': float(vol_long * np.sqrt(252)),
            },
            timestamp=df.index[-1] if hasattr(df, 'index') and len(df.index) > 0 else None,
        )

    # =========================================================================
    # METHOD 2: HIDDEN MARKOV MODEL
    # =========================================================================

    def fit_hmm(self, returns: np.ndarray) -> None:
        """
        Train Hidden Markov Model for regime detection.

        Args:
            returns: Array of returns (not prices)
        """
        if not HMM_AVAILABLE:
            warnings.warn("HMM not available. Skipping.")
            return

        # Prepare data
        returns = np.array(returns).flatten()
        returns = returns[~np.isnan(returns)]
        X = returns.reshape(-1, 1)

        if len(X) < 100:
            warnings.warn("Not enough data to train HMM")
            return

        # Train HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.config.hmm_n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42,
        )

        try:
            self.hmm_model.fit(X)
        except Exception as e:
            warnings.warn(f"HMM training failed: {e}")
            self.hmm_model = None
            return

        # Order states by volatility (for interpretability)
        state_vols = [np.sqrt(self.hmm_model.covars_[i][0, 0])
                      for i in range(self.config.hmm_n_regimes)]
        self.hmm_state_order = np.argsort(state_vols)

        # Store characteristics of each state
        self.hmm_state_characteristics = {}
        for i, state_idx in enumerate(self.hmm_state_order):
            mean_return = self.hmm_model.means_[state_idx][0]
            volatility = state_vols[state_idx]

            self.hmm_state_characteristics[i] = {
                'mean_return': float(mean_return * 252),  # Annualized
                'volatility': float(volatility * np.sqrt(252)),  # Annualized
                'original_idx': int(state_idx),
            }

    def detect_hmm(self, returns: np.ndarray) -> RegimeState:
        """
        Detect regime using HMM.

        Args:
            returns: Array of recent returns

        Returns:
            RegimeState
        """
        if self.hmm_model is None:
            raise ValueError("HMM not fitted. Call fit_hmm first.")

        # Use last N returns
        returns = np.array(returns).flatten()
        returns = returns[~np.isnan(returns)]
        lookback = min(self.config.hmm_lookback, len(returns))
        X = returns[-lookback:].reshape(-1, 1)

        # Predict most likely state
        states = self.hmm_model.predict(X)
        current_state = states[-1]

        # Get probabilities
        state_probs = self.hmm_model.predict_proba(X)[-1]

        # Map to interpretable regime
        ordered_state = np.where(self.hmm_state_order == current_state)[0][0]

        # Characteristics of current state
        char = self.hmm_state_characteristics[ordered_state]

        # Determine trend and volatility
        if char['mean_return'] > 5:  # > 5% annual
            trend = 'bull'
        elif char['mean_return'] < -5:
            trend = 'bear'
        else:
            trend = 'sideways'

        volatility = 'high' if char['volatility'] > 20 else 'low'  # > 20% annual vol

        regime = MarketRegime(f"{trend}_{volatility}_vol")

        return RegimeState(
            regime=regime,
            confidence=float(state_probs[current_state]),
            trend=trend,
            volatility=volatility,
            details={
                'method': 'hmm',
                'hmm_state': int(ordered_state),
                'state_probs': state_probs.tolist(),
                'expected_return': char['mean_return'],
                'expected_vol': char['volatility'],
            },
            timestamp=None,
        )

    # =========================================================================
    # METHOD 3: MOMENTUM-BASED
    # =========================================================================

    def detect_momentum(self, df: pd.DataFrame) -> RegimeState:
        """
        Detect regime using multi-timeframe momentum.

        Args:
            df: DataFrame with 'close' column

        Returns:
            RegimeState
        """
        close = df['close'].values if isinstance(df['close'], pd.Series) else df['close']

        if len(close) < 60:
            return RegimeState(
                regime=MarketRegime.SIDEWAYS_LOW_VOL,
                confidence=0.0,
                trend='sideways',
                volatility='low',
                details={'error': 'insufficient_data'},
            )

        # Momentum at different horizons
        mom_5 = (close[-1] / close[-5] - 1) if len(close) >= 5 else 0
        mom_20 = (close[-1] / close[-20] - 1) if len(close) >= 20 else 0
        mom_60 = (close[-1] / close[-60] - 1) if len(close) >= 60 else 0

        # Momentum score (-1 to +1)
        mom_score = (
            0.5 * np.sign(mom_5) * min(abs(mom_5) / 0.02, 1) +
            0.3 * np.sign(mom_20) * min(abs(mom_20) / 0.05, 1) +
            0.2 * np.sign(mom_60) * min(abs(mom_60) / 0.10, 1)
        )

        # Volatility
        returns = np.diff(close) / close[:-1]
        vol = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.15

        # Volatility percentile (simplified)
        vol_history = pd.Series(returns).rolling(20).std().dropna() * np.sqrt(252)
        if len(vol_history) > 0:
            vol_percentile = (vol_history < vol).mean()
        else:
            vol_percentile = 0.5

        # Determine trend
        if mom_score > 0.3:
            trend = 'bull'
        elif mom_score < -0.3:
            trend = 'bear'
        else:
            trend = 'sideways'

        volatility = 'high' if vol_percentile > 0.7 else 'low'

        regime = MarketRegime(f"{trend}_{volatility}_vol")

        return RegimeState(
            regime=regime,
            confidence=float(abs(mom_score)),
            trend=trend,
            volatility=volatility,
            details={
                'method': 'momentum',
                'mom_5d': float(mom_5),
                'mom_20d': float(mom_20),
                'mom_60d': float(mom_60),
                'mom_score': float(mom_score),
                'volatility_ann': float(vol),
                'vol_percentile': float(vol_percentile),
            },
            timestamp=df.index[-1] if hasattr(df, 'index') and len(df.index) > 0 else None,
        )

    # =========================================================================
    # ENSEMBLE: COMBINE METHODS
    # =========================================================================

    def detect(self, df: pd.DataFrame, use_hmm: bool = True) -> RegimeState:
        """
        Detect regime using ensemble of methods.

        Args:
            df: DataFrame with 'close' column (minimum 200+ rows recommended)
            use_hmm: Whether to use HMM if available and fitted

        Returns:
            RegimeState with ensemble result
        """
        results = {}

        # 1. Rule-based (always)
        results['rule_based'] = self.detect_rule_based(df)

        # 2. HMM (if available and fitted)
        if use_hmm and self.hmm_model is not None:
            try:
                close = df['close'].values if isinstance(df['close'], pd.Series) else df['close']
                returns = np.diff(close) / close[:-1]
                results['hmm'] = self.detect_hmm(returns)
            except Exception as e:
                warnings.warn(f"HMM detection failed: {e}")

        # 3. Momentum
        results['momentum'] = self.detect_momentum(df)

        # ─────────────────────────────────────────────────────────────────────
        # VOTING: Combine results
        # ─────────────────────────────────────────────────────────────────────
        trend_votes = {'bull': 0.0, 'bear': 0.0, 'sideways': 0.0}
        vol_votes = {'high': 0.0, 'low': 0.0}

        weights = {
            'rule_based': self.config.weight_rule_based,
            'hmm': self.config.weight_hmm,
            'momentum': self.config.weight_momentum,
        }

        for method, state in results.items():
            weight = weights.get(method, 0.1)
            trend_votes[state.trend] += weight * state.confidence
            vol_votes[state.volatility] += weight * state.confidence

        # Winners
        final_trend = max(trend_votes, key=trend_votes.get)
        final_vol = max(vol_votes, key=vol_votes.get)

        # Final confidence
        total_weight = sum(weights.get(m, 0.1) for m in results.keys())
        trend_confidence = trend_votes[final_trend] / total_weight if total_weight > 0 else 0
        vol_confidence = vol_votes[final_vol] / total_weight if total_weight > 0 else 0
        final_confidence = (trend_confidence + vol_confidence) / 2

        final_regime = MarketRegime(f"{final_trend}_{final_vol}_vol")

        # ─────────────────────────────────────────────────────────────────────
        # SMOOTHING: Avoid frequent regime changes
        # ─────────────────────────────────────────────────────────────────────
        if len(self.history) > 0:
            last_regime = self.history[-1].regime
            recent_history = self.history[-self.config.regime_min_duration:]
            regime_duration = sum(1 for h in recent_history if h.regime == last_regime)

            # If current regime is very recent, maintain it
            if regime_duration < self.config.regime_min_duration and final_regime != last_regime:
                if final_confidence < 0.7:  # Only change if very confident
                    final_regime = last_regime
                    final_confidence *= 0.8

        # Create final state
        final_state = RegimeState(
            regime=final_regime,
            confidence=final_confidence,
            trend=final_trend,
            volatility=final_vol,
            details={
                'method': 'ensemble',
                'method_results': {m: s.to_dict() for m, s in results.items()},
                'trend_votes': trend_votes,
                'vol_votes': vol_votes,
            },
            timestamp=df.index[-1] if hasattr(df, 'index') and len(df.index) > 0 else None,
        )

        # Save to history
        self.history.append(final_state)

        return final_state

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate simple moving average."""
        if len(data) < period:
            return np.full(len(data), np.mean(data))

        sma = np.convolve(data, np.ones(period) / period, mode='valid')
        # Pad beginning
        padding = np.full(period - 1, sma[0])
        return np.concatenate([padding, sma])

    def get_regime_history(self, n: Optional[int] = None) -> pd.DataFrame:
        """Get regime history as DataFrame."""
        history = self.history[-n:] if n else self.history

        if not history:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'timestamp': h.timestamp,
                'regime': h.regime.value,
                'trend': h.trend,
                'volatility': h.volatility,
                'confidence': h.confidence,
            }
            for h in history
        ])

    def get_regime_stats(self) -> Dict:
        """Get regime statistics."""
        if not self.history:
            return {}

        df = self.get_regime_history()

        return {
            'regime_counts': df['regime'].value_counts().to_dict(),
            'avg_confidence': float(df['confidence'].mean()),
            'regime_changes': int((df['regime'] != df['regime'].shift()).sum()),
            'current_regime': self.history[-1].regime.value,
            'current_duration': sum(
                1 for h in reversed(self.history)
                if h.regime == self.history[-1].regime
            ),
        }

    def reset_history(self):
        """Clear regime history."""
        self.history = []


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def detect_regime(
    closes: np.ndarray,
    config: Optional[RegimeDetectorConfig] = None,
) -> RegimeState:
    """
    Quick regime detection from close prices.

    Args:
        closes: Array of close prices
        config: Optional configuration

    Returns:
        RegimeState
    """
    detector = RegimeDetector(config)
    df = pd.DataFrame({'close': closes})
    return detector.detect(df, use_hmm=False)


def get_regime_features(
    closes: np.ndarray,
    config: Optional[RegimeDetectorConfig] = None,
) -> np.ndarray:
    """
    Get regime as feature vector for RL observation.

    Args:
        closes: Array of close prices
        config: Optional configuration

    Returns:
        Array of 6 regime features
    """
    state = detect_regime(closes, config)
    return state.to_features()
