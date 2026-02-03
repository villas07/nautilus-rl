"""
Enhanced Observation Features

Addresses D-024: Basic features show ~50% accuracy (random).
This module adds more predictive features:

1. Multi-timeframe features (5d, 10d, 20d momentum)
2. Volatility regime detection
3. Trend strength indicators
4. Mean reversion signals
5. Volume profile features

Usage:
    from gym_env.enhanced_observation import EnhancedObservationBuilder

    builder = EnhancedObservationBuilder(lookback=60)
    obs = builder.build(bars_df)
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

import structlog

logger = structlog.get_logger()


class EnhancedObservationBuilder:
    """
    Builds enhanced observation vectors with more predictive features.

    Total features: 60 (vs original 45)
    """

    FEATURE_COUNT = 60

    def __init__(self, lookback: int = 60):
        """
        Initialize builder.

        Args:
            lookback: Number of bars to use for feature calculation.
        """
        self.lookback = lookback

    def build(
        self,
        df: pd.DataFrame,
        position: float = 0.0,
        unrealized_pnl: float = 0.0,
    ) -> np.ndarray:
        """
        Build enhanced observation from bar data.

        Args:
            df: DataFrame with OHLCV data (needs 'open', 'high', 'low', 'close', 'volume')
            position: Current position (-1 to 1)
            unrealized_pnl: Current unrealized PnL

        Returns:
            Numpy array of shape (60,)
        """
        features = []

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        volumes = df["volume"].values

        current_close = closes[-1]

        # =====================================================================
        # 1. RETURN FEATURES (6 features)
        # =====================================================================
        returns = np.diff(closes) / (closes[:-1] + 1e-8)

        features.extend([
            returns[-1] if len(returns) > 0 else 0,      # Last return
            np.mean(returns[-5:]) if len(returns) >= 5 else 0,   # 5d mean return
            np.mean(returns[-20:]) if len(returns) >= 20 else 0, # 20d mean return
            np.std(returns[-5:]) if len(returns) >= 5 else 0,    # 5d volatility
            np.std(returns[-20:]) if len(returns) >= 20 else 0,  # 20d volatility
            np.std(returns[-60:]) if len(returns) >= 60 else np.std(returns),  # 60d volatility
        ])

        # =====================================================================
        # 2. MOMENTUM FEATURES (6 features)
        # =====================================================================
        def safe_momentum(arr, period):
            if len(arr) >= period:
                return (arr[-1] - arr[-period]) / (arr[-period] + 1e-8)
            return 0

        features.extend([
            safe_momentum(closes, 5),    # 5-day momentum
            safe_momentum(closes, 10),   # 10-day momentum
            safe_momentum(closes, 20),   # 20-day momentum
            safe_momentum(closes, 40),   # 40-day momentum
            safe_momentum(closes, 60),   # 60-day momentum (if available)
            # Rate of change of momentum (acceleration)
            safe_momentum(closes, 5) - safe_momentum(closes[:-5] if len(closes) > 5 else closes, 5),
        ])

        # =====================================================================
        # 3. MOVING AVERAGE FEATURES (8 features)
        # =====================================================================
        ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else current_close
        ma10 = np.mean(closes[-10:]) if len(closes) >= 10 else current_close
        ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else current_close
        ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else current_close

        features.extend([
            (current_close - ma5) / (ma5 + 1e-8),
            (current_close - ma10) / (ma10 + 1e-8),
            (current_close - ma20) / (ma20 + 1e-8),
            (current_close - ma50) / (ma50 + 1e-8),
            (ma5 - ma20) / (ma20 + 1e-8),   # Short vs medium trend
            (ma10 - ma50) / (ma50 + 1e-8),  # Medium vs long trend
            (ma5 - ma10) / (ma10 + 1e-8),   # Very short trend
            (ma20 - ma50) / (ma50 + 1e-8),  # Medium-long trend
        ])

        # =====================================================================
        # 4. VOLATILITY REGIME FEATURES (6 features)
        # =====================================================================
        vol_5 = np.std(returns[-5:]) if len(returns) >= 5 else 0.01
        vol_20 = np.std(returns[-20:]) if len(returns) >= 20 else 0.01
        vol_60 = np.std(returns[-60:]) if len(returns) >= 60 else vol_20

        # Volatility ratio (regime detection)
        vol_ratio = vol_5 / (vol_20 + 1e-8)

        # Historical volatility percentile (0-1)
        if len(returns) >= 60:
            rolling_vol = pd.Series(returns).rolling(20).std().dropna().values
            current_vol_percentile = (rolling_vol < vol_20).sum() / len(rolling_vol)
        else:
            current_vol_percentile = 0.5

        # ATR-based features
        if len(highs) >= 20:
            hl = highs[-20:] - lows[-20:]
            hc = np.abs(highs[-19:] - closes[-20:-1])
            lc = np.abs(lows[-19:] - closes[-20:-1])
            tr = np.maximum(hl[1:], np.maximum(hc, lc))
        else:
            tr = highs - lows
        atr = np.mean(tr) if len(tr) > 0 else 1.0

        features.extend([
            vol_ratio,                          # Volatility regime
            np.clip(vol_ratio - 1, -2, 2),      # Volatility expansion/contraction
            current_vol_percentile,             # Vol percentile
            atr / (current_close + 1e-8),       # Normalized ATR
            (vol_5 - vol_60) / (vol_60 + 1e-8), # Vol trend
            1 if vol_ratio > 1.5 else 0,        # High vol regime flag
        ])

        # =====================================================================
        # 5. MEAN REVERSION FEATURES (6 features)
        # =====================================================================
        # RSI
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0.01
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0.01
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_std = np.std(closes[-20:]) if len(closes) >= 20 else 1.0
        bb_upper = ma20 + 2 * bb_std
        bb_lower = ma20 - 2 * bb_std
        bb_width = bb_upper - bb_lower
        bb_position = (current_close - bb_lower) / (bb_width + 1e-8)

        # Z-score
        z_score = (current_close - ma20) / (bb_std + 1e-8)

        features.extend([
            (rsi - 50) / 50,                    # Normalized RSI
            1 if rsi < 30 else (-1 if rsi > 70 else 0),  # RSI signal
            bb_position,                         # Position in BB
            np.clip(z_score, -3, 3) / 3,        # Normalized z-score
            bb_width / (ma20 + 1e-8),           # BB width (volatility)
            1 if bb_position < 0.2 else (-1 if bb_position > 0.8 else 0),  # BB signal
        ])

        # =====================================================================
        # 6. TREND STRENGTH FEATURES (6 features)
        # =====================================================================
        # ADX proxy (simplified)
        high_low = highs - lows
        high_close = np.abs(highs - np.roll(closes, 1))
        low_close = np.abs(lows - np.roll(closes, 1))
        tr_full = np.maximum(high_low, np.maximum(high_close, low_close))[1:]

        if len(tr_full) >= 14:
            # Directional movement
            up_move = highs[1:] - highs[:-1]
            down_move = lows[:-1] - lows[1:]

            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

            atr14 = np.mean(tr_full[-14:])
            plus_di = 100 * np.mean(plus_dm[-14:]) / (atr14 + 1e-8)
            minus_di = 100 * np.mean(minus_dm[-14:]) / (atr14 + 1e-8)

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
            adx = dx  # Simplified
        else:
            plus_di = 50
            minus_di = 50
            adx = 25

        # Linear regression slope
        if len(closes) >= 20:
            x = np.arange(20)
            slope = np.polyfit(x, closes[-20:], 1)[0]
            slope_normalized = slope / (np.std(closes[-20:]) + 1e-8)
        else:
            slope_normalized = 0

        features.extend([
            adx / 100,                          # ADX normalized
            (plus_di - minus_di) / 100,         # DI difference
            1 if adx > 25 else 0,               # Strong trend flag
            np.clip(slope_normalized, -2, 2),   # Trend slope
            # Trend consistency (% of up days in last 20)
            (returns[-20:] > 0).sum() / 20 if len(returns) >= 20 else 0.5,
            # Higher highs / lower lows
            (highs[-5:] > highs[-10:-5].max()).sum() / 5 if len(highs) >= 10 else 0.5,
        ])

        # =====================================================================
        # 7. VOLUME FEATURES (6 features)
        # =====================================================================
        avg_vol_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else 1
        avg_vol_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else 1

        # Volume trend
        vol_ma5 = avg_vol_5
        vol_ma20 = avg_vol_20

        # On-balance volume proxy
        obv_direction = np.sign(returns) if len(returns) > 0 else np.array([0])

        features.extend([
            volumes[-1] / (avg_vol_20 + 1e-8),  # Relative volume
            vol_ma5 / (vol_ma20 + 1e-8),        # Volume trend
            np.std(volumes[-20:]) / (avg_vol_20 + 1e-8) if len(volumes) >= 20 else 0,  # Vol of vol
            # Price-volume correlation (last 20 bars)
            np.corrcoef(returns[-20:], volumes[-21:-1])[0, 1] if len(returns) >= 20 else 0,
            # Volume on up vs down days
            np.mean(volumes[-20:][returns[-20:] > 0]) / (avg_vol_20 + 1e-8) if len(returns) >= 20 and (returns[-20:] > 0).any() else 1,
            1 if volumes[-1] > 2 * avg_vol_20 else 0,  # Volume spike flag
        ])

        # =====================================================================
        # 8. RANGE FEATURES (4 features)
        # =====================================================================
        high_20 = np.max(highs[-20:]) if len(highs) >= 20 else highs[-1]
        low_20 = np.min(lows[-20:]) if len(lows) >= 20 else lows[-1]
        range_20 = high_20 - low_20

        features.extend([
            (current_close - low_20) / (range_20 + 1e-8),   # Position in range
            range_20 / (current_close + 1e-8),              # Range as % of price
            (highs[-1] - lows[-1]) / (atr + 1e-8),          # Today's range vs ATR
            (current_close - lows[-1]) / (highs[-1] - lows[-1] + 1e-8),  # Close position in day's range
        ])

        # =====================================================================
        # 9. POSITION FEATURES (4 features)
        # =====================================================================
        features.extend([
            position,                           # Current position
            1 if position > 0 else 0,          # Is long
            1 if position < 0 else 0,          # Is short
            unrealized_pnl,                    # Unrealized PnL (normalized)
        ])

        # =====================================================================
        # 10. TIME FEATURES (4 features) - Placeholders for live trading
        # =====================================================================
        features.extend([
            0,  # Hour of day (sin)
            0,  # Hour of day (cos)
            0,  # Day of week (sin)
            0,  # Day of week (cos)
        ])

        # =====================================================================
        # 11. MARKET REGIME (4 features)
        # =====================================================================
        # Combine multiple signals for regime detection
        trend_score = (
            (1 if ma5 > ma20 else -1) +
            (1 if ma10 > ma50 else -1) +
            (1 if slope_normalized > 0 else -1)
        ) / 3

        mean_rev_score = (
            (1 if rsi < 30 else (-1 if rsi > 70 else 0)) +
            (1 if bb_position < 0.2 else (-1 if bb_position > 0.8 else 0)) +
            (1 if z_score < -2 else (-1 if z_score > 2 else 0))
        ) / 3

        features.extend([
            trend_score,                        # Overall trend score
            mean_rev_score,                     # Mean reversion score
            vol_ratio,                          # Volatility regime
            adx / 100,                          # Trend strength
        ])

        # Ensure correct length
        features = np.array(features, dtype=np.float32)

        if len(features) < self.FEATURE_COUNT:
            features = np.pad(features, (0, self.FEATURE_COUNT - len(features)))
        elif len(features) > self.FEATURE_COUNT:
            features = features[:self.FEATURE_COUNT]

        # Replace NaN/Inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features

    @staticmethod
    def get_feature_names() -> list:
        """Get names of all features for interpretation."""
        return [
            # Returns (6)
            "return_1d", "return_5d_mean", "return_20d_mean",
            "vol_5d", "vol_20d", "vol_60d",
            # Momentum (6)
            "mom_5d", "mom_10d", "mom_20d", "mom_40d", "mom_60d", "mom_accel",
            # Moving Averages (8)
            "ma5_diff", "ma10_diff", "ma20_diff", "ma50_diff",
            "ma5_ma20", "ma10_ma50", "ma5_ma10", "ma20_ma50",
            # Volatility Regime (6)
            "vol_ratio", "vol_expansion", "vol_percentile",
            "atr_norm", "vol_trend", "high_vol_flag",
            # Mean Reversion (6)
            "rsi_norm", "rsi_signal", "bb_position",
            "z_score", "bb_width", "bb_signal",
            # Trend Strength (6)
            "adx", "di_diff", "strong_trend_flag",
            "slope", "trend_consistency", "higher_highs",
            # Volume (6)
            "rel_volume", "vol_trend", "vol_of_vol",
            "price_vol_corr", "vol_up_days", "vol_spike",
            # Range (4)
            "range_position", "range_pct", "day_range_atr", "close_position",
            # Position (4)
            "position", "is_long", "is_short", "unrealized_pnl",
            # Time (4)
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            # Regime (4)
            "trend_score", "mean_rev_score", "vol_regime", "trend_strength",
        ]


def test_enhanced_features():
    """Test the enhanced feature builder."""
    # Create sample data
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        "open": 100 + np.cumsum(np.random.randn(n) * 0.5),
        "high": 100 + np.cumsum(np.random.randn(n) * 0.5) + abs(np.random.randn(n)),
        "low": 100 + np.cumsum(np.random.randn(n) * 0.5) - abs(np.random.randn(n)),
        "close": 100 + np.cumsum(np.random.randn(n) * 0.5),
        "volume": 1000000 + np.random.randn(n) * 100000,
    })

    builder = EnhancedObservationBuilder()
    obs = builder.build(df)

    print(f"Observation shape: {obs.shape}")
    print(f"Feature count: {len(obs)}")
    print(f"Any NaN: {np.isnan(obs).any()}")
    print(f"Any Inf: {np.isinf(obs).any()}")
    print(f"\nFeature names: {len(builder.get_feature_names())}")

    # Print some key features
    names = builder.get_feature_names()
    print("\nKey features:")
    for i, (name, val) in enumerate(zip(names[:10], obs[:10])):
        print(f"  {name}: {val:.4f}")


if __name__ == "__main__":
    test_enhanced_features()
