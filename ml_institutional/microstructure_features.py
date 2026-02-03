"""
Microstructure Features for Institutional ML.

Based on:
- Kyle (1985) - Market Impact
- Amihud (2002) - Illiquidity Measure
- Easley et al. (2012) - VPIN
- Roll (1984) - Bid-Ask Spread Estimation

Reference: Marcos López de Prado - Advances in Financial Machine Learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MicrostructureConfig:
    """Configuration for microstructure feature calculation."""
    kyle_window: int = 20          # Window for Kyle Lambda
    amihud_window: int = 20        # Window for Amihud illiquidity
    vpin_bucket_size: int = 50     # Volume bucket size for VPIN
    vpin_num_buckets: int = 20     # Number of buckets for VPIN
    roll_window: int = 20          # Window for Roll spread
    corwin_schultz_window: int = 20  # Window for Corwin-Schultz spread


class MicrostructureFeatures:
    """
    Calculate microstructure features from OHLCV data.

    Features calculated:
    1. Kyle Lambda - Market impact coefficient
    2. Amihud Illiquidity - Price impact per dollar volume
    3. VPIN - Volume-Synchronized Probability of Informed Trading
    4. Roll Spread - Bid-ask spread estimator (autocovariance method)
    5. Corwin-Schultz Spread - High-low spread estimator
    6. Flow Toxicity - Order flow imbalance
    7. Volume Clock - Volume-weighted time
    8. Price Efficiency - Variance ratio
    """

    def __init__(self, config: Optional[MicrostructureConfig] = None):
        """Initialize with configuration."""
        self.config = config or MicrostructureConfig()

    def calculate_all(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate all microstructure features.

        Args:
            closes: Array of close prices
            highs: Array of high prices
            lows: Array of low prices
            volumes: Array of volumes

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        # Kyle Lambda
        features['kyle_lambda'] = self.kyle_lambda(closes, volumes)

        # Amihud Illiquidity
        features['amihud_illiquidity'] = self.amihud_illiquidity(closes, volumes)

        # VPIN
        features['vpin'] = self.vpin(closes, volumes)

        # Roll Spread
        features['roll_spread'] = self.roll_spread(closes)

        # Corwin-Schultz Spread
        features['cs_spread'] = self.corwin_schultz_spread(highs, lows)

        # Flow Toxicity
        features['flow_toxicity'] = self.flow_toxicity(closes, volumes)

        # Volume Clock
        features['volume_clock'] = self.volume_clock(volumes)

        # Price Efficiency (Variance Ratio)
        features['price_efficiency'] = self.variance_ratio(closes)

        return features

    def kyle_lambda(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> float:
        """
        Calculate Kyle's Lambda (market impact coefficient).

        Lambda measures the price impact per unit of signed order flow.
        Higher lambda = less liquid market.

        Kyle (1985): ΔP = λ * (signed_volume)

        Args:
            closes: Close prices
            volumes: Volumes

        Returns:
            Kyle lambda coefficient (normalized)
        """
        window = self.config.kyle_window
        if len(closes) < window + 1:
            return 0.0

        # Price changes
        price_changes = np.diff(closes[-window-1:])

        # Signed volume (use price direction as proxy for order sign)
        signs = np.sign(price_changes)
        signed_volumes = signs * volumes[-window:]

        # Avoid division by zero
        if np.std(signed_volumes) < 1e-10:
            return 0.0

        # Regression: price_change = lambda * signed_volume
        # lambda = cov(dp, sv) / var(sv)
        cov = np.cov(price_changes, signed_volumes)[0, 1]
        var = np.var(signed_volumes)

        if var < 1e-10:
            return 0.0

        kyle_lambda = cov / var

        # Normalize by average price for comparability
        avg_price = np.mean(closes[-window:])
        if avg_price > 0:
            kyle_lambda = kyle_lambda * avg_price * 1000  # Scale to reasonable range

        return np.clip(kyle_lambda, -10, 10)

    def amihud_illiquidity(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> float:
        """
        Calculate Amihud Illiquidity Measure.

        Amihud (2002): ILLIQ = |return| / dollar_volume

        Higher values indicate less liquid (more price impact per dollar).

        Args:
            closes: Close prices
            volumes: Volumes

        Returns:
            Amihud illiquidity measure (log-transformed)
        """
        window = self.config.amihud_window
        if len(closes) < window + 1:
            return 0.0

        # Absolute returns
        returns = np.abs(np.diff(closes[-window-1:])) / closes[-window-1:-1]

        # Dollar volume
        dollar_volumes = closes[-window:] * volumes[-window:]

        # Avoid division by zero
        valid_mask = dollar_volumes > 1e-10
        if not np.any(valid_mask):
            return 0.0

        # Average ratio
        ratios = returns[valid_mask] / dollar_volumes[valid_mask]
        amihud = np.mean(ratios)

        # Log transform for better distribution
        if amihud > 0:
            amihud = np.log1p(amihud * 1e6)  # Scale up before log

        return np.clip(amihud, 0, 10)

    def vpin(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> float:
        """
        Calculate Volume-Synchronized Probability of Informed Trading (VPIN).

        VPIN = |buy_volume - sell_volume| / total_volume

        Easley, López de Prado, O'Hara (2012)

        Args:
            closes: Close prices
            volumes: Volumes

        Returns:
            VPIN estimate (0 to 1)
        """
        bucket_size = self.config.vpin_bucket_size
        num_buckets = self.config.vpin_num_buckets

        min_length = bucket_size * num_buckets
        if len(closes) < min_length + 1:
            # Fallback: use all available data
            if len(closes) < 10:
                return 0.5
            bucket_size = max(1, len(closes) // 10)
            num_buckets = min(10, len(closes) // bucket_size)

        # Classify trades as buy/sell based on tick rule
        price_changes = np.diff(closes)

        # Assign volumes to buy/sell based on price direction
        buy_volumes = np.zeros(len(volumes))
        sell_volumes = np.zeros(len(volumes))

        for i in range(1, len(volumes)):
            if price_changes[i-1] > 0:
                buy_volumes[i] = volumes[i]
            elif price_changes[i-1] < 0:
                sell_volumes[i] = volumes[i]
            else:
                # No change: split equally
                buy_volumes[i] = volumes[i] / 2
                sell_volumes[i] = volumes[i] / 2

        # Calculate VPIN over buckets
        total_imbalance = 0
        total_volume = 0

        idx = len(volumes) - 1
        for _ in range(num_buckets):
            bucket_buy = 0
            bucket_sell = 0
            bucket_vol = 0

            while bucket_vol < bucket_size and idx >= 0:
                bucket_buy += buy_volumes[idx]
                bucket_sell += sell_volumes[idx]
                bucket_vol += volumes[idx]
                idx -= 1

            if bucket_vol > 0:
                total_imbalance += abs(bucket_buy - bucket_sell)
                total_volume += bucket_vol

        if total_volume < 1e-10:
            return 0.5

        vpin = total_imbalance / total_volume
        return np.clip(vpin, 0, 1)

    def roll_spread(self, closes: np.ndarray) -> float:
        """
        Estimate bid-ask spread using Roll (1984) model.

        Roll shows that spread can be estimated from serial covariance
        of price changes: spread = 2 * sqrt(-cov(Δp_t, Δp_{t-1}))

        Args:
            closes: Close prices

        Returns:
            Estimated spread as fraction of price
        """
        window = self.config.roll_window
        if len(closes) < window + 2:
            return 0.0

        # Price changes
        price_changes = np.diff(closes[-window-1:])

        # Serial covariance
        cov = np.cov(price_changes[:-1], price_changes[1:])[0, 1]

        # Roll spread (only valid if covariance is negative)
        if cov >= 0:
            return 0.0

        spread = 2 * np.sqrt(-cov)

        # Normalize by average price
        avg_price = np.mean(closes[-window:])
        if avg_price > 0:
            spread = spread / avg_price

        return np.clip(spread, 0, 0.1)  # Cap at 10%

    def corwin_schultz_spread(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> float:
        """
        Estimate spread using Corwin-Schultz (2012) high-low estimator.

        Uses ratio of high-low ranges across two periods.

        Args:
            highs: High prices
            lows: Low prices

        Returns:
            Estimated spread as fraction
        """
        window = self.config.corwin_schultz_window
        if len(highs) < window + 1:
            return 0.0

        spreads = []

        for i in range(window, len(highs)):
            # Two-day high-low
            h2 = max(highs[i], highs[i-1])
            l2 = min(lows[i], lows[i-1])

            # One-day high-lows
            beta = (np.log(highs[i]/lows[i]))**2 + (np.log(highs[i-1]/lows[i-1]))**2
            gamma = (np.log(h2/l2))**2

            # Corwin-Schultz formula
            alpha = (np.sqrt(2*beta) - np.sqrt(beta)) / (3 - 2*np.sqrt(2)) - np.sqrt(gamma/(3-2*np.sqrt(2)))

            if alpha > 0:
                spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
                spreads.append(spread)

        if not spreads:
            return 0.0

        return np.clip(np.mean(spreads), 0, 0.1)

    def flow_toxicity(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
    ) -> float:
        """
        Calculate order flow toxicity measure.

        Measures the imbalance between buying and selling pressure,
        weighted by the informativeness of trades.

        Args:
            closes: Close prices
            volumes: Volumes

        Returns:
            Flow toxicity score (-1 to 1)
        """
        window = min(20, len(closes) - 1)
        if window < 5:
            return 0.0

        # Price changes and their absolute values
        price_changes = np.diff(closes[-window-1:])

        # Volume-weighted direction
        signed_flows = np.sign(price_changes) * volumes[-window:]

        # Toxicity = cumulative imbalance / total volume
        cumulative_imbalance = np.sum(signed_flows)
        total_volume = np.sum(volumes[-window:])

        if total_volume < 1e-10:
            return 0.0

        toxicity = cumulative_imbalance / total_volume
        return np.clip(toxicity, -1, 1)

    def volume_clock(self, volumes: np.ndarray) -> float:
        """
        Calculate volume clock indicator.

        Measures current volume relative to historical patterns,
        indicating where we are in the "volume day".

        Args:
            volumes: Volumes

        Returns:
            Volume clock position (0 to 1)
        """
        window = min(50, len(volumes))
        if window < 5:
            return 0.5

        # Cumulative volume fractions
        recent_volumes = volumes[-window:]
        total_vol = np.sum(recent_volumes)

        if total_vol < 1e-10:
            return 0.5

        # What fraction of expected volume has occurred
        current_vol = volumes[-1]
        avg_vol = total_vol / window

        if avg_vol < 1e-10:
            return 0.5

        # Normalize to 0-1 range
        clock = current_vol / (2 * avg_vol)
        return np.clip(clock, 0, 1)

    def variance_ratio(self, closes: np.ndarray, period: int = 5) -> float:
        """
        Calculate variance ratio for market efficiency.

        Lo & MacKinlay (1988): VR(q) = Var(r_q) / (q * Var(r_1))

        VR = 1 implies efficient market (random walk)
        VR > 1 implies momentum
        VR < 1 implies mean reversion

        Args:
            closes: Close prices
            period: Aggregation period

        Returns:
            Variance ratio (centered around 0)
        """
        if len(closes) < period * 5:
            return 0.0

        # Single-period returns
        returns_1 = np.diff(np.log(closes))

        # Multi-period returns
        log_prices = np.log(closes)
        returns_q = log_prices[period:] - log_prices[:-period]

        var_1 = np.var(returns_1)
        var_q = np.var(returns_q)

        if var_1 < 1e-10:
            return 0.0

        # Variance ratio
        vr = var_q / (period * var_1)

        # Return centered value (0 = efficient, >0 = momentum, <0 = reversion)
        return np.clip(vr - 1, -2, 2)


def get_microstructure_features(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    config: Optional[MicrostructureConfig] = None,
) -> np.ndarray:
    """
    Get microstructure features as numpy array.

    Convenience function for integration with observation builder.

    Args:
        closes: Close prices
        highs: High prices
        lows: Low prices
        volumes: Volumes
        config: Optional configuration

    Returns:
        Array of 8 microstructure features
    """
    calculator = MicrostructureFeatures(config)
    features_dict = calculator.calculate_all(closes, highs, lows, volumes)

    # Return in consistent order
    feature_order = [
        'kyle_lambda',
        'amihud_illiquidity',
        'vpin',
        'roll_spread',
        'cs_spread',
        'flow_toxicity',
        'volume_clock',
        'price_efficiency',
    ]

    return np.array([features_dict[k] for k in feature_order], dtype=np.float32)
