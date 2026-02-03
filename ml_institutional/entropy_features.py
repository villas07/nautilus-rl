"""
Entropy and Complexity Features for Institutional ML.

Based on:
- Shannon (1948) - Information Entropy
- Pincus (1991) - Approximate Entropy
- Richman & Moorman (2000) - Sample Entropy
- Lempel & Ziv (1976) - Complexity Measure

Reference: Marcos López de Prado - Advances in Financial Machine Learning
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class EntropyConfig:
    """Configuration for entropy feature calculation."""
    num_bins: int = 10              # Bins for Shannon entropy
    embed_dim: int = 2              # Embedding dimension for ApEn/SampEn
    tolerance_mult: float = 0.2    # Tolerance = mult * std(data)
    lz_threshold: float = 0.5      # Threshold for Lempel-Ziv binarization


class EntropyFeatures:
    """
    Calculate entropy and complexity features from price data.

    Features calculated:
    1. Shannon Entropy - Information content of price distribution
    2. Approximate Entropy (ApEn) - Regularity measure
    3. Sample Entropy (SampEn) - Improved ApEn (no self-matching)
    4. Lempel-Ziv Complexity - Pattern complexity measure
    """

    def __init__(self, config: Optional[EntropyConfig] = None):
        """Initialize with configuration."""
        self.config = config or EntropyConfig()

    def calculate_all(
        self,
        closes: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate all entropy features.

        Args:
            closes: Array of close prices
            returns: Optional pre-computed returns

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        # Calculate returns if not provided
        if returns is None:
            if len(closes) > 1:
                returns = np.diff(closes) / closes[:-1]
            else:
                returns = np.array([0.0])

        # Shannon Entropy
        features['shannon_entropy'] = self.shannon_entropy(returns)

        # Approximate Entropy
        features['approx_entropy'] = self.approximate_entropy(returns)

        # Sample Entropy
        features['sample_entropy'] = self.sample_entropy(returns)

        # Lempel-Ziv Complexity
        features['lz_complexity'] = self.lempel_ziv_complexity(returns)

        return features

    def shannon_entropy(self, data: np.ndarray) -> float:
        """
        Calculate Shannon entropy of the data distribution.

        H = -sum(p * log(p)) where p is the probability distribution.

        Higher entropy = more unpredictable/random.

        Args:
            data: Input data (typically returns)

        Returns:
            Shannon entropy (normalized to 0-1 range)
        """
        if len(data) < 2:
            return 0.5

        # Remove NaN/Inf
        data = data[np.isfinite(data)]
        if len(data) < 2:
            return 0.5

        # Create histogram
        num_bins = min(self.config.num_bins, len(data) // 2)
        if num_bins < 2:
            num_bins = 2

        hist, _ = np.histogram(data, bins=num_bins, density=True)

        # Convert to probabilities
        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist

        # Calculate entropy
        # Add small epsilon to avoid log(0)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.5

        entropy = -np.sum(hist * np.log2(hist))

        # Normalize by maximum entropy (log2(num_bins))
        max_entropy = np.log2(num_bins)
        if max_entropy > 0:
            entropy = entropy / max_entropy

        return np.clip(entropy, 0, 1)

    def approximate_entropy(
        self,
        data: np.ndarray,
        m: Optional[int] = None,
        r: Optional[float] = None,
    ) -> float:
        """
        Calculate Approximate Entropy (ApEn).

        Pincus (1991): Measures the logarithmic likelihood that patterns
        that are close (within tolerance r) for m consecutive observations
        remain close for m+1 observations.

        Lower ApEn = more regular/predictable
        Higher ApEn = more complex/random

        Args:
            data: Input data
            m: Embedding dimension (default from config)
            r: Tolerance (default: 0.2 * std)

        Returns:
            Approximate entropy (typically 0-2 range, normalized to 0-1)
        """
        if m is None:
            m = self.config.embed_dim
        if r is None:
            r = self.config.tolerance_mult * np.std(data) if np.std(data) > 0 else 0.1

        n = len(data)
        if n < m + 1:
            return 0.5

        # Remove NaN
        data = np.nan_to_num(data, nan=0.0)

        def phi(m_val):
            """Calculate phi for given embedding dimension."""
            patterns = np.array([data[i:i+m_val] for i in range(n - m_val + 1)])
            n_patterns = len(patterns)

            if n_patterns == 0:
                return 0

            counts = np.zeros(n_patterns)

            for i in range(n_patterns):
                # Count patterns within tolerance r
                for j in range(n_patterns):
                    if np.max(np.abs(patterns[i] - patterns[j])) <= r:
                        counts[i] += 1

            # Average log probability
            counts = counts / n_patterns
            counts = counts[counts > 0]

            if len(counts) == 0:
                return 0

            return np.mean(np.log(counts))

        # ApEn = phi(m) - phi(m+1)
        phi_m = phi(m)
        phi_m1 = phi(m + 1)

        apen = phi_m - phi_m1

        # Normalize to 0-1 range (typical ApEn for financial data is 0-2)
        return np.clip(apen / 2, 0, 1)

    def sample_entropy(
        self,
        data: np.ndarray,
        m: Optional[int] = None,
        r: Optional[float] = None,
    ) -> float:
        """
        Calculate Sample Entropy (SampEn).

        Richman & Moorman (2000): Similar to ApEn but excludes self-matching,
        reducing bias for short time series.

        Args:
            data: Input data
            m: Embedding dimension
            r: Tolerance

        Returns:
            Sample entropy (normalized to 0-1)
        """
        if m is None:
            m = self.config.embed_dim
        if r is None:
            r = self.config.tolerance_mult * np.std(data) if np.std(data) > 0 else 0.1

        n = len(data)
        if n < m + 2:
            return 0.5

        # Remove NaN
        data = np.nan_to_num(data, nan=0.0)

        def count_matches(templates, r_val):
            """Count matching template pairs within tolerance."""
            n_templates = len(templates)
            count = 0

            for i in range(n_templates):
                for j in range(i + 1, n_templates):  # Exclude self-matching
                    if np.max(np.abs(templates[i] - templates[j])) <= r_val:
                        count += 1

            return count

        # Create templates of length m and m+1
        templates_m = np.array([data[i:i+m] for i in range(n - m)])
        templates_m1 = np.array([data[i:i+m+1] for i in range(n - m - 1)])

        # Count matches
        count_m = count_matches(templates_m, r)
        count_m1 = count_matches(templates_m1, r)

        # Sample entropy
        if count_m == 0 or count_m1 == 0:
            return 0.5

        sampen = -np.log(count_m1 / count_m)

        # Normalize (typical range 0-3 for financial data)
        return np.clip(sampen / 3, 0, 1)

    def lempel_ziv_complexity(self, data: np.ndarray) -> float:
        """
        Calculate Lempel-Ziv complexity.

        Lempel & Ziv (1976): Measures the number of distinct patterns
        in a binary sequence. Used to quantify the complexity/randomness.

        Args:
            data: Input data (will be binarized)

        Returns:
            Normalized LZ complexity (0-1)
        """
        n = len(data)
        if n < 5:
            return 0.5

        # Remove NaN
        data = np.nan_to_num(data, nan=0.0)

        # Binarize: 1 if above median, 0 otherwise
        threshold = np.median(data)
        binary = (data > threshold).astype(int)

        # LZ complexity calculation
        complexity = 1
        prefix_length = 1
        i = 0

        while i + prefix_length <= n:
            # Check if current substring is in previous part
            substring = binary[i:i + prefix_length]
            found = False

            for j in range(i):
                if j + prefix_length <= i and np.array_equal(binary[j:j + prefix_length], substring):
                    found = True
                    break

            if found:
                prefix_length += 1
            else:
                complexity += 1
                i += prefix_length
                prefix_length = 1

        # Normalize by theoretical maximum complexity
        # For random sequence: c(n) ~ n / log2(n)
        if n > 1:
            max_complexity = n / np.log2(n)
            normalized = complexity / max_complexity
        else:
            normalized = 0.5

        return np.clip(normalized, 0, 1)

    def multiscale_entropy(
        self,
        data: np.ndarray,
        scales: List[int] = [1, 2, 4, 8],
    ) -> np.ndarray:
        """
        Calculate multiscale entropy (MSE).

        Costa et al. (2005): Sample entropy at multiple time scales
        reveals complexity across different temporal resolutions.

        Args:
            data: Input data
            scales: List of scale factors

        Returns:
            Array of entropy values at each scale
        """
        mse_values = []

        for scale in scales:
            if len(data) < scale * 10:
                mse_values.append(0.5)
                continue

            # Coarse-grain the data
            n_points = len(data) // scale
            coarse_data = np.mean(data[:n_points * scale].reshape(-1, scale), axis=1)

            # Calculate sample entropy at this scale
            se = self.sample_entropy(coarse_data)
            mse_values.append(se)

        return np.array(mse_values, dtype=np.float32)


def get_entropy_features(
    closes: np.ndarray,
    config: Optional[EntropyConfig] = None,
) -> np.ndarray:
    """
    Get entropy features as numpy array.

    Convenience function for integration with observation builder.

    Args:
        closes: Close prices
        config: Optional configuration

    Returns:
        Array of 4 entropy features
    """
    calculator = EntropyFeatures(config)

    # Calculate returns
    if len(closes) > 1:
        returns = np.diff(closes) / closes[:-1]
    else:
        returns = np.array([0.0])

    features_dict = calculator.calculate_all(closes, returns)

    # Return in consistent order
    feature_order = [
        'shannon_entropy',
        'approx_entropy',
        'sample_entropy',
        'lz_complexity',
    ]

    return np.array([features_dict[k] for k in feature_order], dtype=np.float32)


def get_multiscale_entropy_features(
    closes: np.ndarray,
    scales: List[int] = [1, 2, 4, 8],
) -> np.ndarray:
    """
    Get multiscale entropy features.

    Args:
        closes: Close prices
        scales: Scale factors

    Returns:
        Array of MSE values at each scale
    """
    calculator = EntropyFeatures()

    if len(closes) > 1:
        returns = np.diff(closes) / closes[:-1]
    else:
        returns = np.array([0.0])

    return calculator.multiscale_entropy(returns, scales)
