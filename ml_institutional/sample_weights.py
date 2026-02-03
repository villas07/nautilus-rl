"""
Sample Weighting for ML Training.

Based on Marcos López de Prado - Advances in Financial Machine Learning.

In financial ML, samples are often not independent:
- Labels can overlap temporally
- Some samples are more "unique" than others

Sample weighting addresses this by:
1. UNIQUENESS: Weight inversely to concurrent labels
2. TIME DECAY: Recent samples get more weight

This reduces overfitting to redundant samples.
"""

import numpy as np
import pandas as pd
from typing import Optional


def get_concurrent_labels(
    t1: pd.Series,
    index: pd.DatetimeIndex
) -> pd.Series:
    """
    Count concurrent labels at each point in time.

    A label is concurrent at time t if:
    - Its start time <= t
    - Its end time (t1) >= t

    Args:
        t1: Series mapping sample index to label end time
        index: Full time index to evaluate

    Returns:
        Series with count of concurrent labels at each time
    """
    # Initialize count series
    concurrent = pd.Series(0, index=index)

    for sample_start, sample_end in t1.items():
        if pd.isna(sample_end):
            continue

        # This label is concurrent from start to end
        mask = (index >= sample_start) & (index <= sample_end)
        concurrent.loc[mask] += 1

    return concurrent


def get_average_uniqueness(
    t1: pd.Series,
    concurrent: pd.Series
) -> pd.Series:
    """
    Calculate average uniqueness for each sample.

    Uniqueness at time t = 1 / concurrent_count(t)
    Average uniqueness = mean(uniqueness) over label lifetime

    Args:
        t1: Label end times
        concurrent: Concurrent label counts

    Returns:
        Series with average uniqueness per sample
    """
    uniqueness = pd.Series(index=t1.index, dtype=float)

    for sample_start, sample_end in t1.items():
        if pd.isna(sample_end):
            uniqueness[sample_start] = 0
            continue

        # Get concurrent counts over label lifetime
        try:
            label_concurrent = concurrent.loc[sample_start:sample_end]
        except KeyError:
            uniqueness[sample_start] = 0
            continue

        if len(label_concurrent) == 0:
            uniqueness[sample_start] = 0
            continue

        # Average uniqueness = mean(1/count)
        label_uniqueness = (1.0 / label_concurrent.replace(0, 1))
        uniqueness[sample_start] = label_uniqueness.mean()

    return uniqueness


def get_time_decay_weights(
    index: pd.DatetimeIndex,
    decay_start: float = 0.5,
    decay_end: float = 1.0
) -> pd.Series:
    """
    Calculate time decay weights (older samples worth less).

    Args:
        index: Sample index
        decay_start: Weight for oldest sample
        decay_end: Weight for newest sample

    Returns:
        Series with time decay weights
    """
    n = len(index)
    if n <= 1:
        return pd.Series(1.0, index=index)

    weights = np.linspace(decay_start, decay_end, n)
    return pd.Series(weights, index=index)


def get_sample_weights(
    t1: pd.Series,
    close: Optional[pd.Series] = None,
    use_uniqueness: bool = True,
    use_time_decay: bool = True,
    decay_start: float = 0.5,
    normalize: bool = True
) -> pd.Series:
    """
    Calculate combined sample weights.

    Combines:
    - Uniqueness weighting (inversely proportional to concurrent labels)
    - Time decay weighting (recent samples worth more)

    Args:
        t1: Series mapping sample index to label end time
        close: Price series (optional, for index alignment)
        use_uniqueness: Include uniqueness weighting
        use_time_decay: Include time decay weighting
        decay_start: Starting weight for time decay
        normalize: Normalize weights to sum to len(t1)

    Returns:
        Series with combined sample weights
    """
    # Initialize weights to 1
    weights = pd.Series(1.0, index=t1.index)

    # Uniqueness weighting
    if use_uniqueness:
        # Get full time index
        if close is not None:
            full_index = close.index
        else:
            full_index = t1.index

        concurrent = get_concurrent_labels(t1, full_index)
        uniqueness = get_average_uniqueness(t1, concurrent)

        # Combine with weights
        weights = weights * uniqueness.clip(lower=0.01)

    # Time decay weighting
    if use_time_decay:
        time_decay = get_time_decay_weights(t1.index, decay_start=decay_start)
        weights = weights * time_decay

    # Handle zeros and NaN
    weights = weights.fillna(0).clip(lower=0.01)

    # Normalize
    if normalize and weights.sum() > 0:
        weights = weights * len(weights) / weights.sum()

    return weights


def test_sample_weights():
    """Test sample weights calculation."""
    np.random.seed(42)

    # Create sample data
    n_samples = 100
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    # t1: Labels expire 5 days after sample
    t1 = pd.Series(dates + pd.Timedelta(days=5), index=dates)

    # Price series
    close = pd.Series(100 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.02)), index=dates)

    print("Testing Sample Weights...")
    print(f"Samples: {n_samples}")
    print(f"Label duration: 5 days\n")

    # Calculate weights
    weights = get_sample_weights(
        t1=t1,
        close=close,
        use_uniqueness=True,
        use_time_decay=True,
        decay_start=0.5
    )

    print(f"Weight statistics:")
    print(f"  Min: {weights.min():.3f}")
    print(f"  Max: {weights.max():.3f}")
    print(f"  Mean: {weights.mean():.3f}")
    print(f"  Std: {weights.std():.3f}")
    print(f"  Sum: {weights.sum():.1f} (should be ~{n_samples})")

    # Show first/last few weights
    print(f"\nFirst 5 weights (oldest, should be lower):")
    print(weights.head().to_string())

    print(f"\nLast 5 weights (newest, should be higher):")
    print(weights.tail().to_string())

    return weights


if __name__ == "__main__":
    test_sample_weights()
