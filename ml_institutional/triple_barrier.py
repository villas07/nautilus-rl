"""
Triple Barrier Labeling Method.

Based on Marcos López de Prado - Advances in Financial Machine Learning.

The Triple Barrier method creates labels based on which barrier is touched first:
- Upper barrier (take profit): Label = +1
- Lower barrier (stop loss): Label = -1
- Vertical barrier (timeout): Label = 0 (or based on return sign)

This is superior to simple next-day direction because:
1. Reflects real trading (you always have SL/TP)
2. Considers the path, not just the endpoint
3. Creates more balanced and meaningful labels
4. Filters noise with minimum return threshold
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class TripleBarrierConfig:
    """Configuration for Triple Barrier labeling."""
    pt_mult: float = 2.0  # Profit taking multiplier (x volatility)
    sl_mult: float = 1.0  # Stop loss multiplier (x volatility)
    min_ret: float = 0.005  # Minimum return to consider (filters noise)
    vertical_days: int = 10  # Maximum holding period
    vol_span: int = 100  # EWMA span for volatility calculation


class TripleBarrierLabeling:
    """
    Triple Barrier Labeling Method.

    Creates labels based on which barrier is touched first:
    - +1: Take profit hit first (long profitable)
    - -1: Stop loss hit first (short profitable)
    -  0: Timeout (vertical barrier hit)

    Example:
        labeler = TripleBarrierLabeling(pt_mult=2.0, sl_mult=1.0)
        labels = labeler.get_labels(df['close'])
    """

    def __init__(
        self,
        pt_mult: float = 2.0,
        sl_mult: float = 1.0,
        min_ret: float = 0.005,
        vertical_days: int = 10,
        vol_span: int = 100
    ):
        """
        Initialize Triple Barrier Labeler.

        Args:
            pt_mult: Profit taking multiplier (x daily volatility)
            sl_mult: Stop loss multiplier (x daily volatility)
            min_ret: Minimum return threshold (filters noise)
            vertical_days: Maximum holding period in days
            vol_span: EWMA span for volatility estimation
        """
        self.config = TripleBarrierConfig(
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            min_ret=min_ret,
            vertical_days=vertical_days,
            vol_span=vol_span
        )

    def get_daily_vol(self, close: pd.Series) -> pd.Series:
        """
        Estimate daily volatility using EWMA.

        Args:
            close: Series of closing prices

        Returns:
            Series of daily volatility estimates
        """
        returns = close.pct_change()
        return returns.ewm(span=self.config.vol_span).std()

    def get_vertical_barriers(
        self,
        events: pd.DatetimeIndex,
        close: pd.Series
    ) -> pd.Series:
        """
        Calculate vertical barriers (time limit).

        Args:
            events: Dates where to evaluate
            close: Price series

        Returns:
            Series of timeout dates
        """
        t1 = events + pd.Timedelta(days=self.config.vertical_days)

        # Adjust to available dates
        def find_closest_date(target_date):
            if target_date > close.index[-1]:
                return pd.NaT
            idx = close.index.searchsorted(target_date, side='right') - 1
            if idx < 0:
                return pd.NaT
            return close.index[idx]

        t1 = pd.Series([find_closest_date(d) for d in t1], index=events)
        return t1

    def apply_triple_barrier(
        self,
        close: pd.Series,
        events: pd.DatetimeIndex,
        daily_vol: pd.Series
    ) -> pd.DataFrame:
        """
        Apply triple barrier method and generate labels.

        Args:
            close: Price series
            events: Dates where to evaluate
            daily_vol: Daily volatility series

        Returns:
            DataFrame with:
            - t1: Time when barrier was touched
            - ret: Return at barrier touch
            - label: +1 (TP), -1 (SL), 0 (timeout)
            - barrier: Which barrier was touched
        """
        # Get vertical barriers
        t1 = self.get_vertical_barriers(events, close)

        results = []

        for event in events:
            if event not in close.index:
                continue

            # Entry price
            entry_price = close.loc[event]

            # Get volatility at entry
            if event not in daily_vol.index:
                continue
            vol = daily_vol.loc[event]

            if pd.isna(vol) or vol == 0:
                continue

            # Calculate barriers
            upper = self.config.pt_mult * vol  # Take profit
            lower = -self.config.sl_mult * vol  # Stop loss
            timeout = t1.loc[event]

            if pd.isna(timeout):
                continue

            # Get price path from entry to timeout
            try:
                path = close.loc[event:timeout]
            except KeyError:
                continue

            if len(path) <= 1:
                continue

            # Calculate returns along the path
            returns = path / entry_price - 1

            # Find which barrier is touched first
            label = 0  # Default: timeout
            touch_time = timeout
            ret = returns.iloc[-1]
            barrier = 'vertical'

            # Check upper barrier (take profit)
            if self.config.pt_mult > 0:
                upper_touches = returns[returns >= upper]
                if len(upper_touches) > 0:
                    upper_time = upper_touches.index[0]
                    if upper_time < touch_time:
                        touch_time = upper_time
                        label = 1
                        ret = upper
                        barrier = 'upper'

            # Check lower barrier (stop loss)
            if self.config.sl_mult > 0:
                lower_touches = returns[returns <= lower]
                if len(lower_touches) > 0:
                    lower_time = lower_touches.index[0]
                    if lower_time < touch_time:
                        touch_time = lower_time
                        label = -1
                        ret = lower
                        barrier = 'lower'

            results.append({
                'event_time': event,
                't1': touch_time,
                'ret': ret,
                'label': label,
                'barrier': barrier,
                'vol': vol
            })

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results).set_index('event_time')

    def get_labels(
        self,
        close: pd.Series,
        events: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """
        Main method: Generate labels for training.

        Args:
            close: Series of closing prices
            events: Dates where to evaluate (default: all dates)

        Returns:
            DataFrame with labels and metadata:
            - label: +1, -1, or 0
            - ret: Return at barrier touch
            - t1: Barrier touch time
            - barrier: Which barrier was touched
        """
        # Use all dates if not specified
        if events is None:
            # Skip initial period needed for volatility
            events = close.index[self.config.vol_span:]

        # Calculate daily volatility
        daily_vol = self.get_daily_vol(close)

        # Apply triple barrier
        labels = self.apply_triple_barrier(close, events, daily_vol)

        if labels.empty:
            return labels

        # Filter by minimum return
        labels = labels[labels['ret'].abs() >= self.config.min_ret]

        return labels

    def get_binary_labels(
        self,
        close: pd.Series,
        events: Optional[pd.DatetimeIndex] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get binary labels (0/1) for classification.

        Converts:
        - +1 (take profit) → 1 (positive outcome)
        - -1 (stop loss) → 0 (negative outcome)
        - 0 (timeout) → based on return sign

        Args:
            close: Price series
            events: Dates to evaluate

        Returns:
            Tuple of (full_labels_df, binary_labels_series)
        """
        labels = self.get_labels(close, events)

        if labels.empty:
            return labels, pd.Series(dtype=int)

        # Convert to binary
        # +1 → 1, -1 → 0, 0 → based on return
        binary = labels['label'].copy()
        binary = binary.replace({1: 1, -1: 0})

        # For timeout (0), use return sign
        timeout_mask = labels['label'] == 0
        binary.loc[timeout_mask] = (labels.loc[timeout_mask, 'ret'] > 0).astype(int)

        return labels, binary


def test_triple_barrier():
    """Test Triple Barrier with sample data."""
    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')

    # Random walk with drift
    returns = np.random.randn(500) * 0.02 + 0.0001
    prices = 100 * np.exp(np.cumsum(returns))
    close = pd.Series(prices, index=dates)

    # Test labeling
    labeler = TripleBarrierLabeling(
        pt_mult=2.0,
        sl_mult=1.0,
        min_ret=0.005,
        vertical_days=10
    )

    labels = labeler.get_labels(close)

    print("Triple Barrier Test Results:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Label distribution:")
    print(f"    +1 (TP): {(labels['label'] == 1).sum()}")
    print(f"    -1 (SL): {(labels['label'] == -1).sum()}")
    print(f"     0 (TO): {(labels['label'] == 0).sum()}")
    print(f"  Barrier distribution:")
    print(labels['barrier'].value_counts())

    # Binary labels
    _, binary = labeler.get_binary_labels(close)
    print(f"\n  Binary balance: {binary.mean():.2%} positive")

    return labels


if __name__ == "__main__":
    test_triple_barrier()
