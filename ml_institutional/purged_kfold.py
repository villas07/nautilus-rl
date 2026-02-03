"""
Purged K-Fold Cross-Validation.

Based on Marcos López de Prado - Advances in Financial Machine Learning.

Standard K-Fold CV has data leakage in financial data because:
1. Labels can overlap temporally (a label at t may depend on data up to t+n)
2. Sequential data means test set info can leak into training

Purged K-Fold solves this by:
1. PURGE: Remove from training any samples whose labels overlap with test
2. EMBARGO: Add a gap after each test set to prevent leakage

Example:
┌────────┬────────┬────────┬────────┬────────┐
│ TRAIN  │EMBARGO │  TEST  │ TRAIN  │ TRAIN  │  Fold 1
├────────┼────────┼────────┼────────┼────────┤
│ TRAIN  │ TRAIN  │EMBARGO │  TEST  │ TRAIN  │  Fold 2
└────────┴────────┴────────┴────────┴────────┘
"""

import numpy as np
import pandas as pd
from typing import Optional, Generator, Tuple, Dict, Any
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold Cross-Validation for time series.

    Prevents data leakage by:
    1. Purging training samples that overlap with test labels
    2. Adding embargo gap between train and test

    Example:
        cv = PurgedKFold(n_splits=5, t1=label_end_times, pct_embargo=0.01)
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: Optional[pd.Series] = None,
        pct_embargo: float = 0.01
    ):
        """
        Initialize Purged K-Fold CV.

        Args:
            n_splits: Number of folds
            t1: Series mapping sample index to label expiration time.
                If provided, purging will remove overlapping samples.
            pct_embargo: Fraction of samples to embargo after test set (0-1)
        """
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices for train/test splits.

        Args:
            X: Feature DataFrame (must have DatetimeIndex)
            y: Target series (optional)
            groups: Group labels (optional, unused)

        Yields:
            Tuples of (train_indices, test_indices)
        """
        # Use sample index if t1 not provided
        if self.t1 is None:
            t1 = pd.Series(X.index, index=X.index)
        else:
            t1 = self.t1

        n_samples = len(X)
        indices = np.arange(n_samples)
        embargo = int(n_samples * self.pct_embargo)

        # Size of each test fold
        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # Test indices for this fold
            test_start = i * test_size
            test_end = (i + 1) * test_size if i < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]

            # Test time range
            test_times = X.index[test_indices]
            test_start_time = test_times.min()
            test_end_time = test_times.max()

            # Build train indices with purging and embargo
            train_indices = []

            for j in indices:
                # Skip if in test set
                if j in test_indices:
                    continue

                # Skip if in embargo zone (right after test)
                if test_end <= j < test_end + embargo:
                    continue

                sample_time = X.index[j]
                sample_end = t1.iloc[j]

                # PURGE: Skip if sample's label overlaps with test period
                # A sample overlaps if its label extends into or past the test period
                if pd.notna(sample_end):
                    # Sample started before test and ends during/after test start
                    if sample_time < test_start_time and sample_end >= test_start_time:
                        continue

                    # Sample is in test period
                    if test_start_time <= sample_time <= test_end_time:
                        continue

                train_indices.append(j)

            yield np.array(train_indices), test_indices

    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Return number of splits."""
        return self.n_splits


def cv_score(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    t1: Optional[pd.Series] = None,
    cv: int = 5,
    pct_embargo: float = 0.01,
    sample_weight: Optional[pd.Series] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Cross-validation with Purged K-Fold.

    Args:
        model: Sklearn-compatible classifier with fit/predict methods
        X: Feature DataFrame
        y: Target series
        t1: Label expiration times (for purging)
        cv: Number of folds
        pct_embargo: Embargo percentage
        sample_weight: Sample weights for training
        verbose: Print progress

    Returns:
        Dict with:
        - scores: Dict of metric lists across folds
        - mean: Dict of mean scores
        - std: Dict of score standard deviations
    """
    purged_cv = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)

    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }

    for fold, (train_idx, test_idx) in enumerate(purged_cv.split(X)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Fit with sample weights if provided
        if sample_weight is not None:
            sw_train = sample_weight.iloc[train_idx]
            try:
                model.fit(X_train, y_train, sample_weight=sw_train)
            except TypeError:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Get probabilities for AUC if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        # Calculate scores
        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        scores['f1'].append(f1_score(y_test, y_pred, zero_division=0))

        try:
            scores['auc'].append(roc_auc_score(y_test, y_proba))
        except ValueError:
            scores['auc'].append(0.5)

        if verbose:
            print(
                f"Fold {fold+1}/{cv}: "
                f"Acc={scores['accuracy'][-1]:.3f}, "
                f"AUC={scores['auc'][-1]:.3f}, "
                f"Train={len(train_idx)}, Test={len(test_idx)}"
            )

    # Calculate summary statistics
    mean_scores = {k: np.mean(v) for k, v in scores.items()}
    std_scores = {k: np.std(v) for k, v in scores.items()}

    if verbose:
        print("\n" + "=" * 50)
        print("PURGED K-FOLD CV RESULTS")
        print("=" * 50)
        for metric in scores:
            print(f"{metric}: {mean_scores[metric]:.3f} (+/- {std_scores[metric]:.3f})")

    return {
        'scores': scores,
        'mean': mean_scores,
        'std': std_scores
    }


def test_purged_kfold():
    """Test Purged K-Fold with sample data."""
    from sklearn.ensemble import RandomForestClassifier

    # Create sample time series data
    np.random.seed(42)
    n_samples = 500
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    # Features: random
    X = pd.DataFrame(
        np.random.randn(n_samples, 10),
        index=dates,
        columns=[f'feature_{i}' for i in range(10)]
    )

    # Target: random binary (to verify ~50% accuracy on random data)
    y = pd.Series(np.random.randint(0, 2, n_samples), index=dates)

    # t1: Each label "expires" 5 days after sample
    t1 = pd.Series(dates + pd.Timedelta(days=5), index=dates)

    print("Testing Purged K-Fold CV...")
    print(f"Samples: {n_samples}, Features: {X.shape[1]}")
    print(f"Class balance: {y.mean():.2%} positive\n")

    # Test with simple model
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)

    results = cv_score(
        model=model,
        X=X,
        y=y,
        t1=t1,
        cv=5,
        pct_embargo=0.01,
        verbose=True
    )

    print(f"\nExpected accuracy ~50% (random data): {results['mean']['accuracy']:.1%}")

    return results


if __name__ == "__main__":
    test_purged_kfold()
