"""
Data Quality Control

Validates and scores data quality:
1. Timestamp validation (market hours, chronological order)
2. Outlier detection (price > 3σ from rolling mean)
3. Volume validation (zero/negative volume)
4. Price sanity checks (OHLC relationships)
5. Gap detection
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, time as dt_time
from typing import List, Tuple, Optional, Dict, Any, Set
from enum import Enum

import pandas as pd
import numpy as np

import structlog

from data.pipeline.config import AssetType, QualityConfig, MarketHours

logger = structlog.get_logger()


class IssueType(str, Enum):
    """Types of data quality issues."""

    TIMESTAMP_OUT_OF_HOURS = "timestamp_out_of_hours"
    TIMESTAMP_FUTURE = "timestamp_future"
    TIMESTAMP_DUPLICATE = "timestamp_duplicate"
    TIMESTAMP_OUT_OF_ORDER = "timestamp_out_of_order"

    PRICE_OUTLIER = "price_outlier"
    PRICE_NEGATIVE = "price_negative"
    PRICE_ZERO = "price_zero"
    PRICE_INVALID_OHLC = "price_invalid_ohlc"

    VOLUME_ZERO = "volume_zero"
    VOLUME_NEGATIVE = "volume_negative"

    GAP_DETECTED = "gap_detected"


@dataclass
class QualityIssue:
    """A data quality issue."""

    issue_type: IssueType
    timestamp: datetime
    field: str
    value: Any
    expected: Optional[Any] = None
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.issue_type.value,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "field": self.field,
            "value": self.value,
            "expected": self.expected,
            "details": self.details,
        }


@dataclass
class QualityReport:
    """Quality assessment report for a dataset."""

    symbol: str
    timeframe: str
    total_rows: int = 0
    valid_rows: int = 0

    # Issue counts by type
    issues_by_type: Dict[IssueType, int] = field(default_factory=dict)

    # Detailed issues (limited to first N per type)
    issues: List[QualityIssue] = field(default_factory=list)

    # Quality scores (0-1)
    timestamp_score: float = 1.0
    price_score: float = 1.0
    volume_score: float = 1.0
    completeness_score: float = 1.0

    # Computed
    overall_score: float = 1.0

    # Statistics
    date_range: Tuple[datetime, datetime] = None
    expected_bars: int = 0
    gaps: List[Dict[str, Any]] = field(default_factory=list)

    def compute_overall_score(self):
        """Compute weighted overall quality score."""
        # Weights: completeness most important, then price, volume, timestamp
        self.overall_score = (
            0.4 * self.completeness_score +
            0.3 * self.price_score +
            0.2 * self.volume_score +
            0.1 * self.timestamp_score
        )

    def add_issue(self, issue: QualityIssue, max_issues_per_type: int = 10):
        """Add an issue to the report."""
        issue_type = issue.issue_type
        self.issues_by_type[issue_type] = self.issues_by_type.get(issue_type, 0) + 1

        # Limit stored details per type
        count = sum(1 for i in self.issues if i.issue_type == issue_type)
        if count < max_issues_per_type:
            self.issues.append(issue)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "issues_by_type": {k.value: v for k, v in self.issues_by_type.items()},
            "scores": {
                "timestamp": self.timestamp_score,
                "price": self.price_score,
                "volume": self.volume_score,
                "completeness": self.completeness_score,
                "overall": self.overall_score,
            },
            "date_range": [d.isoformat() for d in self.date_range] if self.date_range else None,
            "expected_bars": self.expected_bars,
            "actual_bars": self.total_rows,
            "gaps_count": len(self.gaps),
        }


class QualityController:
    """
    Validates data quality and generates quality reports.

    Checks:
    1. Timestamp validity (market hours, order, duplicates)
    2. Price outliers (rolling z-score > threshold)
    3. Volume validity (non-negative, suspicious zeros)
    4. OHLC relationships (high >= low, close in range)
    5. Data completeness (gaps, missing bars)
    """

    # Market hours by asset type
    MARKET_HOURS: Dict[AssetType, MarketHours] = {
        AssetType.STOCK: MarketHours.stock_us(),
        AssetType.ETF: MarketHours.stock_us(),
        AssetType.FUTURE: MarketHours.futures_cme(),
        AssetType.CRYPTO: MarketHours.crypto_24_7(),
        AssetType.FOREX: MarketHours.forex(),
        AssetType.INDEX: MarketHours.stock_us(),
    }

    # Timeframe to expected interval
    TIMEFRAME_INTERVALS: Dict[str, timedelta] = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
    }

    def __init__(self, config: Optional[QualityConfig] = None):
        """
        Initialize quality controller.

        Args:
            config: Quality control configuration.
        """
        self.config = config or QualityConfig()

    def validate(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        asset_type: AssetType,
    ) -> Tuple[pd.DataFrame, QualityReport]:
        """
        Validate a DataFrame and generate quality report.

        Args:
            df: DataFrame with OHLCV data.
            symbol: Symbol for reporting.
            timeframe: Timeframe for gap detection.
            asset_type: Asset type for market hours.

        Returns:
            Tuple of (cleaned DataFrame, QualityReport).
        """
        report = QualityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_rows=len(df),
        )

        if df.empty:
            report.overall_score = 0.0
            return df, report

        df = df.copy()

        # Record date range
        report.date_range = (df["timestamp"].min(), df["timestamp"].max())

        # 1. Validate timestamps
        df, ts_score = self._validate_timestamps(df, asset_type, report)
        report.timestamp_score = ts_score

        # 2. Validate prices
        df, price_score = self._validate_prices(df, report)
        report.price_score = price_score

        # 3. Validate volume
        df, vol_score = self._validate_volume(df, asset_type, report)
        report.volume_score = vol_score

        # 4. Check completeness
        report.completeness_score = self._check_completeness(df, timeframe, asset_type, report)

        # 5. Final cleanup
        df = self._cleanup(df)

        report.valid_rows = len(df)
        report.compute_overall_score()

        logger.info(
            f"Quality check: {symbol} {timeframe}",
            total=report.total_rows,
            valid=report.valid_rows,
            score=f"{report.overall_score:.2f}",
            issues=sum(report.issues_by_type.values()),
        )

        return df, report

    def _validate_timestamps(
        self,
        df: pd.DataFrame,
        asset_type: AssetType,
        report: QualityReport,
    ) -> Tuple[pd.DataFrame, float]:
        """Validate timestamps."""
        initial_count = len(df)
        issues = 0

        # Check for future timestamps
        now = datetime.now(timezone.utc)
        future_mask = df["timestamp"] > now + timedelta(seconds=self.config.max_future_timestamp_seconds)
        future_count = future_mask.sum()
        if future_count > 0:
            issues += future_count
            for ts in df.loc[future_mask, "timestamp"].head(5):
                report.add_issue(QualityIssue(
                    issue_type=IssueType.TIMESTAMP_FUTURE,
                    timestamp=ts,
                    field="timestamp",
                    value=ts,
                    expected=f"<= {now.isoformat()}",
                ))
            df = df[~future_mask]

        # Check for duplicates
        dup_mask = df.duplicated(subset=["timestamp"], keep="first")
        dup_count = dup_mask.sum()
        if dup_count > 0:
            issues += dup_count
            for ts in df.loc[dup_mask, "timestamp"].head(5):
                report.add_issue(QualityIssue(
                    issue_type=IssueType.TIMESTAMP_DUPLICATE,
                    timestamp=ts,
                    field="timestamp",
                    value=ts,
                    details="Duplicate timestamp",
                ))
            df = df[~dup_mask]

        # Check chronological order
        if len(df) > 1:
            out_of_order = df["timestamp"].diff().dt.total_seconds() < 0
            out_of_order_count = out_of_order.sum()
            if out_of_order_count > 0:
                issues += out_of_order_count
                report.add_issue(QualityIssue(
                    issue_type=IssueType.TIMESTAMP_OUT_OF_ORDER,
                    timestamp=datetime.now(timezone.utc),
                    field="timestamp",
                    value=out_of_order_count,
                    details=f"{out_of_order_count} out-of-order timestamps",
                ))
                # Sort to fix
                df = df.sort_values("timestamp")

        # Check market hours (if strict mode)
        if self.config.strict_market_hours and asset_type in self.MARKET_HOURS:
            hours = self.MARKET_HOURS[asset_type]
            df["_weekday"] = df["timestamp"].dt.dayofweek
            df["_time"] = df["timestamp"].dt.time

            in_hours_mask = (
                df["_weekday"].isin(hours.trading_days) &
                (df["_time"] >= hours.open_time) &
                (df["_time"] <= hours.close_time)
            )

            out_of_hours = (~in_hours_mask).sum()
            if out_of_hours > 0:
                issues += out_of_hours
                report.add_issue(QualityIssue(
                    issue_type=IssueType.TIMESTAMP_OUT_OF_HOURS,
                    timestamp=datetime.now(timezone.utc),
                    field="timestamp",
                    value=out_of_hours,
                    details=f"{out_of_hours} bars outside market hours",
                ))
                df = df[in_hours_mask]

            df = df.drop(columns=["_weekday", "_time"], errors="ignore")

        # Calculate score
        if initial_count > 0:
            score = 1.0 - (issues / initial_count)
        else:
            score = 1.0

        return df, max(0.0, score)

    def _validate_prices(
        self,
        df: pd.DataFrame,
        report: QualityReport,
    ) -> Tuple[pd.DataFrame, float]:
        """Validate price data."""
        if df.empty:
            return df, 1.0

        initial_count = len(df)
        issues = 0

        # Check for negative prices
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                neg_mask = df[col] < 0
                neg_count = neg_mask.sum()
                if neg_count > 0:
                    issues += neg_count
                    for _, row in df[neg_mask].head(3).iterrows():
                        report.add_issue(QualityIssue(
                            issue_type=IssueType.PRICE_NEGATIVE,
                            timestamp=row["timestamp"],
                            field=col,
                            value=row[col],
                            expected=">= 0",
                        ))
                    df = df[~neg_mask]

        # Check for zero prices
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                zero_mask = df[col] == 0
                zero_count = zero_mask.sum()
                if zero_count > 0:
                    issues += zero_count
                    for _, row in df[zero_mask].head(3).iterrows():
                        report.add_issue(QualityIssue(
                            issue_type=IssueType.PRICE_ZERO,
                            timestamp=row["timestamp"],
                            field=col,
                            value=0,
                            expected="> 0",
                        ))
                    # Don't remove - might be valid for some instruments

        # Check OHLC relationships
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            # High should be >= Low
            invalid_hl = df["high"] < df["low"]
            if invalid_hl.any():
                issues += invalid_hl.sum()
                for _, row in df[invalid_hl].head(3).iterrows():
                    report.add_issue(QualityIssue(
                        issue_type=IssueType.PRICE_INVALID_OHLC,
                        timestamp=row["timestamp"],
                        field="high/low",
                        value=f"H={row['high']}, L={row['low']}",
                        expected="high >= low",
                    ))

            # Close should be between high and low
            invalid_close = (df["close"] > df["high"]) | (df["close"] < df["low"])
            if invalid_close.any():
                issues += invalid_close.sum()
                for _, row in df[invalid_close].head(3).iterrows():
                    report.add_issue(QualityIssue(
                        issue_type=IssueType.PRICE_INVALID_OHLC,
                        timestamp=row["timestamp"],
                        field="close",
                        value=f"C={row['close']}, H={row['high']}, L={row['low']}",
                        expected="low <= close <= high",
                    ))

        # Detect outliers using rolling z-score
        if "close" in df.columns and len(df) >= 20:
            lookback = min(20, len(df) - 1)
            df["_rolling_mean"] = df["close"].rolling(window=lookback, min_periods=1).mean()
            df["_rolling_std"] = df["close"].rolling(window=lookback, min_periods=1).std()

            # Avoid division by zero
            df["_rolling_std"] = df["_rolling_std"].replace(0, np.nan)

            df["_zscore"] = abs(df["close"] - df["_rolling_mean"]) / df["_rolling_std"]

            outlier_mask = df["_zscore"] > self.config.outlier_std_threshold
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                issues += outlier_count
                for _, row in df[outlier_mask].head(5).iterrows():
                    report.add_issue(QualityIssue(
                        issue_type=IssueType.PRICE_OUTLIER,
                        timestamp=row["timestamp"],
                        field="close",
                        value=row["close"],
                        expected=f"within {self.config.outlier_std_threshold}σ of {row['_rolling_mean']:.2f}",
                        details=f"z-score: {row['_zscore']:.2f}",
                    ))

                # Mark outliers but don't remove (let reconciliation handle)
                df["is_outlier"] = outlier_mask

            df = df.drop(columns=["_rolling_mean", "_rolling_std", "_zscore"], errors="ignore")

        # Calculate score
        if initial_count > 0:
            score = 1.0 - min(1.0, issues / initial_count)
        else:
            score = 1.0

        return df, score

    def _validate_volume(
        self,
        df: pd.DataFrame,
        asset_type: AssetType,
        report: QualityReport,
    ) -> Tuple[pd.DataFrame, float]:
        """Validate volume data."""
        if df.empty or "volume" not in df.columns:
            return df, 1.0

        initial_count = len(df)
        issues = 0

        # Check for negative volume
        neg_mask = df["volume"] < 0
        neg_count = neg_mask.sum()
        if neg_count > 0:
            issues += neg_count
            for _, row in df[neg_mask].head(3).iterrows():
                report.add_issue(QualityIssue(
                    issue_type=IssueType.VOLUME_NEGATIVE,
                    timestamp=row["timestamp"],
                    field="volume",
                    value=row["volume"],
                    expected=">= 0",
                ))
            df = df[~neg_mask]

        # Check for zero volume (suspicious for most markets)
        if not self.config.allow_zero_volume and asset_type != AssetType.FOREX:
            zero_mask = df["volume"] == 0
            zero_count = zero_mask.sum()
            if zero_count > 0:
                issues += zero_count
                report.add_issue(QualityIssue(
                    issue_type=IssueType.VOLUME_ZERO,
                    timestamp=datetime.now(timezone.utc),
                    field="volume",
                    value=zero_count,
                    details=f"{zero_count} bars with zero volume",
                ))
                # Mark as suspicious but don't remove
                df["volume_suspicious"] = df["volume"] == 0

        # Calculate score
        if initial_count > 0:
            score = 1.0 - min(1.0, issues / initial_count)
        else:
            score = 1.0

        return df, score

    def _check_completeness(
        self,
        df: pd.DataFrame,
        timeframe: str,
        asset_type: AssetType,
        report: QualityReport,
    ) -> float:
        """Check data completeness and detect gaps."""
        if df.empty or len(df) < 2:
            return 0.0 if df.empty else 1.0

        interval = self.TIMEFRAME_INTERVALS.get(timeframe)
        if interval is None:
            return 1.0

        # Calculate expected bars
        date_range = df["timestamp"].max() - df["timestamp"].min()
        if interval.total_seconds() > 0:
            expected_bars = int(date_range.total_seconds() / interval.total_seconds()) + 1

            # Adjust for non-24/7 markets
            if asset_type not in [AssetType.CRYPTO]:
                # Rough adjustment: ~6.5 hours per day for stocks, ~23 hours for futures
                if asset_type in [AssetType.STOCK, AssetType.ETF, AssetType.INDEX]:
                    market_fraction = 6.5 / 24  # ~27%
                else:
                    market_fraction = 23 / 24  # ~96%

                # Also exclude weekends
                days = date_range.days
                weekends = days // 7 * 2
                expected_bars = int(expected_bars * market_fraction * (days - weekends + 1) / (days + 1))
        else:
            expected_bars = len(df)

        report.expected_bars = max(expected_bars, 1)

        # Detect gaps
        df = df.sort_values("timestamp")
        time_diffs = df["timestamp"].diff()
        gap_threshold = interval * self.config.max_gap_bars

        gaps = time_diffs > gap_threshold
        gap_count = gaps.sum()

        if gap_count > 0:
            gap_indices = gaps[gaps].index.tolist()
            for idx in gap_indices[:10]:  # Limit to 10 gaps
                if idx > 0:
                    prev_idx = df.index[df.index.get_loc(idx) - 1]
                    gap_start = df.loc[prev_idx, "timestamp"]
                    gap_end = df.loc[idx, "timestamp"]
                    gap_duration = gap_end - gap_start
                    bars_missing = int(gap_duration / interval) - 1

                    report.gaps.append({
                        "start": gap_start.isoformat(),
                        "end": gap_end.isoformat(),
                        "bars_missing": bars_missing,
                    })

                    report.add_issue(QualityIssue(
                        issue_type=IssueType.GAP_DETECTED,
                        timestamp=gap_start,
                        field="timestamp",
                        value=bars_missing,
                        details=f"Gap from {gap_start} to {gap_end}",
                    ))

        # Calculate completeness score
        actual_bars = len(df)
        if report.expected_bars > 0:
            coverage = min(1.0, actual_bars / report.expected_bars)
        else:
            coverage = 1.0

        return coverage

    def _cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup of DataFrame."""
        if df.empty:
            return df

        # Remove any temporary columns
        temp_cols = [c for c in df.columns if c.startswith("_")]
        df = df.drop(columns=temp_cols, errors="ignore")

        # Ensure sorted
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df
