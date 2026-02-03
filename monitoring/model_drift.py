"""
Model Drift Detection (R-013)

Detects drift in model performance and input features:
- Prediction drift (distribution shift)
- Performance drift (accuracy/returns degradation)
- Feature drift (input data distribution changes)
- Concept drift (relationship between features and target changes)

Reference: EVAL-002, governance/evaluations/EVAL-002_gaps_analysis.md
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

import numpy as np
from scipy import stats

import structlog

logger = structlog.get_logger()


class DriftType(str, Enum):
    """Types of drift detected."""

    PREDICTION = "prediction"
    PERFORMANCE = "performance"
    FEATURE = "feature"
    CONCEPT = "concept"


class DriftSeverity(str, Enum):
    """Severity of drift detection."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of a drift detection check."""

    drift_type: DriftType
    severity: DriftSeverity
    metric_name: str
    baseline_value: float
    current_value: float
    change_percent: float
    p_value: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_drifted(self) -> bool:
        """Check if drift is detected."""
        return self.severity != DriftSeverity.NONE


@dataclass
class DriftDetectorConfig:
    """Configuration for drift detection."""

    # Window sizes
    baseline_window: int = 1000  # Samples for baseline
    detection_window: int = 100  # Samples for detection

    # Thresholds
    p_value_threshold: float = 0.05  # Statistical significance
    performance_drop_threshold: float = 0.20  # 20% drop triggers alert
    prediction_shift_threshold: float = 0.15  # 15% distribution shift

    # Severity thresholds (percent change)
    low_threshold: float = 0.10
    medium_threshold: float = 0.20
    high_threshold: float = 0.30
    critical_threshold: float = 0.50

    # Check frequency
    check_interval: int = 100  # Check every N samples


class ModelDriftDetector:
    """
    Detects drift in model behavior and performance.

    Uses statistical tests to identify:
    - Distribution shifts in predictions
    - Performance degradation
    - Feature distribution changes
    """

    def __init__(
        self,
        model_id: str,
        config: Optional[DriftDetectorConfig] = None,
    ):
        """Initialize drift detector for a model."""
        self.model_id = model_id
        self.config = config or DriftDetectorConfig()

        # Data storage
        self._predictions: deque = deque(maxlen=self.config.baseline_window * 2)
        self._outcomes: deque = deque(maxlen=self.config.baseline_window * 2)
        self._features: Dict[str, deque] = {}
        self._returns: deque = deque(maxlen=self.config.baseline_window * 2)

        # Baseline statistics
        self._baseline_prediction_dist: Optional[np.ndarray] = None
        self._baseline_accuracy: Optional[float] = None
        self._baseline_return: Optional[float] = None
        self._baseline_feature_stats: Dict[str, Tuple[float, float]] = {}

        # State
        self._sample_count: int = 0
        self._baseline_established: bool = False
        self._drift_history: List[DriftResult] = []

    def record_prediction(
        self,
        prediction: float,
        features: Optional[Dict[str, float]] = None,
    ) -> None:
        """Record a model prediction."""
        self._predictions.append(prediction)
        self._sample_count += 1

        # Record features
        if features:
            for name, value in features.items():
                if name not in self._features:
                    self._features[name] = deque(
                        maxlen=self.config.baseline_window * 2
                    )
                self._features[name].append(value)

    def record_outcome(
        self,
        actual: float,
        realized_return: Optional[float] = None,
    ) -> None:
        """Record actual outcome for previous prediction."""
        self._outcomes.append(actual)

        if realized_return is not None:
            self._returns.append(realized_return)

    def establish_baseline(self) -> bool:
        """Establish baseline statistics from current data."""
        if len(self._predictions) < self.config.baseline_window:
            logger.warning(
                f"Not enough data for baseline: {len(self._predictions)}/{self.config.baseline_window}"
            )
            return False

        baseline_data = list(self._predictions)[-self.config.baseline_window:]

        # Prediction distribution
        self._baseline_prediction_dist = np.array(baseline_data)

        # Accuracy (if outcomes available)
        if len(self._outcomes) >= self.config.baseline_window:
            predictions = np.array(baseline_data)
            outcomes = np.array(
                list(self._outcomes)[-self.config.baseline_window:]
            )
            # For classification: signal agreement
            self._baseline_accuracy = np.mean(
                np.sign(predictions) == np.sign(outcomes)
            )

        # Returns
        if len(self._returns) >= self.config.baseline_window:
            self._baseline_return = np.mean(
                list(self._returns)[-self.config.baseline_window:]
            )

        # Feature statistics
        for name, values in self._features.items():
            if len(values) >= self.config.baseline_window:
                baseline_values = list(values)[-self.config.baseline_window:]
                self._baseline_feature_stats[name] = (
                    np.mean(baseline_values),
                    np.std(baseline_values),
                )

        self._baseline_established = True
        logger.info(
            f"Baseline established for model {self.model_id}",
            samples=self.config.baseline_window,
        )

        return True

    def check_drift(self) -> List[DriftResult]:
        """Run all drift detection checks."""
        if not self._baseline_established:
            if len(self._predictions) >= self.config.baseline_window:
                self.establish_baseline()
            return []

        results = []

        # Prediction drift
        pred_drift = self._check_prediction_drift()
        if pred_drift:
            results.append(pred_drift)

        # Performance drift
        perf_drift = self._check_performance_drift()
        if perf_drift:
            results.append(perf_drift)

        # Feature drift
        for name in self._features:
            feat_drift = self._check_feature_drift(name)
            if feat_drift:
                results.append(feat_drift)

        # Record drift history
        for result in results:
            if result.is_drifted:
                self._drift_history.append(result)

        return results

    def _check_prediction_drift(self) -> Optional[DriftResult]:
        """Check for drift in prediction distribution."""
        if len(self._predictions) < self.config.detection_window:
            return None

        if self._baseline_prediction_dist is None:
            return None

        recent = np.array(
            list(self._predictions)[-self.config.detection_window:]
        )

        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(
            self._baseline_prediction_dist,
            recent,
        )

        # Calculate distribution metrics
        baseline_mean = np.mean(self._baseline_prediction_dist)
        current_mean = np.mean(recent)

        change_percent = (
            abs(current_mean - baseline_mean) / abs(baseline_mean)
            if baseline_mean != 0 else 0
        )

        severity = self._get_severity(change_percent, p_value)

        return DriftResult(
            drift_type=DriftType.PREDICTION,
            severity=severity,
            metric_name="prediction_distribution",
            baseline_value=baseline_mean,
            current_value=current_mean,
            change_percent=change_percent,
            p_value=p_value,
            details={
                "ks_statistic": ks_stat,
                "baseline_std": float(np.std(self._baseline_prediction_dist)),
                "current_std": float(np.std(recent)),
            },
        )

    def _check_performance_drift(self) -> Optional[DriftResult]:
        """Check for drift in model performance."""
        if len(self._returns) < self.config.detection_window:
            return None

        if self._baseline_return is None:
            return None

        recent_returns = list(self._returns)[-self.config.detection_window:]
        current_return = np.mean(recent_returns)

        change_percent = (
            (current_return - self._baseline_return) / abs(self._baseline_return)
            if self._baseline_return != 0 else 0
        )

        # Performance drop is negative change
        is_degraded = change_percent < -self.config.performance_drop_threshold

        # T-test for significance
        baseline_returns = list(self._returns)[:self.config.baseline_window]
        if len(baseline_returns) >= 30 and len(recent_returns) >= 30:
            _, p_value = stats.ttest_ind(baseline_returns, recent_returns)
        else:
            p_value = None

        severity = (
            self._get_severity(abs(change_percent), p_value)
            if is_degraded else DriftSeverity.NONE
        )

        return DriftResult(
            drift_type=DriftType.PERFORMANCE,
            severity=severity,
            metric_name="average_return",
            baseline_value=self._baseline_return,
            current_value=current_return,
            change_percent=change_percent,
            p_value=p_value,
            details={
                "baseline_sharpe": self._calculate_sharpe(baseline_returns),
                "current_sharpe": self._calculate_sharpe(recent_returns),
            },
        )

    def _check_feature_drift(self, feature_name: str) -> Optional[DriftResult]:
        """Check for drift in a feature distribution."""
        if feature_name not in self._features:
            return None

        values = self._features[feature_name]
        if len(values) < self.config.detection_window:
            return None

        if feature_name not in self._baseline_feature_stats:
            return None

        baseline_mean, baseline_std = self._baseline_feature_stats[feature_name]

        recent = list(values)[-self.config.detection_window:]
        current_mean = np.mean(recent)
        current_std = np.std(recent)

        # Z-score of the mean shift
        if baseline_std > 0:
            z_score = abs(current_mean - baseline_mean) / baseline_std
        else:
            z_score = 0

        change_percent = (
            abs(current_mean - baseline_mean) / abs(baseline_mean)
            if baseline_mean != 0 else 0
        )

        # P-value from z-score
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        severity = self._get_severity(change_percent, p_value)

        return DriftResult(
            drift_type=DriftType.FEATURE,
            severity=severity,
            metric_name=feature_name,
            baseline_value=baseline_mean,
            current_value=current_mean,
            change_percent=change_percent,
            p_value=p_value,
            details={
                "baseline_std": baseline_std,
                "current_std": current_std,
                "z_score": z_score,
            },
        )

    def _get_severity(
        self,
        change_percent: float,
        p_value: Optional[float],
    ) -> DriftSeverity:
        """Determine severity based on change and significance."""
        # Check statistical significance
        if p_value is not None and p_value > self.config.p_value_threshold:
            return DriftSeverity.NONE

        if change_percent >= self.config.critical_threshold:
            return DriftSeverity.CRITICAL
        elif change_percent >= self.config.high_threshold:
            return DriftSeverity.HIGH
        elif change_percent >= self.config.medium_threshold:
            return DriftSeverity.MEDIUM
        elif change_percent >= self.config.low_threshold:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.NONE

    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns."""
        if not returns or len(returns) < 2:
            return 0.0

        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return 0.0

        return float(mean / std * np.sqrt(252))  # Annualized

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection status."""
        recent_drifts = [
            d for d in self._drift_history
            if d.timestamp > datetime.now() - timedelta(hours=24)
        ]

        return {
            "model_id": self.model_id,
            "baseline_established": self._baseline_established,
            "samples_recorded": self._sample_count,
            "recent_drifts": len(recent_drifts),
            "drift_by_type": {
                drift_type.value: sum(
                    1 for d in recent_drifts
                    if d.drift_type == drift_type
                )
                for drift_type in DriftType
            },
            "drift_by_severity": {
                severity.value: sum(
                    1 for d in recent_drifts
                    if d.severity == severity
                )
                for severity in DriftSeverity
                if severity != DriftSeverity.NONE
            },
        }

    def reset_baseline(self) -> None:
        """Reset baseline to current data."""
        self._baseline_established = False
        self.establish_baseline()
        logger.info(f"Baseline reset for model {self.model_id}")


class DriftMonitor:
    """
    Monitors multiple models for drift.

    Aggregates drift detection across all models
    and generates alerts when significant drift is detected.
    """

    def __init__(self, config: Optional[DriftDetectorConfig] = None):
        """Initialize drift monitor."""
        self.config = config or DriftDetectorConfig()
        self._detectors: Dict[str, ModelDriftDetector] = {}

    def register_model(self, model_id: str) -> ModelDriftDetector:
        """Register a model for drift monitoring."""
        detector = ModelDriftDetector(model_id, self.config)
        self._detectors[model_id] = detector
        return detector

    def get_detector(self, model_id: str) -> Optional[ModelDriftDetector]:
        """Get detector for a model."""
        return self._detectors.get(model_id)

    def check_all_models(self) -> Dict[str, List[DriftResult]]:
        """Check drift for all registered models."""
        results = {}

        for model_id, detector in self._detectors.items():
            drift_results = detector.check_drift()
            results[model_id] = drift_results

        return results

    def get_critical_alerts(self) -> List[Tuple[str, DriftResult]]:
        """Get all critical drift alerts."""
        alerts = []

        for model_id, detector in self._detectors.items():
            results = detector.check_drift()

            for result in results:
                if result.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                    alerts.append((model_id, result))

        return alerts

    def get_overall_summary(self) -> Dict[str, Any]:
        """Get overall drift monitoring summary."""
        summaries = {
            model_id: detector.get_drift_summary()
            for model_id, detector in self._detectors.items()
        }

        total_drifts = sum(
            s["recent_drifts"] for s in summaries.values()
        )

        critical_count = sum(
            s["drift_by_severity"].get("critical", 0)
            + s["drift_by_severity"].get("high", 0)
            for s in summaries.values()
        )

        return {
            "total_models": len(self._detectors),
            "total_recent_drifts": total_drifts,
            "critical_alerts": critical_count,
            "model_summaries": summaries,
        }


# Singleton instance
_drift_monitor: Optional[DriftMonitor] = None


def get_drift_monitor() -> DriftMonitor:
    """Get or create the drift monitor instance."""
    global _drift_monitor
    if _drift_monitor is None:
        _drift_monitor = DriftMonitor()
    return _drift_monitor
