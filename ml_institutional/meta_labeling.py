"""
Meta-Labeling (R-009)

Implements meta-labeling from López de Prado's AFML:
- Primary model generates directional signals (buy/sell)
- Meta-model predicts if the signal will be profitable
- Combines signal quality with bet sizing

Benefits:
- Filters out low-quality signals
- Improves precision without sacrificing recall
- Enables probabilistic position sizing

Reference: EVAL-001, Advances in Financial Machine Learning Ch. 3
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

import structlog

logger = structlog.get_logger()


class SignalType(int, Enum):
    """Primary model signal types."""
    SELL = -1
    HOLD = 0
    BUY = 1


@dataclass
class MetaLabelConfig:
    """Configuration for meta-labeling."""

    # Primary model settings
    min_primary_confidence: float = 0.0  # Minimum confidence to consider

    # Meta-model settings
    n_estimators: int = 100
    max_depth: int = 5
    min_samples_leaf: int = 10
    class_weight: str = "balanced"

    # Validation
    cv_folds: int = 5
    min_accuracy: float = 0.52  # Must beat random

    # Bet sizing
    enable_bet_sizing: bool = True
    max_bet_size: float = 1.0
    min_bet_size: float = 0.1


@dataclass
class MetaLabel:
    """Result of meta-labeling a signal."""

    primary_signal: SignalType
    meta_prediction: int  # 1 = take signal, 0 = skip
    meta_probability: float  # Confidence in taking signal
    bet_size: float  # Position size multiplier (0-1)
    features_used: int


@dataclass
class MetaLabelingResult:
    """Results from meta-labeling training."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    cv_scores: List[float]
    feature_importance: Dict[str, float]
    classification_report: str


class PrimaryModelInterface:
    """
    Interface for primary models that generate signals.

    Can wrap RL agents, rule-based strategies, or ML models.
    """

    def predict(self, features: np.ndarray) -> Tuple[SignalType, float]:
        """
        Generate signal and confidence.

        Args:
            features: Feature vector

        Returns:
            Tuple of (signal, confidence)
        """
        raise NotImplementedError

    def predict_batch(
        self,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate signals for batch of features.

        Returns:
            Tuple of (signals array, confidences array)
        """
        raise NotImplementedError


class RLAgentWrapper(PrimaryModelInterface):
    """Wraps an RL agent as a primary model."""

    def __init__(self, model_path: str):
        """Load RL model."""
        from stable_baselines3 import PPO

        self.model = PPO.load(model_path)

    def predict(self, features: np.ndarray) -> Tuple[SignalType, float]:
        """Get signal from RL agent."""
        action, _ = self.model.predict(features, deterministic=True)

        # Convert action to signal
        if action == 0:
            signal = SignalType.SELL
        elif action == 1:
            signal = SignalType.HOLD
        else:
            signal = SignalType.BUY

        # RL doesn't provide confidence directly, use 0.5
        return signal, 0.5

    def predict_batch(
        self,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction."""
        signals = []
        confidences = []

        for i in range(len(features)):
            signal, conf = self.predict(features[i])
            signals.append(signal.value)
            confidences.append(conf)

        return np.array(signals), np.array(confidences)


class VotingSystemWrapper(PrimaryModelInterface):
    """Wraps voting system as primary model."""

    def __init__(self, voting_system):
        """Initialize with voting system."""
        self.voting = voting_system

    def predict(self, features: np.ndarray) -> Tuple[SignalType, float]:
        """Get signal from voting system."""
        result = self.voting.get_signal(features)

        signal = SignalType(result.get("signal", 0))
        confidence = result.get("confidence", 0.5)

        return signal, confidence

    def predict_batch(
        self,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction."""
        signals = []
        confidences = []

        for i in range(len(features)):
            signal, conf = self.predict(features[i])
            signals.append(signal.value)
            confidences.append(conf)

        return np.array(signals), np.array(confidences)


class MetaLabeler:
    """
    Meta-labeling system for signal filtering and bet sizing.

    Workflow:
    1. Primary model generates signals (buy/sell/hold)
    2. Meta-model predicts if signal will be profitable
    3. Bet size determined by meta-model confidence

    The meta-model is trained on historical signals and their outcomes.
    """

    def __init__(self, config: Optional[MetaLabelConfig] = None):
        """Initialize meta-labeler."""
        self.config = config or MetaLabelConfig()
        self._meta_model = None
        self._feature_names: List[str] = []
        self._is_trained = False
        self._training_result: Optional[MetaLabelingResult] = None

    def create_meta_labels(
        self,
        primary_signals: np.ndarray,
        returns: np.ndarray,
        signal_returns: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Create meta-labels from primary signals and outcomes.

        Meta-label = 1 if signal direction matches actual return
        Meta-label = 0 if signal was wrong

        Args:
            primary_signals: Array of signals (-1, 0, 1)
            returns: Array of actual returns
            signal_returns: Optional pre-computed signal returns

        Returns:
            Array of meta-labels (0 or 1)
        """
        if signal_returns is None:
            # Calculate signal return: signal * actual_return
            signal_returns = primary_signals * returns

        # Meta-label = 1 if signal was profitable (or hold with small return)
        meta_labels = np.where(
            signal_returns > 0,
            1,  # Profitable signal
            np.where(
                primary_signals == 0,
                1,  # Hold is always "correct" if no signal
                0,  # Wrong signal
            )
        )

        return meta_labels

    def train(
        self,
        features: np.ndarray,
        primary_signals: np.ndarray,
        returns: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> MetaLabelingResult:
        """
        Train the meta-model.

        Args:
            features: Feature matrix (n_samples, n_features)
            primary_signals: Primary model signals
            returns: Actual returns for each sample
            feature_names: Optional feature names

        Returns:
            Training results
        """
        logger.info(f"Training meta-model on {len(features)} samples")

        # Create meta-labels
        meta_labels = self.create_meta_labels(primary_signals, returns)

        # Filter to only samples with signals (not hold)
        signal_mask = primary_signals != 0
        X = features[signal_mask]
        y = meta_labels[signal_mask]

        if len(X) < 100:
            logger.warning(f"Only {len(X)} signal samples, may not train well")

        # Store feature names
        self._feature_names = feature_names or [
            f"feature_{i}" for i in range(features.shape[1])
        ]

        # Create and train meta-model
        self._meta_model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            class_weight=self.config.class_weight,
            random_state=42,
            n_jobs=-1,
        )

        # Cross-validation
        cv_scores = cross_val_score(
            self._meta_model,
            X, y,
            cv=self.config.cv_folds,
            scoring="accuracy",
        )

        # Train final model
        self._meta_model.fit(X, y)

        # Evaluate
        y_pred = self._meta_model.predict(X)
        y_prob = self._meta_model.predict_proba(X)[:, 1]

        # Metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        try:
            roc_auc = roc_auc_score(y, y_prob)
        except ValueError:
            roc_auc = None

        # Feature importance
        importance = dict(zip(
            self._feature_names,
            self._meta_model.feature_importances_,
        ))
        importance = dict(sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        ))

        # Classification report
        report = classification_report(y, y_pred, target_names=["Skip", "Take"])

        self._training_result = MetaLabelingResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            cv_scores=cv_scores.tolist(),
            feature_importance=importance,
            classification_report=report,
        )

        self._is_trained = True

        logger.info(
            f"Meta-model trained: accuracy={accuracy:.3f}, "
            f"precision={precision:.3f}, recall={recall:.3f}"
        )

        return self._training_result

    def predict(
        self,
        features: np.ndarray,
        primary_signal: SignalType,
        primary_confidence: float = 0.5,
    ) -> MetaLabel:
        """
        Apply meta-labeling to a signal.

        Args:
            features: Feature vector for current state
            primary_signal: Signal from primary model
            primary_confidence: Confidence from primary model

        Returns:
            MetaLabel with prediction and bet size
        """
        if not self._is_trained:
            raise RuntimeError("Meta-model not trained. Call train() first.")

        # If hold signal, don't need meta-label
        if primary_signal == SignalType.HOLD:
            return MetaLabel(
                primary_signal=primary_signal,
                meta_prediction=1,
                meta_probability=1.0,
                bet_size=0.0,
                features_used=len(features),
            )

        # Get meta-model prediction
        features_2d = features.reshape(1, -1)
        meta_pred = self._meta_model.predict(features_2d)[0]
        meta_prob = self._meta_model.predict_proba(features_2d)[0, 1]

        # Calculate bet size
        if self.config.enable_bet_sizing:
            bet_size = self._calculate_bet_size(meta_prob, primary_confidence)
        else:
            bet_size = 1.0 if meta_pred == 1 else 0.0

        return MetaLabel(
            primary_signal=primary_signal,
            meta_prediction=meta_pred,
            meta_probability=meta_prob,
            bet_size=bet_size,
            features_used=len(features),
        )

    def predict_batch(
        self,
        features: np.ndarray,
        primary_signals: np.ndarray,
        primary_confidences: Optional[np.ndarray] = None,
    ) -> List[MetaLabel]:
        """
        Apply meta-labeling to batch of signals.

        Args:
            features: Feature matrix
            primary_signals: Array of primary signals
            primary_confidences: Optional array of confidences

        Returns:
            List of MetaLabels
        """
        if primary_confidences is None:
            primary_confidences = np.full(len(primary_signals), 0.5)

        results = []
        for i in range(len(features)):
            result = self.predict(
                features[i],
                SignalType(primary_signals[i]),
                primary_confidences[i],
            )
            results.append(result)

        return results

    def _calculate_bet_size(
        self,
        meta_probability: float,
        primary_confidence: float,
    ) -> float:
        """
        Calculate bet size from meta-probability and primary confidence.

        Uses geometric mean of both confidences, scaled to [min, max] range.
        """
        # Combine confidences
        combined = np.sqrt(meta_probability * primary_confidence)

        # Scale to bet size range
        if meta_probability < 0.5:
            return 0.0  # Don't bet if meta-model says skip

        bet_size = (
            self.config.min_bet_size +
            (self.config.max_bet_size - self.config.min_bet_size) *
            (combined - 0.5) * 2  # Scale 0.5-1.0 to 0-1
        )

        return np.clip(bet_size, self.config.min_bet_size, self.config.max_bet_size)

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Get top N most important features."""
        if self._training_result is None:
            return {}

        importance = self._training_result.feature_importance
        return dict(list(importance.items())[:top_n])

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if self._training_result is None:
            return {"trained": False}

        return {
            "trained": True,
            "accuracy": self._training_result.accuracy,
            "precision": self._training_result.precision,
            "recall": self._training_result.recall,
            "f1": self._training_result.f1,
            "roc_auc": self._training_result.roc_auc,
            "cv_mean": np.mean(self._training_result.cv_scores),
            "cv_std": np.std(self._training_result.cv_scores),
            "top_features": list(self._training_result.feature_importance.keys())[:5],
        }

    def save(self, path: str) -> None:
        """Save trained meta-model."""
        import joblib

        if not self._is_trained:
            raise RuntimeError("No trained model to save")

        data = {
            "model": self._meta_model,
            "feature_names": self._feature_names,
            "config": self.config,
            "training_result": self._training_result,
        }

        joblib.dump(data, path)
        logger.info(f"Meta-model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MetaLabeler":
        """Load trained meta-model."""
        import joblib

        data = joblib.load(path)

        instance = cls(config=data["config"])
        instance._meta_model = data["model"]
        instance._feature_names = data["feature_names"]
        instance._training_result = data["training_result"]
        instance._is_trained = True

        logger.info(f"Meta-model loaded from {path}")
        return instance


class MetaLabelingPipeline:
    """
    Complete pipeline for meta-labeling with primary model.

    Combines primary model signals with meta-labeling
    for improved signal quality.
    """

    def __init__(
        self,
        primary_model: PrimaryModelInterface,
        meta_config: Optional[MetaLabelConfig] = None,
    ):
        """Initialize pipeline."""
        self.primary_model = primary_model
        self.meta_labeler = MetaLabeler(meta_config)
        self._is_trained = False

    def train_meta_model(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> MetaLabelingResult:
        """
        Train meta-model using primary model signals.

        Args:
            features: Feature matrix
            returns: Actual returns
            feature_names: Optional feature names

        Returns:
            Training result
        """
        # Get primary signals
        signals, confidences = self.primary_model.predict_batch(features)

        # Train meta-model
        result = self.meta_labeler.train(
            features=features,
            primary_signals=signals,
            returns=returns,
            feature_names=feature_names,
        )

        self._is_trained = True
        return result

    def get_signal(
        self,
        features: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Get filtered signal with bet sizing.

        Args:
            features: Current feature vector

        Returns:
            Dict with signal, confidence, and bet_size
        """
        # Get primary signal
        primary_signal, primary_conf = self.primary_model.predict(features)

        # Apply meta-labeling
        if self._is_trained:
            meta_result = self.meta_labeler.predict(
                features,
                primary_signal,
                primary_conf,
            )

            # Filter signal based on meta-prediction
            if meta_result.meta_prediction == 0:
                final_signal = SignalType.HOLD
                bet_size = 0.0
            else:
                final_signal = primary_signal
                bet_size = meta_result.bet_size

            return {
                "signal": final_signal.value,
                "primary_signal": primary_signal.value,
                "primary_confidence": primary_conf,
                "meta_prediction": meta_result.meta_prediction,
                "meta_probability": meta_result.meta_probability,
                "bet_size": bet_size,
                "filtered": meta_result.meta_prediction == 0,
            }

        else:
            # No meta-model, return primary signal directly
            return {
                "signal": primary_signal.value,
                "primary_signal": primary_signal.value,
                "primary_confidence": primary_conf,
                "meta_prediction": 1,
                "meta_probability": 1.0,
                "bet_size": 1.0,
                "filtered": False,
            }

    def evaluate_filtering(
        self,
        features: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate how well meta-labeling filters signals.

        Compares performance with and without meta-filtering.
        """
        # Get signals without meta-filtering
        signals_no_meta, _ = self.primary_model.predict_batch(features)
        returns_no_meta = signals_no_meta * returns

        # Get signals with meta-filtering
        signals_meta = []
        bet_sizes = []

        for i in range(len(features)):
            result = self.get_signal(features[i])
            signals_meta.append(result["signal"])
            bet_sizes.append(result["bet_size"])

        signals_meta = np.array(signals_meta)
        bet_sizes = np.array(bet_sizes)
        returns_meta = signals_meta * returns * bet_sizes

        # Calculate metrics
        def calc_metrics(rets):
            total_return = np.sum(rets)
            sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
            win_rate = np.mean(rets > 0) if len(rets[rets != 0]) > 0 else 0
            n_trades = np.sum(rets != 0)
            return {
                "total_return": float(total_return),
                "sharpe": float(sharpe),
                "win_rate": float(win_rate),
                "n_trades": int(n_trades),
            }

        no_meta_metrics = calc_metrics(returns_no_meta)
        meta_metrics = calc_metrics(returns_meta)

        # Calculate improvement
        improvement = {
            "return_improvement": meta_metrics["total_return"] - no_meta_metrics["total_return"],
            "sharpe_improvement": meta_metrics["sharpe"] - no_meta_metrics["sharpe"],
            "trades_filtered": no_meta_metrics["n_trades"] - meta_metrics["n_trades"],
            "filter_rate": 1 - (meta_metrics["n_trades"] / no_meta_metrics["n_trades"])
            if no_meta_metrics["n_trades"] > 0 else 0,
        }

        return {
            "without_meta": no_meta_metrics,
            "with_meta": meta_metrics,
            "improvement": improvement,
        }


def create_meta_labels_from_triple_barrier(
    signals: np.ndarray,
    tb_labels: np.ndarray,
) -> np.ndarray:
    """
    Create meta-labels from Triple Barrier labels.

    TB label = 1 (hit TP), -1 (hit SL), 0 (timeout)
    Meta-label = 1 if signal matches TB outcome

    Args:
        signals: Primary model signals (-1, 0, 1)
        tb_labels: Triple Barrier labels (-1, 0, 1)

    Returns:
        Meta-labels (0 or 1)
    """
    # Signal matches TB if both positive or both negative
    meta_labels = np.where(
        signals == 0,
        1,  # Hold signals are always "correct"
        np.where(
            signals * tb_labels > 0,
            1,  # Signal direction matched outcome
            0,  # Signal was wrong
        )
    )

    return meta_labels
