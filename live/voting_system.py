"""
Voting System for Agent Ensemble

Aggregates predictions from multiple RL agents to produce
a consensus trading signal with confidence score.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

import structlog

logger = structlog.get_logger()


class VotingMethod(str, Enum):
    """Available voting methods."""

    MAJORITY = "majority"  # Simple majority vote
    WEIGHTED = "weighted"  # Weighted by agent performance
    CONFIDENCE = "confidence"  # Based on action probabilities
    UNANIMOUS = "unanimous"  # All must agree


@dataclass
class VotingResult:
    """Result of voting process."""

    signal: int  # -1 (sell), 0 (hold), 1 (buy)
    confidence: float  # 0.0 to 1.0
    votes: Dict[int, int]  # Action -> count
    agent_signals: Dict[str, int]  # Agent ID -> signal
    reasoning: str


class VotingSystem:
    """
    Aggregates signals from multiple RL agents.

    Supports multiple voting methods and confidence thresholds.
    """

    def __init__(
        self,
        models: List[Any],
        agent_ids: Optional[List[str]] = None,
        method: VotingMethod = VotingMethod.WEIGHTED,
        min_confidence: float = 0.6,
        agent_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize voting system.

        Args:
            models: List of RL model objects.
            agent_ids: Optional list of agent IDs (for tracking).
            method: Voting method to use.
            min_confidence: Minimum confidence threshold.
            agent_weights: Optional weights per agent (by ID).
        """
        self.models = models
        self.agent_ids = agent_ids or [f"agent_{i}" for i in range(len(models))]
        self.method = method
        self.min_confidence = min_confidence
        self.agent_weights = agent_weights or {}

        # Default equal weights
        if not self.agent_weights:
            self.agent_weights = {
                aid: 1.0 for aid in self.agent_ids
            }

    def get_signal(
        self,
        observation: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Get aggregated trading signal.

        Args:
            observation: Current observation vector.

        Returns:
            Tuple of (signal, confidence) where signal is:
            -1 (sell), 0 (hold), 1 (buy)
        """
        result = self.vote(observation)
        return result.signal, result.confidence

    def vote(
        self,
        observation: np.ndarray,
    ) -> VotingResult:
        """
        Run voting process.

        Args:
            observation: Current observation vector.

        Returns:
            VotingResult with signal, confidence, and details.
        """
        # Get predictions from all agents
        agent_signals = self._collect_predictions(observation)

        # Apply voting method
        if self.method == VotingMethod.MAJORITY:
            result = self._majority_vote(agent_signals)
        elif self.method == VotingMethod.WEIGHTED:
            result = self._weighted_vote(agent_signals)
        elif self.method == VotingMethod.CONFIDENCE:
            result = self._confidence_vote(observation)
        elif self.method == VotingMethod.UNANIMOUS:
            result = self._unanimous_vote(agent_signals)
        else:
            result = self._majority_vote(agent_signals)

        return result

    def _collect_predictions(
        self,
        observation: np.ndarray,
    ) -> Dict[str, int]:
        """Collect predictions from all agents."""
        predictions = {}

        for model, agent_id in zip(self.models, self.agent_ids):
            try:
                action, _ = model.predict(observation, deterministic=True)
                # Map action to signal: 0=hold, 1=buy, 2=sell -> 0, 1, -1
                signal = self._action_to_signal(int(action))
                predictions[agent_id] = signal
            except Exception as e:
                logger.warning(f"Prediction failed for {agent_id}: {e}")

        return predictions

    def _action_to_signal(self, action: int) -> int:
        """Convert discrete action to signal."""
        # Assuming: 0=hold, 1=buy, 2=sell
        if action == 1:
            return 1  # Buy
        elif action == 2:
            return -1  # Sell
        else:
            return 0  # Hold

    def _majority_vote(
        self,
        agent_signals: Dict[str, int],
    ) -> VotingResult:
        """Simple majority voting."""
        votes = {-1: 0, 0: 0, 1: 0}

        for signal in agent_signals.values():
            votes[signal] = votes.get(signal, 0) + 1

        total = sum(votes.values())
        if total == 0:
            return VotingResult(
                signal=0,
                confidence=0.0,
                votes=votes,
                agent_signals=agent_signals,
                reasoning="No valid predictions",
            )

        # Find majority
        signal = max(votes, key=votes.get)
        confidence = votes[signal] / total

        return VotingResult(
            signal=signal,
            confidence=confidence,
            votes=votes,
            agent_signals=agent_signals,
            reasoning=f"Majority vote: {votes[signal]}/{total} for signal {signal}",
        )

    def _weighted_vote(
        self,
        agent_signals: Dict[str, int],
    ) -> VotingResult:
        """Weighted voting based on agent weights."""
        weighted_votes = {-1: 0.0, 0: 0.0, 1: 0.0}
        total_weight = 0.0

        for agent_id, signal in agent_signals.items():
            weight = self.agent_weights.get(agent_id, 1.0)
            weighted_votes[signal] += weight
            total_weight += weight

        if total_weight == 0:
            return VotingResult(
                signal=0,
                confidence=0.0,
                votes={-1: 0, 0: 0, 1: 0},
                agent_signals=agent_signals,
                reasoning="No weighted predictions",
            )

        # Find weighted majority
        signal = max(weighted_votes, key=weighted_votes.get)
        confidence = weighted_votes[signal] / total_weight

        # Convert to integer counts for display
        votes = {s: int(v) for s, v in weighted_votes.items()}

        return VotingResult(
            signal=signal,
            confidence=confidence,
            votes=votes,
            agent_signals=agent_signals,
            reasoning=f"Weighted vote: {weighted_votes[signal]:.2f}/{total_weight:.2f} for signal {signal}",
        )

    def _confidence_vote(
        self,
        observation: np.ndarray,
    ) -> VotingResult:
        """Vote based on action probabilities."""
        action_probs = {-1: 0.0, 0: 0.0, 1: 0.0}
        agent_signals = {}
        count = 0

        for model, agent_id in zip(self.models, self.agent_ids):
            try:
                # Get action distribution if available
                action, _ = model.predict(observation, deterministic=True)
                signal = self._action_to_signal(int(action))
                agent_signals[agent_id] = signal

                # Try to get action probabilities
                if hasattr(model, "policy"):
                    obs_tensor = model.policy.obs_to_tensor(observation)[0]
                    with model.policy.evaluate_actions(obs_tensor) as dist:
                        probs = dist.distribution.probs.cpu().numpy()[0]
                        # Add probabilities
                        for i, prob in enumerate(probs):
                            sig = self._action_to_signal(i)
                            action_probs[sig] += prob
                else:
                    # Fallback: add 1 for the selected action
                    action_probs[signal] += 1.0

                count += 1

            except Exception as e:
                logger.debug(f"Confidence extraction failed for {agent_id}: {e}")
                # Fallback to deterministic
                if agent_id in agent_signals:
                    action_probs[agent_signals[agent_id]] += 1.0
                    count += 1

        if count == 0:
            return VotingResult(
                signal=0,
                confidence=0.0,
                votes={-1: 0, 0: 0, 1: 0},
                agent_signals=agent_signals,
                reasoning="No confidence scores",
            )

        # Normalize
        total = sum(action_probs.values())
        for sig in action_probs:
            action_probs[sig] /= total

        signal = max(action_probs, key=action_probs.get)
        confidence = action_probs[signal]

        votes = {s: int(p * 100) for s, p in action_probs.items()}

        return VotingResult(
            signal=signal,
            confidence=confidence,
            votes=votes,
            agent_signals=agent_signals,
            reasoning=f"Confidence-weighted: {action_probs}",
        )

    def _unanimous_vote(
        self,
        agent_signals: Dict[str, int],
    ) -> VotingResult:
        """Unanimous voting (all must agree)."""
        if not agent_signals:
            return VotingResult(
                signal=0,
                confidence=0.0,
                votes={-1: 0, 0: 0, 1: 0},
                agent_signals=agent_signals,
                reasoning="No predictions",
            )

        signals = list(agent_signals.values())
        unique_signals = set(signals)

        if len(unique_signals) == 1:
            signal = signals[0]
            return VotingResult(
                signal=signal,
                confidence=1.0,
                votes={signal: len(signals)},
                agent_signals=agent_signals,
                reasoning=f"Unanimous: all {len(signals)} agents agree on {signal}",
            )
        else:
            # Not unanimous, default to hold
            votes = {-1: 0, 0: 0, 1: 0}
            for s in signals:
                votes[s] = votes.get(s, 0) + 1

            return VotingResult(
                signal=0,
                confidence=0.0,
                votes=votes,
                agent_signals=agent_signals,
                reasoning=f"Not unanimous: {votes}",
            )

    def update_weights(
        self,
        agent_id: str,
        weight: float,
    ) -> None:
        """Update weight for an agent."""
        self.agent_weights[agent_id] = weight
        logger.debug(f"Updated weight for {agent_id}: {weight}")

    def update_weights_from_performance(
        self,
        performance: Dict[str, float],
    ) -> None:
        """
        Update weights based on recent performance.

        Args:
            performance: Dict mapping agent_id to performance metric.
        """
        if not performance:
            return

        # Normalize performance to weights
        min_perf = min(performance.values())
        max_perf = max(performance.values())

        if max_perf == min_perf:
            # Equal weights
            for agent_id in performance:
                self.agent_weights[agent_id] = 1.0
        else:
            # Scale to 0.5-1.5 range
            for agent_id, perf in performance.items():
                normalized = (perf - min_perf) / (max_perf - min_perf)
                weight = 0.5 + normalized  # 0.5 to 1.5
                self.agent_weights[agent_id] = weight

        logger.info(f"Updated weights from performance: {len(performance)} agents")

    def get_statistics(self) -> Dict[str, Any]:
        """Get voting system statistics."""
        return {
            "num_models": len(self.models),
            "method": self.method.value,
            "min_confidence": self.min_confidence,
            "weights": self.agent_weights,
        }
