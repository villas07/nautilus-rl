"""Tests for the voting system."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from live.voting_system import VotingSystem, VotingMethod, VotingResult


class MockModel:
    """Mock RL model for testing."""

    def __init__(self, fixed_action: int = 0):
        self.fixed_action = fixed_action

    def predict(self, obs, deterministic=True):
        return np.array(self.fixed_action), None


class TestVotingSystem:
    """Test cases for VotingSystem."""

    def test_majority_vote_buy(self):
        """Test majority vote with buy signals."""
        models = [
            MockModel(1),  # Buy
            MockModel(1),  # Buy
            MockModel(0),  # Hold
        ]

        vs = VotingSystem(models, method=VotingMethod.MAJORITY)
        obs = np.zeros(45, dtype=np.float32)
        result = vs.vote(obs)

        assert result.signal == 1  # Buy
        assert result.confidence > 0.5

    def test_majority_vote_sell(self):
        """Test majority vote with sell signals."""
        models = [
            MockModel(2),  # Sell
            MockModel(2),  # Sell
            MockModel(2),  # Sell
        ]

        vs = VotingSystem(models, method=VotingMethod.MAJORITY)
        obs = np.zeros(45, dtype=np.float32)
        result = vs.vote(obs)

        assert result.signal == -1  # Sell
        assert result.confidence == 1.0

    def test_weighted_vote(self):
        """Test weighted voting."""
        models = [
            MockModel(1),  # Buy
            MockModel(2),  # Sell
        ]

        agent_weights = {"agent_0": 2.0, "agent_1": 1.0}
        vs = VotingSystem(
            models,
            agent_ids=["agent_0", "agent_1"],
            method=VotingMethod.WEIGHTED,
            agent_weights=agent_weights,
        )

        obs = np.zeros(45, dtype=np.float32)
        result = vs.vote(obs)

        # Agent 0 has higher weight, so buy should win
        assert result.signal == 1

    def test_unanimous_vote_agreement(self):
        """Test unanimous voting with agreement."""
        models = [
            MockModel(1),
            MockModel(1),
            MockModel(1),
        ]

        vs = VotingSystem(models, method=VotingMethod.UNANIMOUS)
        obs = np.zeros(45, dtype=np.float32)
        result = vs.vote(obs)

        assert result.signal == 1
        assert result.confidence == 1.0

    def test_unanimous_vote_disagreement(self):
        """Test unanimous voting with disagreement."""
        models = [
            MockModel(1),
            MockModel(2),
            MockModel(1),
        ]

        vs = VotingSystem(models, method=VotingMethod.UNANIMOUS)
        obs = np.zeros(45, dtype=np.float32)
        result = vs.vote(obs)

        assert result.signal == 0  # Hold on disagreement
        assert result.confidence == 0.0

    def test_get_signal(self):
        """Test get_signal convenience method."""
        models = [MockModel(1), MockModel(1)]
        vs = VotingSystem(models)

        signal, confidence = vs.get_signal(np.zeros(45, dtype=np.float32))

        assert signal in [-1, 0, 1]
        assert 0.0 <= confidence <= 1.0

    def test_update_weights(self):
        """Test weight updates."""
        models = [MockModel(1)]
        vs = VotingSystem(models, agent_ids=["agent_0"])

        vs.update_weights("agent_0", 2.0)
        assert vs.agent_weights["agent_0"] == 2.0

    def test_update_weights_from_performance(self):
        """Test weight updates from performance metrics."""
        models = [MockModel(1), MockModel(1)]
        vs = VotingSystem(models, agent_ids=["agent_0", "agent_1"])

        performance = {
            "agent_0": 2.0,  # Better performance
            "agent_1": 1.0,
        }
        vs.update_weights_from_performance(performance)

        assert vs.agent_weights["agent_0"] > vs.agent_weights["agent_1"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
