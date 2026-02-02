"""
Filter 3: Diversity Filter

Ensures selected agents have diverse strategies:
- Maximum correlation between agents < 0.5
- Different entry/exit patterns
- Coverage of different market conditions

This prevents selecting multiple agents that make the same trades.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

import structlog

logger = structlog.get_logger()


@dataclass
class DiversityCriteria:
    """Criteria for diversity filter."""

    max_correlation: float = 0.5  # Max correlation between any two agents
    min_unique_ratio: float = 0.3  # Min ratio of unique trades
    max_agents_per_symbol: int = 10  # Max agents trading same symbol
    target_agents: int = 50  # Target number of diverse agents


class DiversityFilter:
    """
    Filter 3: Diversity-based selection.

    Selects a diverse subset of agents to avoid redundancy
    in the ensemble.
    """

    def __init__(
        self,
        criteria: Optional[DiversityCriteria] = None,
        models_dir: str = "/app/models",
        catalog_path: str = "/app/data/catalog",
    ):
        """
        Initialize filter.

        Args:
            criteria: Diversity criteria.
            models_dir: Directory containing trained models.
            catalog_path: Path to data catalog.
        """
        self.criteria = criteria or DiversityCriteria()
        self.models_dir = Path(models_dir)
        self.catalog_path = catalog_path

    def filter(
        self,
        agent_ids: List[str],
        agent_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[str]:
        """
        Select diverse subset of agents.

        Args:
            agent_ids: List of candidate agent IDs.
            agent_metrics: Pre-computed metrics (optional).

        Returns:
            List of selected diverse agent IDs.
        """
        if len(agent_ids) <= self.criteria.target_agents:
            return agent_ids

        logger.info(f"Filtering {len(agent_ids)} agents for diversity")

        # Get trade signals for all agents
        agent_signals = self._get_agent_signals(agent_ids)

        # Compute correlation matrix
        correlation_matrix = self._compute_correlation_matrix(agent_signals)

        # Select diverse agents using greedy selection
        selected = self._greedy_select(
            agent_ids,
            correlation_matrix,
            agent_metrics,
        )

        logger.info(f"Selected {len(selected)} diverse agents")
        return selected

    def _get_agent_signals(
        self,
        agent_ids: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Get trade signals for each agent over a common period.

        Returns dict mapping agent_id to signal array.
        """
        signals = {}

        for agent_id in agent_ids:
            try:
                signal = self._generate_signals(agent_id)
                if signal is not None and len(signal) > 0:
                    signals[agent_id] = signal
            except Exception as e:
                logger.warning(f"Failed to get signals for {agent_id}: {e}")

        return signals

    def _generate_signals(self, agent_id: str) -> Optional[np.ndarray]:
        """Generate trade signals for an agent."""
        model = self._load_model(agent_id)
        if model is None:
            return None

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from gym_env.nautilus_env import NautilusBacktestEnv, NautilusEnvConfig

        agent_config = self._load_agent_config(agent_id)
        symbol = agent_config.get("symbol", "SPY")
        venue = agent_config.get("venue", "NASDAQ")
        instrument_id = f"{symbol}.{venue}"

        env_config = NautilusEnvConfig(
            instrument_id=instrument_id,
            venue=venue,
            catalog_path=self.catalog_path,
        )

        env = NautilusBacktestEnv(config=env_config)

        # Run single episode to get signals
        obs, _ = env.reset(seed=42)  # Fixed seed for consistency
        signals = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            signals.append(int(action))
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        env.close()

        return np.array(signals)

    def _compute_correlation_matrix(
        self,
        agent_signals: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Compute correlation matrix between agent signals.

        Returns DataFrame with correlations.
        """
        agent_ids = list(agent_signals.keys())

        if not agent_ids:
            return pd.DataFrame()

        # Pad signals to same length
        max_len = max(len(s) for s in agent_signals.values())
        padded_signals = {}

        for agent_id, signal in agent_signals.items():
            if len(signal) < max_len:
                # Pad with zeros
                padded = np.zeros(max_len)
                padded[:len(signal)] = signal
                padded_signals[agent_id] = padded
            else:
                padded_signals[agent_id] = signal[:max_len]

        # Create DataFrame and compute correlation
        df = pd.DataFrame(padded_signals)
        corr_matrix = df.corr()

        return corr_matrix

    def _greedy_select(
        self,
        agent_ids: List[str],
        correlation_matrix: pd.DataFrame,
        agent_metrics: Optional[Dict[str, Dict[str, float]]],
    ) -> List[str]:
        """
        Greedily select diverse agents.

        Strategy:
        1. Sort agents by performance (Sharpe)
        2. Add best agent
        3. Add next best agent if correlation < threshold with all selected
        4. Repeat until target reached or no more candidates
        """
        if correlation_matrix.empty:
            return agent_ids[:self.criteria.target_agents]

        # Sort by performance
        if agent_metrics:
            sorted_agents = sorted(
                agent_ids,
                key=lambda x: agent_metrics.get(x, {}).get("sharpe_ratio", 0),
                reverse=True,
            )
        else:
            sorted_agents = agent_ids

        # Track symbol counts
        symbol_counts: Dict[str, int] = {}

        selected = []
        for agent_id in sorted_agents:
            if len(selected) >= self.criteria.target_agents:
                break

            if agent_id not in correlation_matrix.index:
                continue

            # Check symbol limit
            agent_config = self._load_agent_config(agent_id)
            symbol = agent_config.get("symbol", "UNKNOWN")

            if symbol_counts.get(symbol, 0) >= self.criteria.max_agents_per_symbol:
                continue

            # Check correlation with already selected agents
            too_correlated = False
            for selected_id in selected:
                if selected_id in correlation_matrix.columns:
                    corr = abs(correlation_matrix.loc[agent_id, selected_id])
                    if corr > self.criteria.max_correlation:
                        too_correlated = True
                        break

            if not too_correlated:
                selected.append(agent_id)
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                logger.debug(f"Selected {agent_id} (symbol: {symbol})")

        return selected

    def compute_ensemble_diversity(
        self,
        agent_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compute diversity metrics for an ensemble.

        Returns dict with:
        - avg_correlation: Average pairwise correlation
        - max_correlation: Maximum pairwise correlation
        - symbol_distribution: Distribution across symbols
        - unique_signal_ratio: Ratio of unique signals
        """
        agent_signals = self._get_agent_signals(agent_ids)
        corr_matrix = self._compute_correlation_matrix(agent_signals)

        if corr_matrix.empty:
            return {"error": "No signals computed"}

        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack().values

        # Symbol distribution
        symbol_counts: Dict[str, int] = {}
        for agent_id in agent_ids:
            config = self._load_agent_config(agent_id)
            symbol = config.get("symbol", "UNKNOWN")
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        return {
            "avg_correlation": float(np.mean(correlations)) if len(correlations) > 0 else 0,
            "max_correlation": float(np.max(correlations)) if len(correlations) > 0 else 0,
            "min_correlation": float(np.min(correlations)) if len(correlations) > 0 else 0,
            "symbol_distribution": symbol_counts,
            "num_agents": len(agent_ids),
        }

    def _load_model(self, agent_id: str) -> Optional[Any]:
        """Load trained model."""
        model_path = self.models_dir / agent_id / f"{agent_id}_final.zip"
        if not model_path.exists():
            model_path = self.models_dir / agent_id / "best" / "best_model.zip"

        if not model_path.exists():
            return None

        try:
            return PPO.load(str(model_path))
        except Exception:
            return None

    def _load_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """Load agent configuration."""
        import json

        config_path = self.models_dir / agent_id / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)

        return {"symbol": "SPY", "venue": "IBKR"}


def filter_diversity(
    agent_ids: List[str],
    agent_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    target_agents: int = 50,
    models_dir: str = "/app/models",
) -> List[str]:
    """
    Convenience function to filter agents for diversity.

    Returns list of diverse agent IDs.
    """
    criteria = DiversityCriteria(target_agents=target_agents)
    filter_obj = DiversityFilter(criteria=criteria, models_dir=models_dir)
    return filter_obj.filter(agent_ids, agent_metrics)
