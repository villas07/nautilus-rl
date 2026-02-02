"""
Filter 2: Cross-Validation

Validates agents on unseen markets to test generalization:
- Test on correlated but different instruments
- Test on different market regimes
- Ensure consistent performance across conditions
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO

import structlog

logger = structlog.get_logger()


@dataclass
class CrossValCriteria:
    """Criteria for cross-validation filter."""

    min_sharpe_on_holdout: float = 0.5  # Lower bar for unseen markets
    max_sharpe_degradation: float = 0.5  # Max drop from training
    min_win_rate_on_holdout: float = 0.45
    max_drawdown_on_holdout: float = 0.20
    min_instruments_passed: int = 2  # Out of holdout set


# Instrument groupings for cross-validation
INSTRUMENT_GROUPS = {
    "us_equity_index": ["SPY", "QQQ", "IWM", "DIA"],
    "us_equity_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
    "crypto_major": ["BTCUSDT", "ETHUSDT"],
    "crypto_alt": ["SOLUSDT", "BNBUSDT", "XRPUSDT"],
    "forex_major": ["EUR/USD", "GBP/USD", "USD/JPY"],
}


def get_holdout_instruments(training_symbol: str) -> List[str]:
    """
    Get holdout instruments for a training symbol.

    Returns instruments from the same group that weren't used in training.
    """
    for group, instruments in INSTRUMENT_GROUPS.items():
        if training_symbol in instruments:
            return [i for i in instruments if i != training_symbol]

    # Default: return some related instruments
    return ["SPY", "QQQ"]


class CrossValidationFilter:
    """
    Filter 2: Cross-validation on unseen instruments.

    Tests if agent generalizes to related but unseen markets.
    """

    def __init__(
        self,
        criteria: Optional[CrossValCriteria] = None,
        models_dir: str = "/app/models",
        catalog_path: str = "/app/data/catalog",
    ):
        """
        Initialize filter.

        Args:
            criteria: Validation criteria.
            models_dir: Directory containing trained models.
            catalog_path: Path to data catalog.
        """
        self.criteria = criteria or CrossValCriteria()
        self.models_dir = Path(models_dir)
        self.catalog_path = catalog_path

    def validate(
        self,
        agent_id: str,
        training_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Validate agent on holdout instruments.

        Args:
            agent_id: Agent identifier.
            training_metrics: Metrics from training (for comparison).

        Returns:
            Validation results.
        """
        logger.info(f"Cross-validating {agent_id}")

        # Load model
        model = self._load_model(agent_id)
        if model is None:
            return {
                "agent_id": agent_id,
                "passed": False,
                "reason": "Model not found",
            }

        # Get agent config
        agent_config = self._load_agent_config(agent_id)
        training_symbol = agent_config.get("symbol", "SPY")
        venue = agent_config.get("venue", "IBKR")
        timeframe = agent_config.get("timeframe", "1h")

        # Get holdout instruments
        holdout_instruments = get_holdout_instruments(training_symbol)

        if not holdout_instruments:
            return {
                "agent_id": agent_id,
                "passed": True,
                "reason": "No holdout instruments available",
                "holdout_results": {},
            }

        # Test on each holdout instrument
        holdout_results = {}
        passed_count = 0

        for instrument in holdout_instruments:
            try:
                metrics = self._evaluate_on_instrument(
                    model, instrument, venue, timeframe
                )
                holdout_results[instrument] = metrics

                # Check if passed on this instrument
                instrument_passed = self._check_instrument_criteria(
                    metrics, training_metrics
                )

                if instrument_passed:
                    passed_count += 1

                logger.info(
                    f"  {instrument}: {'PASS' if instrument_passed else 'FAIL'}",
                    sharpe=metrics["sharpe_ratio"],
                )

            except Exception as e:
                logger.warning(f"  {instrument}: Error - {e}")
                holdout_results[instrument] = {"error": str(e)}

        # Overall pass/fail
        passed = passed_count >= self.criteria.min_instruments_passed

        return {
            "agent_id": agent_id,
            "passed": passed,
            "training_symbol": training_symbol,
            "holdout_instruments": holdout_instruments,
            "holdout_results": holdout_results,
            "passed_count": passed_count,
            "required_count": self.criteria.min_instruments_passed,
        }

    def _evaluate_on_instrument(
        self,
        model: Any,
        symbol: str,
        venue: str,
        timeframe: str,
    ) -> Dict[str, float]:
        """Run evaluation on a specific instrument."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from gym_env.nautilus_env import NautilusGymEnv, EnvConfig

        env_config = EnvConfig(
            symbol=symbol,
            venue=venue,
            timeframe=timeframe,
            catalog_path=self.catalog_path,
        )

        env = NautilusGymEnv(config=env_config)

        # Run episodes
        all_returns = []
        n_episodes = 5

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            all_returns.extend(env._returns)

        env.close()

        returns_arr = np.array(all_returns) if all_returns else np.array([0])

        # Calculate metrics
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            sharpe = np.sqrt(252 * 6) * returns_arr.mean() / returns_arr.std()
        else:
            sharpe = 0.0

        cumulative = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_dd = drawdowns.max() if len(drawdowns) > 0 else 0.0

        return {
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "total_return": float(cumulative[-1] - 1) if len(cumulative) > 0 else 0.0,
        }

    def _check_instrument_criteria(
        self,
        metrics: Dict[str, float],
        training_metrics: Optional[Dict[str, float]],
    ) -> bool:
        """Check if metrics pass criteria for an instrument."""
        # Check absolute thresholds
        if metrics.get("sharpe_ratio", 0) < self.criteria.min_sharpe_on_holdout:
            return False

        if metrics.get("max_drawdown", 1) > self.criteria.max_drawdown_on_holdout:
            return False

        # Check relative to training
        if training_metrics:
            training_sharpe = training_metrics.get("sharpe_ratio", 0)
            holdout_sharpe = metrics.get("sharpe_ratio", 0)

            if training_sharpe - holdout_sharpe > self.criteria.max_sharpe_degradation:
                return False

        return True

    def _load_model(self, agent_id: str) -> Optional[Any]:
        """Load trained model."""
        model_path = self.models_dir / agent_id / f"{agent_id}_final.zip"
        if not model_path.exists():
            model_path = self.models_dir / agent_id / "best" / "best_model.zip"

        if not model_path.exists():
            return None

        try:
            return PPO.load(str(model_path))
        except Exception as e:
            logger.error(f"Failed to load model {agent_id}: {e}")
            return None

    def _load_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """Load agent configuration."""
        import json

        config_path = self.models_dir / agent_id / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)

        return {"symbol": "SPY", "venue": "IBKR", "timeframe": "1h"}


def filter_cross_validation(
    agent_ids: List[str],
    training_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    models_dir: str = "/app/models",
) -> List[str]:
    """
    Convenience function to filter agents by cross-validation.

    Returns list of agent IDs that pass the filter.
    """
    filter_obj = CrossValidationFilter(models_dir=models_dir)
    passed = []

    for agent_id in agent_ids:
        metrics = training_metrics.get(agent_id) if training_metrics else None
        result = filter_obj.validate(agent_id, metrics)
        if result.get("passed", False):
            passed.append(agent_id)

    return passed
