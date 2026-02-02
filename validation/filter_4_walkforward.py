"""
Filter 4: Walk-Forward Validation

Tests agents on truly out-of-sample data:
- Training: 2015-2022
- Validation: 2023
- Test: 2024

Ensures agents haven't overfit to training data.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np

from stable_baselines3 import PPO

import structlog

logger = structlog.get_logger()


@dataclass
class WalkForwardCriteria:
    """Criteria for walk-forward validation."""

    # Test period thresholds
    min_sharpe_test: float = 1.0  # Lower than training to allow some degradation
    max_drawdown_test: float = 0.20
    min_return_test: float = 0.0  # At least break-even

    # Consistency checks
    max_sharpe_drop: float = 1.0  # Max drop from training to test
    min_correlation_train_test: float = 0.3  # Some consistency in behavior

    # Date ranges
    train_start: str = "2015-01-01"
    train_end: str = "2022-12-31"
    validation_start: str = "2023-01-01"
    validation_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2024-12-31"


class WalkForwardFilter:
    """
    Filter 4: Walk-forward out-of-sample testing.

    Tests on 2024 data that wasn't used in training.
    """

    def __init__(
        self,
        criteria: Optional[WalkForwardCriteria] = None,
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
        self.criteria = criteria or WalkForwardCriteria()
        self.models_dir = Path(models_dir)
        self.catalog_path = catalog_path

    def validate(
        self,
        agent_id: str,
        training_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Validate agent using walk-forward testing.

        Args:
            agent_id: Agent identifier.
            training_metrics: Metrics from training period.

        Returns:
            Validation results with test metrics.
        """
        logger.info(f"Walk-forward validating {agent_id}")

        model = self._load_model(agent_id)
        if model is None:
            return {
                "agent_id": agent_id,
                "passed": False,
                "reason": "Model not found",
            }

        agent_config = self._load_agent_config(agent_id)

        # Evaluate on each period
        results = {}

        for period in ["validation", "test"]:
            if period == "validation":
                start = self.criteria.validation_start
                end = self.criteria.validation_end
            else:
                start = self.criteria.test_start
                end = self.criteria.test_end

            try:
                metrics = self._evaluate_period(
                    model,
                    agent_config,
                    start,
                    end,
                )
                results[period] = metrics

                logger.info(
                    f"  {period}: Sharpe={metrics['sharpe_ratio']:.2f}, "
                    f"DD={metrics['max_drawdown']:.1%}, "
                    f"Return={metrics['total_return']:.1%}"
                )

            except Exception as e:
                logger.warning(f"  {period}: Error - {e}")
                results[period] = {"error": str(e)}

        # Check pass criteria
        passed, reasons = self._check_criteria(results, training_metrics)

        return {
            "agent_id": agent_id,
            "passed": passed,
            "reasons": reasons,
            "validation_metrics": results.get("validation", {}),
            "test_metrics": results.get("test", {}),
            "training_metrics": training_metrics,
        }

    def _evaluate_period(
        self,
        model: Any,
        agent_config: Dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> Dict[str, float]:
        """Evaluate model on a specific date range."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from gym_env.nautilus_env import NautilusGymEnv, EnvConfig
        from data.adapters.timescale_adapter import TimescaleAdapter

        # Load data for the specific period
        adapter = TimescaleAdapter()
        df = adapter.fetch_bars(
            symbol=agent_config.get("symbol", "SPY"),
            timeframe=agent_config.get("timeframe", "1h"),
            start_date=start_date,
            end_date=end_date,
        )
        adapter.close()

        if df.empty:
            raise ValueError(f"No data for period {start_date} to {end_date}")

        # Create environment with this data
        env_config = EnvConfig(
            symbol=agent_config.get("symbol", "SPY"),
            venue=agent_config.get("venue", "IBKR"),
            timeframe=agent_config.get("timeframe", "1h"),
            catalog_path=self.catalog_path,
            max_episode_steps=len(df) - 25,  # Leave room for lookback
        )

        env = NautilusGymEnv(config=env_config, data=df)

        # Run episodes
        all_returns = []
        n_episodes = 3  # Multiple runs for robustness

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            all_returns.extend(env._returns)

        env.close()

        # Calculate metrics
        returns_arr = np.array(all_returns) if all_returns else np.array([0])

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
            "num_observations": len(returns_arr),
        }

    def _check_criteria(
        self,
        results: Dict[str, Dict[str, float]],
        training_metrics: Optional[Dict[str, float]],
    ) -> tuple:
        """Check if results pass walk-forward criteria."""
        reasons = []

        test_metrics = results.get("test", {})
        if "error" in test_metrics:
            return False, ["Test period evaluation failed"]

        # Check absolute thresholds
        if test_metrics.get("sharpe_ratio", 0) < self.criteria.min_sharpe_test:
            reasons.append(
                f"Test Sharpe {test_metrics['sharpe_ratio']:.2f} < {self.criteria.min_sharpe_test}"
            )

        if test_metrics.get("max_drawdown", 1) > self.criteria.max_drawdown_test:
            reasons.append(
                f"Test DD {test_metrics['max_drawdown']:.1%} > {self.criteria.max_drawdown_test:.1%}"
            )

        if test_metrics.get("total_return", -1) < self.criteria.min_return_test:
            reasons.append(
                f"Test return {test_metrics['total_return']:.1%} < {self.criteria.min_return_test:.1%}"
            )

        # Check relative to training
        if training_metrics:
            training_sharpe = training_metrics.get("sharpe_ratio", 0)
            test_sharpe = test_metrics.get("sharpe_ratio", 0)

            if training_sharpe - test_sharpe > self.criteria.max_sharpe_drop:
                reasons.append(
                    f"Sharpe drop {training_sharpe:.2f} -> {test_sharpe:.2f} too large"
                )

        passed = len(reasons) == 0
        return passed, reasons

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

        return {"symbol": "SPY", "venue": "IBKR", "timeframe": "1h"}


def filter_walkforward(
    agent_ids: List[str],
    training_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    models_dir: str = "/app/models",
) -> List[str]:
    """
    Convenience function to filter agents with walk-forward validation.

    Returns list of agent IDs that pass.
    """
    filter_obj = WalkForwardFilter(models_dir=models_dir)
    passed = []

    for agent_id in agent_ids:
        metrics = training_metrics.get(agent_id) if training_metrics else None
        result = filter_obj.validate(agent_id, metrics)
        if result.get("passed", False):
            passed.append(agent_id)

    return passed
