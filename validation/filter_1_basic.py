"""
Filter 1: Basic Metrics Validation

Validates agents against minimum performance thresholds:
- Sharpe Ratio > 1.5
- Max Drawdown < 15%
- Win Rate > 50%
- Profit Factor > 1.5
- Number of trades > 50
"""

__version__ = "1.0.0"

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO

import structlog

logger = structlog.get_logger()


@dataclass
class BasicMetricsCriteria:
    """Criteria for basic metrics filter."""

    min_sharpe: float = 1.5
    max_drawdown: float = 0.15
    min_win_rate: float = 0.50
    min_profit_factor: float = 1.5
    min_trades: int = 50
    min_total_return: float = 0.0


@dataclass
class ValidationResult:
    """Result of validation for a single agent."""

    agent_id: str
    passed: bool
    metrics: Dict[str, float]
    failures: List[str]


class BasicMetricsFilter:
    """
    Filter 1: Basic metrics validation.

    Runs backtest on training data and validates against
    minimum performance thresholds.
    """

    def __init__(
        self,
        criteria: Optional[BasicMetricsCriteria] = None,
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
        self.criteria = criteria or BasicMetricsCriteria()
        self.models_dir = Path(models_dir)
        self.catalog_path = catalog_path

    def validate(self, agent_id: str) -> ValidationResult:
        """
        Validate a single agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            ValidationResult with pass/fail and metrics.
        """
        logger.info(f"Validating {agent_id} with basic metrics")

        # Load model
        model = self._load_model(agent_id)
        if model is None:
            return ValidationResult(
                agent_id=agent_id,
                passed=False,
                metrics={},
                failures=["Model not found"],
            )

        # Run backtest
        metrics = self._run_backtest(agent_id, model)

        # Check criteria
        failures = []

        if metrics["sharpe_ratio"] < self.criteria.min_sharpe:
            failures.append(
                f"Sharpe {metrics['sharpe_ratio']:.2f} < {self.criteria.min_sharpe}"
            )

        if metrics["max_drawdown"] > self.criteria.max_drawdown:
            failures.append(
                f"Drawdown {metrics['max_drawdown']:.1%} > {self.criteria.max_drawdown:.1%}"
            )

        if metrics["win_rate"] < self.criteria.min_win_rate:
            failures.append(
                f"Win rate {metrics['win_rate']:.1%} < {self.criteria.min_win_rate:.1%}"
            )

        if metrics["profit_factor"] < self.criteria.min_profit_factor:
            failures.append(
                f"Profit factor {metrics['profit_factor']:.2f} < {self.criteria.min_profit_factor}"
            )

        if metrics["num_trades"] < self.criteria.min_trades:
            failures.append(
                f"Trades {metrics['num_trades']} < {self.criteria.min_trades}"
            )

        if metrics["total_return"] < self.criteria.min_total_return:
            failures.append(
                f"Return {metrics['total_return']:.1%} < {self.criteria.min_total_return:.1%}"
            )

        passed = len(failures) == 0

        logger.info(
            f"Agent {agent_id}: {'PASSED' if passed else 'FAILED'}",
            sharpe=metrics["sharpe_ratio"],
            drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
        )

        return ValidationResult(
            agent_id=agent_id,
            passed=passed,
            metrics=metrics,
            failures=failures,
        )

    def validate_batch(self, agent_ids: List[str]) -> List[ValidationResult]:
        """Validate multiple agents."""
        results = []
        for agent_id in agent_ids:
            result = self.validate(agent_id)
            results.append(result)
        return results

    def _load_model(self, agent_id: str) -> Optional[Any]:
        """Load trained model."""
        model_path = self.models_dir / agent_id / f"{agent_id}_final.zip"
        if not model_path.exists():
            model_path = self.models_dir / agent_id / "best" / "best_model.zip"

        if not model_path.exists():
            logger.warning(f"Model not found: {agent_id}")
            return None

        try:
            return PPO.load(str(model_path))
        except Exception as e:
            logger.error(f"Failed to load model {agent_id}: {e}")
            return None

    def _run_backtest(self, agent_id: str, model: Any) -> Dict[str, float]:
        """
        Run backtest and calculate metrics.

        Returns dict with:
        - sharpe_ratio
        - max_drawdown
        - win_rate
        - profit_factor
        - num_trades
        - total_return
        """
        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from gym_env.nautilus_env import NautilusBacktestEnv, NautilusEnvConfig

        # Get agent config to determine symbol/venue
        agent_config = self._load_agent_config(agent_id)

        # Build instrument_id from symbol and venue
        symbol = agent_config.get("symbol", "SPY")
        venue = agent_config.get("venue", "NASDAQ")
        instrument_id = f"{symbol}.{venue}"

        # Create environment
        env_config = NautilusEnvConfig(
            instrument_id=instrument_id,
            venue=venue,
            catalog_path=self.catalog_path,
        )

        env = NautilusBacktestEnv(config=env_config)

        # Run multiple episodes
        all_returns = []
        all_trades = []
        episode_returns = []

        n_episodes = 10
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_return = 0
            trades = []
            prev_position = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_return += reward

                # Track trades
                current_position = info.get("position", 0)
                if current_position != prev_position:
                    trades.append({
                        "step": info.get("step", 0),
                        "pnl": reward,
                    })
                prev_position = current_position

            all_returns.extend(env._returns)
            all_trades.extend(trades)
            episode_returns.append(info.get("total_return", 0))

        env.close()

        # Calculate metrics
        returns_arr = np.array(all_returns) if all_returns else np.array([0])

        # Sharpe ratio
        if len(returns_arr) > 1 and returns_arr.std() > 0:
            sharpe = np.sqrt(252 * 6) * returns_arr.mean() / returns_arr.std()
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0

        # Win rate and profit factor
        if all_trades:
            trade_pnls = [t["pnl"] for t in all_trades]
            wins = [p for p in trade_pnls if p > 0]
            losses = [p for p in trade_pnls if p < 0]

            win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        else:
            win_rate = 0.0
            profit_factor = 0.0

        return {
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "num_trades": len(all_trades),
            "total_return": float(np.mean(episode_returns)),
            "mean_episode_return": float(np.mean(episode_returns)),
            "std_episode_return": float(np.std(episode_returns)),
        }

    def _load_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """Load agent configuration."""
        import json

        config_path = self.models_dir / agent_id / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)

        # Default config
        return {
            "symbol": "SPY",
            "venue": "IBKR",
            "timeframe": "1h",
        }


def filter_basic_metrics(
    agent_ids: List[str],
    criteria: Optional[BasicMetricsCriteria] = None,
    models_dir: str = "/app/models",
) -> List[str]:
    """
    Convenience function to filter agents by basic metrics.

    Returns list of agent IDs that pass the filter.
    """
    filter_obj = BasicMetricsFilter(criteria=criteria, models_dir=models_dir)
    results = filter_obj.validate_batch(agent_ids)
    return [r.agent_id for r in results if r.passed]
