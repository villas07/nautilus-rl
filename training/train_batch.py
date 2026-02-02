#!/usr/bin/env python3
"""
Batch Training Script

Trains multiple RL agents in parallel using the agents configuration file.

Usage:
    python train_batch.py --config configs/agents_500_pro.yaml
    python train_batch.py --parallel 4 --agents 0-99
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import yaml

import structlog

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_agent import TrainingConfig, train_single_agent

logger = structlog.get_logger()


@dataclass
class BatchConfig:
    """Configuration for batch training."""

    # Agent definitions
    agents: List[Dict[str, Any]]

    # Training defaults
    algorithm: str = "PPO"
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    reward_type: str = "sharpe"

    # Infrastructure
    max_parallel: int = 4
    output_dir: str = "/app/models"
    log_dir: str = "/app/logs"
    catalog_path: str = "/app/data/catalog"

    # MLflow
    use_mlflow: bool = True
    mlflow_experiment: str = "rl-agents-batch"

    @classmethod
    def from_yaml(cls, path: str) -> "BatchConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        agents = data.get("agents", [])
        defaults = data.get("defaults", {})

        return cls(
            agents=agents,
            algorithm=defaults.get("algorithm", "PPO"),
            total_timesteps=defaults.get("total_timesteps", 5_000_000),
            learning_rate=defaults.get("learning_rate", 3e-4),
            reward_type=defaults.get("reward_type", "sharpe"),
            max_parallel=defaults.get("max_parallel", 4),
            output_dir=defaults.get("output_dir", "/app/models"),
            log_dir=defaults.get("log_dir", "/app/logs"),
            catalog_path=defaults.get("catalog_path", "/app/data/catalog"),
            use_mlflow=defaults.get("use_mlflow", True),
            mlflow_experiment=defaults.get("mlflow_experiment", "rl-agents-batch"),
        )


class BatchTrainer:
    """
    Trains multiple agents in parallel.

    Features:
    - Parallel training using ProcessPoolExecutor
    - Resume from checkpoints
    - Progress tracking
    - Results aggregation
    """

    def __init__(self, config: BatchConfig):
        """Initialize batch trainer."""
        self.config = config
        self.results: List[Dict[str, Any]] = []

        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    def get_agent_config(self, agent_def: Dict[str, Any]) -> TrainingConfig:
        """Create TrainingConfig for an agent."""
        return TrainingConfig(
            agent_id=agent_def["agent_id"],
            symbol=agent_def["symbol"],
            venue=agent_def.get("venue", "IBKR"),
            timeframe=agent_def.get("timeframe", "1h"),
            algorithm=agent_def.get("algorithm", self.config.algorithm),
            total_timesteps=agent_def.get("total_timesteps", self.config.total_timesteps),
            learning_rate=agent_def.get("learning_rate", self.config.learning_rate),
            reward_type=agent_def.get("reward_type", self.config.reward_type),
            output_dir=self.config.output_dir,
            log_dir=self.config.log_dir,
            catalog_path=self.config.catalog_path,
            use_mlflow=self.config.use_mlflow,
            mlflow_experiment=self.config.mlflow_experiment,
            n_envs=agent_def.get("n_envs", 2),  # Reduce for parallel training
        )

    def train_agent_worker(self, agent_def: Dict[str, Any]) -> Dict[str, Any]:
        """Worker function for parallel training."""
        agent_id = agent_def["agent_id"]
        logger.info(f"Starting training for {agent_id}")

        try:
            config = self.get_agent_config(agent_def)
            results = train_single_agent(
                agent_id=config.agent_id,
                symbol=config.symbol,
                config=config,
            )
            results["status"] = "success"

        except Exception as e:
            logger.error(f"Training failed for {agent_id}: {e}")
            results = {
                "agent_id": agent_id,
                "status": "failed",
                "error": str(e),
            }

        return results

    def train_sequential(
        self,
        agents: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Train agents sequentially."""
        agents = agents or self.config.agents

        for i, agent_def in enumerate(agents):
            logger.info(f"Training agent {i+1}/{len(agents)}: {agent_def['agent_id']}")
            result = self.train_agent_worker(agent_def)
            self.results.append(result)

        return self.results

    def train_parallel(
        self,
        agents: Optional[List[Dict[str, Any]]] = None,
        max_workers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Train agents in parallel."""
        agents = agents or self.config.agents
        max_workers = max_workers or self.config.max_parallel

        logger.info(
            f"Starting parallel training: {len(agents)} agents, {max_workers} workers"
        )

        # Use spawn context for CUDA compatibility
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(self.train_agent_worker, agent_def): agent_def["agent_id"]
                for agent_def in agents
            }

            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    logger.info(
                        f"Completed {agent_id}: {result.get('status', 'unknown')}"
                    )
                except Exception as e:
                    logger.error(f"Worker failed for {agent_id}: {e}")
                    self.results.append({
                        "agent_id": agent_id,
                        "status": "failed",
                        "error": str(e),
                    })

        return self.results

    def save_results(self, path: Optional[str] = None) -> str:
        """Save training results to JSON."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(self.config.output_dir) / f"batch_results_{timestamp}.json"

        with open(path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_agents": len(self.results),
                "successful": len([r for r in self.results if r.get("status") == "success"]),
                "failed": len([r for r in self.results if r.get("status") == "failed"]),
                "results": self.results,
            }, f, indent=2)

        logger.info(f"Results saved to {path}")
        return str(path)

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        successful = [r for r in self.results if r.get("status") == "success"]
        failed = [r for r in self.results if r.get("status") == "failed"]

        summary = {
            "total": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.results) if self.results else 0,
        }

        if successful:
            rewards = [r.get("mean_reward", 0) for r in successful]
            summary["mean_reward"] = sum(rewards) / len(rewards)
            summary["best_agent"] = max(successful, key=lambda x: x.get("mean_reward", 0))["agent_id"]

        return summary


def train_batch(
    config_path: Optional[str] = None,
    agents: Optional[List[Dict[str, Any]]] = None,
    parallel: bool = True,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Convenience function for batch training.

    Args:
        config_path: Path to batch config YAML.
        agents: List of agent definitions (alternative to config).
        parallel: Use parallel training.
        max_workers: Number of parallel workers.

    Returns:
        List of training results.
    """
    if config_path:
        config = BatchConfig.from_yaml(config_path)
    elif agents:
        config = BatchConfig(agents=agents, max_parallel=max_workers)
    else:
        raise ValueError("Either config_path or agents must be provided")

    trainer = BatchTrainer(config)

    if parallel and len(config.agents) > 1:
        results = trainer.train_parallel(max_workers=max_workers)
    else:
        results = trainer.train_sequential()

    trainer.save_results()

    return results


def parse_agent_range(range_str: str, total: int) -> List[int]:
    """Parse agent range string (e.g., '0-99' or '50-150')."""
    if "-" in range_str:
        start, end = range_str.split("-")
        return list(range(int(start), min(int(end) + 1, total)))
    else:
        return [int(range_str)]


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train multiple RL agents")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/agents_500_pro.yaml",
        help="Path to batch config YAML",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--agents",
        type=str,
        help="Agent range to train (e.g., '0-99')",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Train sequentially instead of parallel",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = BatchConfig.from_yaml(str(config_path))
    else:
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    # Filter agents if range specified
    agents = config.agents
    if args.agents:
        indices = parse_agent_range(args.agents, len(agents))
        agents = [agents[i] for i in indices if i < len(agents)]

    logger.info(f"Training {len(agents)} agents")

    # Create trainer
    trainer = BatchTrainer(config)

    # Train
    if args.sequential:
        results = trainer.train_sequential(agents)
    else:
        results = trainer.train_parallel(agents, max_workers=args.parallel)

    # Save and print summary
    trainer.save_results()
    summary = trainer.get_summary()

    print("\n" + "=" * 50)
    print("BATCH TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Total: {summary['total']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")
    if "mean_reward" in summary:
        print(f"  Mean reward: {summary['mean_reward']:.2f}")
        print(f"  Best agent: {summary['best_agent']}")


if __name__ == "__main__":
    main()
