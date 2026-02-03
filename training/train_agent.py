#!/usr/bin/env python3
"""
Single Agent Training Script

Trains a single RL agent using Stable-Baselines3 and the NautilusGymEnv.

Usage:
    python train_agent.py --agent-id agent_001 --symbol SPY
    python train_agent.py --config configs/agent_config.yaml
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import argparse
import json

import numpy as np
import yaml

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import structlog

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gym_env import NautilusGymEnv, NautilusEnvConfig

logger = structlog.get_logger()


@dataclass
class TrainingConfig:
    """Configuration for agent training."""

    # Agent identification
    agent_id: str = "agent_001"
    symbol: str = "SPY"
    venue: str = "IBKR"
    timeframe: str = "1h"

    # Training parameters
    algorithm: str = "PPO"  # PPO, A2C, SAC
    total_timesteps: int = 5_000_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Environment parameters
    lookback_period: int = 20
    max_episode_steps: int = 1512  # ~1 year of hourly
    initial_capital: float = 100000.0
    reward_type: str = "sharpe"

    # Network architecture
    policy: str = "MlpPolicy"
    net_arch: list = field(default_factory=lambda: [256, 256])

    # Training infrastructure
    n_envs: int = 4
    n_eval_episodes: int = 10
    eval_freq: int = 50000
    save_freq: int = 100000
    log_interval: int = 100

    # Paths
    output_dir: str = "/app/models"
    log_dir: str = "/app/logs"
    catalog_path: str = "data/catalog_nautilus"

    # MLflow
    use_mlflow: bool = True
    mlflow_experiment: str = "rl-agents-training"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v if not isinstance(v, list) else list(v)
            for k, v in self.__dict__.items()
        }


class AgentTrainer:
    """
    Trains a single RL agent.

    Features:
    - Multiple algorithm support (PPO, A2C, SAC)
    - Vectorized environments
    - Checkpointing
    - Evaluation callbacks
    - MLflow integration
    """

    ALGORITHMS = {
        "PPO": PPO,
        "A2C": A2C,
        "SAC": SAC,
    }

    def __init__(self, config: TrainingConfig):
        """Initialize trainer."""
        self.config = config
        self.model = None
        self.env = None
        self.eval_env = None

        # Create directories
        self.model_dir = Path(config.output_dir) / config.agent_id
        self.log_dir = Path(config.log_dir) / config.agent_id
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # MLflow
        self.mlflow_run = None

    def setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        if not self.config.use_mlflow:
            return

        try:
            import mlflow

            mlflow.set_experiment(self.config.mlflow_experiment)
            self.mlflow_run = mlflow.start_run(
                run_name=f"{self.config.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            # Log parameters
            mlflow.log_params({
                "agent_id": self.config.agent_id,
                "symbol": self.config.symbol,
                "algorithm": self.config.algorithm,
                "total_timesteps": self.config.total_timesteps,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "n_epochs": self.config.n_epochs,
                "gamma": self.config.gamma,
                "reward_type": self.config.reward_type,
            })

            logger.info("MLflow tracking enabled", experiment=self.config.mlflow_experiment)

        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")

    def create_env(self) -> DummyVecEnv:
        """Create training environment."""
        # Map timeframe to bar_type
        bar_type_map = {
            "1m": "1-MINUTE-LAST",
            "5m": "5-MINUTE-LAST",
            "15m": "15-MINUTE-LAST",
            "1h": "1-HOUR-LAST",
            "4h": "4-HOUR-LAST",
            "1d": "1-DAY-LAST",
        }
        bar_type = bar_type_map.get(self.config.timeframe, "1-HOUR-LAST")

        env_config = NautilusEnvConfig(
            instrument_id=f"{self.config.symbol}.{self.config.venue}",
            venue=self.config.venue,
            bar_type=bar_type,
            catalog_path=self.config.catalog_path,
            lookback_period=self.config.lookback_period,
            max_episode_steps=self.config.max_episode_steps,
            initial_capital=self.config.initial_capital,
            reward_type=self.config.reward_type,
        )

        def make_env(rank: int):
            def _init():
                env = NautilusGymEnv(config=env_config)
                env = Monitor(env, str(self.log_dir / f"env_{rank}"))
                return env
            return _init

        # Create vectorized environment
        if self.config.n_envs > 1:
            env = SubprocVecEnv([make_env(i) for i in range(self.config.n_envs)])
        else:
            env = DummyVecEnv([make_env(0)])

        return env

    def create_eval_env(self) -> DummyVecEnv:
        """Create evaluation environment."""
        bar_type_map = {
            "1m": "1-MINUTE-LAST",
            "5m": "5-MINUTE-LAST",
            "15m": "15-MINUTE-LAST",
            "1h": "1-HOUR-LAST",
            "4h": "4-HOUR-LAST",
            "1d": "1-DAY-LAST",
        }
        bar_type = bar_type_map.get(self.config.timeframe, "1-HOUR-LAST")

        env_config = NautilusEnvConfig(
            instrument_id=f"{self.config.symbol}.{self.config.venue}",
            venue=self.config.venue,
            bar_type=bar_type,
            catalog_path=self.config.catalog_path,
            lookback_period=self.config.lookback_period,
            max_episode_steps=self.config.max_episode_steps,
            initial_capital=self.config.initial_capital,
            reward_type=self.config.reward_type,
        )

        env = NautilusGymEnv(config=env_config)
        env = Monitor(env, str(self.log_dir / "eval"))

        return DummyVecEnv([lambda: env])

    def create_model(self) -> Any:
        """Create the RL model."""
        AlgorithmClass = self.ALGORITHMS.get(self.config.algorithm)
        if AlgorithmClass is None:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        # Policy kwargs
        policy_kwargs = {
            "net_arch": self.config.net_arch,
        }

        # Create model
        if self.config.algorithm in ["PPO", "A2C"]:
            model = AlgorithmClass(
                policy=self.config.policy,
                env=self.env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs if self.config.algorithm == "PPO" else 1,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range if self.config.algorithm == "PPO" else None,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=str(self.log_dir / "tensorboard"),
            )
        else:  # SAC
            model = AlgorithmClass(
                policy=self.config.policy,
                env=self.env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=str(self.log_dir / "tensorboard"),
            )

        return model

    def create_callbacks(self) -> CallbackList:
        """Create training callbacks."""
        callbacks = []

        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.model_dir / "best"),
            log_path=str(self.log_dir),
            eval_freq=self.config.eval_freq // self.config.n_envs,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq // self.config.n_envs,
            save_path=str(self.model_dir / "checkpoints"),
            name_prefix=self.config.agent_id,
        )
        callbacks.append(checkpoint_callback)

        return CallbackList(callbacks)

    def train(self) -> Dict[str, Any]:
        """
        Run the training loop.

        Returns:
            Training results dictionary.
        """
        logger.info(
            "Starting training",
            agent_id=self.config.agent_id,
            symbol=self.config.symbol,
            algorithm=self.config.algorithm,
            timesteps=self.config.total_timesteps,
        )

        # Setup MLflow
        self.setup_mlflow()

        # Create environments
        self.env = self.create_env()
        self.eval_env = self.create_eval_env()

        # Create model
        self.model = self.create_model()

        # Create callbacks
        callbacks = self.create_callbacks()

        # Train
        start_time = datetime.now()

        try:
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                log_interval=self.config.log_interval,
                progress_bar=True,
            )
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        training_time = (datetime.now() - start_time).total_seconds()

        # Save final model
        final_model_path = self.model_dir / f"{self.config.agent_id}_final.zip"
        self.model.save(str(final_model_path))

        # Save config
        config_path = self.model_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Evaluate final model
        eval_results = self.evaluate()

        # Log to MLflow
        if self.config.use_mlflow and self.mlflow_run:
            try:
                import mlflow

                mlflow.log_metrics({
                    "training_time_seconds": training_time,
                    "final_mean_reward": eval_results["mean_reward"],
                    "final_std_reward": eval_results["std_reward"],
                })
                mlflow.log_artifact(str(final_model_path))
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        results = {
            "agent_id": self.config.agent_id,
            "model_path": str(final_model_path),
            "training_time": training_time,
            **eval_results,
        }

        logger.info("Training complete", **results)

        return results

    def evaluate(self, n_episodes: int = 20) -> Dict[str, float]:
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("No model to evaluate")

        rewards = []
        for _ in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward[0]

            rewards.append(episode_reward)

        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
        }

    def close(self) -> None:
        """Clean up resources."""
        if self.env:
            self.env.close()
        if self.eval_env:
            self.eval_env.close()


def train_single_agent(
    agent_id: str,
    symbol: str,
    config: Optional[TrainingConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to train a single agent.

    Args:
        agent_id: Unique agent identifier.
        symbol: Trading symbol.
        config: Training configuration (or create from kwargs).
        **kwargs: Override config parameters.

    Returns:
        Training results.
    """
    if config is None:
        config = TrainingConfig(agent_id=agent_id, symbol=symbol, **kwargs)
    else:
        config.agent_id = agent_id
        config.symbol = symbol
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    trainer = AgentTrainer(config)

    try:
        results = trainer.train()
    finally:
        trainer.close()

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train a single RL agent")

    parser.add_argument("--agent-id", type=str, default="agent_001")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--venue", type=str, default="IBKR")
    parser.add_argument("--config", type=str, help="Path to YAML config")
    parser.add_argument("--algorithm", type=str, default="PPO")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--catalog-path", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    output_dir = args.output_dir or str(project_root / "models")
    catalog_path = args.catalog_path or str(project_root / "data" / "catalog")
    log_dir = args.log_dir or str(project_root / "logs")

    if args.config:
        config = TrainingConfig.from_yaml(args.config)
        # Override paths from CLI if provided
        config.output_dir = output_dir
        config.catalog_path = catalog_path
        config.log_dir = log_dir
    else:
        config = TrainingConfig(
            agent_id=args.agent_id,
            symbol=args.symbol,
            venue=args.venue,
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            output_dir=output_dir,
            catalog_path=catalog_path,
            log_dir=log_dir,
        )

    results = train_single_agent(
        agent_id=config.agent_id,
        symbol=config.symbol,
        config=config,
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
