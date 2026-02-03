"""
Regime-Based Training Pipeline (R-019)

Trains specialized RL agents for each market regime:
- BULL_LOW_VOL: Momentum strategies
- BULL_HIGH_VOL: Conservative momentum
- BEAR_LOW_VOL: Mean reversion, short
- BEAR_HIGH_VOL: Defensive
- SIDEWAYS_LOW_VOL: Range trading
- SIDEWAYS_HIGH_VOL: Reduce exposure

Reference: EVAL-003, governance/evaluations/EVAL-003_regime_detection.md
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
import json

import numpy as np
import pandas as pd

import structlog

logger = structlog.get_logger()

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_institutional.regime_detector import (
    RegimeDetector,
    RegimeDetectorConfig,
    MarketRegime,
    RegimeState,
)


@dataclass
class RegimeTrainingConfig:
    """Configuration for regime-based training."""

    # Regime detection
    regime_config: Optional[RegimeDetectorConfig] = None

    # Training params per regime
    timesteps_per_agent: int = 1_000_000
    agents_per_regime: int = 10  # 10 agents × 6 regimes = 60 total

    # Data requirements
    min_bars_per_regime: int = 500  # Minimum bars to train
    train_test_split: float = 0.8

    # Output
    models_dir: str = "models/regime_specialized/"
    reports_dir: str = "reports/regime_training/"

    # Regime-specific hyperparameters
    regime_hyperparams: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "bull_low_vol": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "ent_coef": 0.01,  # Low entropy for decisive momentum
        },
        "bull_high_vol": {
            "learning_rate": 1e-4,  # Lower LR for stability
            "gamma": 0.995,
            "ent_coef": 0.02,
        },
        "bear_low_vol": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "ent_coef": 0.01,
        },
        "bear_high_vol": {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "ent_coef": 0.05,  # Higher entropy for caution
        },
        "sideways_low_vol": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "ent_coef": 0.02,
        },
        "sideways_high_vol": {
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "ent_coef": 0.05,
        },
    })


@dataclass
class RegimeDataset:
    """Dataset for a specific regime."""

    regime: MarketRegime
    data: pd.DataFrame
    start_date: date
    end_date: date
    bar_count: int
    pct_of_total: float


@dataclass
class RegimeTrainingResult:
    """Result of training an agent for a regime."""

    agent_id: str
    regime: MarketRegime
    model_path: str
    training_bars: int
    training_time_seconds: float
    final_reward: float
    metrics: Dict[str, float] = field(default_factory=dict)


class RegimeDataLabeler:
    """
    Labels historical data by market regime.

    Uses sliding window regime detection to assign
    each bar to a regime.
    """

    def __init__(self, config: Optional[RegimeDetectorConfig] = None):
        """Initialize labeler."""
        self.config = config or RegimeDetectorConfig()
        self.detector = RegimeDetector(self.config)

    def label_data(
        self,
        df: pd.DataFrame,
        window_size: int = 250,
        step_size: int = 1,
    ) -> pd.DataFrame:
        """
        Label each bar with its regime.

        Args:
            df: DataFrame with OHLCV data
            window_size: Lookback window for regime detection
            step_size: Step size for sliding window

        Returns:
            DataFrame with 'regime' column added
        """
        df = df.copy()
        df["regime"] = None
        df["regime_confidence"] = 0.0

        # Need enough data for detection
        if len(df) < window_size:
            logger.warning(f"Not enough data for regime labeling: {len(df)} < {window_size}")
            return df

        # Slide through data
        for i in range(window_size, len(df), step_size):
            window_df = df.iloc[i - window_size:i]

            try:
                regime_state = self.detector.detect(window_df, use_hmm=False)

                # Label the current bar
                df.iloc[i, df.columns.get_loc("regime")] = regime_state.regime.value
                df.iloc[i, df.columns.get_loc("regime_confidence")] = regime_state.confidence

            except Exception as e:
                logger.debug(f"Regime detection failed at index {i}: {e}")
                continue

        # Forward fill missing values
        df["regime"] = df["regime"].ffill()
        df["regime_confidence"] = df["regime_confidence"].ffill()

        # Fill initial values with first detected regime
        first_regime = df["regime"].dropna().iloc[0] if df["regime"].dropna().any() else "sideways_low_vol"
        df["regime"] = df["regime"].fillna(first_regime)
        df["regime_confidence"] = df["regime_confidence"].fillna(0.5)

        return df

    def split_by_regime(
        self,
        df: pd.DataFrame,
    ) -> Dict[MarketRegime, pd.DataFrame]:
        """
        Split labeled data by regime.

        Args:
            df: DataFrame with 'regime' column

        Returns:
            Dict mapping regime to its data subset
        """
        if "regime" not in df.columns:
            raise ValueError("Data must be labeled first. Call label_data()")

        splits = {}
        for regime in MarketRegime:
            regime_data = df[df["regime"] == regime.value].copy()
            if len(regime_data) > 0:
                splits[regime] = regime_data

        return splits

    def get_regime_statistics(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Get statistics about regime distribution."""
        if "regime" not in df.columns:
            return {}

        total = len(df)
        regime_counts = df["regime"].value_counts()

        stats = {
            "total_bars": total,
            "regime_distribution": {},
            "regime_periods": {},
        }

        for regime in MarketRegime:
            count = regime_counts.get(regime.value, 0)
            stats["regime_distribution"][regime.value] = {
                "count": int(count),
                "pct": float(count / total * 100) if total > 0 else 0,
            }

        return stats


class RegimeTrainingPipeline:
    """
    Pipeline for training regime-specialized agents.

    Workflow:
    1. Label historical data by regime
    2. Split data into regime-specific datasets
    3. Train N agents per regime
    4. Validate within-regime performance
    5. Register with AgentSelector
    """

    def __init__(self, config: Optional[RegimeTrainingConfig] = None):
        """Initialize training pipeline."""
        self.config = config or RegimeTrainingConfig()
        self.labeler = RegimeDataLabeler(self.config.regime_config)
        self._regime_datasets: Dict[MarketRegime, RegimeDataset] = {}
        self._training_results: List[RegimeTrainingResult] = []

        # Create output directories
        os.makedirs(self.config.models_dir, exist_ok=True)
        os.makedirs(self.config.reports_dir, exist_ok=True)

    def prepare_data(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> Dict[MarketRegime, RegimeDataset]:
        """
        Prepare regime-labeled datasets.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name for logging

        Returns:
            Dict of regime datasets
        """
        logger.info(f"Preparing regime data for {symbol} ({len(df)} bars)")

        # Label data
        labeled_df = self.labeler.label_data(df)

        # Get statistics
        stats = self.labeler.get_regime_statistics(labeled_df)
        logger.info(f"Regime distribution: {stats['regime_distribution']}")

        # Split by regime
        splits = self.labeler.split_by_regime(labeled_df)

        # Create datasets
        datasets = {}
        total_bars = len(df)

        for regime, regime_df in splits.items():
            if len(regime_df) >= self.config.min_bars_per_regime:
                datasets[regime] = RegimeDataset(
                    regime=regime,
                    data=regime_df,
                    start_date=regime_df.index[0].date() if hasattr(regime_df.index[0], 'date') else date.today(),
                    end_date=regime_df.index[-1].date() if hasattr(regime_df.index[-1], 'date') else date.today(),
                    bar_count=len(regime_df),
                    pct_of_total=len(regime_df) / total_bars * 100,
                )
                logger.info(
                    f"  {regime.value}: {len(regime_df)} bars ({len(regime_df)/total_bars*100:.1f}%)"
                )
            else:
                logger.warning(
                    f"  {regime.value}: {len(regime_df)} bars (SKIPPED - below minimum {self.config.min_bars_per_regime})"
                )

        self._regime_datasets = datasets
        return datasets

    def get_hyperparams_for_regime(self, regime: MarketRegime) -> Dict[str, Any]:
        """Get optimized hyperparameters for a regime."""
        return self.config.regime_hyperparams.get(
            regime.value,
            {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "ent_coef": 0.01,
            }
        )

    def train_agent_for_regime(
        self,
        regime: MarketRegime,
        agent_id: str,
        symbol: str,
    ) -> Optional[RegimeTrainingResult]:
        """
        Train a single agent for a specific regime.

        Args:
            regime: Market regime to train for
            agent_id: Unique agent identifier
            symbol: Symbol to train on

        Returns:
            Training result or None if failed
        """
        if regime not in self._regime_datasets:
            logger.error(f"No data for regime {regime.value}")
            return None

        dataset = self._regime_datasets[regime]

        logger.info(
            f"Training {agent_id} for {regime.value} "
            f"({dataset.bar_count} bars)"
        )

        start_time = datetime.now()

        try:
            # Get regime-specific hyperparameters
            hyperparams = self.get_hyperparams_for_regime(regime)

            # Create environment with regime data
            from gym_env.nautilus_env import NautilusBacktestEnv

            env = NautilusBacktestEnv(
                symbol=symbol,
                start_date=str(dataset.start_date),
                end_date=str(dataset.end_date),
            )

            # Train with SB3
            from stable_baselines3 import PPO

            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=hyperparams.get("learning_rate", 3e-4),
                gamma=hyperparams.get("gamma", 0.99),
                ent_coef=hyperparams.get("ent_coef", 0.01),
                verbose=0,
            )

            model.learn(
                total_timesteps=self.config.timesteps_per_agent,
                progress_bar=True,
            )

            # Save model
            model_path = os.path.join(
                self.config.models_dir,
                regime.value,
                f"{agent_id}.zip"
            )
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)

            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()

            # Get final metrics
            final_reward = self._evaluate_agent(model, env)

            result = RegimeTrainingResult(
                agent_id=agent_id,
                regime=regime,
                model_path=model_path,
                training_bars=dataset.bar_count,
                training_time_seconds=training_time,
                final_reward=final_reward,
                metrics={
                    "hyperparams": hyperparams,
                },
            )

            self._training_results.append(result)

            logger.info(
                f"  Trained {agent_id}: reward={final_reward:.2f}, "
                f"time={training_time:.1f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Training failed for {agent_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _evaluate_agent(self, model, env, n_episodes: int = 3) -> float:
        """Evaluate agent over N episodes."""
        total_reward = 0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_reward += episode_reward

        return total_reward / n_episodes

    def train_all_regimes(
        self,
        symbol: str,
    ) -> List[RegimeTrainingResult]:
        """
        Train agents for all regimes with sufficient data.

        Args:
            symbol: Symbol to train on

        Returns:
            List of training results
        """
        results = []

        for regime, dataset in self._regime_datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {self.config.agents_per_regime} agents for {regime.value}")
            logger.info(f"{'='*60}")

            for i in range(self.config.agents_per_regime):
                agent_id = f"{regime.value}_agent_{i:03d}"

                result = self.train_agent_for_regime(
                    regime=regime,
                    agent_id=agent_id,
                    symbol=symbol,
                )

                if result:
                    results.append(result)

        return results

    def generate_report(self) -> Dict[str, Any]:
        """Generate training report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "timesteps_per_agent": self.config.timesteps_per_agent,
                "agents_per_regime": self.config.agents_per_regime,
            },
            "datasets": {},
            "training_results": [],
            "summary": {},
        }

        # Dataset info
        for regime, dataset in self._regime_datasets.items():
            report["datasets"][regime.value] = {
                "bar_count": dataset.bar_count,
                "pct_of_total": dataset.pct_of_total,
                "start_date": str(dataset.start_date),
                "end_date": str(dataset.end_date),
            }

        # Training results
        for result in self._training_results:
            report["training_results"].append({
                "agent_id": result.agent_id,
                "regime": result.regime.value,
                "model_path": result.model_path,
                "training_bars": result.training_bars,
                "training_time_seconds": result.training_time_seconds,
                "final_reward": result.final_reward,
            })

        # Summary by regime
        regime_results = {}
        for result in self._training_results:
            regime = result.regime.value
            if regime not in regime_results:
                regime_results[regime] = []
            regime_results[regime].append(result.final_reward)

        for regime, rewards in regime_results.items():
            report["summary"][regime] = {
                "agents_trained": len(rewards),
                "avg_reward": float(np.mean(rewards)),
                "best_reward": float(max(rewards)),
                "worst_reward": float(min(rewards)),
            }

        # Save report
        report_path = os.path.join(
            self.config.reports_dir,
            f"regime_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {report_path}")

        return report

    def register_with_selector(self) -> None:
        """Register trained agents with AgentSelector."""
        from ml_institutional.agent_selector import AgentSelector

        selector = AgentSelector()

        for result in self._training_results:
            selector.register_agent(
                name=result.agent_id,
                model_path=result.model_path,
                regime=result.regime,
                algorithm="PPO",
                strategy_type="regime_specialized",
                metadata={
                    "training_bars": result.training_bars,
                    "final_reward": result.final_reward,
                },
            )

        logger.info(
            f"Registered {len(self._training_results)} agents with AgentSelector"
        )


def create_regime_training_script(
    symbol: str = "SPY.NASDAQ",
    timesteps: int = 100_000,
    agents_per_regime: int = 3,
) -> str:
    """
    Generate a training script for regime-based training.

    Returns script content as string.
    """
    script = f'''#!/usr/bin/env python3
"""
Regime-Based Training Script
Generated: {datetime.now().isoformat()}

Symbol: {symbol}
Timesteps: {timesteps:,}
Agents per regime: {agents_per_regime}
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from training.regime_training import (
    RegimeTrainingPipeline,
    RegimeTrainingConfig,
)

# Configuration
config = RegimeTrainingConfig(
    timesteps_per_agent={timesteps},
    agents_per_regime={agents_per_regime},
    min_bars_per_regime=200,
)

# Initialize pipeline
pipeline = RegimeTrainingPipeline(config)

# Load data
print("Loading data...")
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.model.data import BarType

catalog = ParquetDataCatalog("data/catalog_nautilus")
bar_type = BarType.from_str("{symbol}-1-DAY-LAST-EXTERNAL")
bars = catalog.bars(bar_types=[bar_type])

# Convert to DataFrame
df = pd.DataFrame({{
    'open': [float(b.open) for b in bars],
    'high': [float(b.high) for b in bars],
    'low': [float(b.low) for b in bars],
    'close': [float(b.close) for b in bars],
    'volume': [float(b.volume) for b in bars],
}}, index=pd.to_datetime([b.ts_event for b in bars], unit='ns'))

print(f"Loaded {{len(df)}} bars")

# Prepare regime data
print("\\nLabeling data by regime...")
datasets = pipeline.prepare_data(df, "{symbol}")

# Train
print("\\nStarting training...")
results = pipeline.train_all_regimes("{symbol}")

# Report
print("\\nGenerating report...")
report = pipeline.generate_report()

# Register with selector
print("\\nRegistering with AgentSelector...")
pipeline.register_with_selector()

print("\\n" + "="*60)
print("REGIME TRAINING COMPLETE")
print("="*60)
print(f"Total agents trained: {{len(results)}}")
for regime, summary in report.get("summary", {{}}).items():
    print(f"  {{regime}}: {{summary.get('agents_trained', 0)}} agents, "
          f"avg_reward={{summary.get('avg_reward', 0):.2f}}")
'''

    return script


# Convenience function for quick training
def train_regime_agents(
    data: pd.DataFrame,
    symbol: str,
    timesteps: int = 100_000,
    agents_per_regime: int = 3,
) -> Dict[str, Any]:
    """
    Quick function to train regime-specialized agents.

    Args:
        data: DataFrame with OHLCV data
        symbol: Symbol name
        timesteps: Training timesteps per agent
        agents_per_regime: Number of agents per regime

    Returns:
        Training report
    """
    config = RegimeTrainingConfig(
        timesteps_per_agent=timesteps,
        agents_per_regime=agents_per_regime,
    )

    pipeline = RegimeTrainingPipeline(config)

    # Prepare data
    datasets = pipeline.prepare_data(data, symbol)

    # Train
    results = pipeline.train_all_regimes(symbol)

    # Generate report
    report = pipeline.generate_report()

    # Register
    pipeline.register_with_selector()

    return report
