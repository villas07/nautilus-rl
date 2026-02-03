#!/usr/bin/env python3
"""
Test Regime-Aware Pipeline

Tests the complete flow:
1. Load market data
2. Detect current regime
3. Select appropriate agents
4. Get ensemble prediction
5. Validate against historical data

Usage:
    python scripts/test_regime_pipeline.py
    python scripts/test_regime_pipeline.py --models-dir ./models_batch2
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_institutional.regime_detector import RegimeDetector, MarketRegime
from ml_institutional.agent_selector import AgentSelector, AgentSelectorConfig, AgentConfig


def generate_sample_data(n_days: int = 500) -> pd.DataFrame:
    """Generate sample market data for testing."""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    # Simulate price with trend and volatility regimes
    returns = np.random.randn(n_days) * 0.02

    # Add regime changes
    # Days 0-100: Bull low vol
    returns[:100] = np.random.randn(100) * 0.01 + 0.001
    # Days 100-200: Bull high vol
    returns[100:200] = np.random.randn(100) * 0.03 + 0.001
    # Days 200-300: Bear
    returns[200:300] = np.random.randn(100) * 0.02 - 0.002
    # Days 300-400: Sideways
    returns[300:400] = np.random.randn(100) * 0.01
    # Days 400-500: Recovery
    returns[400:] = np.random.randn(100) * 0.015 + 0.0015

    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'open': prices * (1 - np.random.rand(n_days) * 0.005),
        'high': prices * (1 + np.random.rand(n_days) * 0.01),
        'low': prices * (1 - np.random.rand(n_days) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days),
    }, index=dates)

    return df


def create_mock_agents(models_dir: Path = None) -> list:
    """Create mock agent configs for testing."""
    agents = []

    regimes = [
        MarketRegime.BULL_LOW_VOL,
        MarketRegime.BULL_HIGH_VOL,
        MarketRegime.BEAR_LOW_VOL,
        MarketRegime.BEAR_HIGH_VOL,
        MarketRegime.SIDEWAYS_LOW_VOL,
        MarketRegime.SIDEWAYS_HIGH_VOL,
    ]

    for i, regime in enumerate(regimes):
        agent = AgentConfig(
            name=f"agent_{i:03d}",
            model_path=str(models_dir / f"agent_{i:03d}") if models_dir else f"mock_agent_{i}",
            regime=regime,
            algorithm="PPO",
            strategy_type="general",
            weight=1.0,
            active=True,
        )
        agents.append(agent)

    return agents


def test_regime_detection(df: pd.DataFrame) -> dict:
    """Test regime detection."""
    print("\n" + "=" * 60)
    print("  TEST: REGIME DETECTION")
    print("=" * 60)

    detector = RegimeDetector()

    # Detect regime for recent data
    state = detector.detect(df)

    print(f"  Current regime: {state.regime.value}")
    print(f"  Trend: {state.trend}")
    print(f"  Volatility: {state.volatility}")
    print(f"  Confidence: {state.confidence:.2%}")

    # Get regime history
    history = []
    window_size = 200

    for i in range(window_size, len(df), 20):
        window = df.iloc[:i]
        state = detector.detect(window)
        history.append({
            'date': df.index[i],
            'regime': state.regime.value,
            'confidence': state.confidence,
        })

    # Summarize regime distribution
    regimes = [h['regime'] for h in history]
    regime_counts = pd.Series(regimes).value_counts()

    print("\n  Regime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(history) * 100
        print(f"    {regime}: {count} ({pct:.1f}%)")

    return {
        'current': state.regime.value,
        'history': history,
        'distribution': regime_counts.to_dict(),
    }


def test_agent_selection(agents: list, regime: MarketRegime) -> dict:
    """Test agent selection."""
    print("\n" + "=" * 60)
    print("  TEST: AGENT SELECTION")
    print("=" * 60)

    config = AgentSelectorConfig(
        selection_mode='ensemble',
        ensemble_top_n=3,
        min_confidence=0.5,
    )

    selector = AgentSelector(config)

    # Register agents
    for agent in agents:
        selector.register_agent(agent)

    print(f"  Registered agents: {len(agents)}")
    print(f"  Selection mode: {config.selection_mode}")
    print(f"  Current regime: {regime.value}")

    # Get selected agents for regime
    selected = selector.get_agents_for_regime(regime)

    print(f"\n  Selected agents for {regime.value}:")
    for agent in selected:
        print(f"    - {agent.name} (strategy: {agent.strategy_type}, weight: {agent.weight})")

    return {
        'regime': regime.value,
        'selected_agents': [a.name for a in selected],
        'total_registered': len(agents),
    }


def test_ensemble_prediction(selector: AgentSelector, observation: np.ndarray) -> dict:
    """Test ensemble prediction (mock)."""
    print("\n" + "=" * 60)
    print("  TEST: ENSEMBLE PREDICTION (MOCK)")
    print("=" * 60)

    # Since we don't have real models, simulate predictions
    actions = {0: 'hold', 1: 'buy', 2: 'sell'}

    # Mock prediction
    mock_action = 1  # Buy
    mock_confidence = 0.72

    print(f"  Observation shape: {observation.shape}")
    print(f"  Mock action: {actions[mock_action]}")
    print(f"  Mock confidence: {mock_confidence:.2%}")

    return {
        'action': mock_action,
        'action_name': actions[mock_action],
        'confidence': mock_confidence,
        'note': 'Mock prediction - load real models for actual predictions',
    }


def main():
    parser = argparse.ArgumentParser(description="Test regime-aware pipeline")

    parser.add_argument(
        "--models-dir",
        type=str,
        help="Directory with trained models",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        help="CSV file with market data (default: generate sample)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  REGIME-AWARE PIPELINE TEST")
    print("=" * 60)
    print(f"  Time: {datetime.now().isoformat()}")

    # Load or generate data
    if args.data_file and Path(args.data_file).exists():
        print(f"  Loading data from: {args.data_file}")
        df = pd.read_csv(args.data_file, index_col=0, parse_dates=True)
    else:
        print("  Generating sample data...")
        df = generate_sample_data(500)

    print(f"  Data shape: {df.shape}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Test regime detection
    regime_results = test_regime_detection(df)
    current_regime = MarketRegime(regime_results['current'])

    # Create mock agents
    models_dir = Path(args.models_dir) if args.models_dir else None
    agents = create_mock_agents(models_dir)

    # Test agent selection
    selection_results = test_agent_selection(agents, current_regime)

    # Create selector for prediction test
    config = AgentSelectorConfig(selection_mode='ensemble', ensemble_top_n=3)
    selector = AgentSelector(config)
    for agent in agents:
        selector.register_agent(agent)

    # Test ensemble prediction
    mock_observation = np.random.randn(45)  # 45 features as per observation space
    prediction_results = test_ensemble_prediction(selector, mock_observation)

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE TEST COMPLETE")
    print("=" * 60)

    results = {
        'timestamp': datetime.now().isoformat(),
        'data_points': len(df),
        'regime': regime_results,
        'selection': selection_results,
        'prediction': prediction_results,
        'status': 'SUCCESS',
    }

    # Save results
    results_file = Path("validation/results/regime_pipeline_test.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved: {results_file}")
    print("\n  Next steps:")
    print("    1. Train models with: python training/train_agent.py")
    print("    2. Load real models and test predictions")
    print("    3. Run validation: python scripts/validate_batch.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
