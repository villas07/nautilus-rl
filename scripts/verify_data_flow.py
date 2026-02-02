#!/usr/bin/env python3
"""
Data Flow Verification Script

Verifies the complete data flow:
1. Data Pipeline → Parquet Catalog
2. Parquet Catalog → NautilusTrader
3. NautilusTrader → Gymnasium Environment
4. Gymnasium → RL Training

Run this to ensure all components are properly connected.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Verify all required imports work."""
    print("\n" + "=" * 60)
    print("STEP 1: Checking imports")
    print("=" * 60)

    errors = []

    # Data pipeline
    try:
        from data.pipeline import DataPipeline, PipelineConfig
        print("  ✓ data.pipeline imports OK")
    except ImportError as e:
        errors.append(f"data.pipeline: {e}")
        print(f"  ✗ data.pipeline: {e}")

    # Adapters
    try:
        from data.adapters import (
            BinanceHistoricalAdapter,
            YahooFinanceAdapter,
        )
        print("  ✓ data.adapters imports OK")
    except ImportError as e:
        errors.append(f"data.adapters: {e}")
        print(f"  ✗ data.adapters: {e}")

    # Nautilus catalog writer
    try:
        from data.adapters.nautilus_catalog_writer import NautilusCatalogWriter
        print("  ✓ nautilus_catalog_writer imports OK")
    except ImportError as e:
        errors.append(f"nautilus_catalog_writer: {e}")
        print(f"  ✗ nautilus_catalog_writer: {e}")

    # Gym environment
    try:
        from gym_env import NautilusGymEnv, NautilusEnvConfig
        print("  ✓ gym_env imports OK")
    except ImportError as e:
        errors.append(f"gym_env: {e}")
        print(f"  ✗ gym_env: {e}")

    # NautilusTrader
    try:
        from nautilus_trader.persistence.catalog import ParquetDataCatalog
        from nautilus_trader.backtest.engine import BacktestEngine
        print("  ✓ nautilus_trader imports OK")
    except ImportError as e:
        errors.append(f"nautilus_trader: {e}")
        print(f"  ✗ nautilus_trader: {e}")

    # Stable-Baselines3
    try:
        from stable_baselines3 import PPO
        print("  ✓ stable_baselines3 imports OK")
    except ImportError as e:
        errors.append(f"stable_baselines3: {e}")
        print(f"  ✗ stable_baselines3: {e}")

    return len(errors) == 0


def check_catalog_path(catalog_path: str = "/app/data/catalog"):
    """Verify catalog path exists and has data."""
    print("\n" + "=" * 60)
    print("STEP 2: Checking Parquet Catalog")
    print("=" * 60)

    path = Path(catalog_path)

    if not path.exists():
        print(f"  ✗ Catalog path does not exist: {catalog_path}")
        print(f"    Create it with: mkdir -p {catalog_path}")
        return False

    print(f"  ✓ Catalog path exists: {catalog_path}")

    # Check for data files
    parquet_files = list(path.rglob("*.parquet"))
    if parquet_files:
        print(f"  ✓ Found {len(parquet_files)} parquet files")
        for pf in parquet_files[:5]:
            print(f"    - {pf.relative_to(path)}")
        if len(parquet_files) > 5:
            print(f"    ... and {len(parquet_files) - 5} more")
    else:
        print("  ⚠ No parquet files found in catalog")
        print("    Run data pipeline first to populate")

    return True


def check_nautilus_catalog(catalog_path: str = "/app/data/catalog"):
    """Verify NautilusTrader can read the catalog."""
    print("\n" + "=" * 60)
    print("STEP 3: Testing ParquetDataCatalog")
    print("=" * 60)

    try:
        from nautilus_trader.persistence.catalog import ParquetDataCatalog

        catalog = ParquetDataCatalog(catalog_path)
        print(f"  ✓ ParquetDataCatalog initialized")

        # List instruments
        instruments = catalog.instruments()
        print(f"  ✓ Found {len(instruments)} instruments")

        for inst in instruments[:5]:
            print(f"    - {inst.id}")

        if not instruments:
            print("  ⚠ No instruments in catalog")
            print("    Run NautilusCatalogWriter to add instruments")
            return False

        # Try loading bars for first instrument
        if instruments:
            inst = instruments[0]
            bars = catalog.bars(instrument_ids=[str(inst.id)])
            if bars:
                print(f"  ✓ Loaded {len(bars)} bars for {inst.id}")
                print(f"    First bar: {bars[0].ts_event}")
                print(f"    Last bar: {bars[-1].ts_event}")
            else:
                print(f"  ⚠ No bars found for {inst.id}")

        return True

    except Exception as e:
        print(f"  ✗ ParquetDataCatalog error: {e}")
        return False


def check_gym_environment(catalog_path: str = "/app/data/catalog"):
    """Verify Gymnasium environment works."""
    print("\n" + "=" * 60)
    print("STEP 4: Testing NautilusGymEnv")
    print("=" * 60)

    try:
        from gym_env import NautilusGymEnv, NautilusEnvConfig
        from nautilus_trader.persistence.catalog import ParquetDataCatalog

        # Get first instrument from catalog
        catalog = ParquetDataCatalog(catalog_path)
        instruments = catalog.instruments()

        if not instruments:
            print("  ⚠ No instruments in catalog, skipping gym test")
            return False

        inst = instruments[0]
        print(f"  Using instrument: {inst.id}")

        # Create environment config
        config = NautilusEnvConfig(
            instrument_id=str(inst.id),
            venue=str(inst.id.venue),
            catalog_path=catalog_path,
            max_episode_steps=100,  # Short for testing
        )

        print("  Creating NautilusGymEnv...")
        env = NautilusGymEnv(config)
        print(f"  ✓ Environment created")
        print(f"    Observation space: {env.observation_space}")
        print(f"    Action space: {env.action_space}")

        # Test reset
        print("  Testing reset()...")
        obs, info = env.reset()
        print(f"  ✓ Reset successful")
        print(f"    Observation shape: {obs.shape}")
        print(f"    Info: {info}")

        # Test step
        print("  Testing step()...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  ✓ Step successful")
        print(f"    Reward: {reward:.4f}")
        print(f"    Terminated: {terminated}")
        print(f"    Truncated: {truncated}")

        # Run a few more steps
        steps = 0
        while not (terminated or truncated) and steps < 10:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        print(f"  ✓ Ran {steps} steps successfully")

        env.close()
        print("  ✓ Environment closed")

        return True

    except Exception as e:
        print(f"  ✗ Gym environment error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_integration():
    """Verify training scripts can use the environment."""
    print("\n" + "=" * 60)
    print("STEP 5: Testing Training Integration")
    print("=" * 60)

    try:
        from training.train_agent import TrainingConfig, AgentTrainer

        print("  ✓ Training imports OK")

        # Create a minimal config
        config = TrainingConfig(
            agent_id="test_agent",
            symbol="TEST",
            venue="TEST",
            total_timesteps=100,
        )
        print(f"  ✓ TrainingConfig created")
        print(f"    Agent ID: {config.agent_id}")
        print(f"    Algorithm: {config.algorithm}")
        print(f"    Timesteps: {config.total_timesteps}")

        return True

    except Exception as e:
        print(f"  ✗ Training integration error: {e}")
        return False


def run_verification(catalog_path: str = "/app/data/catalog"):
    """Run all verification steps."""
    print("\n" + "=" * 60)
    print("        DATA FLOW VERIFICATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Catalog path: {catalog_path}")

    results = {
        "imports": check_imports(),
        "catalog_path": check_catalog_path(catalog_path),
    }

    # Only continue if basic checks pass
    if results["imports"] and results["catalog_path"]:
        results["nautilus_catalog"] = check_nautilus_catalog(catalog_path)

        if results["nautilus_catalog"]:
            results["gym_environment"] = check_gym_environment(catalog_path)

    results["training_integration"] = check_training_integration()

    # Summary
    print("\n" + "=" * 60)
    print("        VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check:25s} {status}")
        if not passed:
            all_passed = False

    print("\n" + "-" * 60)
    if all_passed:
        print("  ALL CHECKS PASSED - Data flow is correctly configured")
    else:
        print("  SOME CHECKS FAILED - Review errors above")

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify data flow configuration")
    parser.add_argument(
        "--catalog-path",
        type=str,
        default="/app/data/catalog",
        help="Path to ParquetDataCatalog",
    )

    args = parser.parse_args()

    success = run_verification(args.catalog_path)
    sys.exit(0 if success else 1)
