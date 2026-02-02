#!/usr/bin/env python3
"""
Test complete data flow from catalog to environment.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_catalog_read():
    """Test reading from ParquetDataCatalog."""
    print("\n" + "=" * 60)
    print("TEST 1: ParquetDataCatalog Read")
    print("=" * 60)

    from nautilus_trader.persistence.catalog import ParquetDataCatalog

    catalog_path = Path(__file__).parent.parent / "data" / "catalog"
    catalog = ParquetDataCatalog(str(catalog_path))

    # List instruments
    instruments = catalog.instruments()
    print(f"Instruments found: {len(instruments)}")

    for inst in instruments:
        print(f"  - {inst.id} ({type(inst).__name__})")

    # Read bars for first instrument
    if instruments:
        inst = instruments[0]
        bars = catalog.bars(instrument_ids=[str(inst.id)])
        print(f"\nBars for {inst.id}: {len(bars) if bars else 0}")

        if bars:
            print(f"  First bar: {bars[0].open} / {bars[0].close}")
            print(f"  Last bar: {bars[-1].open} / {bars[-1].close}")
            return True

    return False


def test_backtest_engine():
    """Test BacktestEngine with catalog data."""
    print("\n" + "=" * 60)
    print("TEST 2: BacktestEngine Integration")
    print("=" * 60)

    from nautilus_trader.backtest.engine import BacktestEngine
    from nautilus_trader.config import BacktestEngineConfig
    from nautilus_trader.persistence.catalog import ParquetDataCatalog
    from nautilus_trader.model.identifiers import Venue
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.objects import Money

    catalog_path = Path(__file__).parent.parent / "data" / "catalog"
    catalog = ParquetDataCatalog(str(catalog_path))

    instruments = catalog.instruments()
    if not instruments:
        print("No instruments in catalog")
        return False

    # Use first instrument
    inst = instruments[0]
    print(f"Using instrument: {inst.id}")

    # Create engine
    engine_config = BacktestEngineConfig(trader_id="TESTER-001")
    engine = BacktestEngine(config=engine_config)

    # Add venue
    venue_name = str(inst.id.venue)
    engine.add_venue(
        venue=Venue(venue_name),
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=None,
        starting_balances=[Money(100000, inst.quote_currency if hasattr(inst, 'quote_currency') else inst.currency)],
    )

    # Add instrument
    engine.add_instrument(inst)

    # Add bars
    bars = catalog.bars(instrument_ids=[str(inst.id)])
    if bars:
        engine.add_data(bars)
        print(f"Added {len(bars)} bars to engine")

    # Run empty backtest (no strategy)
    engine.run()
    print("Engine ran successfully")

    engine.dispose()
    return True


def test_gym_environment():
    """Test Gymnasium environment."""
    print("\n" + "=" * 60)
    print("TEST 3: Gymnasium Environment")
    print("=" * 60)

    from gym_env import NautilusGymEnv, NautilusEnvConfig
    from pathlib import Path

    catalog_path = str(Path(__file__).parent.parent / "data" / "catalog")

    # Get an instrument from catalog
    from nautilus_trader.persistence.catalog import ParquetDataCatalog
    catalog = ParquetDataCatalog(catalog_path)
    instruments = catalog.instruments()

    if not instruments:
        print("No instruments in catalog")
        return False

    inst = instruments[0]
    print(f"Testing with instrument: {inst.id}")

    # Create environment config
    config = NautilusEnvConfig(
        instrument_id=str(inst.id),
        venue=str(inst.id.venue),
        catalog_path=catalog_path,
        max_episode_steps=100,  # Short for testing
    )

    print("Creating environment...")
    try:
        env = NautilusGymEnv(config)
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        print("Testing reset...")
        obs, info = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Info: {info}")

        print("Testing step...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")

        # Run a few more steps
        steps = 0
        while not (terminated or truncated) and steps < 10:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        print(f"Ran {steps} additional steps")

        env.close()
        print("Environment test PASSED")
        return True

    except Exception as e:
        print(f"Environment test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_imports():
    """Test training module imports."""
    print("\n" + "=" * 60)
    print("TEST 4: Training Module Imports")
    print("=" * 60)

    try:
        from training.train_agent import TrainingConfig, AgentTrainer
        print("  TrainingConfig: OK")
        print("  AgentTrainer: OK")

        from training.train_batch import BatchConfig, BatchTrainer
        print("  BatchConfig: OK")
        print("  BatchTrainer: OK")

        from stable_baselines3 import PPO, A2C, SAC
        print("  PPO: OK")
        print("  A2C: OK")
        print("  SAC: OK")

        return True

    except Exception as e:
        print(f"Import FAILED: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DATA FLOW VERIFICATION")
    print("=" * 60)

    results = {}

    # Test 1: Catalog read
    try:
        results["catalog_read"] = test_catalog_read()
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
        results["catalog_read"] = False

    # Test 2: Backtest engine
    try:
        results["backtest_engine"] = test_backtest_engine()
    except Exception as e:
        print(f"Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["backtest_engine"] = False

    # Test 3: Gym environment
    try:
        results["gym_environment"] = test_gym_environment()
    except Exception as e:
        print(f"Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["gym_environment"] = False

    # Test 4: Training imports
    try:
        results["training_imports"] = test_training_imports()
    except Exception as e:
        print(f"Test 4 FAILED: {e}")
        results["training_imports"] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests PASSED - Data flow is working correctly")
    else:
        print("\nSome tests FAILED - Check errors above")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
