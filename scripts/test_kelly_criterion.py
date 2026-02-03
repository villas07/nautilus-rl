"""
Test script for R-010: Kelly Criterion Bet Sizing.

Validates:
1. Simple Kelly calculation
2. Continuous Kelly calculation
3. Sharpe-based Kelly calculation
4. Trade history estimation
5. Position limits
6. Integration with meta-labeling

Reference: EVAL-001, López de Prado Ch. 10
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

print("=" * 70)
print("R-010: Testing Kelly Criterion Bet Sizing")
print("=" * 70)

# Test imports
print("\n[1] Testing imports...")
try:
    from ml_institutional.kelly_criterion import (
        KellyCriterion,
        KellyConfig,
        KellyResult,
        KellyMethod,
        calculate_kelly_size,
        optimal_f_from_trades,
    )
    print("    All imports successful")
except ImportError as e:
    print(f"    Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def test_simple_kelly():
    """Test basic Kelly formula."""
    print("\n[2] Testing Simple Kelly Formula...")

    # Classic example: coin flip with 60% win, 2:1 payout
    # Kelly = (bp - q) / b = (2*0.6 - 0.4) / 2 = 0.8 / 2 = 0.4
    kelly_f = calculate_kelly_size(
        win_probability=0.60,
        win_loss_ratio=2.0,
        kelly_fraction=1.0,  # Full Kelly
    )

    print(f"    60% win, 2:1 payout: Kelly = {kelly_f:.3f}")
    assert abs(kelly_f - 0.4) < 0.01, f"Expected 0.4, got {kelly_f}"

    # With 1/4 Kelly (conservative)
    kelly_quarter = calculate_kelly_size(
        win_probability=0.60,
        win_loss_ratio=2.0,
        kelly_fraction=0.25,
    )

    print(f"    Same with 1/4 Kelly: {kelly_quarter:.3f}")
    assert abs(kelly_quarter - 0.1) < 0.01, f"Expected 0.1, got {kelly_quarter}"

    # Edge case: no edge (50% win, 1:1 payout)
    kelly_no_edge = calculate_kelly_size(
        win_probability=0.50,
        win_loss_ratio=1.0,
        kelly_fraction=1.0,
    )

    print(f"    No edge (50% win, 1:1): Kelly = {kelly_no_edge:.3f}")
    assert kelly_no_edge == 0.0, "No edge should give 0"

    # Negative edge (should return 0)
    kelly_negative = calculate_kelly_size(
        win_probability=0.45,
        win_loss_ratio=1.0,
        kelly_fraction=1.0,
    )

    print(f"    Negative edge (45% win): Kelly = {kelly_negative:.3f}")
    assert kelly_negative == 0.0, "Negative edge should give 0"

    print("    Simple Kelly test passed")


def test_kelly_criterion_class():
    """Test KellyCriterion class."""
    print("\n[3] Testing KellyCriterion Class...")

    config = KellyConfig(
        kelly_fraction=0.25,
        min_probability=0.52,
        method=KellyMethod.SIMPLE,
        max_position_pct=0.20,
    )

    kelly = KellyCriterion(config)

    # Direct calculation with known parameters
    result = kelly.calculate(
        probability=0.55,
        win_loss_ratio=1.5,
    )

    print(f"    55% win, 1.5:1 ratio:")
    print(f"      Full Kelly:     {result.kelly_fraction:.3f}")
    print(f"      Adjusted:       {result.adjusted_fraction:.3f}")
    print(f"      Position size:  {result.position_size:.3f}")
    print(f"      Edge:           {result.edge:.3f}")
    print(f"      Should trade:   {result.should_trade}")

    assert result.kelly_fraction > 0, "Should have positive Kelly"
    assert result.adjusted_fraction < result.kelly_fraction, "Adjusted should be smaller"
    assert result.should_trade, "Should trade with positive edge"

    # Test with below-threshold probability
    result_low = kelly.calculate(
        probability=0.51,  # Below 0.52 threshold
        win_loss_ratio=1.5,
    )

    print(f"\n    51% win (below threshold):")
    print(f"      Should trade:   {result_low.should_trade}")
    print(f"      Position size:  {result_low.position_size:.3f}")

    assert not result_low.should_trade, "Should not trade below threshold"
    assert result_low.position_size == 0.0, "Position should be 0"

    print("    KellyCriterion class test passed")


def test_trade_history():
    """Test Kelly estimation from trade history."""
    print("\n[4] Testing Trade History Estimation...")

    config = KellyConfig(
        kelly_fraction=0.25,
        method=KellyMethod.SHARPE,
        lookback_trades=50,
    )

    kelly = KellyCriterion(config)

    # Simulate trades with positive edge
    np.random.seed(42)
    n_trades = 100
    win_rate = 0.55
    avg_win = 0.03  # 3% average win
    avg_loss = 0.02  # 2% average loss

    for i in range(n_trades):
        if np.random.random() < win_rate:
            pnl = avg_win * (0.8 + 0.4 * np.random.random())  # Vary wins
            ret = pnl
        else:
            pnl = -avg_loss * (0.8 + 0.4 * np.random.random())
            ret = pnl

        kelly.add_trade(
            pnl=pnl * 10000,  # $10k position
            entry_price=100,
            exit_price=100 * (1 + ret),
            direction=1,
        )

    # Get stats
    stats = kelly.get_stats()
    print(f"    Trade history stats:")
    print(f"      Trades:    {stats['n_trades']}")
    print(f"      Win rate:  {stats['win_rate']:.1%}")
    print(f"      Avg ret:   {stats['avg_return']:.2%}")
    print(f"      Std ret:   {stats['std_return']:.2%}")
    print(f"      Sharpe:    {stats['sharpe']:.2f}")

    # Calculate Kelly from history
    result = kelly.calculate()

    print(f"\n    Kelly from history:")
    print(f"      Estimated prob:  {result.win_probability:.1%}")
    print(f"      Win/loss ratio:  {result.win_loss_ratio:.2f}")
    print(f"      Kelly fraction:  {result.kelly_fraction:.3f}")
    print(f"      Position size:   {result.position_size:.3f}")
    print(f"      Confidence:      {result.confidence:.2f}")

    assert result.confidence > 0.5, "Should have reasonable confidence"
    assert result.kelly_fraction > 0, "Should have positive Kelly with winning trades"

    print("    Trade history test passed")


def test_optimal_f():
    """Test optimal f calculation from returns."""
    print("\n[5] Testing Optimal F from Returns...")

    # Generate returns with positive edge
    np.random.seed(42)
    n = 200

    # Mix of wins and losses
    wins = np.random.normal(0.02, 0.01, int(n * 0.55))  # 55% wins, 2% avg
    losses = np.random.normal(-0.015, 0.008, int(n * 0.45))  # 45% losses, 1.5% avg
    returns = np.concatenate([wins, losses])
    np.random.shuffle(returns)

    # Find optimal f
    opt_f = optimal_f_from_trades(returns)

    print(f"    Returns stats:")
    print(f"      Mean:    {np.mean(returns):.2%}")
    print(f"      Std:     {np.std(returns):.2%}")
    print(f"      Win %:   {np.mean(returns > 0):.1%}")

    print(f"\n    Optimal f: {opt_f:.3f}")

    # Optimal f should be positive (we have positive edge)
    assert opt_f > 0, f"Optimal f should be positive, got {opt_f}"

    print("    Optimal f test passed")


def test_position_limits():
    """Test position limits are enforced."""
    print("\n[6] Testing Position Limits...")

    config = KellyConfig(
        kelly_fraction=1.0,  # Full Kelly
        min_position_pct=0.05,  # 5% minimum
        max_position_pct=0.15,  # 15% maximum
    )

    kelly = KellyCriterion(config)

    # Very high edge - should hit max
    result_high = kelly.calculate(probability=0.80, win_loss_ratio=3.0)

    print(f"    High edge (80% win, 3:1):")
    print(f"      Full Kelly:    {result_high.kelly_fraction:.3f}")
    print(f"      Position size: {result_high.position_size:.3f} (max: {config.max_position_pct})")

    assert result_high.position_size == config.max_position_pct, "Should hit max"

    # Small edge - check if above min
    config2 = KellyConfig(
        kelly_fraction=0.1,  # Very conservative
        min_position_pct=0.02,
        max_position_pct=0.20,
    )

    kelly2 = KellyCriterion(config2)
    result_low = kelly2.calculate(probability=0.52, win_loss_ratio=1.1)

    print(f"\n    Small edge (52% win, 1.1:1):")
    print(f"      Full Kelly:    {result_low.kelly_fraction:.3f}")
    print(f"      Position size: {result_low.position_size:.3f}")

    assert result_low.position_size >= config2.min_position_pct or not result_low.should_trade

    print("    Position limits test passed")


def test_integration_with_meta_labeling():
    """Test integration with meta-labeling predictions."""
    print("\n[7] Testing Integration with Predictions...")

    config = KellyConfig(
        kelly_fraction=0.25,
        vol_target=0.10,  # 10% target volatility
    )

    kelly = KellyCriterion(config)

    # Simulate meta-labeling prediction
    meta_prob = 0.75  # 75% confidence trade is good
    expected_return = 0.025  # 2.5% expected return
    expected_vol = 0.15  # 15% expected volatility

    result = kelly.calculate_from_predictions(
        predicted_prob=meta_prob,
        predicted_return=expected_return,
        predicted_vol=expected_vol,
    )

    print(f"    Meta-labeling prediction:")
    print(f"      Prob:          {meta_prob:.1%}")
    print(f"      Expected ret:  {expected_return:.2%}")
    print(f"      Expected vol:  {expected_vol:.2%}")

    print(f"\n    Kelly result:")
    print(f"      Position size: {result.position_size:.3f}")
    print(f"      Should trade:  {result.should_trade}")

    # Vol targeting should reduce position (15% vol vs 10% target)
    assert result.position_size < result.adjusted_fraction, "Vol targeting should reduce size"

    print("    Integration test passed")


def test_all_methods():
    """Test all Kelly calculation methods."""
    print("\n[8] Testing All Kelly Methods...")

    # Create trade history
    np.random.seed(42)
    returns = np.concatenate([
        np.random.normal(0.02, 0.01, 60),   # 60% wins
        np.random.normal(-0.015, 0.008, 40)  # 40% losses
    ])
    np.random.shuffle(returns)

    results = {}

    for method in KellyMethod:
        config = KellyConfig(
            kelly_fraction=0.25,
            method=method,
        )
        kelly = KellyCriterion(config)
        kelly.add_trades_from_returns(returns)

        result = kelly.calculate()
        results[method.value] = result

        print(f"    {method.value:12s}: Kelly={result.kelly_fraction:.3f}, "
              f"Pos={result.position_size:.3f}, Trade={result.should_trade}")

    # All methods should agree on direction
    all_positive = all(r.kelly_fraction >= 0 for r in results.values())
    assert all_positive, "All methods should give non-negative Kelly"

    print("    All methods test passed")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Starting R-010 Kelly Criterion Tests")
    print("=" * 70)

    test_simple_kelly()
    test_kelly_criterion_class()
    test_trade_history()
    test_optimal_f()
    test_position_limits()
    test_integration_with_meta_labeling()
    test_all_methods()

    print("\n" + "=" * 70)
    print("R-010: ALL KELLY CRITERION TESTS PASSED")
    print("=" * 70)

    print("\nComponents implemented:")
    print("  - KellyCriterion: Main class for position sizing")
    print("  - KellyConfig: Configuration options")
    print("  - KellyResult: Sizing recommendations")
    print("  - calculate_kelly_size(): Convenience function")
    print("  - optimal_f_from_trades(): Empirical optimization")

    print("\nKelly Methods:")
    print("  - SIMPLE: Classic Kelly (bp - q) / b")
    print("  - CONTINUOUS: For continuous returns mu/sigma^2")
    print("  - SHARPE: Sharpe-based Kelly")

    print("\nFeatures:")
    print("  - Fractional Kelly (default 1/4)")
    print("  - Position limits (min/max)")
    print("  - Time decay weighting")
    print("  - Volatility targeting")
    print("  - Trade history estimation")
    print("  - Integration with meta-labeling")


if __name__ == "__main__":
    main()
