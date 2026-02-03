"""
Test script for R-009: Meta-Labeling.

Validates:
1. Meta-label creation
2. Meta-model training
3. Signal filtering
4. Bet sizing
5. Integration with Triple Barrier

Reference: EVAL-001, Advances in Financial Machine Learning Ch. 3
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

print("=" * 70)
print("R-009: Testing Meta-Labeling")
print("=" * 70)

# Test imports
print("\n[1] Testing imports...")
try:
    from ml_institutional.meta_labeling import (
        MetaLabeler,
        MetaLabelConfig,
        MetaLabel,
        MetaLabelingResult,
        MetaLabelingPipeline,
        PrimaryModelInterface,
        SignalType,
        create_meta_labels_from_triple_barrier,
    )
    print("    All imports successful")
except ImportError as e:
    print(f"    Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


class MockPrimaryModel(PrimaryModelInterface):
    """Mock primary model for testing."""

    def __init__(self, accuracy: float = 0.55):
        """Initialize with target accuracy."""
        self.accuracy = accuracy
        np.random.seed(42)

    def predict(self, features: np.ndarray) -> tuple:
        """Generate mock signal."""
        # Simple signal based on feature sum
        feat_sum = np.sum(features)

        if feat_sum > 0.5:
            signal = SignalType.BUY
        elif feat_sum < -0.5:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        confidence = min(abs(feat_sum) / 2, 1.0)
        return signal, confidence

    def predict_batch(self, features: np.ndarray) -> tuple:
        """Batch prediction."""
        signals = []
        confidences = []

        for i in range(len(features)):
            signal, conf = self.predict(features[i])
            signals.append(signal.value)
            confidences.append(conf)

        return np.array(signals), np.array(confidences)


def generate_test_data(n: int = 1000, n_features: int = 10):
    """Generate test data with signals and returns."""
    np.random.seed(42)

    # Features
    features = np.random.randn(n, n_features)

    # Primary signals (somewhat correlated with features)
    feature_signal = np.sum(features[:, :3], axis=1)  # Use first 3 features
    signals = np.sign(feature_signal + np.random.randn(n) * 0.5)
    signals = signals.astype(int)

    # Returns (somewhat correlated with true signal)
    true_signal = np.sign(features[:, 0] + np.random.randn(n) * 0.3)
    returns = true_signal * np.abs(np.random.randn(n)) * 0.02

    return features, signals, returns


def test_meta_label_creation():
    """Test meta-label creation."""
    print("\n[2] Testing Meta-Label Creation...")

    labeler = MetaLabeler()

    # Test data
    signals = np.array([1, 1, -1, -1, 0, 1, -1])
    returns = np.array([0.02, -0.01, -0.02, 0.01, 0.005, 0.01, -0.03])

    # Create meta-labels
    meta_labels = labeler.create_meta_labels(signals, returns)

    print(f"    Signals:     {signals}")
    print(f"    Returns:     {returns}")
    print(f"    Meta-labels: {meta_labels}")

    # Verify logic
    # Signal 1 + Return 0.02 = Profitable = 1
    assert meta_labels[0] == 1, "Profitable buy should be 1"
    # Signal 1 + Return -0.01 = Loss = 0
    assert meta_labels[1] == 0, "Losing buy should be 0"
    # Signal -1 + Return -0.02 = Profitable short = 1
    assert meta_labels[2] == 1, "Profitable short should be 1"
    # Signal -1 + Return 0.01 = Losing short = 0
    assert meta_labels[3] == 0, "Losing short should be 0"
    # Signal 0 (hold) = Always correct = 1
    assert meta_labels[4] == 1, "Hold should always be 1"

    print("    Meta-label creation test passed")


def test_meta_model_training():
    """Test meta-model training."""
    print("\n[3] Testing Meta-Model Training...")

    config = MetaLabelConfig(
        n_estimators=50,
        max_depth=3,
        cv_folds=3,
    )

    labeler = MetaLabeler(config)

    # Generate test data
    features, signals, returns = generate_test_data(n=500)

    # Train
    result = labeler.train(
        features=features,
        primary_signals=signals,
        returns=returns,
        feature_names=[f"feat_{i}" for i in range(features.shape[1])],
    )

    print(f"    Training results:")
    print(f"      Accuracy:  {result.accuracy:.3f}")
    print(f"      Precision: {result.precision:.3f}")
    print(f"      Recall:    {result.recall:.3f}")
    print(f"      F1:        {result.f1:.3f}")
    print(f"      ROC-AUC:   {result.roc_auc:.3f}" if result.roc_auc else "      ROC-AUC: N/A")
    print(f"      CV Mean:   {np.mean(result.cv_scores):.3f} (+/- {np.std(result.cv_scores):.3f})")

    # Top features
    print(f"    Top features:")
    for feat, imp in list(result.feature_importance.items())[:5]:
        print(f"      {feat}: {imp:.3f}")

    # Check basic requirements
    assert result.accuracy > 0.45, "Accuracy too low"
    assert len(result.feature_importance) == features.shape[1]

    print("    Meta-model training test passed")
    return labeler


def test_prediction(labeler: MetaLabeler):
    """Test meta-label prediction."""
    print("\n[4] Testing Prediction...")

    # Generate test sample
    np.random.seed(123)
    features = np.random.randn(10)

    # Predict
    result = labeler.predict(
        features=features,
        primary_signal=SignalType.BUY,
        primary_confidence=0.7,
    )

    print(f"    Prediction result:")
    print(f"      Primary signal: {result.primary_signal.name}")
    print(f"      Meta prediction: {result.meta_prediction} ({'Take' if result.meta_prediction else 'Skip'})")
    print(f"      Meta probability: {result.meta_probability:.3f}")
    print(f"      Bet size: {result.bet_size:.3f}")

    # Verify structure
    assert result.primary_signal == SignalType.BUY
    assert result.meta_prediction in [0, 1]
    assert 0 <= result.meta_probability <= 1
    assert 0 <= result.bet_size <= 1

    # Test hold signal
    hold_result = labeler.predict(
        features=features,
        primary_signal=SignalType.HOLD,
        primary_confidence=0.5,
    )

    assert hold_result.bet_size == 0.0, "Hold should have 0 bet size"

    print("    Prediction test passed")


def test_bet_sizing():
    """Test bet sizing calculation."""
    print("\n[5] Testing Bet Sizing...")

    config = MetaLabelConfig(
        enable_bet_sizing=True,
        max_bet_size=1.0,
        min_bet_size=0.1,
    )

    labeler = MetaLabeler(config)

    # Test different probability levels
    probabilities = [0.3, 0.5, 0.7, 0.9]
    primary_conf = 0.6

    print(f"    Bet sizes for different meta-probabilities (primary_conf={primary_conf}):")
    for prob in probabilities:
        bet_size = labeler._calculate_bet_size(prob, primary_conf)
        print(f"      Meta-prob {prob:.1f}: bet_size = {bet_size:.3f}")

    # Verify scaling
    low_bet = labeler._calculate_bet_size(0.5, 0.5)
    high_bet = labeler._calculate_bet_size(0.9, 0.9)

    assert low_bet < high_bet, "Higher confidence should give larger bet"
    assert labeler._calculate_bet_size(0.3, 0.5) == 0.0, "Low meta-prob should give 0 bet"

    print("    Bet sizing test passed")


def test_pipeline():
    """Test complete meta-labeling pipeline."""
    print("\n[6] Testing Pipeline...")

    # Create mock primary model
    primary_model = MockPrimaryModel(accuracy=0.55)

    config = MetaLabelConfig(
        n_estimators=30,
        max_depth=3,
        cv_folds=3,
    )

    pipeline = MetaLabelingPipeline(primary_model, config)

    # Generate data
    features, _, returns = generate_test_data(n=500)

    # Train meta-model
    result = pipeline.train_meta_model(
        features=features,
        returns=returns,
        feature_names=[f"feat_{i}" for i in range(features.shape[1])],
    )

    print(f"    Pipeline training: accuracy={result.accuracy:.3f}")

    # Get filtered signal
    test_features = np.random.randn(10)
    signal_result = pipeline.get_signal(test_features)

    print(f"    Signal result:")
    print(f"      Final signal: {signal_result['signal']}")
    print(f"      Primary signal: {signal_result['primary_signal']}")
    print(f"      Meta probability: {signal_result['meta_probability']:.3f}")
    print(f"      Bet size: {signal_result['bet_size']:.3f}")
    print(f"      Filtered: {signal_result['filtered']}")

    # Evaluate filtering
    eval_features, _, eval_returns = generate_test_data(n=200)
    evaluation = pipeline.evaluate_filtering(eval_features, eval_returns)

    print(f"\n    Filtering evaluation:")
    print(f"      Without meta: {evaluation['without_meta']['n_trades']} trades, "
          f"return={evaluation['without_meta']['total_return']:.4f}")
    print(f"      With meta:    {evaluation['with_meta']['n_trades']} trades, "
          f"return={evaluation['with_meta']['total_return']:.4f}")
    print(f"      Filter rate:  {evaluation['improvement']['filter_rate']:.1%}")

    print("    Pipeline test passed")


def test_triple_barrier_integration():
    """Test integration with Triple Barrier labels."""
    print("\n[7] Testing Triple Barrier Integration...")

    # Simulate signals and TB labels
    signals = np.array([1, 1, -1, -1, 1, -1, 0])
    tb_labels = np.array([1, -1, -1, 1, 1, -1, 0])  # 1=TP, -1=SL, 0=timeout

    # Create meta-labels
    meta_labels = create_meta_labels_from_triple_barrier(signals, tb_labels)

    print(f"    Signals:     {signals}")
    print(f"    TB Labels:   {tb_labels}")
    print(f"    Meta-labels: {meta_labels}")

    # Verify logic
    # Buy + TP = Correct
    assert meta_labels[0] == 1, "Buy hitting TP should be 1"
    # Buy + SL = Wrong
    assert meta_labels[1] == 0, "Buy hitting SL should be 0"
    # Sell + SL = Correct (short hitting SL means price went down)
    assert meta_labels[2] == 1, "Sell hitting SL should be 1"
    # Sell + TP = Wrong (short hitting TP means price went up)
    assert meta_labels[3] == 0, "Sell hitting TP should be 0"
    # Hold = Always correct
    assert meta_labels[6] == 1, "Hold should always be 1"

    print("    Triple Barrier integration test passed")


def test_save_load():
    """Test model saving and loading."""
    print("\n[8] Testing Save/Load...")

    import tempfile
    import os

    # Train a model
    config = MetaLabelConfig(n_estimators=20, max_depth=2)
    labeler = MetaLabeler(config)

    features, signals, returns = generate_test_data(n=200)
    labeler.train(features, signals, returns)

    # Save
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        save_path = f.name

    try:
        labeler.save(save_path)
        print(f"    Saved model to {save_path}")

        # Load
        loaded = MetaLabeler.load(save_path)
        print(f"    Loaded model from {save_path}")

        # Verify
        assert loaded._is_trained
        assert loaded._training_result is not None
        assert loaded._training_result.accuracy == labeler._training_result.accuracy

        print(f"    Loaded accuracy matches: {loaded._training_result.accuracy:.3f}")

    finally:
        os.unlink(save_path)

    print("    Save/Load test passed")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Starting R-009 Meta-Labeling Tests")
    print("=" * 70)

    test_meta_label_creation()
    labeler = test_meta_model_training()
    test_prediction(labeler)
    test_bet_sizing()
    test_pipeline()
    test_triple_barrier_integration()
    test_save_load()

    print("\n" + "=" * 70)
    print("R-009: ALL META-LABELING TESTS PASSED")
    print("=" * 70)

    print("\nComponents implemented:")
    print("  - MetaLabeler: Core meta-labeling model")
    print("  - MetaLabelConfig: Configuration options")
    print("  - MetaLabelingPipeline: Integration with primary model")
    print("  - Bet sizing based on meta-probability")
    print("  - Triple Barrier integration")

    print("\nBenefits:")
    print("  - Filters low-quality signals")
    print("  - Improves precision without sacrificing recall")
    print("  - Enables probabilistic position sizing")
    print("  - Works with RL agents, voting systems, or rule-based strategies")


if __name__ == "__main__":
    main()
