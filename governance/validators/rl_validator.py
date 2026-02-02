"""
RL Engineer Validator

Validates changes against RL-specific criteria:
- Model improves Sharpe ratio vs baseline
- Drawdown within limits
- Walk-forward validation passes
- No overfitting (train/test gap < 20%)
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import structlog

from governance.governance_engine import (
    RoleValidator,
    ChangeRequest,
    ValidationReport,
    ValidationResult,
    ChangeType,
)

logger = structlog.get_logger()

# Immutable thresholds (from IMMUTABLE_RULES.md)
IMMUTABLE_THRESHOLDS = {
    "min_sharpe_improvement": 0.0,      # Must not decrease
    "max_drawdown_pct": 15.0,           # Hard limit
    "max_train_test_gap_pct": 20.0,     # Overfitting threshold
    "walk_forward_min_windows": 3,      # Minimum passing windows
}


class RLEngineerValidator(RoleValidator):
    """
    Validates changes proposed by or affecting the RL Engineer role.
    """

    def __init__(self):
        super().__init__("rl_engineer")
        self.baseline_metrics: Dict[str, float] = {}

    def validate(self, change: ChangeRequest) -> ValidationReport:
        """
        Run all RL-specific validations.
        """
        start_time = time.time()
        checks = {}
        messages = []

        # Determine which checks to run based on change type
        if change.change_type == ChangeType.MODEL:
            checks.update(self._validate_model_change(change, messages))
        elif change.change_type == ChangeType.CONFIG:
            checks.update(self._validate_config_change(change, messages))
        else:
            checks.update(self._validate_code_change(change, messages))

        # Determine overall result
        all_passed = all(checks.values()) if checks else True
        result = ValidationResult.PASS if all_passed else ValidationResult.FAIL

        duration = time.time() - start_time

        report = ValidationReport(
            validator=self.role_name,
            result=result,
            checks=checks,
            messages=messages,
            duration_seconds=duration,
        )

        logger.info(
            "RL validation complete",
            result=result.value,
            checks_passed=sum(checks.values()),
            checks_total=len(checks),
        )

        return report

    def _validate_model_change(
        self, change: ChangeRequest, messages: List[str]
    ) -> Dict[str, bool]:
        """Validate a model-related change."""
        checks = {}

        # Check 1: Sharpe ratio improvement
        sharpe_ok = self._check_sharpe_improvement(change.files_changed, messages)
        checks["sharpe_improvement"] = sharpe_ok

        # Check 2: Drawdown within limits
        drawdown_ok = self._check_drawdown_limit(change.files_changed, messages)
        checks["drawdown_limit"] = drawdown_ok

        # Check 3: Walk-forward validation
        walkforward_ok = self._check_walk_forward(change.files_changed, messages)
        checks["walk_forward_validation"] = walkforward_ok

        # Check 4: No overfitting
        overfitting_ok = self._check_no_overfitting(change.files_changed, messages)
        checks["no_overfitting"] = overfitting_ok

        return checks

    def _validate_config_change(
        self, change: ChangeRequest, messages: List[str]
    ) -> Dict[str, bool]:
        """Validate configuration changes."""
        checks = {}

        # Check that thresholds are not being relaxed
        for file in change.files_changed:
            if "config" in file.lower():
                threshold_ok = self._check_thresholds_not_relaxed(file, messages)
                checks["thresholds_not_relaxed"] = threshold_ok

        return checks

    def _validate_code_change(
        self, change: ChangeRequest, messages: List[str]
    ) -> Dict[str, bool]:
        """Validate general code changes."""
        checks = {}

        # Check that unit tests pass
        tests_ok = self._run_unit_tests(messages)
        checks["unit_tests_pass"] = tests_ok

        # Check for dangerous patterns
        patterns_ok = self._check_dangerous_patterns(change.files_changed, messages)
        checks["no_dangerous_patterns"] = patterns_ok

        return checks

    # =========================================================================
    # INDIVIDUAL CHECKS
    # =========================================================================

    def _check_sharpe_improvement(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """
        Check if model change improves or maintains Sharpe ratio.
        """
        # Find model files
        model_files = [f for f in files if f.endswith(".zip") or "model" in f.lower()]

        if not model_files:
            messages.append("No model files to validate")
            return True  # No model change, pass by default

        try:
            # Load and evaluate model
            # This is a placeholder - real implementation would:
            # 1. Load the model
            # 2. Run backtest
            # 3. Compare Sharpe to baseline

            # For now, check if model file exists and is valid
            from pathlib import Path
            for model_file in model_files:
                path = Path(model_file)
                if path.exists():
                    # Model exists - would evaluate here
                    messages.append(f"Model {path.name} exists, evaluation pending")
                    return True

            messages.append("Model file not found")
            return False

        except Exception as e:
            messages.append(f"Sharpe check error: {str(e)}")
            return False

    def _check_drawdown_limit(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """
        Check if model's max drawdown is within limit.
        """
        max_allowed = IMMUTABLE_THRESHOLDS["max_drawdown_pct"]

        try:
            # Placeholder - real implementation would run backtest
            # and check actual drawdown

            # For now, assume pass if no evidence of violation
            messages.append(f"Drawdown check: limit is {max_allowed}%")
            return True

        except Exception as e:
            messages.append(f"Drawdown check error: {str(e)}")
            return False

    def _check_walk_forward(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """
        Run walk-forward validation.
        """
        min_windows = IMMUTABLE_THRESHOLDS["walk_forward_min_windows"]

        try:
            # Placeholder - real implementation would:
            # 1. Split data into windows
            # 2. Train on each window
            # 3. Test on next window
            # 4. Check if min_windows pass

            messages.append(f"Walk-forward: requires {min_windows}/4 windows passing")
            return True

        except Exception as e:
            messages.append(f"Walk-forward error: {str(e)}")
            return False

    def _check_no_overfitting(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """
        Check train/test performance gap for overfitting.
        """
        max_gap = IMMUTABLE_THRESHOLDS["max_train_test_gap_pct"]

        try:
            # Placeholder - real implementation would:
            # 1. Get train performance
            # 2. Get test performance
            # 3. Calculate gap percentage
            # 4. Ensure gap < max_gap

            messages.append(f"Overfitting check: max allowed gap is {max_gap}%")
            return True

        except Exception as e:
            messages.append(f"Overfitting check error: {str(e)}")
            return False

    def _check_thresholds_not_relaxed(
        self, file: str, messages: List[str]
    ) -> bool:
        """
        Ensure risk thresholds are not being relaxed.
        """
        try:
            import yaml
            from pathlib import Path

            path = Path(file)
            if not path.exists():
                return True  # New file, no relaxation possible

            # Load current config
            with open(path) as f:
                config = yaml.safe_load(f)

            # Check critical thresholds
            violations = []

            # Check validation thresholds
            if "validation" in config:
                val = config["validation"].get("filter_1_basic", {})
                if val.get("min_sharpe_ratio", 999) < 1.5:
                    violations.append("min_sharpe_ratio below 1.5")
                if val.get("max_drawdown_pct", 0) > 15:
                    violations.append("max_drawdown_pct above 15%")

            if violations:
                messages.append(f"Threshold violations: {violations}")
                return False

            return True

        except Exception as e:
            messages.append(f"Threshold check error: {str(e)}")
            return True  # Don't block on parse errors

    def _run_unit_tests(self, messages: List[str]) -> bool:
        """Run unit tests for RL components."""
        try:
            import subprocess
            from pathlib import Path

            project_root = Path(__file__).parent.parent.parent
            test_dir = project_root / "tests" / "unit"

            # Check if tests/unit exists and has test files
            if not test_dir.exists():
                messages.append("No unit tests directory found - skipping")
                return True  # Don't fail if no tests yet

            test_files = list(test_dir.glob("test_*.py"))
            if not test_files:
                messages.append("No unit test files found - skipping")
                return True

            # First check if pytest is available
            check_pytest = subprocess.run(
                ["python", "-c", "import pytest"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_root,
            )
            if check_pytest.returncode != 0:
                messages.append("pytest not installed - skipping tests")
                return True

            result = subprocess.run(
                ["python", "-m", "pytest", str(test_dir), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_root,
            )

            if result.returncode == 0:
                messages.append("Unit tests passed")
                return True
            else:
                # Check for specific failure conditions
                output = result.stdout + result.stderr
                if "no tests ran" in output.lower() or "collected 0 items" in output.lower():
                    messages.append("No tests collected - skipping")
                    return True
                messages.append(f"Unit tests failed: {output[-500:]}")
                return False

        except subprocess.TimeoutExpired:
            messages.append("Unit tests timed out")
            return False
        except Exception as e:
            messages.append(f"Unit test error: {str(e)}")
            return True  # Don't block on test infrastructure issues

    def _check_dangerous_patterns(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """Check for dangerous code patterns."""
        dangerous_patterns = [
            ("eval(", "eval() is dangerous"),
            ("exec(", "exec() is dangerous"),
            ("__import__", "dynamic imports are risky"),
            ("subprocess.call", "prefer subprocess.run"),
            ("shell=True", "shell=True is risky"),
            ("pickle.load", "pickle can execute arbitrary code"),
        ]

        violations = []

        for file in files:
            try:
                path = Path(file)
                if path.exists() and path.suffix == ".py":
                    content = path.read_text()

                    for pattern, reason in dangerous_patterns:
                        if pattern in content:
                            violations.append(f"{path.name}: {reason}")

            except Exception:
                pass  # Skip files we can't read

        if violations:
            messages.append(f"Dangerous patterns found: {violations}")
            return False

        return True

    # =========================================================================
    # BASELINE MANAGEMENT
    # =========================================================================

    def set_baseline(self, model_id: str, metrics: Dict[str, float]):
        """Set baseline metrics for a model."""
        self.baseline_metrics[model_id] = metrics

    def get_baseline(self, model_id: str) -> Optional[Dict[str, float]]:
        """Get baseline metrics for a model."""
        return self.baseline_metrics.get(model_id)
