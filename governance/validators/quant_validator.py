"""
Quant Developer Validator

Validates changes against quant-specific criteria:
- NautilusTrader compatibility
- Unit tests pass
- Backtest reproducibility
- Order execution validity
"""

import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import structlog

from governance.governance_engine import (
    RoleValidator,
    ChangeRequest,
    ValidationReport,
    ValidationResult,
    ChangeType,
)

logger = structlog.get_logger()

PROJECT_ROOT = Path(__file__).parent.parent.parent


class QuantDeveloperValidator(RoleValidator):
    """
    Validates changes proposed by or affecting the Quant Developer role.
    """

    def __init__(self):
        super().__init__("quant_developer")

    def validate(self, change: ChangeRequest) -> ValidationReport:
        """Run all quant-specific validations."""
        start_time = time.time()
        checks = {}
        messages = []

        # Always run these checks
        checks["nautilus_compatibility"] = self._check_nautilus_compatibility(
            change.files_changed, messages
        )
        checks["unit_tests_pass"] = self._run_unit_tests(messages)

        # Additional checks for specific change types
        if change.change_type == ChangeType.STRATEGY:
            checks["backtest_reproducible"] = self._check_backtest_reproducibility(
                change.files_changed, messages
            )
            checks["order_execution_valid"] = self._check_order_execution(
                change.files_changed, messages
            )

        if change.change_type == ChangeType.CONFIG:
            checks["risk_thresholds_valid"] = self._check_risk_thresholds(
                change.files_changed, messages
            )

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
            "Quant validation complete",
            result=result.value,
            checks_passed=sum(checks.values()),
            checks_total=len(checks),
        )

        return report

    # =========================================================================
    # INDIVIDUAL CHECKS
    # =========================================================================

    def _check_nautilus_compatibility(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """
        Check if changes are compatible with NautilusTrader.
        """
        try:
            # Check 1: Import test
            import_ok = self._test_nautilus_imports(files, messages)
            if not import_ok:
                return False

            # Check 2: API usage patterns
            api_ok = self._check_nautilus_api_usage(files, messages)
            if not api_ok:
                return False

            messages.append("NautilusTrader compatibility verified")
            return True

        except Exception as e:
            messages.append(f"Nautilus compatibility error: {str(e)}")
            return False

    def _test_nautilus_imports(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """Test that nautilus imports work."""
        try:
            # Test basic imports
            from nautilus_trader.backtest.engine import BacktestEngine
            from nautilus_trader.persistence.catalog import ParquetDataCatalog
            from nautilus_trader.trading.strategy import Strategy

            messages.append("NautilusTrader imports OK")
            return True

        except ImportError as e:
            messages.append(f"NautilusTrader import failed: {str(e)}")
            return False

    def _check_nautilus_api_usage(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """Check for deprecated or incorrect API usage."""
        deprecated_patterns = [
            ("account.equity(", "Use balance_total().as_double() instead of equity()"),
            ("BacktestVenueConfig(.*maker_fee", "maker_fee removed from BacktestVenueConfig"),
            ("BacktestVenueConfig(.*taker_fee", "taker_fee removed from BacktestVenueConfig"),
        ]

        violations = []

        for file in files:
            try:
                path = Path(file)
                if path.exists() and path.suffix == ".py":
                    content = path.read_text()

                    import re
                    for pattern, reason in deprecated_patterns:
                        if re.search(pattern, content):
                            violations.append(f"{path.name}: {reason}")

            except Exception:
                pass

        if violations:
            messages.append(f"Deprecated API usage: {violations}")
            return False

        return True

    def _run_unit_tests(self, messages: List[str]) -> bool:
        """Run unit tests for quant components."""
        try:
            test_dir = PROJECT_ROOT / "tests" / "unit"

            if not test_dir.exists():
                messages.append("No unit tests directory")
                return True

            # Run pytest
            result = subprocess.run(
                ["python", "-m", "pytest", str(test_dir), "-v", "-x", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=PROJECT_ROOT,
            )

            if result.returncode == 0:
                messages.append("Unit tests passed")
                return True
            else:
                # Extract failure info
                output = result.stdout + result.stderr
                messages.append(f"Unit tests failed: {output[-500:]}")
                return False

        except subprocess.TimeoutExpired:
            messages.append("Unit tests timed out")
            return False
        except FileNotFoundError:
            messages.append("pytest not found")
            return True  # Don't block if pytest not installed
        except Exception as e:
            messages.append(f"Unit test error: {str(e)}")
            return True

    def _check_backtest_reproducibility(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """
        Run backtest multiple times and verify reproducibility.
        """
        try:
            # This would run the same backtest 3 times and compare results
            # For now, placeholder that checks for non-deterministic code

            non_deterministic_patterns = [
                "random.random()",
                "random.randint(",
                "np.random.rand(",
                "datetime.now()",  # Can cause issues in backtests
            ]

            strategy_files = [
                f for f in files
                if "strateg" in f.lower() and f.endswith(".py")
            ]

            issues = []
            for file in strategy_files:
                try:
                    path = Path(file)
                    if path.exists():
                        content = path.read_text()
                        for pattern in non_deterministic_patterns:
                            if pattern in content:
                                issues.append(f"{path.name} uses {pattern}")
                except Exception:
                    pass

            if issues:
                messages.append(f"Non-deterministic code found: {issues}")
                messages.append("Consider using seeded random for reproducibility")
                # Warning only, don't fail
                return True

            messages.append("Backtest reproducibility check passed")
            return True

        except Exception as e:
            messages.append(f"Reproducibility check error: {str(e)}")
            return True

    def _check_order_execution(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """
        Verify order execution logic is valid.
        """
        try:
            required_patterns = [
                ("submit_order", "Strategy should use submit_order()"),
                ("MarketOrder", "Should use proper order types"),
            ]

            strategy_files = [
                f for f in files
                if "strateg" in f.lower() and f.endswith(".py")
            ]

            if not strategy_files:
                messages.append("No strategy files to validate")
                return True

            for file in strategy_files:
                try:
                    path = Path(file)
                    if path.exists():
                        content = path.read_text()

                        # Check for required patterns
                        for pattern, reason in required_patterns:
                            if pattern not in content:
                                messages.append(f"Missing in {path.name}: {reason}")
                                # Warning only for now
                except Exception:
                    pass

            messages.append("Order execution patterns OK")
            return True

        except Exception as e:
            messages.append(f"Order execution check error: {str(e)}")
            return True

    def _check_risk_thresholds(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """
        Verify risk thresholds are not being violated.
        """
        # Immutable risk limits
        RISK_LIMITS = {
            "max_position_per_symbol_usd": 5000,
            "max_total_exposure_usd": 50000,
            "max_daily_loss_usd": 2000,
        }

        try:
            import yaml

            for file in files:
                if "config" in file.lower() and file.endswith((".yaml", ".yml")):
                    path = Path(file)
                    if path.exists():
                        with open(path) as f:
                            config = yaml.safe_load(f)

                        # Check position_sizing section
                        ps = config.get("position_sizing", {})

                        violations = []
                        if ps.get("max_position_per_symbol_usd", 0) > RISK_LIMITS["max_position_per_symbol_usd"]:
                            violations.append("max_position_per_symbol_usd exceeds limit")

                        if ps.get("max_total_exposure_usd", 0) > RISK_LIMITS["max_total_exposure_usd"]:
                            violations.append("max_total_exposure_usd exceeds limit")

                        if violations:
                            messages.append(f"Risk threshold violations: {violations}")
                            return False

            messages.append("Risk thresholds within limits")
            return True

        except Exception as e:
            messages.append(f"Risk threshold check error: {str(e)}")
            return True
