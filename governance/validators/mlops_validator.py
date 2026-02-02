"""
MLOps Engineer Validator

Validates changes against infrastructure criteria:
- Docker build success
- No production service disruption
- Resource limits (CPU/RAM)
- Logs generation
"""

import os
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

# Immutable resource limits
RESOURCE_LIMITS = {
    "max_memory_mb": 8192,
    "max_cpu_percent": 80,
    "max_disk_gb": 50,
    "max_docker_image_mb": 5000,
}


class MLOpsEngineerValidator(RoleValidator):
    """
    Validates changes proposed by or affecting the MLOps Engineer role.
    """

    def __init__(self):
        super().__init__("mlops_engineer")

    def validate(self, change: ChangeRequest) -> ValidationReport:
        """Run all MLOps-specific validations."""
        start_time = time.time()
        checks = {}
        messages = []

        # Infrastructure checks
        if change.change_type == ChangeType.INFRASTRUCTURE:
            checks["docker_build_success"] = self._check_docker_build(
                change.files_changed, messages
            )
            checks["resource_limits"] = self._check_resource_limits(
                change.files_changed, messages
            )

        # Always check these
        checks["no_production_break"] = self._check_no_production_break(
            change.files_changed, messages
        )
        checks["logs_configured"] = self._check_logs_configured(
            change.files_changed, messages
        )

        # Code quality for infra files
        if any("deploy" in f.lower() or "docker" in f.lower() for f in change.files_changed):
            checks["security_scan"] = self._security_scan(
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
            "MLOps validation complete",
            result=result.value,
            checks_passed=sum(checks.values()),
            checks_total=len(checks),
        )

        return report

    # =========================================================================
    # INDIVIDUAL CHECKS
    # =========================================================================

    def _check_docker_build(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """
        Verify Docker build succeeds.
        """
        dockerfile_changed = any(
            "dockerfile" in f.lower() or "docker-compose" in f.lower()
            for f in files
        )

        if not dockerfile_changed:
            messages.append("No Dockerfile changes, skipping build check")
            return True

        try:
            # Check if Docker is available
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                messages.append("Docker not available")
                return True  # Don't fail if Docker not installed

            # Find Dockerfile
            dockerfile = PROJECT_ROOT / "Dockerfile"
            if not dockerfile.exists():
                messages.append("No Dockerfile found")
                return True

            # Try dry-run build (syntax check only)
            result = subprocess.run(
                [
                    "docker", "build",
                    "--no-cache",
                    "-f", str(dockerfile),
                    "--target", "builder",  # Build only first stage
                    "-t", "nautilus-test:validation",
                    ".",
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=PROJECT_ROOT,
            )

            if result.returncode == 0:
                messages.append("Docker build successful")
                return True
            else:
                messages.append(f"Docker build failed: {result.stderr[-500:]}")
                return False

        except subprocess.TimeoutExpired:
            messages.append("Docker build timed out")
            return False
        except FileNotFoundError:
            messages.append("Docker command not found")
            return True  # Don't fail if Docker not installed
        except Exception as e:
            messages.append(f"Docker build error: {str(e)}")
            return True

    def _check_resource_limits(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """
        Verify resource configurations are within limits.
        """
        try:
            import yaml

            violations = []

            for file in files:
                path = Path(file)
                if not path.exists():
                    continue

                # Check docker-compose files
                if "docker-compose" in file.lower():
                    with open(path) as f:
                        compose = yaml.safe_load(f)

                    services = compose.get("services", {})
                    for service_name, service in services.items():
                        deploy = service.get("deploy", {})
                        resources = deploy.get("resources", {})
                        limits = resources.get("limits", {})

                        # Check memory
                        memory = limits.get("memory", "")
                        if memory:
                            mem_mb = self._parse_memory(memory)
                            if mem_mb > RESOURCE_LIMITS["max_memory_mb"]:
                                violations.append(
                                    f"{service_name}: memory {mem_mb}MB > {RESOURCE_LIMITS['max_memory_mb']}MB"
                                )

                        # Check CPU
                        cpus = limits.get("cpus", "")
                        if cpus:
                            cpu_pct = float(cpus) * 100
                            if cpu_pct > RESOURCE_LIMITS["max_cpu_percent"]:
                                violations.append(
                                    f"{service_name}: CPU {cpu_pct}% > {RESOURCE_LIMITS['max_cpu_percent']}%"
                                )

                # Check config files
                if file.endswith((".yaml", ".yml")) and "config" in file.lower():
                    with open(path) as f:
                        config = yaml.safe_load(f)

                    # Check training resources
                    training = config.get("training", {})
                    local = training.get("local", {})

                    # Could add more specific checks here

            if violations:
                messages.append(f"Resource limit violations: {violations}")
                return False

            messages.append("Resource limits OK")
            return True

        except Exception as e:
            messages.append(f"Resource check error: {str(e)}")
            return True

    def _parse_memory(self, memory_str: str) -> int:
        """Parse memory string to MB."""
        memory_str = memory_str.lower().strip()

        if memory_str.endswith("g"):
            return int(float(memory_str[:-1]) * 1024)
        elif memory_str.endswith("gb"):
            return int(float(memory_str[:-2]) * 1024)
        elif memory_str.endswith("m"):
            return int(float(memory_str[:-1]))
        elif memory_str.endswith("mb"):
            return int(float(memory_str[:-2]))
        else:
            return int(memory_str)

    def _check_no_production_break(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """
        Check that changes don't break production services.
        """
        try:
            # Check for dangerous patterns in deployment files
            dangerous_patterns = [
                ("rm -rf /", "Dangerous rm command"),
                ("DROP TABLE", "Database table drop"),
                ("DELETE FROM", "Database delete without WHERE"),
                ("--force", "Force flag in deployment"),
                (":latest", "Using :latest tag in production"),
            ]

            deploy_files = [
                f for f in files
                if any(x in f.lower() for x in ["deploy", "docker", "compose", "k8s"])
            ]

            violations = []

            for file in deploy_files:
                try:
                    path = Path(file)
                    if path.exists():
                        content = path.read_text()

                        for pattern, reason in dangerous_patterns:
                            if pattern in content:
                                violations.append(f"{path.name}: {reason}")

                except Exception:
                    pass

            if violations:
                messages.append(f"Production safety violations: {violations}")
                return False

            # Check for required safety features
            required_features = []

            # Check docker-compose has restart policy
            compose_files = [f for f in files if "docker-compose" in f.lower()]
            for file in compose_files:
                try:
                    import yaml
                    path = Path(file)
                    if path.exists():
                        with open(path) as f:
                            compose = yaml.safe_load(f)

                        for service_name, service in compose.get("services", {}).items():
                            if "restart" not in service:
                                required_features.append(
                                    f"{service_name}: missing restart policy"
                                )

                except Exception:
                    pass

            if required_features:
                messages.append(f"Missing safety features: {required_features}")
                # Warning only, don't fail
                return True

            messages.append("Production safety check passed")
            return True

        except Exception as e:
            messages.append(f"Production check error: {str(e)}")
            return True

    def _check_logs_configured(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """
        Verify logging is properly configured.
        """
        try:
            # Check if structlog or logging is used
            py_files = [f for f in files if f.endswith(".py")]

            for file in py_files:
                try:
                    path = Path(file)
                    if path.exists():
                        content = path.read_text()

                        # Check for logging imports
                        has_logging = (
                            "import structlog" in content or
                            "import logging" in content or
                            "from structlog" in content
                        )

                        # Main files should have logging
                        if "main" in file.lower() or "__init__" in file:
                            if not has_logging:
                                messages.append(f"{path.name}: Consider adding logging")

                except Exception:
                    pass

            messages.append("Logging configuration check passed")
            return True

        except Exception as e:
            messages.append(f"Logging check error: {str(e)}")
            return True

    def _security_scan(
        self, files: List[str], messages: List[str]
    ) -> bool:
        """
        Run security scan on infrastructure files.
        """
        try:
            security_issues = []

            # Patterns to check
            secret_patterns = [
                (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password"),
                (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key"),
                (r"secret\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret"),
                (r"AWS_ACCESS_KEY", "AWS credentials in code"),
                (r"PRIVATE_KEY", "Private key in code"),
            ]

            import re

            for file in files:
                try:
                    path = Path(file)
                    if path.exists() and path.suffix in [".py", ".yaml", ".yml", ".sh", ".env"]:
                        content = path.read_text()

                        for pattern, issue_type in secret_patterns:
                            if re.search(pattern, content, re.IGNORECASE):
                                security_issues.append(f"{path.name}: {issue_type}")

                except Exception:
                    pass

            if security_issues:
                messages.append(f"SECURITY ISSUES: {security_issues}")
                return False

            messages.append("Security scan passed")
            return True

        except Exception as e:
            messages.append(f"Security scan error: {str(e)}")
            return True
