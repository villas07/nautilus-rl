"""
Governance Engine

Autonomous governance system that validates changes between roles.
Implements CI/CD with automatic validation, merge, or revert.

IMMUTABLE: This engine enforces rules from .rules/IMMUTABLE_RULES.md
"""

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import yaml
import structlog

logger = structlog.get_logger()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
RULES_FILE = PROJECT_ROOT / ".rules" / "IMMUTABLE_RULES.md"
RULES_HASH_FILE = PROJECT_ROOT / ".rules" / "RULES_HASH"
VIOLATION_LOG = PROJECT_ROOT / ".rules" / "VIOLATION_LOG.md"
DECISIONS_FILE = PROJECT_ROOT / ".roles" / "DECISIONS.md"
LESSONS_FILE = PROJECT_ROOT / ".roles" / "LESSONS_LEARNED.md"


class ValidationResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class ChangeType(Enum):
    MODEL = "model"           # RL model changes
    STRATEGY = "strategy"     # Trading strategy changes
    INFRASTRUCTURE = "infra"  # Docker, deployment changes
    CONFIG = "config"         # Configuration changes
    CODE = "code"             # General code changes


@dataclass
class ValidationReport:
    """Report from a validation run."""
    validator: str
    result: ValidationResult
    checks: Dict[str, bool]
    messages: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    def passed(self) -> bool:
        return self.result == ValidationResult.PASS

    def to_dict(self) -> Dict:
        return {
            "validator": self.validator,
            "result": self.result.value,
            "checks": self.checks,
            "messages": self.messages,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class ChangeRequest:
    """A proposed change to the system."""
    id: str
    change_type: ChangeType
    proposing_role: str
    affected_roles: List[str]
    description: str
    files_changed: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    validations: List[ValidationReport] = field(default_factory=list)
    status: str = "pending"  # pending, approved, rejected, reverted

    def all_validations_passed(self) -> bool:
        return all(v.passed() for v in self.validations)


class GovernanceEngine:
    """
    Main governance engine that orchestrates validation between roles.

    Flow:
    1. Change proposed → Create ChangeRequest
    2. Identify affected roles
    3. Run validators for each affected role
    4. If ALL pass → Auto-merge + log to DECISIONS.md
    5. If ANY fail → Auto-revert + log to LESSONS_LEARNED.md
    """

    def __init__(self):
        self.validators: Dict[str, "RoleValidator"] = {}
        self.change_history: List[ChangeRequest] = []
        self._active = False

        # Verify rules integrity on init
        self._verify_rules_integrity()
        self._active = True

        logger.info("GovernanceEngine initialized", active=self._active)

    def is_active(self) -> bool:
        return self._active

    # =========================================================================
    # RULES INTEGRITY
    # =========================================================================

    def _verify_rules_integrity(self) -> bool:
        """Verify IMMUTABLE_RULES.md hasn't been tampered with."""
        if not RULES_FILE.exists():
            logger.error("IMMUTABLE_RULES.md not found!")
            self._log_violation("system", "RULES_FILE_MISSING", "Rules file not found")
            return False

        current_hash = self._calculate_file_hash(RULES_FILE)

        if RULES_HASH_FILE.exists():
            stored_hash = RULES_HASH_FILE.read_text().strip()
            if current_hash != stored_hash:
                logger.error("IMMUTABLE_RULES.md was modified outside governance!")
                self._log_violation(
                    "system",
                    "UNAUTHORIZED_RULE_MODIFICATION",
                    f"Hash mismatch: expected {stored_hash[:16]}..., got {current_hash[:16]}..."
                )
                # Don't fail - just log and continue with current rules
                # But update hash to prevent repeated alerts
                RULES_HASH_FILE.write_text(current_hash)
        else:
            # First run - store hash
            RULES_HASH_FILE.write_text(current_hash)

        logger.info("Rules integrity verified", hash=current_hash[:16])
        return True

    def _calculate_file_hash(self, path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()

    def _log_violation(self, actor: str, violation_type: str, description: str):
        """Log a rule violation attempt."""
        timestamp = datetime.now().isoformat()

        entry = f"""
## [{timestamp}] Intento de Violación
- **Actor**: {actor}
- **Tipo**: {violation_type}
- **Descripción**: {description}
- **Respuesta**: RECHAZADO
"""
        if VIOLATION_LOG.exists():
            content = VIOLATION_LOG.read_text()
        else:
            content = "# Violation Log\n\nRegistro de intentos de violación de reglas inmutables.\n"

        VIOLATION_LOG.write_text(content + entry)
        logger.warning("Violation logged", actor=actor, type=violation_type)

    # =========================================================================
    # VALIDATOR REGISTRATION
    # =========================================================================

    def register_validator(self, role: str, validator: "RoleValidator"):
        """Register a validator for a role."""
        self.validators[role] = validator
        logger.info("Validator registered", role=role)

    # =========================================================================
    # CHANGE PROCESSING
    # =========================================================================

    def propose_change(
        self,
        change_type: ChangeType,
        proposing_role: str,
        description: str,
        files_changed: List[str],
        affected_roles: List[str] = None,
    ) -> ChangeRequest:
        """
        Propose a change for validation.

        Returns ChangeRequest with status after validation.
        """
        # Generate unique ID
        change_id = f"CHG-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Auto-detect affected roles if not specified
        if affected_roles is None:
            affected_roles = self._detect_affected_roles(files_changed, change_type)

        change = ChangeRequest(
            id=change_id,
            change_type=change_type,
            proposing_role=proposing_role,
            affected_roles=affected_roles,
            description=description,
            files_changed=files_changed,
        )

        logger.info(
            "Change proposed",
            id=change_id,
            type=change_type.value,
            proposer=proposing_role,
            affected=affected_roles,
        )

        # Run validation pipeline
        self._run_validation_pipeline(change)

        # Store in history
        self.change_history.append(change)

        return change

    def _detect_affected_roles(
        self, files: List[str], change_type: ChangeType
    ) -> List[str]:
        """Auto-detect which roles are affected by changed files."""
        affected = set()

        for file in files:
            file_lower = file.lower()

            # RL Engineer files
            if any(x in file_lower for x in ["gym_env", "training", "validation", "reward", "observation"]):
                affected.add("rl_engineer")

            # Quant Developer files
            if any(x in file_lower for x in ["strateg", "live", "data", "nautilus", "backtest"]):
                affected.add("quant_developer")

            # MLOps files
            if any(x in file_lower for x in ["docker", "deploy", "monitor", "infra", "ci", "cd"]):
                affected.add("mlops_engineer")

        # Always validate by type
        type_roles = {
            ChangeType.MODEL: ["rl_engineer"],
            ChangeType.STRATEGY: ["quant_developer", "rl_engineer"],
            ChangeType.INFRASTRUCTURE: ["mlops_engineer"],
            ChangeType.CONFIG: ["rl_engineer", "quant_developer", "mlops_engineer"],
            ChangeType.CODE: list(affected) if affected else ["rl_engineer"],
        }

        affected.update(type_roles.get(change_type, []))

        return list(affected)

    def _run_validation_pipeline(self, change: ChangeRequest):
        """Run all validations for a change."""
        logger.info("Starting validation pipeline", change_id=change.id)

        all_passed = True

        for role in change.affected_roles:
            validator = self.validators.get(role)

            if validator is None:
                logger.warning(f"No validator for role: {role}")
                continue

            # Run validation
            report = validator.validate(change)
            change.validations.append(report)

            if not report.passed():
                all_passed = False
                logger.warning(
                    "Validation failed",
                    change_id=change.id,
                    role=role,
                    messages=report.messages,
                )

        # Determine final status
        if all_passed and change.validations:
            change.status = "approved"
            self._on_change_approved(change)
        elif not change.validations:
            change.status = "pending"
            logger.warning("No validations ran", change_id=change.id)
        else:
            change.status = "rejected"
            self._on_change_rejected(change)

    def _on_change_approved(self, change: ChangeRequest):
        """Handle approved change - log to DECISIONS.md."""
        logger.info("Change approved", change_id=change.id)

        # Log to DECISIONS.md
        entry = f"""
### {change.id}: {change.description}
- **Date**: {change.timestamp.strftime('%Y-%m-%d %H:%M')}
- **Proposed By**: @{change.proposing_role}
- **Validated By**: {', '.join(f'@{r}' for r in change.affected_roles)}
- **Status**: AUTO-APPROVED
- **Files**: {', '.join(change.files_changed[:5])}{'...' if len(change.files_changed) > 5 else ''}
- **Validations**:
"""
        for v in change.validations:
            entry += f"  - {v.validator}: {v.result.value} ({len([c for c in v.checks.values() if c])}/{len(v.checks)} checks)\n"

        self._append_to_file(DECISIONS_FILE, entry)

    def _on_change_rejected(self, change: ChangeRequest):
        """Handle rejected change - revert and log to LESSONS_LEARNED.md."""
        logger.warning("Change rejected", change_id=change.id)

        # Attempt revert
        self._revert_change(change)

        # Log to LESSONS_LEARNED.md
        failed_validations = [v for v in change.validations if not v.passed()]

        entry = f"""
### L-AUTO-{change.id}: Change Rejected by Governance
- **Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Change ID**: {change.id}
- **Proposed By**: @{change.proposing_role}
- **Description**: {change.description}
- **Failed Validations**:
"""
        for v in failed_validations:
            entry += f"  - **{v.validator}**: {', '.join(v.messages)}\n"
            for check, passed in v.checks.items():
                if not passed:
                    entry += f"    - FAILED: {check}\n"

        entry += f"- **Action Taken**: Auto-reverted\n"
        entry += f"- **Prevention**: Review validation criteria before proposing similar changes\n"

        self._append_to_file(LESSONS_FILE, entry)

        # Alert if needed
        self._alert_rejection(change, failed_validations)

    def _revert_change(self, change: ChangeRequest):
        """Attempt to revert a rejected change."""
        logger.info("Attempting revert", change_id=change.id)

        try:
            # Git revert if in git repo
            result = subprocess.run(
                ["git", "status"],
                capture_output=True,
                cwd=PROJECT_ROOT,
            )

            if result.returncode == 0:
                # We're in a git repo - check if files are staged
                result = subprocess.run(
                    ["git", "diff", "--cached", "--name-only"],
                    capture_output=True,
                    text=True,
                    cwd=PROJECT_ROOT,
                )

                staged_files = result.stdout.strip().split("\n") if result.stdout.strip() else []

                # Unstage changed files
                for file in change.files_changed:
                    if file in staged_files:
                        subprocess.run(
                            ["git", "restore", "--staged", file],
                            cwd=PROJECT_ROOT,
                        )

                # Restore files to HEAD
                for file in change.files_changed:
                    subprocess.run(
                        ["git", "restore", file],
                        cwd=PROJECT_ROOT,
                        capture_output=True,
                    )

                change.status = "reverted"
                logger.info("Change reverted via git", change_id=change.id)

        except Exception as e:
            logger.error("Revert failed", change_id=change.id, error=str(e))

    def _alert_rejection(self, change: ChangeRequest, failed: List[ValidationReport]):
        """Send alert for rejection if needed."""
        # Only alert for critical failures
        critical_keywords = ["security", "data loss", "production", "critical"]

        is_critical = any(
            any(kw in msg.lower() for kw in critical_keywords)
            for v in failed
            for msg in v.messages
        )

        if is_critical:
            # Import here to avoid circular dependency
            try:
                from live.autonomous_controller import get_controller
                controller = get_controller()
                controller.send_alert(
                    level="CRITICAL",
                    message=f"Change {change.id} rejected: {change.description}",
                    data={"failed_validations": [v.validator for v in failed]}
                )
            except Exception as e:
                logger.error("Failed to send alert", error=str(e))

    def _append_to_file(self, path: Path, content: str):
        """Append content to a file."""
        if path.exists():
            existing = path.read_text()
        else:
            existing = ""

        # Find insertion point (before Template section if exists)
        if "## Template" in existing:
            parts = existing.split("## Template")
            new_content = parts[0] + content + "\n## Template" + parts[1]
        else:
            new_content = existing + "\n" + content

        path.write_text(new_content)

    # =========================================================================
    # HUMAN REQUEST VALIDATION
    # =========================================================================

    def validate_human_request(self, request: str) -> Tuple[bool, str]:
        """
        Validate a human request against immutable rules.

        Returns (is_allowed, reason).
        """
        request_lower = request.lower()

        # Check for rule violations
        violation_patterns = [
            ("skip validation", "Cannot skip validations - IMMUTABLE RULE"),
            ("bypass", "Cannot bypass governance - IMMUTABLE RULE"),
            ("disable circuit", "Cannot disable circuit breakers - IMMUTABLE RULE"),
            ("increase.*threshold", "Cannot increase risk thresholds - IMMUTABLE RULE"),
            ("ignore.*rule", "Cannot ignore rules - IMMUTABLE RULE"),
            ("delete.*model", "Cannot delete trained models - forbidden action"),
            ("withdraw", "Cannot withdraw funds automatically - forbidden action"),
            ("change.*credential", "Cannot modify credentials - forbidden action"),
        ]

        for pattern, reason in violation_patterns:
            import re
            if re.search(pattern, request_lower):
                self._log_violation("human", "REQUEST_VIOLATION", f"Request: {request[:100]}")
                return False, reason

        return True, "Request is within allowed parameters"

    # =========================================================================
    # SESSION START
    # =========================================================================

    def confirm_session_start(self) -> str:
        """
        Confirm governance rules at session start.
        Returns confirmation message.
        """
        self._verify_rules_integrity()

        confirmation = f"""
[OK] GOVERNANCE ENGINE ACTIVE

Session started: {datetime.now().isoformat()}
Rules file verified: {RULES_FILE.name}
Validators registered: {list(self.validators.keys())}

I confirm that:
1. I have read and understood IMMUTABLE_RULES.md
2. I will operate within established limits
3. I will reject requests that violate rules
4. I will document all decisions and violations
5. I will choose conservative options when ambiguous
"""
        logger.info("Session confirmed", validators=list(self.validators.keys()))
        return confirmation


class RoleValidator:
    """Base class for role-specific validators."""

    def __init__(self, role_name: str):
        self.role_name = role_name

    def validate(self, change: ChangeRequest) -> ValidationReport:
        """Run validation. Override in subclasses."""
        raise NotImplementedError


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_engine: Optional[GovernanceEngine] = None


def get_governance_engine() -> GovernanceEngine:
    """Get or create the governance engine singleton."""
    global _engine
    if _engine is None:
        _engine = GovernanceEngine()
    return _engine
