"""
Governance Module

Autonomous governance system for the nautilus-agents project.
Implements CI/CD with automatic validation between roles.
"""

from governance.governance_engine import (
    GovernanceEngine,
    get_governance_engine,
    ChangeRequest,
    ChangeType,
    ValidationResult,
    ValidationReport,
    RoleValidator,
)

__all__ = [
    "GovernanceEngine",
    "get_governance_engine",
    "ChangeRequest",
    "ChangeType",
    "ValidationResult",
    "ValidationReport",
    "RoleValidator",
]
