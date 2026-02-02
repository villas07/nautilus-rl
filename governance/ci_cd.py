#!/usr/bin/env python3
"""
CI/CD Pipeline

Autonomous CI/CD that validates all changes through governance.
Can be run as:
- Pre-commit hook
- CLI command
- GitHub Actions integration
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import structlog

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from governance.governance_engine import (
    get_governance_engine,
    ChangeType,
    ChangeRequest,
)
from governance.validators import (
    RLEngineerValidator,
    QuantDeveloperValidator,
    MLOpsEngineerValidator,
)

logger = structlog.get_logger()


class CICDPipeline:
    """
    Autonomous CI/CD Pipeline.

    Flow:
    1. Detect changed files
    2. Classify change type
    3. Submit to governance for validation
    4. Auto-merge or auto-revert based on result
    """

    def __init__(self):
        self.engine = get_governance_engine()
        self._register_validators()

    def _register_validators(self):
        """Register all role validators."""
        self.engine.register_validator("rl_engineer", RLEngineerValidator())
        self.engine.register_validator("quant_developer", QuantDeveloperValidator())
        self.engine.register_validator("mlops_engineer", MLOpsEngineerValidator())

    def get_changed_files(self, staged_only: bool = True) -> List[str]:
        """Get list of changed files from git."""
        try:
            if staged_only:
                # Get staged files
                result = subprocess.run(
                    ["git", "diff", "--cached", "--name-only"],
                    capture_output=True,
                    text=True,
                    cwd=PROJECT_ROOT,
                )
            else:
                # Get all changed files (staged + unstaged)
                result = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD"],
                    capture_output=True,
                    text=True,
                    cwd=PROJECT_ROOT,
                )

            if result.returncode == 0:
                files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
                return files
            else:
                return []

        except Exception as e:
            logger.error("Failed to get changed files", error=str(e))
            return []

    def classify_change(self, files: List[str]) -> Tuple[ChangeType, str]:
        """Classify the type of change based on files."""
        # Count files by category
        categories = {
            "model": 0,
            "strategy": 0,
            "infra": 0,
            "config": 0,
            "code": 0,
        }

        for f in files:
            f_lower = f.lower()

            if f.endswith(".zip") or "model" in f_lower:
                categories["model"] += 1
            elif any(x in f_lower for x in ["strateg", "trading", "order", "execution"]):
                categories["strategy"] += 1
            elif any(x in f_lower for x in ["docker", "deploy", "compose", "k8s", "ci", "cd"]):
                categories["infra"] += 1
            elif any(x in f_lower for x in ["config", ".yaml", ".yml", ".json", ".toml"]):
                categories["config"] += 1
            else:
                categories["code"] += 1

        # Determine primary type
        max_category = max(categories, key=categories.get)

        type_map = {
            "model": ChangeType.MODEL,
            "strategy": ChangeType.STRATEGY,
            "infra": ChangeType.INFRASTRUCTURE,
            "config": ChangeType.CONFIG,
            "code": ChangeType.CODE,
        }

        change_type = type_map[max_category]

        # Generate description
        description = f"{len(files)} files changed: {max_category} ({categories[max_category]} files)"

        return change_type, description

    def detect_proposing_role(self, files: List[str]) -> str:
        """Detect which role is proposing the change."""
        # Simple heuristic based on file paths
        role_patterns = {
            "rl_engineer": ["gym_env", "training", "validation", "reward", "observation", "model"],
            "quant_developer": ["strateg", "live", "data", "backtest", "nautilus"],
            "mlops_engineer": ["docker", "deploy", "monitor", "infra", "ci", "governance"],
        }

        scores = {role: 0 for role in role_patterns}

        for f in files:
            f_lower = f.lower()
            for role, patterns in role_patterns.items():
                if any(p in f_lower for p in patterns):
                    scores[role] += 1

        return max(scores, key=scores.get) or "rl_engineer"

    def run_pipeline(
        self,
        files: Optional[List[str]] = None,
        staged_only: bool = True,
        dry_run: bool = False,
    ) -> Tuple[bool, ChangeRequest]:
        """
        Run the full CI/CD pipeline.

        Returns:
            Tuple of (success, change_request)
        """
        # Get files if not provided
        if files is None:
            files = self.get_changed_files(staged_only)

        if not files:
            logger.info("No files to validate")
            return True, None

        logger.info("Starting CI/CD pipeline", files=files)

        # Classify change
        change_type, description = self.classify_change(files)
        proposing_role = self.detect_proposing_role(files)

        # Submit to governance
        if dry_run:
            logger.info(
                "DRY RUN - would submit to governance",
                change_type=change_type.value,
                proposer=proposing_role,
                files=files,
            )
            return True, None

        change = self.engine.propose_change(
            change_type=change_type,
            proposing_role=proposing_role,
            description=description,
            files_changed=files,
        )

        # Report result
        success = change.status == "approved"

        if success:
            logger.info(
                "CI/CD PASSED",
                change_id=change.id,
                status=change.status,
            )
        else:
            logger.error(
                "CI/CD FAILED",
                change_id=change.id,
                status=change.status,
                validations=[v.to_dict() for v in change.validations],
            )

        return success, change

    def generate_report(self, change: ChangeRequest) -> str:
        """Generate a human-readable report."""
        if change is None:
            return "No changes to report."

        lines = [
            "=" * 60,
            "CI/CD PIPELINE REPORT",
            "=" * 60,
            f"Change ID: {change.id}",
            f"Type: {change.change_type.value}",
            f"Status: {change.status.upper()}",
            f"Proposer: @{change.proposing_role}",
            f"Affected Roles: {', '.join(f'@{r}' for r in change.affected_roles)}",
            "",
            "Files Changed:",
        ]

        for f in change.files_changed[:10]:
            lines.append(f"  - {f}")
        if len(change.files_changed) > 10:
            lines.append(f"  ... and {len(change.files_changed) - 10} more")

        lines.append("")
        lines.append("Validations:")

        for v in change.validations:
            status = "[PASS]" if v.passed() else "[FAIL]"
            lines.append(f"  {v.validator}: {status}")

            for check, passed in v.checks.items():
                icon = "[+]" if passed else "[-]"
                lines.append(f"    {icon} {check}")

            if v.messages:
                for msg in v.messages:
                    lines.append(f"    -> {msg}")

        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CI/CD Pipeline with Autonomous Governance"
    )
    parser.add_argument(
        "--staged-only",
        action="store_true",
        default=True,
        help="Only validate staged files (default: True)",
    )
    parser.add_argument(
        "--all-changes",
        action="store_true",
        help="Validate all changed files, not just staged",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific files to validate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually submit to governance",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print detailed report",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    pipeline = CICDPipeline()

    staged_only = not args.all_changes

    success, change = pipeline.run_pipeline(
        files=args.files,
        staged_only=staged_only,
        dry_run=args.dry_run,
    )

    if args.report and change:
        print(pipeline.generate_report(change))
    elif args.json and change:
        output = {
            "success": success,
            "change_id": change.id,
            "status": change.status,
            "validations": [v.to_dict() for v in change.validations],
        }
        print(json.dumps(output, indent=2, default=str))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
