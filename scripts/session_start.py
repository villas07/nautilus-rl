#!/usr/bin/env python3
"""
Session Start Protocol

MANDATORY: This script MUST be executed at the start of every Claude Code session.

It performs:
1. Integrity verification of protected files
2. Governance engine initialization
3. State loading and summary display
4. Session marker creation

If this script is not run, the system will log protocol violations.
"""

import sys
from datetime import datetime
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    """Print session start banner."""
    print("""
+==============================================================================+
|                        NAUTILUS-AGENTS SESSION START                         |
|                                                                              |
|  This script MUST be run at the start of every session.                      |
|  It verifies system integrity and initializes governance.                    |
+==============================================================================+
""")


def verify_integrity():
    """Verify integrity of all protected files."""
    print("[1/5] Verifying file integrity...")

    from governance.integrity import verify_integrity, initialize_checksums, load_checksums

    stored = load_checksums()
    if not stored:
        print("      First run detected - initializing checksums...")
        initialize_checksums()
        print("      [OK] Checksums initialized")
        return True

    ok, violations = verify_integrity()

    if ok:
        print("      [OK] All protected files verified")
        return True
    else:
        print("      [ALERT] INTEGRITY VIOLATIONS DETECTED:")
        for v in violations:
            print(f"        - {v}")
        print("")
        print("      WARNING: Protected files have been modified outside governance!")
        print("      Review changes carefully before proceeding.")
        return False


def verify_immutable_rules():
    """Verify IMMUTABLE_RULES.md integrity."""
    print("[2/5] Verifying immutable rules...")

    rules_file = PROJECT_ROOT / ".rules" / "IMMUTABLE_RULES.md"
    hash_file = PROJECT_ROOT / ".rules" / "RULES_HASH"

    if not rules_file.exists():
        print("      [CRITICAL] IMMUTABLE_RULES.md NOT FOUND!")
        return False

    import hashlib
    content = rules_file.read_bytes()
    current_hash = hashlib.sha256(content).hexdigest()

    if hash_file.exists():
        stored_hash = hash_file.read_text(encoding="utf-8").strip()
        if current_hash != stored_hash:
            print("      [ALERT] IMMUTABLE_RULES.md has been modified!")
            print(f"        Expected: {stored_hash[:32]}...")
            print(f"        Got:      {current_hash[:32]}...")

            # Log the violation
            from governance.integrity import log_warning
            log_warning(
                f"IMMUTABLE_RULES.md modified. Expected hash {stored_hash[:32]}, "
                f"got {current_hash[:32]}",
                severity="CRITICAL"
            )
            return False
    else:
        hash_file.write_text(current_hash, encoding="utf-8")

    print("      [OK] Immutable rules verified")
    return True


def initialize_governance():
    """Initialize the governance engine."""
    print("[3/5] Initializing governance engine...")

    try:
        from governance.governance_engine import get_governance_engine
        from governance.validators import (
            RLEngineerValidator,
            QuantDeveloperValidator,
            MLOpsEngineerValidator,
        )

        engine = get_governance_engine()
        engine.register_validator("rl_engineer", RLEngineerValidator())
        engine.register_validator("quant_developer", QuantDeveloperValidator())
        engine.register_validator("mlops_engineer", MLOpsEngineerValidator())

        print(f"      [OK] Governance engine active")
        print(f"      [OK] Validators: {list(engine.validators.keys())}")
        return True

    except Exception as e:
        print(f"      [ERROR] Failed to initialize governance: {e}")
        return False


def load_current_state():
    """Load and display current project state."""
    print("[4/5] Loading current state...")

    # Check role states
    roles_dir = PROJECT_ROOT / ".roles"
    states = {}

    for role in ["rl_engineer", "quant_developer", "mlops_engineer"]:
        state_file = roles_dir / role / "STATE.md"
        if state_file.exists():
            content = state_file.read_text(encoding="utf-8")
            # Extract current focus from state file
            if "Current Focus:" in content:
                focus = content.split("Current Focus:")[1].split("\n")[0].strip()
                states[role] = focus
            elif "## Current" in content:
                focus = content.split("## Current")[1].split("\n")[1].strip()
                states[role] = focus[:50] + "..." if len(focus) > 50 else focus
            else:
                states[role] = "State file exists"

    print("      Role States:")
    for role, state in states.items():
        print(f"        @{role}: {state}")

    # Check pending decisions
    queue_file = roles_dir / "DECISION_QUEUE.md"
    if queue_file.exists():
        content = queue_file.read_text(encoding="utf-8")
        pending = content.count("Status**: Pending")
        if pending > 0:
            print(f"      [!] {pending} pending decisions in queue")

    # Check recent warnings
    warnings_file = PROJECT_ROOT / ".rules" / "WARNINGS.md"
    if warnings_file.exists():
        content = warnings_file.read_text(encoding="utf-8")
        recent = content.count(datetime.now().strftime("%Y-%m-%d"))
        if recent > 0:
            print(f"      [!] {recent} warnings logged today")

    print("      [OK] State loaded")
    return True


def mark_session_started():
    """Mark session as properly started."""
    print("[5/5] Marking session active...")

    from governance.integrity import mark_session_active
    mark_session_active()

    # Also log to session log
    session_log = PROJECT_ROOT / ".rules" / "SESSION_LOG.md"
    timestamp = datetime.now().isoformat()
    entry = f"- {timestamp}: Session started via protocol\n"

    if session_log.exists():
        content = session_log.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        if len(lines) > 100:
            lines = lines[-100:]
        content = "\n".join(lines) + "\n"
    else:
        content = "# Session Log\n\n"

    session_log.write_text(content + entry, encoding="utf-8")

    print("      [OK] Session marked active")
    return True


def display_summary(integrity_ok: bool, rules_ok: bool, governance_ok: bool):
    """Display session summary."""
    all_ok = integrity_ok and rules_ok and governance_ok

    if all_ok:
        status = "[OK] ACTIVE"
    elif rules_ok:
        status = "[WARN] DEGRADED"
    else:
        status = "[FAIL] COMPROMISED"

    print(f"""
+==============================================================================+
|                           SESSION SUMMARY                                    |
+==============================================================================+
|  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<50} |
|  Status: {status:<53} |
|  Integrity: {'VERIFIED' if integrity_ok else 'VIOLATIONS DETECTED':<51} |
|  Rules: {'VERIFIED' if rules_ok else 'COMPROMISED':<55} |
|  Governance: {'ACTIVE' if governance_ok else 'INACTIVE':<50} |
+==============================================================================+
|                                                                              |
|  CONFIRMATION:                                                               |
|  1. I have verified system integrity                                         |
|  2. I will operate within IMMUTABLE_RULES.md                                 |
|  3. I will use governance for all changes                                    |
|  4. I will document decisions and violations                                 |
|                                                                              |
+==============================================================================+
""")

    if not all_ok:
        print("[WARNING] System is not fully operational. Review issues above.")
        print("")

    # Quick reference
    print("Quick Reference:")
    print("  - Roles: @rl_engineer, @quant_developer, @mlops_engineer")
    print("  - Rules: .rules/IMMUTABLE_RULES.md")
    print("  - Config: config/autonomous_config.yaml")
    print("  - Decisions: .roles/DECISIONS.md")
    print("")


def main():
    """Main session start routine."""
    print_banner()

    integrity_ok = verify_integrity()
    rules_ok = verify_immutable_rules()
    governance_ok = initialize_governance()
    load_current_state()
    mark_session_started()

    display_summary(integrity_ok, rules_ok, governance_ok)

    # Return appropriate exit code
    if integrity_ok and rules_ok and governance_ok:
        return 0
    elif rules_ok:
        return 1  # Degraded but functional
    else:
        return 2  # Critical issues


if __name__ == "__main__":
    sys.exit(main())
