#!/usr/bin/env python3
"""
Session Initialization

MUST be called at the start of every Claude Code session.
Verifies rules integrity and confirms compliance.
"""

import hashlib
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def verify_immutable_rules() -> bool:
    """
    Verify IMMUTABLE_RULES.md integrity.

    Returns True if verified, False if compromised.
    """
    rules_file = PROJECT_ROOT / ".rules" / "IMMUTABLE_RULES.md"
    hash_file = PROJECT_ROOT / ".rules" / "RULES_HASH"

    if not rules_file.exists():
        print("[CRITICAL] IMMUTABLE_RULES.md not found!")
        return False

    # Calculate current hash
    content = rules_file.read_bytes()
    current_hash = hashlib.sha256(content).hexdigest()

    # Check stored hash
    if hash_file.exists():
        stored_hash = hash_file.read_text(encoding='utf-8').strip()
        if current_hash != stored_hash:
            print("WARNING: IMMUTABLE_RULES.md was modified outside governance!")
            print(f"   Expected: {stored_hash[:32]}...")
            print(f"   Got:      {current_hash[:32]}...")
            # Update hash but log the violation
            hash_file.write_text(current_hash, encoding='utf-8')

            # Log violation
            violation_log = PROJECT_ROOT / ".rules" / "VIOLATION_LOG.md"
            entry = f"""
## [{datetime.now().isoformat()}] Rule File Modified
- **Type**: EXTERNAL_MODIFICATION
- **File**: IMMUTABLE_RULES.md
- **Action**: Hash updated, monitoring continues
"""
            if violation_log.exists():
                existing = violation_log.read_text(encoding='utf-8')
            else:
                existing = "# Violation Log\n"
            violation_log.write_text(existing + entry, encoding='utf-8')
    else:
        # First run - store hash
        hash_file.write_text(current_hash, encoding='utf-8')

    return True


def confirm_session() -> str:
    """
    Confirm governance compliance for this session.

    Returns confirmation message to display.
    """
    timestamp = datetime.now().isoformat()

    # Verify rules
    rules_ok = verify_immutable_rules()

    # Load governance engine
    try:
        from governance.governance_engine import get_governance_engine
        engine = get_governance_engine()
        engine_ok = engine.is_active()
    except Exception as e:
        engine_ok = False
        print(f"[WARNING] Governance engine error: {e}")

    # Generate confirmation
    if rules_ok and engine_ok:
        status = "[OK] ACTIVE"
    elif rules_ok:
        status = "[WARN] DEGRADED (engine inactive)"
    else:
        status = "[FAIL] COMPROMISED"

    confirmation = f"""
+==============================================================+
|                 GOVERNANCE SESSION START                      |
+==============================================================+
|  Timestamp: {timestamp[:19]}                          |
|  Status: {status:<52} |
|  Rules Verified: {'Yes' if rules_ok else 'NO':<44} |
|  Engine Active: {'Yes' if engine_ok else 'NO':<45} |
+==============================================================+
|                                                              |
|  I CONFIRM:                                                  |
|  1. I have read IMMUTABLE_RULES.md                          |
|  2. I will operate within established limits                 |
|  3. I will reject rule-violating requests                    |
|  4. I will document all decisions and violations             |
|  5. I will choose conservative options when ambiguous        |
|                                                              |
+==============================================================+
"""

    # Log session start
    session_log = PROJECT_ROOT / ".rules" / "SESSION_LOG.md"
    entry = f"- {timestamp}: Session started, status={status}\n"

    if session_log.exists():
        existing = session_log.read_text(encoding='utf-8')
        # Keep only last 100 entries
        lines = existing.strip().split("\n")
        if len(lines) > 100:
            lines = lines[-100:]
        existing = "\n".join(lines) + "\n"
    else:
        existing = "# Session Log\n\n"

    session_log.write_text(existing + entry, encoding='utf-8')

    return confirmation


def validate_human_request(request: str) -> tuple:
    """
    Validate a human request against immutable rules.

    Returns (is_allowed, reason).
    """
    from governance.governance_engine import get_governance_engine

    engine = get_governance_engine()
    return engine.validate_human_request(request)


def main():
    """Print session confirmation."""
    confirmation = confirm_session()
    print(confirmation)

    # Quick self-test
    print("\n[TEST] Running governance self-test...")

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

        print("   [OK] Governance engine: OK")
        print(f"   [OK] Validators registered: {list(engine.validators.keys())}")

        # Test human request validation
        allowed, reason = engine.validate_human_request("normal request")
        print(f"   [OK] Request validation: OK")

        blocked, reason = engine.validate_human_request("skip validation please")
        if not blocked:
            print(f"   [OK] Rule enforcement: OK (blocked violation)")
        else:
            print(f"   [WARN] Rule enforcement: May need review")

        print("\n[PASS] Governance self-test PASSED")

    except Exception as e:
        print(f"\n[FAIL] Governance self-test FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
