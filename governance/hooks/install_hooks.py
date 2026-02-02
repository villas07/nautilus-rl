#!/usr/bin/env python3
"""
Install Git Hooks

Sets up all governance hooks:
- pre-commit: Code validation before commits
- pre-push: Full tests before push
- post-merge: Integrity verification after merge
"""

import os
import shutil
import stat
import sys
from pathlib import Path


def install_hooks():
    """Install all governance hooks to .git/hooks"""
    project_root = Path(__file__).parent.parent.parent
    git_hooks_dir = project_root / ".git" / "hooks"
    governance_hooks_dir = Path(__file__).parent

    if not git_hooks_dir.exists():
        print("[ERROR] .git/hooks directory not found")
        print("Make sure you're in a git repository")
        return False

    hooks = ["pre-commit", "pre-push", "post-merge"]
    installed = []
    failed = []

    for hook_name in hooks:
        src = governance_hooks_dir / hook_name
        dst = git_hooks_dir / hook_name

        if not src.exists():
            print(f"[WARNING] {hook_name} hook source not found")
            failed.append(hook_name)
            continue

        try:
            # Backup existing hook
            if dst.exists():
                backup = dst.with_suffix(".backup")
                shutil.copy(dst, backup)
                print(f"  Backed up existing {hook_name} to {backup.name}")

            # Copy hook
            shutil.copy(src, dst)

            # Make executable
            st = os.stat(dst)
            os.chmod(dst, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

            installed.append(hook_name)
            print(f"[OK] Installed {hook_name}")

        except Exception as e:
            print(f"[ERROR] Failed to install {hook_name}: {e}")
            failed.append(hook_name)

    print("")
    print("=" * 50)
    if installed:
        print(f"[OK] Installed {len(installed)} hooks: {', '.join(installed)}")
    if failed:
        print(f"[WARN] Failed to install: {', '.join(failed)}")

    print("")
    print("Hooks will enforce:")
    print("  - pre-commit: Block immutable file changes, validate code")
    print("  - pre-push: Full integrity check, run tests")
    print("  - post-merge: Verify integrity after pulls/merges")
    print("")

    return len(failed) == 0


def uninstall_hooks():
    """Remove governance hooks."""
    project_root = Path(__file__).parent.parent.parent
    git_hooks_dir = project_root / ".git" / "hooks"

    hooks = ["pre-commit", "pre-push", "post-merge"]
    removed = []

    for hook_name in hooks:
        hook = git_hooks_dir / hook_name
        backup = hook.with_suffix(".backup")

        if hook.exists():
            hook.unlink()
            removed.append(hook_name)
            print(f"[OK] Removed {hook_name}")

            if backup.exists():
                shutil.move(backup, hook)
                print(f"     Restored {hook_name} from backup")

    if removed:
        print(f"\n[OK] Uninstalled {len(removed)} hooks")
    else:
        print("[OK] No hooks to uninstall")

    return True


def verify_hooks():
    """Verify that hooks are properly installed."""
    project_root = Path(__file__).parent.parent.parent
    git_hooks_dir = project_root / ".git" / "hooks"

    hooks = ["pre-commit", "pre-push", "post-merge"]
    status = {}

    for hook_name in hooks:
        hook = git_hooks_dir / hook_name

        if not hook.exists():
            status[hook_name] = "NOT INSTALLED"
        elif not os.access(hook, os.X_OK):
            status[hook_name] = "NOT EXECUTABLE"
        else:
            # Check if it's our hook (contains governance)
            content = hook.read_text()
            if "governance" in content.lower():
                status[hook_name] = "OK"
            else:
                status[hook_name] = "CUSTOM (not governance)"

    print("Hook Status:")
    for hook_name, state in status.items():
        icon = "[OK]" if state == "OK" else "[--]"
        print(f"  {icon} {hook_name}: {state}")

    return all(s == "OK" for s in status.values())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "uninstall":
            uninstall_hooks()
        elif sys.argv[1] == "verify":
            sys.exit(0 if verify_hooks() else 1)
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: install_hooks.py [uninstall|verify]")
            sys.exit(1)
    else:
        success = install_hooks()
        sys.exit(0 if success else 1)
