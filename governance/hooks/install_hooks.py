#!/usr/bin/env python3
"""
Install Git Hooks

Sets up pre-commit hook for automatic governance validation.
"""

import os
import shutil
import stat
from pathlib import Path


def install_hooks():
    """Install governance hooks to .git/hooks"""
    project_root = Path(__file__).parent.parent.parent
    git_hooks_dir = project_root / ".git" / "hooks"
    governance_hooks_dir = project_root / "governance" / "hooks"

    if not git_hooks_dir.exists():
        print("Error: .git/hooks directory not found")
        print("Make sure you're in a git repository")
        return False

    # Install pre-commit hook
    src = governance_hooks_dir / "pre-commit"
    dst = git_hooks_dir / "pre-commit"

    if src.exists():
        # Backup existing hook
        if dst.exists():
            backup = dst.with_suffix(".backup")
            shutil.copy(dst, backup)
            print(f"Backed up existing hook to {backup}")

        # Copy hook
        shutil.copy(src, dst)

        # Make executable
        st = os.stat(dst)
        os.chmod(dst, st.st_mode | stat.S_IEXEC)

        print(f"[OK] Installed pre-commit hook to {dst}")
        return True
    else:
        print(f"Warning: {src} not found")
        return False


def uninstall_hooks():
    """Remove governance hooks."""
    project_root = Path(__file__).parent.parent.parent
    git_hooks_dir = project_root / ".git" / "hooks"

    hook = git_hooks_dir / "pre-commit"
    backup = hook.with_suffix(".backup")

    if hook.exists():
        hook.unlink()
        print(f"Removed {hook}")

        if backup.exists():
            shutil.move(backup, hook)
            print(f"Restored backup from {backup}")

        return True

    print("No hooks to uninstall")
    return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "uninstall":
        uninstall_hooks()
    else:
        install_hooks()
