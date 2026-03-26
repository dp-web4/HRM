#!/usr/bin/env python3
"""
Check for updates and relaunch script if needed.

This script:
1. Pulls latest changes from git
2. Checks snarc version
3. Rebuilds snarc if needed
4. Relaunches the calling script if updates were pulled

Usage: Call from beginning of main() in session scripts
"""

import subprocess
import sys
import os
import json
from pathlib import Path


def get_snarc_version():
    """Get current snarc version from package.json."""
    snarc_path = Path.home() / "ai-workspace" / "snarc"
    package_json = snarc_path / "package.json"

    if not package_json.exists():
        return None

    try:
        with open(package_json) as f:
            data = json.load(f)
            return data.get("version")
    except Exception:
        return None


def check_and_update_sage():
    """Pull latest SAGE changes. Returns True if updates pulled."""
    sage_path = Path.home() / "ai-workspace" / "SAGE"

    try:
        # Check if there are remote changes
        os.chdir(sage_path)
        subprocess.run(["git", "fetch"], check=True, capture_output=True)

        # Check if behind
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            capture_output=True, text=True, check=True
        )

        behind = int(result.stdout.strip())
        if behind > 0:
            print(f"📥 SAGE is {behind} commits behind. Pulling updates...")
            subprocess.run(["git", "pull", "--rebase"], check=True)
            print("✅ SAGE updated")
            return True
        else:
            print("✅ SAGE is up to date")
            return False

    except Exception as e:
        print(f"⚠️  Could not check SAGE updates: {e}")
        return False


def check_and_update_snarc():
    """Pull and rebuild snarc if needed. Returns True if updated."""
    snarc_path = Path.home() / "ai-workspace" / "snarc"

    if not snarc_path.exists():
        print("⚠️  SNARC not found")
        return False

    try:
        old_version = get_snarc_version()

        # Pull updates
        os.chdir(snarc_path)
        subprocess.run(["git", "fetch"], check=True, capture_output=True)

        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD..origin/main"],
            capture_output=True, text=True, check=True
        )

        behind = int(result.stdout.strip())
        if behind > 0:
            print(f"📥 SNARC is {behind} commits behind. Pulling updates...")
            subprocess.run(["git", "pull", "--rebase"], check=True)

            new_version = get_snarc_version()
            if new_version != old_version:
                print(f"🔄 SNARC version changed: {old_version} → {new_version}")
                print("🔨 Rebuilding snarc...")
                subprocess.run(["npm", "run", "build"], check=True)
                print("✅ SNARC rebuilt")

                # Restart SAGE daemon if running
                try:
                    result = subprocess.run(
                        ["systemctl", "--user", "is-active", "sage-daemon-sprout"],
                        capture_output=True, text=True
                    )
                    if result.stdout.strip() == "active":
                        print("🔄 Restarting SAGE daemon...")
                        subprocess.run(
                            ["systemctl", "--user", "restart", "sage-daemon-sprout"],
                            check=True
                        )
                        print("✅ SAGE daemon restarted")
                except Exception as e:
                    print(f"⚠️  Could not restart daemon: {e}")

                return True
            else:
                print("✅ SNARC updated (version unchanged)")
                return True
        else:
            print(f"✅ SNARC {old_version} is up to date")
            return False

    except Exception as e:
        print(f"⚠️  Could not check snarc updates: {e}")
        return False


def relaunch_if_needed(script_path, argv):
    """
    Check for updates and relaunch script if needed.

    Args:
        script_path: Path to the script (__file__)
        argv: Command line arguments (sys.argv)

    Returns True if script was relaunched (caller should exit)
    """
    print("🔍 Checking for updates...")

    sage_updated = check_and_update_sage()
    snarc_updated = check_and_update_snarc()

    if sage_updated:
        print("\n🔄 SAGE was updated. Relaunching script...")
        print(f"   Command: {' '.join(argv)}")
        print()
        os.execv(sys.executable, [sys.executable] + argv)
        # Never returns if successful

    if snarc_updated:
        print("✅ SNARC was updated. Continuing with new version...")
        # Give daemon time to restart
        import time
        time.sleep(2)

    print()
    return False


if __name__ == "__main__":
    # For testing
    print("Testing update check...")
    relaunch_if_needed(__file__, sys.argv)
    print("No updates needed or script relaunched")
