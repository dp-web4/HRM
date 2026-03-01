#!/usr/bin/env python3
"""
Snapshot SAGE instance state for git persistence.

Live state files (identity.json, experience_buffer.json, peer_trust.json,
daemon_state.json) are gitignored because they're written continuously by
running daemons and cause merge conflicts across machines.

This script copies those files into a git-tracked snapshots/ directory
inside the instance dir. Raising scripts call this before git commit so
that state is persisted without cross-machine conflicts.

Usage:
    # Auto-detect instance from machine
    python3 -m sage.scripts.snapshot_state

    # Explicit machine
    python3 -m sage.scripts.snapshot_state --machine nomad

    # All instances on this machine
    python3 -m sage.scripts.snapshot_state --all
"""

import sys
import os
from pathlib import Path

# Resolve HRM root
SCRIPT_DIR = Path(__file__).parent.resolve()
HRM_ROOT = SCRIPT_DIR.parent.parent.resolve()
sys.path.insert(0, str(HRM_ROOT))

import argparse
from sage.instances.resolver import InstancePaths


def snapshot_instance(paths: InstancePaths) -> bool:
    """Snapshot a single instance. Returns True if files were snapshotted."""
    if not paths.root.is_dir():
        print(f"  Instance dir not found: {paths.root}")
        return False

    result = paths.snapshot()
    if result:
        print(f"  Snapshotted {paths.slug} → {result}")
        return True
    else:
        print(f"  No state files to snapshot for {paths.slug}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Snapshot SAGE instance state for git")
    parser.add_argument("--machine", type=str, help="Machine name (auto-detected if not set)")
    parser.add_argument("--model", type=str, help="Model override")
    parser.add_argument("--all", action="store_true", help="Snapshot all instances")
    args = parser.parse_args()

    if args.all:
        instances = InstancePaths.list_instances()
        if not instances:
            print("No instances found.")
            return
        print(f"Snapshotting {len(instances)} instances...")
        for paths in instances:
            snapshot_instance(paths)
    else:
        try:
            paths = InstancePaths.resolve(machine=args.machine, model=args.model)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        snapshot_instance(paths)


if __name__ == "__main__":
    main()
