"""
Snapshot live daemon state files for git persistence.

Thin wrapper around InstancePaths.snapshot() for backward compatibility.
Raising scripts import snapshot_instance() from here; the actual logic
lives in resolver.py (InstancePaths.snapshot) which is the fleet-wide
canonical implementation.

Usage from raising sessions:
    from sage.instances.snapshot import snapshot_instance
    snapshot_instance()  # auto-detects machine/instance

Usage from CLI (preferred):
    python3 -m sage.scripts.snapshot_state --machine sprout
"""

from pathlib import Path
from typing import Optional

from sage.instances.resolver import InstancePaths


def snapshot_instance(
    instance: Optional[InstancePaths] = None,
    tag: Optional[str] = None,
) -> Optional[Path]:
    """Snapshot live state files for git persistence.

    Delegates to InstancePaths.snapshot() which writes flat files to
    snapshots/ with latest.json metadata and archive/ for identity history.

    Args:
        instance: InstancePaths to snapshot. Auto-resolves if None.
        tag: Ignored (kept for backward compat). Use snapshot_state.py CLI
             for tagged snapshots.

    Returns:
        Path to the snapshots directory, or None if nothing to snapshot.
    """
    if instance is None:
        instance = InstancePaths.resolve()

    return instance.snapshot()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Snapshot SAGE instance state (use sage.scripts.snapshot_state instead)'
    )
    parser.add_argument('--machine', help='Machine name (auto-detect if omitted)')
    parser.add_argument('--tag', help='(Ignored, kept for compat)')
    args = parser.parse_args()

    if args.machine:
        instance = InstancePaths.resolve(machine=args.machine)
    else:
        instance = InstancePaths.resolve()

    result = snapshot_instance(instance)
    if result:
        print(f"Snapshot saved to: {result}")
