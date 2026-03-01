"""
Snapshot live daemon state files for git persistence.

Live state files (identity.json, experience_buffer.json, peer_trust.json,
daemon_state.json) are gitignored because daemons write them continuously.
This module copies them to a timestamped snapshots/ directory inside the
instance, which IS tracked by git.

Usage from raising sessions:
    from sage.instances.snapshot import snapshot_instance
    snapshot_instance()  # auto-detects machine/instance

Usage from CLI:
    python3 -m sage.instances.snapshot
    python3 -m sage.instances.snapshot --machine sprout --tag session-60
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from sage.instances.resolver import InstancePaths

# Files to snapshot (relative to instance root)
SNAPSHOT_FILES = [
    'identity.json',
    'experience_buffer.json',
    'peer_trust.json',
    'daemon_state.json',
]


def snapshot_instance(
    instance: Optional[InstancePaths] = None,
    tag: Optional[str] = None,
) -> Optional[Path]:
    """Snapshot live state files to a timestamped directory.

    Args:
        instance: InstancePaths to snapshot. Auto-resolves if None.
        tag: Optional label appended to snapshot dir name (e.g. 'session-60').

    Returns:
        Path to the snapshot directory, or None if nothing to snapshot.
    """
    if instance is None:
        instance = InstancePaths.resolve()

    if not instance.root.is_dir():
        print(f"[snapshot] Instance dir not found: {instance.root}")
        return None

    # Build snapshot dir name: snapshots/20260301_012345[_tag]/
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    dirname = f"{ts}_{tag}" if tag else ts
    snap_dir = instance.root / 'snapshots' / dirname
    snap_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for fname in SNAPSHOT_FILES:
        src = instance.root / fname
        if src.exists():
            shutil.copy2(src, snap_dir / fname)
            copied += 1

    if copied == 0:
        snap_dir.rmdir()
        print("[snapshot] No state files found to snapshot")
        return None

    # Write a small manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'instance': instance.slug,
        'tag': tag,
        'files': [f for f in SNAPSHOT_FILES if (instance.root / f).exists()],
    }
    with open(snap_dir / 'snapshot.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"[snapshot] {copied} files → {snap_dir.relative_to(instance.root)}")
    return snap_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Snapshot SAGE instance state')
    parser.add_argument('--machine', help='Machine name (auto-detect if omitted)')
    parser.add_argument('--tag', help='Label for this snapshot (e.g. session-60)')
    args = parser.parse_args()

    if args.machine:
        instance = InstancePaths.resolve(machine=args.machine)
    else:
        instance = InstancePaths.resolve()

    result = snapshot_instance(instance, tag=args.tag)
    if result:
        print(f"Snapshot saved to: {result}")
