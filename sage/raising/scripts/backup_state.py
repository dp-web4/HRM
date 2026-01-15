#!/usr/bin/env python3
"""
State Backup Utility
====================

Creates timestamped backups of raising state files and provides recovery
functionality. Protects against data loss from crashes or corruption.

Usage:
    python backup_state.py                  # Create backup of all state files
    python backup_state.py --primary        # Backup primary track only
    python backup_state.py --training       # Backup training track only
    python backup_state.py --list           # List available backups
    python backup_state.py --restore <name> # Restore from backup

Backups are stored in:
    state/backups/identity_YYYYMMDD_HHMMSS.json
    tracks/training/backups/state_YYYYMMDD_HHMMSS.json

Created: 2026-01-15 (Sprout autonomous R&D)
"""

import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional


SCRIPT_DIR = Path(__file__).parent.resolve()
RAISING_DIR = SCRIPT_DIR.parent

# State file locations
PRIMARY_STATE = RAISING_DIR / "state" / "identity.json"
PRIMARY_BACKUP_DIR = RAISING_DIR / "state" / "backups"

TRAINING_STATE = RAISING_DIR / "tracks" / "training" / "state.json"
TRAINING_BACKUP_DIR = RAISING_DIR / "tracks" / "training" / "backups"

# Retention settings
MAX_BACKUPS = 20  # Keep last N backups per track


def ensure_backup_dirs():
    """Create backup directories if they don't exist."""
    PRIMARY_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def create_backup(state_file: Path, backup_dir: Path, name_prefix: str) -> Optional[Path]:
    """Create a timestamped backup of a state file."""
    if not state_file.exists():
        print(f"  Warning: State file not found: {state_file}")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{name_prefix}_{timestamp}.json"
    backup_path = backup_dir / backup_name

    # Load and re-save to verify JSON validity
    try:
        with open(state_file) as f:
            state = json.load(f)

        with open(backup_path, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"  Created: {backup_path.name}")
        return backup_path

    except json.JSONDecodeError as e:
        print(f"  Error: Corrupt state file - {e}")
        # Still create raw backup for recovery attempts
        raw_backup = backup_dir / f"{name_prefix}_{timestamp}_corrupted.txt"
        shutil.copy2(state_file, raw_backup)
        print(f"  Raw backup: {raw_backup.name}")
        return None


def cleanup_old_backups(backup_dir: Path, prefix: str):
    """Remove old backups, keeping only MAX_BACKUPS most recent."""
    backups = sorted(backup_dir.glob(f"{prefix}_*.json"), reverse=True)

    if len(backups) > MAX_BACKUPS:
        for old_backup in backups[MAX_BACKUPS:]:
            old_backup.unlink()
            print(f"  Cleaned up: {old_backup.name}")


def list_backups(backup_dir: Path, prefix: str) -> List[Path]:
    """List available backups for a track."""
    backups = sorted(backup_dir.glob(f"{prefix}_*.json"), reverse=True)
    return backups


def restore_backup(backup_path: Path, state_file: Path) -> bool:
    """Restore a backup to the active state file."""
    if not backup_path.exists():
        print(f"  Error: Backup not found: {backup_path}")
        return False

    try:
        # Verify backup is valid JSON
        with open(backup_path) as f:
            state = json.load(f)

        # Create safety backup of current state
        if state_file.exists():
            safety_backup = state_file.with_suffix('.json.pre_restore')
            shutil.copy2(state_file, safety_backup)
            print(f"  Safety backup: {safety_backup.name}")

        # Restore
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"  Restored: {state_file.name} from {backup_path.name}")
        return True

    except json.JSONDecodeError as e:
        print(f"  Error: Backup file is corrupted - {e}")
        return False


def backup_primary():
    """Backup primary track state."""
    print("Primary Track Backup:")
    ensure_backup_dirs()
    create_backup(PRIMARY_STATE, PRIMARY_BACKUP_DIR, "identity")
    cleanup_old_backups(PRIMARY_BACKUP_DIR, "identity")


def backup_training():
    """Backup training track state."""
    print("Training Track Backup:")
    ensure_backup_dirs()
    create_backup(TRAINING_STATE, TRAINING_BACKUP_DIR, "state")
    cleanup_old_backups(TRAINING_BACKUP_DIR, "state")


def show_backups():
    """List all available backups."""
    ensure_backup_dirs()

    print()
    print("=" * 60)
    print("Available Backups")
    print("=" * 60)

    print("\nPrimary Track:")
    print("-" * 40)
    primary_backups = list_backups(PRIMARY_BACKUP_DIR, "identity")
    if primary_backups:
        for backup in primary_backups[:10]:
            size = backup.stat().st_size
            print(f"  {backup.name} ({size:,} bytes)")
        if len(primary_backups) > 10:
            print(f"  ... and {len(primary_backups) - 10} more")
    else:
        print("  No backups found")

    print("\nTraining Track:")
    print("-" * 40)
    training_backups = list_backups(TRAINING_BACKUP_DIR, "state")
    if training_backups:
        for backup in training_backups[:10]:
            size = backup.stat().st_size
            print(f"  {backup.name} ({size:,} bytes)")
        if len(training_backups) > 10:
            print(f"  ... and {len(training_backups) - 10} more")
    else:
        print("  No backups found")

    print()


def restore_from_backup(backup_name: str) -> bool:
    """Restore from a named backup."""
    ensure_backup_dirs()

    # Try primary track first
    if backup_name.startswith("identity"):
        backup_path = PRIMARY_BACKUP_DIR / backup_name
        if not backup_name.endswith('.json'):
            backup_path = PRIMARY_BACKUP_DIR / f"{backup_name}.json"
        if backup_path.exists():
            print(f"Restoring primary track from {backup_name}:")
            return restore_backup(backup_path, PRIMARY_STATE)

    # Try training track
    if backup_name.startswith("state"):
        backup_path = TRAINING_BACKUP_DIR / backup_name
        if not backup_name.endswith('.json'):
            backup_path = TRAINING_BACKUP_DIR / f"{backup_name}.json"
        if backup_path.exists():
            print(f"Restoring training track from {backup_name}:")
            return restore_backup(backup_path, TRAINING_STATE)

    # Search both directories
    for backup_dir, state_file in [(PRIMARY_BACKUP_DIR, PRIMARY_STATE),
                                    (TRAINING_BACKUP_DIR, TRAINING_STATE)]:
        for backup in backup_dir.glob(f"*{backup_name}*"):
            if backup.suffix == '.json':
                print(f"Found backup: {backup.name}")
                return restore_backup(backup, state_file)

    print(f"Error: Backup not found: {backup_name}")
    return False


def main():
    parser = argparse.ArgumentParser(description="SAGE-Sprout State Backup Utility")
    parser.add_argument("--primary", action="store_true", help="Backup primary track only")
    parser.add_argument("--training", action="store_true", help="Backup training track only")
    parser.add_argument("--list", action="store_true", help="List available backups")
    parser.add_argument("--restore", type=str, metavar="NAME", help="Restore from backup")

    args = parser.parse_args()

    if args.list:
        show_backups()
        return

    if args.restore:
        success = restore_from_backup(args.restore)
        exit(0 if success else 1)

    # Default: backup everything
    print()
    print("=" * 60)
    print("Creating State Backups")
    print("=" * 60)
    print()

    if args.primary or (not args.primary and not args.training):
        backup_primary()
        print()

    if args.training or (not args.primary and not args.training):
        backup_training()
        print()

    print("Backup complete.")
    print()


if __name__ == "__main__":
    main()
