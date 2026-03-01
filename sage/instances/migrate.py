"""
Migrate existing SAGE state files into instance directories.

One-shot migration for existing machines. Copies files (not moves) so
original data is preserved. Idempotent — safe to run multiple times.

Usage:
    # Migrate all known machines (dry run first)
    python3 -m sage.instances.migrate --dry-run

    # Actually migrate
    python3 -m sage.instances.migrate

    # Migrate a specific machine
    python3 -m sage.instances.migrate --machine sprout
"""

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import List, Tuple

from sage.instances.resolver import INSTANCES_ROOT, make_slug
from sage.instances.init import init_instance


# Raising directory (source of all existing state)
RAISING_DIR = Path(__file__).parent.parent / "raising"


def _copy_file(src: Path, dst: Path, dry_run: bool = False) -> bool:
    """Copy a single file, creating parent dirs. Returns True if copied."""
    if not src.exists():
        return False
    if dry_run:
        print(f"  [DRY] {src.name} → {dst}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_dir_contents(src_dir: Path, dst_dir: Path, pattern: str = "*.json",
                       dry_run: bool = False) -> int:
    """Copy all matching files from src_dir to dst_dir. Returns count."""
    if not src_dir.exists():
        return 0
    count = 0
    for f in sorted(src_dir.glob(pattern)):
        if f.is_file():
            if _copy_file(f, dst_dir / f.name, dry_run):
                count += 1
    return count


def _patch_identity_name(identity_path: Path, new_name: str):
    """Patch identity.name in an identity.json file."""
    data = json.loads(identity_path.read_text())
    if 'identity' in data and data['identity'].get('name') != new_name:
        data['identity']['name'] = new_name
        identity_path.write_text(json.dumps(data, indent=2) + '\n')


def migrate_sprout(dry_run: bool = False) -> List[str]:
    """Migrate Sprout (primary track owner) to sprout-qwen2.5-0.5b/."""
    slug = "sprout-qwen2.5-0.5b"
    instance_dir = INSTANCES_ROOT / slug
    state = RAISING_DIR / "state"
    log = []

    if not instance_dir.exists():
        if dry_run:
            log.append(f"  [DRY] Would init instance: {slug}")
        else:
            init_instance("sprout", "qwen2.5-0.5b", device_hint="cuda", backend="local")
        log.append(f"  Initialized {slug}")

    # Identity (primary owner)
    if _copy_file(state / "identity.json", instance_dir / "identity.json", dry_run):
        log.append("  identity.json")

    # Experience buffer (main 1.2MB)
    if _copy_file(state / "experience_buffer.json", instance_dir / "experience_buffer.json", dry_run):
        log.append("  experience_buffer.json (primary)")

    # Latent exploration state
    if _copy_file(state / "latent_exploration_state.json", instance_dir / "latent_exploration_state.json", dry_run):
        log.append("  latent_exploration_state.json")

    # Daemon state (generic — sprout used the plain one)
    if _copy_file(state / "daemon_state.json", instance_dir / "daemon_state.json", dry_run):
        log.append("  daemon_state.json")

    # Sessions: text (109), conversations (72), latent_exploration (91), explorations (7)
    sessions_dst = instance_dir / "sessions"
    for subdir in ["text", "conversations", "latent_exploration", "explorations"]:
        src = RAISING_DIR / "sessions" / subdir
        n = _copy_dir_contents(src, sessions_dst, "*.json", dry_run)
        if n:
            log.append(f"  sessions/{subdir}: {n} files")

    # Training sessions
    training_src = RAISING_DIR / "tracks" / "training"
    if _copy_file(training_src / "state.json", instance_dir / "training" / "state.json", dry_run):
        log.append("  training/state.json")
    n = _copy_dir_contents(training_src / "sessions", instance_dir / "training" / "sessions", "*.json", dry_run)
    if n:
        log.append(f"  training/sessions: {n} files")

    # Backups
    n = _copy_dir_contents(state / "backups", instance_dir / "backups", "*.json", dry_run)
    if n:
        log.append(f"  backups: {n} files")

    # Also copy the experience buffer backups from state/ root
    for bak in sorted(state.glob("experience_buffer.backup_*")):
        if _copy_file(bak, instance_dir / "backups" / bak.name, dry_run):
            log.append(f"  backups/{bak.name}")
    for bak in sorted(state.glob("experience_buffer.json.backup_*")):
        if _copy_file(bak, instance_dir / "backups" / bak.name, dry_run):
            log.append(f"  backups/{bak.name}")

    # Update instance.json
    if not dry_run:
        manifest = instance_dir / "instance.json"
        data = json.loads(manifest.read_text())
        data['has_training_track'] = True
        data['has_lora'] = True
        data['device_hint'] = 'cuda'
        data['backend'] = 'local'
        manifest.write_text(json.dumps(data, indent=2) + '\n')

    return log


def migrate_thor(dry_run: bool = False) -> List[str]:
    """Migrate Thor to thor-qwen2.5-14b/."""
    slug = "thor-qwen2.5-14b"
    instance_dir = INSTANCES_ROOT / slug
    state = RAISING_DIR / "state"
    log = []

    if not instance_dir.exists():
        if dry_run:
            log.append(f"  [DRY] Would init instance: {slug}")
        else:
            init_instance("thor", "qwen2.5-14b", device_hint="cuda", backend="local")
        log.append(f"  Initialized {slug}")

    # Identity — fork from shared identity.json, patch name
    if _copy_file(state / "identity.json", instance_dir / "identity.json", dry_run):
        log.append("  identity.json (forked from shared, patching name)")
        if not dry_run:
            _patch_identity_name(instance_dir / "identity.json", "SAGE-Thor")

    # Experience buffer (copy shared — Thor wrote to the same file)
    if _copy_file(state / "experience_buffer.json", instance_dir / "experience_buffer.json", dry_run):
        log.append("  experience_buffer.json (copy of shared)")

    # R14B sessions
    r14b_sessions = RAISING_DIR / "tracks" / "raising-14b" / "sessions"
    sessions_dst = instance_dir / "sessions"
    n = _copy_dir_contents(r14b_sessions, sessions_dst, "*.json", dry_run)
    if n:
        log.append(f"  sessions (R14B): {n} files")

    # R14B state
    r14b_state = RAISING_DIR / "tracks" / "raising-14b" / "state.json"
    if _copy_file(r14b_state, instance_dir / "training" / "state.json", dry_run):
        log.append("  training/state.json (R14B)")

    if not dry_run:
        manifest = instance_dir / "instance.json"
        data = json.loads(manifest.read_text())
        data['device_hint'] = 'cuda'
        data['backend'] = 'local'
        manifest.write_text(json.dumps(data, indent=2) + '\n')

    return log


def migrate_legion(dry_run: bool = False) -> List[str]:
    """Migrate Legion to legion-qwen2-0.5b/."""
    slug = "legion-qwen2-0.5b"
    instance_dir = INSTANCES_ROOT / slug
    state = RAISING_DIR / "state"
    log = []

    if not instance_dir.exists():
        if dry_run:
            log.append(f"  [DRY] Would init instance: {slug}")
        else:
            init_instance("legion", "qwen2:0.5b", device_hint="cuda")
        log.append(f"  Initialized {slug}")

    # Identity (legion-specific)
    if _copy_file(state / "legion_identity.json", instance_dir / "identity.json", dry_run):
        log.append("  identity.json (from legion_identity.json)")

    # Experience buffer (legion-specific)
    if _copy_file(state / "experience_buffer_legion_qwen2_0.5b.json", instance_dir / "experience_buffer.json", dry_run):
        log.append("  experience_buffer.json")

    # Peer trust
    if _copy_file(state / "peer_trust_legion.json", instance_dir / "peer_trust.json", dry_run):
        log.append("  peer_trust.json")

    # Daemon state
    if _copy_file(state / "daemon_state_legion.json", instance_dir / "daemon_state.json", dry_run):
        log.append("  daemon_state.json")

    # Sessions
    sessions_dst = instance_dir / "sessions"
    n = _copy_dir_contents(RAISING_DIR / "sessions" / "legion", sessions_dst, "*.json", dry_run)
    if n:
        log.append(f"  sessions: {n} files")

    return log


def migrate_mcnugget(dry_run: bool = False) -> List[str]:
    """Migrate McNugget to mcnugget-gemma3-12b/."""
    slug = "mcnugget-gemma3-12b"
    instance_dir = INSTANCES_ROOT / slug
    state = RAISING_DIR / "state"
    log = []

    if not instance_dir.exists():
        if dry_run:
            log.append(f"  [DRY] Would init instance: {slug}")
        else:
            init_instance("mcnugget", "gemma3:12b", device_hint="mps")
        log.append(f"  Initialized {slug}")

    # Identity (mcnugget-specific)
    if _copy_file(state / "mcnugget_identity.json", instance_dir / "identity.json", dry_run):
        log.append("  identity.json (from mcnugget_identity.json)")

    # Experience buffer (mcnugget-specific)
    if _copy_file(state / "experience_buffer_mcnugget_gemma3_12b.json", instance_dir / "experience_buffer.json", dry_run):
        log.append("  experience_buffer.json")

    # Peer trust
    if _copy_file(state / "peer_trust_mcnugget.json", instance_dir / "peer_trust.json", dry_run):
        log.append("  peer_trust.json")

    # Sessions
    sessions_dst = instance_dir / "sessions"
    n = _copy_dir_contents(RAISING_DIR / "sessions" / "mcnugget", sessions_dst, "*.json", dry_run)
    if n:
        log.append(f"  sessions: {n} files")

    return log


def migrate_cbp(dry_run: bool = False) -> List[str]:
    """Migrate CBP to cbp-tinyllama-latest/."""
    slug = "cbp-tinyllama-latest"
    instance_dir = INSTANCES_ROOT / slug
    state = RAISING_DIR / "state"
    log = []

    if not instance_dir.exists():
        if dry_run:
            log.append(f"  [DRY] Would init instance: {slug}")
        else:
            init_instance("cbp", "tinyllama:latest", device_hint="cpu")
        log.append(f"  Initialized {slug}")

    # Identity — fork from shared identity.json, patch name
    if _copy_file(state / "identity.json", instance_dir / "identity.json", dry_run):
        log.append("  identity.json (forked from shared, patching name)")
        if not dry_run:
            _patch_identity_name(instance_dir / "identity.json", "SAGE-CBP")

    # Experience buffer (cbp-specific)
    if _copy_file(state / "experience_buffer_cbp_tinyllama_latest.json", instance_dir / "experience_buffer.json", dry_run):
        log.append("  experience_buffer.json")

    # Peer trust
    if _copy_file(state / "peer_trust_cbp.json", instance_dir / "peer_trust.json", dry_run):
        log.append("  peer_trust.json")

    return log


def migrate_nomad(dry_run: bool = False) -> List[str]:
    """Initialize Nomad from seed (no existing data to migrate)."""
    slug = "nomad-gemma3-4b"
    instance_dir = INSTANCES_ROOT / slug
    log = []

    if not instance_dir.exists():
        if dry_run:
            log.append(f"  [DRY] Would init instance: {slug}")
        else:
            init_instance("nomad", "gemma3:4b", device_hint="cuda")
        log.append(f"  Initialized {slug} (from seed, no existing data)")
    else:
        log.append(f"  {slug} already exists, skipping")

    return log


_MIGRATORS = {
    'sprout': migrate_sprout,
    'thor': migrate_thor,
    'legion': migrate_legion,
    'mcnugget': migrate_mcnugget,
    'cbp': migrate_cbp,
    'nomad': migrate_nomad,
}


def migrate_all(dry_run: bool = False, machines: List[str] = None):
    """Run migration for all (or specified) machines."""
    targets = machines or list(_MIGRATORS.keys())
    mode = "[DRY RUN]" if dry_run else "[MIGRATE]"

    print(f"\n{mode} SAGE Instance Migration")
    print(f"{'=' * 50}")
    print(f"Source: {RAISING_DIR}")
    print(f"Target: {INSTANCES_ROOT}")
    print()

    total_actions = 0
    for machine in targets:
        migrator = _MIGRATORS.get(machine)
        if not migrator:
            print(f"[WARN] Unknown machine: {machine}, skipping")
            continue

        print(f"--- {machine} ---")
        log = migrator(dry_run=dry_run)
        for line in log:
            print(line)
        total_actions += len(log)
        print()

    print(f"{'=' * 50}")
    print(f"Total: {total_actions} actions for {len(targets)} machines")

    if dry_run:
        print("\nThis was a dry run. Run without --dry-run to execute.")

    # Verify
    if not dry_run:
        print("\nVerification:")
        for machine in targets:
            slug = _MIGRATORS.get(machine) and make_slug(machine,
                {'sprout': 'qwen2.5-0.5b', 'thor': 'qwen2.5-14b',
                 'legion': 'qwen2:0.5b', 'mcnugget': 'gemma3:12b',
                 'cbp': 'tinyllama:latest', 'nomad': 'gemma3:4b'}[machine])
            idir = INSTANCES_ROOT / slug
            if idir.exists():
                files = list(idir.rglob("*"))
                file_count = sum(1 for f in files if f.is_file())
                print(f"  {slug}: {file_count} files")
            else:
                print(f"  {slug}: MISSING")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing SAGE state into instance directories.",
        epilog=(
            "Examples:\n"
            "  python3 -m sage.instances.migrate --dry-run\n"
            "  python3 -m sage.instances.migrate\n"
            "  python3 -m sage.instances.migrate --machine legion\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')
    parser.add_argument('--machine', action='append', dest='machines',
                        help='Migrate only this machine (can repeat)')
    args = parser.parse_args()

    migrate_all(dry_run=args.dry_run, machines=args.machines)


if __name__ == "__main__":
    main()
