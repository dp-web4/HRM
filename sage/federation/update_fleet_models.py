#!/usr/bin/env python3
"""
Update this machine's entry in sage-fleet-models.json.

Each machine runs this to register its current model, backend, and
LoRA capability. Called at the start of raising sessions or any time
the model configuration changes.

Usage:
    python3 -m sage.federation.update_fleet_models
    python3 -m sage.federation.update_fleet_models --model gemma3:4b --lora
    python3 -m sage.federation.update_fleet_models --dry-run
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent / "sage-fleet-models.json"


def detect_machine() -> str:
    """Detect current machine name from SAGE machine config."""
    try:
        from sage.gateway.machine_config import detect_machine as _detect
        return _detect()
    except ImportError:
        import os
        return os.environ.get("SAGE_MACHINE", "unknown")


def detect_backend() -> str:
    """Detect whether ollama or transformers is the primary backend."""
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        if result.returncode == 0:
            return "ollama"
    except Exception:
        pass
    try:
        import transformers  # noqa
        return "transformers"
    except ImportError:
        pass
    return "unknown"


def detect_transformers_available() -> bool:
    try:
        import transformers  # noqa
        return True
    except ImportError:
        return False


def detect_os() -> str:
    """Detect OS: WSL2, macOS, or Linux (Ubuntu)."""
    import platform
    if platform.system() == "Darwin":
        return "macOS"
    # Check for WSL2
    try:
        with open("/proc/version") as f:
            if "microsoft" in f.read().lower():
                return "WSL2 (Ubuntu on Windows)"
    except Exception:
        pass
    return "Linux (Ubuntu)"


def detect_lora_capable() -> bool:
    """LoRA requires transformers + peft."""
    try:
        import transformers  # noqa
        import peft  # noqa
        return True
    except ImportError:
        return False


def detect_current_model(machine: str) -> tuple[str, str]:
    """Returns (model_id, model_display) from machine config or ollama."""
    try:
        from sage.gateway.machine_config import get_machine_config
        cfg = get_machine_config(machine)
        model = cfg.get("model", "unknown")
        # Strip 'ollama:' prefix if present
        model_id = model.replace("ollama:", "")
        # Display: capitalize and format nicely
        model_display = model_id.replace(":", " ").replace("-", " ").title()
        return model_id, model_display
    except Exception:
        return "unknown", "Unknown"


def git_push(manifest_path: Path):
    """Commit and push the updated manifest."""
    repo_root = manifest_path.parent
    while repo_root != repo_root.parent:
        if (repo_root / ".git").exists():
            break
        repo_root = repo_root.parent

    rel_path = manifest_path.relative_to(repo_root)

    try:
        subprocess.run(["git", "add", str(rel_path)], cwd=repo_root, check=True)
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_root, capture_output=True
        )
        if result.returncode == 0:
            print("  No changes to commit.")
            return

        machine = detect_machine()
        subprocess.run(
            ["git", "commit", "-m", f"fleet-models: {machine} updated model entry"],
            cwd=repo_root, check=True
        )
        subprocess.run(["git", "push"], cwd=repo_root, check=True)
        print("  Pushed to remote.")
    except subprocess.CalledProcessError as e:
        print(f"  Git operation failed: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Update this machine's fleet model entry")
    parser.add_argument("--machine", help="Machine name (auto-detected if omitted)")
    parser.add_argument("--model", help="Model ID (e.g. gemma3:4b)")
    parser.add_argument("--model-display", help="Human-readable model name")
    parser.add_argument("--backend", choices=["ollama", "transformers"], help="Inference backend")
    parser.add_argument("--lora", action="store_true", help="Mark as LoRA-capable")
    parser.add_argument("--no-lora", action="store_true", help="Mark as not LoRA-capable")
    parser.add_argument("--lora-plugins", nargs="*", default=None, help="LoRA plugin names")
    parser.add_argument("--notes", help="Inference notes")
    parser.add_argument("--role", help="Machine role description")
    parser.add_argument("--no-push", action="store_true", help="Update file but don't push")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change, don't write")
    args = parser.parse_args()

    machine = args.machine or detect_machine()
    print(f"Updating fleet models entry for: {machine}")

    # Load manifest
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    if machine not in manifest["machines"]:
        print(f"WARNING: '{machine}' not in manifest. Creating new entry.")
        manifest["machines"][machine] = {}

    entry = manifest["machines"][machine]

    # Auto-detect what wasn't specified
    backend = args.backend or detect_backend()
    transformers_avail = detect_transformers_available()
    lora_capable = args.lora or (not args.no_lora and detect_lora_capable())

    if args.model:
        model_id = args.model
        model_display = args.model_display or model_id.replace(":", " ").title()
    else:
        model_id, model_display = detect_current_model(machine)
        if args.model_display:
            model_display = args.model_display

    updates = {
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "updated_by": machine,
        "os": detect_os(),
        "model": model_id,
        "model_display": model_display,
        "backend": backend,
        "transformers_available": transformers_avail,
        "lora_capable": lora_capable,
    }
    if args.lora_plugins is not None:
        updates["lora_plugins"] = args.lora_plugins
    elif "lora_plugins" not in entry:
        updates["lora_plugins"] = []
    if args.notes is not None:
        updates["inference_notes"] = args.notes
    elif "inference_notes" not in entry:
        updates["inference_notes"] = ""
    if args.role is not None:
        updates["role"] = args.role

    # Show diff
    changes = {k: (entry.get(k), v) for k, v in updates.items() if entry.get(k) != v}
    if changes:
        print("  Changes:")
        for k, (old, new) in changes.items():
            print(f"    {k}: {old!r} -> {new!r}")
    else:
        print("  No changes detected.")

    if args.dry_run:
        print("  (dry-run, not writing)")
        return

    entry.update(updates)
    manifest["machines"][machine] = entry

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Written to {MANIFEST_PATH}")

    if not args.no_push:
        git_push(MANIFEST_PATH)


if __name__ == "__main__":
    main()
