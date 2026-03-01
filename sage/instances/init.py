"""
Initialize a new SAGE instance from the seed template.

Usage:
    # External user bootstrapping a new machine
    python3 -m sage.instances.init --machine mybox --model llama3.2:3b

    # Internal machine initialization
    python3 -m sage.instances.init --machine nomad --model gemma3:4b

    # Force re-init (overwrites existing)
    python3 -m sage.instances.init --machine nomad --model gemma3:4b --force
"""

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

from sage.instances.resolver import INSTANCES_ROOT, make_slug, model_to_slug


SEED_DIR = INSTANCES_ROOT / "_seed"

# Model family lookup (best-effort)
_MODEL_FAMILIES = {
    'qwen': 'alibaba-qwen',
    'gemma': 'google-gemma',
    'llama': 'meta-llama',
    'tinyllama': 'tinyllama',
    'phi': 'microsoft-phi',
    'smollm': 'huggingface-smollm',
    'mistral': 'mistral-ai',
}


def _guess_family(model: str) -> str:
    """Guess model family from model name."""
    lower = model.lower()
    for prefix, family in _MODEL_FAMILIES.items():
        if prefix in lower:
            return family
    return "unknown"


def init_instance(machine: str, model: str, force: bool = False,
                  device_hint: str = "cpu", backend: str = "ollama",
                  operator_name: str = "operator") -> Path:
    """Create a new instance directory from the seed template.

    Args:
        machine: Machine name (e.g. 'mybox', 'nomad')
        model: Model identifier (e.g. 'gemma3:4b', 'llama3.2:3b')
        force: Overwrite existing instance directory
        device_hint: Device hint for instance.json
        backend: Backend type ('ollama' or 'local')
        operator_name: Name for the operator relationship (default: 'operator')

    Returns:
        Path to created instance directory.

    Raises:
        FileExistsError: If instance dir exists and force=False
    """
    slug = make_slug(machine, model)
    instance_dir = INSTANCES_ROOT / slug

    if instance_dir.exists():
        if not force:
            raise FileExistsError(
                f"Instance directory already exists: {instance_dir}\n"
                f"Use --force to overwrite."
            )
        shutil.rmtree(instance_dir)

    # Copy seed template
    shutil.copytree(SEED_DIR, instance_dir)

    today = date.today().isoformat()
    model_slug = model_to_slug(model)
    family = _guess_family(model)

    # Placeholder replacements
    replacements = {
        '__MACHINE__': machine,
        '__MODEL__': model,
        '__FAMILY__': family,
        '__SLUG__': slug,
        '__DATE__': today,
        '__OPERATOR__': operator_name,
    }

    # Process all JSON and Markdown files — replace placeholders
    for template_file in list(instance_dir.rglob('*.json')) + list(instance_dir.rglob('*.md')):
        text = template_file.read_text()
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        template_file.write_text(text)

    # Update instance.json with runtime-specific fields
    manifest_path = instance_dir / "instance.json"
    manifest = json.loads(manifest_path.read_text())
    manifest['device_hint'] = device_hint
    manifest['backend'] = backend
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    # Create empty state files
    (instance_dir / "experience_buffer.json").write_text('[]\n')
    (instance_dir / "peer_trust.json").write_text('{}\n')
    (instance_dir / "daemon_state.json").write_text('{}\n')

    return instance_dir


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a new SAGE instance from the seed template.",
        epilog=(
            "Examples:\n"
            "  python3 -m sage.instances.init --machine mybox --model llama3.2:3b\n"
            "  python3 -m sage.instances.init --machine nomad --model gemma3:4b --device cuda\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--machine', required=True, help='Machine name (e.g. mybox, nomad)')
    parser.add_argument('--model', required=True, help='Model identifier (e.g. gemma3:4b, llama3.2:3b)')
    parser.add_argument('--device', default='cpu', help='Device hint (cpu, cuda, mps)')
    parser.add_argument('--backend', default='ollama', choices=['ollama', 'local'],
                        help='Model backend')
    parser.add_argument('--operator-name', default='operator',
                        help='Name for the operator relationship (default: operator)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing instance')
    args = parser.parse_args()

    try:
        instance_dir = init_instance(
            machine=args.machine,
            model=args.model,
            force=args.force,
            device_hint=args.device,
            backend=args.backend,
            operator_name=args.operator_name,
        )
    except FileExistsError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    slug = instance_dir.name
    print(f"SAGE instance initialized: {slug}")
    print(f"  Directory: {instance_dir}")
    print(f"  Identity:  {instance_dir / 'identity.json'}")
    print(f"  Manifest:  {instance_dir / 'instance.json'}")
    print()
    print("Next steps:")
    print(f"  1. Start the daemon:")
    print(f"     SAGE_MACHINE={args.machine} python3 -m sage.gateway.sage_daemon")
    print(f"  2. Or set the instance explicitly:")
    print(f"     SAGE_INSTANCE={slug} python3 -m sage.gateway.sage_daemon")


if __name__ == "__main__":
    main()
