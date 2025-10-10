#!/usr/bin/env python3
"""
Add SAGE/ARC metadata to the cached GR00T model.

This patches the metadata.json in the HuggingFace cache to include our custom embodiment.
"""

import json
from pathlib import Path


def patch_groot_metadata():
    """Add sage_arc embodiment to GR00T's metadata.json."""

    # Path to cached model metadata
    cache_path = Path.home() / ".cache" / "huggingface" / "hub"
    model_dirs = list(cache_path.glob("models--nvidia--GR00T-N1.5-3B"))

    if not model_dirs:
        print("❌ GR00T model not found in cache")
        return False

    # Find the actual metadata file (it's a symlink)
    metadata_file = model_dirs[0] / "snapshots" / "869830fc749c35f34771aa5209f923ac57e4564e" / "experiment_cfg" / "metadata.json"

    # Resolve symlink to actual blob file
    if metadata_file.is_symlink():
        actual_file = metadata_file.resolve()
    else:
        actual_file = metadata_file

    print(f"📂 Found metadata at: {actual_file}")

    # Load existing metadata
    with open(actual_file, 'r') as f:
        existing_metadata = json.load(f)

    print(f"📊 Existing embodiments: {list(existing_metadata.keys())}")

    # Load our custom metadata
    custom_metadata_path = Path("/home/dp/ai-workspace/HRM/sage/orchestration/groot_arc_setup/metadata.json")
    with open(custom_metadata_path, 'r') as f:
        custom_metadata = json.load(f)

    # Merge
    existing_metadata.update(custom_metadata)

    # Write back
    with open(actual_file, 'w') as f:
        json.dump(existing_metadata, f, indent=2)

    print(f"✅ Added new embodiments: {list(custom_metadata.keys())}")
    print(f"📊 All embodiments now: {list(existing_metadata.keys())}")

    return True


def main():
    print("=" * 80)
    print("Patching GR00T Model Metadata")
    print("=" * 80)

    success = patch_groot_metadata()

    if success:
        print("\n✅ Metadata patched successfully!")
        print("   You can now use EmbodimentTag.NEW_EMBODIMENT with Gr00tPolicy")
    else:
        print("\n❌ Failed to patch metadata")


if __name__ == "__main__":
    main()
