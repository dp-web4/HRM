#!/usr/bin/env python3
"""
Download NVIDIA Llama-3.1-Nemotron-4B-Instruct to model zoo.

This script downloads the Nemotron 4B model from HuggingFace to the local
model zoo for SAGE integration.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def download_nemotron():
    """Download Nemotron 4B Instruct model."""

    model_id = "nvidia/Nemotron-H-4B-Instruct-128K"
    local_dir = "model-zoo/sage/language-models/nemotron-h-4b-instruct-128k"

    print("=" * 80)
    print("Downloading NVIDIA Nemotron-H-4B-Instruct-128K")
    print("=" * 80)
    print()
    print(f"📥 Source: {model_id}")
    print(f"💾 Destination: {local_dir}")
    print()
    print("This will download:")
    print("  - Model weights (~8GB for BF16)")
    print("  - Tokenizer files")
    print("  - Configuration files")
    print()
    print("⏱️  Estimated time: 5-10 minutes (depending on connection)")
    print()

    try:
        # Download with progress
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print()
        print("=" * 80)
        print("✅ Download Complete!")
        print("=" * 80)
        print()
        print(f"Model saved to: {Path(local_dir).absolute()}")
        print()

        # Check downloaded files
        model_path = Path(local_dir)
        files = list(model_path.rglob("*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())

        print("📊 Download Summary:")
        print(f"  Total files: {len([f for f in files if f.is_file()])}")
        print(f"  Total size: {total_size / (1024**3):.2f} GB")
        print()
        print("Key files:")
        for pattern in ["*.safetensors", "*.json", "tokenizer*"]:
            matching = list(model_path.glob(pattern))
            if matching:
                for f in matching[:5]:  # Show first 5
                    size_mb = f.stat().st_size / (1024**2)
                    print(f"  ✓ {f.name} ({size_mb:.1f} MB)")
        print()

        return 0

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ Download Failed")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify HuggingFace access (may need token for gated models)")
        print("  3. Check disk space (need ~10GB free)")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(download_nemotron())
