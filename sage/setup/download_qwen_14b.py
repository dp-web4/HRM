#!/usr/bin/env python3
"""
Download Qwen2.5-14B-Instruct for Thor SAGE H-Module

This downloads the base instruct model which will serve as Thor's
primary reasoning module (H-Module).

Model: Qwen/Qwen2.5-14B-Instruct
Size: ~28GB
Purpose: Strategic reasoning for Thor SAGE instance
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def download_qwen_14b():
    """Download Qwen2.5-14B-Instruct to model zoo."""

    model_id = "Qwen/Qwen2.5-14B-Instruct"
    local_dir = Path("model-zoo/sage/epistemic-stances/qwen2.5-14b/base-instruct")

    print("=" * 70)
    print("Downloading Qwen2.5-14B-Instruct for Thor SAGE")
    print("=" * 70)
    print()
    print(f"Model: {model_id}")
    print(f"Target: {local_dir}")
    print(f"Size: ~28GB (this will take a while)")
    print()
    print("Progress:")
    print("-" * 70)

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.bin"],  # We want safetensors
        )

        print()
        print("=" * 70)
        print("✅ Download Complete!")
        print("=" * 70)
        print()
        print(f"Model location: {local_dir.absolute()}")
        print()
        print("Next steps:")
        print("1. Configure Thor H-Module to use this model")
        print("2. Set up epistemic stancing fine-tuning")
        print("3. Test first awakening with 14B reasoning")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ Download Failed")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("- Check internet connection")
        print("- Verify disk space (~30GB needed)")
        print("- Try running again (download will resume)")
        print()

        return False

if __name__ == "__main__":
    success = download_qwen_14b()
    sys.exit(0 if success else 1)
