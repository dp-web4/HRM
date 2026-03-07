#!/usr/bin/env python3
"""
Download Qwen/Qwen3.5-27B transformers model from HuggingFace.

This downloads the full model for PyTorch + PEFT LoRA training.
Model will be saved to model-zoo/qwen3.5-27b-transformers/
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Paths
MODEL_ZOO = Path(__file__).parent.parent.parent.parent / "model-zoo"
MODEL_DIR = MODEL_ZOO / "qwen3.5-27b-transformers"

def download_model():
    """Download Qwen3.5-27B transformers model."""
    print("=" * 60)
    print("Downloading Qwen/Qwen3.5-27B (HuggingFace Transformers)")
    print("=" * 60)

    print(f"\nDestination: {MODEL_DIR}")
    print(f"Repository: Qwen/Qwen3.5-27B")
    print(f"Expected size: ~54GB (BF16 weights)")
    print(f"\nThis will take some time...\n")

    # Create model zoo directory if it doesn't exist
    MODEL_ZOO.mkdir(parents=True, exist_ok=True)

    try:
        # Download the model
        # ignore_patterns to skip unnecessary files and reduce download size
        downloaded_path = snapshot_download(
            repo_id="Qwen/Qwen3.5-27B",
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
            ignore_patterns=[
                "*.md",  # Skip markdown files
                "*.txt",  # Skip text files
                ".gitattributes",
                "*.safetensors.index.json",  # We'll use the consolidated weights
            ],
            resume_download=True,  # Resume if interrupted
        )

        print(f"\n{'='*60}")
        print("✓ Download complete!")
        print(f"{'='*60}")
        print(f"\nModel saved to: {downloaded_path}")

        # List downloaded files
        print("\nDownloaded files:")
        files = sorted(MODEL_DIR.glob("*"))
        total_size = 0
        for f in files:
            if f.is_file():
                size_gb = f.stat().st_size / 1e9
                total_size += size_gb
                print(f"  {f.name}: {size_gb:.2f}GB")

        print(f"\nTotal size: {total_size:.2f}GB")

        return True

    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user")
        print(f"Partial download saved to: {MODEL_DIR}")
        print("Run this script again to resume")
        return False

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print(f"Python: {sys.version}")
    print(f"Working directory: {Path.cwd()}\n")

    # Check disk space
    import shutil
    stat = shutil.disk_usage("/")
    free_gb = stat.free / 1e9
    print(f"Disk space available: {free_gb:.1f}GB")

    if free_gb < 60:
        print(f"⚠ WARNING: Low disk space ({free_gb:.1f}GB free, need ~54GB)")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted")
            sys.exit(1)

    success = download_model()
    sys.exit(0 if success else 1)
