#!/usr/bin/env python3
"""
Download NVIDIA Llama-3.1-Nemotron-Nano-4B-v1.1 model.

This is the CORRECT Jetson-optimized Nemotron variant:
- Pure Transformer architecture (no mamba-ssm dependency)
- Explicitly tested on Jetson AGX Thor
- Compatible with standard transformers library on ARM64
- Supports AWQ 4-bit quantization for edge deployment

Model: nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1
Size: ~8GB (BF16)
Context: 128K tokens
Architecture: Dense decoder-only Transformer (Llama 3.1 Minitron Width 4B Base)

Usage:
    python3 sage/models/download_nemotron_nano.py
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    """Download Llama Nemotron Nano 4B model."""

    # Model details
    model_id = "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1"
    local_dir = Path("model-zoo/sage/language-models/llama-nemotron-nano-4b")

    print("="*80)
    print("Downloading NVIDIA Llama-3.1-Nemotron-Nano-4B-v1.1")
    print("="*80)
    print()
    print("Model Details:")
    print(f"  HuggingFace ID: {model_id}")
    print(f"  Local Directory: {local_dir}")
    print(f"  Size: ~8GB (BF16)")
    print(f"  Context: 128K tokens")
    print(f"  Architecture: Pure Transformer (Llama 3.1 Minitron)")
    print()
    print("Jetson Compatibility:")
    print("  ✅ Pure Transformer (no mamba-ssm)")
    print("  ✅ Tested on Jetson AGX Thor")
    print("  ✅ ARM64 compatible")
    print("  ✅ Standard transformers library")
    print("  ✅ AWQ 4-bit quantization available")
    print()

    # Create directory
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("Starting download...")
        print(f"This may take several minutes depending on your connection speed.")
        print()

        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print()
        print("="*80)
        print("✅ Download Complete!")
        print("="*80)
        print()
        print(f"Model saved to: {local_dir}")
        print()
        print("Next Steps:")
        print("  1. Test basic inference:")
        print(f"     python3 sage/irp/plugins/nemotron_nano_irp.py")
        print()
        print("  2. Run benchmarks vs existing models:")
        print(f"     python3 sage/tests/test_nemotron_nano_vs_qwen.py")
        print()
        print("  3. Deploy on Jetson:")
        print(f"     - Standard: Use transformers library (works immediately)")
        print(f"     - Optimized: AWQ 4-bit quantization with TensorRT-LLM")
        print()

        # Verify download
        if not (local_dir / "config.json").exists():
            print("⚠️  Warning: config.json not found. Download may be incomplete.")
            return 1

        print("✅ Verification passed: config.json found")
        print()

        return 0

    except Exception as e:
        print()
        print("="*80)
        print("❌ Download Failed")
        print("="*80)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check internet connection")
        print("  2. Verify HuggingFace access (may need auth token for some models)")
        print("  3. Check disk space (~8GB required)")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
