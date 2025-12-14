#!/usr/bin/env python3
"""
Download Qwen3-Omni-30B Model for Thor SAGE

Downloads the Qwen3-Omni-30B-A3B-Instruct model - a TRUE omni-modal model
that natively processes audio, video, and text without translation layers.

This eliminates:
- Separate audio/vision/text models
- VAE translation between modalities
- Piecemeal processing overhead

Model: Qwen/Qwen3-Omni-30B-A3B-Instruct
Size: ~30GB (A3B indicates optimized quantization)
Capabilities: Audio, Video, Text - unified processing

For Thor (122GB unified memory):
- Can run alongside 14B text model when needed
- Or run standalone for full omni capabilities

Usage:
    python3 sage/setup/download_qwen3_omni_30b.py
"""

import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def download_qwen3_omni():
    """Download Qwen3-Omni-30B-A3B-Instruct model."""

    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    local_dir = Path("model-zoo/sage/omni-modal/qwen3-omni-30b")

    print("=" * 70)
    print("Downloading Qwen3-Omni-30B-A3B-Instruct for Thor SAGE")
    print("=" * 70)
    print()
    print(f"Model: {model_id}")
    print(f"Target: {local_dir}")
    print(f"Size: ~30GB (A3B optimized)")
    print()
    print("Capabilities:")
    print("  ✅ Audio processing (native)")
    print("  ✅ Video processing (native)")
    print("  ✅ Text processing (native)")
    print("  ✅ Unified multi-modal reasoning")
    print()
    print("This eliminates the need for:")
    print("  ❌ Separate audio/vision models")
    print("  ❌ VAE translation layers")
    print("  ❌ Modality-specific pipelines")
    print()
    print("Progress:")
    print("-" * 70)

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.bin"],  # Use safetensors only
        )

        print()
        print("=" * 70)
        print("✅ Download Complete!")
        print("=" * 70)
        print()
        print(f"Model location: {local_dir.absolute()}")
        print()
        print("Next steps:")
        print("1. Update multi_model_loader.py to include Qwen3-Omni option")
        print("2. Create omni-modal IRP plugin for audio/video/text")
        print("3. Test memory usage (30GB + headroom)")
        print("4. Benchmark unified processing vs piecemeal approach")
        print("5. Update Thor identity to include omni-modal capabilities")
        print()

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check internet connection")
        print("2. Verify HuggingFace hub access")
        print("3. Check available disk space (need ~40GB free)")
        print(f"4. Verify model exists: https://huggingface.co/{model_id}")
        print("5. Try manual download if needed")
        return False

    return True


def main():
    """Main entry point."""
    success = download_qwen3_omni()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
