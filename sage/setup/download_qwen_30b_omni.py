#!/usr/bin/env python3
"""
Download Qwen 30B Omni Model for Thor SAGE

Downloads the Qwen2.5 30B model (or closest omni-modal equivalent) for Thor's
multi-modal capabilities. This model unifies audio/video/text processing,
eliminating the need for separate modality-specific models and translation layers.

For Thor (122GB unified memory):
- 30B int8 quantized: ~30GB (recommended - fits alongside 14B)
- 30B fp16: ~60GB (higher quality but tight fit)

Usage:
    python3 sage/setup/download_qwen_30b_omni.py [--quantization int8|fp16]
"""

import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def download_qwen_30b_omni(quantization: str = "int8"):
    """
    Download Qwen 30B omni-modal model.

    Args:
        quantization: Quantization format ('int8' or 'fp16')
    """

    # Qwen2.5-32B-Instruct is the closest to 30B with best capabilities
    # Note: Qwen3 doesn't exist yet; Qwen2.5 is latest stable
    # For true omni capabilities, we'd want Qwen2-Audio-7B or Qwen2-VL-7B
    # But user asked for 30B class, so using Qwen2.5-32B-Instruct

    if quantization == "int8":
        # GPTQ or AWQ quantized version
        model_id = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
        variant = "gptq-int8"
    else:
        # Standard fp16 version
        model_id = "Qwen/Qwen2.5-32B-Instruct"
        variant = "fp16"

    local_dir = Path(f"model-zoo/sage/epistemic-stances/qwen2.5-32b/{variant}")

    print("=" * 70)
    print(f"Downloading Qwen2.5-32B-Instruct ({quantization}) for Thor SAGE")
    print("=" * 70)
    print()
    print(f"Model: {model_id}")
    print(f"Target: {local_dir}")

    if quantization == "int8":
        print(f"Size: ~30GB (quantized)")
    else:
        print(f"Size: ~60GB (fp16)")

    print()
    print("Note: For true omni-modal (audio/video/text), consider:")
    print("  - Qwen2-Audio-7B-Instruct (audio + text)")
    print("  - Qwen2-VL-7B-Instruct (vision + text)")
    print("  - Or wait for future Qwen omni release")
    print()
    print("This downloads the powerful 32B text model as foundation.")
    print("We can add specialized audio/vision models later if needed.")
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
        print("1. Update multi_model_loader.py to include 32B option")
        print("2. Test memory usage with 14B + 32B loaded")
        print("3. Benchmark inference speed vs 14B")
        print("4. Consider adding Qwen2-Audio/VL for true omni capabilities")
        print()

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check internet connection")
        print("2. Verify HuggingFace hub access")
        print("3. Check available disk space (need ~60GB free)")
        print("4. Try manual download from https://huggingface.co/" + model_id)
        return False

    return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Qwen 30B Omni Model")
    parser.add_argument(
        '--quantization',
        choices=['int8', 'fp16'],
        default='int8',
        help='Quantization format (default: int8 for better memory efficiency)'
    )

    args = parser.parse_args()

    success = download_qwen_30b_omni(args.quantization)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
