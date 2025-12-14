#!/usr/bin/env python3
"""
Download Qwen3-Omni-30B INT8 AWQ Quantized Model

AWQ (Activation-aware Weight Quantization) is optimized for inference.
Expected size: ~35GB (vs 70GB for FP16)
"""

from huggingface_hub import snapshot_download
from pathlib import Path

def download_qwen3_omni_int8():
    # Using AWQ 8-bit quantization (better for inference than GPTQ)
    model_id = "cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-8bit"
    local_dir = Path("model-zoo/sage/omni-modal/qwen3-omni-30b-int8-awq")

    print("=" * 70)
    print(f"Downloading Qwen3-Omni-30B INT8 AWQ from {model_id}")
    print(f"Target directory: {local_dir}")
    print("=" * 70)
    print()
    print("Expected size: ~35GB (50% reduction from 70GB FP16)")
    print("AWQ quantization maintains high quality while reducing memory")
    print()

    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.bin"],  # Only download safetensors
        )

        print()
        print("=" * 70)
        print("✅ Download complete!")
        print("=" * 70)
        print(f"\nModel saved to: {local_dir.absolute()}")

        # Show downloaded files
        print("\nDownloaded files:")
        for f in sorted(local_dir.glob("*")):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name}: {size_mb:.1f} MB")

        return True

    except Exception as e:
        print(f"\n❌ Error during download: {e}")
        return False

if __name__ == "__main__":
    success = download_qwen3_omni_int8()
    exit(0 if success else 1)
