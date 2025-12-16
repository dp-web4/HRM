#!/usr/bin/env python3
"""Check if final norm weights are reasonable"""

import safetensors
from pathlib import Path

final_norm_path = Path("model-zoo/sage/omni-modal/qwen3-omni-30b-extracted/final_norm/thinker_final_norm.safetensors")

print(f"Checking: {final_norm_path}")
print(f"Exists: {final_norm_path.exists()}")

if final_norm_path.exists():
    with safetensors.safe_open(final_norm_path, framework="pt") as f:
        weight = f.get_tensor('thinker.model.norm.weight')
        print(f"\nFinal norm weight:")
        print(f"  Shape: {weight.shape}")
        print(f"  Mean: {weight.mean():.6f}")
        print(f"  Std: {weight.std():.6f}")
        print(f"  Min: {weight.min():.6f}")
        print(f"  Max: {weight.max():.6f}")
        print(f"\n  First 10 values: {weight[:10].tolist()}")

        # Check if it's all ones (default initialization)
        import torch
        if torch.allclose(weight, torch.ones_like(weight)):
            print("\n⚠️  WARNING: Final norm is all ones (default initialization)")
        else:
            print("\n✅ Final norm has been trained (not default initialization)")
