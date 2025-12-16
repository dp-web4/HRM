#!/usr/bin/env python3
"""Check if Q/K normalization weights are loaded correctly"""

import safetensors
from pathlib import Path

# Check layer 0 attention file
attn_file = Path("model-zoo/sage/omni-modal/qwen3-omni-30b-extracted/attention/thinker_attention_layer_00.safetensors")

print(f"Checking: {attn_file}")
print(f"Exists: {attn_file.exists()}\n")

if attn_file.exists():
    with safetensors.safe_open(attn_file, framework="pt") as f:
        print("Keys in attention file:")
        for key in sorted(f.keys()):
            tensor = f.get_tensor(key)
            print(f"  {key}: {list(tensor.shape)}")

        print("\nChecking Q/K norms:")
        prefix = "thinker.model.layers.0.self_attn"

        # Check q_norm
        q_norm_key = f"{prefix}.q_norm.weight"
        if q_norm_key in f.keys():
            q_norm = f.get_tensor(q_norm_key)
            print(f"\n✅ Q norm found!")
            print(f"   Shape: {q_norm.shape}")
            print(f"   Mean: {q_norm.mean():.6f}")
            print(f"   Std: {q_norm.std():.6f}")
            print(f"   Values: {q_norm[:10].tolist()}")
        else:
            print(f"\n❌ Q norm NOT found! Key '{q_norm_key}' missing")

        # Check k_norm
        k_norm_key = f"{prefix}.k_norm.weight"
        if k_norm_key in f.keys():
            k_norm = f.get_tensor(k_norm_key)
            print(f"\n✅ K norm found!")
            print(f"   Shape: {k_norm.shape}")
            print(f"   Mean: {k_norm.mean():.6f}")
            print(f"   Std: {k_norm.std():.6f}")
            print(f"   Values: {k_norm[:10].tolist()}")
        else:
            print(f"\n❌ K norm NOT found! Key '{k_norm_key}' missing")
