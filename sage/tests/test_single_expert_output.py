#!/usr/bin/env python3
"""Test single expert computation in isolation"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compression'))

import torch
import torch.nn.functional as F
from selective_expert_loader import SelectiveExpertLoader

def manual_expert_forward(x, gate_weight, up_weight, down_weight):
    """Manually compute expert forward to verify"""
    # x: [batch, seq, hidden] or [1, hidden]
    gate_out = F.linear(x, gate_weight)
    up_out = F.linear(x, up_weight)
    intermediate = F.silu(gate_out) * up_out
    output = F.linear(intermediate, down_weight)
    return output

def main():
    print("="*80)
    print("🔍 DEBUG: Single Expert Computation")
    print("="*80)

    # Load expert
    loader = SelectiveExpertLoader(
        extraction_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted",
        component="thinker",
        device="cpu",
        max_loaded_experts=16
    )

    expert_weights = loader.load_expert(50, 0)

    # Extract weights
    gate_proj = None
    up_proj = None
    down_proj = None

    for key, weight in expert_weights.items():
        if 'gate_proj' in key:
            gate_proj = weight
        elif 'up_proj' in key:
            up_proj = weight
        elif 'down_proj' in key:
            down_proj = weight

    print("\n✅ Loaded expert 50, layer 0")
    print(f"   gate_proj: {gate_proj.shape}")
    print(f"   up_proj: {up_proj.shape}")
    print(f"   down_proj: {down_proj.shape}")

    # Create random input
    torch.manual_seed(42)
    x = torch.randn(1, 2048)  # [1, hidden_size]

    print(f"\n📊 Input:")
    print(f"   Shape: {x.shape}")
    print(f"   Mean: {x.mean():.6f}, Std: {x.std():.6f}")

    # Compute expert output
    output = manual_expert_forward(x, gate_proj, up_proj, down_proj)

    print(f"\n📊 Expert output:")
    print(f"   Shape: {output.shape}")
    print(f"   Mean: {output.mean():.6f}, Std: {output.std():.6f}")
    print(f"   Min: {output.min():.6f}, Max: {output.max():.6f}")

    # Check for issues
    if torch.isnan(output).any():
        print("   ❌ Contains NaN!")
    elif torch.isinf(output).any():
        print("   ❌ Contains Inf!")
    elif output.abs().max() > 1e6:
        print("   ⚠️  Very large values (potential numerical instability)")
    elif output.abs().max() < 1e-6:
        print("   ⚠️  Very small values (potential vanishing)")
    else:
        print("   ✅ Looks numerically healthy")

    # Also test with a real embedding
    print(f"\n{'='*80}")
    print("Testing with real model embedding")
    print(f"{'='*80}")

    # Load real embeddings
    import safetensors
    embed_file = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted/embeddings/thinker_embeddings.safetensors"
    with safetensors.safe_open(embed_file, framework="pt") as f:
        embeddings = f.get_tensor("embed_tokens.weight")

    # Get embedding for token "Paris" (ID 59604)
    paris_embedding = embeddings[59604:59604+1, :].float()  # [1, 2048] - convert to float32
    print(f"\n📊 'Paris' token embedding:")
    print(f"   Shape: {paris_embedding.shape}")
    print(f"   Dtype: {paris_embedding.dtype}")
    print(f"   Mean: {paris_embedding.mean():.6f}, Std: {paris_embedding.std():.6f}")

    # Process through expert
    paris_output = manual_expert_forward(paris_embedding, gate_proj, up_proj, down_proj)
    print(f"\n📊 'Paris' after expert 50:")
    print(f"   Shape: {paris_output.shape}")
    print(f"   Mean: {paris_output.mean():.6f}, Std: {paris_output.std():.6f}")
    print(f"   Min: {paris_output.min():.6f}, Max: {paris_output.max():.6f}")

if __name__ == "__main__":
    main()
