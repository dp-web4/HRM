#!/usr/bin/env python3
"""Debug expert weight shapes and orientations"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compression'))

from selective_expert_loader import SelectiveExpertLoader

def main():
    print("="*80)
    print("🔍 DEBUG: Expert Weight Shapes")
    print("="*80)

    # Initialize loader
    loader = SelectiveExpertLoader(
        extraction_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted",
        component="thinker",
        device="cpu",
        max_loaded_experts=16
    )

    # Load one expert
    print("\n📦 Loading expert 50 from layer 0...")
    expert_weights = loader.load_expert(50, 0)

    print("\n📊 Weight shapes:")
    for key, weight in expert_weights.items():
        print(f"   {key}: {weight.shape}")
        print(f"      mean={weight.mean():.6f}, std={weight.std():.6f}")

    # Expected shapes for thinker (hidden_size=2048, intermediate=768)
    print("\n✅ Expected shapes (for thinker):")
    print("   gate_proj: [768, 2048]  (out=768, in=2048)")
    print("   up_proj:   [768, 2048]  (out=768, in=2048)")
    print("   down_proj: [2048, 768]  (out=2048, in=768)")

    # Check if shapes match
    gate_shape = None
    up_shape = None
    down_shape = None

    for key, weight in expert_weights.items():
        if 'gate_proj' in key:
            gate_shape = weight.shape
        elif 'up_proj' in key:
            up_shape = weight.shape
        elif 'down_proj' in key:
            down_shape = weight.shape

    print("\n🔍 Validation:")
    print(f"   gate_proj: {gate_shape} {'✅' if gate_shape == (768, 2048) else '❌'}")
    print(f"   up_proj:   {up_shape} {'✅' if up_shape == (768, 2048) else '❌'}")
    print(f"   down_proj: {down_shape} {'✅' if down_shape == (2048, 768) else '❌'}")

if __name__ == "__main__":
    main()
