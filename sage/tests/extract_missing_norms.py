#!/usr/bin/env python3
"""Extract missing norm weights for layers 1, 5, 9, etc."""

import os
import sys
import json
import safetensors
import safetensors.torch
import torch

# Layers missing norm files
MISSING_LAYERS = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]

def extract_norms(model_dir: str, extraction_dir: str, component: str = "thinker"):
    """Extract norm weights for missing layers"""

    # Load index
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_file) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Create output directory
    norms_dir = os.path.join(extraction_dir, "norms")
    os.makedirs(norms_dir, exist_ok=True)

    print(f"Extracting missing {component} norms...")

    for layer_id in MISSING_LAYERS:
        print(f"\n📦 Layer {layer_id}:")

        prefix = f"{component}.model.layers.{layer_id}"
        input_norm_key = f"{prefix}.input_layernorm.weight"
        post_norm_key = f"{prefix}.post_attention_layernorm.weight"

        # Find which shard files contain these weights
        input_norm_shard = weight_map.get(input_norm_key)
        post_norm_shard = weight_map.get(post_norm_key)

        if not input_norm_shard or not post_norm_shard:
            print(f"  ❌ Norm weights not found in index")
            continue

        # Load the weights
        norms = {}

        # Load input norm
        shard_path = os.path.join(model_dir, input_norm_shard)
        with safetensors.safe_open(shard_path, framework="pt") as f:
            if input_norm_key in f.keys():
                norms[input_norm_key] = f.get_tensor(input_norm_key)
                print(f"  ✅ Loaded input_layernorm from {input_norm_shard}")

        # Load post-attention norm (might be in different shard)
        shard_path = os.path.join(model_dir, post_norm_shard)
        with safetensors.safe_open(shard_path, framework="pt") as f:
            if post_norm_key in f.keys():
                norms[post_norm_key] = f.get_tensor(post_norm_key)
                print(f"  ✅ Loaded post_attention_layernorm from {post_norm_shard}")

        # Save to extraction directory
        if len(norms) == 2:
            output_file = os.path.join(norms_dir, f"{component}_norms_layer_{layer_id:02d}.safetensors")
            safetensors.torch.save_file(norms, output_file)
            print(f"  💾 Saved to {output_file}")
        else:
            print(f"  ⚠️  Only found {len(norms)}/2 norms, skipping")

if __name__ == "__main__":
    model_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b"
    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"

    extract_norms(model_dir, extraction_dir, component="thinker")

    print("\n" + "="*80)
    print("✅ Extraction complete!")
    print("="*80)
