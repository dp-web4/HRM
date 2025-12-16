#!/usr/bin/env python3
"""
Fix corrupted expert files that are missing up_proj or down_proj weights
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compression'))

import safetensors
from pathlib import Path
from expert_extractor import ExpertExtractor

def find_corrupted_experts():
    """Find all expert files with incomplete weights"""
    experts_dir = Path("model-zoo/sage/omni-modal/qwen3-omni-30b-extracted/experts")
    corrupted = []

    print("Scanning for corrupted expert files...")
    for expert_file in sorted(experts_dir.glob("*.safetensors")):
        with safetensors.safe_open(expert_file, framework="pt") as f:
            num_weights = len(list(f.keys()))

            if num_weights < 3:  # Should have gate, up, down
                # Parse filename: thinker_expert_XXX_layer_YY.safetensors
                parts = expert_file.stem.split('_')
                expert_id = int(parts[2])
                layer_id = int(parts[4])

                corrupted.append((expert_file, expert_id, layer_id, num_weights))
                print(f"  ❌ {expert_file.name}: only {num_weights}/3 weights")

    return corrupted

def main():
    print("="*80)
    print("FIXING CORRUPTED EXPERT FILES")
    print("="*80)

    # Find corrupted files
    corrupted = find_corrupted_experts()

    if not corrupted:
        print("\n✅ No corrupted expert files found!")
        return

    print(f"\nFound {len(corrupted)} corrupted expert files")
    print("Re-extracting with fixed extractor (searches all shards)...\n")

    # Initialize extractor
    extractor = ExpertExtractor(
        model_path="model-zoo/sage/omni-modal/qwen3-omni-30b",
        output_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"
    )

    # Re-extract corrupted experts
    fixed = 0
    for expert_file, expert_id, layer_id, old_count in corrupted:
        print(f"Re-extracting expert {expert_id} layer {layer_id}...")

        # Delete old corrupted file
        expert_file.unlink()

        # Re-extract with fixed code
        result = extractor.extract_expert(expert_id, layer_id, component="thinker", force=True)

        if result and result.exists():
            # Verify it has all 3 weights now
            with safetensors.safe_open(result, framework="pt") as f:
                new_count = len(list(f.keys()))
                if new_count == 3:
                    print(f"  ✅ Fixed! Now has {new_count}/3 weights")
                    fixed += 1
                else:
                    print(f"  ⚠️  Still incomplete: {new_count}/3 weights")
        else:
            print(f"  ❌ Re-extraction failed")

    print(f"\n{'='*80}")
    print(f"✅ Fixed {fixed}/{len(corrupted)} corrupted experts")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
