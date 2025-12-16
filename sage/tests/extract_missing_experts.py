#!/usr/bin/env python3
"""Extract the 8 missing experts that were corrupted"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compression'))

from expert_extractor import ExpertExtractor

# The 8 experts that were corrupted (from previous scan)
missing_experts = [
    (77, 5),
    (78, 9),
    (81, 17),
    (82, 21),
    (85, 29),
    (86, 33),
    (89, 41),
    (90, 45),
]

def main():
    print("="*80)
    print("EXTRACTING 8 MISSING EXPERTS")
    print("="*80)

    extractor = ExpertExtractor(
        model_path="model-zoo/sage/omni-modal/qwen3-omni-30b",
        output_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"
    )

    success = 0
    for expert_id, layer_id in missing_experts:
        print(f"\nExtracting expert {expert_id} layer {layer_id}...")
        result = extractor.extract_expert(expert_id, layer_id, component="thinker", force=True)

        if result:
            print(f"  ✅ Success: {result.name}")
            success += 1
        else:
            print(f"  ❌ Failed")

    print(f"\n{'='*80}")
    print(f"✅ Extracted {success}/{len(missing_experts)} experts")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
