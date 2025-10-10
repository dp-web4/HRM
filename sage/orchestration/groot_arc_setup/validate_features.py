#!/usr/bin/env python3
"""
Validate extracted GR00T features.
"""

import torch
import json
from pathlib import Path


def main():
    print("="*80)
    print("Validating GR00T Feature Extraction")
    print("="*80)

    data_dir = Path("/home/dp/ai-workspace/HRM/sage/data/arc_groot_features/validation_10")
    features_dir = data_dir / "features"
    metadata_path = data_dir / "metadata.json"

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\n📊 Dataset Summary:")
    print(f"   Tasks processed: {metadata['num_tasks']}")
    print(f"   Examples extracted: {metadata['num_examples']}")
    print(f"   Split: {metadata['split']}")

    # Load first example
    example = metadata["examples"][0]
    feature_id = example["feature_id"]
    feature_path = features_dir / f"{feature_id}.pt"

    print(f"\n🔬 Inspecting example: {feature_id}")
    print(f"   Task: {example['task_id']}")
    print(f"   Input shape: {example['input_shape']}")
    print(f"   Output shape: {example['output_shape']}")

    # Load feature data
    data = torch.load(feature_path)

    features = data["features"]
    attention_mask = data["attention_mask"]
    input_grid = data["input_grid"]
    output_grid = data["output_grid"]

    print(f"\n✅ Feature data loaded successfully!")
    print(f"   Features shape: {features.shape}")
    print(f"   Features dtype: {features.dtype}")
    print(f"   Features range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"   Features mean: {features.mean():.4f}")
    print(f"   Features std: {features.std():.4f}")

    if attention_mask is not None:
        print(f"   Attention mask shape: {attention_mask.shape}")
        print(f"   Active tokens: {attention_mask.sum()}/{attention_mask.numel()}")

    print(f"\n📐 Grid data:")
    print(f"   Input grid shape: {input_grid.shape}")
    print(f"   Output grid shape: {output_grid.shape}")
    print(f"   Input grid values: {set(input_grid.flatten().tolist())}")
    print(f"   Output grid values: {set(output_grid.flatten().tolist())}")

    print(f"\n   Input grid:")
    for row in input_grid:
        print(f"   {row.tolist()}")

    print(f"\n   Output grid:")
    for row in output_grid:
        print(f"   {row.tolist()}")

    # Check all features
    print(f"\n🔍 Checking all {metadata['num_examples']} examples...")
    feature_shapes = []
    for ex in metadata["examples"]:
        fpath = features_dir / f"{ex['feature_id']}.pt"
        if fpath.exists():
            fdata = torch.load(fpath)
            feature_shapes.append(fdata["features"].shape)
        else:
            print(f"   ⚠️  Missing: {ex['feature_id']}")

    # Check consistency
    unique_shapes = set(feature_shapes)
    print(f"   Unique feature shapes: {len(unique_shapes)}")
    for shape in unique_shapes:
        count = feature_shapes.count(shape)
        print(f"      {shape}: {count} examples")

    print(f"\n✅ Validation complete!")
    print(f"   All {len(feature_shapes)} feature files loaded successfully")
    print(f"   Ready for SAGE distillation training!")


if __name__ == "__main__":
    main()
