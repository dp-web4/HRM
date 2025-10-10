#!/usr/bin/env python3
"""
Build ARC-AGI dataset with GR00T features for SAGE distillation.

This script:
1. Loads ARC tasks from JSON files
2. Renders grids as 900x900 RGB images with ARC color palette
3. Extracts features using GR00T backbone
4. Saves features and metadata for SAGE training
"""

import sys
sys.path.insert(0, "/home/dp/ai-workspace/isaac-gr00t")

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
from PIL import Image

from gr00t.model.policy import Gr00tPolicy
from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.transforms import GR00TTransform


# ARC-AGI color palette (standard 10 colors)
ARC_COLORS = [
    (0, 0, 0),        # 0: Black
    (0, 116, 217),    # 1: Blue
    (255, 65, 54),    # 2: Red
    (46, 204, 64),    # 3: Green
    (255, 220, 0),    # 4: Yellow
    (170, 170, 170),  # 5: Grey
    (240, 18, 190),   # 6: Magenta
    (255, 133, 27),   # 7: Orange
    (127, 219, 255),  # 8: Light Blue
    (135, 12, 37),    # 9: Dark Red
]


def render_arc_grid(grid: np.ndarray, cell_size: int = 30) -> np.ndarray:
    """
    Render ARC grid as RGB image.

    Args:
        grid: [H, W] array with values 0-9
        cell_size: Pixels per cell

    Returns:
        RGB image [H*cell_size, W*cell_size, 3]
    """
    h, w = grid.shape
    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            color = ARC_COLORS[grid[i, j]]
            img[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = color

    return img


def pad_grid_to_30x30(grid: np.ndarray) -> np.ndarray:
    """Pad grid to 30x30 (ARC max size) with black (0)."""
    h, w = grid.shape
    padded = np.zeros((30, 30), dtype=np.uint8)
    padded[:h, :w] = grid
    return padded


def load_arc_task(task_path: Path) -> Dict:
    """Load single ARC task from JSON."""
    with open(task_path, 'r') as f:
        return json.load(f)


def extract_groot_features(
    policy: Gr00tPolicy,
    input_img: np.ndarray,
    output_img: np.ndarray,
    task_id: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract GR00T features from input/output pair.

    Args:
        policy: Loaded Gr00tPolicy
        input_img: Input grid rendered as [H, W, 3]
        output_img: Output grid rendered as [H, W, 3]
        task_id: Task identifier for annotation

    Returns:
        (features, attention_mask): Features [1, seq_len, 2048], mask [1, seq_len]
    """
    # Prepare observations in GR00T format
    # Video: [B=1, T=2, V=1, H=900, W=900, C=3]
    # - T=2: input and output as temporal sequence
    # - V=1: single view (the grid)
    video_frames = np.stack([input_img, output_img], axis=0)  # [T=2, H, W, C]
    video_with_view = video_frames[:, np.newaxis, :, :, :]  # [T=2, V=1, H, W, C]
    video_batched = video_with_view[np.newaxis, :, :, :, :, :]  # [B=1, T=2, V=1, H, W, C]

    # State: Simple task encoding [B=1, T=1, 16]
    # For now, just a zero vector (could encode task properties later)
    state_batched = np.zeros((1, 1, 16), dtype=np.float32)

    observations = {
        "video": video_batched,
        "state": state_batched,
        "annotation.0": np.array([f"Solve ARC task {task_id}"]),
    }

    # Apply transforms
    normalized_obs = policy.apply_transforms(observations)

    # Extract backbone features
    with torch.no_grad():
        backbone_inputs, _ = policy.model.prepare_input(normalized_obs)
        backbone_outputs = policy.model.backbone(backbone_inputs)

        features = backbone_outputs["backbone_features"]  # [B, seq_len, 2048]
        attention_mask = backbone_outputs.get("backbone_attention_mask")  # [B, seq_len]

    return features, attention_mask


def process_arc_dataset(
    arc_data_dir: Path,
    output_dir: Path,
    policy: Gr00tPolicy,
    max_tasks: int = None,
    split: str = "training",
):
    """
    Process entire ARC dataset and extract GR00T features.

    Args:
        arc_data_dir: Path to ARC-AGI/data directory
        output_dir: Path to save features
        policy: Loaded Gr00tPolicy
        max_tasks: Maximum number of tasks to process (None = all)
        split: "training" or "evaluation"
    """
    print(f"\n{'='*80}")
    print(f"Processing ARC {split} dataset")
    print(f"{'='*80}")

    # Get all task files
    task_dir = arc_data_dir / split
    task_files = sorted(list(task_dir.glob("*.json")))

    if max_tasks:
        task_files = task_files[:max_tasks]

    print(f"Found {len(task_files)} tasks")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)

    # Process each task
    dataset_metadata = []

    for task_file in tqdm(task_files, desc="Extracting features"):
        task_id = task_file.stem
        task_data = load_arc_task(task_file)

        # Process training examples
        for example_idx, example in enumerate(task_data["train"]):
            input_grid = np.array(example["input"], dtype=np.uint8)
            output_grid = np.array(example["output"], dtype=np.uint8)

            # Pad to 30x30
            input_padded = pad_grid_to_30x30(input_grid)
            output_padded = pad_grid_to_30x30(output_grid)

            # Render as 900x900 RGB images
            input_img = render_arc_grid(input_padded, cell_size=30)
            output_img = render_arc_grid(output_padded, cell_size=30)

            # Extract GR00T features
            try:
                features, attention_mask = extract_groot_features(
                    policy, input_img, output_img, task_id
                )

                # Save features
                feature_id = f"{task_id}_train_{example_idx}"
                feature_path = features_dir / f"{feature_id}.pt"
                torch.save({
                    "features": features.cpu(),
                    "attention_mask": attention_mask.cpu() if attention_mask is not None else None,
                    "input_grid": input_grid,
                    "output_grid": output_grid,
                    "task_id": task_id,
                    "split": "train",
                    "example_idx": example_idx,
                }, feature_path)

                # Add to metadata
                dataset_metadata.append({
                    "feature_id": feature_id,
                    "task_id": task_id,
                    "split": "train",
                    "example_idx": example_idx,
                    "input_shape": input_grid.shape,
                    "output_shape": output_grid.shape,
                    "feature_shape": list(features.shape),
                })

            except Exception as e:
                print(f"\n⚠️  Error processing {task_id} train example {example_idx}: {e}")
                continue

        # Process test examples
        for example_idx, example in enumerate(task_data["test"]):
            input_grid = np.array(example["input"], dtype=np.uint8)
            output_grid = np.array(example["output"], dtype=np.uint8)

            input_padded = pad_grid_to_30x30(input_grid)
            output_padded = pad_grid_to_30x30(output_grid)

            input_img = render_arc_grid(input_padded, cell_size=30)
            output_img = render_arc_grid(output_padded, cell_size=30)

            try:
                features, attention_mask = extract_groot_features(
                    policy, input_img, output_img, task_id
                )

                feature_id = f"{task_id}_test_{example_idx}"
                feature_path = features_dir / f"{feature_id}.pt"
                torch.save({
                    "features": features.cpu(),
                    "attention_mask": attention_mask.cpu() if attention_mask is not None else None,
                    "input_grid": input_grid,
                    "output_grid": output_grid,
                    "task_id": task_id,
                    "split": "test",
                    "example_idx": example_idx,
                }, feature_path)

                dataset_metadata.append({
                    "feature_id": feature_id,
                    "task_id": task_id,
                    "split": "test",
                    "example_idx": example_idx,
                    "input_shape": input_grid.shape,
                    "output_shape": output_grid.shape,
                    "feature_shape": list(features.shape),
                })

            except Exception as e:
                print(f"\n⚠️  Error processing {task_id} test example {example_idx}: {e}")
                continue

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "num_tasks": len(task_files),
            "num_examples": len(dataset_metadata),
            "split": split,
            "examples": dataset_metadata,
        }, f, indent=2)

    print(f"\n✅ Processing complete!")
    print(f"   Processed {len(task_files)} tasks")
    print(f"   Extracted {len(dataset_metadata)} examples")
    print(f"   Features saved to: {features_dir}")
    print(f"   Metadata saved to: {metadata_path}")


def main():
    print("="*80)
    print("ARC-AGI GR00T Feature Extraction")
    print("="*80)

    # Paths
    arc_data_dir = Path("/home/dp/ai-workspace/HRM/dataset/raw-data/ARC-AGI/data")
    output_dir = Path("/home/dp/ai-workspace/HRM/sage/data/arc_groot_features")

    # Load GR00T policy
    print("\n📦 Loading GR00tPolicy with SAGE/ARC metadata...")

    modality_config = {
        "video": ModalityConfig(
            delta_indices=[-1, 0],
            modality_keys=["video.input_grid"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state.task_encoding"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["action.grid_output"],
        ),
    }

    transform = GR00TTransform(
        max_state_dim=32,
        max_action_dim=32,
        state_horizon=1,
        action_horizon=16,
    )

    policy = Gr00tPolicy(
        model_path="nvidia/GR00T-N1.5-3B",
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_config=modality_config,
        modality_transform=transform,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"✅ Policy loaded on {policy.device}")
    print(f"   Model: {sum(p.numel() for p in policy.model.parameters()) / 1e9:.2f}B params")

    # Process full training dataset
    print("\n🚀 Processing all 400 training tasks...")
    print("   Estimated time: ~12 minutes (1.7s per task)")
    print("   Estimated storage: ~50GB features + grids")
    print()

    process_arc_dataset(
        arc_data_dir=arc_data_dir,
        output_dir=output_dir / "training_full",
        policy=policy,
        max_tasks=None,  # Process all tasks
        split="training",
    )

    print("\n" + "="*80)
    print("Full dataset extraction complete!")
    print("Ready for SAGE distillation training!")
    print("="*80)


if __name__ == "__main__":
    main()
