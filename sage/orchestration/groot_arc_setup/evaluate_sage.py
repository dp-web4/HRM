#!/usr/bin/env python3
"""
Evaluate trained SAGE model on validation set.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import time

from sage_student_model import create_sage_student
from train_sage import ARCGrootDataset, collate_fn


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Dict:
    """Evaluate model and return detailed metrics."""
    model.eval()

    total_correct = 0
    total_pixels = 0
    task_accuracies = []
    perfect_grids = 0
    total_grids = 0

    all_predictions = []
    all_targets = []

    print("\nEvaluating model...")
    for batch in tqdm(dataloader, desc="Evaluation"):
        groot_features = batch["groot_features"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_grid = batch["target_grid"].to(device)

        # Forward pass
        grid_logits, _ = model(groot_features, attention_mask)
        predictions = grid_logits.argmax(dim=-1)  # [batch, 30, 30]

        # Pixel-wise accuracy
        correct = (predictions == target_grid).sum().item()
        total_correct += correct
        total_pixels += target_grid.numel()

        # Per-grid accuracy (all pixels must be correct)
        batch_size = target_grid.shape[0]
        for i in range(batch_size):
            grid_correct = (predictions[i] == target_grid[i]).all().item()
            perfect_grids += int(grid_correct)
            total_grids += 1

            # Store for confusion matrix
            all_predictions.append(predictions[i].cpu().numpy())
            all_targets.append(target_grid[i].cpu().numpy())

    pixel_accuracy = 100 * total_correct / total_pixels
    grid_accuracy = 100 * perfect_grids / total_grids

    # Calculate per-color accuracy
    all_preds = np.concatenate([p.flatten() for p in all_predictions])
    all_targs = np.concatenate([t.flatten() for t in all_targets])

    color_stats = {}
    for color in range(10):
        mask = all_targs == color
        if mask.sum() > 0:
            color_acc = 100 * (all_preds[mask] == color).sum() / mask.sum()
            color_stats[f"color_{color}"] = {
                "accuracy": color_acc,
                "count": int(mask.sum())
            }

    return {
        "pixel_accuracy": pixel_accuracy,
        "grid_accuracy": grid_accuracy,
        "perfect_grids": perfect_grids,
        "total_grids": total_grids,
        "color_stats": color_stats,
    }


def main():
    print("="*80)
    print("SAGE Model Evaluation")
    print("="*80)

    # Paths
    data_dir = Path("/home/dp/ai-workspace/HRM/sage/data/arc_groot_features/training_full")
    features_dir = data_dir / "features"
    metadata_path = data_dir / "metadata.json"
    checkpoint_path = Path("/home/dp/ai-workspace/HRM/sage/checkpoints/sage_student/best_model.pt")

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    # Load dataset
    print("\n📦 Loading dataset...")
    dataset = ARCGrootDataset(features_dir, metadata_path, split="train")

    # Use same split as training (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Validation examples: {len(val_dataset)}")

    # Create dataloader (batch_size=1 to avoid OOM)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Load model
    print("\n📦 Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_sage_student(hidden_dim=512, num_layers=6, num_heads=8)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Checkpoint val accuracy: {checkpoint.get('val_accuracy', 'N/A'):.2f}%")
    print(f"   Device: {device}")

    # Evaluate
    start_time = time.time()
    metrics = evaluate_model(model, val_loader, device)
    eval_time = time.time() - start_time

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\n📊 Overall Metrics:")
    print(f"   Pixel Accuracy: {metrics['pixel_accuracy']:.2f}%")
    print(f"   Grid Accuracy: {metrics['grid_accuracy']:.2f}%")
    print(f"   Perfect Grids: {metrics['perfect_grids']}/{metrics['total_grids']}")
    print(f"   Evaluation Time: {eval_time:.2f}s")
    print(f"   Time per Grid: {eval_time/metrics['total_grids']*1000:.1f}ms")

    print(f"\n🎨 Per-Color Accuracy:")
    for color, stats in metrics['color_stats'].items():
        color_num = color.split('_')[1]
        print(f"   Color {color_num}: {stats['accuracy']:6.2f}% ({stats['count']:,} pixels)")

    # Save results
    results_path = checkpoint_path.parent / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")

    # Compare with GR00T
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"   GR00T (Teacher):  2.7B params, ~150ms inference")
    print(f"   SAGE (Student):   44.1M params, ~{eval_time/metrics['total_grids']*1000:.0f}ms inference")
    print(f"   Compression:      61x smaller")
    print(f"   Speedup:          ~{150/(eval_time/metrics['total_grids']*1000):.1f}x faster")
    print(f"   Accuracy:         {metrics['pixel_accuracy']:.2f}% (pixel-wise)")
    print(f"                     {metrics['grid_accuracy']:.2f}% (exact match)")


if __name__ == "__main__":
    main()
