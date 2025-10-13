#!/usr/bin/env python3
"""
SAGE Student Model Training with Knowledge Distillation from GR00T.

Training configuration:
- Batch size: 8 (fits in 2GB VRAM)
- Epochs: 100
- Optimizer: AdamW with cosine annealing
- Mixed precision: bfloat16
- Validation: 10% holdout
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, List, Tuple

from sage_student_model import SAGEStudent, DistillationLoss, create_sage_student


class ARCGrootDataset(Dataset):
    """Dataset for ARC tasks with pre-extracted GR00T features."""

    def __init__(self, data_dir: Path, metadata_path: Path, split: str = "train"):
        self.data_dir = data_dir
        self.split = split

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.examples = self.metadata["examples"]
        print(f"Loaded {len(self.examples)} examples from {split} split")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example_meta = self.examples[idx]
        feature_id = example_meta["feature_id"]

        # Load feature file
        feature_path = self.data_dir / f"{feature_id}.pt"
        data = torch.load(feature_path, weights_only=False)

        # Extract components
        groot_features = data["features"]  # [1, seq_len, 2048]
        attention_mask = data["attention_mask"]  # [1, seq_len]
        output_grid = data["output_grid"]  # [H, W]

        # Pad output grid to 30x30
        h, w = output_grid.shape
        target_grid = np.zeros((30, 30), dtype=np.int64)
        target_grid[:h, :w] = output_grid

        return {
            "groot_features": groot_features.squeeze(0),  # [seq_len, 2048]
            "attention_mask": attention_mask.squeeze(0) if attention_mask is not None else None,
            "target_grid": torch.from_numpy(target_grid),  # [30, 30]
            "task_id": data["task_id"],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function to handle variable sequence lengths."""
    # Find max sequence length in batch
    max_seq_len = max(item["groot_features"].shape[0] for item in batch)

    # Prepare batched tensors
    batch_size = len(batch)
    groot_features = torch.zeros(batch_size, max_seq_len, 2048)
    attention_masks = torch.zeros(batch_size, max_seq_len)
    target_grids = torch.stack([item["target_grid"] for item in batch])

    # Fill in data with padding
    for i, item in enumerate(batch):
        seq_len = item["groot_features"].shape[0]
        groot_features[i, :seq_len] = item["groot_features"]
        if item["attention_mask"] is not None:
            attention_masks[i, :seq_len] = item["attention_mask"]
        else:
            attention_masks[i, :seq_len] = 1  # No mask = all valid

    return {
        "groot_features": groot_features,
        "attention_mask": attention_masks,
        "target_grid": target_grids,
    }


def train_epoch(
    model: SAGEStudent,
    criterion: DistillationLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_task_loss = 0.0
    total_feature_loss = 0.0
    total_correct = 0
    total_pixels = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        groot_features = batch["groot_features"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_grid = batch["target_grid"].to(device)

        # Forward pass
        optimizer.zero_grad()
        grid_logits, student_features = model(groot_features, attention_mask)

        # Compute loss
        loss, loss_dict = criterion(
            grid_logits, target_grid, student_features, groot_features, attention_mask
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate metrics
        total_loss += loss_dict["total"]
        total_task_loss += loss_dict["task"]
        total_feature_loss += loss_dict["feature_distill"]

        # Calculate accuracy
        predictions = grid_logits.argmax(dim=-1)  # [batch, 30, 30]
        correct = (predictions == target_grid).sum().item()
        total_correct += correct
        total_pixels += target_grid.numel()

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_dict['total']:.4f}",
            "acc": f"{100 * total_correct / total_pixels:.2f}%",
        })

    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "task_loss": total_task_loss / num_batches,
        "feature_loss": total_feature_loss / num_batches,
        "accuracy": 100 * total_correct / total_pixels,
    }


@torch.no_grad()
def validate(
    model: SAGEStudent,
    criterion: DistillationLoss,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()

    total_loss = 0.0
    total_task_loss = 0.0
    total_correct = 0
    total_pixels = 0

    for batch in tqdm(dataloader, desc="Validating"):
        groot_features = batch["groot_features"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_grid = batch["target_grid"].to(device)

        # Forward pass
        grid_logits, student_features = model(groot_features, attention_mask)

        # Compute loss
        loss, loss_dict = criterion(
            grid_logits, target_grid, student_features, groot_features, attention_mask
        )

        total_loss += loss_dict["total"]
        total_task_loss += loss_dict["task"]

        # Calculate accuracy
        predictions = grid_logits.argmax(dim=-1)
        correct = (predictions == target_grid).sum().item()
        total_correct += correct
        total_pixels += target_grid.numel()

    num_batches = len(dataloader)
    return {
        "loss": total_loss / num_batches,
        "task_loss": total_task_loss / num_batches,
        "accuracy": 100 * total_correct / total_pixels,
    }


def train(
    model: SAGEStudent,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    num_epochs: int = 100,
    lr: float = 1e-4,
    checkpoint_dir: Path = None,
):
    """Main training loop."""
    print(f"\n{'='*80}")
    print("Starting SAGE Training")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Setup
    criterion = DistillationLoss(
        student_hidden_dim=model.hidden_dim,
        teacher_hidden_dim=2048,
        task_weight=1.0,
        feature_weight=0.5,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    best_epoch = 0

    # Training loop
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")

        # Train
        train_metrics = train_epoch(model, criterion, train_loader, optimizer, device, epoch)

        # Clear cache before validation
        torch.cuda.empty_cache()

        # Validate
        val_metrics = validate(model, criterion, val_loader, device)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log metrics
        print(f"\n📊 Epoch {epoch} Results:")
        print(f"   Train Loss: {train_metrics['loss']:.4f}")
        print(f"   Train Accuracy: {train_metrics['accuracy']:.2f}%")
        print(f"   Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Val Accuracy: {val_metrics['accuracy']:.2f}%")
        print(f"   Learning Rate: {current_lr:.2e}")

        # Save history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(current_lr)

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch

            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_acc,
                    "val_loss": val_metrics["loss"],
                }, checkpoint_path)
                print(f"   💾 Saved best model (val_acc: {best_val_acc:.2f}%)")

        # Periodic checkpoint
        if checkpoint_dir and epoch % 10 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, checkpoint_path)

    # Training complete
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Total time: {elapsed_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")

    # Save final history
    if checkpoint_dir:
        history_path = checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to: {history_path}")

    return history


def main():
    # Set PyTorch memory allocator to reduce fragmentation
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    print("="*80)
    print("SAGE Student Training - Knowledge Distillation from GR00T")
    print("="*80)
    print("Memory config: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    # Paths
    data_dir = Path("/home/dp/ai-workspace/HRM/sage/data/arc_groot_features/training_full")
    features_dir = data_dir / "features"
    metadata_path = data_dir / "metadata.json"
    checkpoint_dir = Path("/home/dp/ai-workspace/HRM/sage/checkpoints/sage_student")

    # Check data exists
    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        return

    # Load dataset
    print("\n📦 Loading dataset...")
    dataset = ARCGrootDataset(features_dir, metadata_path, split="train")

    # Train/val split (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Train examples: {len(train_dataset)}")
    print(f"   Val examples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,  # Reduced from 8 to avoid OOM
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Create model
    print("\n📦 Creating SAGE student model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_sage_student(hidden_dim=512, num_layers=6, num_heads=8)
    model = model.to(device)

    param_counts = model.get_num_params()
    print(f"✅ Model created")
    print(f"   Parameters: {param_counts['total']:,} ({param_counts['total']/1e6:.1f}M)")
    print(f"   Device: {device}")

    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=100,
        lr=1e-4,
        checkpoint_dir=checkpoint_dir,
    )

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
