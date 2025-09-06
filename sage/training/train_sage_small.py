"""
Small-scale SAGE training for memory-constrained environments

This script trains a smaller version of SAGE that fits in limited GPU memory.
"""

import torch
import sys
sys.path.append('..')

from core.sage_config import SAGEPresets
from train_sage import SAGETrainer, SyntheticAttentionDataset
from torch.utils.data import DataLoader


def main():
    """Run small-scale training"""
    print("Initializing small-scale SAGE training...")
    
    # Use development config (25M params instead of 100M)
    config = SAGEPresets.development()
    config.batch_size = 4  # Very small batch size
    config.gradient_accumulation_steps = 8  # Effective batch = 32
    config.gradient_checkpointing = True  # Save memory
    config.num_workers = 0  # Avoid multiprocessing overhead
    config.checkpoint_interval = 100  # More frequent checkpoints for testing
    config.log_interval = 10
    
    print(f"Configuration:")
    print(f"  Model size: ~25M parameters")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    
    # Create small datasets
    print("Creating datasets...")
    train_dataset = SyntheticAttentionDataset(
        num_samples=100,  # Very small for testing
        seq_len=50,  # Shorter sequences
        num_classes=10,
        context_dim=config.context_dim
    )
    
    val_dataset = SyntheticAttentionDataset(
        num_samples=20,
        seq_len=50,
        num_classes=10,
        context_dim=config.context_dim
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False  # Save memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = SAGETrainer(config)
    
    # Check memory before training
    if torch.cuda.is_available():
        print(f"GPU memory before training:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Run short training
    print("\nStarting training...")
    try:
        trainer.train(train_loader, val_loader, num_epochs=1)
        print("\nTraining completed successfully!")
    except torch.cuda.OutOfMemoryError as e:
        print(f"\nGPU out of memory: {e}")
        print("Try reducing batch_size or seq_len further")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Check final memory usage
    if torch.cuda.is_available():
        print(f"\nGPU memory after training:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    main()