#!/usr/bin/env python3
"""
ARC Training Script for HRM on Legion RTX 4090
Trains the Hierarchical Reasoning Model on Abstract Reasoning Corpus
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import Dict, Any, Tuple
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config
)

# Configuration
MODEL_CONFIG = {
    'batch_size': 32,
    'seq_len': 900,  # 30x30 grid max
    'puzzle_emb_ndim': 128,
    'vocab_size': 11,  # 0-9 colors + padding
    'H_cycles': 4,
    'L_cycles': 8,
    'H_layers': 6,
    'L_layers': 4,
    'hidden_size': 512,
    'expansion': 4.0,
    'num_heads': 16,
    'pos_encodings': 'rope',
    'halt_max_steps': 12,
    'halt_exploration_prob': 0.1,
    'forward_dtype': 'bfloat16',
}

TRAINING_CONFIG = {
    'learning_rate': 3e-4,
    'warmup_steps': 500,
    'max_epochs': 100,
    'gradient_accumulation_steps': 2,
    'mixed_precision': True,
    'gradient_clip': 1.0,
    'weight_decay': 0.01,
    'dropout': 0.1,
    'eval_frequency': 100,
    'checkpoint_frequency': 500,
}

class ARCDataset(Dataset):
    """Simple ARC dataset loader"""
    def __init__(self, data_dir: str, split: str):
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples = []
        
        # Load all puzzle files
        split_dir = self.data_dir / split
        if split_dir.exists():
            for puzzle_file in split_dir.glob("*.json"):
                with open(puzzle_file, 'r') as f:
                    puzzle = json.load(f)
                    for example in puzzle.get('train', []) + puzzle.get('test', []):
                        if 'input' in example and 'output' in example:
                            self.samples.append({
                                'input': self._grid_to_tensor(example['input']),
                                'output': self._grid_to_tensor(example['output']),
                                'puzzle_id': puzzle_file.stem
                            })
        
        print(f"Loaded {len(self.samples)} {split} samples from {split_dir}")
    
    def _grid_to_tensor(self, grid):
        """Convert grid to tensor, padding to max size"""
        tensor = torch.tensor(grid, dtype=torch.long)
        # Pad to 30x30
        if tensor.shape[0] < 30 or tensor.shape[1] < 30:
            pad_h = 30 - tensor.shape[0]
            pad_w = 30 - tensor.shape[1]
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=10)  # 10 = padding token
        return tensor.flatten()[:900]  # Ensure exactly 900 elements
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input': sample['input'],
            'target': sample['output'],
            'puzzle_id': hash(sample['puzzle_id']) % 1000  # Simple hash for puzzle ID
        }

def train_arc():
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create datasets
    print("\n📊 Loading ARC dataset...")
    train_dataset = ARCDataset("data/arc-aug-1000", "train")
    val_dataset = ARCDataset("data/arc-aug-1000", "val")
    
    # Skip if no data found
    if len(train_dataset) == 0:
        print("⚠️  No training data found. Please run dataset builder first:")
        print("   python dataset/build_arc_dataset.py --output-dir data/arc-aug-1000 --num-aug 1000")
        return
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False,
    ) if len(val_dataset) > 0 else None
    
    # Create model
    print("\n🧠 Creating HRM model...")
    model_config = HierarchicalReasoningModel_ACTV1Config(
        **MODEL_CONFIG,
        num_puzzle_identifiers=1000  # Max puzzle IDs
    )
    
    model = HierarchicalReasoningModel_ACTV1(model_config).to(device)
    
    # Print model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {param_count/1e6:.1f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * TRAINING_CONFIG['max_epochs']
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if TRAINING_CONFIG['mixed_precision'] and device.type == 'cuda' else None
    
    # Training loop
    print("\n🎯 Starting training...")
    best_val_loss = float('inf')
    global_step = 0
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(TRAINING_CONFIG['max_epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['max_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if scaler and device.type == 'cuda':
                with autocast():
                    outputs, carry = model(batch)
                    
                    # Simple cross-entropy loss
                    loss = F.cross_entropy(
                        outputs.reshape(-1, MODEL_CONFIG['vocab_size']),
                        batch['target'].reshape(-1)
                    )
                    
                    # Add ACT penalty if available
                    if hasattr(carry, 'steps'):
                        act_penalty = 0.01 * carry.steps.mean()
                        loss = loss + act_penalty
            else:
                outputs, carry = model(batch)
                loss = F.cross_entropy(
                    outputs.reshape(-1, MODEL_CONFIG['vocab_size']),
                    batch['target'].reshape(-1)
                )
                if hasattr(carry, 'steps'):
                    act_penalty = 0.01 * carry.steps.mean()
                    loss = loss + act_penalty
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % TRAINING_CONFIG['gradient_accumulation_steps'] == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        TRAINING_CONFIG['gradient_clip']
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        TRAINING_CONFIG['gradient_clip']
                    )
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            predictions = outputs.argmax(dim=-1)
            train_correct += (predictions == batch['target']).sum().item()
            train_total += batch['target'].numel()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/max(train_total, 1):.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })
            
            # Validation
            if val_loader and global_step % TRAINING_CONFIG['eval_frequency'] == 0:
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for val_batch in tqdm(val_loader, desc="Validation", leave=False):
                        val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in val_batch.items()}
                        
                        if scaler and device.type == 'cuda':
                            with autocast():
                                outputs, _ = model(val_batch)
                                loss = F.cross_entropy(
                                    outputs.reshape(-1, MODEL_CONFIG['vocab_size']),
                                    val_batch['target'].reshape(-1)
                                )
                        else:
                            outputs, _ = model(val_batch)
                            loss = F.cross_entropy(
                                outputs.reshape(-1, MODEL_CONFIG['vocab_size']),
                                val_batch['target'].reshape(-1)
                            )
                        
                        val_loss += loss.item()
                        predictions = outputs.argmax(dim=-1)
                        val_correct += (predictions == val_batch['target']).sum().item()
                        val_total += val_batch['target'].numel()
                
                val_accuracy = val_correct / max(val_total, 1)
                avg_val_loss = val_loss / len(val_loader)
                
                print(f"\n📈 Validation - Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f}")
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'config': model_config.dict()
                    }, 'checkpoints/hrm_arc_best.pt')
                    print(f"✅ New best model saved!")
                
                model.train()
            
            # Checkpoint
            if global_step % TRAINING_CONFIG['checkpoint_frequency'] == 0:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': model_config.dict()
                }, f'checkpoints/hrm_arc_step_{global_step}.pt')
                print(f"💾 Checkpoint saved at step {global_step}")
            
            global_step += 1
        
        # Epoch summary
        epoch_acc = train_correct / max(train_total, 1)
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"   Train Accuracy: {epoch_acc:.4f}")
        if val_loader:
            print(f"   Best Val Loss: {best_val_loss:.4f}")
        print("-" * 50)
    
    print(f"\n🎉 Training complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Model saved to: checkpoints/hrm_arc_best.pt")

if __name__ == "__main__":
    train_arc()