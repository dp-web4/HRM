#!/usr/bin/env python3
"""
Simple Sudoku training script for HRM testing
"""

import os
import sys
import torch
from pathlib import Path

# Add HRM to path
sys.path.append(str(Path(__file__).parent))

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

def train_sudoku():
    """Train HRM on Sudoku puzzles"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Dataset config
    dataset_config = PuzzleDatasetConfig(
        root="data/sudoku-test",
        vocab_size=1000,
        puzzle_emb_ndim=128,
        seq_len=128,
    )
    
    # Load dataset
    print("\n📊 Loading Sudoku dataset...")
    train_dataset = PuzzleDataset(dataset_config, split="train")
    val_dataset = PuzzleDataset(dataset_config, split="val")
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # Small batch for testing
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Model config
    model_config = {
        'batch_size': 8,
        'seq_len': 128,
        'puzzle_emb_ndim': 128,
        'num_puzzle_identifiers': train_dataset.metadata.num_puzzles,
        'vocab_size': 1000,
        'H_cycles': 2,
        'L_cycles': 4,
        'H_layers': 3,
        'L_layers': 2,
        'hidden_size': 256,
        'expansion': 4.0,
        'num_heads': 8,
        'pos_encodings': 'rope',
        'halt_max_steps': 5,
        'halt_exploration_prob': 0.1,
        'forward_dtype': 'bfloat16' if device.type == 'cuda' else 'float32',
    }
    
    # Create model
    print("\n🧠 Creating HRM model...")
    model = HierarchicalReasoningModel_ACTV1(model_config).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count/1e6:.1f}M")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    print("\n🚀 Starting training...")
    model.train()
    
    for epoch in range(3):  # Just 3 epochs for testing
        print(f"\n📈 Epoch {epoch+1}/3")
        
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            carry = model.initial_carry(batch)
            carry, outputs = model(carry, batch)
            
            # Calculate loss
            logits = outputs['logits']
            targets = batch['targets']
            
            # Reshape for cross entropy
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                
                carry = model.initial_carry(batch)
                carry, outputs = model(carry, batch)
                
                logits = outputs['logits']
                targets = batch['targets']
                
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                
                loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
                val_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        
        model.train()
    
    print("\n✅ Training complete!")
    
    # Save model
    save_path = "sudoku_hrm_checkpoint.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model_config,
    }, save_path)
    print(f"💾 Model saved to {save_path}")


if __name__ == "__main__":
    train_sudoku()