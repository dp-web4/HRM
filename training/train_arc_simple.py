#!/usr/bin/env python3
"""
Simple ARC Training Script for HRM
Minimal setup to get training started quickly
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
import time
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Simple configuration
BATCH_SIZE = 8  # Small batch for testing
LEARNING_RATE = 3e-4
MAX_EPOCHS = 10  # Quick test
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleARCDataset(Dataset):
    """Simple dataset loader for ARC puzzles"""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load data files
        split_dir = self.data_dir / split
        if split_dir.exists():
            self.inputs = np.load(split_dir / 'all__inputs.npy')
            self.labels = np.load(split_dir / 'all__labels.npy')
            print(f"Loaded {len(self.inputs)} {split} samples")
        else:
            # Create dummy data for testing
            print(f"Creating dummy {split} data")
            self.inputs = np.random.randint(0, 10, (100, 900))
            self.labels = np.random.randint(0, 10, (100, 900))
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'input': torch.from_numpy(self.inputs[idx]).long(),
            'target': torch.from_numpy(self.labels[idx]).long()
        }

class SimpleHRM(nn.Module):
    """Simplified HRM model for testing"""
    
    def __init__(self, vocab_size=11, hidden_size=256, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.output = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.output(x)
        return x

def train():
    """Main training loop"""
    print(f"🚀 Training on {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Create datasets
    train_dataset = SimpleARCDataset('data/arc-aug-100', 'train')
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Single worker for simplicity
    )
    
    # Create model
    model = SimpleHRM().to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {param_count/1e6:.1f}M")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n🏃 Starting training...")
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = F.cross_entropy(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}", 'avg_loss': f"{avg_loss:.4f}"})
            
            # Break early for testing
            if batch_idx >= 10:  # Just 10 steps per epoch for testing
                break
        
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    # Save model
    checkpoint_path = 'checkpoints/hrm_arc_simple.pt'
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': avg_loss
    }, checkpoint_path)
    print(f"\n✅ Model saved to {checkpoint_path}")
    print("🎉 Training complete!")

if __name__ == "__main__":
    train()