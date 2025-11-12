#!/usr/bin/env python3
"""
Train SAGE V3 using human-like predictions.
Simplified version using the FaithfulModel architecture.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import math
import random
from typing import List, Dict

# Model configuration
MODEL_CONFIG = {
    'seq_len': 900,  # 30x30 grid
    'vocab_size': 12,  # 0-9 colors + padding
    'hidden_size': 256,
    'num_heads': 8,
    'num_h_layers': 4,  # Strategic layers
    'num_l_layers': 3,  # Tactical layers
    'dropout': 0.1,  # Some dropout for training
    'max_cycles': 8
}

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class FaithfulModel(nn.Module):
    """Model to learn from human-like reasoning"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.pos_encoding = PositionalEncoding(config['hidden_size'])
        self.dropout = nn.Dropout(config['dropout'])
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['hidden_size'],
            nhead=config['num_heads'],
            dim_feedforward=config['hidden_size'] * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['num_h_layers'] + config['num_l_layers']
        )
        
        # Output layer
        self.output_layer = nn.Linear(config['hidden_size'], config['vocab_size'])
    
    def forward(self, x):
        # Embed tokens
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transform
        x = self.transformer(x)
        
        # Output
        return self.output_layer(x)

def grid_to_tensor(grid: List[List[int]], size: int = 30) -> torch.LongTensor:
    """Convert grid to flattened tensor with padding"""
    tensor = torch.zeros(size, size, dtype=torch.long)
    
    if grid:
        h, w = len(grid), len(grid[0]) if grid[0] else 0
        for i in range(min(h, size)):
            for j in range(min(w, size)):
                tensor[i, j] = grid[i][j]
    
    # Flatten to sequence
    return tensor.flatten()

def load_training_data(path: str = 'sage_v3_training_data.json'):
    """Load and prepare training data"""
    with open(path, 'r') as f:
        data = json.load(f)
    
    pairs = []
    for task_id, task in data.items():
        # Training examples
        for ex in task.get('train', []):
            if 'input' in ex and 'output' in ex:
                pairs.append((ex['input'], ex['output']))
        
        # Test examples with predictions
        for ex in task.get('test', []):
            if 'input' in ex and 'output' in ex:
                pairs.append((ex['input'], ex['output']))
    
    return pairs

def train():
    """Train SAGE V3"""
    print("=" * 60)
    print("SAGE V3 Training - Human-like Visual Reasoning")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\nLoading training data...")
    pairs = load_training_data()
    print(f"Loaded {len(pairs)} input-output pairs")
    
    # Prepare tensors
    print("Preparing tensors...")
    inputs = []
    targets = []
    
    for inp, out in pairs:
        inputs.append(grid_to_tensor(inp))
        targets.append(grid_to_tensor(out))
    
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=16,  # Small batch for memory
        shuffle=True
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = FaithfulModel(MODEL_CONFIG).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,
        total_steps=len(dataloader) * 10,  # 10 epochs
        pct_start=0.1
    )
    
    # Training loop
    num_epochs = 10  # Reduced for quick training
    print(f"\nTraining for {num_epochs} epochs...")
    print("-" * 40)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward
            outputs = model(batch_inputs)
            
            # Reshape for loss
            outputs = outputs.view(-1, MODEL_CONFIG['vocab_size'])
            batch_targets = batch_targets.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, batch_targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            mask = batch_targets != 0  # Ignore padding
            correct += (predicted[mask] == batch_targets[mask]).sum().item()
            total += mask.sum().item()
        
        # Epoch statistics
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total if total > 0 else 0
        
        # Print every epoch for quick training
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}% | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
                'config': MODEL_CONFIG
            }, 'sage_v3_best.pt')
    
    print("-" * 40)
    print(f"Training complete! Best loss: {best_loss:.4f}")
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': MODEL_CONFIG
    }, 'sage_v3_final.pt')
    
    print(f"\nModels saved:")
    print(f"  - sage_v3_best.pt (best loss)")
    print(f"  - sage_v3_final.pt (final epoch)")
    
    return model

if __name__ == '__main__':
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Train
    model = train()