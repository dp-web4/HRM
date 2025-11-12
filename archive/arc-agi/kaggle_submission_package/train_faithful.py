#!/usr/bin/env python3
"""
Train model to FAITHFULLY reproduce Claude's predictions.
Focus on exact reproduction, not general learning.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import math
from pathlib import Path

# Model configuration
MODEL_CONFIG = {
    'seq_len': 900,
    'vocab_size': 12,
    'hidden_size': 256,
    'num_heads': 8,
    'num_h_layers': 4,
    'num_l_layers': 3,
    'dropout': 0.1,  # Add some dropout for regularization
    'max_cycles': 8
}

def preprocess_grid(grid, max_size=30):
    """Convert ARC grid to model input tensor"""
    grid_array = np.array(grid, dtype=np.int32)
    h, w = grid_array.shape
    
    # Pad to max_size x max_size with zeros
    padded = np.zeros((max_size, max_size), dtype=np.int32)
    padded[:min(h, max_size), :min(w, max_size)] = grid_array[:min(h, max_size), :min(w, max_size)]
    
    # Flatten to sequence
    return torch.tensor(padded.flatten(), dtype=torch.long)

class ClaudeDataset(Dataset):
    """Dataset focused on Claude's predictions only"""
    
    def __init__(self, data_path, max_size=30, claude_only=True):
        self.max_size = max_size
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.examples = []
        for task_id, task_data in self.data.items():
            # Focus on Claude's predictions (in 'test' key)
            for example in task_data.get('test', []):
                self.examples.append({
                    'input': example['input'],
                    'output': example['output'],
                    'task_id': task_id,
                    'source': 'claude'
                })
            
            # Optionally include original training data
            if not claude_only:
                for example in task_data.get('train', []):
                    self.examples.append({
                        'input': example['input'],
                        'output': example['output'],
                        'task_id': task_id,
                        'source': 'original'
                    })
        
        print(f"Dataset loaded: {len(self.examples)} examples")
        claude_count = sum(1 for e in self.examples if e['source'] == 'claude')
        print(f"  Claude predictions: {claude_count}")
        if not claude_only:
            print(f"  Original training: {len(self.examples) - claude_count}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input_tensor = preprocess_grid(example['input'], self.max_size)
        output_tensor = preprocess_grid(example['output'], self.max_size)
        return input_tensor, output_tensor, example['source']

class PositionalEncoding(nn.Module):
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
    """Model optimized for faithful reproduction"""
    
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

def compute_weighted_loss(outputs, targets, zero_weight=0.1):
    """
    Compute loss with different weights for zero and non-zero predictions.
    Lower weight for zeros since they're common and less important.
    """
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss = criterion(outputs.transpose(1, 2), targets)
    
    # Create weight mask: lower weight for zero targets
    weights = torch.ones_like(targets, dtype=torch.float)
    weights[targets == 0] = zero_weight
    
    # Apply weights and average
    weighted_loss = (loss * weights).mean()
    return weighted_loss

def main():
    print("=" * 60)
    print("Faithful Claude Reproduction Training")
    print("=" * 60)
    
    # Configuration
    BATCH_SIZE = 4  # Smaller batch for better convergence
    LEARNING_RATE = 5e-4
    NUM_EPOCHS = 100  # Many epochs for memorization
    SAVE_EVERY = 10
    ZERO_WEIGHT = 0.1  # Lower weight for zero predictions
    CLAUDE_ONLY = True  # Train only on Claude's predictions
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\nLoading Claude's predictions...")
    dataset = ClaudeDataset('claude_reasoning_training_data.json', claude_only=CLAUDE_ONLY)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create model
    print("\nCreating model...")
    model = FaithfulModel(MODEL_CONFIG).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    print(f"Focus: Faithful reproduction of Claude's outputs")
    print("-" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        non_zero_accuracy = 0
        non_zero_count = 0
        
        for batch_idx, (inputs, targets, sources) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Weighted loss
            loss = compute_weighted_loss(outputs, targets, zero_weight=ZERO_WEIGHT)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(2)
            
            # Overall accuracy
            correct = predicted.eq(targets).float()
            correct_predictions += correct.sum().item()
            total_predictions += targets.numel()
            
            # Non-zero accuracy (more important!)
            non_zero_mask = targets != 0
            if non_zero_mask.any():
                non_zero_correct = (predicted[non_zero_mask] == targets[non_zero_mask]).float()
                non_zero_accuracy += non_zero_correct.sum().item()
                non_zero_count += non_zero_mask.sum().item()
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}")
        
        # Epoch statistics
        avg_loss = total_loss / len(dataloader)
        overall_acc = 100. * correct_predictions / total_predictions
        nz_acc = 100. * non_zero_accuracy / non_zero_count if non_zero_count > 0 else 0
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Overall Accuracy: {overall_acc:.2f}%")
        print(f"  Non-Zero Accuracy: {nz_acc:.2f}% (KEY METRIC!)")
        
        if device.type == 'cuda':
            print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e6:.1f}MB")
        
        # Save checkpoint
        if (epoch + 1) % SAVE_EVERY == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_name = 'faithful_model_best.pt'
            else:
                checkpoint_name = f'faithful_model_epoch_{epoch+1}.pt'
            
            print(f"  Saving {checkpoint_name}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': MODEL_CONFIG,
                'epoch': epoch + 1,
                'loss': avg_loss,
                'overall_accuracy': overall_acc,
                'non_zero_accuracy': nz_acc
            }, checkpoint_name)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best loss: {best_loss:.4f}")
    print("Model saved as 'faithful_model_best.pt'")
    print("=" * 60)

if __name__ == '__main__':
    main()