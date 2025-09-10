#!/usr/bin/env python3
"""
Standalone training script - no slow imports!
Train SAGE-7M from scratch using Claude's reasoning predictions.
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
    'dropout': 0.0,
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

class HierarchicalReasoningModule(nn.Module):
    """Simplified HRM for faster training"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.pos_encoding = PositionalEncoding(config['hidden_size'])
        
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
        
        # Transform
        x = self.transformer(x)
        
        # Output
        return self.output_layer(x)

class ARCDataset(Dataset):
    def __init__(self, data_path, max_size=30):
        self.max_size = max_size
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.examples = []
        for task_id, task_data in self.data.items():
            for example in task_data.get('train', []):
                self.examples.append({
                    'input': example['input'],
                    'output': example['output'],
                    'source': 'original'
                })
            for example in task_data.get('test', []):
                self.examples.append({
                    'input': example['input'],
                    'output': example['output'],
                    'source': 'claude'
                })
        
        print(f"Loaded {len(self.examples)} examples")
        claude_count = sum(1 for e in self.examples if e['source'] == 'claude')
        print(f"  Claude predictions: {claude_count}")
        print(f"  Original training: {len(self.examples) - claude_count}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        input_tensor = preprocess_grid(example['input'], self.max_size)
        output_tensor = preprocess_grid(example['output'], self.max_size)
        return input_tensor, output_tensor, example['source']

def main():
    print("=" * 60)
    print("SAGE-7M Standalone Training")
    print("=" * 60)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\nLoading data...")
    dataset = ARCDataset('claude_reasoning_training_data.json')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create model
    print("\nCreating model...")
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(10):  # Quick test with 10 epochs
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets, sources) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
            _, predicted = outputs.max(2)
            correct += predicted.eq(targets).float().mean().item()
            total += 1
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}")
        
        # Epoch stats
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
        
        # Check GPU
        if device.type == 'cuda':
            print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e6:.1f}MB")
    
    # Save model
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': MODEL_CONFIG,
        'final_loss': avg_loss,
        'final_accuracy': accuracy
    }, 'sage_standalone_trained.pt')
    
    print("Training complete!")

if __name__ == '__main__':
    main()