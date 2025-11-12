#!/usr/bin/env python3
"""
V3 Distillation Training - Learn to reproduce Claude's reasoned solutions
This model will be trained to faithfully reproduce the context-aware solutions
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from pathlib import Path
import time

# Model configuration
MODEL_CONFIG = {
    'seq_len': 900,  # 30x30 grid
    'vocab_size': 12,  # 0-9 colors + padding
    'hidden_size': 256,
    'num_heads': 8,
    'num_layers': 6,  # Simplified from H+L layers
    'dropout': 0.1,  # Add some dropout for regularization
    'pattern_dims': 16,  # For pattern classification
}

class ARCDataset(Dataset):
    """Dataset for ARC puzzles with Claude's solutions"""
    
    def __init__(self, data_path, max_size=30):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.max_size = max_size
        self.examples = []
        
        # Extract all input-output pairs
        for task_id, task_data in self.data.items():
            pattern = task_data.get('pattern', 'unknown')
            
            for example in task_data.get('test', []):
                self.examples.append({
                    'input': example['input'],
                    'output': example['output'],
                    'pattern': pattern,
                    'task_id': task_id
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Preprocess input
        input_grid = np.array(example['input'], dtype=np.int32)
        h_in, w_in = input_grid.shape
        input_padded = np.zeros((self.max_size, self.max_size), dtype=np.int32)
        input_padded[:min(h_in, self.max_size), :min(w_in, self.max_size)] = \
            input_grid[:min(h_in, self.max_size), :min(w_in, self.max_size)]
        
        # Preprocess output
        output_grid = np.array(example['output'], dtype=np.int32)
        h_out, w_out = output_grid.shape
        output_padded = np.zeros((self.max_size, self.max_size), dtype=np.int32)
        output_padded[:min(h_out, self.max_size), :min(w_out, self.max_size)] = \
            output_grid[:min(h_out, self.max_size), :min(w_out, self.max_size)]
        
        # Flatten to sequences
        input_seq = torch.tensor(input_padded.flatten(), dtype=torch.long)
        output_seq = torch.tensor(output_padded.flatten(), dtype=torch.long)
        
        # Pattern encoding
        pattern_map = {'unknown': 0, 'fill_rectangles': 1, 'extract_pattern': 2, 
                      'tile_3x3': 3, 'tile_2x2': 4}
        pattern_id = pattern_map.get(example['pattern'], 0)
        
        return {
            'input': input_seq,
            'output': output_seq,
            'pattern': torch.tensor(pattern_id, dtype=torch.long),
            'output_h': h_out,
            'output_w': w_out
        }

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

class V3ReasoningModel(nn.Module):
    """Model that learns to reproduce Claude's reasoned solutions"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.pattern_embedding = nn.Embedding(5, config['pattern_dims'])  # 5 pattern types
        self.pos_encoding = PositionalEncoding(config['hidden_size'])
        
        # Pattern-aware projection
        self.input_projection = nn.Linear(
            config['hidden_size'] + config['pattern_dims'], 
            config['hidden_size']
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['hidden_size'],
            nhead=config['num_heads'],
            dim_feedforward=config['hidden_size'] * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        
        # Output head
        self.output_layer = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x, pattern=None):
        batch_size, seq_len = x.shape
        
        # Embed tokens
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        
        # Add pattern information if provided
        if pattern is not None:
            pattern_emb = self.pattern_embedding(pattern)  # [batch, pattern_dims]
            pattern_emb = pattern_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, pattern_dims]
            x = torch.cat([x, pattern_emb], dim=-1)  # [batch, seq, hidden+pattern]
            x = self.input_projection(x)  # [batch, seq, hidden]
        
        x = self.dropout(x)
        
        # Transform
        x = self.transformer(x)
        
        # Output
        return self.output_layer(x)

def compute_accuracy(outputs, targets, ignore_index=0):
    """Compute pixel-wise accuracy"""
    pred = outputs.argmax(dim=-1)
    mask = targets != ignore_index  # Don't count padding
    correct = (pred == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy

def compute_non_zero_accuracy(outputs, targets):
    """Compute accuracy on non-zero predictions"""
    pred = outputs.argmax(dim=-1)
    
    # Check non-zero patterns
    target_non_zero = targets != 0
    pred_non_zero = pred != 0
    
    # Accuracy on non-zero positions
    non_zero_correct = (pred == targets) & target_non_zero
    if target_non_zero.sum() > 0:
        return non_zero_correct.sum().float() / target_non_zero.sum().float()
    else:
        return torch.tensor(1.0)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_non_zero_acc = 0
    num_batches = 0
    
    for batch in dataloader:
        inputs = batch['input'].to(device)
        targets = batch['output'].to(device)
        patterns = batch['pattern'].to(device)
        
        # Forward pass
        outputs = model(inputs, patterns)
        
        # Compute loss
        loss = criterion(outputs.transpose(1, 2), targets)
        
        # Compute accuracy
        accuracy = compute_accuracy(outputs, targets)
        non_zero_acc = compute_non_zero_accuracy(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        total_non_zero_acc += non_zero_acc.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_accuracy / num_batches,
        'non_zero_accuracy': total_non_zero_acc / num_batches
    }

def main():
    print("=" * 60)
    print("V3 DISTILLATION TRAINING")
    print("Learning to reproduce Claude's reasoned solutions")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = ARCDataset('claude_v3_training_data.json')
    print(f"Total examples: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Create model
    print("\nInitializing model...")
    model = V3ReasoningModel(MODEL_CONFIG).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training loop
    print("\nStarting training...")
    print("-" * 60)
    
    best_accuracy = 0
    num_epochs = 100
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        metrics = train_epoch(model, dataloader, criterion, optimizer, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Non-zero Accuracy: {metrics['non_zero_accuracy']*100:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': metrics['accuracy'],
                'non_zero_accuracy': metrics['non_zero_accuracy'],
                'loss': metrics['loss'],
                'config': MODEL_CONFIG
            }
            torch.save(checkpoint, 'v3_reasoning_model.pt')
            print(f"  ✓ Saved best model (accuracy: {best_accuracy*100:.2f}%)")
        
        # Early stopping if perfect
        if metrics['accuracy'] > 0.99 and metrics['non_zero_accuracy'] > 0.98:
            print("\n✨ Achieved excellent accuracy - stopping early")
            break
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best accuracy: {best_accuracy*100:.2f}%")
    print(f"Model saved as: v3_reasoning_model.pt")
    print("=" * 60)

if __name__ == '__main__':
    main()