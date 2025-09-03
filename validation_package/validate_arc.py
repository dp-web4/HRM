#!/usr/bin/env python3
"""
Standalone ARC Validation Script for HRM Model
Can be run on any machine with the model checkpoint and dataset
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
import math
import argparse
from typing import Dict, Any, Tuple, Optional

# Configuration (matching training script)
MODEL_CONFIG = {
    'batch_size': 8,
    'seq_len': 900,
    'vocab_size': 12,
    'hidden_size': 256,
    'num_heads': 8,
    'num_h_layers': 4,
    'num_l_layers': 3,
    'dropout': 0.1,
    'max_cycles': 8,
}

class ARCDataset(Dataset):
    """Dataset loader for ARC puzzles"""
    
    def __init__(self, data_dir, split='test'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load data files
        split_dir = self.data_dir / split
        if split_dir.exists() and (split_dir / 'all__inputs.npy').exists():
            self.inputs = np.load(split_dir / 'all__inputs.npy')
            self.labels = np.load(split_dir / 'all__labels.npy')
            print(f"Loaded {len(self.inputs)} {split} samples")
        else:
            raise FileNotFoundError(f"Dataset not found at {split_dir}")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'input': torch.from_numpy(self.inputs[idx]).long(),
            'target': torch.from_numpy(self.labels[idx]).long()
        }

class PositionalEncoding(nn.Module):
    """RoPE-style positional encoding"""
    
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
    """Simplified HRM-style architecture"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.pos_encoding = PositionalEncoding(config['hidden_size'])
        
        # H-level (strategic) layers
        self.h_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_h_layers'])
        ])
        
        # L-level (tactical) layers
        self.l_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_l_layers'])
        ])
        
        # Interaction layers
        self.h_to_l = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.l_to_h = nn.Linear(config['hidden_size'], config['hidden_size'])
        
        # Halting mechanism
        self.halt_predictor = nn.Linear(config['hidden_size'] * 2, 1)
        
        # Output layer
        self.output = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        # Layer norms
        self.h_norm = nn.LayerNorm(config['hidden_size'])
        self.l_norm = nn.LayerNorm(config['hidden_size'])
        
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x, max_cycles=None):
        batch_size, seq_len = x.shape
        max_cycles = max_cycles or self.config['max_cycles']
        
        # Embed input
        x_emb = self.token_embedding(x)
        x_emb = self.pos_encoding(x_emb)
        x_emb = self.dropout(x_emb)
        
        # Initialize H and L states
        h_state = x_emb.clone()
        l_state = x_emb.clone()
        
        # Store halting probabilities
        halt_probs = []
        
        # Reasoning cycles
        for cycle in range(max_cycles):
            # H-level processing (strategic)
            h_prev = h_state.clone()
            for h_layer in self.h_layers:
                h_state = h_layer(h_state)
            h_state = self.h_norm(h_state)
            
            # L-level processing (tactical)
            l_prev = l_state.clone()
            # Incorporate H-level guidance
            l_state = l_state + self.h_to_l(h_state)
            for l_layer in self.l_layers:
                l_state = l_layer(l_state)
            l_state = self.l_norm(l_state)
            
            # L to H feedback
            h_state = h_state + self.l_to_h(l_state)
            
            # Compute halting probability
            combined = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)
            halt_prob = torch.sigmoid(self.halt_predictor(combined))
            halt_probs.append(halt_prob)
            
            # Early stopping based on halt probability
            if cycle > 0 and halt_prob.mean() > 0.9:
                break
        
        # Final output from L-level (execution)
        output = self.output(l_state)
        
        return output, halt_probs

def validate(model_path, data_dir, device='cuda', batch_size=8, max_batches=None):
    """Run validation on a model checkpoint"""
    
    print(f"🔍 Validating model: {model_path}")
    print(f"📊 Dataset: {data_dir}")
    print(f"🖥️  Device: {device}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get model info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {param_count/1e6:.1f}M")
    
    if 'global_step' in checkpoint:
        print(f"📍 Checkpoint from step: {checkpoint['global_step']}")
    if 'best_val_loss' in checkpoint:
        print(f"📈 Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    # Load dataset
    dataset = ARCDataset(data_dir, 'test')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Validation loop
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_cycles = 0
    num_batches = 0
    
    print("\n🏃 Running validation...")
    progress_bar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            if max_batches and batch_idx >= max_batches:
                break
                
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs, halt_probs = model(inputs)
            
            # Calculate loss
            loss = F.cross_entropy(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )
            
            # Calculate accuracy
            predictions = outputs.argmax(dim=-1)
            correct = (predictions == targets).float().sum().item()
            num_elements = targets.numel()
            
            # Update metrics
            total_loss += loss.item()
            total_correct += correct
            total_samples += num_elements
            total_cycles += len(halt_probs)
            num_batches += 1
            
            # Update progress bar
            current_acc = correct / num_elements
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{current_acc:.4f}",
                'cycles': len(halt_probs)
            })
    
    # Final metrics
    avg_loss = total_loss / num_batches
    avg_accuracy = total_correct / total_samples
    avg_cycles = total_cycles / num_batches
    
    print("\n" + "="*50)
    print("📊 VALIDATION RESULTS")
    print("="*50)
    print(f"Loss:     {avg_loss:.4f}")
    print(f"Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"Avg Cycles: {avg_cycles:.2f}")
    print(f"Total Samples: {total_samples}")
    print("="*50)
    
    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'cycles': avg_cycles,
        'total_samples': total_samples
    }

def main():
    parser = argparse.ArgumentParser(description='Validate HRM model on ARC dataset')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--max-batches', type=int, help='Max batches to validate (for quick tests)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Run validation
    results = validate(
        model_path=args.model,
        data_dir=args.data,
        device=args.device,
        batch_size=args.batch_size,
        max_batches=args.max_batches
    )
    
    # Save results
    results_file = Path(args.model).stem + '_validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {results_file}")

if __name__ == "__main__":
    main()