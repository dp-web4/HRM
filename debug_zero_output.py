#!/usr/bin/env python3
"""
Debug why HRM always outputs zeros
Following Nova's systematic debugging approach
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
import math
from pathlib import Path
from collections import OrderedDict

# Add parent directory to path
sys.path.append('.')
sys.path.append('training')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define HRM model inline to ensure consistency
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.pos_encoding = PositionalEncoding(config['hidden_size'])
        
        # H-layers (high-level reasoning)
        self.h_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_h_layers'])
        ])
        
        # L-layers (low-level processing)
        self.l_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_l_layers'])
        ])
        
        # Cross-layer connections
        self.h_to_l = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.l_to_h = nn.Linear(config['hidden_size'], config['hidden_size'])
        
        # Halt predictor
        self.halt_predictor = nn.Linear(config['hidden_size'] * 2, 1)
        
        # Cycle embedding
        self.cycle_embedding = nn.Embedding(config['max_cycles'], config['hidden_size'])
        
        # Output projection - CRITICAL: Should output vocab_size classes
        self.output = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        # Layer norms
        self.h_norm = nn.LayerNorm(config['hidden_size'])
        self.l_norm = nn.LayerNorm(config['hidden_size'])
        
        # Dropout
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x, max_cycles=None):
        batch_size, seq_len = x.shape
        max_cycles = max_cycles or self.config['max_cycles']
        
        # Embed tokens
        x_emb = self.token_embedding(x)
        x_emb = self.pos_encoding(x_emb)
        x_emb = self.dropout(x_emb)
        
        h_state = x_emb.clone()
        l_state = x_emb.clone()
        
        halt_probs = []
        cumulative_halt = torch.zeros(batch_size, 1).to(x.device)
        
        for cycle in range(max_cycles):
            # Add cycle embedding
            cycle_emb = self.cycle_embedding(torch.tensor([cycle], device=x.device))
            cycle_emb = cycle_emb.expand(batch_size, seq_len, -1)
            
            # H-level processing
            h_state = h_state + 0.1 * cycle_emb
            for h_layer in self.h_layers:
                h_state = h_layer(h_state)
            h_state = self.h_norm(h_state)
            
            # L-level processing with H influence
            l_state = l_state + self.h_to_l(h_state)
            for l_layer in self.l_layers:
                l_state = l_layer(l_state)
            l_state = self.l_norm(l_state)
            
            # Update H with L information
            h_state = h_state + self.l_to_h(l_state)
            
            # Compute halt probability
            combined = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)
            halt_logit = self.halt_predictor(combined)
            halt_prob = torch.sigmoid(halt_logit)
            halt_probs.append(halt_prob)
            
            cumulative_halt = cumulative_halt + halt_prob
            
            # Check stopping condition
            if cycle >= 3 and cumulative_halt.mean() > 1.0:
                break
        
        # Generate output
        output = self.output(l_state)
        
        return output, halt_probs

def debug_hrm_zero_output():
    """Comprehensive debugging following Nova's suggestions"""
    
    print("=" * 80)
    print("HRM ZERO OUTPUT DEBUGGING")
    print("Following Nova's systematic approach")
    print("=" * 80)
    
    # Model configuration
    model_config = {
        'vocab_size': 12,  # ARC has 0-10 plus padding
        'hidden_size': 256,
        'num_heads': 8,
        'num_h_layers': 4,
        'num_l_layers': 6,
        'dropout': 0.1,
        'max_cycles': 8
    }
    
    print("\n" + "=" * 80)
    print("A. ARCHITECTURE SANITY CHECK")
    print("=" * 80)
    
    # Create model
    model = HierarchicalReasoningModule(model_config).to(DEVICE)
    
    # Check architecture
    print(f"Model class: {model.__class__.__name__}")
    print(f"Config vocab_size: {model_config['vocab_size']}")
    
    # Find output layer
    output_layer = None
    for name, module in model.named_modules():
        if name == 'output':
            output_layer = module
            break
    
    if output_layer:
        print(f"Output layer found: {output_layer}")
        print(f"Output dimensions: in={output_layer.in_features}, out={output_layer.out_features}")
        assert output_layer.out_features == model_config['vocab_size'], \
            f"Output layer should have {model_config['vocab_size']} outputs, has {output_layer.out_features}"
        print("✅ Output layer correctly configured for 12 classes")
    else:
        print("❌ No output layer found!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\n" + "=" * 80)
    print("B. CHECKPOINT LOADING VERIFICATION")
    print("=" * 80)
    
    # Load checkpoint
    checkpoint_paths = [
        'training/checkpoints/hrm_arc_step_193000.pt',
        'training/checkpoints/hrm_arc_step_125000.pt',
        'training/checkpoints/hrm_arc_step_100000.pt',
        'training/checkpoints/hrm_arc_best.pt'
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if Path(path).exists():
            checkpoint_path = path
            break
    
    if not checkpoint_path:
        print("❌ No checkpoint found!")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Check what's in the checkpoint
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Try loading with strict=False to see issues
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    print(f"Missing keys: {len(missing)}")
    if missing:
        print(f"  Sample missing (first 5): {missing[:5]}")
    
    print(f"Unexpected keys: {len(unexpected)}")
    if unexpected:
        print(f"  Sample unexpected (first 5): {unexpected[:5]}")
    
    if len(missing) == 0 and len(unexpected) == 0:
        print("✅ Checkpoint loaded perfectly")
    elif len(missing) > 0:
        print("⚠️  Some keys missing - model may not be fully initialized")
    
    # Check if keys have prefixes
    sample_keys = list(state_dict.keys())[:5]
    print(f"Sample state dict keys: {sample_keys}")
    
    # Check output layer weights
    if 'output.weight' in state_dict:
        output_weights = state_dict['output.weight']
        print(f"Output layer weight shape in checkpoint: {output_weights.shape}")
        print(f"Output weight stats: min={output_weights.min():.3f}, max={output_weights.max():.3f}, mean={output_weights.mean():.3f}")
    else:
        print("⚠️  No 'output.weight' in checkpoint!")
    
    print("\n" + "=" * 80)
    print("C. INPUT ENCODING CHECK")
    print("=" * 80)
    
    # Create test input
    test_input = torch.tensor([[1, 2, 3, 0, 0, 4, 5, 6, 0] * 100], dtype=torch.long).to(DEVICE)  # 900 length
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input unique values: {torch.unique(test_input).tolist()}")
    print(f"Test input dtype: {test_input.dtype}")
    
    print("\n" + "=" * 80)
    print("D. FORWARD PASS INSPECTION")
    print("=" * 80)
    
    model.eval()
    with torch.no_grad():
        output, halt_probs = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [1, 900, 12]")
    
    # Check output statistics
    print(f"Output min: {output.min():.3f}")
    print(f"Output max: {output.max():.3f}")
    print(f"Output mean: {output.mean():.3f}")
    print(f"Output std: {output.std():.3f}")
    
    # Check per-class statistics
    if output.shape[-1] == 12:
        print("\nPer-class logit means:")
        for i in range(12):
            class_mean = output[:, :, i].mean().item()
            print(f"  Class {i}: {class_mean:.3f}")
    
    # Get predictions
    predictions = output.argmax(dim=-1)
    unique_preds = torch.unique(predictions)
    print(f"\nUnique predictions: {unique_preds.tolist()}")
    
    # Count zeros
    zeros = (predictions == 0).sum().item()
    total = predictions.numel()
    print(f"Zeros in predictions: {zeros}/{total} ({zeros/total*100:.1f}%)")
    
    # Check if all predictions are the same
    if len(unique_preds) == 1:
        print(f"❌ Model predicts only class {unique_preds[0].item()}")
    else:
        print(f"✅ Model predicts {len(unique_preds)} different classes")
    
    print("\n" + "=" * 80)
    print("E. TEST WITH RANDOM OUTPUTS")
    print("=" * 80)
    
    # Generate random outputs to test decoding
    random_output = torch.randn(1, 900, 12).to(DEVICE)
    random_preds = random_output.argmax(dim=-1)
    random_unique = torch.unique(random_preds)
    
    print(f"Random output shape: {random_output.shape}")
    print(f"Random unique predictions: {random_unique.tolist()}")
    print(f"Random prediction distribution:")
    for i in range(12):
        count = (random_preds == i).sum().item()
        if count > 0:
            print(f"  Class {i}: {count} ({count/900*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("F. DIAGNOSIS")
    print("=" * 80)
    
    # Analyze the problem
    if len(unique_preds) == 1 and unique_preds[0] == 0:
        print("❌ PROBLEM CONFIRMED: Model outputs all zeros")
        
        # Check which is most likely cause
        if output.std() < 0.01:
            print("  → Likely cause: Weights not loaded properly or model at initialization")
        elif output[:, :, 0].mean() > output[:, :, 1:].mean() + 1.0:
            print("  → Likely cause: Class 0 heavily biased in output layer")
        else:
            print("  → Likely cause: Training issue - model learned to always predict 0")
    else:
        print("✅ Model produces diverse outputs")
    
    # Additional checks
    print("\n" + "=" * 80)
    print("ADDITIONAL DEBUGGING")
    print("=" * 80)
    
    # Check embedding layer
    embedding_weights = model.token_embedding.weight
    print(f"Embedding shape: {embedding_weights.shape}")
    print(f"Embedding stats: min={embedding_weights.min():.3f}, max={embedding_weights.max():.3f}, std={embedding_weights.std():.3f}")
    
    # Check if embeddings are initialized
    if embedding_weights.std() < 0.01:
        print("⚠️  Embeddings might not be initialized properly")
    
    # Test with different checkpoint if available
    if Path('training/checkpoints/hrm_arc_best.pt').exists():
        print("\nTrying hrm_arc_best.pt checkpoint...")
        best_checkpoint = torch.load('training/checkpoints/hrm_arc_best.pt', map_location=DEVICE, weights_only=False)
        
        if 'model_state_dict' in best_checkpoint:
            best_state = best_checkpoint['model_state_dict']
        else:
            best_state = best_checkpoint
            
        model2 = HierarchicalReasoningModule(model_config).to(DEVICE)
        model2.load_state_dict(best_state, strict=False)
        model2.eval()
        
        with torch.no_grad():
            output2, _ = model2(test_input)
            preds2 = output2.argmax(dim=-1)
            unique2 = torch.unique(preds2)
            print(f"Best checkpoint predictions: {unique2.tolist()}")

if __name__ == "__main__":
    debug_hrm_zero_output()