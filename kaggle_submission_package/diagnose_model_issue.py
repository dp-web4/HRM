#!/usr/bin/env python3
"""
Comprehensive diagnostic to understand why the model scores 0.00 on ARC tasks.
This script checks:
1. Model architecture and checkpoint loading
2. Weight initialization patterns
3. Output behavior on simple patterns
4. Token prediction distributions
5. Whether the model is actually learning or just memorizing
"""

import json
import torch
import torch.nn as nn
import numpy as np
import math
from pathlib import Path
from collections import Counter
import os

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
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.pos_encoding = PositionalEncoding(config['hidden_size'])
        self.dropout = nn.Dropout(config['dropout'])
        
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
        
        self.output_layer = nn.Linear(config['hidden_size'], config['vocab_size'])
    
    def forward(self, x):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        x = self.transformer(x)
        return self.output_layer(x)

def preprocess_grid(grid, max_size=30):
    """Convert ARC grid to model input tensor"""
    grid_array = np.array(grid, dtype=np.int32)
    h, w = grid_array.shape
    
    padded = np.zeros((max_size, max_size), dtype=np.int32)
    padded[:min(h, max_size), :min(w, max_size)] = grid_array[:min(h, max_size), :min(w, max_size)]
    
    return torch.tensor(padded.flatten(), dtype=torch.long)

def check_model_initialization(model):
    """Check if model weights are properly initialized"""
    print("\n1. MODEL WEIGHT ANALYSIS")
    print("-" * 60)
    
    # Check for zero weights (uninitialized)
    zero_params = 0
    near_zero_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            zeros = (param.data == 0).sum().item()
            near_zeros = (param.data.abs() < 1e-6).sum().item()
            
            zero_params += zeros
            near_zero_params += near_zeros
            
            if zeros > param.numel() * 0.1:  # More than 10% zeros
                print(f"  WARNING: {name} has {zeros}/{param.numel()} zero weights!")
    
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Zero weights: {zero_params:,} ({100*zero_params/total_params:.2f}%)")
    print(f"  Near-zero weights: {near_zero_params:,} ({100*near_zero_params/total_params:.2f}%)")
    
    # Check weight distribution
    all_weights = []
    for param in model.parameters():
        if param.requires_grad:
            all_weights.extend(param.data.cpu().numpy().flatten())
    
    all_weights = np.array(all_weights)
    print(f"\n  Weight statistics:")
    print(f"    Mean: {all_weights.mean():.6f}")
    print(f"    Std:  {all_weights.std():.6f}")
    print(f"    Min:  {all_weights.min():.6f}")
    print(f"    Max:  {all_weights.max():.6f}")
    
    # Check if weights look like they were trained
    if all_weights.std() < 0.01:
        print("  ⚠️ WARNING: Very low weight variance - model may not be trained!")
    elif all_weights.std() > 10:
        print("  ⚠️ WARNING: Very high weight variance - possible instability!")

def test_output_behavior(model, device):
    """Test model's output behavior on various inputs"""
    print("\n2. OUTPUT BEHAVIOR ANALYSIS")
    print("-" * 60)
    
    model.eval()
    
    # Test patterns
    test_cases = [
        ("Empty (all zeros)", [[0, 0], [0, 0]]),
        ("Single pixel", [[1, 0], [0, 0]]),
        ("All same (1s)", [[1, 1], [1, 1]]),
        ("All same (5s)", [[5, 5], [5, 5]]),
        ("Pattern", [[1, 2], [3, 4]]),
        ("Large values", [[7, 8], [9, 9]])
    ]
    
    all_predictions = []
    
    for name, pattern in test_cases:
        input_tensor = preprocess_grid(pattern).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output[0], dim=-1)
            predictions = output[0].argmax(dim=-1).cpu().numpy()
            
            # Get the first 4 predictions (2x2 grid)
            pred_grid = predictions.reshape(30, 30)[:2, :2]
            
            # Analyze confidence
            max_probs = probs.max(dim=-1)[0]
            avg_conf = max_probs.mean().item()
            
            print(f"\n  {name}:")
            print(f"    Input:  {pattern}")
            print(f"    Output: {pred_grid.tolist()}")
            print(f"    Avg confidence: {avg_conf:.3f}")
            
            all_predictions.extend(predictions.flatten())
    
    # Analyze prediction distribution
    pred_counter = Counter(all_predictions)
    print(f"\n  Overall token distribution across all tests:")
    for token, count in sorted(pred_counter.items()):
        print(f"    Token {token}: {count} times ({100*count/len(all_predictions):.1f}%)")
    
    # Check if model always predicts the same thing
    if len(pred_counter) == 1:
        print("  ⚠️ WARNING: Model only predicts one token value!")
    elif len(pred_counter) <= 3:
        print("  ⚠️ WARNING: Model has very limited output diversity!")

def test_gradient_flow(model, device):
    """Test if gradients flow properly through the model"""
    print("\n3. GRADIENT FLOW CHECK")
    print("-" * 60)
    
    model.train()
    
    # Create a simple input/output pair
    input_grid = [[1, 2], [3, 4]]
    output_grid = [[4, 3], [2, 1]]  # Reversed pattern
    
    input_tensor = preprocess_grid(input_grid).unsqueeze(0).to(device)
    target_tensor = preprocess_grid(output_grid).unsqueeze(0).to(device)
    
    # Forward pass
    output = model(input_tensor)
    loss = nn.CrossEntropyLoss()(output.view(-1, MODEL_CONFIG['vocab_size']), target_tensor.view(-1))
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Check gradients
    grad_norms = {}
    zero_grad_layers = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
            
            if grad_norm == 0:
                zero_grad_layers.append(name)
        else:
            zero_grad_layers.append(name)
    
    print(f"  Loss value: {loss.item():.4f}")
    print(f"  Layers with gradients: {len(grad_norms)}/{len(list(model.named_parameters()))}")
    
    if zero_grad_layers:
        print(f"  ⚠️ WARNING: {len(zero_grad_layers)} layers have zero/no gradients!")
        for layer in zero_grad_layers[:5]:  # Show first 5
            print(f"    - {layer}")
    
    # Show gradient magnitudes for key layers
    print("\n  Gradient norms for key layers:")
    key_layers = ['token_embedding.weight', 'output_layer.weight', 'transformer.layers.0.self_attn.out_proj.weight']
    for layer in key_layers:
        if layer in grad_norms:
            print(f"    {layer}: {grad_norms[layer]:.6f}")

def check_checkpoint_info(checkpoint_path):
    """Analyze checkpoint contents"""
    print("\n4. CHECKPOINT ANALYSIS")
    print("-" * 60)
    
    if not os.path.exists(checkpoint_path):
        print(f"  Checkpoint not found at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict):
        print("  Checkpoint contents:")
        for key in checkpoint.keys():
            if key == 'model_state_dict':
                num_params = len(checkpoint[key])
                print(f"    - {key}: {num_params} parameter tensors")
            elif key == 'optimizer_state_dict':
                print(f"    - {key}: optimizer state present")
            elif isinstance(checkpoint[key], (int, float)):
                print(f"    - {key}: {checkpoint[key]}")
            else:
                print(f"    - {key}: {type(checkpoint[key]).__name__}")
        
        # Check training metrics if available
        if 'overall_accuracy' in checkpoint:
            print(f"\n  Training metrics:")
            print(f"    Overall accuracy: {checkpoint.get('overall_accuracy', 'N/A'):.2f}%")
            print(f"    Non-zero accuracy: {checkpoint.get('non_zero_accuracy', 'N/A'):.2f}%")
            print(f"    Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"    Best loss: {checkpoint.get('best_loss', 'N/A')}")
    else:
        print("  Checkpoint is raw state_dict (no metadata)")
    
    return checkpoint

def test_actual_arc_task(model, device):
    """Test on an actual ARC task to see specific failure mode"""
    print("\n5. ACTUAL ARC TASK TEST")
    print("-" * 60)
    
    # Load one training task
    train_path = 'arc-prize-2025/arc-agi_training_challenges.json'
    if not os.path.exists(train_path):
        train_path = 'arc-prize-2025/arc-agi_test_challenges.json'
    
    with open(train_path, 'r') as f:
        data = json.load(f)
    
    # Get first task
    task_id = list(data.keys())[0]
    task = data[task_id]
    
    if 'train' in task and task['train']:
        example = task['train'][0]
        input_grid = example['input']
        output_grid = example['output']
        
        print(f"  Task: {task_id}")
        print(f"  Input shape: {len(input_grid)}x{len(input_grid[0])}")
        print(f"  Output shape: {len(output_grid)}x{len(output_grid[0])}")
        
        # Get model prediction
        input_tensor = preprocess_grid(input_grid).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            predictions = output[0].argmax(dim=-1).cpu().numpy()
            pred_grid = predictions.reshape(30, 30)[:len(output_grid), :len(output_grid[0])]
        
        # Compare patterns
        input_tokens = set(np.array(input_grid).flatten())
        output_tokens = set(np.array(output_grid).flatten())
        pred_tokens = set(pred_grid.flatten())
        
        print(f"\n  Token usage:")
        print(f"    Input uses tokens: {sorted(input_tokens)}")
        print(f"    Expected output uses: {sorted(output_tokens)}")
        print(f"    Model predicts: {sorted(pred_tokens)}")
        
        # Check if it's copying
        if len(input_grid) == len(output_grid) and len(input_grid[0]) == len(output_grid[0]):
            input_flat = np.array(input_grid).flatten()
            output_flat = np.array(output_grid).flatten()
            pred_flat = pred_grid.flatten()
            
            copy_score = (pred_flat == input_flat).mean()
            correct_score = (pred_flat == output_flat).mean()
            
            print(f"\n  Behavior analysis:")
            print(f"    Similarity to input (copying): {copy_score:.2%}")
            print(f"    Similarity to expected output: {correct_score:.2%}")
            
            if copy_score > 0.7:
                print("    ⚠️ Model is mostly COPYING the input!")
            elif correct_score < 0.2:
                print("    ⚠️ Model output is very different from expected!")

def main():
    print("=" * 80)
    print("COMPREHENSIVE MODEL DIAGNOSTIC")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Check checkpoint first
    checkpoint_path = 'faithful_model_best.pt'
    checkpoint = check_checkpoint_info(checkpoint_path)
    
    # Load model
    model = FaithfulModel(MODEL_CONFIG).to(device)
    
    if checkpoint and isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("\n✓ Model loaded from checkpoint")
        else:
            model.load_state_dict(checkpoint)
            print("\n✓ Model loaded from raw state dict")
    elif checkpoint:
        model.load_state_dict(checkpoint)
        print("\n✓ Model loaded from checkpoint")
    else:
        print("\n⚠️ Using random initialization!")
    
    # Run diagnostics
    check_model_initialization(model)
    test_output_behavior(model, device)
    test_gradient_flow(model, device)
    test_actual_arc_task(model, device)
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    print("""
Based on the analysis above, check for:
1. Zero/near-zero weights → Model not properly trained
2. Single token predictions → Model collapsed to trivial solution
3. Copying behavior → Model just memorizing inputs
4. No gradient flow → Architecture or training issue
5. Low output diversity → Model not learning patterns
""")

if __name__ == "__main__":
    main()