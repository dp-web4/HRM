#!/usr/bin/env python3
"""
Analyze what the model is actually learning by examining its outputs more closely.
"""

import json
import torch
import torch.nn as nn
import numpy as np
import math
from pathlib import Path
# import matplotlib.pyplot as plt  # Not needed for this analysis
from collections import Counter

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

def analyze_output_distribution(output_logits):
    """Analyze the distribution of predicted tokens"""
    # Get probabilities
    probs = torch.softmax(output_logits, dim=-1)
    
    # Get most likely predictions
    predictions = output_logits.argmax(dim=-1)
    
    # Count frequency of each predicted token
    pred_counts = Counter(predictions.cpu().numpy().flatten())
    
    # Get average confidence for top predictions
    max_probs = probs.max(dim=-1)[0]
    avg_confidence = max_probs.mean().item()
    
    return {
        'token_distribution': dict(pred_counts),
        'avg_confidence': avg_confidence,
        'min_confidence': max_probs.min().item(),
        'max_confidence': max_probs.max().item()
    }

def main():
    print("=" * 80)
    print("Analyzing Model Behavior")
    print("=" * 80)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaithfulModel(MODEL_CONFIG).to(device)
    
    model_path = 'faithful_model_best.pt'
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Test on different input patterns
    print("\n1. Testing on various input patterns:")
    print("-" * 40)
    
    test_patterns = [
        # All zeros
        ("All zeros", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        # All ones
        ("All ones", [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        # Diagonal
        ("Diagonal", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        # Checkerboard
        ("Checkerboard", [[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
        # Random
        ("Random", [[2, 5, 3], [7, 1, 9], [4, 8, 6]])
    ]
    
    for name, pattern in test_patterns:
        print(f"\n{name}:")
        print("Input:", pattern)
        
        input_tensor = preprocess_grid(pattern).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_logits = model(input_tensor)
            
            # Analyze the output distribution
            analysis = analyze_output_distribution(output_logits[0])
            
            # Get actual prediction for 3x3 grid
            predictions = output_logits[0].argmax(dim=-1).cpu().numpy()
            pred_grid = predictions.reshape(30, 30)[:3, :3]
            
            print("Output:", pred_grid.tolist())
            print(f"Token distribution: {analysis['token_distribution']}")
            print(f"Average confidence: {analysis['avg_confidence']:.3f}")
            print(f"Confidence range: [{analysis['min_confidence']:.3f}, {analysis['max_confidence']:.3f}]")
    
    # Analyze model's general behavior
    print("\n" + "=" * 80)
    print("2. Model Statistics:")
    print("-" * 40)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check if model weights look reasonable
    print("\n3. Weight Statistics:")
    print("-" * 40)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            mean = param.data.mean().item()
            std = param.data.std().item()
            min_val = param.data.min().item()
            max_val = param.data.max().item()
            
            print(f"{name:40s} mean={mean:7.4f} std={std:7.4f} [{min_val:7.4f}, {max_val:7.4f}]")
            
            # Check for potential issues
            if abs(mean) > 10:
                print(f"  ⚠️  WARNING: Large mean value!")
            if std < 0.001:
                print(f"  ⚠️  WARNING: Very small std - possible dead neurons!")
            if std > 10:
                print(f"  ⚠️  WARNING: Very large std - possible instability!")
    
    # Test what happens with actual ARC training data
    print("\n" + "=" * 80)
    print("4. Testing on actual ARC patterns:")
    print("-" * 40)
    
    # Load one real ARC task
    with open('arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
        data = json.load(f)
    
    task_id = list(data.keys())[0]
    task = data[task_id]
    
    if 'train' in task and task['train']:
        example = task['train'][0]
        input_grid = example['input']
        output_grid = example['output']
        
        print(f"\nTask: {task_id}")
        print(f"Input shape: {len(input_grid)}x{len(input_grid[0])}")
        print(f"Output shape: {len(output_grid)}x{len(output_grid[0])}")
        
        input_tensor = preprocess_grid(input_grid).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output_logits = model(input_tensor)
            analysis = analyze_output_distribution(output_logits[0])
            
            predictions = output_logits[0].argmax(dim=-1).cpu().numpy()
            pred_grid = predictions.reshape(30, 30)[:len(output_grid), :len(output_grid[0])]
            
            # Compare input/output token distributions
            input_tokens = Counter(np.array(input_grid).flatten())
            output_tokens = Counter(np.array(output_grid).flatten())
            pred_tokens = Counter(pred_grid.flatten())
            
            print(f"\nToken distributions:")
            print(f"  Input tokens: {dict(input_tokens)}")
            print(f"  Expected output tokens: {dict(output_tokens)}")
            print(f"  Predicted output tokens: {dict(pred_tokens)}")
            
            # Check if model is just copying input
            input_flat = np.array(input_grid).flatten()
            pred_flat = pred_grid.flatten()[:len(input_flat)]
            
            copy_similarity = (input_flat == pred_flat[:len(input_flat)]).mean()
            print(f"\nSimilarity to input (copying): {copy_similarity:.2%}")
            
            if copy_similarity > 0.8:
                print("  ⚠️  Model appears to be mostly copying the input!")

if __name__ == "__main__":
    main()