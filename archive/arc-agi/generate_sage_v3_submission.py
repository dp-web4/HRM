#!/usr/bin/env python3
"""
Generate Kaggle submission using SAGE V3 (human-like reasoning model).
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import math

# Model configuration
MODEL_CONFIG = {
    'seq_len': 900,  # 30x30 grid
    'vocab_size': 12,  # 0-9 colors + padding
    'hidden_size': 256,
    'num_heads': 8,
    'num_h_layers': 4,
    'num_l_layers': 3,
    'dropout': 0.0,  # No dropout for inference
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
    """Model trained on human-like reasoning"""
    
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

def grid_to_tensor(grid, size=30):
    """Convert grid to padded tensor"""
    tensor = torch.zeros(size, size, dtype=torch.long)
    
    if grid:
        h = len(grid)
        w = len(grid[0]) if grid[0] else 0
        
        for i in range(min(h, size)):
            for j in range(min(w, size)):
                if i < len(grid) and j < len(grid[i]):
                    tensor[i, j] = grid[i][j]
    
    return tensor.flatten()

def tensor_to_grid(tensor, orig_h, orig_w):
    """Convert tensor back to grid with original size"""
    # Reshape to 30x30
    grid = tensor.reshape(30, 30)
    
    # Extract the relevant portion
    result = []
    for i in range(min(orig_h, 30)):
        row = []
        for j in range(min(orig_w, 30)):
            row.append(int(grid[i, j].item()))
        result.append(row)
    
    return result

def generate_submission():
    """Generate submission using SAGE V3"""
    print("=" * 60)
    print("SAGE V3 Submission Generator")
    print("Human-like Visual Reasoning Model")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    print("\nLoading SAGE V3 model...")
    checkpoint = torch.load('sage_v3_best.pt', map_location=device)
    
    model = FaithfulModel(MODEL_CONFIG)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch'] + 1}")
    print(f"Training loss: {checkpoint['loss']:.4f}")
    print(f"Training accuracy: {checkpoint.get('accuracy', 0):.1f}%")
    
    # Load test data
    print("\nLoading test challenges...")
    test_path = Path('kaggle_submission_package/arc-prize-2025/arc-agi_test_challenges.json')
    with open(test_path, 'r') as f:
        test_tasks = json.load(f)
    
    print(f"Found {len(test_tasks)} test tasks")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = {}
    
    with torch.no_grad():
        for idx, (task_id, task_data) in enumerate(test_tasks.items()):
            if idx % 20 == 0:
                print(f"  Progress: {idx}/{len(test_tasks)} tasks...")
            
            task_predictions = []
            
            for test_case in task_data.get('test', []):
                input_grid = test_case['input']
                orig_h = len(input_grid)
                orig_w = len(input_grid[0]) if input_grid else 0
                
                # Convert to tensor
                input_tensor = grid_to_tensor(input_grid).unsqueeze(0).to(device)
                
                # Generate prediction
                output_logits = model(input_tensor)
                
                # Get predicted classes
                predicted = output_logits.argmax(dim=-1).squeeze(0)
                
                # Convert back to grid
                pred_grid = tensor_to_grid(predicted, orig_h, orig_w)
                
                # Create second attempt (simple variation)
                # Try color inversion for variety
                colors = set(c for row in pred_grid for c in row)
                if len(colors) == 2:
                    color_list = sorted(list(colors))
                    attempt2 = [[color_list[1] if c == color_list[0] else color_list[0] 
                                for c in row] for row in pred_grid]
                else:
                    # Or try small perturbation
                    attempt2 = pred_grid
                
                task_predictions.append({
                    'attempt_1': pred_grid,
                    'attempt_2': attempt2
                })
            
            predictions[task_id] = task_predictions
    
    print(f"  Progress: {len(test_tasks)}/{len(test_tasks)} tasks... Done!")
    
    # Save submission
    output_path = 'sage_v3_submission.json'
    with open(output_path, 'w') as f:
        json.dump(predictions, f, separators=(',', ':'))
    
    print(f"\n✓ Submission saved to {output_path}")
    
    # Quick validation
    non_zero = 0
    total = 0
    for task_preds in predictions.values():
        for pred in task_preds:
            total += 1
            if any(any(c != 0 for c in row) for row in pred['attempt_1']):
                non_zero += 1
    
    print(f"\nSubmission statistics:")
    print(f"  Total predictions: {total}")
    print(f"  Non-zero predictions: {non_zero} ({non_zero/total*100:.1f}%)")
    print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")
    
    print("\n" + "=" * 60)
    print("SAGE V3 submission ready!")
    print("This model was trained on human-like visual reasoning")
    print("rather than complex algorithmic patterns.")
    print("=" * 60)

if __name__ == '__main__':
    generate_submission()