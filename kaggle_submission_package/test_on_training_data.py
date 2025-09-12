#!/usr/bin/env python3
"""
Test our model on the public training data where we have ground truth outputs.
This will help us understand why we're scoring 0.00.
"""

import json
import torch
import torch.nn as nn
import numpy as np
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple
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

def postprocess_output(output, height, width, max_size=30):
    """Convert model output to ARC grid format"""
    pred = output.argmax(dim=-1).cpu().numpy()
    grid_full = pred.reshape(max_size, max_size)
    grid = grid_full[:height, :width]
    grid = np.clip(grid, 0, 9)
    return grid.tolist()

def visualize_grid(grid, title=""):
    """Simple ASCII visualization of a grid"""
    print(f"\n{title}")
    print("-" * (len(grid[0]) * 2 + 1))
    for row in grid:
        print("|" + "".join([str(val) if val > 0 else " " for val in row]) + "|")
    print("-" * (len(grid[0]) * 2 + 1))

def compare_grids(pred, truth):
    """Compare two grids and return accuracy metrics"""
    if len(pred) != len(truth) or (len(pred) > 0 and len(pred[0]) != len(truth[0])):
        return {
            'exact_match': False,
            'pixel_accuracy': 0.0,
            'size_mismatch': True,
            'pred_size': (len(pred), len(pred[0]) if pred else 0),
            'true_size': (len(truth), len(truth[0]) if truth else 0)
        }
    
    pred_flat = np.array(pred).flatten()
    true_flat = np.array(truth).flatten()
    
    exact = (pred_flat == true_flat).all()
    accuracy = (pred_flat == true_flat).mean()
    
    # Find differences
    diff_positions = []
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            if pred[i][j] != truth[i][j]:
                diff_positions.append((i, j, pred[i][j], truth[i][j]))
    
    return {
        'exact_match': exact,
        'pixel_accuracy': float(accuracy),
        'size_mismatch': False,
        'num_differences': len(diff_positions),
        'diff_positions': diff_positions[:5]  # Show first 5 differences
    }

def main():
    print("=" * 80)
    print("Testing ARC Model on Training Data")
    print("=" * 80)
    
    # Check if we're in the kaggle_submission_package directory
    if not os.path.exists('arc-prize-2025'):
        print("ERROR: arc-prize-2025 directory not found!")
        print("Current directory:", os.getcwd())
        return
    
    # Load model
    print("\n1. Loading model...")
    model_path = 'faithful_model_best.pt'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Available model files:", [f for f in os.listdir('.') if f.endswith('.pt')])
        # Try alternative path
        model_path = 'SAGE-7M-V2.pt'
        if not os.path.exists(model_path):
            print(f"Alternative model {model_path} also not found!")
            return
    
    print(f"Loading model from: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = FaithfulModel(MODEL_CONFIG).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully!")
    
    # Load training data
    print("\n2. Loading training data...")
    train_dir = Path('arc-prize-2025/arc-agi_training_challenges.json')
    if not train_dir.exists():
        # Try alternative structure
        train_files = list(Path('arc-prize-2025').glob('*.json'))
        print(f"Found JSON files: {[f.name for f in train_files]}")
        train_dir = Path('arc-prize-2025/arc-agi_test_challenges.json')  # Use test for now
        if not train_dir.exists():
            print("Training data not found!")
            return
    
    with open(train_dir, 'r') as f:
        training_data = json.load(f)
    
    print(f"Loaded {len(training_data)} training tasks")
    
    # Test on a subset of tasks
    num_test = min(5, len(training_data))  # Test on first 5 tasks
    task_ids = list(training_data.keys())[:num_test]
    
    print(f"\n3. Testing on {num_test} tasks...")
    print("=" * 80)
    
    total_exact = 0
    total_tests = 0
    all_accuracies = []
    
    for task_idx, task_id in enumerate(task_ids):
        print(f"\nTask {task_idx + 1}/{num_test}: {task_id}")
        print("-" * 40)
        
        task = training_data[task_id]
        
        # Get training examples
        train_examples = task.get('train', [])
        test_examples = task.get('test', [])
        
        print(f"  Training examples: {len(train_examples)}")
        print(f"  Test examples: {len(test_examples)}")
        
        # For now, let's just try to predict the first test example if it has an output
        # Otherwise, try the last training example
        if test_examples and 'output' in test_examples[0]:
            test_input = test_examples[0]['input']
            test_output = test_examples[0]['output']
            example_type = "test"
        elif train_examples and 'output' in train_examples[-1]:
            test_input = train_examples[-1]['input']
            test_output = train_examples[-1]['output']
            example_type = "train"
        else:
            print("  No output available for comparison!")
            continue
        
        print(f"  Using {example_type} example for testing")
        print(f"  Input size: {len(test_input)}x{len(test_input[0]) if test_input else 0}")
        print(f"  Output size: {len(test_output)}x{len(test_output[0]) if test_output else 0}")
        
        # Preprocess input
        input_tensor = preprocess_grid(test_input).unsqueeze(0).to(device)
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            pred_grid = postprocess_output(
                output[0], 
                len(test_output), 
                len(test_output[0]) if test_output else 0
            )
        
        # Compare with ground truth
        comparison = compare_grids(pred_grid, test_output)
        
        print(f"  Results:")
        print(f"    Exact match: {comparison['exact_match']}")
        print(f"    Pixel accuracy: {comparison['pixel_accuracy']:.2%}")
        
        if comparison['size_mismatch']:
            print(f"    SIZE MISMATCH!")
            print(f"    Predicted: {comparison['pred_size']}")
            print(f"    Expected: {comparison['true_size']}")
        else:
            print(f"    Differences: {comparison['num_differences']} pixels")
            if comparison['num_differences'] > 0 and comparison['num_differences'] <= 10:
                print(f"    First differences (row, col, pred, true):")
                for diff in comparison['diff_positions']:
                    print(f"      {diff}")
        
        # Show a sample of the grids if they're small enough
        if len(test_output) <= 10 and len(test_output[0]) <= 10:
            visualize_grid(test_input, "Input:")
            visualize_grid(test_output, "Expected Output:")
            visualize_grid(pred_grid, "Predicted Output:")
        
        total_tests += 1
        if comparison['exact_match']:
            total_exact += 1
        all_accuracies.append(comparison['pixel_accuracy'])
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Tasks tested: {total_tests}")
    print(f"Exact matches: {total_exact}/{total_tests} ({total_exact/total_tests*100:.1f}%)")
    print(f"Average pixel accuracy: {np.mean(all_accuracies):.2%}")
    print(f"Min pixel accuracy: {np.min(all_accuracies):.2%}")
    print(f"Max pixel accuracy: {np.max(all_accuracies):.2%}")
    
    if total_exact == 0:
        print("\n⚠️  WARNING: No exact matches! This explains the 0.00 score.")
        print("The model is not learning the patterns correctly.")

if __name__ == "__main__":
    main()