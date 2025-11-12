#!/usr/bin/env python3
"""
Test how faithfully the trained model reproduces Claude's predictions.
"""

import json
import torch
import torch.nn as nn
import numpy as np
import math
from pathlib import Path

# Model configuration (must match training)
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

def postprocess_output(output, height, width, max_size=30):
    """Convert model output to ARC grid format"""
    # Get predicted tokens
    pred = output.argmax(dim=-1).cpu().numpy()
    
    # Reshape from flattened sequence to 30x30 grid
    grid_full = pred.reshape(max_size, max_size)
    
    # Extract the actual dimensions we need
    grid = grid_full[:height, :width].tolist()
    
    return grid

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

def grid_similarity(grid1, grid2):
    """Calculate similarity between two grids"""
    g1 = np.array(grid1)
    g2 = np.array(grid2)
    
    # Ensure same shape
    if g1.shape != g2.shape:
        return 0.0
    
    # Calculate pixel-wise accuracy
    correct = (g1 == g2).sum()
    total = g1.size
    
    return correct / total

def main():
    print("=" * 60)
    print("Testing Faithful Reproduction of Claude's Predictions")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading trained model...")
    checkpoint = torch.load('faithful_model_best.pt', map_location=device)
    
    model = FaithfulModel(MODEL_CONFIG).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"Training accuracy: {checkpoint.get('overall_accuracy', 'N/A'):.2f}%")
    print(f"Non-zero accuracy: {checkpoint.get('non_zero_accuracy', 'N/A'):.2f}%")
    
    # Load training data with inputs and Claude's outputs
    print("\nLoading training data with Claude's predictions...")
    with open('claude_reasoning_training_data.json', 'r') as f:
        training_data = json.load(f)
    
    # Test reproduction accuracy
    print("\nTesting reproduction accuracy...")
    
    total_similarity = 0
    perfect_matches = 0
    high_accuracy = 0  # >95% similarity
    non_zero_accuracy = 0
    num_tests = 0
    
    with torch.no_grad():
        for task_id, task_data in training_data.items():
            for test_example in task_data.get('test', []):
                # Get the actual input and Claude's output
                test_input = test_example['input']
                claude_output = test_example['output']
                
                # Get dimensions from Claude's output
                h = len(claude_output)
                w = len(claude_output[0]) if h > 0 else 1
                
                # Generate model prediction from the actual input
                input_tensor = preprocess_grid(test_input, 30).unsqueeze(0).to(device)
                output = model(input_tensor)
                model_pred = postprocess_output(output[0], h, w, 30)
                
                # Calculate similarity to Claude's output
                similarity = grid_similarity(model_pred, claude_output)
                
                total_similarity += similarity
                if similarity == 1.0:
                    perfect_matches += 1
                if similarity > 0.95:
                    high_accuracy += 1
                
                # Check non-zero accuracy
                claude_has_nonzero = any(any(cell != 0 for cell in row) for row in claude_output)
                model_has_nonzero = any(any(cell != 0 for cell in row) for row in model_pred)
                if claude_has_nonzero == model_has_nonzero:
                    non_zero_accuracy += 1
                
                num_tests += 1
                
                if num_tests <= 5:
                    print(f"\n  Task {task_id}:")
                    print(f"    Similarity: {similarity:.2%}")
                    print(f"    Non-zero match: {claude_has_nonzero == model_has_nonzero}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("REPRODUCTION ACCURACY SUMMARY")
    print("=" * 60)
    print(f"Total test cases: {num_tests}")
    print(f"Average similarity: {100 * total_similarity / num_tests:.2f}%")
    print(f"Perfect matches: {perfect_matches}/{num_tests} ({100 * perfect_matches / num_tests:.1f}%)")
    print(f"High accuracy (>95%): {high_accuracy}/{num_tests} ({100 * high_accuracy / num_tests:.1f}%)")
    print(f"Non-zero pattern match: {non_zero_accuracy}/{num_tests} ({100 * non_zero_accuracy / num_tests:.1f}%)")
    
    if total_similarity / num_tests > 0.90:
        print("\n✓ Model achieves EXCELLENT faithful reproduction!")
    elif total_similarity / num_tests > 0.80:
        print("\n✓ Model achieves good faithful reproduction.")
    else:
        print("\n⚠ Model needs more training for faithful reproduction.")

if __name__ == '__main__':
    main()