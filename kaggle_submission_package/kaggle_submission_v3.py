#!/usr/bin/env python3
"""
ARC Prize 2025 Kaggle Submission - SAGE-7M V3
Human-like Visual Reasoning Model
Team: dp-web4

This uses the model trained on human-like visual reasoning patterns.
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import math
from typing import List, Dict, Any

# Paths - automatically detect Kaggle vs local environment
import os
if os.path.exists('/kaggle/input'):
    # Kaggle environment
    INPUT_PATH = Path('/kaggle/input/arc-prize-2025')
    OUTPUT_PATH = Path('/kaggle/working')
    MODEL_PATH = Path('/kaggle/input/sage-7m-v3/pytorch/default/1/SAGE-V3-human.pt')
else:
    # Local testing environment
    INPUT_PATH = Path('arc-prize-2025')
    OUTPUT_PATH = Path('.')
    MODEL_PATH = Path('SAGE-V3-human.pt')
    
    # Create output directory if it doesn't exist
    OUTPUT_PATH.mkdir(exist_ok=True)

# Model configuration - must match the trained model exactly
MODEL_CONFIG = {
    'seq_len': 900,  # 30x30 grid
    'vocab_size': 12,  # 0-9 colors + padding
    'hidden_size': 256,
    'num_heads': 8,
    'num_h_layers': 4,  # Strategic layers
    'num_l_layers': 3,  # Tactical layers
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
    """Model trained to faithfully reproduce Claude's reasoning"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.pos_encoding = PositionalEncoding(config['hidden_size'])
        self.dropout = nn.Dropout(config['dropout'])
        
        # Transformer layers (combined H+L layers)
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

def preprocess_grid(grid: List[List[int]], max_size: int = 30) -> torch.Tensor:
    """Convert ARC grid to model input tensor"""
    grid_array = np.array(grid, dtype=np.int32)
    h, w = grid_array.shape
    
    # Pad to max_size x max_size with zeros
    padded = np.zeros((max_size, max_size), dtype=np.int32)
    padded[:min(h, max_size), :min(w, max_size)] = grid_array[:min(h, max_size), :min(w, max_size)]
    
    # Flatten to sequence
    return torch.tensor(padded.flatten(), dtype=torch.long)

def postprocess_output(output: torch.Tensor, height: int, width: int, max_size: int = 30) -> List[List[int]]:
    """Convert model output to ARC grid format"""
    # Get predicted tokens
    pred = output.argmax(dim=-1).cpu().numpy()
    
    # Reshape from flattened sequence to 30x30 grid
    grid_full = pred.reshape(max_size, max_size)
    
    # Extract the actual dimensions we need
    grid = grid_full[:height, :width]
    
    # Ensure valid colors (0-9)
    grid = np.clip(grid, 0, 9)
    
    return grid.tolist()

def solve_task(model: nn.Module, task: Dict[str, Any], device: torch.device) -> List[Dict[str, List[List[int]]]]:
    """Solve a single ARC task using the trained model"""
    
    # Get test cases (can be multiple per task)
    test_cases = task.get('test', [])
    
    # Process each test case
    task_attempts = []
    for test_case in test_cases:
        test_input = test_case['input']
        test_h = len(test_input)
        test_w = len(test_input[0]) if test_input else 1
        
        # Preprocess input
        test_tensor = preprocess_grid(test_input).unsqueeze(0).to(device)
        
        # Run model inference
        model.eval()
        with torch.no_grad():
            output = model(test_tensor)
        
        # Postprocess to get prediction
        solution = postprocess_output(output[0], test_h, test_w)
        
        # Generate two attempts with slight variation
        attempt_1 = solution
        
        # For attempt_2, apply slight temperature variation
        with torch.no_grad():
            # Apply temperature scaling for variation
            logits = output[0] / 1.05  # Slight temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from distribution
            pred_varied = torch.multinomial(probs.view(-1, MODEL_CONFIG['vocab_size']), 1).view(-1)
            grid_varied = pred_varied.cpu().numpy().reshape(30, 30)
            attempt_2 = grid_varied[:test_h, :test_w].tolist()
            
            # Ensure valid colors
            attempt_2 = np.clip(attempt_2, 0, 9).tolist()
        
        # Create attempt dictionary in Kaggle format
        attempt = {
            "attempt_1": attempt_1,
            "attempt_2": attempt_2
        }
        task_attempts.append(attempt)
    
    return task_attempts

def main():
    """Main submission entry point"""
    print("=" * 60)
    print("ARC Prize 2025 - SAGE-7M V2 Submission")
    print("Faithful Claude Distillation Model")
    print("Team: dp-web4")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\nCreating SAGE-7M model architecture...")
    model = FaithfulModel(MODEL_CONFIG).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Load checkpoint
    if MODEL_PATH.exists():
        print(f"\nLoading checkpoint from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model weights loaded successfully")
            
            # Show training metrics if available
            if 'epoch' in checkpoint:
                print(f"  Training epoch: {checkpoint['epoch']}")
            if 'overall_accuracy' in checkpoint:
                print(f"  Training accuracy: {checkpoint['overall_accuracy']:.2f}%")
            if 'non_zero_accuracy' in checkpoint:
                print(f"  Non-zero accuracy: {checkpoint['non_zero_accuracy']:.2f}%")
        else:
            # Direct state dict format
            model.load_state_dict(checkpoint)
            print("✓ Model weights loaded (direct format)")
    else:
        print(f"⚠ Warning: No checkpoint found at {MODEL_PATH}")
        print("  Using random initialization (not recommended)")
    
    model.eval()
    
    # Load test tasks
    test_path = INPUT_PATH / 'arc-agi_test_challenges.json'
    if not test_path.exists():
        # Fallback for local testing
        test_path = Path('arc-agi_test_challenges.json')
        if not test_path.exists():
            test_path = Path('arc-prize-2025/arc-agi_test_challenges.json')
    
    print(f"\nLoading test tasks from {test_path}")
    with open(test_path, 'r') as f:
        test_tasks = json.load(f)
    
    print(f"Found {len(test_tasks)} tasks to solve")
    
    # Solve each task
    submission = {}
    non_zero_count = 0
    
    print("\nGenerating predictions...")
    for i, (task_id, task_data) in enumerate(test_tasks.items()):
        if i % 20 == 0:  # Progress update
            print(f"  Processing task {i+1}/{len(test_tasks)}...")
        
        try:
            # Generate solution using the model
            solution = solve_task(model, task_data, device)
            submission[task_id] = solution
            
            # Check if solution has non-zero values
            for attempt in solution:
                grid = attempt.get('attempt_1', [])
                if any(any(val != 0 for val in row) for row in grid):
                    non_zero_count += 1
                    break
                    
        except Exception as e:
            print(f"  Error on task {task_id}: {e}")
            # Create fallback solution
            test_cases = task_data.get('test', [])
            fallback_attempts = []
            for test_case in test_cases:
                test_input = test_case['input']
                h = len(test_input)
                w = len(test_input[0]) if h > 0 else 1
                # Use input as baseline (identity mapping)
                fallback_attempts.append({
                    "attempt_1": test_input,
                    "attempt_2": test_input
                })
            submission[task_id] = fallback_attempts
    
    # Save submission
    submission_path = OUTPUT_PATH / 'submission.json'
    with open(submission_path, 'w') as f:
        json.dump(submission, f, separators=(',', ':'))
    
    print("\n" + "=" * 60)
    print("SUBMISSION COMPLETE")
    print("=" * 60)
    print(f"✓ Submission saved to {submission_path}")
    print(f"  Total tasks: {len(submission)}")
    print(f"  Tasks with non-zero predictions: {non_zero_count}/{len(submission)}")
    
    # Validate submission format
    sample_id = list(submission.keys())[0]
    print(f"\nFormat validation (task {sample_id}):")
    print(f"  Test cases in task: {len(submission[sample_id])}")
    print(f"  Keys per test case: {list(submission[sample_id][0].keys())}")
    first_grid = submission[sample_id][0]['attempt_1']
    print(f"  Grid dimensions: {len(first_grid)}x{len(first_grid[0]) if first_grid else 0}")
    
    print("\nReady for Kaggle submission!")

if __name__ == '__main__':
    main()