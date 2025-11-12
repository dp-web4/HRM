#!/usr/bin/env python3
"""
ARC Prize 2025 Kaggle Submission - V3 Context-Aware Model
Using Claude's reasoned solutions distilled into a 4.8M parameter model
Team: dp-web4
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
    MODEL_PATH = Path('/kaggle/input/sage-5m-r/pytorch/default/1/v3_reasoning_model.pt')
else:
    # Local testing environment
    INPUT_PATH = Path('arc-prize-2025')
    OUTPUT_PATH = Path('.')
    MODEL_PATH = Path('v3_reasoning_model.pt')

# Model configuration - must match training
MODEL_CONFIG = {
    'seq_len': 900,
    'vocab_size': 12,
    'hidden_size': 256,
    'num_heads': 8,
    'num_layers': 6,
    'dropout': 0.0,  # No dropout for inference
    'pattern_dims': 16,
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

class V3ReasoningModel(nn.Module):
    """Model that reproduces Claude's reasoned solutions"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.pattern_embedding = nn.Embedding(5, config['pattern_dims'])
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
            pattern_emb = self.pattern_embedding(pattern)
            pattern_emb = pattern_emb.unsqueeze(1).expand(-1, seq_len, -1)
            x = torch.cat([x, pattern_emb], dim=-1)
            x = self.input_projection(x)
        
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

def analyze_input_pattern(grid):
    """Quick pattern analysis to determine pattern type"""
    grid_array = np.array(grid)
    h, w = grid_array.shape
    
    # Check for small grids that might be tiled
    if h <= 5 and w <= 5:
        return 3  # Possible tile pattern
    
    # Check for rectangles (presence of color 3 forming patterns)
    if np.any(grid_array == 3):
        # Look for rectangular patterns
        green_mask = grid_array == 3
        if np.sum(green_mask) > 4:  # At least 4 green cells
            return 1  # fill_rectangles pattern
    
    # Check if mostly sparse (extraction pattern)
    non_zero_ratio = np.count_nonzero(grid_array) / (h * w)
    if non_zero_ratio < 0.3:
        return 2  # extract_pattern
    
    return 0  # unknown

def solve_task(model: nn.Module, task: Dict[str, Any], device: torch.device) -> List[Dict[str, List[List[int]]]]:
    """Solve a single ARC task using V3 model"""
    
    test_cases = task.get('test', [])
    task_attempts = []
    
    for test_case in test_cases:
        test_input = test_case['input']
        test_h = len(test_input)
        test_w = len(test_input[0]) if test_input else 1
        
        # Preprocess input
        test_tensor = preprocess_grid(test_input).unsqueeze(0).to(device)
        
        # Analyze pattern type
        pattern_type = analyze_input_pattern(test_input)
        pattern_tensor = torch.tensor([pattern_type], dtype=torch.long).to(device)
        
        # Run model inference
        model.eval()
        with torch.no_grad():
            output = model(test_tensor, pattern_tensor)
        
        # Generate primary solution
        solution = postprocess_output(output[0], test_h, test_w)
        
        # Generate two attempts with slight variation
        attempt_1 = solution
        
        # For attempt_2, apply slight temperature variation
        with torch.no_grad():
            # Slight temperature variation
            logits = output[0] / 1.02
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
    print("ARC Prize 2025 - V3 Context-Aware Submission")
    print("Claude's Reasoned Solutions (4.8M parameters)")
    print("Team: dp-web4")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\nInitializing V3 model...")
    model = V3ReasoningModel(MODEL_CONFIG).to(device)
    
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
            if 'accuracy' in checkpoint:
                print(f"  Training accuracy: {checkpoint['accuracy']*100:.2f}%")
            if 'non_zero_accuracy' in checkpoint:
                print(f"  Non-zero accuracy: {checkpoint['non_zero_accuracy']*100:.2f}%")
        else:
            # Direct state dict format
            model.load_state_dict(checkpoint)
            print("✓ Model weights loaded (direct format)")
    else:
        print(f"❌ ERROR: No checkpoint found at {MODEL_PATH}")
        print("  The model file is required for predictions!")
        print("  Please ensure the model is uploaded to the correct Kaggle dataset.")
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
    
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
    
    print("\n✨ V3 Ready for Kaggle submission!")
    print("This version uses context-aware reasoning patterns")

if __name__ == '__main__':
    main()