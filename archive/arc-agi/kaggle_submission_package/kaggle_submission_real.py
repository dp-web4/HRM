#!/usr/bin/env python3
"""
ARC Prize 2025 Kaggle Submission - REAL MODEL VERSION
SAGE-7M (Sentient Agentic Generative Engine) - 6.95M parameters
Team: dp-web4

This version uses the actual trained HRM model architecture, not simplified placeholders.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import math
from typing import List, Dict, Any, Tuple, Optional

# Paths - automatically detect Kaggle vs local environment
import os
if os.path.exists('/kaggle/input'):
    # Kaggle environment
    INPUT_PATH = Path('/kaggle/input/arc-prize-2025')
    OUTPUT_PATH = Path('/kaggle/working')
    MODEL_PATH = Path('/kaggle/input/sage-7m/pytorch/default/1/hrm_arc_best.pt')
else:
    # Local testing environment - use local Kaggle structure
    INPUT_PATH = Path('arc-prize-2025')
    OUTPUT_PATH = Path('working')
    MODEL_PATH = Path('hrm-model/hrm_arc_best.pt')
    
    # Create output directory if it doesn't exist
    OUTPUT_PATH.mkdir(exist_ok=True)

# SAGE-7M configuration - must match the trained model exactly
MODEL_CONFIG = {
    'seq_len': 900,  # 30x30 grid
    'vocab_size': 12,  # 0-9 colors + padding/blank
    'hidden_size': 256,
    'num_heads': 8,
    'num_h_layers': 4,  # Strategic layers
    'num_l_layers': 3,  # Tactical layers
    'dropout': 0.0,  # No dropout for inference
    'max_cycles': 8
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
    """HRM architecture exactly matching the trained checkpoint"""
    
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
        
        # Interaction layers (CRITICAL: H↔L bidirectional communication)
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
            for h_layer in self.h_layers:
                h_state = h_layer(h_state)
            h_state = self.h_norm(h_state)
            
            # L-level processing (tactical) with H guidance
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
            
            # Early stopping
            if cycle > 0 and halt_prob.mean() > 0.9:
                break
        
        # Final output from L-level
        output = self.output(l_state)
        
        return output, halt_probs

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

def solve_task(model: nn.Module, task: Dict[str, Any], device: torch.device) -> List[List[int]]:
    """Solve a single ARC task using the trained model"""
    
    # Get training examples for context (future: use for few-shot prompting)
    train_examples = task.get('train', [])
    
    # Get test input
    test_input = task['test'][0]['input']
    test_h = len(test_input)
    test_w = len(test_input[0]) if test_input else 1
    
    # Preprocess - pad to 30x30 with zeros
    test_tensor = preprocess_grid(test_input).unsqueeze(0).to(device)
    
    # Run model inference
    model.eval()
    with torch.no_grad():
        output, halt_probs = model(test_tensor)  # Model returns tuple!
    
    # Postprocess (output shape is [batch, seq_len, vocab_size])
    solution = postprocess_output(output[0], test_h, test_w)
    
    return solution

def main():
    """Main submission entry point"""
    print("ARC Prize 2025 - SAGE-7M Submission (REAL MODEL)")
    print("Model: SAGE-7M (6.95M parameters)")
    print("Architecture: H↔L Bidirectional Reasoning")
    print("Team: dp-web4")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model with correct architecture
    print("\nLoading SAGE-7M model...")
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(device)
    
    # Count parameters to verify
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Load checkpoint
    if MODEL_PATH.exists():
        print(f"Loading checkpoint from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model weights loaded successfully")
        else:
            print("⚠ Warning: No model_state_dict in checkpoint, using random weights")
            
        # Show checkpoint info
        if 'epoch' in checkpoint:
            print(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'global_step' in checkpoint:
            print(f"  Checkpoint step: {checkpoint['global_step']}")
    else:
        print(f"⚠ Warning: No checkpoint found at {MODEL_PATH}, using random weights")
    
    model.eval()
    
    # Load test tasks
    test_path = INPUT_PATH / 'arc-agi_test_challenges.json'
    if not test_path.exists():
        # Fallback for local testing
        test_path = Path('arc-agi_test_challenges.json')
    
    print(f"\nLoading test tasks from {test_path}")
    with open(test_path, 'r') as f:
        test_tasks = json.load(f)
    
    print(f"Found {len(test_tasks)} tasks")
    
    # Solve each task
    submission = {}
    for i, (task_id, task_data) in enumerate(test_tasks.items()):
        if i % 20 == 0:  # Progress update every 20 tasks
            print(f"Processing task {i+1}/{len(test_tasks)}: {task_id}")
        
        try:
            # Generate solution using the real model
            solution = solve_task(model, task_data, device)
            submission[task_id] = solution
            
        except Exception as e:
            print(f"Error on task {task_id}: {e}")
            # Fallback solution
            test_input = task_data['test'][0]['input']
            h = len(test_input)
            w = len(test_input[0]) if h > 0 else 1
            submission[task_id] = [[0 for _ in range(w)] for _ in range(h)]
    
    # Save submission
    submission_path = OUTPUT_PATH / 'submission.json'
    with open(submission_path, 'w') as f:
        json.dump(submission, f)
    
    print(f"\n✓ Submission saved to {submission_path}")
    print(f"  Total tasks: {len(submission)}")
    
    # Quick validation
    non_zero_tasks = 0
    for task_id, solution in submission.items():
        flat = [val for row in solution for val in row]
        if any(val != 0 for val in flat):
            non_zero_tasks += 1
    
    print(f"  Tasks with non-zero predictions: {non_zero_tasks}/{len(submission)}")
    
    print("\nDone!")

if __name__ == '__main__':
    main()