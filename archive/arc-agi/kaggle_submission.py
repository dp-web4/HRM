#!/usr/bin/env python3
"""
ARC Prize 2025 Kaggle Submission
SAGE-7M (Situation-Aware Governance Engine) - 6.95M parameters
An evolution of Sapient's HRM with 75% parameter reduction
Team: dp-web4
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
    MODEL_PATH = INPUT_PATH / 'hrm-model' / 'hrm_arc_best.pt'
else:
    # Local testing environment
    INPUT_PATH = Path('.')
    OUTPUT_PATH = Path('.')
    MODEL_PATH = Path('hrm-model/hrm_arc_best.pt')

# SAGE-7M configuration (6.95M parameters, 75% smaller than original HRM's 27M)
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
    """HRM architecture matching the trained checkpoint"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.pos_encoding = PositionalEncoding(config['hidden_size'])
        
        # H-level (strategic) layers - process high-level patterns
        self.h_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_h_layers'])
        ])
        
        # L-level (tactical) layers - refine solutions
        self.l_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_l_layers'])
        ])
        
        # Cross-level connections (CRITICAL for matching checkpoint)
        self.h_to_l = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.l_to_h = nn.Linear(config['hidden_size'], config['hidden_size'])
        
        # Layer normalization (CRITICAL for matching checkpoint)
        self.h_norm = nn.LayerNorm(config['hidden_size'])
        self.l_norm = nn.LayerNorm(config['hidden_size'])
        
        # Adaptive computation (note: checkpoint has 512 input size, we'll handle this)
        self.halt_predictor = nn.Linear(config.get('halt_input_size', 512), 1)
        
        # Output layer (matches checkpoint naming)
        self.output = nn.Linear(config['hidden_size'], config['vocab_size'])
        
    def forward(self, x, max_cycles=None):
        if max_cycles is None:
            max_cycles = self.config['max_cycles']
        
        batch_size, seq_len = x.shape
        device = x.device
        
        # Embedding
        x_emb = self.token_embedding(x)
        x_emb = self.pos_encoding(x_emb)
        
        # Adaptive computation through H-L cycles
        halted = torch.zeros(batch_size, device=device).bool()
        output = torch.zeros_like(x_emb)
        halt_probs = []
        
        # Initialize H and L states
        h_state = x_emb.clone()
        l_state = x_emb.clone()
        
        for cycle in range(max_cycles):
            # H-level (strategic reasoning)
            for h_layer in self.h_layers:
                h_state = h_layer(h_state)
            h_state = self.h_norm(h_state)
            
            # L-level (tactical refinement) with H-level guidance
            l_state = l_state + self.h_to_l(h_state)
            for l_layer in self.l_layers:
                l_state = l_layer(l_state)
            l_state = self.l_norm(l_state)
            
            # L to H feedback
            h_state = h_state + self.l_to_h(l_state)
            
            # Adaptive halting - concatenate H and L features for halt decision
            combined = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)  # [batch, 512]
            halt_logit = self.halt_predictor(combined)
            halt_prob = torch.sigmoid(halt_logit).squeeze(-1)
            halt_probs.append(halt_prob)
            
            # Update output for non-halted samples
            not_halted = ~halted
            output[not_halted] = output[not_halted] + l_state[not_halted]
            
            # Update halting status
            should_halt = halt_prob > 0.5
            halted = halted | should_halt
            
            # Early termination if all samples halted
            if halted.all():
                break
        
        # Generate output tokens
        logits = self.output(output)
        return logits

def preprocess_grid(grid: List[List[int]]) -> torch.Tensor:
    """Convert ARC grid to model input tensor"""
    flat = []
    for row in grid:
        flat.extend(row)
    # Pad to 30x30 if needed
    while len(flat) < 900:
        flat.append(11)  # Padding token
    return torch.tensor(flat[:900], dtype=torch.long)

def postprocess_output(output: torch.Tensor, height: int, width: int) -> List[List[int]]:
    """Convert model output to ARC grid format"""
    # Get predicted tokens
    pred = output.argmax(dim=-1).cpu().numpy()
    
    # Reshape to grid
    grid = []
    idx = 0
    for i in range(height):
        row = []
        for j in range(width):
            if idx < len(pred):
                val = int(pred[idx])
                # Ensure valid color (0-9)
                val = min(max(val, 0), 9)
                row.append(val)
                idx += 1
            else:
                row.append(0)
        grid.append(row)
    
    return grid

def solve_task(model: nn.Module, task: Dict[str, Any], device: torch.device) -> List[List[int]]:
    """Solve a single ARC task"""
    # Get last training example as context
    train_examples = task['train']
    if not train_examples:
        # Fallback to zeros if no training examples
        return [[0]]
    
    last_example = train_examples[-1]
    input_grid = last_example['input']
    output_grid = last_example['output']
    
    # Get test input
    test_input = task['test'][0]['input']
    test_h = len(test_input)
    test_w = len(test_input[0]) if test_input else 1
    
    # Preprocess
    test_tensor = preprocess_grid(test_input).unsqueeze(0).to(device)
    
    # Run model inference
    model.eval()
    with torch.no_grad():
        output = model(test_tensor)
    
    # Postprocess
    solution = postprocess_output(output[0], test_h, test_w)
    
    return solution

def main():
    """Main submission entry point"""
    print("ARC Prize 2025 - SAGE-7M Submission")
    print("Model: SAGE-7M (6.95M parameters)")
    print("Evolution of Sapient's HRM with 75% parameter reduction")
    print("Team: dp-web4")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading SAGE-7M model...")
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(device)
    
    # Load checkpoint if available
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded from checkpoint")
    else:
        print("Warning: No checkpoint found, using random weights")
    
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
    for task_id, task_data in test_tasks.items():
        print(f"Solving task {task_id}...")
        
        try:
            # Generate solution
            solution = solve_task(model, task_data, device)
            
            # Format output
            submission[task_id] = solution
            
        except Exception as e:
            print(f"Error on task {task_id}: {e}")
            # Fallback solution
            submission[task_id] = [[0]]
    
    # Save submission
    submission_path = OUTPUT_PATH / 'submission.json'
    with open(submission_path, 'w') as f:
        json.dump(submission, f)
    
    print(f"\nSubmission saved to {submission_path}")
    print("Done!")

if __name__ == '__main__':
    main()