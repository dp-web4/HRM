#!/usr/bin/env python3
"""
Evaluate HRM model on ARC-AGI-2 using the CORRECT architecture from train_arc_full_nova.py
This matches the actual checkpoint that achieved 71.36% accuracy
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import math
from typing import List, Dict, Any, Tuple

# Model configuration (from checkpoint)
MODEL_CONFIG = {
    'batch_size': 20,
    'seq_len': 900,  # 30x30 grid
    'vocab_size': 12,
    'hidden_size': 256,
    'num_heads': 8,
    'num_h_layers': 4,
    'num_l_layers': 3,
    'dropout': 0.1,
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
    """HRM architecture exactly matching train_arc_full_nova.py"""
    
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
        
        # Interaction layers (CRITICAL: these are what were missing!)
        self.h_to_l = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.l_to_h = nn.Linear(config['hidden_size'], config['hidden_size'])
        
        # Halting mechanism (CRITICAL: takes 2*hidden_size!)
        self.halt_predictor = nn.Linear(config['hidden_size'] * 2, 1)
        
        # Output layer (CRITICAL: named 'output' not 'output_layer')
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
            h_prev = h_state.clone()
            for h_layer in self.h_layers:
                h_state = h_layer(h_state)
            h_state = self.h_norm(h_state)
            
            # L-level processing (tactical)
            l_prev = l_state.clone()
            # Incorporate H-level guidance
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
            
            # Early stopping based on halt probability
            if cycle > 0 and halt_prob.mean() > 0.9:
                break
        
        # Final output from L-level (execution)
        output = self.output(l_state)
        
        return output, halt_probs

def load_arc_agi2_task(json_path: Path) -> Dict:
    """Load a single ARC-AGI-2 task from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def grid_to_sequence(grid: List[List[int]], max_size: int = 30) -> np.ndarray:
    """Convert 2D grid to padded sequence for HRM input"""
    grid_array = np.array(grid, dtype=np.int32)
    h, w = grid_array.shape
    
    # Pad to max_size x max_size
    padded = np.zeros((max_size, max_size), dtype=np.int32)
    padded[:min(h, max_size), :min(w, max_size)] = grid_array[:min(h, max_size), :min(w, max_size)]
    
    # Flatten to sequence
    return padded.flatten()

def sequence_to_grid(sequence: np.ndarray, original_shape: Tuple[int, int], max_size: int = 30) -> List[List[int]]:
    """Convert sequence back to 2D grid"""
    # Reshape to max_size x max_size
    grid = sequence.reshape(max_size, max_size)
    
    # Extract original dimensions
    h, w = original_shape
    grid = grid[:h, :w]
    
    return grid.tolist()

def evaluate_task(model: nn.Module, task_data: Dict, device: torch.device) -> Dict:
    """Evaluate model on a single ARC task"""
    
    train_examples = task_data.get('train', [])
    test_examples = task_data.get('test', [])
    
    results = {
        'predictions': [],
        'has_ground_truth': False,
        'accuracy': None
    }
    
    # For each test example
    for test_idx, test_example in enumerate(test_examples):
        test_input = test_example['input']
        test_shape = (len(test_input), len(test_input[0]))
        
        # Convert to sequence
        input_seq = grid_to_sequence(test_input)
        input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
        
        # Run model inference
        with torch.no_grad():
            output, halt_probs = model(input_tensor)  # Note: model returns output AND halt_probs
            pred_seq = output.argmax(dim=-1).squeeze().cpu().numpy()
        
        # Convert back to grid
        pred_grid = sequence_to_grid(pred_seq, test_shape)
        results['predictions'].append(pred_grid)
        
        # Check accuracy if ground truth available
        if 'output' in test_example:
            results['has_ground_truth'] = True
            ground_truth = test_example['output']
            
            # Calculate pixel accuracy
            pred_array = np.array(pred_grid)
            gt_array = np.array(ground_truth)
            
            # Resize if dimensions don't match
            if pred_array.shape != gt_array.shape:
                # Take minimum dimensions
                min_h = min(pred_array.shape[0], gt_array.shape[0])
                min_w = min(pred_array.shape[1], gt_array.shape[1])
                pred_array = pred_array[:min_h, :min_w]
                gt_array = gt_array[:min_h, :min_w]
            
            accuracy = np.mean(pred_array == gt_array)
            
            if results['accuracy'] is None:
                results['accuracy'] = []
            results['accuracy'].append(accuracy)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate HRM on ARC-AGI-2')
    parser.add_argument('--model', type=str, default='validation_package/hrm_arc_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='arc-agi-2/data',
                        help='Path to ARC-AGI-2 data directory')
    parser.add_argument('--split', type=str, default='evaluation',
                        choices=['training', 'evaluation'],
                        help='Which split to evaluate on')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--max-tasks', type=int, default=10,
                        help='Maximum number of tasks to evaluate (default: 10 for quick test)')
    parser.add_argument('--output', type=str, default='arc_agi2_hrm_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    # Create model with CORRECT architecture
    model = HierarchicalReasoningModule(MODEL_CONFIG)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  - Training accuracy on ARC-AGI-1: {checkpoint['metrics']['val_accuracy']:.2%}")
    print(f"  - Model trained for {checkpoint.get('global_step', 'unknown')} steps")
    
    # Get task files
    data_dir = Path(args.data) / args.split
    task_files = sorted(data_dir.glob('*.json'))
    
    if args.max_tasks:
        task_files = task_files[:args.max_tasks]
    
    print(f"\nEvaluating on {len(task_files)} tasks from ARC-AGI-2 {args.split} split")
    print("="*60)
    
    # Evaluate tasks
    all_results = []
    total_accuracy = []
    
    for task_file in tqdm(task_files, desc="Evaluating"):
        task_id = task_file.stem
        
        # Load task
        task_data = load_arc_agi2_task(task_file)
        
        # Evaluate
        results = evaluate_task(model, task_data, device)
        
        # Store results
        task_result = {
            'task_id': task_id,
            'num_test': len(results['predictions']),
            'has_ground_truth': results['has_ground_truth']
        }
        
        if results['accuracy'] is not None:
            mean_acc = np.mean(results['accuracy'])
            task_result['accuracy'] = float(mean_acc)
            total_accuracy.append(mean_acc)
        
        all_results.append(task_result)
    
    # Calculate overall statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: HRM with H↔L bidirectional communication")
    print(f"Parameters: 6.95M")
    print(f"Original ARC-AGI-1 accuracy: 71.36%")
    print("-"*60)
    print(f"ARC-AGI-2 Results:")
    print(f"  Total tasks evaluated: {len(all_results)}")
    
    if total_accuracy:
        mean_accuracy = np.mean(total_accuracy)
        print(f"  Tasks with ground truth: {len(total_accuracy)}")
        print(f"  Average accuracy: {mean_accuracy:.2%}")
        print(f"  Min accuracy: {min(total_accuracy):.2%}")
        print(f"  Max accuracy: {max(total_accuracy):.2%}")
        print(f"  Std deviation: {np.std(total_accuracy):.2%}")
    else:
        print("  No ground truth available for accuracy calculation")
    
    # Save results
    output_data = {
        'model': args.model,
        'model_info': {
            'parameters': '6.95M',
            'architecture': 'HRM with H↔L bidirectional',
            'original_arc_accuracy': 0.7136,
            'training_steps': checkpoint.get('global_step', 7000)
        },
        'dataset': 'ARC-AGI-2',
        'split': args.split,
        'num_tasks': len(all_results),
        'average_accuracy': float(np.mean(total_accuracy)) if total_accuracy else None,
        'per_task_results': all_results
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Comparison with ARC Prize target
    if total_accuracy:
        print("\n" + "="*60)
        print("ARC PRIZE ANALYSIS")
        print("="*60)
        print(f"Current accuracy: {mean_accuracy:.2%}")
        print(f"ARC Prize target: 85.00%")
        print(f"Gap to target: {85.0 - mean_accuracy*100:.2%}")
        print(f"Performance ratio: {mean_accuracy/0.85:.2%} of target")

if __name__ == "__main__":
    main()