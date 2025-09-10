#!/usr/bin/env python3
"""
Test the trained SAGE-7M model on public ARC evaluation sets.
Compares model predictions against ground truth solutions.
"""

import json
import torch
import torch.nn as nn
import numpy as np
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

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

def grid_accuracy(pred_grid, true_grid):
    """Calculate pixel-wise accuracy between two grids"""
    if len(pred_grid) != len(true_grid) or len(pred_grid[0]) != len(true_grid[0]):
        return 0.0
    
    pred_flat = np.array(pred_grid).flatten()
    true_flat = np.array(true_grid).flatten()
    
    return (pred_flat == true_flat).mean()

def exact_match(pred_grid, true_grid):
    """Check if two grids are exactly the same"""
    return pred_grid == true_grid

def evaluate_model(model, challenges, solutions, device, max_tasks=None):
    """Evaluate model on a set of challenges with known solutions"""
    
    model.eval()
    
    total_tasks = 0
    exact_matches = 0
    total_accuracy = 0
    non_zero_correct = 0
    
    task_results = []
    
    with torch.no_grad():
        for i, (task_id, task_data) in enumerate(challenges.items()):
            if max_tasks and i >= max_tasks:
                break
            
            if task_id not in solutions:
                continue
            
            solution_data = solutions[task_id]
            
            # Process each test case
            for test_idx, test_case in enumerate(task_data.get('test', [])):
                test_input = test_case['input']
                
                # Get true output
                if test_idx < len(solution_data):
                    true_output = solution_data[test_idx]
                else:
                    continue
                
                h = len(true_output)
                w = len(true_output[0]) if h > 0 else 1
                
                # Get model prediction
                input_tensor = preprocess_grid(test_input, 30).unsqueeze(0).to(device)
                output = model(input_tensor)
                pred_output = postprocess_output(output[0], h, w, 30)
                
                # Calculate metrics
                acc = grid_accuracy(pred_output, true_output)
                is_exact = exact_match(pred_output, true_output)
                
                # Check non-zero pattern
                true_has_nonzero = any(any(cell != 0 for cell in row) for row in true_output)
                pred_has_nonzero = any(any(cell != 0 for cell in row) for row in pred_output)
                
                task_results.append({
                    'task_id': task_id,
                    'test_idx': test_idx,
                    'accuracy': acc,
                    'exact_match': is_exact,
                    'non_zero_match': true_has_nonzero == pred_has_nonzero
                })
                
                total_tasks += 1
                total_accuracy += acc
                if is_exact:
                    exact_matches += 1
                if true_has_nonzero == pred_has_nonzero:
                    non_zero_correct += 1
    
    return {
        'total_tasks': total_tasks,
        'exact_matches': exact_matches,
        'exact_match_rate': exact_matches / total_tasks if total_tasks > 0 else 0,
        'average_accuracy': total_accuracy / total_tasks if total_tasks > 0 else 0,
        'non_zero_accuracy': non_zero_correct / total_tasks if total_tasks > 0 else 0,
        'task_results': task_results
    }

def main():
    print("=" * 60)
    print("Testing SAGE-7M Model on ARC Evaluation Sets")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading trained model...")
    checkpoint = torch.load('faithful_model_best.pt', map_location=device, weights_only=False)
    
    model = FaithfulModel(MODEL_CONFIG).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"Training accuracy: {checkpoint.get('overall_accuracy', 'N/A'):.2f}%")
    
    # Load evaluation data
    print("\nLoading evaluation data...")
    
    # Load challenges and solutions
    eval_challenges_path = Path('arc-prize-2025/arc-agi_evaluation_challenges.json')
    eval_solutions_path = Path('arc-prize-2025/arc-agi_evaluation_solutions.json')
    
    with open(eval_challenges_path, 'r') as f:
        eval_challenges = json.load(f)
    
    with open(eval_solutions_path, 'r') as f:
        eval_solutions = json.load(f)
    
    print(f"Loaded {len(eval_challenges)} evaluation tasks")
    
    # Evaluate on evaluation set
    print("\n" + "=" * 60)
    print("EVALUATING ON ARC-AGI EVALUATION SET")
    print("=" * 60)
    
    eval_results = evaluate_model(model, eval_challenges, eval_solutions, device)
    
    print(f"\nResults on {eval_results['total_tasks']} test cases:")
    print(f"  Exact matches: {eval_results['exact_matches']}/{eval_results['total_tasks']} ({eval_results['exact_match_rate']*100:.1f}%)")
    print(f"  Average pixel accuracy: {eval_results['average_accuracy']*100:.1f}%")
    print(f"  Non-zero pattern accuracy: {eval_results['non_zero_accuracy']*100:.1f}%")
    
    # Show some example results
    print("\nSample results (first 10 tasks):")
    for result in eval_results['task_results'][:10]:
        status = "✓" if result['exact_match'] else "✗"
        print(f"  {status} Task {result['task_id']}: {result['accuracy']*100:.1f}% accuracy")
    
    # Analyze performance by accuracy ranges
    accuracies = [r['accuracy'] for r in eval_results['task_results']]
    perfect = sum(1 for a in accuracies if a == 1.0)
    high = sum(1 for a in accuracies if 0.9 <= a < 1.0)
    medium = sum(1 for a in accuracies if 0.5 <= a < 0.9)
    low = sum(1 for a in accuracies if a < 0.5)
    
    print("\nAccuracy distribution:")
    print(f"  Perfect (100%): {perfect} tasks")
    print(f"  High (90-99%): {high} tasks")
    print(f"  Medium (50-89%): {medium} tasks")
    print(f"  Low (<50%): {low} tasks")
    
    # Test on training set samples (to verify overfitting)
    print("\n" + "=" * 60)
    print("TESTING ON TRAINING SET SAMPLES")
    print("=" * 60)
    
    train_challenges_path = Path('arc-prize-2025/arc-agi_training_challenges.json')
    train_solutions_path = Path('arc-prize-2025/arc-agi_training_solutions.json')
    
    with open(train_challenges_path, 'r') as f:
        train_challenges = json.load(f)
    
    with open(train_solutions_path, 'r') as f:
        train_solutions = json.load(f)
    
    # Test on first 50 training tasks
    train_sample = dict(list(train_challenges.items())[:50])
    train_results = evaluate_model(model, train_sample, train_solutions, device)
    
    print(f"\nResults on {train_results['total_tasks']} training samples:")
    print(f"  Exact matches: {train_results['exact_matches']}/{train_results['total_tasks']} ({train_results['exact_match_rate']*100:.1f}%)")
    print(f"  Average pixel accuracy: {train_results['average_accuracy']*100:.1f}%")
    print(f"  Non-zero pattern accuracy: {train_results['non_zero_accuracy']*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"The model was trained to faithfully reproduce Claude's predictions.")
    print(f"On evaluation set: {eval_results['exact_match_rate']*100:.1f}% exact match rate")
    print(f"On training samples: {train_results['exact_match_rate']*100:.1f}% exact match rate")
    
    if eval_results['non_zero_accuracy'] > 0.95:
        print("\n✓ Model successfully avoids Agent Zero problem!")
    else:
        print("\n⚠ Model may have Agent Zero tendencies")

if __name__ == '__main__':
    main()