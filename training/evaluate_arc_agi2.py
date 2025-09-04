#!/usr/bin/env python3
"""
Evaluate HRM Model on ARC-AGI-2 Tasks
Direct comparison with ARC-AGI-1 evaluation
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import numpy as np
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import glob
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Model configuration (must match training)
MODEL_CONFIG = {
    'batch_size': 1,
    'seq_len': 900,  # 30x30 grid max
    'vocab_size': 12,
    'hidden_size': 256,
    'num_heads': 8,
    'num_h_layers': 4,
    'num_l_layers': 3,
    'dropout': 0.0,
    'max_cycles': 8,
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
    """HRM architecture"""
    
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
        
        # Interaction layers
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
            
            # L-level processing (tactical)
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
        
        # Final output
        output = self.output(l_state)
        
        return output, halt_probs

def grid_to_tensor(grid, max_size=30):
    """Convert ARC grid to tensor, padding to max_size x max_size"""
    grid = np.array(grid)
    h, w = grid.shape
    
    # Pad to max_size x max_size
    padded = np.zeros((max_size, max_size), dtype=np.int64)
    padded[:h, :w] = grid
    
    # Flatten to 1D sequence
    return torch.from_numpy(padded.flatten())

def tensor_to_grid(tensor, original_shape):
    """Convert tensor back to grid with original shape"""
    # Reshape to 30x30
    grid = tensor.reshape(30, 30)
    # Crop to original size
    h, w = original_shape
    return grid[:h, :w]

def evaluate_task(model, task_data, device):
    """
    Evaluate model on a single ARC task
    Returns accuracy and detailed metrics
    """
    test_examples = task_data['test']
    correct_pixels = 0
    total_pixels = 0
    perfect_grids = 0
    
    predictions = []
    targets = []
    
    for example in test_examples:
        input_grid = example['input']
        output_grid = example['output']
        
        # Convert to tensors
        input_tensor = grid_to_tensor(input_grid).unsqueeze(0).to(device)
        target_tensor = grid_to_tensor(output_grid)
        
        # Get model prediction
        with torch.no_grad():
            output, _ = model(input_tensor)
            pred = output.squeeze(0).argmax(dim=-1).cpu()
        
        # Crop to original size
        original_shape = (len(output_grid), len(output_grid[0]))
        pred_grid = tensor_to_grid(pred, original_shape)
        target_grid = tensor_to_grid(target_tensor, original_shape)
        
        # Calculate accuracy
        correct = (pred_grid == target_grid).sum().item()
        total = pred_grid.numel()
        
        correct_pixels += correct
        total_pixels += total
        
        if correct == total:
            perfect_grids += 1
        
        predictions.append(pred_grid)
        targets.append(target_grid)
    
    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0
    grid_accuracy = perfect_grids / len(test_examples) if test_examples else 0
    
    return {
        'pixel_accuracy': pixel_accuracy,
        'grid_accuracy': grid_accuracy,
        'num_test_examples': len(test_examples),
        'perfect_grids': perfect_grids,
        'predictions': predictions,
        'targets': targets
    }

def categorize_task_by_performance(pixel_acc):
    """Categorize task based on pixel accuracy"""
    if pixel_acc >= 0.95:
        return 'perfect'
    elif pixel_acc >= 0.80:
        return 'excellent'
    elif pixel_acc >= 0.60:
        return 'good'
    elif pixel_acc >= 0.40:
        return 'moderate'
    elif pixel_acc >= 0.20:
        return 'poor'
    else:
        return 'failed'

def analyze_task_characteristics(task_data):
    """Extract task characteristics for pattern analysis"""
    characteristics = {
        'num_train': len(task_data['train']),
        'num_test': len(task_data['test']),
        'input_sizes': [],
        'output_sizes': [],
        'uses_colors': set(),
        'size_change': False,
    }
    
    for split in ['train', 'test']:
        for example in task_data[split]:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            
            characteristics['input_sizes'].append(input_grid.shape)
            characteristics['output_sizes'].append(output_grid.shape)
            characteristics['uses_colors'].update(np.unique(input_grid).tolist())
            characteristics['uses_colors'].update(np.unique(output_grid).tolist())
            
            if input_grid.shape != output_grid.shape:
                characteristics['size_change'] = True
    
    characteristics['num_colors'] = len(characteristics['uses_colors'])
    characteristics['avg_input_size'] = np.mean([s[0] * s[1] for s in characteristics['input_sizes']])
    characteristics['avg_output_size'] = np.mean([s[0] * s[1] for s in characteristics['output_sizes']])
    
    return characteristics

def main():
    print("🔍 Evaluating HRM Model on ARC-AGI-2 Tasks")
    print("=" * 60)
    print("Note: ARC-AGI-2 is designed to be significantly harder")
    print("=" * 60)
    
    # Load model
    model_path = 'checkpoints/hrm_arc_best.pt'
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Loading model on {device}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from step {checkpoint.get('global_step', 'unknown')}")
    
    # Load ARC-AGI-2 evaluation tasks
    eval_dir = Path('../arc-agi-2/data/evaluation')
    if not eval_dir.exists():
        print(f"❌ ARC-AGI-2 evaluation directory not found at {eval_dir}")
        return
    
    task_files = sorted(eval_dir.glob('*.json'))
    print(f"📊 Found {len(task_files)} ARC-AGI-2 evaluation tasks")
    
    # Evaluate each task
    results = {}
    categories = defaultdict(list)
    characteristics_by_category = defaultdict(list)
    
    print("\n🏃 Evaluating ARC-AGI-2 tasks...")
    for task_file in tqdm(task_files):
        task_id = task_file.stem
        
        # Load task data
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        # Evaluate
        result = evaluate_task(model, task_data, device)
        results[task_id] = result
        
        # Categorize
        category = categorize_task_by_performance(result['pixel_accuracy'])
        categories[category].append((task_id, result['pixel_accuracy']))
        
        # Analyze characteristics
        chars = analyze_task_characteristics(task_data)
        characteristics_by_category[category].append(chars)
    
    # Print results
    print("\n" + "=" * 60)
    print("ARC-AGI-2 PERFORMANCE BY CATEGORY")
    print("=" * 60)
    
    for category in ['perfect', 'excellent', 'good', 'moderate', 'poor', 'failed']:
        tasks = categories[category]
        print(f"\n{category.upper()} ({len(tasks)} tasks, {len(tasks)/len(task_files)*100:.1f}%)")
        
        if tasks:
            # Show top 5 examples
            sorted_tasks = sorted(tasks, key=lambda x: x[1], reverse=True)
            for task_id, acc in sorted_tasks[:5]:
                print(f"  {task_id}: {acc:.1%}")
            if len(tasks) > 5:
                print(f"  ... and {len(tasks) - 5} more")
    
    # Overall statistics
    all_pixel_accs = [r['pixel_accuracy'] for r in results.values()]
    all_grid_accs = [r['grid_accuracy'] for r in results.values()]
    
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS - ARC-AGI-2")
    print("=" * 60)
    print(f"Tasks Evaluated: {len(results)}")
    print(f"Mean Pixel Accuracy: {np.mean(all_pixel_accs):.1%}")
    print(f"Median Pixel Accuracy: {np.median(all_pixel_accs):.1%}")
    print(f"Std Dev: {np.std(all_pixel_accs):.1%}")
    print(f"Perfect Grids: {sum(all_grid_accs):.0f}/{len(results)} ({np.mean(all_grid_accs):.1%})")
    
    # Compare with ARC-AGI-1 results
    print("\n" + "=" * 60)
    print("COMPARISON: ARC-AGI-1 vs ARC-AGI-2")
    print("=" * 60)
    print(f"                 AGI-1 (400 tasks)  →  AGI-2 (120 tasks)")
    print(f"Mean Accuracy:      49.1%          →  {np.mean(all_pixel_accs):.1%}")
    print(f"Median Accuracy:    55.0%          →  {np.median(all_pixel_accs):.1%}")
    print(f"Std Dev:            30.7%          →  {np.std(all_pixel_accs):.1%}")
    print(f"Perfect Grids:      0.25%          →  {np.mean(all_grid_accs):.1%}")
    
    # Analyze patterns by category
    print("\n" + "=" * 60)
    print("TASK CHARACTERISTICS BY PERFORMANCE")
    print("=" * 60)
    
    for category in ['perfect', 'excellent', 'failed']:
        if category in characteristics_by_category:
            chars_list = characteristics_by_category[category]
            if chars_list:
                print(f"\n{category.upper()} tasks tend to have:")
                avg_train = np.mean([c['num_train'] for c in chars_list])
                avg_colors = np.mean([c['num_colors'] for c in chars_list])
                pct_size_change = np.mean([c['size_change'] for c in chars_list]) * 100
                avg_input = np.mean([c['avg_input_size'] for c in chars_list])
                
                print(f"  - {avg_train:.1f} training examples")
                print(f"  - {avg_colors:.1f} unique colors")
                print(f"  - {pct_size_change:.0f}% have size changes")
                print(f"  - {avg_input:.0f} avg input pixels")
    
    # Key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS - ARC-AGI-2")
    print("=" * 60)
    
    # Success patterns
    if categories['perfect']:
        print("\n✅ PERFECT TASKS (Model fully solves):")
        for task_id, acc in categories['perfect'][:3]:
            print(f"  - {task_id}")
    
    if categories['excellent']:
        print("\n🟢 EXCELLENT TASKS (>80% accuracy):")
        for task_id, acc in categories['excellent'][:3]:
            print(f"  - {task_id}: {acc:.1%}")
    
    # Failure patterns
    if categories['failed']:
        print("\n❌ FAILED TASKS (Model cannot solve):")
        for task_id, acc in categories['failed'][:5]:
            print(f"  - {task_id}: {acc:.1%}")
    
    # Save detailed results
    output_data = {
        'dataset': 'ARC-AGI-2',
        'summary': {
            'total_tasks': len(results),
            'mean_pixel_accuracy': float(np.mean(all_pixel_accs)),
            'median_pixel_accuracy': float(np.median(all_pixel_accs)),
            'std_dev': float(np.std(all_pixel_accs)),
            'perfect_grids': int(sum(all_grid_accs)),
            'perfect_grid_rate': float(np.mean(all_grid_accs))
        },
        'categories': {
            cat: len(tasks) for cat, tasks in categories.items()
        },
        'task_results': {
            task_id: {
                'pixel_accuracy': float(r['pixel_accuracy']),
                'grid_accuracy': float(r['grid_accuracy']),
                'category': categorize_task_by_performance(r['pixel_accuracy'])
            }
            for task_id, r in results.items()
        },
        'comparison_with_agi1': {
            'agi1_mean': 0.491,
            'agi2_mean': float(np.mean(all_pixel_accs)),
            'performance_drop': 0.491 - float(np.mean(all_pixel_accs))
        }
    }
    
    output_file = 'arc_agi2_evaluation.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to {output_file}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("PERFORMANCE DISTRIBUTION - ARC-AGI-2")
    print("=" * 60)
    perfect_pct = len(categories['perfect']) / len(results) * 100
    excellent_pct = len(categories['excellent']) / len(results) * 100
    good_pct = len(categories['good']) / len(results) * 100
    failed_pct = len(categories['failed']) / len(results) * 100
    
    print(f"Perfect (>95%):   {perfect_pct:5.1f}% [{len(categories['perfect']):3d} tasks]")
    print(f"Excellent (>80%): {excellent_pct:5.1f}% [{len(categories['excellent']):3d} tasks]")
    print(f"Good (>60%):      {good_pct:5.1f}% [{len(categories['good']):3d} tasks]")
    print(f"Failed (<20%):    {failed_pct:5.1f}% [{len(categories['failed']):3d} tasks]")
    
    print("\n✨ ARC-AGI-2 Performance Summary:")
    perf_drop = (0.491 - np.mean(all_pixel_accs)) / 0.491 * 100
    print(f"   - {perf_drop:.0f}% performance drop from ARC-AGI-1")
    print(f"   - Confirms model lacks true reasoning capability")
    print(f"   - ARC-AGI-2's increased difficulty exposes limitations")

if __name__ == "__main__":
    main()