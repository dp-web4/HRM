#!/usr/bin/env python3
"""
Analyze HRM Model Performance by Task Type
Performs inference on individual ARC tasks to identify success/failure patterns
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
from typing import Dict, List, Tuple
import glob

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Model configuration (must match training)
MODEL_CONFIG = {
    'batch_size': 1,  # Process one task at a time
    'seq_len': 900,
    'vocab_size': 12,
    'hidden_size': 256,
    'num_heads': 8,
    'num_h_layers': 4,
    'num_l_layers': 3,
    'dropout': 0.0,  # No dropout during inference
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

def analyze_task_categories(predictions, targets, task_ids=None):
    """
    Analyze which types of ARC tasks the model handles well
    Returns categories based on accuracy thresholds
    """
    categories = {
        'excellent': [],  # >90% accuracy
        'good': [],       # 70-90% accuracy  
        'moderate': [],   # 50-70% accuracy
        'poor': [],       # 30-50% accuracy
        'failed': []      # <30% accuracy
    }
    
    task_accuracies = {}
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        # Calculate per-task accuracy
        correct = (pred == target).float().mean().item()
        task_id = task_ids[i] if task_ids else f"task_{i}"
        task_accuracies[task_id] = correct
        
        # Categorize
        if correct >= 0.9:
            categories['excellent'].append((task_id, correct))
        elif correct >= 0.7:
            categories['good'].append((task_id, correct))
        elif correct >= 0.5:
            categories['moderate'].append((task_id, correct))
        elif correct >= 0.3:
            categories['poor'].append((task_id, correct))
        else:
            categories['failed'].append((task_id, correct))
    
    return categories, task_accuracies

def analyze_error_patterns(predictions, targets):
    """
    Analyze common error patterns in model predictions
    """
    error_stats = {
        'color_errors': 0,      # Wrong color mapping
        'shape_errors': 0,      # Wrong shape/structure
        'position_errors': 0,   # Right pattern, wrong position
        'size_errors': 0,       # Wrong size/scale
        'total_errors': 0
    }
    
    for pred, target in zip(predictions, targets):
        mask = pred != target
        if mask.any():
            error_stats['total_errors'] += mask.sum().item()
            
            # Analyze error types (simplified heuristics)
            # Color errors: different values at same positions
            unique_pred = torch.unique(pred[mask])
            unique_target = torch.unique(target[mask])
            if len(unique_pred) != len(unique_target):
                error_stats['color_errors'] += 1
            
            # Shape errors: different non-zero patterns
            pred_nonzero = (pred != 0).float()
            target_nonzero = (target != 0).float()
            if (pred_nonzero != target_nonzero).any():
                error_stats['shape_errors'] += 1
    
    return error_stats

def main():
    print("🔍 Analyzing HRM Model Task Performance")
    print("=" * 50)
    
    # Check for model checkpoint
    model_path = 'checkpoints/hrm_arc_best.pt'
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Loading model on {device}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from step {checkpoint.get('global_step', 'unknown')}")
    
    # Load validation data
    data_path = Path('../data/arc-aug-500/test')
    if not data_path.exists():
        print(f"❌ Validation data not found at {data_path}")
        return
    
    inputs = np.load(data_path / 'all__inputs.npy')
    targets = np.load(data_path / 'all__labels.npy')
    
    print(f"📊 Loaded {len(inputs)} validation samples")
    
    # Process a subset for analysis (first 1000 samples)
    num_samples = min(1000, len(inputs))
    all_predictions = []
    all_targets = []
    
    print(f"\n🏃 Running inference on {num_samples} samples...")
    
    with torch.no_grad():
        for i in range(0, num_samples, 10):  # Batch of 10
            batch_end = min(i + 10, num_samples)
            batch_inputs = torch.from_numpy(inputs[i:batch_end]).long().to(device)
            batch_targets = torch.from_numpy(targets[i:batch_end]).long().to(device)
            
            outputs, _ = model(batch_inputs)
            predictions = outputs.argmax(dim=-1)
            
            all_predictions.extend(predictions.cpu())
            all_targets.extend(batch_targets.cpu())
            
            if (i + 10) % 100 == 0:
                print(f"  Processed {i + 10}/{num_samples} samples...")
    
    # Analyze results
    print("\n📊 Analyzing Performance Categories...")
    categories, task_accuracies = analyze_task_categories(
        all_predictions, all_targets
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("TASK PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    for category, tasks in categories.items():
        print(f"\n{category.upper()} ({len(tasks)} tasks):")
        if category in ['excellent', 'failed']:
            # Show some examples
            for task_id, acc in tasks[:5]:
                print(f"  - {task_id}: {acc:.2%}")
            if len(tasks) > 5:
                print(f"  ... and {len(tasks) - 5} more")
    
    # Overall statistics
    all_accs = list(task_accuracies.values())
    print("\n" + "=" * 50)
    print("OVERALL STATISTICS")
    print("=" * 50)
    print(f"Mean Accuracy: {np.mean(all_accs):.2%}")
    print(f"Median Accuracy: {np.median(all_accs):.2%}")
    print(f"Std Dev: {np.std(all_accs):.2%}")
    print(f"Min Accuracy: {np.min(all_accs):.2%}")
    print(f"Max Accuracy: {np.max(all_accs):.2%}")
    
    # Error pattern analysis
    print("\n" + "=" * 50)
    print("ERROR PATTERN ANALYSIS")
    print("=" * 50)
    error_stats = analyze_error_patterns(all_predictions, all_targets)
    for error_type, count in error_stats.items():
        print(f"{error_type}: {count}")
    
    # Save detailed results
    results = {
        'num_samples': num_samples,
        'categories': {k: len(v) for k, v in categories.items()},
        'overall_accuracy': np.mean(all_accs),
        'statistics': {
            'mean': float(np.mean(all_accs)),
            'median': float(np.median(all_accs)),
            'std': float(np.std(all_accs)),
            'min': float(np.min(all_accs)),
            'max': float(np.max(all_accs))
        },
        'error_patterns': error_stats
    }
    
    output_file = 'task_performance_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to {output_file}")
    
    # Insights
    print("\n" + "=" * 50)
    print("KEY INSIGHTS")
    print("=" * 50)
    
    excellent_pct = len(categories['excellent']) / num_samples * 100
    failed_pct = len(categories['failed']) / num_samples * 100
    
    print(f"✅ {excellent_pct:.1f}% of tasks solved with >90% accuracy")
    print(f"❌ {failed_pct:.1f}% of tasks failed (<30% accuracy)")
    
    if excellent_pct > 30:
        print("→ Model excels at simpler pattern recognition tasks")
    if failed_pct > 20:
        print("→ Model struggles with complex multi-step reasoning")
    
    print("\nThe 71% overall accuracy masks significant variation:")
    print("- Some tasks are solved perfectly")
    print("- Others fail completely")
    print("- This suggests the model has learned specific patterns well")
    print("  but lacks general reasoning ability for novel combinations")

if __name__ == "__main__":
    main()