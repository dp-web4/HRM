#!/usr/bin/env python3
"""
Evaluate fine-tuned reasoning models on AGI-1 and AGI-2
"""
import os
import sys
import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Import evaluation functions from existing scripts
sys.path.append('.')
from evaluate_original_arc import (
    evaluate_task,
    categorize_task_by_performance,
    MODEL_CONFIG
)
# Import the enhanced model from fine-tuning script
try:
    from finetune_reasoning_fixed import HierarchicalReasoningModule
except ImportError:
    from finetune_reasoning import HierarchicalReasoningModule

def evaluate_finetuned_model(checkpoint_path, dataset='agi-1'):
    """Evaluate a fine-tuned model on specified dataset"""
    
    print(f"🔍 Evaluating fine-tuned model on {dataset.upper()}")
    print(f"📂 Checkpoint: {checkpoint_path}")
    print("=" * 60)
    
    # Update config for reasoning model
    eval_config = MODEL_CONFIG.copy()
    eval_config['max_cycles'] = 20  # Extended reasoning cycles
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Update config from checkpoint if available
    if 'config' in checkpoint:
        eval_config.update(checkpoint['config'])
    
    model = HierarchicalReasoningModule(eval_config).to(device)
    
    # Load with strict=False to handle architecture differences
    # The fine-tuned model has an enhanced halt_predictor
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"✅ Model loaded (max cycles: {eval_config['max_cycles']})")
    
    # Determine dataset path
    if dataset == 'agi-1':
        eval_dir = Path('../dataset/raw-data/ARC-AGI/data/evaluation')
    else:  # agi-2
        eval_dir = Path('../arc-agi-2/data/evaluation')
    
    if not eval_dir.exists():
        print(f"❌ Dataset not found at {eval_dir}")
        return None
    
    # Load task files
    task_files = sorted(eval_dir.glob('*.json'))
    print(f"📊 Found {len(task_files)} tasks")
    
    # Evaluate
    results = {}
    categories = {}
    
    print("\n🏃 Evaluating...")
    for task_file in tqdm(task_files):
        task_id = task_file.stem
        
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        result = evaluate_task(model, task_data, device)
        results[task_id] = result
        
        category = categorize_task_by_performance(result['pixel_accuracy'])
        if category not in categories:
            categories[category] = []
        categories[category].append((task_id, result['pixel_accuracy']))
    
    # Calculate statistics
    all_pixel_accs = [r['pixel_accuracy'] for r in results.values()]
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Mean Accuracy: {np.mean(all_pixel_accs):.1%}")
    print(f"Median Accuracy: {np.median(all_pixel_accs):.1%}")
    print(f"Std Dev: {np.std(all_pixel_accs):.1%}")
    
    # Show distribution
    print("\nPerformance Distribution:")
    for cat in ['perfect', 'excellent', 'good', 'moderate', 'poor', 'failed']:
        if cat in categories:
            count = len(categories[cat])
            pct = count / len(results) * 100
            print(f"  {cat.capitalize():10} ({count:3d} tasks, {pct:5.1f}%)")
    
    return {
        'mean': np.mean(all_pixel_accs),
        'median': np.median(all_pixel_accs),
        'std': np.std(all_pixel_accs),
        'categories': {k: len(v) for k, v in categories.items()}
    }

def compare_models():
    """Compare original vs fine-tuned models"""
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON: Original vs Fine-tuned")
    print("=" * 60)
    
    # Original model results (from our previous evaluation)
    original_agi1 = {
        'mean': 0.491,
        'median': 0.550,
        'std': 0.307
    }
    
    original_agi2 = {
        'mean': 0.187,
        'median': 0.000,
        'std': 0.282
    }
    
    # Evaluate fine-tuned AGI-1 model
    agi1_checkpoint = 'checkpoints/hrm_reasoning_agi1_final.pt'
    if not Path(agi1_checkpoint).exists():
        # Try latest checkpoint
        agi1_checkpoint = 'checkpoints/hrm_reasoning_agi1_step_6000.pt'
    
    if Path(agi1_checkpoint).exists():
        print("\n📊 Evaluating AGI-1 fine-tuned model...")
        finetuned_agi1 = evaluate_finetuned_model(agi1_checkpoint, 'agi-1')
    else:
        print(f"❌ AGI-1 checkpoint not found")
        finetuned_agi1 = None
    
    # Evaluate fine-tuned AGI-2 model (if exists)
    agi2_checkpoint = 'checkpoints/hrm_reasoning_agi2_final.pt'
    if Path(agi2_checkpoint).exists():
        print("\n📊 Evaluating AGI-2 fine-tuned model...")
        finetuned_agi2 = evaluate_finetuned_model(agi2_checkpoint, 'agi-2')
    else:
        finetuned_agi2 = None
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print("                  Original → Fine-tuned  | Improvement")
    print("-" * 60)
    
    if finetuned_agi1:
        improvement = (finetuned_agi1['mean'] - original_agi1['mean']) / original_agi1['mean'] * 100
        print(f"AGI-1 Mean:       {original_agi1['mean']:.1%} → {finetuned_agi1['mean']:.1%}  | {improvement:+.1f}%")
        print(f"AGI-1 Median:     {original_agi1['median']:.1%} → {finetuned_agi1['median']:.1%}")
    
    if finetuned_agi2:
        improvement = (finetuned_agi2['mean'] - original_agi2['mean']) / original_agi2['mean'] * 100
        print(f"AGI-2 Mean:       {original_agi2['mean']:.1%} → {finetuned_agi2['mean']:.1%}  | {improvement:+.1f}%")
        print(f"AGI-2 Median:     {original_agi2['median']:.1%} → {finetuned_agi2['median']:.1%}")
    
    print("-" * 60)
    print("\nKey Changes:")
    print("- Extended reasoning: 8 → 20 cycles")
    print("- Training data: Augmented → Original tasks")
    print("- Focus: Pattern matching → Multi-step reasoning")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned models')
    parser.add_argument('--checkpoint', type=str, help='Specific checkpoint to evaluate')
    parser.add_argument('--dataset', type=str, default='agi-1', choices=['agi-1', 'agi-2'])
    parser.add_argument('--compare', action='store_true', help='Run full comparison')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    elif args.checkpoint:
        evaluate_finetuned_model(args.checkpoint, args.dataset)
    else:
        # Default: evaluate latest AGI-1 checkpoint
        checkpoint = 'checkpoints/hrm_reasoning_agi1_step_6000.pt'
        if Path(checkpoint).exists():
            evaluate_finetuned_model(checkpoint, 'agi-1')
        else:
            print("No checkpoint found. Run fine-tuning first.")