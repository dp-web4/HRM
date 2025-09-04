#!/usr/bin/env python3
"""
Test SAGE-7M on ARC-AGI-2 public evaluation set
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math
from typing import List, Dict, Any, Tuple, Optional
import sys
import os

# Add kaggle_submission_package to path to import the model
sys.path.insert(0, 'kaggle_submission_package')
from kaggle_submission import SAGE7M, MODEL_CONFIG, preprocess_grid, postprocess_output

def evaluate_prediction(pred: List[List[int]], target: List[List[int]]) -> bool:
    """Check if prediction matches target exactly"""
    if len(pred) != len(target):
        return False
    for pred_row, target_row in zip(pred, target):
        if len(pred_row) != len(target_row):
            return False
        if pred_row != target_row:
            return False
    return True

def solve_task_with_examples(model: nn.Module, task_data: Dict, device: torch.device) -> List[List[int]]:
    """Solve a task using all training examples as context"""
    # For now, using the same simple approach as the submission
    # In future, could concatenate all examples or use more sophisticated context
    
    if not task_data['train']:
        return [[0]]
    
    # Use last training example as context
    last_example = task_data['train'][-1]
    
    # Get test input (first test case)
    test_input = task_data['test'][0]['input']
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
    """Test SAGE-7M on ARC-AGI-2 evaluation set"""
    
    print("=" * 60)
    print("SAGE-7M ARC-AGI-2 Evaluation")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading SAGE-7M model...")
    model = SAGE7M(MODEL_CONFIG).to(device)
    
    # Load checkpoint
    checkpoint_path = Path('kaggle_submission_package/hrm-model/hrm_arc_best.pt')
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Model loaded from checkpoint")
    else:
        print("⚠ Warning: No checkpoint found, using random weights")
        return
    
    model.eval()
    
    # Load evaluation tasks
    eval_dir = Path('arc-agi-2/data/evaluation')
    task_files = sorted(eval_dir.glob('*.json'))
    print(f"\nFound {len(task_files)} evaluation tasks")
    
    # Evaluate
    correct = 0
    total = 0
    results = {}
    
    print("\nEvaluating...")
    for task_file in tqdm(task_files):
        task_id = task_file.stem
        
        # Load task
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        try:
            # Generate prediction
            prediction = solve_task_with_examples(model, task_data, device)
            
            # Get ground truth (first test output)
            ground_truth = task_data['test'][0]['output']
            
            # Evaluate
            is_correct = evaluate_prediction(prediction, ground_truth)
            
            results[task_id] = {
                'correct': is_correct,
                'prediction': prediction,
                'ground_truth': ground_truth
            }
            
            if is_correct:
                correct += 1
            total += 1
            
        except Exception as e:
            print(f"\nError on task {task_id}: {e}")
            results[task_id] = {
                'correct': False,
                'error': str(e)
            }
            total += 1
    
    # Calculate accuracy
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Tasks evaluated: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 60)
    
    # Save detailed results
    output_file = 'sage7m_arc_agi2_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model': 'SAGE-7M',
            'parameters': '6.95M',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'results': results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    
    # Show some examples of correct and incorrect predictions
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    correct_examples = [k for k, v in results.items() if v.get('correct', False)][:3]
    incorrect_examples = [k for k, v in results.items() if not v.get('correct', False)][:3]
    
    if correct_examples:
        print("\n✓ Correct predictions:")
        for task_id in correct_examples:
            print(f"  - {task_id}")
    
    if incorrect_examples:
        print("\n✗ Incorrect predictions:")
        for task_id in incorrect_examples:
            print(f"  - {task_id}")
    
    print("\n" + "=" * 60)
    print(f"Final Score: {accuracy:.2f}% on ARC-AGI-2 public evaluation set")
    print("=" * 60)

if __name__ == '__main__':
    main()