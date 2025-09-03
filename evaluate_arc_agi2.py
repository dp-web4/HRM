#!/usr/bin/env python3
"""
Evaluate HRM model on ARC-AGI-2 test set
"""

import json
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "training"))

def load_arc_agi2_task(json_path):
    """Load a single ARC-AGI-2 task from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def preprocess_arc_task(task_data, max_size=30):
    """Convert ARC-AGI-2 task to HRM input format"""
    # For evaluation, we only have test inputs (no outputs)
    # We'll use the train examples to understand the pattern
    
    train_examples = task_data.get('train', [])
    test_examples = task_data.get('test', [])
    
    processed_data = {
        'train_inputs': [],
        'train_outputs': [],
        'test_inputs': []
    }
    
    # Process training examples
    for example in train_examples:
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Pad to max_size
        h, w = input_grid.shape
        if h <= max_size and w <= max_size:
            padded_input = np.zeros((max_size, max_size), dtype=np.int32)
            padded_input[:h, :w] = input_grid
            
            oh, ow = output_grid.shape
            padded_output = np.zeros((max_size, max_size), dtype=np.int32)
            padded_output[:oh, :ow] = output_grid
            
            processed_data['train_inputs'].append(padded_input)
            processed_data['train_outputs'].append(padded_output)
    
    # Process test examples
    for example in test_examples:
        input_grid = np.array(example['input'])
        h, w = input_grid.shape
        if h <= max_size and w <= max_size:
            padded_input = np.zeros((max_size, max_size), dtype=np.int32)
            padded_input[:h, :w] = input_grid
            processed_data['test_inputs'].append(padded_input)
            
            # Store actual output for scoring if available
            if 'output' in example:
                output_grid = np.array(example['output'])
                oh, ow = output_grid.shape
                padded_output = np.zeros((max_size, max_size), dtype=np.int32)
                padded_output[:oh, :ow] = output_grid
                if 'test_outputs' not in processed_data:
                    processed_data['test_outputs'] = []
                processed_data['test_outputs'].append(padded_output)
    
    return processed_data

def evaluate_model_on_task(model, task_data, device='cuda'):
    """Evaluate HRM model on a single ARC task"""
    # This is a placeholder - actual implementation would need the HRM model architecture
    # and proper inference code from the training directory
    
    # For now, return a dummy prediction
    test_inputs = task_data['test_inputs']
    predictions = []
    
    for test_input in test_inputs:
        # Convert to tensor
        input_tensor = torch.tensor(test_input, dtype=torch.long).unsqueeze(0).to(device)
        
        # TODO: Actual model inference here
        # For now, just return the input as a placeholder
        pred = test_input.copy()
        predictions.append(pred)
    
    return predictions

def calculate_accuracy(predictions, ground_truths):
    """Calculate pixel-wise accuracy"""
    if len(predictions) != len(ground_truths):
        return 0.0
    
    correct = 0
    total = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_array = np.array(pred)
        gt_array = np.array(gt)
        
        # Only compare non-zero regions
        mask = gt_array != 0
        correct += np.sum(pred_array[mask] == gt_array[mask])
        total += np.sum(mask)
    
    return correct / total if total > 0 else 0.0

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
    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Maximum number of tasks to evaluate')
    parser.add_argument('--output', type=str, default='arc_agi2_results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model checkpoint not found at {model_path}")
        print("Please ensure the model file exists or provide correct path")
        return
    
    # Load model
    print(f"Loading model from {model_path}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # TODO: Load actual HRM model here
    # checkpoint = torch.load(model_path, map_location=device)
    # model = load_hrm_model(checkpoint)
    # model.to(device)
    # model.eval()
    
    # For now, we'll use a placeholder
    model = None
    
    # Get task files
    data_dir = Path(args.data) / args.split
    task_files = sorted(data_dir.glob('*.json'))
    
    if args.max_tasks:
        task_files = task_files[:args.max_tasks]
    
    print(f"Found {len(task_files)} tasks in {args.split} split")
    
    # Evaluate on each task
    results = []
    total_accuracy = 0
    tasks_with_ground_truth = 0
    
    for task_file in tqdm(task_files, desc="Evaluating tasks"):
        task_id = task_file.stem
        
        # Load task
        task_data = load_arc_agi2_task(task_file)
        
        # Preprocess
        processed_data = preprocess_arc_task(task_data)
        
        # Skip if no test examples
        if not processed_data['test_inputs']:
            continue
        
        # Evaluate
        predictions = evaluate_model_on_task(model, processed_data, device)
        
        # Calculate accuracy if ground truth is available
        accuracy = None
        if 'test_outputs' in processed_data:
            accuracy = calculate_accuracy(predictions, processed_data['test_outputs'])
            total_accuracy += accuracy
            tasks_with_ground_truth += 1
        
        # Store result
        result = {
            'task_id': task_id,
            'num_train': len(processed_data['train_inputs']),
            'num_test': len(processed_data['test_inputs']),
            'accuracy': accuracy
        }
        results.append(result)
    
    # Calculate overall statistics
    overall_stats = {
        'total_tasks': len(results),
        'tasks_with_ground_truth': tasks_with_ground_truth,
        'average_accuracy': total_accuracy / tasks_with_ground_truth if tasks_with_ground_truth > 0 else None,
        'split': args.split,
        'model': str(model_path)
    }
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total tasks evaluated: {overall_stats['total_tasks']}")
    print(f"Tasks with ground truth: {overall_stats['tasks_with_ground_truth']}")
    if overall_stats['average_accuracy'] is not None:
        print(f"Average accuracy: {overall_stats['average_accuracy']:.2%}")
    
    # Save results
    output_data = {
        'overall_stats': overall_stats,
        'per_task_results': results
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()