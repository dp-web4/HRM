#!/usr/bin/env python3
"""
Verify if the training data (Claude's predictions) matches the actual ARC ground truth.
This will help us understand if the model was trained on incorrect labels.
"""

import json
import numpy as np
from pathlib import Path

def compare_solutions(claude_pred, ground_truth):
    """Compare Claude's prediction with ground truth"""
    if len(claude_pred) != len(ground_truth):
        return False, f"Size mismatch: {len(claude_pred)}x{len(claude_pred[0])} vs {len(ground_truth)}x{len(ground_truth[0])}"
    
    if len(claude_pred[0]) != len(ground_truth[0]):
        return False, f"Size mismatch: {len(claude_pred)}x{len(claude_pred[0])} vs {len(ground_truth)}x{len(ground_truth[0])}"
    
    claude_flat = np.array(claude_pred).flatten()
    truth_flat = np.array(ground_truth).flatten()
    
    if np.array_equal(claude_flat, truth_flat):
        return True, "Perfect match"
    else:
        accuracy = (claude_flat == truth_flat).mean()
        return False, f"Content mismatch, {accuracy:.1%} pixel accuracy"

def main():
    print("=" * 80)
    print("TRAINING DATA VERIFICATION")
    print("Comparing Claude's predictions with actual ARC ground truth")
    print("=" * 80)
    
    # Load Claude's training data
    print("\n1. Loading Claude's training data...")
    with open('claude_reasoning_training_data.json', 'r') as f:
        claude_data = json.load(f)
    print(f"   Loaded {len(claude_data)} tasks from Claude")
    
    # Load actual ARC training data with ground truth
    print("\n2. Loading ARC ground truth...")
    arc_train_path = Path('arc-prize-2025/arc-agi_training_challenges.json')
    arc_solutions_path = Path('arc-prize-2025/arc-agi_training_solutions.json')
    
    if not arc_train_path.exists() or not arc_solutions_path.exists():
        print("   ERROR: ARC training data not found!")
        return
    
    with open(arc_train_path, 'r') as f:
        arc_challenges = json.load(f)
    
    with open(arc_solutions_path, 'r') as f:
        arc_solutions = json.load(f)
    
    print(f"   Loaded {len(arc_challenges)} ARC challenges")
    print(f"   Loaded {len(arc_solutions)} ARC solutions")
    
    # Compare Claude's predictions with ground truth
    print("\n3. Comparing predictions with ground truth...")
    print("-" * 60)
    
    total_examples = 0
    correct_predictions = 0
    size_mismatches = 0
    content_mismatches = 0
    missing_tasks = 0
    
    # Track specific error patterns
    error_details = {
        'size_errors': [],
        'wrong_predictions': [],
        'missing_in_claude': []
    }
    
    # Check first 10 tasks in detail
    detailed_check_count = 10
    checked = 0
    
    for task_id in list(arc_challenges.keys())[:100]:  # Check first 100 tasks
        if task_id not in claude_data:
            missing_tasks += 1
            error_details['missing_in_claude'].append(task_id)
            continue
        
        claude_task = claude_data[task_id]
        arc_task = arc_challenges[task_id]
        
        # Compare test examples
        if 'test' in claude_task and 'test' in arc_task:
            for idx, claude_test in enumerate(claude_task['test']):
                if idx < len(arc_solutions[task_id]):
                    total_examples += 1
                    
                    claude_output = claude_test['output']
                    ground_truth = arc_solutions[task_id][idx]
                    
                    match, reason = compare_solutions(claude_output, ground_truth)
                    
                    if match:
                        correct_predictions += 1
                    else:
                        if "Size mismatch" in reason:
                            size_mismatches += 1
                            if checked < detailed_check_count:
                                error_details['size_errors'].append({
                                    'task_id': task_id,
                                    'reason': reason,
                                    'claude_shape': f"{len(claude_output)}x{len(claude_output[0])}",
                                    'truth_shape': f"{len(ground_truth)}x{len(ground_truth[0])}"
                                })
                        else:
                            content_mismatches += 1
                            if checked < detailed_check_count:
                                error_details['wrong_predictions'].append({
                                    'task_id': task_id,
                                    'reason': reason
                                })
                    
                    # Show details for first few
                    if checked < detailed_check_count:
                        print(f"\nTask {task_id} (test {idx}):")
                        print(f"  Claude output: {len(claude_output)}x{len(claude_output[0])}")
                        print(f"  Ground truth:  {len(ground_truth)}x{len(ground_truth[0])}")
                        print(f"  Match: {match} - {reason}")
                        checked += 1
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal test examples checked: {total_examples}")
    print(f"Correct predictions: {correct_predictions} ({100*correct_predictions/total_examples:.1f}%)")
    print(f"Size mismatches: {size_mismatches} ({100*size_mismatches/total_examples:.1f}%)")
    print(f"Content mismatches: {content_mismatches} ({100*content_mismatches/total_examples:.1f}%)")
    print(f"Missing tasks in Claude data: {missing_tasks}")
    
    if size_mismatches > 0:
        print("\n⚠️ CRITICAL ISSUE: Claude's predictions have wrong output dimensions!")
        print("   The model was trained on incorrect labels.")
        print("\nExample size errors:")
        for error in error_details['size_errors'][:5]:
            print(f"  - Task {error['task_id']}: Claude={error['claude_shape']}, Truth={error['truth_shape']}")
    
    if correct_predictions == 0:
        print("\n🔴 FATAL: No correct predictions in training data!")
        print("   The model learned from completely wrong labels.")
    elif correct_predictions < total_examples * 0.5:
        print("\n⚠️ WARNING: Less than 50% of training labels are correct!")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if correct_predictions == 0:
        print("""
The model scored 0.00 because it was trained on INCORRECT labels!
Claude's predictions don't match the actual ARC ground truth.

The model faithfully learned to reproduce Claude's wrong predictions,
which explains why it gets 99% accuracy on training but 0% on evaluation.

SOLUTION: Need to generate correct training data from actual ARC solutions,
not from Claude's incorrect predictions.
""")
    else:
        accuracy = 100 * correct_predictions / total_examples
        print(f"""
Training data accuracy: {accuracy:.1f}%

The model was trained on partially incorrect data.
This explains the poor performance on actual ARC tasks.
""")

if __name__ == "__main__":
    main()