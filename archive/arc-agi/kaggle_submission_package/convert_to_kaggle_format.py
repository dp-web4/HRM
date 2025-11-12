#!/usr/bin/env python3
"""
Convert Agent Zero submission to Kaggle ARC Prize format.

Agent Zero outputs all zeros, but we need to format it correctly:
- Each task value must be a list
- Each list item must have attempt_1 and attempt_2
- Tasks with multiple test cases need multiple list items
"""

import json
from pathlib import Path

def convert_to_kaggle_format(input_json_path, output_json_path, test_data_path):
    """
    Convert simple format to Kaggle's expected format.
    
    Kaggle expects:
    {
        "task_id": [
            {"attempt_1": [[grid]], "attempt_2": [[grid]]},
            # Additional items if task has multiple test cases
        ]
    }
    """
    
    # Load the current submission
    with open(input_json_path, 'r') as f:
        current_submission = json.load(f)
    
    # Load test data to check how many test cases each task has
    with open(test_data_path, 'r') as f:
        test_tasks = json.load(f)
    
    # Convert to Kaggle format
    kaggle_submission = {}
    
    for task_id, grid in current_submission.items():
        # Check how many test cases this task has
        if task_id in test_tasks:
            num_test_cases = len(test_tasks[task_id].get('test', []))
        else:
            num_test_cases = 1  # Default to 1 if not found
        
        # Create the list of attempts for this task
        task_attempts = []
        for _ in range(num_test_cases):
            # Agent Zero: both attempts are the same (all zeros)
            attempt = {
                "attempt_1": grid,
                "attempt_2": grid  # Same as attempt_1 for Agent Zero
            }
            task_attempts.append(attempt)
        
        kaggle_submission[task_id] = task_attempts
    
    # Save the converted submission
    with open(output_json_path, 'w') as f:
        json.dump(kaggle_submission, f, separators=(',', ':'))  # Compact format
    
    return kaggle_submission

def main():
    # Paths
    input_path = Path('submission.json')
    output_path = Path('submission_kaggle.json')
    
    # Try to find test data
    test_paths = [
        Path('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'),
        Path('arc-prize-2025/arc-agi_test_challenges.json'),
        Path('arc-agi_test_challenges.json')
    ]
    
    test_path = None
    for path in test_paths:
        if path.exists():
            test_path = path
            break
    
    if not test_path:
        print("Warning: Could not find test data file. Using single test case per task.")
        # Create a dummy test structure
        with open(input_path, 'r') as f:
            current = json.load(f)
        test_data = {task_id: {'test': [{}]} for task_id in current.keys()}
        test_path = Path('dummy_test.json')
        with open(test_path, 'w') as f:
            json.dump(test_data, f)
    
    print(f"Converting {input_path} to Kaggle format...")
    print(f"Using test data from: {test_path}")
    
    # Convert
    result = convert_to_kaggle_format(input_path, output_path, test_path)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Output saved to: {output_path}")
    print(f"  Total tasks: {len(result)}")
    
    # Show sample
    sample_id = list(result.keys())[0]
    print(f"\nSample output for task {sample_id}:")
    print(f"  Number of test cases: {len(result[sample_id])}")
    print(f"  First test case has attempts: {list(result[sample_id][0].keys())}")
    print(f"  Grid shape: {len(result[sample_id][0]['attempt_1'])}x{len(result[sample_id][0]['attempt_1'][0]) if result[sample_id][0]['attempt_1'] else 0}")

if __name__ == '__main__':
    main()