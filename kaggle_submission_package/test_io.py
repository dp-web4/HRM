#!/usr/bin/env python3
"""
Test script to verify file I/O works without PyTorch dependencies
"""

import json
from pathlib import Path
import os

print("=" * 60)
print("Testing Kaggle submission I/O")
print("=" * 60)

# Paths - automatically detect Kaggle vs local environment
if os.path.exists('/kaggle/input'):
    # Kaggle environment
    INPUT_PATH = Path('/kaggle/input/arc-prize-2025')
    OUTPUT_PATH = Path('/kaggle/working')
else:
    # Local testing environment - use local Kaggle structure
    INPUT_PATH = Path('arc-prize-2025')
    OUTPUT_PATH = Path('working')
    
    # Create output directory if it doesn't exist
    OUTPUT_PATH.mkdir(exist_ok=True)

print(f"\nPaths:")
print(f"  INPUT_PATH: {INPUT_PATH}")
print(f"  OUTPUT_PATH: {OUTPUT_PATH}")
print(f"  INPUT_PATH exists: {INPUT_PATH.exists()}")
print(f"  OUTPUT_PATH exists: {OUTPUT_PATH.exists()}")

# List input files
print(f"\nContents of {INPUT_PATH}:")
if INPUT_PATH.exists():
    for item in INPUT_PATH.iterdir():
        size = item.stat().st_size if item.is_file() else 0
        print(f"  - {item.name} ({size:,} bytes)")
else:
    print("  INPUT_PATH does not exist!")

# Load test tasks
test_path = INPUT_PATH / 'arc-agi_test_challenges.json'
print(f"\nLoading test file from: {test_path}")
print(f"  Test file exists: {test_path.exists()}")

if test_path.exists():
    with open(test_path, 'r') as f:
        test_tasks = json.load(f)
    
    print(f"  ✓ Loaded {len(test_tasks)} test tasks")
    
    # Show first few task IDs
    task_ids = list(test_tasks.keys())[:5]
    print(f"  First 5 task IDs: {task_ids}")
    
    # Check structure of first task
    first_task = test_tasks[task_ids[0]]
    print(f"\n  First task structure:")
    print(f"    - train examples: {len(first_task.get('train', []))}")
    print(f"    - test examples: {len(first_task.get('test', []))}")
    if first_task.get('test'):
        test_input = first_task['test'][0]['input']
        print(f"    - test input shape: {len(test_input)}x{len(test_input[0]) if test_input else 0}")
else:
    print("  ✗ Test file not found!")
    test_tasks = {}

# Create dummy submission
print(f"\nCreating dummy submission...")
submission = {}

for i, task_id in enumerate(list(test_tasks.keys())[:10]):  # Just first 10 for testing
    # Create dummy solution (all zeros, same size as input)
    task = test_tasks[task_id]
    if task.get('test') and task['test'][0].get('input'):
        test_input = task['test'][0]['input']
        height = len(test_input)
        width = len(test_input[0]) if height > 0 else 1
        solution = [[0 for _ in range(width)] for _ in range(height)]
    else:
        solution = [[0]]
    
    submission[task_id] = solution
    print(f"  Task {i+1}: {task_id} -> solution shape {len(solution)}x{len(solution[0])}")

# Save submission
submission_path = OUTPUT_PATH / 'submission.json'
print(f"\nSaving submission to: {submission_path}")
print(f"  Parent directory exists: {submission_path.parent.exists()}")

try:
    with open(submission_path, 'w') as f:
        json.dump(submission, f, indent=2)
    print(f"  ✓ File saved successfully")
    
    # Verify it was written
    if submission_path.exists():
        size = submission_path.stat().st_size
        print(f"  ✓ File exists with size: {size:,} bytes")
        
        # Try to read it back
        with open(submission_path, 'r') as f:
            loaded = json.load(f)
        print(f"  ✓ File can be read back, contains {len(loaded)} tasks")
    else:
        print(f"  ✗ File not found after saving!")
        
except Exception as e:
    print(f"  ✗ Error saving file: {e}")

# List output directory
print(f"\nContents of {OUTPUT_PATH}:")
if OUTPUT_PATH.exists():
    files = list(OUTPUT_PATH.iterdir())
    if files:
        for item in files:
            size = item.stat().st_size if item.is_file() else 0
            print(f"  - {item.name} ({size:,} bytes)")
    else:
        print("  Directory is empty!")
else:
    print("  OUTPUT_PATH does not exist!")

# Also check current directory
print(f"\nContents of current directory:")
for item in Path('.').glob('submission*'):
    size = item.stat().st_size if item.is_file() else 0
    print(f"  - {item.name} ({size:,} bytes)")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)