#!/usr/bin/env python3
"""
ARC Prize 2025 Kaggle Submission - DEBUG VERSION
SAGE-7M (Situation-Aware Governance Engine) - 6.95M parameters
This version has extensive logging to debug output issues
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import math
import os
import sys
from typing import List, Dict, Any, Tuple, Optional

print("=" * 60)
print("SAGE-7M ARC Prize Submission - DEBUG VERSION")
print("=" * 60)

# Debug: Show environment
print(f"\nPython version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")

# Paths - automatically detect Kaggle vs local environment
if os.path.exists('/kaggle/input'):
    print("\n✓ Detected Kaggle environment")
    # Kaggle environment
    INPUT_PATH = Path('/kaggle/input/arc-prize-2025')
    OUTPUT_PATH = Path('/kaggle/working')
    MODEL_PATH = Path('/kaggle/input/sage-7m/pytorch/default/1/hrm_arc_best.pt')
    
    # Alternative paths to try
    ALT_MODEL_PATHS = [
        Path('/kaggle/input/sage-7m-2/hrm_arc_best.pt'),
        Path('/kaggle/input/sage-7m-2/hrm-model/hrm_arc_best.pt'),
    ]
else:
    print("\n✓ Detected local environment")
    # Local testing environment
    INPUT_PATH = Path('.')
    OUTPUT_PATH = Path('.')
    MODEL_PATH = Path('hrm-model/hrm_arc_best.pt')
    ALT_MODEL_PATHS = []

print(f"INPUT_PATH: {INPUT_PATH}")
print(f"OUTPUT_PATH: {OUTPUT_PATH}")
print(f"MODEL_PATH: {MODEL_PATH}")

# Check paths exist
print(f"\nINPUT_PATH exists: {INPUT_PATH.exists()}")
print(f"OUTPUT_PATH exists: {OUTPUT_PATH.exists()}")
print(f"MODEL_PATH exists: {MODEL_PATH.exists()}")

# If model not found, try alternatives
if not MODEL_PATH.exists() and ALT_MODEL_PATHS:
    print(f"\n⚠ Model not found at {MODEL_PATH}, trying alternatives...")
    for alt_path in ALT_MODEL_PATHS:
        print(f"  Trying {alt_path}: {alt_path.exists()}")
        if alt_path.exists():
            MODEL_PATH = alt_path
            print(f"  ✓ Found model at {MODEL_PATH}")
            break

# List input directory contents
if INPUT_PATH.exists():
    print(f"\nContents of {INPUT_PATH}:")
    for item in INPUT_PATH.iterdir():
        print(f"  - {item.name}")

# List /kaggle/input if on Kaggle
if os.path.exists('/kaggle/input'):
    print(f"\nContents of /kaggle/input:")
    for item in Path('/kaggle/input').iterdir():
        print(f"  - {item.name}")
        # Also list subdirectories
        if item.is_dir():
            for subitem in item.iterdir():
                print(f"    - {subitem.name}")

# SAGE-7M configuration
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

# Create minimal model class (simplified for debugging)
class HRMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Minimal architecture just for testing
        self.embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.output = nn.Linear(config['hidden_size'], config['vocab_size'])
    
    def forward(self, x):
        # Minimal forward pass
        x = self.embedding(x)
        x = self.output(x.mean(dim=1))  # Simple pooling
        return x

def solve_task(model, task_data, device):
    """Solve a single ARC task - simplified for debugging"""
    print(f"  Processing task with {len(task_data['train'])} train examples")
    
    # For debugging, just return a simple pattern
    if task_data['test']:
        test_input = task_data['test'][0]['input']
        height = len(test_input)
        width = len(test_input[0]) if height > 0 else 1
        # Return same size grid with all zeros
        return [[0 for _ in range(width)] for _ in range(height)]
    
    return [[0]]

def main():
    print("\n" + "=" * 60)
    print("Starting main execution")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print(f"\nAttempting to load model from {MODEL_PATH}")
    model = HRMModel(MODEL_CONFIG).to(device)
    
    if MODEL_PATH.exists():
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            print(f"✓ Checkpoint loaded, keys: {list(checkpoint.keys())[:5]}...")
            # For debugging, don't load weights, just verify file is readable
        except Exception as e:
            print(f"⚠ Error loading checkpoint: {e}")
    else:
        print(f"⚠ Model file not found at {MODEL_PATH}")
    
    model.eval()
    print("✓ Model in eval mode")
    
    # Load test tasks
    test_path = INPUT_PATH / 'arc-agi_test_challenges.json'
    alt_test_paths = [
        Path('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'),
        INPUT_PATH / 'test_challenges.json',
        Path('arc-agi_test_challenges.json')
    ]
    
    print(f"\nLooking for test file at {test_path}")
    print(f"Test file exists: {test_path.exists()}")
    
    if not test_path.exists():
        print("Trying alternative paths...")
        for alt_path in alt_test_paths:
            print(f"  Trying {alt_path}: {alt_path.exists()}")
            if alt_path.exists():
                test_path = alt_path
                print(f"  ✓ Found test file at {test_path}")
                break
    
    if not test_path.exists():
        # Create dummy data for testing
        print("\n⚠ No test file found, creating dummy data")
        test_tasks = {
            "task_001": {
                "train": [{"input": [[0,1]], "output": [[1,0]]}],
                "test": [{"input": [[0,1]]}]
            },
            "task_002": {
                "train": [{"input": [[1,0]], "output": [[0,1]]}],
                "test": [{"input": [[1,0]]}]
            }
        }
    else:
        print(f"\n✓ Loading test tasks from {test_path}")
        with open(test_path, 'r') as f:
            test_tasks = json.load(f)
    
    print(f"✓ Loaded {len(test_tasks)} tasks")
    
    # Solve each task
    print("\nSolving tasks...")
    submission = {}
    
    for i, (task_id, task_data) in enumerate(test_tasks.items()):
        if i < 3:  # Only process first 3 for debugging
            print(f"  Task {i+1}/{min(3, len(test_tasks))}: {task_id}")
            try:
                solution = solve_task(model, task_data, device)
                submission[task_id] = solution
                print(f"    ✓ Solution shape: {len(solution)}x{len(solution[0]) if solution else 0}")
            except Exception as e:
                print(f"    ⚠ Error: {e}")
                submission[task_id] = [[0]]
    
    print(f"\n✓ Processed {len(submission)} tasks")
    
    # Save submission
    print("\n" + "=" * 60)
    print("Saving submission")
    print("=" * 60)
    
    # Try multiple output locations
    output_locations = [
        OUTPUT_PATH / 'submission.json',
        Path('/kaggle/working/submission.json'),
        Path('./submission.json'),
        Path('submission.json')
    ]
    
    saved = False
    for output_path in output_locations:
        try:
            print(f"\nAttempting to save to {output_path}")
            print(f"  Parent directory: {output_path.parent}")
            print(f"  Parent exists: {output_path.parent.exists()}")
            
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file
            with open(output_path, 'w') as f:
                json.dump(submission, f, indent=2)
            
            # Verify it was written
            if output_path.exists():
                size = output_path.stat().st_size
                print(f"  ✓ SUCCESS: File saved ({size} bytes)")
                saved = True
                break
            else:
                print(f"  ⚠ File not found after writing")
                
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
    
    if not saved:
        print("\n⚠ WARNING: Could not save submission file!")
    
    # List output directory to confirm
    print(f"\nContents of {OUTPUT_PATH}:")
    if OUTPUT_PATH.exists():
        for item in OUTPUT_PATH.iterdir():
            print(f"  - {item.name} ({item.stat().st_size} bytes)")
    else:
        print("  OUTPUT_PATH does not exist!")
    
    # Also check /kaggle/working directly
    if os.path.exists('/kaggle/working'):
        print(f"\nContents of /kaggle/working:")
        for item in Path('/kaggle/working').iterdir():
            print(f"  - {item.name} ({item.stat().st_size} bytes)")
    
    print("\n" + "=" * 60)
    print("DONE - Script completed")
    print("=" * 60)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n⚠ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    # Final check of working directory
    print("\n" + "=" * 60)
    print("FINAL CHECK")
    print("=" * 60)
    print(f"Current directory: {os.getcwd()}")
    print("Files in current directory:")
    for item in Path('.').iterdir():
        if item.is_file():
            print(f"  - {item.name} ({item.stat().st_size} bytes)")