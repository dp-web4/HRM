#!/usr/bin/env python3
"""
Debug script to check why predictions are all zeros
"""

import json
import torch
import sys
import os

# Add parent to path for imports
sys.path.insert(0, '/mnt/c/projects/ai-agents/HRM')

print("Checking submission predictions...")

# Load submission
with open('working/submission.json', 'r') as f:
    submission = json.load(f)

# Analyze predictions
all_zeros = 0
non_zeros = 0
task_stats = []

for task_id, solution in submission.items():
    # Flatten solution to check values
    flat = []
    for row in solution:
        flat.extend(row)
    
    # Count non-zero values
    non_zero_count = sum(1 for val in flat if val != 0)
    total_cells = len(flat)
    
    if non_zero_count == 0:
        all_zeros += 1
    else:
        non_zeros += 1
    
    task_stats.append({
        'task_id': task_id,
        'shape': f"{len(solution)}x{len(solution[0]) if solution else 0}",
        'non_zeros': non_zero_count,
        'total': total_cells,
        'unique_vals': len(set(flat))
    })

print(f"\nSubmission Analysis:")
print(f"  Total tasks: {len(submission)}")
print(f"  All zeros: {all_zeros}")
print(f"  Has predictions: {non_zeros}")

# Show first few tasks with non-zero predictions
print(f"\nFirst 10 tasks:")
for stat in task_stats[:10]:
    print(f"  {stat['task_id']}: {stat['shape']}, non-zeros: {stat['non_zeros']}/{stat['total']}, unique: {stat['unique_vals']}")

# Check if model is actually producing varied outputs
print("\nChecking model outputs directly...")

# Import model components
sys.path.append('.')
from kaggle_submission import HierarchicalReasoningModule, MODEL_CONFIG, MODEL_PATH, preprocess_grid

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = HierarchicalReasoningModule(MODEL_CONFIG).to(device)
print(f"Model created, parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check if checkpoint exists and loads
if MODEL_PATH.exists():
    print(f"\nLoading checkpoint from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    print(f"Checkpoint keys: {list(checkpoint.keys())[:5]}")
    
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model weights loaded from checkpoint")
        except Exception as e:
            print(f"✗ Error loading weights: {e}")
    else:
        print("✗ No model_state_dict in checkpoint")
        
    # Check if model weights are non-zero
    sample_weight = next(model.parameters())
    print(f"Sample weight stats: min={sample_weight.min():.4f}, max={sample_weight.max():.4f}, mean={sample_weight.mean():.4f}")
    if torch.allclose(sample_weight, torch.zeros_like(sample_weight)):
        print("⚠️ WARNING: Model weights appear to be all zeros!")
else:
    print(f"✗ Checkpoint not found at {MODEL_PATH}")

# Test with a simple input
print("\nTesting model with random input...")
test_input = torch.randint(0, 10, (1, 900)).to(device)
model.eval()
with torch.no_grad():
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
    
    # Get predictions
    preds = output.argmax(dim=-1)
    print(f"Predictions shape: {preds.shape}")
    print(f"Unique predicted values: {torch.unique(preds).cpu().tolist()}")
    print(f"First 20 predictions: {preds[0][:20].cpu().tolist()}")

# Test with actual task
print("\nTesting with actual task...")
with open('arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
    test_tasks = json.load(f)

first_task = test_tasks[list(test_tasks.keys())[0]]
test_grid = first_task['test'][0]['input']
print(f"Test grid shape: {len(test_grid)}x{len(test_grid[0])}")

test_tensor = preprocess_grid(test_grid).unsqueeze(0).to(device)
with torch.no_grad():
    output = model(test_tensor)
    preds = output.argmax(dim=-1)
    print(f"Predictions for first task: {preds[0][:20].cpu().tolist()}")
    print(f"Unique values: {torch.unique(preds).cpu().tolist()}")