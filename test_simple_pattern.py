#!/usr/bin/env python3
"""Test SAGE-7M on a simple pattern"""

import torch
import sys
import json
sys.path.insert(0, 'kaggle_submission_package')
from kaggle_submission import SAGE7M, MODEL_CONFIG, preprocess_grid, postprocess_output

def test_simple():
    device = torch.device('cpu')
    
    # Load model
    model = SAGE7M(MODEL_CONFIG).to(device)
    checkpoint = torch.load('kaggle_submission_package/hrm-model/hrm_arc_best.pt', map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Create simple test patterns
    test_cases = [
        # All zeros
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        # All ones
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        # Diagonal
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        # Cross
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    ]
    
    for i, pattern in enumerate(test_cases):
        print(f"\nTest {i+1}:")
        print("Input:", pattern)
        
        # Preprocess
        input_tensor = preprocess_grid(pattern).unsqueeze(0).to(device)
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Run model
        with torch.no_grad():
            output = model(input_tensor)
            print(f"Output shape: {output.shape}")
            
            # Get predictions
            pred = output[0].argmax(dim=-1).cpu()[:9]  # First 9 values (3x3)
            pred_grid = pred.reshape(3, 3).tolist()
            print("Prediction:", pred_grid)
            
            # Check what values model prefers
            probs = torch.softmax(output[0][:9], dim=-1)
            top_values = probs.argmax(dim=-1)
            print(f"Top predicted values: {top_values.tolist()}")
            print(f"Max probs: {probs.max(dim=-1).values.tolist()[:3]}...")

if __name__ == '__main__':
    test_simple()