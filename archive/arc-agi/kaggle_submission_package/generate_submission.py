#!/usr/bin/env python3
"""
Generate Kaggle submission using our trained SAGE model.
This creates LEGAL predictions from the model, not direct Claude predictions.
"""

import json
import torch
import torch.nn as nn
import numpy as np
import math
from pathlib import Path

# Model configuration (must match training)
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

def preprocess_grid(grid, max_size=30):
    """Convert ARC grid to model input tensor"""
    grid_array = np.array(grid, dtype=np.int32)
    h, w = grid_array.shape
    
    # Pad to max_size x max_size with zeros
    padded = np.zeros((max_size, max_size), dtype=np.int32)
    padded[:min(h, max_size), :min(w, max_size)] = grid_array[:min(h, max_size), :min(w, max_size)]
    
    # Flatten to sequence
    return torch.tensor(padded.flatten(), dtype=torch.long)

def postprocess_output(output, height, width, max_size=30):
    """Convert model output to ARC grid format"""
    # Get predicted tokens
    pred = output.argmax(dim=-1).cpu().numpy()
    
    # Reshape from flattened sequence to 30x30 grid
    grid_full = pred.reshape(max_size, max_size)
    
    # Extract the actual dimensions we need
    grid = grid_full[:height, :width].tolist()
    
    return grid

class PositionalEncoding(nn.Module):
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
    """Simplified HRM architecture matching training"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.pos_encoding = PositionalEncoding(config['hidden_size'])
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['hidden_size'],
            nhead=config['num_heads'],
            dim_feedforward=config['hidden_size'] * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['num_h_layers'] + config['num_l_layers']
        )
        
        # Output layer
        self.output_layer = nn.Linear(config['hidden_size'], config['vocab_size'])
    
    def forward(self, x):
        # Embed tokens
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        
        # Transform
        x = self.transformer(x)
        
        # Output
        return self.output_layer(x)

def main():
    print("=" * 60)
    print("Generating Kaggle Submission from Trained Model")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading trained model...")
    checkpoint = torch.load('sage_standalone_trained.pt', map_location=device)
    
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from checkpoint")
    print(f"Training accuracy was: {checkpoint.get('final_accuracy', 'N/A'):.2f}%")
    
    # Load test tasks
    print("\nLoading test tasks...")
    test_path = Path('arc-prize-2025/arc-agi_test_challenges.json')
    if not test_path.exists():
        print("Error: Test file not found. Please ensure arc-prize-2025 directory exists.")
        return
    
    with open(test_path, 'r') as f:
        test_tasks = json.load(f)
    
    print(f"Loaded {len(test_tasks)} test tasks")
    
    # Generate predictions
    print("\nGenerating predictions...")
    submission = {}
    
    with torch.no_grad():
        for i, (task_id, task_data) in enumerate(test_tasks.items()):
            if i % 20 == 0:
                print(f"  Processing task {i+1}/{len(test_tasks)}...")
            
            task_attempts = []
            
            # Process each test case in the task
            for test_case in task_data.get('test', []):
                test_input = test_case['input']
                h = len(test_input)
                w = len(test_input[0]) if h > 0 else 1
                
                # Preprocess input
                input_tensor = preprocess_grid(test_input, 30).unsqueeze(0).to(device)
                
                # Generate prediction
                output = model(input_tensor)
                
                # Postprocess to grid
                predicted_grid = postprocess_output(output[0], h, w, 30)
                
                # Create attempts (attempt_2 is a slight variation)
                attempt_1 = predicted_grid
                
                # For attempt_2, use the same prediction (could add temperature sampling)
                attempt_2 = predicted_grid
                
                task_attempts.append({
                    "attempt_1": attempt_1,
                    "attempt_2": attempt_2
                })
            
            submission[task_id] = task_attempts
    
    # Save submission
    submission_path = 'model_submission.json'
    with open(submission_path, 'w') as f:
        json.dump(submission, f, separators=(',', ':'))
    
    print(f"\n✓ Submission saved to {submission_path}")
    print(f"  Total tasks: {len(submission)}")
    
    # Quick validation
    non_zero_tasks = 0
    for task_id, attempts in submission.items():
        for attempt in attempts:
            grid = attempt.get('attempt_1', [])
            if any(any(cell != 0 for cell in row) for row in grid):
                non_zero_tasks += 1
                break
    
    print(f"  Tasks with non-zero predictions: {non_zero_tasks}/{len(submission)}")
    print("\nDone! This submission is from our trained model, not direct Claude predictions.")

if __name__ == '__main__':
    main()