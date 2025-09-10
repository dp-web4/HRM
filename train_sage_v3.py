#!/usr/bin/env python3
"""
Train SAGE V3 using human-like predictions.
This version uses simple, visual reasoning patterns instead of complex algorithms.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

# Import the SAGE model architecture
import sys
sys.path.append('.')
from kaggle_submission import HRMModel  # Reuse existing architecture

class ARCDataset(Dataset):
    """Dataset for ARC tasks with human-like predictions"""
    
    def __init__(self, data_path: str, max_size: int = 30):
        """Load training data from JSON"""
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.max_size = max_size
        self.task_ids = list(self.data.keys())
        
        # Flatten all input-output pairs
        self.pairs = []
        for task_id in self.task_ids:
            task = self.data[task_id]
            # Add training examples
            for example in task.get('train', []):
                self.pairs.append((example['input'], example['output']))
            # Add test examples (with our predictions as outputs)
            for example in task.get('test', []):
                if 'output' in example:  # Our predictions
                    self.pairs.append((example['input'], example['output']))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_grid, output_grid = self.pairs[idx]
        
        # Pad to max_size
        input_tensor = self.grid_to_tensor(input_grid)
        output_tensor = self.grid_to_tensor(output_grid)
        
        return input_tensor, output_tensor
    
    def grid_to_tensor(self, grid):
        """Convert grid to padded tensor"""
        h, w = len(grid), len(grid[0]) if grid else 0
        
        # Create padded tensor
        tensor = torch.zeros((self.max_size, self.max_size), dtype=torch.long)
        
        # Copy grid into tensor
        for i in range(min(h, self.max_size)):
            for j in range(min(w, self.max_size)):
                if i < len(grid) and j < len(grid[i]):
                    tensor[i, j] = grid[i][j]
        
        return tensor

def train_sage_v3():
    """Train SAGE V3 with human-like predictions"""
    print("Training SAGE V3 with Human-like Reasoning")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = ARCDataset('sage_v3_training_data.json')
    print(f"Loaded {len(dataset)} training pairs")
    
    # Create dataloader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = HRMModel(
        input_dim=30*30,
        hidden_dim=256,
        num_layers=6,
        num_colors=10,
        max_size=30
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training loop
    num_epochs = 100
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, 10)  # 10 color classes
            targets = targets.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'training_approach': 'human-like visual reasoning'
    }
    
    save_path = 'sage_v3_human_trained.pt'
    torch.save(checkpoint, save_path)
    print(f"\n✓ Model saved to {save_path}")
    
    # Generate submission
    print("\nGenerating Kaggle submission...")
    model.eval()
    
    # Load test data and generate predictions
    test_path = Path('kaggle_submission_package/arc-prize-2025/arc-agi_test_challenges.json')
    with open(test_path, 'r') as f:
        test_tasks = json.load(f)
    
    predictions = {}
    
    with torch.no_grad():
        for task_id, task_data in test_tasks.items():
            task_predictions = []
            
            for test_case in task_data.get('test', []):
                # Convert to tensor
                input_grid = test_case['input']
                input_tensor = dataset.grid_to_tensor(input_grid).unsqueeze(0).to(device)
                
                # Generate prediction
                output = model(input_tensor)
                
                # Convert back to grid
                pred_grid = output.squeeze(0).argmax(dim=-1).cpu().numpy()
                
                # Crop to actual size (remove padding)
                h, w = len(input_grid), len(input_grid[0])
                pred_grid = pred_grid[:h, :w].tolist()
                
                # Create two attempts (second is color inverted)
                attempt1 = pred_grid
                
                # Simple variation for attempt 2
                colors = set(c for row in attempt1 for c in row)
                if len(colors) == 2:
                    color_list = sorted(list(colors))
                    attempt2 = [[color_list[1] if c == color_list[0] else color_list[0] 
                                for c in row] for row in attempt1]
                else:
                    attempt2 = attempt1
                
                task_predictions.append({
                    'attempt_1': attempt1,
                    'attempt_2': attempt2
                })
            
            predictions[task_id] = task_predictions
    
    # Save submission
    submission_path = 'sage_v3_submission.json'
    with open(submission_path, 'w') as f:
        json.dump(predictions, f, separators=(',', ':'))
    
    print(f"✓ Submission saved to {submission_path}")
    print("\nSAGE V3 Training Complete!")
    print("This version learned from human-like visual reasoning patterns.")

if __name__ == '__main__':
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    train_sage_v3()