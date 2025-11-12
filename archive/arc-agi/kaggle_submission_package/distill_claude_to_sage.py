#!/usr/bin/env python3
"""
Distillation training script: Transfer Claude's knowledge to SAGE-7M.
Uses claude_training_data.json as the training set.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from typing import List, Dict, Any, Tuple

# Import the model from kaggle_submission.py
import sys
sys.path.append(str(Path(__file__).parent))
from kaggle_submission import HierarchicalReasoningModule, MODEL_CONFIG, preprocess_grid

class ARCDistillationDataset(Dataset):
    """Dataset for distillation training using Claude's predictions"""
    
    def __init__(self, data_path: str, max_size: int = 30):
        self.max_size = max_size
        
        # Load Claude's training data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Flatten all examples
        self.examples = []
        for task_id, task_data in self.data.items():
            # Add training examples (original ARC data)
            for example in task_data.get('train', []):
                self.examples.append({
                    'input': example['input'],
                    'output': example['output'],
                    'task_id': task_id,
                    'source': 'train'
                })
            
            # Add test examples with Claude's predictions
            for example in task_data.get('test', []):
                self.examples.append({
                    'input': example['input'],
                    'output': example['output'],  # Claude's prediction
                    'task_id': task_id,
                    'source': 'claude'
                })
        
        print(f"Loaded {len(self.examples)} examples for distillation")
        claude_examples = sum(1 for e in self.examples if e['source'] == 'claude')
        print(f"  - Claude predictions: {claude_examples}")
        print(f"  - Original training: {len(self.examples) - claude_examples}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Preprocess input
        input_tensor = preprocess_grid(example['input'], self.max_size)
        
        # Preprocess output (target)
        output_tensor = preprocess_grid(example['output'], self.max_size)
        
        return {
            'input': input_tensor,
            'target': output_tensor,
            'task_id': example['task_id'],
            'source': example['source']
        }

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs, _ = model(inputs)
        
        # Reshape for loss calculation
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs, _ = model(inputs)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def main():
    """Main distillation training loop"""
    print("SAGE-7M Distillation Training")
    print("Teaching Claude's knowledge to SAGE-7M")
    print("=" * 50)
    
    # Configuration
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    VALIDATION_SPLIT = 0.1
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = ARCDistillationDataset('claude_training_data.json')
    
    # Split into train/val
    dataset_size = len(dataset)
    val_size = int(VALIDATION_SPLIT * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {train_size} examples")
    print(f"  Validation: {val_size} examples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print(f"\nInitializing SAGE-7M model...")
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(device)
    
    # Load existing checkpoint if available
    checkpoint_path = Path('hrm-model/hrm_arc_best.pt')
    start_epoch = 0
    best_val_loss = float('inf')
    
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Loaded pre-trained weights")
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"  Resuming from epoch {start_epoch}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    print(f"\nStarting distillation training...")
    print(f"Training for {NUM_EPOCHS} epochs")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"✓ New best validation loss! Saving checkpoint...")
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'config': MODEL_CONFIG
            }
            
            # Save distilled model
            distilled_path = Path('hrm-model/sage_7m_distilled.pt')
            distilled_path.parent.mkdir(exist_ok=True)
            torch.save(checkpoint, distilled_path)
            print(f"  Saved to {distilled_path}")
    
    print(f"\n{'='*50}")
    print("Distillation training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"\nDistilled model saved to: hrm-model/sage_7m_distilled.pt")
    print("\nTo use the distilled model for Kaggle submission:")
    print("  1. Update MODEL_PATH in kaggle_submission.py to point to sage_7m_distilled.pt")
    print("  2. Run kaggle_submission.py to generate predictions")
    print("  3. Submit to Kaggle!")

if __name__ == '__main__':
    main()