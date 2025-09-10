#!/usr/bin/env python3
"""
Train SAGE-7M from scratch using Claude's reasoning predictions.
No checkpoint loading - pure distillation from Claude's knowledge.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import numpy as np
from typing import List, Dict, Any

# Import model from kaggle_submission
from kaggle_submission import HierarchicalReasoningModule, MODEL_CONFIG, preprocess_grid

class ARCDistillationDataset(Dataset):
    """Dataset for distillation training using Claude's reasoning predictions"""
    
    def __init__(self, data_path: str, max_size: int = 30):
        self.max_size = max_size
        
        # Load Claude's training data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Flatten all examples
        self.examples = []
        for task_id, task_data in self.data.items():
            # Original training examples
            for example in task_data.get('train', []):
                self.examples.append({
                    'input': example['input'],
                    'output': example['output'],
                    'task_id': task_id,
                    'source': 'original'
                })
            
            # Claude's reasoning predictions (the key distillation data!)
            for example in task_data.get('test', []):
                self.examples.append({
                    'input': example['input'],
                    'output': example['output'],
                    'task_id': task_id,
                    'source': 'claude_reasoning'
                })
        
        print(f"Dataset loaded: {len(self.examples)} total examples")
        claude_examples = sum(1 for e in self.examples if e['source'] == 'claude_reasoning')
        print(f"  - Claude reasoning predictions: {claude_examples}")
        print(f"  - Original ARC training data: {len(self.examples) - claude_examples}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Preprocess
        input_tensor = preprocess_grid(example['input'], self.max_size)
        output_tensor = preprocess_grid(example['output'], self.max_size)
        
        return {
            'input': input_tensor,
            'target': output_tensor,
            'task_id': example['task_id'],
            'source': example['source']
        }

def main():
    """Train from scratch with Claude's knowledge"""
    print("=" * 60)
    print("SAGE-7M Training from Scratch")
    print("Distilling Claude's reasoning into neural weights")
    print("=" * 60)
    
    # Configuration
    BATCH_SIZE = 8  # Smaller batch for better convergence
    LEARNING_RATE = 1e-3  # Start with higher LR
    NUM_EPOCHS = 100  # Fewer epochs but more focused
    SAVE_EVERY = 10
    LOG_EVERY = 1
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataset
    print("\nLoading Claude's reasoning data...")
    data_path = Path('claude_reasoning_training_data.json')
    if not data_path.exists():
        print("Warning: Reasoning predictions not found, using basic predictions")
        data_path = Path('claude_training_data.json')
    
    dataset = ARCDistillationDataset(str(data_path))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create model FROM SCRATCH - no checkpoint!
    print("\nCreating model from random initialization...")
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(device)
    
    # Verify it's on GPU
    print(f"Model device: {next(model.parameters()).device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Batch size: {BATCH_SIZE}, Steps per epoch: {len(dataloader)}")
    print("-" * 60)
    
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        claude_correct = 0
        claude_total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            sources = batch['source']
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs.transpose(1, 2), targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            _, predicted = outputs.max(2)
            batch_correct = predicted.eq(targets).float().mean(dim=1)
            correct += batch_correct.sum().item()
            total += targets.size(0)
            
            # Track Claude examples separately
            for i, source in enumerate(sources):
                if source == 'claude_reasoning':
                    claude_correct += batch_correct[i].item()
                    claude_total += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: Loss={loss.item():.4f}")
        
        # Epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100. * correct / total
        claude_acc = 100. * claude_correct / claude_total if claude_total > 0 else 0
        
        # Learning rate step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log epoch results
        elapsed = time.time() - start_time
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"  Loss: {avg_loss:.6f} | Acc: {accuracy:.2f}% | Claude Acc: {claude_acc:.2f}%")
        print(f"  LR: {current_lr:.6f} | Time: {elapsed/60:.1f}m")
        
        if device.type == 'cuda':
            print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e6:.1f}MB")
            print(f"  GPU Utilization: Check nvidia-smi")
        
        # Save checkpoint
        if (epoch + 1) % SAVE_EVERY == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_name = 'sage_7m_scratch_best.pt'
                print(f"  ✓ New best loss! Saving to {checkpoint_name}")
            else:
                checkpoint_name = f'sage_7m_scratch_epoch_{epoch+1}.pt'
                print(f"  Saving checkpoint to {checkpoint_name}")
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
                'claude_accuracy': claude_acc,
                'config': MODEL_CONFIG
            }
            torch.save(checkpoint, checkpoint_name)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Final accuracy: {accuracy:.2f}%")
    print(f"Claude reasoning accuracy: {claude_acc:.2f}%")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print("=" * 60)

if __name__ == '__main__':
    main()