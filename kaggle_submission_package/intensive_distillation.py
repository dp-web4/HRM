#!/usr/bin/env python3
"""
Intensive distillation training: Transfer Claude's reasoning to SAGE-7M.
Optimized for small dataset with hundreds of epochs.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
from typing import List, Dict, Any

# Import the model
import sys
sys.path.append(str(Path(__file__).parent))
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
            
            # Claude's reasoning predictions
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
    """Intensive distillation training"""
    print("=" * 60)
    print("SAGE-7M Intensive Distillation Training")
    print("Transferring Claude's reasoning capabilities")
    print("=" * 60)
    
    # Configuration for intensive training
    BATCH_SIZE = 16  # Larger batch for small dataset
    LEARNING_RATE = 5e-4  # Higher LR for faster convergence
    NUM_EPOCHS = 500  # Many epochs for thorough learning
    SAVE_EVERY = 50  # Save checkpoint every N epochs
    LOG_EVERY = 10  # Log stats every N epochs
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataset - use reasoning predictions
    data_path = Path('claude_reasoning_training_data.json')
    if not data_path.exists():
        print("Warning: Reasoning predictions not found, using basic predictions")
        data_path = Path('claude_training_data.json')
    
    dataset = ARCDistillationDataset(str(data_path))
    
    # Single dataloader - no validation split for maximum training
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead
        pin_memory=(device.type == 'cuda')
    )
    
    print(f"\nTraining configuration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Total epochs: {NUM_EPOCHS}")
    print(f"  Steps per epoch: {len(dataloader)}")
    print(f"  Total steps: {len(dataloader) * NUM_EPOCHS}")
    
    # Initialize model
    print(f"\nInitializing SAGE-7M...")
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Load checkpoint if available (for continued training)
    checkpoint_path = Path('hrm-model/sage_7m_distilled.pt')
    start_epoch = 0
    best_loss = float('inf')
    
    if checkpoint_path.exists():
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"  Resuming from epoch {start_epoch}")
            if 'best_loss' in checkpoint:
                best_loss = checkpoint['best_loss']
                print(f"  Best loss so far: {best_loss:.6f}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler - cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=50,  # Restart every 50 epochs
        T_mult=2,  # Double the period after each restart
        eta_min=1e-6
    )
    
    # Training statistics
    epoch_losses = []
    epoch_accuracies = []
    
    print(f"\n{'='*60}")
    print("Starting intensive training...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        
        epoch_loss = 0
        correct = 0
        total = 0
        
        # Training loop
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, halt_probs = model(inputs)
            
            # Reshape for loss
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Add regularization for halting (encourage fewer cycles)
            if halt_probs:
                halt_reg = sum(hp.mean() for hp in halt_probs) * 0.01
                loss = loss + halt_reg
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        
        # Epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100. * correct / total
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)
        
        # Learning rate step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Logging
        if (epoch + 1) % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1 - start_epoch)) * (NUM_EPOCHS - epoch - 1)
            
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            print(f"  Loss: {avg_loss:.6f} | Acc: {accuracy:.2f}% | LR: {current_lr:.6f}")
            print(f"  Time: {elapsed/60:.1f}m elapsed, {eta/60:.1f}m remaining")
            
            # Show improvement
            if epoch > start_epoch:
                loss_change = avg_loss - epoch_losses[-min(10, len(epoch_losses))]
                acc_change = accuracy - epoch_accuracies[-min(10, len(epoch_accuracies))]
                print(f"  Δ10: Loss {loss_change:+.6f}, Acc {acc_change:+.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % SAVE_EVERY == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_name = 'sage_7m_distilled_best.pt'
                print(f"  ✓ New best loss! Saving to {checkpoint_name}")
            else:
                checkpoint_name = f'sage_7m_epoch_{epoch+1}.pt'
                print(f"  Saving checkpoint to {checkpoint_name}")
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
                'best_loss': best_loss,
                'config': MODEL_CONFIG,
                'epoch_losses': epoch_losses,
                'epoch_accuracies': epoch_accuracies
            }
            
            save_path = Path('hrm-model') / checkpoint_name
            save_path.parent.mkdir(exist_ok=True)
            torch.save(checkpoint, save_path)
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Final loss: {epoch_losses[-1]:.6f}")
    print(f"Final accuracy: {epoch_accuracies[-1]:.2f}%")
    print(f"Best loss achieved: {best_loss:.6f}")
    
    # Save final model
    final_checkpoint = {
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_loss': epoch_losses[-1],
        'final_accuracy': epoch_accuracies[-1],
        'best_loss': best_loss,
        'config': MODEL_CONFIG,
        'epoch_losses': epoch_losses,
        'epoch_accuracies': epoch_accuracies,
        'training_time': total_time
    }
    
    final_path = Path('hrm-model/sage_7m_distilled_final.pt')
    torch.save(final_checkpoint, final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    # Print summary statistics
    print(f"\nTraining Summary:")
    print(f"  Average loss reduction: {(epoch_losses[0] - epoch_losses[-1]):.6f}")
    print(f"  Average accuracy gain: {(epoch_accuracies[-1] - epoch_accuracies[0]):.2f}%")
    print(f"  Best checkpoint: sage_7m_distilled_best.pt (loss: {best_loss:.6f})")
    
    print(f"\nNext steps:")
    print("1. Update kaggle_submission.py to use 'sage_7m_distilled_best.pt'")
    print("2. Generate predictions with the distilled model")
    print("3. Submit to Kaggle!")
    print("\nThe model has learned from Claude's reasoning patterns!")

if __name__ == '__main__':
    main()