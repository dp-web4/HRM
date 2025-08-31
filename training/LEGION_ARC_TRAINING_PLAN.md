# Legion RTX 4090 ARC Training Plan for HRM

## Overview
This plan sets up HRM training on the ARC (Abstraction and Reasoning Corpus) dataset using Legion's RTX 4090 GPU. The goal is to train the 27M parameter Hierarchical Reasoning Model to solve abstract reasoning puzzles with minimal data.

## Hardware Specifications
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **System**: Legion Pro 7 with 64GB RAM
- **CUDA**: 12.1 compatible
- **Expected Training Time**: 4-8 hours for full convergence

## Training Configuration

### Model Parameters
```python
MODEL_CONFIG = {
    'batch_size': 32,  # RTX 4090 can handle larger batches
    'seq_len': 900,  # 30x30 grid max = 900 positions
    'puzzle_emb_ndim': 128,
    'vocab_size': 11,  # 0-9 colors + padding
    'H_cycles': 4,  # More cycles for complex reasoning
    'L_cycles': 8,  # More tactical iterations
    'H_layers': 6,  # Deeper strategic reasoning
    'L_layers': 4,  # Deeper tactical execution
    'hidden_size': 512,  # Larger hidden dimension
    'expansion': 4.0,
    'num_heads': 16,  # More attention heads
    'pos_encodings': 'rope',
    'halt_max_steps': 12,  # Allow more computation steps
    'halt_exploration_prob': 0.1,
    'forward_dtype': 'bfloat16',  # RTX 4090 supports bfloat16
}
```

### Training Hyperparameters
```python
TRAINING_CONFIG = {
    'learning_rate': 3e-4,
    'warmup_steps': 500,
    'max_epochs': 100,
    'gradient_accumulation_steps': 2,  # Effective batch size = 64
    'mixed_precision': True,  # Use automatic mixed precision
    'gradient_clip': 1.0,
    'weight_decay': 0.01,
    'dropout': 0.1,
    'eval_frequency': 100,  # Validate every 100 steps
    'checkpoint_frequency': 500,  # Save every 500 steps
}
```

### Dataset Configuration
```python
DATASET_CONFIG = {
    'train_dir': 'data/arc-aug-1000/train',
    'val_dir': 'data/arc-aug-1000/val',
    'test_dir': 'data/arc-aug-1000/test',
    'num_augmentations': 1000,  # As per HRM paper
    'max_grid_size': 30,
    'num_workers': 8,  # Parallel data loading
    'pin_memory': True,
    'persistent_workers': True,
}
```

## Setup Steps

### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv arc_training_env
source arc_training_env/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy pandas tqdm tensorboard wandb argdantic pydantic
pip install einops accelerate transformers datasets
```

### 2. Data Preparation
```bash
# Download ARC dataset if not present
mkdir -p dataset/raw-data
cd dataset/raw-data

# Clone ARC repositories
git clone https://github.com/fchollet/ARC-AGI.git
git clone https://github.com/victorvikram/ConceptARC.git

# Return to project root
cd ../..

# Generate augmented dataset
python dataset/build_arc_dataset.py \
    --dataset-dirs dataset/raw-data/ARC-AGI/data dataset/raw-data/ConceptARC/corpus \
    --output-dir data/arc-aug-1000 \
    --num-aug 1000 \
    --seed 42
```

### 3. Model Initialization
```bash
# Verify GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
python -c "import torch; print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

## Training Script

Create `training/train_arc_legion.py`:

```python
#!/usr/bin/env python3
"""
ARC Training Script for HRM on Legion RTX 4090
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import wandb
from tqdm import tqdm
import json
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config
)
from dataset.common import PuzzleDataset, PuzzleDatasetConfig
from models.losses import ACTLoss

def train_arc():
    # Initialize wandb for experiment tracking
    wandb.init(
        project="hrm-arc-training",
        name=f"arc-legion-{time.strftime('%Y%m%d-%H%M%S')}",
        config={**MODEL_CONFIG, **TRAINING_CONFIG}
    )
    
    device = torch.device('cuda')
    print(f"🚀 Training on {torch.cuda.get_device_name(0)}")
    
    # Dataset setup
    dataset_config = PuzzleDatasetConfig(
        root="data/arc-aug-1000",
        vocab_size=11,
        puzzle_emb_ndim=128,
        seq_len=900,
    )
    
    train_dataset = PuzzleDataset(dataset_config, split="train")
    val_dataset = PuzzleDataset(dataset_config, split="val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=True,
        num_workers=DATASET_CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=False,
        num_workers=DATASET_CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True,
    )
    
    # Model setup
    model_config = HierarchicalReasoningModel_ACTV1Config(
        **MODEL_CONFIG,
        num_puzzle_identifiers=len(train_dataset)
    )
    
    model = HierarchicalReasoningModel_ACTV1(model_config).to(device)
    
    # Print model size
    param_count = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {param_count/1e6:.1f}M")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * TRAINING_CONFIG['max_epochs']
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Loss function with ACT
    criterion = ACTLoss(
        halt_penalty=0.01,
        correct_prediction_reward=1.0
    )
    
    # Training loop
    best_val_acc = 0
    global_step = 0
    
    for epoch in range(TRAINING_CONFIG['max_epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast():
                outputs, carry = model(batch)
                loss, metrics = criterion(outputs, batch['target'], carry)
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % TRAINING_CONFIG['gradient_accumulation_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    TRAINING_CONFIG['gradient_clip']
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            predictions = outputs.argmax(dim=-1)
            train_correct += (predictions == batch['target']).sum().item()
            train_total += batch['target'].numel()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/train_total:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })
            
            # Log to wandb
            if global_step % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/accuracy': train_correct/train_total,
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'train/halt_steps': metrics.get('avg_halt_steps', 0),
                    'global_step': global_step
                })
            
            # Validation
            if global_step % TRAINING_CONFIG['eval_frequency'] == 0:
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for val_batch in tqdm(val_loader, desc="Validation"):
                        val_batch = {k: v.to(device) for k, v in val_batch.items()}
                        
                        with autocast():
                            outputs, carry = model(val_batch)
                            loss, _ = criterion(outputs, val_batch['target'], carry)
                        
                        val_loss += loss.item()
                        predictions = outputs.argmax(dim=-1)
                        val_correct += (predictions == val_batch['target']).sum().item()
                        val_total += val_batch['target'].numel()
                
                val_accuracy = val_correct / val_total
                wandb.log({
                    'val/loss': val_loss / len(val_loader),
                    'val/accuracy': val_accuracy,
                    'global_step': global_step
                })
                
                # Save best model
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_acc': best_val_acc,
                        'config': model_config.dict()
                    }, f'checkpoints/hrm_arc_best.pt')
                    print(f"✅ New best model saved: {val_accuracy:.4f}")
                
                model.train()
            
            # Checkpoint
            if global_step % TRAINING_CONFIG['checkpoint_frequency'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                    'config': model_config.dict()
                }, f'checkpoints/hrm_arc_step_{global_step}.pt')
            
            global_step += 1
        
        # Epoch summary
        epoch_acc = train_correct / train_total
        print(f"Epoch {epoch+1} - Train Acc: {epoch_acc:.4f}, Best Val Acc: {best_val_acc:.4f}")
    
    wandb.finish()
    print(f"🎉 Training complete! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load configs from this file
    exec(open(__file__).read().split('## Training Script')[0])
    
    # Start training
    train_arc()
```

## Monitoring and Evaluation

### Real-time Monitoring
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tensorboard --logdir logs/

# View wandb dashboard
# Go to https://wandb.ai/your-username/hrm-arc-training
```

### Evaluation Script
```bash
# Test on held-out ARC puzzles
python evaluate_arc.py --checkpoint checkpoints/hrm_arc_best.pt --test-dir data/arc-aug-1000/test
```

## Expected Results

Based on the HRM paper:
- **Training Accuracy**: >95% on augmented training set
- **Validation Accuracy**: >85% on held-out puzzles
- **Test Accuracy**: >75% on completely novel ARC tasks
- **Training Time**: 4-8 hours on RTX 4090
- **Convergence**: Should see rapid improvement in first 20 epochs

## Troubleshooting

### Out of Memory
- Reduce batch_size to 16 or 8
- Reduce hidden_size to 384
- Use gradient checkpointing

### Slow Training
- Ensure CUDA and cuDNN are properly installed
- Check data loading isn't bottleneck (increase num_workers)
- Verify mixed precision is enabled

### Poor Convergence
- Try learning rate sweep: [1e-4, 3e-4, 1e-3]
- Increase num_augmentations to 2000
- Add more H_cycles/L_cycles for deeper reasoning

## Next Steps After Training

1. **Evaluate on Original ARC Test Set**: Test generalization to unseen puzzles
2. **Analyze Failure Cases**: Understand which abstract concepts are challenging
3. **Fine-tune on Specific Patterns**: Target weak areas with additional training
4. **Export for Edge Deployment**: Optimize model for Jetson deployment
5. **Integrate with SAGE**: Use trained HRM as reasoning module in larger system