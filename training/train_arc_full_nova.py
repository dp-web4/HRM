#!/usr/bin/env python3
"""
Full ARC Training Script for HRM-style Model - Nova's Optimized Version
Production training with hierarchical reasoning architecture
Implements Nova's performance and stability improvements
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import json
import time
from tqdm import tqdm
import numpy as np
import math
from typing import Dict, Any, Tuple, Optional
import wandb
import glob

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Enable cuDNN benchmark for small tensors
torch.backends.cudnn.benchmark = True

# Configuration
MODEL_CONFIG = {
    'batch_size': 20,  # Keep stable batch size
    'seq_len': 900,  # 30x30 grid max
    'vocab_size': 12,  # 0-9 colors + padding/blank
    'hidden_size': 256,
    'num_heads': 8,
    'num_h_layers': 4,  # Strategic layers
    'num_l_layers': 3,  # Tactical layers
    'dropout': 0.1,
    'max_cycles': 8,  # Maximum reasoning cycles
}

TRAINING_CONFIG = {
    'learning_rate': 3e-4,
    'warmup_steps': 500,
    'max_epochs': 100,
    'gradient_accumulation_steps': 2,  # Effective batch size = 40
    'gradient_clip': 1.0,
    'weight_decay': 0.01,
    'eval_frequency': 1000,  # Fast validation frequency
    'full_eval_every': 10000,  # Full validation frequency
    'checkpoint_frequency': 500,
    'use_amp': True,
    'patience': 10,
    'max_val_batches': None,  # Will be set to 10% of val set
}

# DataLoader settings per Nova's recommendations
DATALOADER_CONFIG = {
    'num_workers': 4,  # Increased from 2
    'prefetch_factor': 2,
    'persistent_workers': True,
    'pin_memory': True,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ARCDataset(Dataset):
    """Dataset loader for ARC puzzles"""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load data files
        split_dir = self.data_dir / split
        if split_dir.exists() and (split_dir / 'all__inputs.npy').exists():
            self.inputs = np.load(split_dir / 'all__inputs.npy')
            self.labels = np.load(split_dir / 'all__labels.npy')
            print(f"Loaded {len(self.inputs)} {split} samples")
        else:
            # Create dummy data if dataset not ready
            print(f"Warning: Dataset not found, using dummy data for {split}")
            self.inputs = np.random.randint(0, 10, (1000, 900))
            self.labels = np.random.randint(0, 10, (1000, 900))
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'input': torch.from_numpy(self.inputs[idx]).long(),
            'target': torch.from_numpy(self.labels[idx]).long()
        }

class PositionalEncoding(nn.Module):
    """RoPE-style positional encoding"""
    
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
    """Simplified HRM-style architecture"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.pos_encoding = PositionalEncoding(config['hidden_size'])
        
        # H-level (strategic) layers
        self.h_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_h_layers'])
        ])
        
        # L-level (tactical) layers
        self.l_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_l_layers'])
        ])
        
        # Interaction layers
        self.h_to_l = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.l_to_h = nn.Linear(config['hidden_size'], config['hidden_size'])
        
        # Halting mechanism
        self.halt_predictor = nn.Linear(config['hidden_size'] * 2, 1)
        
        # Output layer
        self.output = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        # Layer norms
        self.h_norm = nn.LayerNorm(config['hidden_size'])
        self.l_norm = nn.LayerNorm(config['hidden_size'])
        
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x, max_cycles=None):
        batch_size, seq_len = x.shape
        max_cycles = max_cycles or self.config['max_cycles']
        
        # Embed input
        x_emb = self.token_embedding(x)
        x_emb = self.pos_encoding(x_emb)
        x_emb = self.dropout(x_emb)
        
        # Initialize H and L states
        h_state = x_emb.clone()
        l_state = x_emb.clone()
        
        # Store halting probabilities
        halt_probs = []
        
        # Reasoning cycles
        for cycle in range(max_cycles):
            # H-level processing (strategic)
            h_prev = h_state.clone()
            for h_layer in self.h_layers:
                h_state = h_layer(h_state)
            h_state = self.h_norm(h_state)
            
            # L-level processing (tactical)
            l_prev = l_state.clone()
            # Incorporate H-level guidance
            l_state = l_state + self.h_to_l(h_state)
            for l_layer in self.l_layers:
                l_state = l_layer(l_state)
            l_state = self.l_norm(l_state)
            
            # L to H feedback
            h_state = h_state + self.l_to_h(l_state)
            
            # Compute halting probability
            combined = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)
            halt_prob = torch.sigmoid(self.halt_predictor(combined))
            halt_probs.append(halt_prob)
            
            # Early stopping based on halt probability
            if cycle > 0 and halt_prob.mean() > 0.9:
                break
        
        # Final output from L-level (execution)
        output = self.output(l_state)
        
        return output, halt_probs

class ACTLoss(nn.Module):
    """Adaptive Computation Time loss"""
    
    def __init__(self, halt_penalty=0.01):
        super().__init__()
        self.halt_penalty = halt_penalty
    
    def forward(self, outputs, targets, halt_probs):
        # Main task loss
        task_loss = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1)
        )
        
        # Halting loss (encourage fewer cycles)
        if halt_probs:
            halt_loss = sum(p.mean() for p in halt_probs) / len(halt_probs)
        else:
            halt_loss = 0.0
        
        total_loss = task_loss + self.halt_penalty * halt_loss
        
        return total_loss, {
            'task_loss': task_loss.item(),
            'halt_loss': halt_loss.item() if halt_probs else 0.0,
            'num_cycles': len(halt_probs)
        }

class TrainingMetrics:
    """Track detailed training metrics per Nova's recommendations"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.data_wait_time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.optimizer_time = 0
        self.batch_count = 0
        self.sample_count = 0
        self.start_time = time.perf_counter()
    
    def log_batch(self, batch_size):
        self.batch_count += 1
        self.sample_count += batch_size
    
    def get_stats(self):
        elapsed = time.perf_counter() - self.start_time
        if elapsed == 0:
            return {}
        
        return {
            'samples_per_sec': self.sample_count / elapsed,
            'batches_per_sec': self.batch_count / elapsed,
            'avg_data_wait_ms': (self.data_wait_time / max(1, self.batch_count)) * 1000,
            'avg_forward_ms': (self.forward_time / max(1, self.batch_count)) * 1000,
            'avg_backward_ms': (self.backward_time / max(1, self.batch_count)) * 1000,
            'avg_optimizer_ms': (self.optimizer_time / max(1, self.batch_count)) * 1000,
        }

def find_latest_checkpoint():
    """Find the latest checkpoint to resume from"""
    step_checkpoints = glob.glob('checkpoints/hrm_arc_step_*.pt')
    if step_checkpoints:
        # Sort by step number and get the latest
        step_checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return step_checkpoints[-1], 'step'
    elif Path('checkpoints/hrm_arc_best.pt').exists():
        return 'checkpoints/hrm_arc_best.pt', 'best'
    return None, None

def save_training_state(path, epoch, global_step, model, optimizer, scheduler, 
                       best_val_loss, config, metrics=None):
    """Save complete training state"""
    state = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'config': config,
    }
    if metrics:
        state['metrics'] = metrics
    
    torch.save(state, path)
    
    # Also save a JSON sidecar with key info
    json_path = str(path).replace('.pt', '_info.json')
    info = {
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': float(best_val_loss),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if metrics:
        info['metrics'] = {k: float(v) if isinstance(v, (int, float)) else v 
                          for k, v in metrics.items()}
    
    with open(json_path, 'w') as f:
        json.dump(info, f, indent=2)

def train():
    """Main training loop with Nova's improvements"""
    print(f"🚀 Starting Nova-optimized HRM training on {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Check for existing dataset, use largest available
    if Path('../data/arc-aug-1000/train/all__inputs.npy').exists():
        data_dir = '../data/arc-aug-1000'
        print("Using full dataset (1000 augmentations)")
    elif Path('../data/arc-aug-500/train/all__inputs.npy').exists():
        data_dir = '../data/arc-aug-500'
        print("Using medium dataset (500 augmentations)")
    elif Path('../data/arc-aug-100/train/all__inputs.npy').exists():
        data_dir = '../data/arc-aug-100'
        print("Using smaller dataset (100 augmentations)")
    else:
        data_dir = '../data/arc-dummy'
        print("Warning: No dataset found, using dummy data")
    
    # Initialize wandb
    run = wandb.init(
        project="hrm-arc-training",
        name=f"arc-nova-{time.strftime('%Y%m%d-%H%M%S')}",
        config={**MODEL_CONFIG, **TRAINING_CONFIG, **DATALOADER_CONFIG},
        mode="offline"  # Offline mode, sync later
    )
    
    # Create datasets
    train_dataset = ARCDataset(data_dir, 'train')
    val_dataset = ARCDataset(data_dir, 'test') if Path(f'{data_dir}/test').exists() else None
    
    # Set max_val_batches to 10% of validation set
    if val_dataset:
        TRAINING_CONFIG['max_val_batches'] = max(1, len(val_dataset) // (MODEL_CONFIG['batch_size'] * 10))
        print(f"Fast validation will use {TRAINING_CONFIG['max_val_batches']} batches (~10% of val set)")
    
    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=True,
        **DATALOADER_CONFIG
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=False,
        **DATALOADER_CONFIG
    ) if val_dataset else None
    
    # Create model
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(DEVICE)
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
    
    # Load checkpoint if exists
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    best_val_step = 0
    
    checkpoint_path, checkpoint_type = find_latest_checkpoint()
    if checkpoint_path:
        print(f"📂 Loading {checkpoint_type} checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint_type == 'step':  # Only restore optimizer for step checkpoints
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'global_step' in checkpoint:
                global_step = checkpoint.get('global_step', 0)
        
        # Always try to load best val loss from best checkpoint
        if Path('checkpoints/hrm_arc_best.pt').exists():
            best_checkpoint = torch.load('checkpoints/hrm_arc_best.pt', map_location='cpu')
            if 'best_val_loss' in best_checkpoint:
                best_val_loss = best_checkpoint['best_val_loss']
        
        print(f"✅ Resumed from epoch {start_epoch}, step {global_step}, best val loss: {best_val_loss:.4f}")
    else:
        print("🆕 Starting fresh training (no checkpoint found)")
    
    # Mixed precision training
    scaler = GradScaler() if TRAINING_CONFIG['use_amp'] else None
    
    # Loss function
    criterion = ACTLoss(halt_penalty=0.01)
    
    # Metrics tracker
    metrics = TrainingMetrics()
    
    print("\n🏃 Starting training...")
    
    for epoch in range(start_epoch, TRAINING_CONFIG['max_epochs']):
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_task_loss = 0
        epoch_halt_loss = 0
        total_cycles = 0
        num_batches = 0
        
        metrics.reset()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['max_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Skip batches if resuming (simplified - just skip based on global_step)
            if epoch == start_epoch and batch_idx == 0 and global_step > 0:
                # Skip to approximately the right position
                skip_batches = global_step % len(train_loader)
                if batch_idx < skip_batches:
                    continue
                print(f"Resumed training from approximately batch {skip_batches}")
            
            # Track data loading time
            data_start = time.perf_counter()
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            metrics.data_wait_time += time.perf_counter() - data_start
            
            # Forward pass with timing
            forward_start = time.perf_counter()
            if TRAINING_CONFIG['use_amp']:
                with autocast():
                    outputs, halt_probs = model(inputs)
                    loss, loss_metrics = criterion(outputs, targets, halt_probs)
            else:
                outputs, halt_probs = model(inputs)
                loss, loss_metrics = criterion(outputs, targets, halt_probs)
            
            # Scale loss for gradient accumulation
            loss = loss / TRAINING_CONFIG['gradient_accumulation_steps']
            metrics.forward_time += time.perf_counter() - forward_start
            
            # Backward pass with timing
            backward_start = time.perf_counter()
            if TRAINING_CONFIG['use_amp']:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            metrics.backward_time += time.perf_counter() - backward_start
            
            # Gradient accumulation and optimizer step
            if (batch_idx + 1) % TRAINING_CONFIG['gradient_accumulation_steps'] == 0:
                optimizer_start = time.perf_counter()
                
                if TRAINING_CONFIG['use_amp']:
                    scaler.unscale_(optimizer)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    TRAINING_CONFIG['gradient_clip']
                )
                
                if TRAINING_CONFIG['use_amp']:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                metrics.optimizer_time += time.perf_counter() - optimizer_start
            
            # Update metrics
            metrics.log_batch(inputs.size(0))
            epoch_loss += loss.item() * TRAINING_CONFIG['gradient_accumulation_steps']
            epoch_task_loss += loss_metrics['task_loss']
            epoch_halt_loss += loss_metrics['halt_loss']
            total_cycles += loss_metrics['num_cycles']
            num_batches += 1
            
            # Calculate accuracy
            predictions = outputs.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            
            # Update progress bar with actual metrics
            stats = metrics.get_stats()
            progress_bar.set_postfix({
                'loss': f"{loss.item()*TRAINING_CONFIG['gradient_accumulation_steps']:.4f}",
                'acc': f"{accuracy:.4f}",
                'cycles': f"{loss_metrics['num_cycles']:.1f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}",
                'samples/s': f"{stats.get('samples_per_sec', 0):.1f}"
            })
            
            # Log to wandb
            if global_step % 10 == 0:
                wandb.log({
                    'train/loss': loss.item() * TRAINING_CONFIG['gradient_accumulation_steps'],
                    'train/task_loss': loss_metrics['task_loss'],
                    'train/halt_loss': loss_metrics['halt_loss'],
                    'train/accuracy': accuracy,
                    'train/num_cycles': loss_metrics['num_cycles'],
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'train/samples_per_sec': stats.get('samples_per_sec', 0),
                    'train/data_wait_ms': stats.get('avg_data_wait_ms', 0),
                    'train/forward_ms': stats.get('avg_forward_ms', 0),
                    'train/backward_ms': stats.get('avg_backward_ms', 0),
                    'global_step': global_step
                })
            
            # Fast validation (10% of val set)
            if val_loader and global_step % TRAINING_CONFIG['eval_frequency'] == 0 and global_step > 0:
                model.eval()
                val_loss = 0
                val_acc = 0
                val_cycles = 0
                val_batches = 0
                
                with torch.no_grad():
                    for val_idx, val_batch in enumerate(tqdm(val_loader, 
                                                            desc="Fast Validation", 
                                                            leave=False)):
                        if TRAINING_CONFIG['max_val_batches'] and val_idx >= TRAINING_CONFIG['max_val_batches']:
                            break
                        
                        val_inputs = val_batch['input'].to(DEVICE)
                        val_targets = val_batch['target'].to(DEVICE)
                        
                        if TRAINING_CONFIG['use_amp']:
                            with autocast():
                                val_outputs, val_halt_probs = model(val_inputs)
                                v_loss, v_metrics = criterion(val_outputs, val_targets, val_halt_probs)
                        else:
                            val_outputs, val_halt_probs = model(val_inputs)
                            v_loss, v_metrics = criterion(val_outputs, val_targets, val_halt_probs)
                        
                        val_loss += v_loss.item()
                        val_predictions = val_outputs.argmax(dim=-1)
                        val_acc += (val_predictions == val_targets).float().mean().item()
                        val_cycles += v_metrics['num_cycles']
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                avg_val_acc = val_acc / val_batches
                avg_val_cycles = val_cycles / val_batches
                
                wandb.log({
                    'val/loss': avg_val_loss,
                    'val/accuracy': avg_val_acc,
                    'val/num_cycles': avg_val_cycles,
                    'val/type': 'fast',
                    'global_step': global_step
                })
                
                print(f"\n📊 Fast Val (step {global_step}): loss={avg_val_loss:.4f}, acc={avg_val_acc:.4f}")
                
                # Run full validation if fast val improved
                run_full_val = avg_val_loss < best_val_loss * 1.1  # Within 10% of best
                
                # Also run full val at regular mega-intervals
                if global_step % TRAINING_CONFIG['full_eval_every'] == 0:
                    run_full_val = True
                
                if run_full_val and val_loader:
                    print("Running full validation...")
                    full_val_loss = 0
                    full_val_acc = 0
                    full_val_cycles = 0
                    full_val_batches = 0
                    
                    with torch.no_grad():
                        for val_batch in tqdm(val_loader, desc="Full Validation", leave=False):
                            val_inputs = val_batch['input'].to(DEVICE)
                            val_targets = val_batch['target'].to(DEVICE)
                            
                            if TRAINING_CONFIG['use_amp']:
                                with autocast():
                                    val_outputs, val_halt_probs = model(val_inputs)
                                    v_loss, v_metrics = criterion(val_outputs, val_targets, val_halt_probs)
                            else:
                                val_outputs, val_halt_probs = model(val_inputs)
                                v_loss, v_metrics = criterion(val_outputs, val_targets, val_halt_probs)
                            
                            full_val_loss += v_loss.item()
                            val_predictions = val_outputs.argmax(dim=-1)
                            full_val_acc += (val_predictions == val_targets).float().mean().item()
                            full_val_cycles += v_metrics['num_cycles']
                            full_val_batches += 1
                    
                    avg_full_val_loss = full_val_loss / full_val_batches
                    avg_full_val_acc = full_val_acc / full_val_batches
                    avg_full_val_cycles = full_val_cycles / full_val_batches
                    
                    wandb.log({
                        'val/full_loss': avg_full_val_loss,
                        'val/full_accuracy': avg_full_val_acc,
                        'val/full_num_cycles': avg_full_val_cycles,
                        'val/type': 'full',
                        'global_step': global_step
                    })
                    
                    # Save best model
                    if avg_full_val_loss < best_val_loss:
                        best_val_loss = avg_full_val_loss
                        best_val_step = global_step
                        save_training_state(
                            'checkpoints/hrm_arc_best.pt',
                            epoch, global_step, model, optimizer, scheduler,
                            best_val_loss, MODEL_CONFIG,
                            {'val_accuracy': avg_full_val_acc, 'val_cycles': avg_full_val_cycles}
                        )
                        print(f"\n✅ New best model! loss={avg_full_val_loss:.4f}, acc={avg_full_val_acc:.4f}")
                
                model.train()
            
            # Checkpoint
            if global_step % TRAINING_CONFIG['checkpoint_frequency'] == 0 and global_step > 0:
                save_training_state(
                    f'checkpoints/hrm_arc_step_{global_step}.pt',
                    epoch, global_step, model, optimizer, scheduler,
                    best_val_loss, MODEL_CONFIG, metrics.get_stats()
                )
                print(f"\n💾 Checkpoint saved at step {global_step}")
            
            global_step += 1
            
            # Early termination for testing
            if global_step > 100 and data_dir == '../data/arc-dummy':
                print("\nStopping early - using dummy data")
                break
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_task_loss = epoch_task_loss / num_batches
        avg_epoch_halt_loss = epoch_halt_loss / num_batches
        avg_epoch_cycles = total_cycles / num_batches
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Loss: {avg_epoch_loss:.4f} (task: {avg_epoch_task_loss:.4f}, halt: {avg_epoch_halt_loss:.4f})")
        print(f"  Avg cycles: {avg_epoch_cycles:.2f}")
        print(f"  Training metrics: {metrics.get_stats()}")
        
        # Early termination for testing
        if global_step > 100 and data_dir == '../data/arc-dummy':
            break
    
    # Final save
    save_training_state(
        'checkpoints/hrm_arc_final.pt',
        epoch, global_step, model, optimizer, scheduler,
        best_val_loss, MODEL_CONFIG, metrics.get_stats()
    )
    
    wandb.finish()
    print("\n🎉 Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f} at step {best_val_step}")
    print("Models saved in checkpoints/")

if __name__ == "__main__":
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)
    
    try:
        train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted! Saving checkpoint...")
        print("💾 You can resume training by running the script again.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise