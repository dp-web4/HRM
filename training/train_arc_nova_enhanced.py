#!/usr/bin/env python3
"""
Enhanced ARC Training Script with Nova's Optimizations
Implements status tracking, smart validation, label smoothing, and LR restarts
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
import uuid
from datetime import datetime
import threading

# Configuration
MODEL_CONFIG = {
    'batch_size': 20,  # Back to efficient batch size
    'seq_len': 900,
    'vocab_size': 12,
    'hidden_size': 256,
    'num_heads': 8,
    'num_h_layers': 4,
    'num_l_layers': 3,
    'dropout': 0.1,
    'max_cycles': 8,
}

TRAINING_CONFIG = {
    'learning_rate': 3e-4,
    'warmup_steps': 500,
    'max_epochs': 100,
    'gradient_accumulation_steps': 2,  # Effective batch = 40
    'gradient_clip': 1.0,
    'weight_decay': 0.01,
    'eval_frequency': 10000,  # Fast validation every 10k steps
    'full_eval_every': 10000,  # Not used with smart validation
    'checkpoint_frequency': 500,
    'use_amp': True,
    'patience': 10,
    'max_val_batches': None,
    # New optimizations
    'label_smoothing': 0.1,
    'lr_restart_every': 20000,  # Warm restart every 20k steps
    'smart_validation': True,  # Only full val if fast val improves
    'status_update_interval': 60,  # Update status.json every minute
}

DATALOADER_CONFIG = {
    'num_workers': 4,  # Back to 4 workers
    'prefetch_factor': 2,
    'persistent_workers': True,
    'pin_memory': True,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate unique run ID
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(uuid.uuid4())[:8]
print(f"🔖 Run ID: {RUN_ID}")

class StatusTracker:
    """Tracks and saves training status to JSON"""
    
    def __init__(self, status_file='status.json'):
        self.status_file = status_file
        self.status = {
            'run_id': RUN_ID,
            'start_time': datetime.now().isoformat(),
            'current_step': 0,
            'current_epoch': 0,
            'best_val_loss': float('inf'),
            'best_val_acc': 0.0,
            'best_step': 0,
            'last_checkpoint': None,
            'last_validation': None,
            'training_loss': 0.0,
            'training_acc': 0.0,
            'samples_per_sec': 0.0,
            'estimated_time_remaining': 'calculating...',
            'last_update': None
        }
        self.running = True
        self.lock = threading.Lock()
        self.update_thread = threading.Thread(target=self._periodic_update)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def update(self, **kwargs):
        """Update status with new values"""
        with self.lock:
            self.status.update(kwargs)
            self.status['last_update'] = datetime.now().isoformat()
    
    def _save_status(self):
        """Save status to JSON file"""
        with self.lock:
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
    
    def _periodic_update(self):
        """Periodically save status to file"""
        while self.running:
            time.sleep(TRAINING_CONFIG['status_update_interval'])
            self._save_status()
    
    def stop(self):
        """Stop the status tracker"""
        self.running = False
        self._save_status()

class ARCDataset(Dataset):
    """Dataset loader for ARC puzzles"""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        split_dir = self.data_dir / split
        if split_dir.exists() and (split_dir / 'all__inputs.npy').exists():
            self.inputs = np.load(split_dir / 'all__inputs.npy')
            self.labels = np.load(split_dir / 'all__labels.npy')
            print(f"Loaded {len(self.inputs)} {split} samples")
        else:
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
    """HRM architecture with H/L dual-loop structure"""
    
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

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(-1)
        one_hot = torch.zeros_like(pred).scatter(-1, target.unsqueeze(-1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(smooth_one_hot * log_probs).sum(dim=-1).mean()
        return loss

class ACTLoss(nn.Module):
    """Adaptive Computation Time loss with label smoothing"""
    
    def __init__(self, halt_penalty=0.01, label_smoothing=0.1):
        super().__init__()
        self.halt_penalty = halt_penalty
        self.task_loss = LabelSmoothingCrossEntropy(label_smoothing)
    
    def forward(self, outputs, targets, halt_probs):
        # Main task loss with label smoothing
        task_loss = self.task_loss(
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

class WarmRestartScheduler:
    """Cosine annealing with warm restarts"""
    
    def __init__(self, optimizer, lr_max, restart_every, lr_min=1e-6):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.restart_every = restart_every
        self.current_step = 0
    
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        cycle_step = self.current_step % self.restart_every
        
        if cycle_step == 0:
            # Restart
            lr = self.lr_max
        else:
            # Cosine annealing
            progress = cycle_step / self.restart_every
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def load_state_dict(self, state_dict):
        self.current_step = state_dict.get('current_step', 0)
    
    def state_dict(self):
        return {'current_step': self.current_step}

def find_latest_checkpoint():
    """Find the latest checkpoint to resume from"""
    step_checkpoints = glob.glob('checkpoints/hrm_arc_step_*.pt')
    if step_checkpoints:
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
        'run_id': RUN_ID
    }
    if metrics:
        state['metrics'] = metrics
    
    torch.save(state, path)
    
    # Save JSON sidecar
    json_path = str(path).replace('.pt', '_info.json')
    info = {
        'run_id': RUN_ID,
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
    """Main training loop with enhanced features"""
    print(f"🚀 Starting Enhanced Nova Training on {DEVICE}")
    print(f"📊 Batch size: {MODEL_CONFIG['batch_size']}, Workers: {DATALOADER_CONFIG['num_workers']}")
    print(f"✨ Optimizations: Label smoothing, LR restarts, Smart validation, Status tracking")
    
    # Initialize status tracker
    status_tracker = StatusTracker()
    
    # Check for dataset
    if Path('../data/arc-aug-500/train/all__inputs.npy').exists():
        data_dir = '../data/arc-aug-500'
        print("Using arc-aug-500 dataset")
    elif Path('../data/arc-aug-100/train/all__inputs.npy').exists():
        data_dir = '../data/arc-aug-100'
        print("Using arc-aug-100 dataset")
    else:
        data_dir = '../data/arc-dummy'
        print("Warning: No dataset found, using dummy data")
    
    # Initialize wandb
    run = wandb.init(
        project="hrm-arc-training",
        name=f"arc-nova-enhanced-{RUN_ID}",
        config={**MODEL_CONFIG, **TRAINING_CONFIG, **DATALOADER_CONFIG, 'run_id': RUN_ID},
        mode="offline"
    )
    
    # Create datasets
    train_dataset = ARCDataset(data_dir, 'train')
    val_dataset = ARCDataset(data_dir, 'test') if Path(f'{data_dir}/test').exists() else None
    
    # Set max_val_batches
    if val_dataset:
        TRAINING_CONFIG['max_val_batches'] = max(1, len(val_dataset) // (MODEL_CONFIG['batch_size'] * 10))
        print(f"Fast validation will use {TRAINING_CONFIG['max_val_batches']} batches (~10% of val set)")
    
    # Create data loaders
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
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # Scheduler with warm restarts
    scheduler = WarmRestartScheduler(
        optimizer,
        lr_max=TRAINING_CONFIG['learning_rate'],
        restart_every=TRAINING_CONFIG['lr_restart_every']
    )
    
    # Load checkpoint if exists
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_step = 0
    last_fast_val_loss = float('inf')
    
    checkpoint_path, checkpoint_type = find_latest_checkpoint()
    if checkpoint_path:
        print(f"📂 Loading {checkpoint_type} checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint_type == 'step':
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
            if 'global_step' in checkpoint:
                global_step = checkpoint.get('global_step', 0)
        
        if Path('checkpoints/hrm_arc_best.pt').exists():
            best_checkpoint = torch.load('checkpoints/hrm_arc_best.pt', map_location='cpu')
            if 'best_val_loss' in best_checkpoint:
                best_val_loss = best_checkpoint['best_val_loss']
            if 'metrics' in best_checkpoint and 'val_accuracy' in best_checkpoint['metrics']:
                best_val_acc = best_checkpoint['metrics']['val_accuracy']
        
        print(f"✅ Resumed from epoch {start_epoch}, step {global_step}")
        print(f"📈 Best val: loss={best_val_loss:.4f}, acc={best_val_acc:.4f}")
    
    # Mixed precision
    scaler = GradScaler() if TRAINING_CONFIG['use_amp'] else None
    
    # Loss function with label smoothing
    criterion = ACTLoss(
        halt_penalty=0.01, 
        label_smoothing=TRAINING_CONFIG['label_smoothing']
    )
    
    # Training metrics
    samples_processed = 0
    training_start = time.time()
    
    print("\n🏃 Starting training...")
    
    for epoch in range(start_epoch, TRAINING_CONFIG['max_epochs']):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAINING_CONFIG['max_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Skip batches if resuming
            if epoch == start_epoch and batch_idx == 0 and global_step > 0:
                skip_batches = global_step % len(train_loader)
                if batch_idx < skip_batches:
                    continue
            
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            
            # Forward pass
            if TRAINING_CONFIG['use_amp']:
                with autocast():
                    outputs, halt_probs = model(inputs)
                    loss, loss_metrics = criterion(outputs, targets, halt_probs)
            else:
                outputs, halt_probs = model(inputs)
                loss, loss_metrics = criterion(outputs, targets, halt_probs)
            
            # Scale loss for accumulation
            loss = loss / TRAINING_CONFIG['gradient_accumulation_steps']
            
            # Backward pass
            if TRAINING_CONFIG['use_amp']:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % TRAINING_CONFIG['gradient_accumulation_steps'] == 0:
                if TRAINING_CONFIG['use_amp']:
                    scaler.unscale_(optimizer)
                
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
            
            # Metrics
            predictions = outputs.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()
            epoch_loss += loss.item() * TRAINING_CONFIG['gradient_accumulation_steps']
            epoch_acc += accuracy
            num_batches += 1
            samples_processed += inputs.size(0)
            
            # Calculate throughput
            elapsed = time.time() - training_start
            samples_per_sec = samples_processed / elapsed
            
            # Update status
            status_tracker.update(
                current_step=global_step,
                current_epoch=epoch,
                training_loss=loss.item() * TRAINING_CONFIG['gradient_accumulation_steps'],
                training_acc=accuracy,
                samples_per_sec=samples_per_sec,
                last_checkpoint=f"step_{global_step}"
            )
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item()*TRAINING_CONFIG['gradient_accumulation_steps']:.4f}",
                'acc': f"{accuracy:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}",
                'samples/s': f"{samples_per_sec:.1f}"
            })
            
            # Log to wandb
            if global_step % 10 == 0:
                wandb.log({
                    'train/loss': loss.item() * TRAINING_CONFIG['gradient_accumulation_steps'],
                    'train/accuracy': accuracy,
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'train/samples_per_sec': samples_per_sec,
                    'global_step': global_step
                })
            
            # Fast validation
            if val_loader and global_step % TRAINING_CONFIG['eval_frequency'] == 0 and global_step > 0:
                model.eval()
                val_loss = 0
                val_acc = 0
                val_batches = 0
                
                with torch.no_grad():
                    for val_idx, val_batch in enumerate(tqdm(val_loader, desc="Fast Validation", leave=False)):
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
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                avg_val_acc = val_acc / val_batches
                
                print(f"\n📊 Fast Val (step {global_step}): loss={avg_val_loss:.4f}, acc={avg_val_acc:.4f}")
                
                wandb.log({
                    'val/fast_loss': avg_val_loss,
                    'val/fast_accuracy': avg_val_acc,
                    'global_step': global_step
                })
                
                # Smart validation: only run full val if fast val improved
                should_run_full = False
                if TRAINING_CONFIG['smart_validation']:
                    if avg_val_loss < last_fast_val_loss * 0.99:  # 1% improvement threshold
                        should_run_full = True
                        print("✨ Fast validation improved! Running full validation...")
                    else:
                        print(f"📊 No improvement (current: {avg_val_loss:.4f}, previous: {last_fast_val_loss:.4f})")
                    last_fast_val_loss = avg_val_loss
                else:
                    # Only if smart validation is disabled
                    if global_step % TRAINING_CONFIG['full_eval_every'] == 0:
                        should_run_full = True
                
                # Full validation
                if should_run_full and val_loader:
                    print("Running full validation...")
                    full_val_loss = 0
                    full_val_acc = 0
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
                            full_val_batches += 1
                    
                    avg_full_val_loss = full_val_loss / full_val_batches
                    avg_full_val_acc = full_val_acc / full_val_batches
                    
                    print(f"📊 Full Val: loss={avg_full_val_loss:.4f}, acc={avg_full_val_acc:.4f}")
                    
                    wandb.log({
                        'val/full_loss': avg_full_val_loss,
                        'val/full_accuracy': avg_full_val_acc,
                        'global_step': global_step
                    })
                    
                    # Save best model
                    if avg_full_val_acc > best_val_acc:
                        best_val_loss = avg_full_val_loss
                        best_val_acc = avg_full_val_acc
                        best_val_step = global_step
                        save_training_state(
                            'checkpoints/hrm_arc_best.pt',
                            epoch, global_step, model, optimizer, scheduler,
                            best_val_loss, MODEL_CONFIG,
                            {'val_accuracy': avg_full_val_acc}
                        )
                        print(f"✅ New best model! acc={avg_full_val_acc:.4f}")
                        
                        # Archive the best model with its score
                        archive_path = f'checkpoints/hrm_arc_best_step{global_step}_acc{avg_full_val_acc*100:.1f}.pt'
                        save_training_state(
                            archive_path,
                            epoch, global_step, model, optimizer, scheduler,
                            best_val_loss, MODEL_CONFIG,
                            {'val_accuracy': avg_full_val_acc}
                        )
                    
                    status_tracker.update(
                        best_val_loss=best_val_loss,
                        best_val_acc=best_val_acc,
                        best_step=best_val_step,
                        last_validation=f"step_{global_step}"
                    )
                
                model.train()
            
            # Checkpoint
            if global_step % TRAINING_CONFIG['checkpoint_frequency'] == 0 and global_step > 0:
                save_training_state(
                    f'checkpoints/hrm_arc_step_{global_step}.pt',
                    epoch, global_step, model, optimizer, scheduler,
                    best_val_loss, MODEL_CONFIG
                )
                print(f"💾 Checkpoint saved at step {global_step}")
            
            global_step += 1
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_acc = epoch_acc / num_batches
        print(f"\nEpoch {epoch+1}: loss={avg_epoch_loss:.4f}, acc={avg_epoch_acc:.4f}")
    
    # Final save
    save_training_state(
        f'checkpoints/hrm_arc_final_{RUN_ID}.pt',
        epoch, global_step, model, optimizer, scheduler,
        best_val_loss, MODEL_CONFIG
    )
    
    status_tracker.stop()
    wandb.finish()
    
    print("\n🎉 Training complete!")
    print(f"Best model: step {best_val_step}, acc={best_val_acc:.4f}")
    print(f"Run ID: {RUN_ID}")

if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok=True)
    
    try:
        train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted! Checkpoint saved.")
        print("💾 Resume by running the script again.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise