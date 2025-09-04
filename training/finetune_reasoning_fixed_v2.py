#!/usr/bin/env python3
"""
Fine-tune HRM for Real Reasoning on Non-Augmented ARC Tasks - FIXED V2
Key changes:
- Allow up to 20 reasoning cycles (vs 8)
- Train on original tasks only (no augmentation)
- FIXED V2: Proper halt predictor that actually increases probability over cycles
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
from typing import Dict, Any, Tuple, Optional, List
import wandb
from datetime import datetime

# Configuration
MODEL_CONFIG = {
    'batch_size': 4,  # Smaller batch for fine-tuning
    'seq_len': 900,
    'vocab_size': 12,
    'hidden_size': 256,
    'num_heads': 8,
    'num_h_layers': 4,
    'num_l_layers': 3,
    'dropout': 0.1,
    'max_cycles': 20,  # INCREASED from 8 to allow deeper reasoning
}

TRAINING_CONFIG = {
    'learning_rate': 5e-5,  # Lower LR for fine-tuning
    'num_steps': 10000,  # Fixed 10K steps as requested
    'gradient_clip': 1.0,
    'weight_decay': 0.01,
    'use_amp': True,
    'checkpoint_frequency': 1000,
    'gradient_accumulation_steps': 4,  # Effective batch = 16
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    """HRM with PROPERLY FIXED extended reasoning cycles"""
    
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
        
        # Halt predictor - keep simple but add cycle awareness
        self.halt_predictor = nn.Linear(config['hidden_size'] * 2, 1)
        
        # Cycle embedding to help the model know where it is in the reasoning process
        self.cycle_embedding = nn.Embedding(config['max_cycles'], config['hidden_size'])
        
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
        
        # Store halting probabilities and reasoning depth
        halt_probs = []
        
        # Track cumulative halt probability for proper ACT
        cumulative_halt = torch.zeros(batch_size, 1).to(x.device)
        
        # Reasoning cycles with adaptive depth
        for cycle in range(max_cycles):
            # Add cycle information
            cycle_emb = self.cycle_embedding(torch.tensor([cycle], device=x.device))
            cycle_emb = cycle_emb.expand(batch_size, seq_len, -1)
            
            # H-level processing (strategic planning) with cycle awareness
            h_state = h_state + 0.1 * cycle_emb  # Gentle cycle conditioning
            for h_layer in self.h_layers:
                h_state = h_layer(h_state)
            h_state = self.h_norm(h_state)
            
            # L-level processing (tactical execution)
            l_state = l_state + self.h_to_l(h_state)
            for l_layer in self.l_layers:
                l_state = l_layer(l_state)
            l_state = self.l_norm(l_state)
            
            # L to H feedback
            h_state = h_state + self.l_to_h(l_state)
            
            # Compute halting probability
            combined = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)
            halt_logit = self.halt_predictor(combined)
            
            # FIXED V2.3: Ultra-gentle halting mechanism
            # Progresses from -0.2 to +0.9 over 20 cycles (user-specified)
            cycle_bias = -0.2 + (cycle / (max_cycles - 1)) * 1.1
            
            # Apply bias and compute probability
            biased_logit = halt_logit + cycle_bias
            halt_prob = torch.sigmoid(biased_logit)
            halt_probs.append(halt_prob)
            
            # Debug logging every 100 steps
            if cycle == 19 and hasattr(self, '_debug_step_count'):
                self._debug_step_count += 1
                if self._debug_step_count % 100 == 0:
                    print(f"DEBUG: halt_logit={halt_logit.mean().item():.3f}, bias={cycle_bias:.3f}, biased_logit={biased_logit.mean().item():.3f}, halt_prob={halt_prob.mean().item():.3f}")
            elif not hasattr(self, '_debug_step_count'):
                self._debug_step_count = 0
            
            # Update cumulative halt probability
            cumulative_halt = cumulative_halt + halt_prob
            
            # Adaptive early stopping with proper ACT
            if cycle >= 3:  # Minimum 3 cycles
                # Stop if cumulative probability exceeds threshold
                # OR if we're very confident at this specific cycle
                if cumulative_halt.mean() > 1.0 or halt_prob.mean() > 0.95:
                    break
        
        # Final output
        output = self.output(l_state)
        
        return output, halt_probs

class ARCOriginalDataset(Dataset):
    """Dataset for original (non-augmented) ARC tasks"""
    
    def __init__(self, data_dir, split='training'):
        self.data_dir = Path(data_dir)
        self.split = split
        self.tasks = []
        self.examples = []
        
        # Load all task files
        task_dir = self.data_dir / split
        if task_dir.exists():
            for task_file in sorted(task_dir.glob('*.json')):
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                    task_id = task_file.stem
                    
                    # Add all training examples
                    for example in task_data['train']:
                        self.examples.append({
                            'task_id': task_id,
                            'input': example['input'],
                            'output': example['output'],
                            'type': 'train'
                        })
                    
                    # Add test examples for training
                    for example in task_data['test']:
                        self.examples.append({
                            'task_id': task_id,
                            'input': example['input'],
                            'output': example['output'],
                            'type': 'test'
                        })
        
        print(f"Loaded {len(self.examples)} examples from {len(set(e['task_id'] for e in self.examples))} tasks")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert grids to tensors
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Pad to 30x30
        input_tensor = self.grid_to_tensor(input_grid)
        output_tensor = self.grid_to_tensor(output_grid)
        
        return {
            'input': input_tensor,
            'target': output_tensor,
            'task_id': example['task_id']
        }
    
    def grid_to_tensor(self, grid, max_size=30):
        """Convert grid to padded tensor"""
        h, w = grid.shape
        padded = np.zeros((max_size, max_size), dtype=np.int64)
        padded[:h, :w] = grid
        return torch.from_numpy(padded.flatten())

class AdaptiveReasoningLoss(nn.Module):
    """Loss function that properly encourages adaptive reasoning depth"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, targets, halt_probs):
        # Main task loss
        task_loss = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1),
            ignore_index=0  # Ignore padding
        )
        
        # Adaptive computation loss
        if halt_probs:
            num_cycles = len(halt_probs)
            
            # Compute ACT loss - penalize total computation
            # But allow the model to use cycles as needed
            act_loss = 0.0
            cumulative = 0.0
            
            for i, p in enumerate(halt_probs):
                # Accumulate halt probabilities
                cumulative += p.mean()
                
                # Penalize excessive computation
                # But scale penalty based on cycle number
                # Early cycles: small penalty (exploration is ok)
                # Late cycles: larger penalty (should have decided by now)
                cycle_weight = (i + 1) / len(halt_probs)
                act_loss += cycle_weight * p.mean() * 0.01
            
            # Also penalize if we never become confident
            if halt_probs[-1].mean() < 0.5 and num_cycles == 20:
                # Model used all cycles but still not confident
                # This suggests it's lost/confused
                act_loss += 0.1
            
            # Encourage using 5-12 cycles on average
            # Penalize too few (<3) or too many (>15)
            if num_cycles < 3:
                act_loss += 0.2
            elif num_cycles > 15:
                act_loss += 0.05 * (num_cycles - 15)
            
            total_loss = task_loss + act_loss
        else:
            total_loss = task_loss
            act_loss = 0.0
        
        return total_loss, {
            'task_loss': task_loss.item(),
            'act_loss': act_loss if isinstance(act_loss, float) else act_loss.item(),
            'num_cycles': len(halt_probs) if halt_probs else 0,
            'final_halt_prob': halt_probs[-1].mean().item() if halt_probs else 0.0
        }

def finetune(dataset_name='agi-1'):
    """Fine-tune on original ARC tasks for reasoning"""
    
    print(f"🎯 Fine-tuning for reasoning on {dataset_name.upper()} - FIXED V2.3")
    print(f"📊 Max reasoning cycles: {MODEL_CONFIG['max_cycles']}")
    print("✅ Halt predictor with ULTRA-GENTLE bias (-0.2 to +0.9)")
    print("✅ Adaptive computation - minimal intervention")
    print("=" * 60)
    
    # Determine dataset path
    if dataset_name == 'agi-1':
        data_path = Path('../dataset/raw-data/ARC-AGI/data')
        output_suffix = 'agi1'
    elif dataset_name == 'agi-2':
        data_path = Path('../arc-agi-2/data')
        output_suffix = 'agi2'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load best checkpoint
    checkpoint_path = 'checkpoints/hrm_arc_best.pt'
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Create model with extended reasoning
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(DEVICE)
    
    # Load weights carefully
    model_state = model.state_dict()
    checkpoint_state = checkpoint['model_state_dict']
    
    # Load all matching keys
    for key in checkpoint_state:
        if key in model_state and model_state[key].shape == checkpoint_state[key].shape:
            model_state[key] = checkpoint_state[key]
            print(f"  ✓ Loaded: {key}")
        else:
            print(f"  ✗ Skipped: {key} (shape mismatch or not found)")
    
    model.load_state_dict(model_state)
    print(f"✅ Model loaded from step {checkpoint.get('global_step', 'unknown')}")
    
    # Test initial halt behavior
    print("\n🔍 Testing initial halt behavior...")
    model.eval()
    test_input = torch.randint(0, 10, (2, 900)).to(DEVICE)
    with torch.no_grad():
        _, test_halt_probs = model(test_input)
        halt_values = [p.mean().item() for p in test_halt_probs]
        print(f"  Initial halt probs: {[f'{v:.3f}' for v in halt_values[:10]]}")
        print(f"  Cycles used: {len(test_halt_probs)}")
        print(f"  Pattern: {'INCREASING' if halt_values[-1] > halt_values[0] else 'DECREASING'} ✓")
    model.train()
    
    # Create dataset
    train_dataset = ARCOriginalDataset(data_path, 'training')
    train_loader = DataLoader(
        train_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Optimizer for fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # Mixed precision
    scaler = GradScaler() if TRAINING_CONFIG['use_amp'] else None
    
    # Loss function
    criterion = AdaptiveReasoningLoss()
    
    # Initialize wandb
    run = wandb.init(
        project="hrm-reasoning-finetune-v2",
        name=f"reasoning-{output_suffix}-v2.3-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={**MODEL_CONFIG, **TRAINING_CONFIG, 'dataset': dataset_name},
        mode="offline"
    )
    
    # Training loop
    print("\n🏃 Starting fine-tuning for reasoning (V2 - Proper ACT)...")
    model.train()
    
    global_step = 0
    accumulated_loss = 0
    accumulated_cycles = 0
    cycle_distribution = []  # Track how many cycles are used
    
    progress_bar = tqdm(total=TRAINING_CONFIG['num_steps'], desc="Fine-tuning")
    
    while global_step < TRAINING_CONFIG['num_steps']:
        for batch in train_loader:
            if global_step >= TRAINING_CONFIG['num_steps']:
                break
            
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            
            # Forward pass
            if TRAINING_CONFIG['use_amp']:
                with autocast():
                    outputs, halt_probs = model(inputs)
                    loss, metrics = criterion(outputs, targets, halt_probs)
            else:
                outputs, halt_probs = model(inputs)
                loss, metrics = criterion(outputs, targets, halt_probs)
            
            # Track cycle usage
            cycle_distribution.append(metrics['num_cycles'])
            
            # Scale for gradient accumulation
            loss = loss / TRAINING_CONFIG['gradient_accumulation_steps']
            accumulated_loss += loss.item()
            accumulated_cycles += metrics['num_cycles']
            
            # Backward pass
            if TRAINING_CONFIG['use_amp']:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (global_step + 1) % TRAINING_CONFIG['gradient_accumulation_steps'] == 0:
                if TRAINING_CONFIG['use_amp']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['gradient_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['gradient_clip'])
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Log metrics
                avg_loss = accumulated_loss
                avg_cycles = accumulated_cycles / TRAINING_CONFIG['gradient_accumulation_steps']
                
                if global_step % 10 == 0:
                    # Compute cycle distribution stats
                    if len(cycle_distribution) > 100:
                        recent_cycles = cycle_distribution[-100:]
                        cycle_stats = {
                            'cycles_mean': np.mean(recent_cycles),
                            'cycles_std': np.std(recent_cycles),
                            'cycles_min': np.min(recent_cycles),
                            'cycles_max': np.max(recent_cycles),
                        }
                    else:
                        cycle_stats = {}
                    
                    wandb.log({
                        'loss': avg_loss,
                        'task_loss': metrics['task_loss'],
                        'act_loss': metrics['act_loss'],
                        'reasoning_cycles': avg_cycles,
                        'final_halt_prob': metrics['final_halt_prob'],
                        'learning_rate': TRAINING_CONFIG['learning_rate'],
                        'step': global_step,
                        **cycle_stats
                    })
                
                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'cycles': f"{avg_cycles:.1f}",
                    'halt_p': f"{metrics['final_halt_prob']:.3f}"
                })
                
                accumulated_loss = 0
                accumulated_cycles = 0
            
            # Checkpoint
            if global_step % TRAINING_CONFIG['checkpoint_frequency'] == 0 and global_step > 0:
                save_path = f'checkpoints/hrm_reasoning_{output_suffix}_v2_step_{global_step}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'config': MODEL_CONFIG,
                    'dataset': dataset_name
                }, save_path)
                print(f"\n💾 Checkpoint saved: {save_path}")
                
                # Test halt behavior periodically
                model.eval()
                with torch.no_grad():
                    _, test_halt = model(test_input)
                    cycle_count = len(test_halt)
                    halt_values = [p.mean().item() for p in test_halt]
                    print(f"  Cycles at step {global_step}: {cycle_count}")
                    print(f"  Halt probs: {[f'{v:.3f}' for v in halt_values[:5]]} ... {[f'{v:.3f}' for v in halt_values[-3:]]}")
                    print(f"  Pattern: {'INCREASING ✓' if halt_values[-1] > halt_values[0] else 'DECREASING ✗'}")
                model.train()
            
            global_step += 1
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Save final model
    final_path = f'checkpoints/hrm_reasoning_{output_suffix}_v2_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'config': MODEL_CONFIG,
        'dataset': dataset_name
    }, final_path)
    
    wandb.finish()
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"💾 Final model saved: {final_path}")
    print(f"📊 Cycle usage statistics:")
    if cycle_distribution:
        print(f"   Mean: {np.mean(cycle_distribution):.1f} cycles")
        print(f"   Std: {np.std(cycle_distribution):.1f}")
        print(f"   Range: {np.min(cycle_distribution)}-{np.max(cycle_distribution)}")
    
    return final_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tune HRM for reasoning (FIXED V2)')
    parser.add_argument('--dataset', type=str, default='agi-1', choices=['agi-1', 'agi-2'],
                       help='Dataset to fine-tune on')
    args = parser.parse_args()
    
    finetune(args.dataset)