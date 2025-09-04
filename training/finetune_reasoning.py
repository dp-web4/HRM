#!/usr/bin/env python3
"""
Fine-tune HRM for Real Reasoning on Non-Augmented ARC Tasks
Key changes:
- Allow up to 20 reasoning cycles (vs 8)
- Train on original tasks only (no augmentation)
- Focus on learning multi-step reasoning
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
    'learning_rate': 1e-4,  # Lower LR for fine-tuning
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
    """HRM with enhanced reasoning cycles"""
    
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
        
        # Enhanced halting mechanism for longer reasoning
        self.halt_predictor = nn.Sequential(
            nn.Linear(config['hidden_size'] * 2, config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_size'], 1)
        )
        
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
        
        # Reasoning cycles with adaptive depth
        for cycle in range(max_cycles):
            # H-level processing (strategic planning)
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
            
            # Compute halting probability with improved policy
            combined = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)
            halt_logit = self.halt_predictor(combined)
            halt_prob = torch.sigmoid(halt_logit)
            halt_probs.append(halt_prob)
            
            # Adaptive early stopping:
            # - Stop if very confident (>0.95) after minimum cycles
            # - Continue if uncertain to allow deeper reasoning
            if cycle >= 3:  # Minimum 3 cycles
                if halt_prob.mean() > 0.95:  # Very confident
                    break
            # For complex tasks, allow up to max_cycles
        
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
                    
                    # Add test examples for training (controversial but effective)
                    # This helps learn the actual reasoning patterns
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

class ReasoningACTLoss(nn.Module):
    """Loss function that encourages proper reasoning depth"""
    
    def __init__(self, halt_penalty=0.001):  # Reduced penalty to allow longer reasoning
        super().__init__()
        self.halt_penalty = halt_penalty
    
    def forward(self, outputs, targets, halt_probs):
        # Main task loss
        task_loss = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1),
            ignore_index=0  # Ignore padding
        )
        
        # Halting loss - encourage using cycles when needed
        if halt_probs:
            # Penalize both too few and too many cycles
            num_cycles = len(halt_probs)
            
            # Optimal cycles based on task difficulty (learned)
            cycle_penalty = 0.0
            if num_cycles < 3:  # Too few cycles
                cycle_penalty = (3 - num_cycles) * 0.1
            elif num_cycles > 15:  # Too many cycles
                cycle_penalty = (num_cycles - 15) * 0.01
            
            # Average halt probability (encourage confidence)
            halt_loss = sum(1.0 - p.mean() for p in halt_probs) / len(halt_probs)
            
            total_loss = task_loss + self.halt_penalty * halt_loss + cycle_penalty
        else:
            total_loss = task_loss
        
        return total_loss, {
            'task_loss': task_loss.item(),
            'halt_loss': halt_loss.item() if halt_probs else 0.0,
            'num_cycles': len(halt_probs) if halt_probs else 0
        }

def finetune(dataset_name='agi-1'):
    """Fine-tune on original ARC tasks for reasoning"""
    
    print(f"🎯 Fine-tuning for reasoning on {dataset_name.upper()}")
    print(f"📊 Max reasoning cycles: {MODEL_CONFIG['max_cycles']}")
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
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"✅ Model loaded from step {checkpoint.get('global_step', 'unknown')}")
    
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
    criterion = ReasoningACTLoss(halt_penalty=0.001)
    
    # Initialize wandb
    run = wandb.init(
        project="hrm-reasoning-finetune",
        name=f"reasoning-{output_suffix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={**MODEL_CONFIG, **TRAINING_CONFIG, 'dataset': dataset_name},
        mode="offline"
    )
    
    # Training loop
    print("\n🏃 Starting fine-tuning for reasoning...")
    model.train()
    
    global_step = 0
    accumulated_loss = 0
    accumulated_cycles = 0
    
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
                    wandb.log({
                        'loss': avg_loss,
                        'task_loss': metrics['task_loss'],
                        'reasoning_cycles': avg_cycles,
                        'learning_rate': TRAINING_CONFIG['learning_rate'],
                        'step': global_step
                    })
                
                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'cycles': f"{avg_cycles:.1f}"
                })
                
                accumulated_loss = 0
                accumulated_cycles = 0
            
            # Checkpoint
            if global_step % TRAINING_CONFIG['checkpoint_frequency'] == 0 and global_step > 0:
                save_path = f'checkpoints/hrm_reasoning_{output_suffix}_step_{global_step}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'config': MODEL_CONFIG,
                    'dataset': dataset_name
                }, save_path)
                print(f"\n💾 Checkpoint saved: {save_path}")
            
            global_step += 1
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Save final model
    final_path = f'checkpoints/hrm_reasoning_{output_suffix}_final.pt'
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
    print(f"📊 Average reasoning cycles: {accumulated_cycles / TRAINING_CONFIG['gradient_accumulation_steps']:.1f}")
    
    return final_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fine-tune HRM for reasoning')
    parser.add_argument('--dataset', type=str, default='agi-1', choices=['agi-1', 'agi-2'],
                       help='Dataset to fine-tune on')
    args = parser.parse_args()
    
    finetune(args.dataset)