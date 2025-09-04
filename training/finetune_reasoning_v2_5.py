#!/usr/bin/env python3
"""
Fine-tune HRM V2.5 - NORMALIZED HALT PREDICTOR
Key fix: Normalize halt predictor output to 0-1.0 range before applying bias
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

# Import base classes
sys.path.append('.')
from finetune_reasoning_fixed_v2 import ARCOriginalDataset, AdaptiveReasoningLoss, MODEL_CONFIG, TRAINING_CONFIG, PositionalEncoding

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HierarchicalReasoningModuleV25(nn.Module):
    """HRM V2.5 with normalized halt predictor"""
    
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
        
        # NORMALIZED halt predictor - outputs raw logits that we'll normalize
        self.halt_predictor = nn.Linear(config['hidden_size'] * 2, 1)
        
        # Cycle embedding
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
        
        halt_probs = []
        cumulative_halt = torch.zeros(batch_size, 1).to(x.device)
        
        # Reasoning cycles with normalized halt predictor
        for cycle in range(max_cycles):
            # Add cycle information
            cycle_emb = self.cycle_embedding(torch.tensor([cycle], device=x.device))
            cycle_emb = cycle_emb.expand(batch_size, seq_len, -1)
            
            # H-level processing
            h_state = h_state + 0.1 * cycle_emb
            for h_layer in self.h_layers:
                h_state = h_layer(h_state)
            h_state = self.h_norm(h_state)
            
            # L-level processing
            l_state = l_state + self.h_to_l(h_state)
            for l_layer in self.l_layers:
                l_state = l_layer(l_state)
            l_state = self.l_norm(l_state)
            
            # L to H feedback
            h_state = h_state + self.l_to_h(l_state)
            
            # Compute halting probability with normalization
            combined = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)
            halt_logit = self.halt_predictor(combined)
            
            # KEY FIX: Normalize halt_logit to reasonable range before bias
            # Use tanh to bound to [-1, 1], then shift to [0, 1]
            normalized_logit = (torch.tanh(halt_logit) + 1.0) / 2.0  # Now in [0, 1]
            
            # Convert back to logit space for bias application
            # logit(p) = log(p / (1-p))
            eps = 1e-6
            normalized_logit = torch.clamp(normalized_logit, eps, 1-eps)
            normalized_logit_space = torch.log(normalized_logit / (1 - normalized_logit))
            
            # Now apply cycle bias in normalized logit space
            cycle_bias = -0.2 + (cycle / (max_cycles - 1)) * 1.1
            biased_logit = normalized_logit_space + cycle_bias
            halt_prob = torch.sigmoid(biased_logit)
            halt_probs.append(halt_prob)
            
            # Update cumulative halt probability
            cumulative_halt = cumulative_halt + halt_prob
            
            # Adaptive early stopping
            if cycle >= 3:
                if cumulative_halt.mean() > 1.0 or halt_prob.mean() > 0.95:
                    break
        
        # Final output
        output = self.output(l_state)
        return output, halt_probs

def finetune_v2_5(dataset_name='agi-1'):
    """V2.5: Normalized halt predictor"""
    
    print(f"🎯 Fine-tuning V2.5 - NORMALIZED HALT PREDICTOR")
    print(f"🔧 Fix: Normalize halt predictor output before applying bias")
    print("=" * 60)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/hrm_arc_best.pt'
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Create model
    model = HierarchicalReasoningModuleV25(MODEL_CONFIG).to(DEVICE)
    
    # Load weights (halt predictor will be fresh)
    model_state = model.state_dict()
    checkpoint_state = checkpoint['model_state_dict']
    
    loaded_keys = []
    for key in checkpoint_state:
        if key in model_state and model_state[key].shape == checkpoint_state[key].shape:
            if 'halt_predictor' not in key:  # Fresh halt predictor
                model_state[key] = checkpoint_state[key]
                loaded_keys.append(key)
    
    model.load_state_dict(model_state)
    print(f"✅ Loaded {len(loaded_keys)} layers, fresh halt predictor")
    
    # Test initial behavior
    print("\n🔍 Testing V2.5 normalized halt behavior...")
    model.eval()
    test_input = torch.randint(0, 10, (2, 900)).to(DEVICE)
    with torch.no_grad():
        _, test_halt_probs = model(test_input)
        halt_values = [p.mean().item() for p in test_halt_probs]
        print(f"  V2.5 halt probs: {[f'{v:.3f}' for v in halt_values]}")
        print(f"  Cycles used: {len(test_halt_probs)}")
        
        # Verify the bias math is working
        for i, prob in enumerate(halt_values):
            expected_min = 1.0 / (1.0 + math.exp(-(-0.2 + (i / 19) * 1.1)))
            print(f"  Cycle {i}: prob={prob:.3f}, min_expected={expected_min:.3f}")
    model.train()
    
    # Dataset setup
    data_path = Path('../dataset/raw-data/ARC-AGI/data') if dataset_name == 'agi-1' else Path('../arc-agi-2/data')
    output_suffix = 'agi1' if dataset_name == 'agi-1' else 'agi2'
    
    train_dataset = ARCOriginalDataset(data_path, 'training')
    train_loader = DataLoader(
        train_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    scaler = GradScaler() if TRAINING_CONFIG['use_amp'] else None
    criterion = AdaptiveReasoningLoss()
    
    # Initialize wandb
    run = wandb.init(
        project="hrm-reasoning-finetune-v2",
        name=f"reasoning-{output_suffix}-v2.5-normalized-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={**MODEL_CONFIG, **TRAINING_CONFIG, 'dataset': dataset_name, 'version': 'v2.5-normalized'},
        mode="offline"
    )
    
    # Training loop
    print("\n🏃 Starting V2.5 fine-tuning (Normalized Halt Predictor)...")
    model.train()
    
    global_step = 0
    accumulated_loss = 0
    accumulated_cycles = 0
    cycle_distribution = []
    
    progress_bar = tqdm(total=TRAINING_CONFIG['num_steps'], desc="Fine-tuning V2.5")
    
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
                    # Cycle distribution stats
                    if len(cycle_distribution) > 100:
                        recent_cycles = cycle_distribution[-100:]
                        cycle_stats = {
                            'cycles_mean': np.mean(recent_cycles),
                            'cycles_std': np.std(recent_cycles),
                            'cycles_min': np.min(recent_cycles),
                            'cycles_max': np.max(recent_cycles),
                            'cycles_unique': len(set(recent_cycles))
                        }
                    else:
                        cycle_stats = {}
                    
                    wandb.log({
                        'loss': avg_loss,
                        'task_loss': metrics['task_loss'],
                        'act_loss': metrics['act_loss'],
                        'reasoning_cycles': avg_cycles,
                        'final_halt_prob': metrics['final_halt_prob'],
                        'step': global_step,
                        **cycle_stats
                    })
                
                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'cycles': f"{avg_cycles:.1f}",
                    'halt_p': f"{metrics['final_halt_prob']:.3f}",
                    'unique': f"{len(set(cycle_distribution[-50:])) if len(cycle_distribution) >= 50 else len(set(cycle_distribution))}"
                })
                
                accumulated_loss = 0
                accumulated_cycles = 0
            
            # Checkpoint
            if global_step % TRAINING_CONFIG['checkpoint_frequency'] == 0 and global_step > 0:
                save_path = f'checkpoints/hrm_reasoning_{dataset_name}_v2_5_step_{global_step}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'config': MODEL_CONFIG,
                    'dataset': dataset_name,
                    'version': 'v2.5-normalized'
                }, save_path)
                print(f"\n💾 V2.5 Checkpoint: {save_path}")
                
                # Test adaptive behavior
                model.eval()
                with torch.no_grad():
                    _, test_halt = model(test_input[:2])
                    cycle_count = len(test_halt)
                    halt_values = [p.mean().item() for p in test_halt]
                    unique_counts = len(set(cycle_distribution[-200:]))
                    
                    print(f"  Cycles at step {global_step}: {cycle_count}")
                    print(f"  Halt progression: {[f'{v:.3f}' for v in halt_values]}")
                    print(f"  Cycle variety (last 200): {unique_counts} different counts")
                model.train()
            
            global_step += 1
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Final save and analysis
    final_path = f'checkpoints/hrm_reasoning_{dataset_name}_v2_5_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'config': MODEL_CONFIG,
        'dataset': dataset_name,
        'version': 'v2.5-normalized'
    }, final_path)
    
    wandb.finish()
    
    print(f"\n✅ V2.5 Fine-tuning complete!")
    print(f"💾 Final model: {final_path}")
    
    # Final cycle analysis
    if cycle_distribution:
        unique_cycles = set(cycle_distribution)
        print(f"📊 Cycle Distribution Analysis:")
        print(f"   Mean: {np.mean(cycle_distribution):.1f} cycles")
        print(f"   Range: {min(cycle_distribution)}-{max(cycle_distribution)}")
        print(f"   Unique counts: {len(unique_cycles)}")
        print(f"   Distribution: {dict(zip(*np.unique(cycle_distribution, return_counts=True)))}")
    
    return final_path

if __name__ == "__main__":
    # Start fresh test
    test_input = torch.randint(0, 10, (2, 900)).to(DEVICE)
    finetune_v2_5('agi-1')