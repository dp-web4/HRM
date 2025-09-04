#!/usr/bin/env python3
"""
Fine-tune HRM V2.4 - RESET HALT PREDICTOR
Key fix: Reinitialize halt predictor to fix extreme negative logits
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

# Import the model from our fixed version
sys.path.append('.')
from finetune_reasoning_fixed_v2 import HierarchicalReasoningModule, ARCOriginalDataset, AdaptiveReasoningLoss, MODEL_CONFIG, TRAINING_CONFIG

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reset_halt_predictor(model):
    """Reinitialize just the halt predictor to fix extreme negative logits"""
    print("🔧 Reinitializing halt predictor...")
    
    # Save current halt predictor values for debugging
    with torch.no_grad():
        old_weight = model.halt_predictor.weight.clone()
        old_bias = model.halt_predictor.bias.clone()
        
        print(f"  Old weights range: [{old_weight.min().item():.3f}, {old_weight.max().item():.3f}]")
        print(f"  Old bias: {old_bias.item():.3f}")
        
        # Reinitialize with careful scaling
        # We want initial outputs around 0, not extreme negatives
        fan_in = model.halt_predictor.in_features
        bound = 1 / math.sqrt(fan_in) * 0.1  # Much gentler initialization
        
        model.halt_predictor.weight.uniform_(-bound, bound)
        model.halt_predictor.bias.uniform_(-0.1, 0.1)  # Small bias range
        
        print(f"  New weights range: [{model.halt_predictor.weight.min().item():.3f}, {model.halt_predictor.weight.max().item():.3f}]")
        print(f"  New bias: {model.halt_predictor.bias.item():.3f}")
    
    return model

def finetune_v2_4(dataset_name='agi-1'):
    """V2.4: Reset halt predictor and retrain"""
    
    print(f"🎯 Fine-tuning V2.4 - RESET HALT PREDICTOR")
    print(f"🔧 Fix: Reinitialize halt predictor to eliminate extreme negatives")
    print("=" * 60)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/hrm_arc_best.pt'
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Create and load model
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(DEVICE)
    model_state = model.state_dict()
    checkpoint_state = checkpoint['model_state_dict']
    
    # Load all weights except halt predictor
    for key in checkpoint_state:
        if key in model_state and model_state[key].shape == checkpoint_state[key].shape:
            if 'halt_predictor' not in key:  # Skip halt predictor
                model_state[key] = checkpoint_state[key]
                print(f"  ✓ Loaded: {key}")
            else:
                print(f"  🔧 Skipped (will reinit): {key}")
    
    model.load_state_dict(model_state)
    
    # Reset halt predictor
    model = reset_halt_predictor(model)
    
    # Test halt behavior
    print("\n🔍 Testing reset halt behavior...")
    model.eval()
    test_input = torch.randint(0, 10, (2, 900)).to(DEVICE)
    with torch.no_grad():
        _, test_halt_probs = model(test_input)
        halt_values = [p.mean().item() for p in test_halt_probs]
        print(f"  Reset halt probs: {[f'{v:.3f}' for v in halt_values]}")
        print(f"  Cycles used: {len(test_halt_probs)}")
    model.train()
    
    # Dataset and training setup (same as before)
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
        name=f"reasoning-{output_suffix}-v2.4-reset-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={**MODEL_CONFIG, **TRAINING_CONFIG, 'dataset': dataset_name},
        mode="offline"
    )
    
    # Training loop
    print("\n🏃 Starting V2.4 fine-tuning (Reset Halt Predictor)...")
    model.train()
    
    global_step = 0
    accumulated_loss = 0
    accumulated_cycles = 0
    
    progress_bar = tqdm(total=TRAINING_CONFIG['num_steps'], desc="Fine-tuning V2.4")
    
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
                        'act_loss': metrics['act_loss'],
                        'reasoning_cycles': avg_cycles,
                        'final_halt_prob': metrics['final_halt_prob'],
                        'step': global_step,
                    })
                
                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'cycles': f"{avg_cycles:.1f}",
                    'halt_p': f"{metrics['final_halt_prob']:.3f}"
                })
                
                accumulated_loss = 0
                accumulated_cycles = 0
            
            # Checkpoint with halt behavior check
            if global_step % TRAINING_CONFIG['checkpoint_frequency'] == 0 and global_step > 0:
                save_path = f'checkpoints/hrm_reasoning_{output_suffix}_v2_4_step_{global_step}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'config': MODEL_CONFIG,
                    'dataset': dataset_name
                }, save_path)
                print(f"\n💾 Checkpoint saved: {save_path}")
                
                # Test halt behavior
                model.eval()
                with torch.no_grad():
                    _, test_halt = model(test_input)
                    cycle_count = len(test_halt)
                    halt_values = [p.mean().item() for p in test_halt]
                    print(f"  Cycles at step {global_step}: {cycle_count}")
                    print(f"  Halt probs: {[f'{v:.3f}' for v in halt_values[:5]]} ... {[f'{v:.3f}' for v in halt_values[-3:]]}")
                    print(f"  Last 3 cycles should be >0.5: {halt_values[-3:] if len(halt_values) >= 3 else halt_values}")
                model.train()
            
            global_step += 1
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Save final model
    final_path = f'checkpoints/hrm_reasoning_{output_suffix}_v2_4_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'config': MODEL_CONFIG,
        'dataset': dataset_name
    }, final_path)
    
    wandb.finish()
    
    print(f"\n✅ V2.4 Fine-tuning complete!")
    print(f"💾 Final model saved: {final_path}")
    
    return final_path

if __name__ == "__main__":
    finetune_v2_4('agi-1')