#!/usr/bin/env python3
"""
Fine-tune HRM V2.6 - EFFICIENCY-REWARDING LOSS
Key fix: Reward efficient correctness, not just correctness
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
from finetune_reasoning_fixed_v2 import HierarchicalReasoningModule, ARCOriginalDataset, MODEL_CONFIG, TRAINING_CONFIG

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EfficiencyRewardingLoss(nn.Module):
    """Loss that rewards efficient correctness"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, targets, halt_probs):
        batch_size = outputs.size(0)
        
        # Compute per-example accuracy
        predicted = outputs.argmax(dim=-1)
        targets_flat = targets.reshape(batch_size, -1)
        predicted_flat = predicted.reshape(batch_size, -1)
        
        # Accuracy per example (ignoring padding tokens)
        correct_per_example = []
        for i in range(batch_size):
            # Only count non-padding positions
            mask = targets_flat[i] != 0
            if mask.sum() > 0:
                correct = (predicted_flat[i][mask] == targets_flat[i][mask]).float().mean()
                correct_per_example.append(correct.item())
            else:
                correct_per_example.append(0.0)
        
        correct_per_example = torch.tensor(correct_per_example, device=outputs.device)
        
        # Base task loss
        task_loss = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1),
            ignore_index=0,
            reduction='none'
        ).reshape(batch_size, -1).mean(dim=1)
        
        # Cycle efficiency analysis
        num_cycles = len(halt_probs)
        
        # EFFICIENCY REWARD SYSTEM
        efficiency_adjustment = torch.zeros_like(task_loss)
        
        for i in range(batch_size):
            accuracy = correct_per_example[i]
            
            if accuracy > 0.8:  # Correct answer (threshold for "got it right")
                # REWARD for efficient correctness
                if num_cycles <= 6:
                    efficiency_adjustment[i] = -0.5  # Big efficiency bonus
                elif num_cycles <= 10:
                    efficiency_adjustment[i] = -0.2  # Medium bonus
                elif num_cycles <= 15:
                    efficiency_adjustment[i] = -0.1  # Small bonus
                else:
                    # Used too many cycles even though correct
                    efficiency_adjustment[i] = +0.1 * (num_cycles - 15)
                    
            elif accuracy > 0.3:  # Partial credit
                # Mild reward for reasonable efficiency
                if num_cycles <= 8:
                    efficiency_adjustment[i] = -0.1
                elif num_cycles > 16:
                    efficiency_adjustment[i] = +0.05
                    
            else:  # Wrong answer
                # No efficiency considerations - just standard task loss
                # But slight penalty for using max cycles and still being wrong
                if num_cycles >= 18:
                    efficiency_adjustment[i] = +0.2  # "You used max cycles and still failed"
        
        # Apply efficiency adjustments
        adjusted_task_loss = task_loss + efficiency_adjustment
        
        # Minimal ACT loss - just prevent immediate stopping
        act_loss = 0.0
        if halt_probs and num_cycles < 3:
            act_loss = 0.1  # Mild penalty for stopping too early
        
        total_loss = adjusted_task_loss.mean() + act_loss
        
        # Compute efficiency metrics
        avg_accuracy = correct_per_example.mean().item()
        efficiency_bonus = -efficiency_adjustment[efficiency_adjustment < 0].sum().item()
        efficiency_penalty = efficiency_adjustment[efficiency_adjustment > 0].sum().item()
        
        return total_loss, {
            'task_loss': task_loss.mean().item(),
            'efficiency_bonus': efficiency_bonus,
            'efficiency_penalty': efficiency_penalty, 
            'act_loss': act_loss,
            'num_cycles': num_cycles,
            'accuracy': avg_accuracy,
            'final_halt_prob': halt_probs[-1].mean().item() if halt_probs else 0.0
        }

def reset_halt_predictor_gentle(model):
    """Reset halt predictor with even gentler initialization"""
    print("🔧 Gentle halt predictor reset...")
    
    with torch.no_grad():
        # Very small weights, slight positive bias
        fan_in = model.halt_predictor.in_features
        bound = 0.5 / math.sqrt(fan_in)  # Even gentler
        
        model.halt_predictor.weight.uniform_(-bound, bound)
        model.halt_predictor.bias.fill_(0.2)  # Slight positive bias toward halting
        
        print(f"  Reset: weights ±{bound:.4f}, bias=0.2")
    
    return model

def finetune_v2_6(dataset_name='agi-1'):
    """V2.6: Efficiency-rewarding loss function"""
    
    print(f"🎯 Fine-tuning V2.6 - EFFICIENCY REWARDS")
    print(f"🏆 Reward efficient correctness, not just correctness")
    print(f"🚫 Stop adversarial cycle escalation")
    print("=" * 60)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/hrm_arc_best.pt'
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Create model
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(DEVICE)
    
    # Load weights (skip halt predictor)
    model_state = model.state_dict()
    checkpoint_state = checkpoint['model_state_dict']
    
    loaded_keys = []
    for key in checkpoint_state:
        if key in model_state and model_state[key].shape == checkpoint_state[key].shape:
            if 'halt_predictor' not in key:
                model_state[key] = checkpoint_state[key]
                loaded_keys.append(key)
    
    model.load_state_dict(model_state)
    model = reset_halt_predictor_gentle(model)
    print(f"✅ Loaded {len(loaded_keys)} layers")
    
    # Test initial behavior with efficiency rewards
    print("\n🔍 Testing V2.6 efficiency-aware training...")
    model.eval()
    test_input = torch.randint(0, 10, (2, 900)).to(DEVICE)
    with torch.no_grad():
        outputs, test_halt_probs = model(test_input)
        halt_values = [p.mean().item() for p in test_halt_probs]
        print(f"  Initial cycles: {len(test_halt_probs)}")
        print(f"  Initial halt progression: {[f'{v:.3f}' for v in halt_values]}")
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
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    scaler = GradScaler() if TRAINING_CONFIG['use_amp'] else None
    criterion = EfficiencyRewardingLoss()
    
    # Initialize wandb
    run = wandb.init(
        project="hrm-reasoning-finetune-v2",
        name=f"reasoning-{output_suffix}-v2.6-efficiency-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={**MODEL_CONFIG, **TRAINING_CONFIG, 'dataset': dataset_name, 'version': 'v2.6-efficiency'},
        mode="offline"
    )
    
    # Training loop
    print("\n🏃 Starting V2.6 fine-tuning (Efficiency Rewards)...")
    model.train()
    
    global_step = 0
    accumulated_metrics = {
        'loss': 0, 'task_loss': 0, 'efficiency_bonus': 0, 
        'efficiency_penalty': 0, 'cycles': 0, 'accuracy': 0
    }
    cycle_distribution = []
    efficiency_distribution = []
    
    progress_bar = tqdm(total=TRAINING_CONFIG['num_steps'], desc="V2.6 Efficiency")
    
    while global_step < TRAINING_CONFIG['num_steps']:
        for batch in train_loader:
            if global_step >= TRAINING_CONFIG['num_steps']:
                break
            
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            
            # Forward pass with efficiency rewards
            if TRAINING_CONFIG['use_amp']:
                with autocast():
                    outputs, halt_probs = model(inputs)
                    loss, metrics = criterion(outputs, targets, halt_probs)
            else:
                outputs, halt_probs = model(inputs)
                loss, metrics = criterion(outputs, targets, halt_probs)
            
            # Track distributions
            cycle_distribution.append(metrics['num_cycles'])
            efficiency_score = metrics['efficiency_bonus'] - metrics['efficiency_penalty']
            efficiency_distribution.append(efficiency_score)
            
            # Scale for accumulation
            loss = loss / TRAINING_CONFIG['gradient_accumulation_steps']
            
            # Accumulate metrics
            for key in accumulated_metrics:
                if key == 'cycles':
                    accumulated_metrics[key] += metrics['num_cycles']
                elif key == 'loss':
                    accumulated_metrics[key] += loss.item()
                else:
                    accumulated_metrics[key] += metrics.get(key, 0)
            
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
                
                # Average accumulated metrics
                avg_metrics = {k: v / TRAINING_CONFIG['gradient_accumulation_steps'] 
                             for k, v in accumulated_metrics.items()}
                
                if global_step % 10 == 0:
                    # Efficiency analysis
                    if len(cycle_distribution) > 100:
                        recent_cycles = cycle_distribution[-100:]
                        recent_efficiency = efficiency_distribution[-100:]
                        
                        log_data = {
                            'loss': avg_metrics['loss'],
                            'task_loss': avg_metrics['task_loss'],
                            'efficiency_bonus': avg_metrics['efficiency_bonus'],
                            'efficiency_penalty': avg_metrics['efficiency_penalty'],
                            'net_efficiency': avg_metrics['efficiency_bonus'] - avg_metrics['efficiency_penalty'],
                            'accuracy': avg_metrics['accuracy'],
                            'reasoning_cycles': avg_metrics['cycles'],
                            'final_halt_prob': metrics['final_halt_prob'],
                            'cycles_mean': np.mean(recent_cycles),
                            'cycles_std': np.std(recent_cycles),
                            'cycles_min': np.min(recent_cycles),
                            'cycles_max': np.max(recent_cycles),
                            'cycles_unique': len(set(recent_cycles)),
                            'efficiency_mean': np.mean(recent_efficiency),
                            'step': global_step,
                        }
                        wandb.log(log_data)
                
                progress_bar.set_postfix({
                    'loss': f"{avg_metrics['loss']:.3f}",
                    'acc': f"{avg_metrics['accuracy']:.2f}",
                    'cyc': f"{avg_metrics['cycles']:.1f}",
                    'eff': f"{avg_metrics['efficiency_bonus']-avg_metrics['efficiency_penalty']:+.2f}",
                    'halt': f"{metrics['final_halt_prob']:.3f}"
                })
                
                # Reset accumulators
                for key in accumulated_metrics:
                    accumulated_metrics[key] = 0
            
            # Enhanced checkpoint with efficiency analysis
            if global_step % TRAINING_CONFIG['checkpoint_frequency'] == 0 and global_step > 0:
                save_path = f'checkpoints/hrm_reasoning_{dataset_name}_v2_6_step_{global_step}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'config': MODEL_CONFIG,
                    'dataset': dataset_name,
                    'version': 'v2.6-efficiency'
                }, save_path)
                print(f"\n💾 V2.6 Checkpoint: {save_path}")
                
                # Comprehensive efficiency test
                model.eval()
                with torch.no_grad():
                    # Test on multiple inputs to see adaptation
                    test_cycles = []
                    test_halt_probs = []
                    test_accuracies = []
                    
                    for test_i in range(5):
                        test_input_batch = torch.randint(0, 10, (1, 900)).to(DEVICE)
                        test_output, test_halt = model(test_input_batch)
                        
                        test_cycles.append(len(test_halt))
                        test_halt_probs.append([p.mean().item() for p in test_halt])
                    
                    cycle_variety = len(set(test_cycles))
                    avg_test_cycles = np.mean(test_cycles)
                    
                    print(f"  Test cycles: {test_cycles} (variety: {cycle_variety})")
                    print(f"  Average cycles: {avg_test_cycles:.1f}")
                    print(f"  Recent training cycles: {cycle_distribution[-10:] if len(cycle_distribution) >= 10 else cycle_distribution}")
                    print(f"  Efficiency trend: {efficiency_distribution[-5:] if len(efficiency_distribution) >= 5 else efficiency_distribution}")
                    
                    # Check if halt predictor logits are staying reasonable
                    test_input_debug = torch.randint(0, 10, (2, 900)).to(DEVICE)
                    _, halt_debug = model(test_input_debug)
                    # We'd need to modify the model to extract raw logits for debugging
                    
                model.train()
            
            global_step += 1
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Final analysis
    final_path = f'checkpoints/hrm_reasoning_{dataset_name}_v2_6_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'config': MODEL_CONFIG,
        'dataset': dataset_name,
        'version': 'v2.6-efficiency'
    }, final_path)
    
    wandb.finish()
    
    print(f"\n✅ V2.6 Efficiency Training Complete!")
    print(f"💾 Final model: {final_path}")
    
    # Efficiency analysis
    if cycle_distribution and efficiency_distribution:
        unique_cycles = len(set(cycle_distribution))
        avg_cycles = np.mean(cycle_distribution)
        avg_efficiency = np.mean(efficiency_distribution)
        
        print(f"\n📊 Efficiency Analysis:")
        print(f"   Average cycles: {avg_cycles:.1f}")
        print(f"   Cycle variety: {unique_cycles} different counts")
        print(f"   Average efficiency score: {avg_efficiency:+.3f}")
        print(f"   Efficiency trend: {'IMPROVING' if efficiency_distribution[-50:] and np.mean(efficiency_distribution[-50:]) > np.mean(efficiency_distribution[:50]) else 'STABLE'}")
        
        # Cycle distribution breakdown
        cycle_counts = np.bincount(cycle_distribution)
        print(f"   Cycle usage:")
        for cycles, count in enumerate(cycle_counts):
            if count > 0:
                percentage = (count / len(cycle_distribution)) * 100
                print(f"     {cycles:2d} cycles: {percentage:5.1f}% ({count} times)")
    
    return final_path

if __name__ == "__main__":
    finetune_v2_6('agi-1')