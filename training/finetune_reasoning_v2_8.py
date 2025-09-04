#!/usr/bin/env python3
"""
Fine-tune HRM V2.8 - CONFIDENCE-DRIVEN HALTING
Key innovation: Halt based on prediction stability, not input complexity
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
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

def compute_prediction_delta(prev_outputs, curr_outputs):
    """Compute change in predictions between cycles"""
    if prev_outputs is None:
        return torch.ones(curr_outputs.size(0)).to(curr_outputs.device)  # High delta for first cycle
    
    # Compare prediction probabilities
    prev_probs = F.softmax(prev_outputs, dim=-1)
    curr_probs = F.softmax(curr_outputs, dim=-1)
    
    # KL divergence between probability distributions (confidence measure)
    kl_div = F.kl_div(curr_probs.log(), prev_probs, reduction='none').sum(dim=-1).mean(dim=-1)
    
    # Alternative: L2 norm of logit differences
    logit_delta = torch.norm(curr_outputs - prev_outputs, dim=-1).mean(dim=-1)
    
    # Alternative: Top-1 prediction agreement
    prev_pred = prev_outputs.argmax(dim=-1)
    curr_pred = curr_outputs.argmax(dim=-1)
    disagreement = (prev_pred != curr_pred).float().mean(dim=-1)
    
    # Combine metrics (prioritize KL divergence as primary confidence measure)
    confidence_delta = kl_div + 0.1 * logit_delta + 0.5 * disagreement
    
    return confidence_delta

class ConfidenceDrivenLoss(nn.Module):
    """Loss function that teaches confidence-based halting"""
    
    def __init__(self):
        super().__init__()
        self.stability_threshold = 0.1  # When prediction delta drops below this, model should halt
        self.confidence_weight = 0.2    # Weight for confidence alignment loss
    
    def forward(self, all_cycle_outputs, targets, halt_probs, prediction_deltas=None):
        """
        Args:
            all_cycle_outputs: List of outputs from each reasoning cycle
            targets: Ground truth targets
            halt_probs: List of halt probabilities from each cycle
            prediction_deltas: List of prediction stability measures
        """
        batch_size = targets.size(0)
        num_cycles = len(all_cycle_outputs)
        
        # Use final cycle output for task loss
        final_outputs = all_cycle_outputs[-1]
        
        # Standard task loss
        task_loss = F.cross_entropy(
            final_outputs.reshape(-1, final_outputs.size(-1)),
            targets.reshape(-1),
            ignore_index=0,
            reduction='none'
        ).reshape(batch_size, -1).mean(dim=1)
        
        # Compute per-example accuracy for efficiency analysis
        predicted = final_outputs.argmax(dim=-1)
        targets_flat = targets.reshape(batch_size, -1)
        predicted_flat = predicted.reshape(batch_size, -1)
        
        correct_per_example = []
        for i in range(batch_size):
            mask = targets_flat[i] != 0
            if mask.sum() > 0:
                correct = (predicted_flat[i][mask] == targets_flat[i][mask]).float().mean()
                correct_per_example.append(correct.item())
            else:
                correct_per_example.append(0.0)
        
        correct_per_example = torch.tensor(correct_per_example, device=final_outputs.device, dtype=torch.float32)
        
        # CONFIDENCE-DRIVEN HALT TRAINING
        confidence_loss = 0.0
        
        if prediction_deltas and len(prediction_deltas) > 1:
            # Train halt predictor to correlate with prediction stability
            for cycle_idx in range(1, len(prediction_deltas)):  # Skip first cycle (no previous prediction)
                stability = prediction_deltas[cycle_idx]  # Lower = more stable
                halt_prob = halt_probs[cycle_idx - 1].squeeze()  # Halt prob from previous cycle
                
                # Target: High halt probability when predictions are stable (low delta)
                # Sigmoid maps stability to target halt probability
                target_halt_prob = torch.sigmoid(-stability * 10 + 2)  # High stability -> high halt prob
                
                # MSE loss between actual and target halt probabilities
                halt_mse = F.mse_loss(halt_prob, target_halt_prob.detach())
                confidence_loss += halt_mse * self.confidence_weight
        
        # EFFICIENCY REWARDS (adapted from V2.6)
        efficiency_adjustment = torch.zeros_like(task_loss)
        
        for i in range(batch_size):
            accuracy = correct_per_example[i]
            
            if accuracy > 0.8:  # Correct answer
                if num_cycles <= 5:
                    efficiency_adjustment[i] = -0.3  # Good efficiency
                elif num_cycles <= 8:
                    efficiency_adjustment[i] = -0.1  # Moderate efficiency
                elif num_cycles > 15:
                    efficiency_adjustment[i] = +0.1  # Too many cycles
                    
            elif accuracy > 0.3:  # Partial credit
                if num_cycles > 18:
                    efficiency_adjustment[i] = +0.1  # Penalty for max cycles
                    
            else:  # Wrong answer
                if num_cycles >= 18:
                    efficiency_adjustment[i] = +0.2  # Max cycles + wrong = bad
        
        # Combine losses
        total_loss = (task_loss + efficiency_adjustment).mean() + confidence_loss
        
        # Metrics for logging
        final_delta = prediction_deltas[-1].mean().item() if prediction_deltas else 0.0
        stability_achieved = (final_delta < self.stability_threshold) if prediction_deltas else False
        
        return total_loss, {
            'task_loss': task_loss.mean().item(),
            'confidence_loss': confidence_loss if isinstance(confidence_loss, float) else confidence_loss.item(),
            'efficiency_bonus': -efficiency_adjustment[efficiency_adjustment < 0].sum().item(),
            'efficiency_penalty': efficiency_adjustment[efficiency_adjustment > 0].sum().item(),
            'num_cycles': num_cycles,
            'accuracy': correct_per_example.mean().item(),
            'final_delta': final_delta,
            'stability_achieved': stability_achieved,
            'avg_delta': np.mean([d.mean().item() for d in prediction_deltas]) if prediction_deltas else 0.0,
            'final_halt_prob': halt_probs[-1].mean().item() if halt_probs else 0.0
        }

class HierarchicalReasoningModuleV28(nn.Module):
    """HRM V2.8 with confidence-driven halt predictor"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Standard HRM components
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.pos_encoding = PositionalEncoding(config['hidden_size'])
        
        self.h_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_h_layers'])
        ])
        
        self.l_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_l_layers'])
        ])
        
        self.h_to_l = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.l_to_h = nn.Linear(config['hidden_size'], config['hidden_size'])
        
        # CONFIDENCE-AWARE halt predictor
        # Takes current state + prediction delta from previous cycle
        self.halt_predictor = nn.Sequential(
            nn.Linear(config['hidden_size'] * 2 + 1, config['hidden_size']),  # +1 for prediction delta
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['hidden_size'], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.cycle_embedding = nn.Embedding(config['max_cycles'], config['hidden_size'])
        self.output = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        self.h_norm = nn.LayerNorm(config['hidden_size'])
        self.l_norm = nn.LayerNorm(config['hidden_size'])
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x, max_cycles=None, confidence_threshold=0.1):
        batch_size, seq_len = x.shape
        max_cycles = max_cycles or self.config['max_cycles']
        
        # Embed input
        x_emb = self.token_embedding(x)
        x_emb = self.pos_encoding(x_emb)
        x_emb = self.dropout(x_emb)
        
        h_state = x_emb.clone()
        l_state = x_emb.clone()
        
        all_outputs = []
        halt_probs = []
        prediction_deltas = []
        cumulative_halt = torch.zeros(batch_size, 1).to(x.device)
        
        prev_outputs = None
        
        for cycle in range(max_cycles):
            # Cycle embedding
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
            
            h_state = h_state + self.l_to_h(l_state)
            
            # Generate output for this cycle
            curr_outputs = self.output(l_state)
            all_outputs.append(curr_outputs)
            
            # Compute prediction delta (confidence measure)
            pred_delta = compute_prediction_delta(prev_outputs, curr_outputs)
            prediction_deltas.append(pred_delta)
            
            # CONFIDENCE-DRIVEN halt prediction
            combined_state = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)
            # Include prediction delta as confidence signal
            confidence_input = torch.cat([
                combined_state, 
                pred_delta.unsqueeze(-1)
            ], dim=-1)
            
            halt_logit = self.halt_predictor(confidence_input)
            
            # Gentle cycle bias (much reduced from previous versions)
            cycle_bias = cycle * 0.02  # Very gentle progression
            biased_logit = halt_logit + cycle_bias
            halt_prob = torch.sigmoid(biased_logit)
            halt_probs.append(halt_prob)
            
            cumulative_halt = cumulative_halt + halt_prob
            
            # CONFIDENCE-BASED stopping
            # Stop when predictions are stable (low delta) AND model wants to halt
            stable_predictions = pred_delta.mean() < confidence_threshold
            wants_to_halt = halt_prob.mean() > 0.7
            sufficient_cycles = cycle >= 3
            
            if sufficient_cycles and (stable_predictions or wants_to_halt or cumulative_halt.mean() > 1.5):
                break
                
            prev_outputs = curr_outputs.detach()  # Detach to prevent gradients through cycles
        
        # Store intermediate results for loss computation
        self.all_cycle_outputs = all_outputs
        self.prediction_deltas = prediction_deltas
        
        return all_outputs[-1], halt_probs  # Return final output and halt probs

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

def finetune_v2_8(dataset_name='agi-1'):
    """V2.8: Confidence-driven halting based on prediction stability"""
    
    print(f"🎯 Fine-tuning V2.8 - CONFIDENCE-DRIVEN HALTING")
    print(f"🧠 Halt based on prediction stability, not complexity")
    print(f"📈 Train halt predictor to recognize solution convergence")
    print("=" * 60)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/hrm_arc_best.pt'
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Create enhanced model
    model = HierarchicalReasoningModuleV28(MODEL_CONFIG).to(DEVICE)
    
    # Load compatible weights (skip halt predictor - it has different input size now)
    model_state = model.state_dict()
    checkpoint_state = checkpoint['model_state_dict']
    
    loaded_keys = []
    for key in checkpoint_state:
        if key in model_state and model_state[key].shape == checkpoint_state[key].shape:
            if 'halt_predictor' not in key:
                model_state[key] = checkpoint_state[key]
                loaded_keys.append(key)
    
    model.load_state_dict(model_state, strict=False)
    print(f"✅ Loaded {len(loaded_keys)} layers, fresh confidence-aware halt predictor")
    
    # Test confidence-driven behavior
    print("\\n🔍 Testing confidence-driven halt behavior...")
    model.eval()
    
    test_cases = [
        ("Simple pattern", torch.cat([torch.zeros(1, 450, dtype=torch.long), torch.ones(1, 450, dtype=torch.long)], dim=1).to(DEVICE)),
        ("Complex pattern", torch.randint(0, 10, (1, 900)).to(DEVICE)),
    ]
    
    with torch.no_grad():
        for name, test_input in test_cases:
            outputs, test_halt_probs = model(test_input, confidence_threshold=0.1)
            cycles_used = len(test_halt_probs)
            
            # Show prediction stability
            deltas = [d.mean().item() for d in model.prediction_deltas]
            halt_values = [p.mean().item() for p in test_halt_probs]
            
            print(f"  {name}:")
            print(f"    Cycles: {cycles_used}")
            print(f"    Prediction deltas: {' → '.join([f'{d:.3f}' for d in deltas])}")
            print(f"    Halt probabilities: {' → '.join([f'{h:.3f}' for h in halt_values])}")
            print(f"    Final stability: {'STABLE' if deltas[-1] < 0.1 else 'UNSTABLE'}")
    
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
    
    scaler = GradScaler('cuda') if TRAINING_CONFIG['use_amp'] else None
    criterion = ConfidenceDrivenLoss()
    
    # Initialize wandb
    run = wandb.init(
        project="hrm-reasoning-finetune-v2",
        name=f"reasoning-{output_suffix}-v2.8-confidence-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={**MODEL_CONFIG, **TRAINING_CONFIG, 'dataset': dataset_name, 'version': 'v2.8-confidence'},
        mode="offline"
    )
    
    # Training loop with confidence tracking
    print("\\n🏃 Starting V2.8 fine-tuning (Confidence-Driven Halting)...")
    model.train()
    
    global_step = 0
    accumulated_metrics = {
        'loss': 0, 'task_loss': 0, 'confidence_loss': 0,
        'efficiency_bonus': 0, 'efficiency_penalty': 0, 
        'cycles': 0, 'accuracy': 0, 'final_delta': 0, 'avg_delta': 0
    }
    
    cycle_distribution = []
    delta_distribution = []
    stability_count = 0
    
    progress_bar = tqdm(total=TRAINING_CONFIG['num_steps'], desc="V2.8 Confidence")
    
    while global_step < TRAINING_CONFIG['num_steps']:
        for batch in train_loader:
            if global_step >= TRAINING_CONFIG['num_steps']:
                break
            
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            
            # Forward pass with confidence tracking
            if TRAINING_CONFIG['use_amp']:
                with autocast('cuda'):
                    outputs, halt_probs = model(inputs, confidence_threshold=0.1)
                    loss, metrics = criterion(model.all_cycle_outputs, targets, halt_probs, model.prediction_deltas)
            else:
                outputs, halt_probs = model(inputs, confidence_threshold=0.1)
                loss, metrics = criterion(model.all_cycle_outputs, targets, halt_probs, model.prediction_deltas)
            
            # Track distributions
            cycle_distribution.append(metrics['num_cycles'])
            delta_distribution.append(metrics['final_delta'])
            if metrics['stability_achieved']:
                stability_count += 1
            
            # Accumulate
            loss = loss / TRAINING_CONFIG['gradient_accumulation_steps']
            
            for key in accumulated_metrics:
                if key == 'cycles':
                    accumulated_metrics[key] += metrics['num_cycles']
                elif key == 'loss':
                    accumulated_metrics[key] += loss.item()
                else:
                    accumulated_metrics[key] += metrics.get(key, 0)
            
            # Backward
            if TRAINING_CONFIG['use_amp']:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Step
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
                
                # Log
                avg_metrics = {k: v / TRAINING_CONFIG['gradient_accumulation_steps'] 
                             for k, v in accumulated_metrics.items()}
                
                if global_step % 10 == 0:
                    # Enhanced logging with confidence tracking
                    if len(cycle_distribution) > 50:
                        recent_cycles = cycle_distribution[-50:]
                        recent_deltas = delta_distribution[-50:]
                        recent_stability_rate = stability_count / len(cycle_distribution) * 100
                        
                        log_data = {
                            **{f'avg_{k}': v for k, v in avg_metrics.items()},
                            'cycles_mean': np.mean(recent_cycles),
                            'cycles_std': np.std(recent_cycles),
                            'cycles_unique': len(set(recent_cycles)),
                            'delta_mean': np.mean(recent_deltas),
                            'delta_std': np.std(recent_deltas),
                            'stability_rate': recent_stability_rate,
                            'step': global_step,
                        }
                        wandb.log(log_data)
                
                progress_bar.set_postfix({
                    'loss': f"{avg_metrics['loss']:.3f}",
                    'acc': f"{avg_metrics['accuracy']:.2f}",
                    'cyc': f"{avg_metrics['cycles']:.1f}",
                    'delta': f"{avg_metrics['final_delta']:.3f}",
                    'conf': f"{avg_metrics['confidence_loss']:.3f}",
                    'var': f"{len(set(cycle_distribution[-20:])) if len(cycle_distribution) >= 20 else 1}"
                })
                
                # Reset
                for key in accumulated_metrics:
                    accumulated_metrics[key] = 0
            
            # Enhanced checkpoint
            if global_step % TRAINING_CONFIG['checkpoint_frequency'] == 0 and global_step > 0:
                save_path = f'checkpoints/hrm_reasoning_{dataset_name}_v2_8_step_{global_step}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'config': MODEL_CONFIG,
                    'dataset': dataset_name,
                    'version': 'v2.8-confidence'
                }, save_path)
                print(f"\\n💾 V2.8 Checkpoint: {save_path}")
                
                # Comprehensive confidence test
                model.eval()
                test_results = []
                
                with torch.no_grad():
                    for _ in range(10):
                        test_input = torch.randint(0, 10, (1, 900)).to(DEVICE)
                        test_output, test_halt = model(test_input, confidence_threshold=0.1)
                        
                        test_cycles = len(test_halt)
                        test_deltas = [d.mean().item() for d in model.prediction_deltas]
                        final_stability = test_deltas[-1] < 0.1
                        
                        test_results.append({
                            'cycles': test_cycles,
                            'final_delta': test_deltas[-1],
                            'stable': final_stability
                        })
                
                cycles = [r['cycles'] for r in test_results]
                deltas = [r['final_delta'] for r in test_results]
                stable_count = sum(r['stable'] for r in test_results)
                
                print(f"  Test cycles: {cycles}")
                print(f"  Test deltas: {[f'{d:.3f}' for d in deltas]}")
                print(f"  Stable solutions: {stable_count}/10")
                print(f"  Cycle variety: {len(set(cycles))}")
                print(f"  Training variety (last 100): {len(set(cycle_distribution[-100:]))}")
                
                model.train()
            
            global_step += 1
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Final analysis
    final_path = f'checkpoints/hrm_reasoning_{dataset_name}_v2_8_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'config': MODEL_CONFIG,
        'dataset': dataset_name,
        'version': 'v2.8-confidence'
    }, final_path)
    
    wandb.finish()
    
    print(f"\\n✅ V2.8 Confidence-Driven Training Complete!")
    print(f"💾 Final model: {final_path}")
    
    # Confidence analysis
    if cycle_distribution and delta_distribution:
        unique_cycles = len(set(cycle_distribution))
        avg_cycles = np.mean(cycle_distribution)
        avg_delta = np.mean(delta_distribution)
        final_stability_rate = stability_count / len(cycle_distribution) * 100
        
        print(f"\\n📊 Confidence Analysis:")
        print(f"   Average cycles: {avg_cycles:.1f}")
        print(f"   Cycle variety: {unique_cycles} different counts")
        print(f"   Average final delta: {avg_delta:.3f}")
        print(f"   Stability achievement rate: {final_stability_rate:.1f}%")
        
        # Cycle distribution
        cycle_counts = np.bincount(cycle_distribution)
        print(f"   Cycle usage:")
        for cycles, count in enumerate(cycle_counts):
            if count > 0:
                percentage = (count / len(cycle_distribution)) * 100
                print(f"     {cycles:2d} cycles: {percentage:5.1f}% ({count} times)")
        
        # Correlation between stability and cycle count
        if len(set(cycle_distribution)) > 1 and len(set(delta_distribution)) > 1:
            correlation = np.corrcoef(delta_distribution, cycle_distribution)[0,1]
            print(f"   Delta-Cycle correlation: {correlation:.3f}")
            print(f"   {'✅ GOOD: Stable predictions → fewer cycles' if correlation < -0.3 else '⚠️ WEAK: Low correlation' if abs(correlation) < 0.3 else '❌ BAD: Unstable predictions → fewer cycles'}")
    
    return final_path

if __name__ == "__main__":
    finetune_v2_8('agi-1')