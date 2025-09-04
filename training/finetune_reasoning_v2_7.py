#!/usr/bin/env python3
"""
Fine-tune HRM V2.7 - TASK DIFFICULTY AWARENESS
Key fix: Train halt predictor to actually evaluate input complexity
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

def estimate_task_difficulty(inputs, targets):
    """Estimate task difficulty based on input/output characteristics"""
    batch_size = inputs.size(0)
    difficulties = []
    
    for i in range(batch_size):
        input_grid = inputs[i].cpu().numpy().reshape(30, 30)
        target_grid = targets[i].cpu().numpy().reshape(30, 30)
        
        # Count complexity indicators
        complexity_score = 0
        
        # 1. Number of unique colors (more colors = harder)
        unique_colors = len(set(input_grid.flatten())) - 1  # Remove padding
        complexity_score += min(unique_colors, 10) * 0.1
        
        # 2. Grid sparsity (more filled = potentially harder)
        non_zero = np.count_nonzero(input_grid)
        sparsity = non_zero / 900
        complexity_score += sparsity * 0.3
        
        # 3. Pattern irregularity (more entropy = harder)
        # Simple entropy measure
        values, counts = np.unique(input_grid, return_counts=True)
        if len(counts) > 1:
            entropy = -np.sum((counts/counts.sum()) * np.log2(counts/counts.sum() + 1e-8))
            complexity_score += entropy * 0.1
        
        # 4. Output size change (size changes are often harder)
        input_bbox = get_content_bbox(input_grid)
        target_bbox = get_content_bbox(target_grid)
        
        if input_bbox and target_bbox:
            input_area = (input_bbox[2] - input_bbox[0]) * (input_bbox[3] - input_bbox[1])
            target_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])
            
            if input_area != target_area:
                complexity_score += 0.3  # Size change is hard
        
        # Convert to target cycles (3-15 range)
        target_cycles = 3 + min(complexity_score * 8, 12)
        
        # Convert to logit space for halt predictor training
        # Higher difficulty = lower initial halt probability
        target_halt_logit = -2.0 + (15 - target_cycles) / 12 * 3.0  # Maps 3-15 cycles to +1.0 to -2.0 logits
        difficulties.append(target_halt_logit)
    
    return torch.tensor(difficulties, device=inputs.device, dtype=torch.float32)

def get_content_bbox(grid):
    """Get bounding box of non-zero content"""
    non_zero = np.where(grid != 0)
    if len(non_zero[0]) == 0:
        return None
    return (non_zero[0].min(), non_zero[1].min(), non_zero[0].max(), non_zero[1].max())

class TaskAwareLoss(nn.Module):
    """Loss function that teaches halt predictor task difficulty"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, targets, halt_probs, inputs):
        batch_size = outputs.size(0)
        
        # Standard accuracy computation
        predicted = outputs.argmax(dim=-1)
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
        
        correct_per_example = torch.tensor(correct_per_example, device=outputs.device, dtype=torch.float32)
        
        # Base task loss
        task_loss = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1),
            ignore_index=0,
            reduction='none'
        ).reshape(batch_size, -1).mean(dim=1)
        
        # TASK DIFFICULTY AWARENESS LOSS
        difficulty_targets = estimate_task_difficulty(inputs, targets)
        
        # The halt predictor should output logits that match task difficulty
        # We'll use the first cycle's halt logit as the "difficulty assessment"
        if halt_probs:
            # Extract first cycle halt logit (we need to modify model to return this)
            # For now, estimate from halt probability
            first_halt_prob = halt_probs[0].mean()
            # Convert back to approximate logit (before bias)
            first_logit = torch.log(first_halt_prob / (1 - first_halt_prob + 1e-8)) + 0.2  # Remove cycle 0 bias
            
            # Difficulty alignment loss
            difficulty_loss = F.mse_loss(first_logit.unsqueeze(0), difficulty_targets.mean().unsqueeze(0)) * 0.1
        else:
            difficulty_loss = 0.0
        
        # Efficiency rewards (same as V2.6)
        num_cycles = len(halt_probs)
        efficiency_adjustment = torch.zeros_like(task_loss)
        
        for i in range(batch_size):
            accuracy = correct_per_example[i]
            expected_difficulty = difficulty_targets[i].item()
            
            # Estimate optimal cycles from difficulty
            optimal_cycles = max(3, min(15, 3 + int((2.0 - expected_difficulty) * 3)))  # Map logit to cycles
            
            if accuracy > 0.8:  # Correct
                cycle_diff = abs(num_cycles - optimal_cycles)
                if cycle_diff <= 2:
                    efficiency_adjustment[i] = -0.3  # Reward for near-optimal
                elif cycle_diff <= 5:
                    efficiency_adjustment[i] = -0.1  # Small reward
                else:
                    efficiency_adjustment[i] = +0.1 * cycle_diff  # Penalty for far from optimal
            elif accuracy > 0.3:  # Partial
                if num_cycles <= optimal_cycles + 3:
                    efficiency_adjustment[i] = -0.1  # Mild reward
            else:  # Wrong
                if num_cycles >= 18:
                    efficiency_adjustment[i] = +0.2  # Penalty for max cycles + wrong
        
        # Adaptive threshold based on learned difficulty
        # Higher difficulty tasks should accumulate more before stopping
        avg_difficulty = difficulty_targets.mean().item()
        adaptive_threshold = 0.8 + (2.0 - avg_difficulty) * 0.3  # Range: 0.8 - 1.4
        
        total_loss = (task_loss + efficiency_adjustment).mean() + difficulty_loss
        
        return total_loss, {
            'task_loss': task_loss.mean().item(),
            'difficulty_loss': difficulty_loss if isinstance(difficulty_loss, float) else difficulty_loss.item(),
            'efficiency_bonus': -efficiency_adjustment[efficiency_adjustment < 0].sum().item(),
            'efficiency_penalty': efficiency_adjustment[efficiency_adjustment > 0].sum().item(),
            'num_cycles': num_cycles,
            'accuracy': correct_per_example.mean().item(),
            'avg_difficulty': avg_difficulty,
            'adaptive_threshold': adaptive_threshold,
            'final_halt_prob': halt_probs[-1].mean().item() if halt_probs else 0.0
        }

class HierarchicalReasoningModuleV27(nn.Module):
    """HRM V2.7 with task-aware halt predictor"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Standard HRM components (same as before)
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
        
        # ENHANCED halt predictor with difficulty awareness
        self.halt_predictor = nn.Sequential(
            nn.Linear(config['hidden_size'] * 2, config['hidden_size']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['hidden_size'], 1)
        )
        
        self.cycle_embedding = nn.Embedding(config['max_cycles'], config['hidden_size'])
        self.output = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        self.h_norm = nn.LayerNorm(config['hidden_size'])
        self.l_norm = nn.LayerNorm(config['hidden_size'])
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x, max_cycles=None, adaptive_threshold=None):
        batch_size, seq_len = x.shape
        max_cycles = max_cycles or self.config['max_cycles']
        adaptive_threshold = adaptive_threshold or 1.2  # Higher default threshold
        
        # Embed input
        x_emb = self.token_embedding(x)
        x_emb = self.pos_encoding(x_emb)
        x_emb = self.dropout(x_emb)
        
        h_state = x_emb.clone()
        l_state = x_emb.clone()
        
        halt_probs = []
        halt_logits = []  # Store for difficulty training
        cumulative_halt = torch.zeros(batch_size, 1).to(x.device)
        
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
            
            # Enhanced halt prediction
            combined = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)
            halt_logit = self.halt_predictor(combined)
            halt_logits.append(halt_logit)
            
            # Apply gentler bias
            cycle_bias = -0.2 + (cycle / (max_cycles - 1)) * 1.1
            biased_logit = halt_logit + cycle_bias
            halt_prob = torch.sigmoid(biased_logit)
            halt_probs.append(halt_prob)
            
            cumulative_halt = cumulative_halt + halt_prob
            
            # Adaptive stopping with variable threshold
            if cycle >= 3:
                if cumulative_halt.mean() > adaptive_threshold or halt_prob.mean() > 0.95:
                    break
        
        output = self.output(l_state)
        
        # Store halt logits for difficulty training
        self.last_halt_logits = halt_logits
        
        return output, halt_probs

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

def finetune_v2_7(dataset_name='agi-1'):
    """V2.7: Task difficulty awareness"""
    
    print(f"🎯 Fine-tuning V2.7 - TASK DIFFICULTY AWARENESS")
    print(f"🧠 Teach halt predictor to evaluate input complexity")
    print(f"🎚️ Adaptive thresholds based on estimated difficulty")
    print("=" * 60)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/hrm_arc_best.pt'
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Create enhanced model
    model = HierarchicalReasoningModuleV27(MODEL_CONFIG).to(DEVICE)
    
    # Load compatible weights
    model_state = model.state_dict()
    checkpoint_state = checkpoint['model_state_dict']
    
    loaded_keys = []
    for key in checkpoint_state:
        if key in model_state and model_state[key].shape == checkpoint_state[key].shape:
            # Skip halt predictor - we want fresh multi-layer predictor
            if 'halt_predictor' not in key:
                model_state[key] = checkpoint_state[key]
                loaded_keys.append(key)
    
    model.load_state_dict(model_state, strict=False)
    print(f"✅ Loaded {len(loaded_keys)} layers, fresh enhanced halt predictor")
    
    # Test with different difficulty examples
    print("\n🔍 Testing difficulty-aware halt behavior...")
    model.eval()
    
    # Create test inputs with different complexities
    simple_input = torch.zeros(1, 900, dtype=torch.long).to(DEVICE)  # All zeros = simple
    simple_input[0, :100] = 1  # Small pattern
    
    complex_input = torch.randint(0, 10, (1, 900)).to(DEVICE)  # Random = complex
    
    test_cases = [
        ("Simple task", simple_input, 0.8),  # Lower threshold for simple
        ("Complex task", complex_input, 1.4),  # Higher threshold for complex
    ]
    
    with torch.no_grad():
        for name, test_input, threshold in test_cases:
            outputs, test_halt_probs = model(test_input, adaptive_threshold=threshold)
            cycles_used = len(test_halt_probs)
            halt_values = [p.mean().item() for p in test_halt_probs]
            
            print(f"  {name}: {cycles_used} cycles, threshold={threshold}")
            print(f"    Halt progression: {' → '.join([f'{v:.3f}' for v in halt_values])}")
            
            # Check difficulty estimation
            difficulty = estimate_task_difficulty(test_input, test_input)  # Self as target for test
            print(f"    Estimated difficulty: {difficulty.item():.3f}")
    
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
    criterion = TaskAwareLoss()
    
    # Initialize wandb
    run = wandb.init(
        project="hrm-reasoning-finetune-v2",
        name=f"reasoning-{output_suffix}-v2.7-taskaware-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={**MODEL_CONFIG, **TRAINING_CONFIG, 'dataset': dataset_name, 'version': 'v2.7-taskaware'},
        mode="offline"
    )
    
    # Training loop with difficulty awareness
    print("\n🏃 Starting V2.7 fine-tuning (Task Difficulty Awareness)...")
    model.train()
    
    global_step = 0
    accumulated_metrics = {
        'loss': 0, 'task_loss': 0, 'difficulty_loss': 0,
        'efficiency_bonus': 0, 'efficiency_penalty': 0, 
        'cycles': 0, 'accuracy': 0, 'avg_difficulty': 0, 'adaptive_threshold': 0
    }
    
    cycle_distribution = []
    difficulty_distribution = []
    threshold_distribution = []
    
    progress_bar = tqdm(total=TRAINING_CONFIG['num_steps'], desc="V2.7 TaskAware")
    
    while global_step < TRAINING_CONFIG['num_steps']:
        for batch in train_loader:
            if global_step >= TRAINING_CONFIG['num_steps']:
                break
            
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            
            # Estimate batch difficulty and set adaptive threshold
            batch_difficulties = estimate_task_difficulty(inputs, targets)
            avg_batch_difficulty = batch_difficulties.mean().item()
            adaptive_threshold = 0.8 + (2.0 - avg_batch_difficulty) * 0.3
            adaptive_threshold = np.clip(adaptive_threshold, 0.6, 1.8)
            
            # Forward pass with adaptive threshold
            if TRAINING_CONFIG['use_amp']:
                with autocast('cuda'):
                    outputs, halt_probs = model(inputs, adaptive_threshold=adaptive_threshold)
                    loss, metrics = criterion(outputs, targets, halt_probs, inputs)
            else:
                outputs, halt_probs = model(inputs, adaptive_threshold=adaptive_threshold)
                loss, metrics = criterion(outputs, targets, halt_probs, inputs)
            
            # Track distributions
            cycle_distribution.append(metrics['num_cycles'])
            difficulty_distribution.append(metrics['avg_difficulty'])
            threshold_distribution.append(metrics['adaptive_threshold'])
            
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
                    # Enhanced logging with difficulty tracking
                    if len(cycle_distribution) > 50:
                        recent_cycles = cycle_distribution[-50:]
                        recent_difficulties = difficulty_distribution[-50:]
                        recent_thresholds = threshold_distribution[-50:]
                        
                        log_data = {
                            **{f'avg_{k}': v for k, v in avg_metrics.items()},
                            'cycles_mean': np.mean(recent_cycles),
                            'cycles_std': np.std(recent_cycles),
                            'cycles_unique': len(set(recent_cycles)),
                            'difficulty_mean': np.mean(recent_difficulties),
                            'threshold_mean': np.mean(recent_thresholds),
                            'step': global_step,
                        }
                        wandb.log(log_data)
                
                progress_bar.set_postfix({
                    'loss': f"{avg_metrics['loss']:.3f}",
                    'acc': f"{avg_metrics['accuracy']:.2f}",
                    'cyc': f"{avg_metrics['cycles']:.1f}",
                    'diff': f"{avg_metrics['avg_difficulty']:.2f}",
                    'thr': f"{avg_metrics['adaptive_threshold']:.2f}",
                    'var': f"{len(set(cycle_distribution[-20:])) if len(cycle_distribution) >= 20 else 1}"
                })
                
                # Reset
                for key in accumulated_metrics:
                    accumulated_metrics[key] = 0
            
            # Enhanced checkpoint
            if global_step % TRAINING_CONFIG['checkpoint_frequency'] == 0 and global_step > 0:
                save_path = f'checkpoints/hrm_reasoning_{dataset_name}_v2_7_step_{global_step}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                    'config': MODEL_CONFIG,
                    'dataset': dataset_name,
                    'version': 'v2.7-taskaware'
                }, save_path)
                print(f"\n💾 V2.7 Checkpoint: {save_path}")
                
                # Comprehensive variability test
                model.eval()
                test_cycles = []
                test_difficulties = []
                test_thresholds = []
                
                with torch.no_grad():
                    for _ in range(10):
                        test_input = torch.randint(0, 10, (1, 900)).to(DEVICE)
                        test_difficulty = estimate_task_difficulty(test_input, test_input).item()
                        test_threshold = 0.8 + (2.0 - test_difficulty) * 0.3
                        test_threshold = np.clip(test_threshold, 0.6, 1.8)
                        
                        _, test_halt = model(test_input, adaptive_threshold=test_threshold)
                        
                        test_cycles.append(len(test_halt))
                        test_difficulties.append(test_difficulty)
                        test_thresholds.append(test_threshold)
                
                cycle_variety = len(set(test_cycles))
                print(f"  Test cycles: {test_cycles}")
                print(f"  Difficulties: {[f'{d:.2f}' for d in test_difficulties]}")
                print(f"  Thresholds: {[f'{t:.2f}' for t in test_thresholds]}")
                print(f"  Cycle variety: {cycle_variety}")
                print(f"  Training variety (last 100): {len(set(cycle_distribution[-100:]))}")
                
                model.train()
            
            global_step += 1
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Final comprehensive analysis
    final_path = f'checkpoints/hrm_reasoning_{dataset_name}_v2_7_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'config': MODEL_CONFIG,
        'dataset': dataset_name,
        'version': 'v2.7-taskaware'
    }, final_path)
    
    wandb.finish()
    
    print(f"\n✅ V2.7 Task-Aware Training Complete!")
    print(f"💾 Final model: {final_path}")
    
    # Final variability analysis
    if cycle_distribution:
        unique_cycles = sorted(set(cycle_distribution))
        print(f"\n📊 FINAL VARIABILITY ANALYSIS:")
        print(f"   Unique cycle counts: {unique_cycles}")
        print(f"   Total variety: {len(unique_cycles)}")
        print(f"   Range: {min(cycle_distribution)}-{max(cycle_distribution)}")
        
        # Distribution breakdown
        cycle_counts = {}
        for c in cycle_distribution:
            cycle_counts[c] = cycle_counts.get(c, 0) + 1
        
        print(f"   Distribution:")
        for cycles in sorted(cycle_counts.keys()):
            count = cycle_counts[cycles]
            percentage = (count / len(cycle_distribution)) * 100
            print(f"     {cycles:2d} cycles: {percentage:5.1f}% ({count:4d} times)")
        
        # Correlation analysis
        if len(difficulty_distribution) == len(cycle_distribution):
            correlation = np.corrcoef(difficulty_distribution, cycle_distribution)[0,1]
            print(f"   Difficulty-Cycle correlation: {correlation:.3f}")
            print(f"   {'✅ GOOD: Negative correlation (harder→more cycles)' if correlation < -0.3 else '⚠️ WEAK: Low correlation' if abs(correlation) < 0.3 else '❌ BAD: Wrong correlation'}")
    
    return final_path

if __name__ == "__main__":
    finetune_v2_7('agi-1')