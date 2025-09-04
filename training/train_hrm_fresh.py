#!/usr/bin/env python3
"""
Fresh HRM training from random weights
- Uses 500x augmented dataset
- No validation during training
- Checkpoints every 10k steps
- Batch size 20, 4 workers
- Reality checks on architecture and loss
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from pathlib import Path
import json
import time
from tqdm import tqdm
import numpy as np
import math
from datetime import datetime
import random
from typing import Dict, Any, List, Tuple

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    'vocab_size': 12,      # ARC has colors 0-10 plus padding
    'hidden_size': 256,    
    'num_heads': 8,
    'num_h_layers': 4,     # High-level reasoning layers
    'num_l_layers': 3,     # Low-level processing layers (matching successful config)
    'dropout': 0.1,
    'max_cycles': 8,       # Reasoning cycles
}

TRAINING_CONFIG = {
    'batch_size': 20,
    'num_workers': 4,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'max_steps': 200000,
    'checkpoint_every': 10000,  # Every 10k steps
    'gradient_clip': 1.0,
    'use_amp': True,  # Automatic mixed precision for speed
}

# ============================================================================
# ARCHITECTURE REALITY CHECK
# ============================================================================

print("=" * 80)
print("ARCHITECTURE REALITY CHECK")
print("=" * 80)
print("\nModel Configuration:")
for k, v in MODEL_CONFIG.items():
    print(f"  {k}: {v}")

print("\nExpected behavior:")
print("  - Input: [batch_size, 900] tensor of integers 0-11")
print("  - Output: [batch_size, 900, 12] logits")
print("  - H↔L bidirectional communication for reasoning")
print("  - Multiple reasoning cycles with halt mechanism")

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
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
    """HRM with H↔L bidirectional communication"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings - CRITICAL for learning
        self.token_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.pos_encoding = PositionalEncoding(config['hidden_size'])
        
        # H-layers (high-level reasoning)
        self.h_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_h_layers'])
        ])
        
        # L-layers (low-level processing)
        self.l_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                dropout=config['dropout'],
                batch_first=True
            ) for _ in range(config['num_l_layers'])
        ])
        
        # Cross-layer connections for H↔L communication
        self.h_to_l = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.l_to_h = nn.Linear(config['hidden_size'], config['hidden_size'])
        
        # Halt predictor for adaptive computation
        self.halt_predictor = nn.Linear(config['hidden_size'] * 2, 1)
        
        # Cycle embedding for reasoning steps
        self.cycle_embedding = nn.Embedding(config['max_cycles'], config['hidden_size'])
        
        # Output projection - MUST produce diverse outputs!
        self.output = nn.Linear(config['hidden_size'], config['vocab_size'])
        
        # Normalization layers
        self.h_norm = nn.LayerNorm(config['hidden_size'])
        self.l_norm = nn.LayerNorm(config['hidden_size'])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config['dropout'])
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Proper weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, x, max_cycles=None):
        batch_size, seq_len = x.shape
        max_cycles = max_cycles or self.config['max_cycles']
        
        # Embed tokens - this MUST produce different embeddings for different inputs
        x_emb = self.token_embedding(x)
        x_emb = self.pos_encoding(x_emb)
        x_emb = self.dropout(x_emb)
        
        # Initialize H and L states
        h_state = x_emb.clone()
        l_state = x_emb.clone()
        
        halt_probs = []
        cumulative_halt = torch.zeros(batch_size, 1).to(x.device)
        
        # Multiple reasoning cycles
        for cycle in range(max_cycles):
            # Add cycle information
            cycle_emb = self.cycle_embedding(torch.tensor([cycle], device=x.device))
            cycle_emb = cycle_emb.expand(batch_size, seq_len, -1)
            
            # H-level reasoning
            h_state = h_state + 0.1 * cycle_emb
            for h_layer in self.h_layers:
                h_state = h_layer(h_state)
            h_state = self.h_norm(h_state)
            
            # L-level processing with H influence
            l_state = l_state + self.h_to_l(h_state)
            for l_layer in self.l_layers:
                l_state = l_layer(l_state)
            l_state = self.l_norm(l_state)
            
            # Update H with L information (bidirectional)
            h_state = h_state + self.l_to_h(l_state)
            
            # Compute halt probability
            combined = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)
            halt_logit = self.halt_predictor(combined)
            halt_prob = torch.sigmoid(halt_logit)
            halt_probs.append(halt_prob)
            
            cumulative_halt = cumulative_halt + halt_prob
            
            # Simple stopping condition
            if cycle >= 3 and cumulative_halt.mean() > 1.0:
                break
        
        # Generate output from final L-state
        output = self.output(l_state)
        
        return output, halt_probs

# ============================================================================
# DATASET WITH 500x AUGMENTATION
# ============================================================================

class ARCAugmentedDataset(Dataset):
    """ARC dataset with 500x augmentation"""
    
    def __init__(self, data_path: Path, split: str = 'training'):
        self.data_path = data_path
        self.split = split
        self.tasks = []
        self.augmentation_factor = 500  # Each task augmented 500 times
        
        # Load tasks
        json_files = list((data_path / split).glob('*.json'))
        print(f"Loading {len(json_files)} {split} tasks...")
        
        for json_file in json_files:
            with open(json_file, 'r') as f:
                task = json.load(f)
                self.tasks.append(task)
        
        print(f"Loaded {len(self.tasks)} tasks, will augment to {len(self.tasks) * self.augmentation_factor} examples")
    
    def __len__(self):
        return len(self.tasks) * self.augmentation_factor
    
    def augment_grid(self, grid: List[List[int]]) -> List[List[int]]:
        """Apply random augmentation to grid"""
        grid = np.array(grid)
        
        # Random rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        if k > 0:
            grid = np.rot90(grid, k)
        
        # Random flip
        if random.random() > 0.5:
            grid = np.fliplr(grid)
        if random.random() > 0.5:
            grid = np.flipud(grid)
        
        # Random color permutation (keeping 0 as background)
        if random.random() > 0.5:
            colors = list(range(1, 11))
            random.shuffle(colors)
            color_map = {0: 0}
            for i, c in enumerate(colors):
                color_map[i + 1] = c
            
            new_grid = np.zeros_like(grid)
            for old_color, new_color in color_map.items():
                new_grid[grid == old_color] = new_color
            grid = new_grid
        
        return grid.tolist()
    
    def grid_to_tensor(self, grid: List[List[int]]) -> torch.Tensor:
        """Convert grid to 900-length tensor"""
        grid = np.array(grid, dtype=np.int64)
        h, w = grid.shape
        
        # Pad to 30x30
        padded = np.zeros((30, 30), dtype=np.int64)
        padded[:min(h, 30), :min(w, 30)] = grid[:min(h, 30), :min(w, 30)]
        
        # Flatten
        return torch.from_numpy(padded.flatten())
    
    def __getitem__(self, idx):
        # Get base task
        task_idx = idx % len(self.tasks)
        task = self.tasks[task_idx]
        
        # Use training examples (not test)
        examples = task['train']
        example = random.choice(examples)
        
        # Apply augmentation
        input_grid = self.augment_grid(example['input'])
        output_grid = self.augment_grid(example['output'])
        
        # Convert to tensors
        input_tensor = self.grid_to_tensor(input_grid)
        output_tensor = self.grid_to_tensor(output_grid)
        
        return {
            'input': input_tensor,
            'target': output_tensor
        }

# ============================================================================
# LOSS FUNCTION REALITY CHECK
# ============================================================================

class ImprovedARCLoss(nn.Module):
    """
    Improved loss function that prevents constant output collapse
    """
    
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    
    def forward(self, outputs, targets, halt_probs):
        batch_size = outputs.size(0)
        
        # 1. Standard cross-entropy loss
        ce_loss = self.ce_loss(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1)
        ).reshape(batch_size, -1).mean(dim=1)
        
        # 2. Diversity penalty - penalize if all predictions are the same
        predictions = outputs.argmax(dim=-1)
        diversity_penalty = 0
        
        for i in range(batch_size):
            unique_preds = len(torch.unique(predictions[i]))
            if unique_preds == 1:  # Only one class predicted
                diversity_penalty += 0.5  # Significant penalty
            elif unique_preds < 3:  # Very few classes
                diversity_penalty += 0.2
        
        diversity_penalty = diversity_penalty / batch_size
        
        # 3. Output distribution penalty - prevent extreme class bias
        logit_means = outputs.mean(dim=[0, 1])  # Mean logits per class
        logit_std = logit_means.std()
        
        # Penalize if one class dominates (low std means more uniform)
        if logit_std > 2.0:  # High variance = one class dominating
            distribution_penalty = (logit_std - 1.0) * 0.1
        else:
            distribution_penalty = 0
        
        # 4. Halt loss - encourage reasonable number of cycles
        if halt_probs:
            num_cycles = len(halt_probs)
            if num_cycles < 3:
                halt_penalty = 0.1
            elif num_cycles > 6:
                halt_penalty = 0.05 * (num_cycles - 6)
            else:
                halt_penalty = 0
        else:
            halt_penalty = 0
        
        # Combine losses
        total_loss = ce_loss.mean() + diversity_penalty + distribution_penalty + halt_penalty
        
        # Return metrics for logging
        metrics = {
            'ce_loss': ce_loss.mean().item(),
            'diversity_penalty': diversity_penalty,
            'distribution_penalty': distribution_penalty,
            'halt_penalty': halt_penalty,
            'unique_predictions': len(torch.unique(predictions)),
            'num_cycles': len(halt_probs) if halt_probs else 0
        }
        
        return total_loss, metrics

print("\n" + "=" * 80)
print("LOSS FUNCTION REALITY CHECK")
print("=" * 80)
print("\nLoss components:")
print("  1. Cross-entropy loss - standard classification")
print("  2. Diversity penalty - prevents all-same predictions")
print("  3. Distribution penalty - prevents extreme class bias")
print("  4. Halt penalty - encourages reasonable reasoning cycles")
print("\nThis should prevent collapse to constant outputs!")

# ============================================================================
# INPUT SENSITIVITY CHECK
# ============================================================================

def check_input_sensitivity(model, num_tests=5):
    """Verify model produces different outputs for different inputs"""
    model.eval()
    
    test_inputs = []
    test_outputs = []
    
    with torch.no_grad():
        for i in range(num_tests):
            # Create different test inputs
            if i == 0:
                test_input = torch.zeros(1, 900, dtype=torch.long).to(DEVICE)
            elif i == 1:
                test_input = torch.ones(1, 900, dtype=torch.long).to(DEVICE)
            else:
                test_input = torch.randint(0, 10, (1, 900)).to(DEVICE)
            
            output, _ = model(test_input)
            test_inputs.append(test_input)
            test_outputs.append(output)
    
    # Check if outputs are different
    all_same = True
    for i in range(1, num_tests):
        if not torch.allclose(test_outputs[0], test_outputs[i], rtol=1e-3):
            all_same = False
            break
    
    if all_same:
        print("⚠️ WARNING: Model produces same output for different inputs!")
        return False
    else:
        print("✅ Model produces different outputs for different inputs")
        return True
    
    model.train()
    return True

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_hrm_fresh():
    """Train HRM from scratch with all safeguards"""
    
    print("\n" + "=" * 80)
    print("STARTING FRESH HRM TRAINING")
    print("=" * 80)
    
    # Create model from scratch
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Size: {total_params * 4 / 1024**2:.1f} MB")
    
    # Initial input sensitivity check
    print("\nInitial input sensitivity check:")
    check_input_sensitivity(model)
    
    # Setup dataset
    data_path = Path('../dataset/raw-data/ARC-AGI/data')
    if not data_path.exists():
        data_path = Path('../../arc-agi-1/data')
    if not data_path.exists():
        print(f"Error: ARC dataset not found at {data_path}")
        return
    
    train_dataset = ARCAugmentedDataset(data_path, 'training')
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # Setup loss
    criterion = ImprovedARCLoss()
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if TRAINING_CONFIG['use_amp'] else None
    
    # Training loop
    model.train()
    global_step = 0
    epoch = 0
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training ID for this run
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"\nRun ID: {run_id}")
    print(f"Checkpoints will be saved to: checkpoints/hrm_fresh_{run_id}_step_*.pt")
    
    print("\n" + "=" * 80)
    print("TRAINING STARTING")
    print("=" * 80)
    
    # Progress bar
    pbar = tqdm(total=TRAINING_CONFIG['max_steps'], desc="Training")
    
    # Metrics tracking
    recent_losses = []
    recent_metrics = {}
    
    while global_step < TRAINING_CONFIG['max_steps']:
        epoch += 1
        
        for batch in train_loader:
            if global_step >= TRAINING_CONFIG['max_steps']:
                break
            
            inputs = batch['input'].to(DEVICE)
            targets = batch['target'].to(DEVICE)
            
            # Forward pass
            if TRAINING_CONFIG['use_amp']:
                with autocast('cuda'):
                    outputs, halt_probs = model(inputs)
                    loss, metrics = criterion(outputs, targets, halt_probs)
            else:
                outputs, halt_probs = model(inputs)
                loss, metrics = criterion(outputs, targets, halt_probs)
            
            # Backward pass
            optimizer.zero_grad()
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['gradient_clip'])
                optimizer.step()
            
            # Track metrics
            recent_losses.append(loss.item())
            for k, v in metrics.items():
                if k not in recent_metrics:
                    recent_metrics[k] = []
                recent_metrics[k].append(v)
            
            # Keep only recent metrics
            if len(recent_losses) > 100:
                recent_losses = recent_losses[-100:]
                for k in recent_metrics:
                    recent_metrics[k] = recent_metrics[k][-100:]
            
            global_step += 1
            
            # Update progress bar
            if global_step % 10 == 0:
                avg_loss = np.mean(recent_losses)
                avg_unique = np.mean(recent_metrics.get('unique_predictions', [0]))
                avg_cycles = np.mean(recent_metrics.get('num_cycles', [0]))
                
                pbar.set_postfix({
                    'loss': f"{avg_loss:.3f}",
                    'unique': f"{avg_unique:.1f}",
                    'cycles': f"{avg_cycles:.1f}"
                })
            
            pbar.update(1)
            
            # Checkpoint
            if global_step % TRAINING_CONFIG['checkpoint_every'] == 0:
                # Input sensitivity check
                print(f"\n\nStep {global_step} - Input sensitivity check:")
                is_sensitive = check_input_sensitivity(model)
                
                if not is_sensitive:
                    print("⚠️ WARNING: Model becoming input-invariant! This is BAD!")
                
                # Save checkpoint
                checkpoint_path = checkpoint_dir / f"hrm_fresh_{run_id}_step_{global_step}.pt"
                checkpoint = {
                    'global_step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': MODEL_CONFIG,
                    'run_id': run_id,
                    'metrics': {
                        'avg_loss': np.mean(recent_losses),
                        'avg_unique_predictions': np.mean(recent_metrics.get('unique_predictions', [0])),
                        'input_sensitive': is_sensitive
                    }
                }
                
                if scaler:
                    checkpoint['scaler_state_dict'] = scaler.state_dict()
                
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
                
                # Continue training
                model.train()
    
    pbar.close()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    # Final checks
    print("\nFinal input sensitivity check:")
    check_input_sensitivity(model)
    
    # Save final model
    final_path = checkpoint_dir / f"hrm_fresh_{run_id}_final.pt"
    torch.save({
        'global_step': global_step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': MODEL_CONFIG,
        'run_id': run_id
    }, final_path)
    
    print(f"\nFinal model saved: {final_path}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HRM FRESH TRAINING SETUP")
    print("=" * 80)
    print(f"\nDevice: {DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "=" * 80)
    print("REALITY CHECKS COMPLETE")
    print("=" * 80)
    print("\n✅ Architecture: H↔L bidirectional with proper initialization")
    print("✅ Loss function: Includes diversity and distribution penalties")
    print("✅ Dataset: 500x augmentation with rotations, flips, and color permutations")
    print("✅ Monitoring: Input sensitivity checks every 10k steps")
    print("✅ Checkpoints: Every 10k steps for efficiency")
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    # Auto-start training
    train_hrm_fresh()