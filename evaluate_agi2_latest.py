#!/usr/bin/env python3
"""
Evaluate latest HRM checkpoint (step 193000) on ARC-AGI-2 public test set
Compare to baseline performance to see if model has learned beyond outputting zeros
"""
import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.append('.')
sys.path.append('training')

# Import HRM model - try different import paths
try:
    from training.train_hrm import HierarchicalReasoningModule
except:
    try:
        from train_hrm import HierarchicalReasoningModule
    except:
        # Define the model inline if imports fail
        import math
        
        class PositionalEncoding(nn.Module):
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
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # Token and position embeddings
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
                
                # Cross-layer connections
                self.h_to_l = nn.Linear(config['hidden_size'], config['hidden_size'])
                self.l_to_h = nn.Linear(config['hidden_size'], config['hidden_size'])
                
                # Halt predictor
                self.halt_predictor = nn.Linear(config['hidden_size'] * 2, 1)
                
                # Cycle embedding
                self.cycle_embedding = nn.Embedding(config['max_cycles'], config['hidden_size'])
                
                # Output projection
                self.output = nn.Linear(config['hidden_size'], config['vocab_size'])
                
                # Layer norms
                self.h_norm = nn.LayerNorm(config['hidden_size'])
                self.l_norm = nn.LayerNorm(config['hidden_size'])
                
                # Dropout
                self.dropout = nn.Dropout(config['dropout'])
            
            def forward(self, x, max_cycles=None):
                batch_size, seq_len = x.shape
                max_cycles = max_cycles or self.config['max_cycles']
                
                # Embed tokens
                x_emb = self.token_embedding(x)
                x_emb = self.pos_encoding(x_emb)
                x_emb = self.dropout(x_emb)
                
                h_state = x_emb.clone()
                l_state = x_emb.clone()
                
                halt_probs = []
                cumulative_halt = torch.zeros(batch_size, 1).to(x.device)
                
                for cycle in range(max_cycles):
                    # Add cycle embedding
                    cycle_emb = self.cycle_embedding(torch.tensor([cycle], device=x.device))
                    cycle_emb = cycle_emb.expand(batch_size, seq_len, -1)
                    
                    # H-level processing
                    h_state = h_state + 0.1 * cycle_emb
                    for h_layer in self.h_layers:
                        h_state = h_layer(h_state)
                    h_state = self.h_norm(h_state)
                    
                    # L-level processing with H influence
                    l_state = l_state + self.h_to_l(h_state)
                    for l_layer in self.l_layers:
                        l_state = l_layer(l_state)
                    l_state = self.l_norm(l_state)
                    
                    # Update H with L information
                    h_state = h_state + self.l_to_h(l_state)
                    
                    # Compute halt probability
                    combined = torch.cat([h_state.mean(dim=1), l_state.mean(dim=1)], dim=-1)
                    halt_logit = self.halt_predictor(combined)
                    halt_prob = torch.sigmoid(halt_logit)
                    halt_probs.append(halt_prob)
                    
                    cumulative_halt = cumulative_halt + halt_prob
                    
                    # Check stopping condition
                    if cycle >= 3 and cumulative_halt.mean() > 1.0:
                        break
                
                # Generate output
                output = self.output(l_state)
                
                return output, halt_probs

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_arc_agi2_test_tasks(data_path: Path) -> List[Dict]:
    """Load ARC-AGI-2 test tasks"""
    test_tasks = []
    
    # Check both possible locations
    test_paths = [
        data_path / 'arc-agi_test_challenges.json',
        data_path / 'test_challenges.json',
        data_path / 'arc-prize-2025' / 'arc-agi_test_challenges.json'
    ]
    
    test_file = None
    for path in test_paths:
        if path.exists():
            test_file = path
            print(f"Found test file: {test_file}")
            break
    
    if not test_file:
        print(f"Warning: No test file found in {data_path}")
        print(f"Tried: {test_paths}")
        return []
    
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    # Convert to list format
    for task_id, task_data in test_data.items():
        task = {
            'id': task_id,
            'train': task_data.get('train', []),
            'test': task_data.get('test', [])
        }
        test_tasks.append(task)
    
    return test_tasks

def preprocess_grid(grid: List[List[int]], max_size: int = 30) -> torch.Tensor:
    """Preprocess ARC grid to tensor format"""
    grid_array = np.array(grid, dtype=np.int64)
    h, w = grid_array.shape
    
    # Pad to max_size x max_size
    padded = np.zeros((max_size, max_size), dtype=np.int64)
    padded[:min(h, max_size), :min(w, max_size)] = grid_array[:min(h, max_size), :min(w, max_size)]
    
    # Flatten to sequence
    flat = padded.flatten()
    
    return torch.tensor(flat, dtype=torch.long)

def postprocess_output(output: torch.Tensor, target_shape: Tuple[int, int]) -> List[List[int]]:
    """Convert model output back to ARC grid format"""
    # Get predictions
    if len(output.shape) == 3:
        predictions = output.argmax(dim=-1)
    else:
        predictions = output
    
    # Reshape to grid
    predictions = predictions.squeeze(0).cpu().numpy()
    
    # Reshape to 30x30
    grid = predictions.reshape(30, 30)
    
    # Crop to target size
    h, w = target_shape
    result = grid[:h, :w].tolist()
    
    return result

def calculate_pixel_accuracy(pred: List[List[int]], target: List[List[int]]) -> float:
    """Calculate pixel-wise accuracy"""
    pred_array = np.array(pred)
    target_array = np.array(target)
    
    # Ensure same shape
    if pred_array.shape != target_array.shape:
        return 0.0
    
    correct = (pred_array == target_array).sum()
    total = pred_array.size
    
    return correct / total

def calculate_zero_baseline(target: List[List[int]]) -> float:
    """Calculate accuracy if we predict all zeros"""
    target_array = np.array(target)
    zeros = (target_array == 0).sum()
    total = target_array.size
    return zeros / total

def evaluate_latest_checkpoint():
    """Evaluate step 193000 checkpoint on ARC-AGI-2"""
    
    print("=" * 60)
    print("EVALUATING HRM STEP 193000 ON ARC-AGI-2 PUBLIC TEST")
    print("=" * 60)
    
    # Load model
    checkpoint_path = Path('training/checkpoints/hrm_arc_step_193000.pt')
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Print checkpoint info
    if 'global_step' in checkpoint:
        print(f"   Step: {checkpoint['global_step']}")
    if 'epoch' in checkpoint:
        print(f"   Epoch: {checkpoint['epoch']}")
    
    # Initialize model
    model_config = checkpoint.get('config', {
        'vocab_size': 12,
        'hidden_size': 256,
        'num_heads': 8,
        'num_h_layers': 4,
        'num_l_layers': 6,
        'dropout': 0.1,
        'max_cycles': 8
    })
    
    model = HierarchicalReasoningModule(model_config).to(DEVICE)
    
    # Load weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning loading model: {e}")
    
    model.eval()
    
    # Load test data
    data_paths = [
        Path('../arc-agi-2/data'),
        Path('kaggle_submission_package/arc-prize-2025'),
        Path('../dataset/arc-agi-2')
    ]
    
    test_tasks = []
    for data_path in data_paths:
        if data_path.exists():
            test_tasks = load_arc_agi2_test_tasks(data_path)
            if test_tasks:
                break
    
    if not test_tasks:
        print("❌ No ARC-AGI-2 test tasks found")
        return
    
    print(f"\n📊 Found {len(test_tasks)} test tasks")
    
    # Evaluate
    results = []
    all_zeros_count = 0
    above_baseline_count = 0
    perfect_solves = 0
    
    print("\n🔍 Evaluating tasks...")
    for task in tqdm(test_tasks[:100], desc="Testing"):  # Test first 100 tasks
        task_id = task['id']
        
        # Use first test example
        if not task['test']:
            continue
            
        test_input = task['test'][0]['input']
        
        # Preprocess
        input_tensor = preprocess_grid(test_input).unsqueeze(0).to(DEVICE)
        
        # Forward pass
        with torch.no_grad():
            output, halt_probs = model(input_tensor)
        
        # Get predictions
        predictions = postprocess_output(output, (len(test_input), len(test_input[0])))
        
        # Check if all zeros
        pred_array = np.array(predictions)
        unique_values = np.unique(pred_array)
        
        if len(unique_values) == 1 and unique_values[0] == 0:
            all_zeros_count += 1
        
        # Store result
        result = {
            'task_id': task_id,
            'prediction': predictions,
            'unique_values': unique_values.tolist(),
            'num_unique': len(unique_values)
        }
        results.append(result)
        
        # If we have ground truth (for validation set)
        if 'output' in task['test'][0]:
            target = task['test'][0]['output']
            accuracy = calculate_pixel_accuracy(predictions, target)
            baseline = calculate_zero_baseline(target)
            
            result['accuracy'] = accuracy
            result['baseline'] = baseline
            result['above_baseline'] = accuracy > baseline + 0.1
            
            if accuracy > baseline + 0.1:
                above_baseline_count += 1
            if accuracy == 1.0:
                perfect_solves += 1
    
    # Analysis
    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)
    
    print(f"\n📊 Prediction Diversity:")
    print(f"   Tasks outputting all zeros: {all_zeros_count}/{len(results)} ({all_zeros_count/len(results)*100:.1f}%)")
    
    # Count unique prediction patterns
    unique_patterns = {}
    for result in results:
        pattern = tuple(result['unique_values'])
        unique_patterns[pattern] = unique_patterns.get(pattern, 0) + 1
    
    print(f"   Unique prediction patterns: {len(unique_patterns)}")
    print(f"\n   Top patterns:")
    for pattern, count in sorted(unique_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"      {pattern}: {count} tasks ({count/len(results)*100:.1f}%)")
    
    # Check specific outputs
    print(f"\n🔍 Sample Predictions (first 5 tasks):")
    for i, result in enumerate(results[:5]):
        print(f"\n   Task {result['task_id']}:")
        print(f"      Unique values: {result['unique_values']}")
        if 'accuracy' in result:
            print(f"      Accuracy: {result['accuracy']:.1%}")
            print(f"      Baseline: {result['baseline']:.1%}")
            print(f"      Above baseline: {result['above_baseline']}")
    
    # Final verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    
    if all_zeros_count == len(results):
        print("❌ Model still outputs ALL ZEROS - no learning beyond step 7000")
        print("   The model has not learned to generate meaningful outputs")
    elif all_zeros_count > len(results) * 0.9:
        print("⚠️  Model mostly outputs zeros (>90%) - minimal learning")
        print("   Some variation but still dominated by zero predictions")
    elif len(unique_patterns) > 5:
        print("✅ Model shows output diversity - has learned beyond baseline!")
        print(f"   {len(unique_patterns)} different output patterns detected")
        if above_baseline_count > 0:
            print(f"   {above_baseline_count} tasks above baseline accuracy")
    else:
        print("⚠️  Limited output diversity - some learning but restricted")
    
    # Save results
    output_file = 'agi2_evaluation_step193000.json'
    with open(output_file, 'w') as f:
        json.dump({
            'checkpoint': str(checkpoint_path),
            'step': 193000,
            'num_tasks': len(results),
            'all_zeros_count': all_zeros_count,
            'unique_patterns': len(unique_patterns),
            'above_baseline': above_baseline_count,
            'perfect_solves': perfect_solves,
            'results': results[:20]  # Save first 20 for inspection
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    return results

if __name__ == "__main__":
    evaluate_latest_checkpoint()