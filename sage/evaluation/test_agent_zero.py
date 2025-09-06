"""
Test if our SAGE model is behaving like Agent Zero (outputting all zeros)

This script:
1. Loads our best SAGE checkpoint
2. Runs it on ARC-AGI-2 test data
3. Checks if outputs are all zeros
4. Calculates accuracy to verify the ~18-20% score
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directories to path
sys.path.append('..')
sys.path.append('../..')

from core.sage_core import SAGECore
from core.sage_config import SAGEConfig


def load_arc_agi2_data(data_dir: str = "../../dataset/raw-data/ARC-AGI-2") -> List[Dict]:
    """Load ARC-AGI-2 test data"""
    test_data = []
    
    # Check for test files
    test_files = []
    if os.path.exists(data_dir):
        # Look for test JSON files
        for file in os.listdir(data_dir):
            if file.endswith('.json') and 'test' in file.lower():
                test_files.append(os.path.join(data_dir, file))
        
        # If no test files, try to load any JSON files
        if not test_files:
            for file in os.listdir(data_dir):
                if file.endswith('.json'):
                    test_files.append(os.path.join(data_dir, file))
                    if len(test_files) >= 10:  # Limit for testing
                        break
    
    # Load the data
    for file_path in test_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # ARC format with train/test splits
                    if 'test' in data:
                        for test_case in data['test']:
                            test_data.append({
                                'input': test_case['input'],
                                'output': test_case.get('output', None)
                            })
                    elif 'train' in data:
                        # Use train as fallback
                        for train_case in data['train']:
                            test_data.append({
                                'input': train_case['input'],
                                'output': train_case.get('output', None)
                            })
                elif isinstance(data, list):
                    # Direct list of examples
                    test_data.extend(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return test_data


def grid_to_tensor(grid: List[List[int]], max_size: int = 10) -> torch.Tensor:
    """Convert ARC grid to tensor, padding if necessary"""
    grid_array = np.array(grid, dtype=np.int64)
    h, w = grid_array.shape
    
    # Truncate or pad to max_size x max_size
    if h > max_size or w > max_size:
        # Truncate if too large
        grid_array = grid_array[:max_size, :max_size]
    elif h < max_size or w < max_size:
        padded = np.zeros((max_size, max_size), dtype=np.int64)
        padded[:min(h, max_size), :min(w, max_size)] = grid_array[:min(h, max_size), :min(w, max_size)]
        grid_array = padded
    
    return torch.tensor(grid_array, dtype=torch.long)


def analyze_outputs(outputs: torch.Tensor) -> Dict:
    """Analyze model outputs to check for Agent Zero behavior"""
    # outputs shape: [batch, seq, num_classes] or [seq, num_classes]
    
    if outputs.dim() == 3:
        # Get predictions
        predictions = outputs.argmax(dim=-1)  # [batch, seq]
        predictions_flat = predictions.flatten()
    else:
        predictions = outputs.argmax(dim=-1)  # [seq]
        predictions_flat = predictions
    
    # Count unique predictions
    unique_values = torch.unique(predictions_flat)
    
    # Check if all zeros
    all_zeros = (predictions_flat == 0).all().item()
    
    # Calculate statistics
    stats = {
        'all_zeros': all_zeros,
        'unique_values': len(unique_values),
        'unique_list': unique_values.tolist(),
        'zero_percentage': (predictions_flat == 0).float().mean().item() * 100,
        'mode': predictions_flat.mode().values.item(),
        'predictions_sample': predictions_flat[:100].tolist()  # First 100 predictions
    }
    
    return stats


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate pixel-wise accuracy"""
    correct = (predictions == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()


def test_on_arc_agi2():
    """Main evaluation function"""
    print("=" * 60)
    print("SAGE Model - Agent Zero Reality Check")
    print("=" * 60)
    
    # Load model checkpoint
    checkpoint_path = "../training/checkpoints/sage/sage_best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Looking for alternative checkpoints...")
        
        # Search for any checkpoint
        alt_checkpoints = list(Path("..").glob("**/sage*.pt"))
        if alt_checkpoints:
            checkpoint_path = str(alt_checkpoints[0])
            print(f"Found: {checkpoint_path}")
        else:
            print("No SAGE checkpoints found!")
            return
    
    print(f"\n1. Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        # Create config from the saved dict
        from core.sage_config import SAGEPresets
        # Use development config since the checkpoint is from the smaller model
        config = SAGEPresets.development()
        # Update with saved values
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    else:
        print("Warning: No config in checkpoint, using development config")
        from core.sage_config import SAGEPresets
        config = SAGEPresets.development()
    
    # Initialize model
    model = SAGECore(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   Model loaded: {model.get_num_params()/1e6:.1f}M parameters")
    
    # Load ARC-AGI-2 data
    print("\n2. Loading ARC-AGI-2 test data...")
    test_data = load_arc_agi2_data()
    
    if not test_data:
        print("   No test data found. Creating synthetic test...")
        # Create synthetic sparse grids similar to ARC
        test_data = []
        for _ in range(100):
            grid = np.random.choice([0, 0, 0, 0, 1, 2], size=(10, 10))  # 80% zeros
            test_data.append({
                'input': grid.tolist(),
                'output': grid.tolist()  # Use same as output for testing
            })
    
    print(f"   Loaded {len(test_data)} test examples")
    
    # Run evaluation
    print("\n3. Running model on test data...")
    all_predictions = []
    all_targets = []
    output_analyses = []
    
    # Use CPU to avoid CUDA indexing issues
    device = torch.device('cpu')
    model = model.to(device)
    
    with torch.no_grad():
        for i, example in enumerate(test_data[:50]):  # Test on first 50
            # Prepare input
            input_grid = grid_to_tensor(example['input'])
            input_flat = input_grid.flatten().unsqueeze(0).to(device)  # [1, 900]
            
            # Generate random context (since we don't have proper context)
            context = torch.randn(1, config.context_dim).to(device)
            
            # Run model
            outputs = model(input_flat, context)
            
            # Analyze outputs
            analysis = analyze_outputs(outputs['output'])
            output_analyses.append(analysis)
            
            # Get predictions
            predictions = outputs['output'].argmax(dim=-1).cpu()
            all_predictions.append(predictions)
            
            # Store targets if available
            if example.get('output'):
                target_grid = grid_to_tensor(example['output'])
                target_flat = target_grid.flatten()
                all_targets.append(target_flat)
            
            if i % 10 == 0:
                print(f"   Processed {i+1}/{min(50, len(test_data))} examples")
    
    # Analyze results
    print("\n4. Analysis Results:")
    print("-" * 40)
    
    # Check Agent Zero behavior
    all_zeros_count = sum(1 for a in output_analyses if a['all_zeros'])
    avg_zero_percentage = np.mean([a['zero_percentage'] for a in output_analyses])
    avg_unique_values = np.mean([a['unique_values'] for a in output_analyses])
    
    print(f"   Examples outputting ALL zeros: {all_zeros_count}/{len(output_analyses)} "
          f"({all_zeros_count/len(output_analyses)*100:.1f}%)")
    print(f"   Average percentage of zeros: {avg_zero_percentage:.1f}%")
    print(f"   Average unique values per output: {avg_unique_values:.2f}")
    
    # Sample analysis
    print(f"\n   Sample output analysis (first example):")
    first_analysis = output_analyses[0]
    print(f"   - All zeros: {first_analysis['all_zeros']}")
    print(f"   - Unique values: {first_analysis['unique_list']}")
    print(f"   - Zero percentage: {first_analysis['zero_percentage']:.1f}%")
    print(f"   - Mode (most common): {first_analysis['mode']}")
    print(f"   - First 20 predictions: {first_analysis['predictions_sample'][:20]}")
    
    # Calculate accuracy if we have targets
    if all_targets:
        accuracies = []
        for pred, target in zip(all_predictions, all_targets):
            acc = calculate_accuracy(pred.squeeze(), target)
            accuracies.append(acc)
        
        avg_accuracy = np.mean(accuracies)
        print(f"\n5. Accuracy Results:")
        print(f"   Average accuracy: {avg_accuracy*100:.2f}%")
        
        # Calculate what accuracy would be with all zeros
        zero_accuracies = []
        for target in all_targets:
            zero_pred = torch.zeros_like(target)
            zero_acc = calculate_accuracy(zero_pred, target)
            zero_accuracies.append(zero_acc)
        
        avg_zero_accuracy = np.mean(zero_accuracies)
        print(f"   Accuracy if outputting all zeros: {avg_zero_accuracy*100:.2f}%")
        print(f"   Difference: {(avg_accuracy - avg_zero_accuracy)*100:+.2f}%")
    
    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT:")
    
    is_agent_zero = (avg_zero_percentage > 90) or (all_zeros_count > len(output_analyses) * 0.8)
    
    if is_agent_zero:
        print("❌ This model IS behaving like Agent Zero!")
        print("   It's outputting mostly/all zeros regardless of input.")
    elif avg_unique_values < 2:
        print("⚠️  This model shows Agent Zero tendencies!")
        print("   It's outputting very few unique values (near-constant output).")
    else:
        print("✅ This model is NOT Agent Zero!")
        print(f"   It outputs {avg_unique_values:.1f} unique values on average.")
    
    print("=" * 60)
    
    return {
        'is_agent_zero': is_agent_zero,
        'avg_zero_percentage': avg_zero_percentage,
        'avg_unique_values': avg_unique_values,
        'accuracy': avg_accuracy if all_targets else None
    }


if __name__ == "__main__":
    # Run the test
    results = test_on_arc_agi2()
    
    # Save results
    with open("agent_zero_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to agent_zero_test_results.json")