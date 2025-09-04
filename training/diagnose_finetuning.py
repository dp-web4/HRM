#!/usr/bin/env python3
"""
Diagnose what's wrong with the fine-tuning process
"""
import torch
import sys
sys.path.append('.')
from pathlib import Path
from finetune_reasoning import HierarchicalReasoningModule

def compare_models():
    """Compare original vs fine-tuned models to see what changed"""
    
    # Load original model
    original_path = 'checkpoints/hrm_arc_best.pt'
    original_checkpoint = torch.load(original_path, map_location='cpu', weights_only=False)
    
    # Load fine-tuned model at different stages
    checkpoints = [
        ('checkpoints/hrm_reasoning_agi1_step_3000.pt', '3000 steps'),
        ('checkpoints/hrm_reasoning_agi1_step_6000.pt', '6000 steps')
    ]
    
    print("FINE-TUNING DIAGNOSIS")
    print("=" * 60)
    print(f"Original model: step {original_checkpoint.get('global_step', 'unknown')}")
    print(f"Original loss: {original_checkpoint.get('best_loss', 'N/A')}")
    print()
    
    for checkpoint_path, label in checkpoints:
        if Path(checkpoint_path).exists():
            print(f"Loading {label}...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Check if model structure changed
            original_keys = set(original_checkpoint['model_state_dict'].keys())
            finetuned_keys = set(checkpoint['model_state_dict'].keys())
            
            added_keys = finetuned_keys - original_keys
            removed_keys = original_keys - finetuned_keys
            
            print(f"\n{label}:")
            print(f"  Step: {checkpoint.get('global_step', 'unknown')}")
            print(f"  Config max_cycles: {checkpoint['config']['max_cycles']}")
            print(f"  Added parameters: {added_keys if added_keys else 'None'}")
            print(f"  Removed parameters: {removed_keys if removed_keys else 'None'}")
            
            # Check parameter magnitudes
            print("\n  Parameter magnitude changes:")
            for key in ['token_embedding.weight', 'output.weight', 'h_layers.0.self_attn.in_proj_weight']:
                if key in original_checkpoint['model_state_dict'] and key in checkpoint['model_state_dict']:
                    orig_norm = original_checkpoint['model_state_dict'][key].norm().item()
                    fine_norm = checkpoint['model_state_dict'][key].norm().item()
                    change = (fine_norm - orig_norm) / orig_norm * 100
                    print(f"    {key[:30]:30} | orig: {orig_norm:.2f} | fine: {fine_norm:.2f} | change: {change:+.1f}%")
            
            # Check halt predictor specifically
            print("\n  Halt predictor analysis:")
            halt_keys = [k for k in checkpoint['model_state_dict'].keys() if 'halt' in k]
            for key in halt_keys[:3]:  # Show first 3
                param = checkpoint['model_state_dict'][key]
                print(f"    {key}: shape={param.shape}, norm={param.norm().item():.2f}")

def analyze_training_data():
    """Check the training data being used"""
    from finetune_reasoning import ARCOriginalDataset
    from torch.utils.data import DataLoader
    
    print("\n" + "=" * 60)
    print("TRAINING DATA ANALYSIS")
    print("=" * 60)
    
    # Load dataset
    data_path = Path('../dataset/raw-data/ARC-AGI/data')
    dataset = ARCOriginalDataset(data_path, 'training')
    
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Unique tasks: {len(set(e['task_id'] for e in dataset.examples))}")
    
    # Sample a few examples
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    
    print(f"\nSample batch:")
    print(f"  Input shape: {batch['input'].shape}")
    print(f"  Target shape: {batch['target'].shape}")
    print(f"  Task IDs: {batch['task_id'][:4]}")
    
    # Check if targets are reasonable
    print(f"\nTarget statistics:")
    print(f"  Min value: {batch['target'].min().item()}")
    print(f"  Max value: {batch['target'].max().item()}")
    print(f"  Unique values: {batch['target'].unique().tolist()}")

def test_forward_pass():
    """Test if the model can still do forward passes correctly"""
    from finetune_reasoning import HierarchicalReasoningModule, MODEL_CONFIG
    import numpy as np
    
    print("\n" + "=" * 60)
    print("FORWARD PASS TEST")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with extended reasoning
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(device)
    
    # Load fine-tuned checkpoint
    checkpoint_path = 'checkpoints/hrm_reasoning_agi1_step_3000.pt'
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        print("Model loaded successfully")
        
        # Create dummy input
        batch_size = 2
        seq_len = 900
        dummy_input = torch.randint(0, 10, (batch_size, seq_len)).to(device)
        
        print(f"\nTesting with input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output, halt_probs = model(dummy_input, max_cycles=20)
            
        print(f"Output shape: {output.shape}")
        print(f"Number of reasoning cycles used: {len(halt_probs)}")
        if halt_probs:
            halt_values = [p.mean().item() for p in halt_probs]
            print(f"Halt probabilities: {halt_values}")
            print(f"Average halt prob: {np.mean(halt_values):.3f}")
        
        # Check if output is reasonable
        probs = torch.softmax(output[0, 0], dim=-1)
        print(f"\nOutput distribution (first position):")
        print(f"  Max prob: {probs.max().item():.3f}")
        print(f"  Entropy: {-(probs * probs.log()).sum().item():.3f}")

if __name__ == "__main__":
    compare_models()
    analyze_training_data()
    test_forward_pass()