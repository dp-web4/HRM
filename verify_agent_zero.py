#!/usr/bin/env python3
"""
Verify that the HRM model is outputting all zeros (Agent Zero behavior)
This reproduces the documented results from September 2025
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "training"))

# Import model  
from forum.bucket.hrm_models.hrm import HRM as HierarchicalReasoningModule

# Configuration (from the original training)
MODEL_CONFIG = {
    'hidden_size': 256,
    'num_layers_h': 4,
    'num_layers_l': 3,
    'num_attention_heads': 8,
    'dropout': 0.1,
    'use_bidirectional': True,
    'max_seq_len': 900,
    'num_classes': 12,
    'puzzle_embedding_size': 1024,
    'halt_threshold': 0.95,
    'max_cycles': 8
}

def test_model_outputs():
    """Test if model outputs all zeros"""
    print("=" * 60)
    print("AGENT ZERO VERIFICATION TEST")
    print("Checking if HRM model outputs all zeros")
    print("=" * 60)
    
    # Load model
    checkpoint_path = Path("validation_package/hrm_arc_best.pt")
    if not checkpoint_path.exists():
        # Try kaggle submission package
        checkpoint_path = Path("kaggle_submission_package/hrm-model/hrm_arc_best.pt")
    
    print(f"\n1. Loading checkpoint: {checkpoint_path}")
    device = torch.device('cpu')  # Use CPU for consistency
    
    model = HierarchicalReasoningModule(MODEL_CONFIG).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"   Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Test with various inputs
    print("\n2. Testing with different inputs:")
    
    test_cases = [
        ("All zeros", torch.zeros(1, 900, dtype=torch.long)),
        ("All ones", torch.ones(1, 900, dtype=torch.long)),
        ("Random values", torch.randint(0, 10, (1, 900))),
        ("Specific pattern", torch.tensor([[1,2,3,4,5] * 180]).reshape(1, 900))
    ]
    
    all_predictions = []
    
    with torch.no_grad():
        for name, input_tensor in test_cases:
            output, halt_probs = model(input_tensor.to(device))
            predictions = output.argmax(dim=-1)
            unique_preds = torch.unique(predictions)
            
            print(f"\n   {name}:")
            print(f"   - Input shape: {input_tensor.shape}")
            print(f"   - Output shape: {output.shape}")
            print(f"   - Unique predictions: {unique_preds.cpu().tolist()}")
            print(f"   - First 20 predictions: {predictions[0][:20].cpu().tolist()}")
            
            # Check first position logits
            first_logits = output[0, 0, :].cpu()
            print(f"   - First position logits: {first_logits[:5].tolist()}")
            print(f"   - Class with highest logit: {first_logits.argmax().item()}")
            
            all_predictions.append(predictions)
    
    # Check if all outputs are identical
    print("\n3. Checking if outputs are input-invariant:")
    all_same = True
    for i in range(1, len(all_predictions)):
        if not torch.equal(all_predictions[0], all_predictions[i]):
            all_same = False
            break
    
    if all_same:
        print("   ❌ CONFIRMED: Model outputs are IDENTICAL for all inputs!")
        print("   This is Agent Zero behavior - complete input invariance.")
    else:
        print("   ✓ Model outputs vary with input")
    
    # Calculate what accuracy would be on sparse grids
    print("\n4. Zero-baseline accuracy calculation:")
    print("   For ARC grids that are ~80% zeros:")
    print("   - If model outputs all zeros: ~80% pixel accuracy")
    print("   - Reported ARC-AGI-2 accuracy: 18-34%")
    print("   - This matches sparse grid statistics")
    
    # Summary
    print("\n" + "=" * 60)
    print("VERDICT:")
    
    unique_all = set()
    for pred in all_predictions:
        unique_all.update(pred.cpu().flatten().tolist())
    
    if len(unique_all) == 1 and 0 in unique_all:
        print("✅ AGENT ZERO CONFIRMED!")
        print("The model outputs all zeros regardless of input.")
        print("The reported 18-34% accuracy is from zero-baseline on sparse grids.")
    else:
        print(f"Model outputs {len(unique_all)} unique values: {sorted(unique_all)}")
        if len(unique_all) <= 2:
            print("⚠️  Near-constant output detected (Agent Zero tendencies)")
    
    print("=" * 60)
    
    return len(unique_all) == 1 and 0 in unique_all

if __name__ == "__main__":
    is_agent_zero = test_model_outputs()
    
    if is_agent_zero:
        print("\nThis confirms the documentation:")
        print("- HRM checkpoint at step 7000 outputs all zeros")
        print("- The 71% AGI-1 and 20% AGI-2 scores are from zero-baseline")
        print("- No actual pattern solving is happening")
        print("\nNext steps: Find later checkpoints or continue training")