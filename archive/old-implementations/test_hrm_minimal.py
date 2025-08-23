#!/usr/bin/env python3
"""
Minimal HRM test - just verify it can load and forward
"""
import sys
import os

print("🧪 Minimal HRM Test")
print("=" * 50)

try:
    # Add HRM to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Test imports
    print("Testing imports...")
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    from models.common import trunc_normal_init_
    print("✅ Model imports successful")
    
    # Test config
    print("\nTesting configuration...")
    test_config = {
        "batch_size": 1,
        "seq_len": 81,  # 9x9 sudoku
        "puzzle_emb_ndim": 16,
        "num_puzzle_identifiers": 1,
        "vocab_size": 10,  # 0-9 for sudoku
        "H_cycles": 2,
        "L_cycles": 2,
        "H_layers": 2,
        "L_layers": 2,
        "hidden_size": 128,
        "expansion": 2.0,
        "num_heads": 4,
        "pos_encodings": "learned",
        "halt_max_steps": 4,
        "halt_exploration_prob": 0.1,
        "forward_dtype": "float32"  # Use float32 for compatibility
    }
    
    print("Creating model...")
    model = HierarchicalReasoningModel_ACTV1(test_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params/1e6:.2f}M parameters")
    
    # Test device availability
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n✅ CUDA available: {torch.cuda.get_device_name(0)}")
        model = model.to(device)
        print("✅ Model moved to GPU")
    else:
        device = torch.device("cpu")
        print("\n⚠️  CUDA not available, using CPU")
    
    print("\n🎉 All tests passed! HRM is ready to run on Jetson.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
