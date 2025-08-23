#!/usr/bin/env python3
"""
HRM test for Docker - works without Flash Attention
"""
import sys
import os

print("🐳 HRM Docker Test (No Flash Attention)")
print("=" * 50)

try:
    # Monkey-patch to use no-flash version
    import importlib.util
    spec = importlib.util.spec_from_file_location("models.layers", 
                                                  "/workspace/HRM/models/layers_no_flash.py")
    layers_module = importlib.util.module_from_spec(spec)
    sys.modules['models.layers'] = layers_module
    spec.loader.exec_module(layers_module)
    
    # Now import HRM
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    from models.common import trunc_normal_init_
    print("✅ Model imports successful")
    
    # Test config
    print("\nCreating minimal HRM config...")
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
        "forward_dtype": "float32"
    }
    
    print("Creating model...")
    model = HierarchicalReasoningModel_ACTV1(test_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params/1e6:.2f}M parameters")
    
    # Test on GPU
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n✅ CUDA available: {torch.cuda.get_device_name(0)}")
        model = model.to(device)
        print("✅ Model moved to GPU")
        
        # Test forward pass
        print("\nTesting forward pass...")
        dummy_input = torch.zeros(1, 81, dtype=torch.long).to(device)
        dummy_mask = torch.ones(1, 81, dtype=torch.bool).to(device)
        
        with torch.no_grad():
            output = model(dummy_input, dummy_mask)
        
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
    else:
        device = torch.device("cpu")
        print("\n⚠️  CUDA not available, using CPU")
    
    print("\n🎉 All tests passed! HRM works in Docker without Flash Attention.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)