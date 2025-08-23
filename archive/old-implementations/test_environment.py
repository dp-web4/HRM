#!/usr/bin/env python3
"""Test script to verify HRM environment setup"""

import torch
import sys
import os

# Add HRM to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use no-flash version for WSL2 environments
import models.layers_no_flash as layers
sys.modules['models.layers'] = layers

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1

def test_environment():
    """Test PyTorch and CUDA environment"""
    print("=" * 60)
    print("ENVIRONMENT TEST")
    print("=" * 60)
    
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return torch.cuda.is_available()

def test_hrm_model():
    """Test HRM model initialization and forward pass"""
    print("\n" + "=" * 60)
    print("HRM MODEL TEST")
    print("=" * 60)
    
    # Proper config based on hrm_v1.yaml
    config = {
        'batch_size': 2,
        'seq_len': 16,
        'num_puzzle_identifiers': 100,
        'vocab_size': 512,
        'H_cycles': 2,
        'L_cycles': 2,
        'H_layers': 4,
        'L_layers': 4,
        'hidden_size': 256,  # Reduced for testing
        'expansion': 4,
        'num_heads': 4,
        'pos_encodings': 'rope',
        'halt_max_steps': 16,
        'halt_exploration_prob': 0.1,
        'enable_activation_checkpointing': False
    }
    
    try:
        # Create model
        model = HierarchicalReasoningModel_ACTV1(config)
        print(f"✓ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params / 1e6:.2f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
        
        # Test forward pass on CPU
        batch = {
            'inputs': torch.randint(0, config['vocab_size'], (config['batch_size'], config['seq_len'])),
            'puzzle_identifiers': torch.arange(config['batch_size'])
        }
        
        carry = model.initial_carry(batch)
        carry, outputs = model(carry, batch)
        print(f"✓ CPU forward pass successful")
        if isinstance(outputs, dict):
            print(f"  Output keys: {list(outputs.keys())}")
            if 'logits' in outputs:
                print(f"  Logits shape: {outputs['logits'].shape}")
        else:
            print(f"  Output shape: {outputs.shape}")
        
        # Test on GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # Note: initial_carry must be called after model is on GPU
            with torch.cuda.device(0):
                carry_gpu = model.initial_carry(batch_gpu)
                carry_gpu, outputs_gpu = model(carry_gpu, batch_gpu)
            print(f"✓ GPU forward pass successful")
            if isinstance(outputs_gpu, dict):
                if 'logits' in outputs_gpu:
                    print(f"  GPU logits shape: {outputs_gpu['logits'].shape}")
            else:
                print(f"  GPU output shape: {outputs_gpu.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n🚀 SAGE/HRM Environment Test Suite\n")
    
    # Test environment
    has_cuda = test_environment()
    
    # Test HRM model
    hrm_success = test_hrm_model()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✓ PyTorch installed and working")
    print(f"{'✓' if has_cuda else '✗'} CUDA available and working")
    print(f"{'✓' if hrm_success else '✗'} HRM model working")
    
    if not has_cuda:
        print("\n⚠️  Note: CUDA not available - models will run on CPU only")
        print("   For WSL2, ensure you have:")
        print("   - Windows 11 or Windows 10 version 21H2 or higher")
        print("   - NVIDIA GPU drivers installed on Windows (not WSL)")
        print("   - WSL2 with GPU support enabled")
    
    if hrm_success:
        print("\n✅ Environment is ready for SAGE/HRM experiments!")
    else:
        print("\n❌ Some issues need to be resolved")
    
    return has_cuda and hrm_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)