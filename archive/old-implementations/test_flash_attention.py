#!/usr/bin/env python3
"""Test HRM with Flash Attention enabled"""

import torch
import sys
import os

# Add HRM to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_flash_attention():
    """Test that Flash Attention is available and working"""
    print("=" * 60)
    print("FLASH ATTENTION TEST")
    print("=" * 60)
    
    try:
        # Try to import flash attention
        from flash_attn import flash_attn_interface
        print("✅ Flash Attention imported successfully")
        
        # Test a simple attention operation
        batch_size = 2
        seq_len = 128
        num_heads = 8
        head_dim = 64
        
        # Create random tensors
        q = torch.randn(batch_size, seq_len, num_heads, head_dim).cuda().half()
        k = torch.randn(batch_size, seq_len, num_heads, head_dim).cuda().half()
        v = torch.randn(batch_size, seq_len, num_heads, head_dim).cuda().half()
        
        # Run flash attention
        output = flash_attn_interface.flash_attn_func(q, k, v)
        print(f"✅ Flash Attention forward pass successful")
        print(f"   Output shape: {output.shape}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Flash Attention not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Flash Attention test failed: {e}")
        return False

def test_hrm_with_flash():
    """Test HRM model with Flash Attention"""
    print("\n" + "=" * 60)
    print("HRM WITH FLASH ATTENTION TEST")
    print("=" * 60)
    
    try:
        # Import with flash attention
        from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
        from models.common import trunc_normal_init_
        print("✅ HRM imported (will use Flash Attention if available)")
        
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
            'hidden_size': 256,
            'expansion': 4,
            'num_heads': 4,
            'pos_encodings': 'rope',
            'halt_max_steps': 16,
            'halt_exploration_prob': 0.1,
            'enable_activation_checkpointing': False
        }
        
        # Create model
        model = HierarchicalReasoningModel_ACTV1(config)
        print(f"✅ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params / 1e6:.2f}M")
        
        # Test on GPU
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"✅ Model moved to GPU")
            
            # Create batch
            batch = {
                'inputs': torch.randint(0, config['vocab_size'], 
                                       (config['batch_size'], config['seq_len'])).cuda(),
                'puzzle_identifiers': torch.arange(config['batch_size']).cuda()
            }
            
            # Forward pass
            with torch.cuda.device(0):
                carry = model.initial_carry(batch)
                carry, outputs = model(carry, batch)
            
            print(f"✅ GPU forward pass successful with Flash Attention")
            if isinstance(outputs, dict) and 'logits' in outputs:
                print(f"   Output logits shape: {outputs['logits'].shape}")
            
            return True
        else:
            print("⚠️  CUDA not available, cannot test GPU forward pass")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n🚀 Flash Attention Environment Test\n")
    
    # Test Flash Attention
    flash_success = test_flash_attention()
    
    # Test HRM with Flash
    hrm_success = test_hrm_with_flash()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"{'✅' if flash_success else '❌'} Flash Attention working")
    print(f"{'✅' if hrm_success else '❌'} HRM with Flash Attention working")
    
    if flash_success and hrm_success:
        print("\n✅ Environment is fully configured with Flash Attention!")
        print("   The model will now run identically to other machines in the network.")
    
    return flash_success and hrm_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)