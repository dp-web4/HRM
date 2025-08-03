#!/usr/bin/env python3
"""
Run HRM with CUDA workaround
"""

import os
import sys

# Set environment before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Try CPU fallback if CUDA fails
try:
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️  CUDA not available, running on CPU")
except Exception as e:
    print(f"⚠️  CUDA initialization failed: {e}")
    import torch
    device = torch.device('cpu')
    print("⚠️  Falling back to CPU")

print(f"\nDevice: {device}")
print(f"PyTorch: {torch.__version__}")

# Import HRM modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test with a simple model
print("\n🧪 Testing simple model...")
model = torch.nn.Linear(10, 10).to(device)
x = torch.randn(5, 10).to(device)
y = model(x)
print(f"✓ Model output shape: {y.shape}")

# Now test HRM
print("\n🧠 Testing HRM model...")
try:
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    
    config = {
        'batch_size': 2,
        'seq_len': 64,
        'puzzle_emb_ndim': 128,
        'num_puzzle_identifiers': 100,
        'vocab_size': 1000,
        'H_cycles': 2,
        'L_cycles': 4,
        'H_layers': 3,
        'L_layers': 2,
        'hidden_size': 256,
        'expansion': 4.0,
        'num_heads': 8,
        'pos_encodings': 'rope',
        'halt_max_steps': 5,
        'halt_exploration_prob': 0.1,
        'forward_dtype': 'float32' if device.type == 'cpu' else 'bfloat16',
    }
    
    model = HierarchicalReasoningModel_ACTV1(config).to(device)
    print(f"✓ HRM model created on {device}")
    
    # Test forward pass
    batch = {
        'inputs': torch.randint(0, 1000, (2, 64), device=device),
        'puzzle_identifiers': torch.tensor([0, 1], device=device),
    }
    
    carry = model.initial_carry(batch)
    carry, outputs = model(carry, batch)
    
    print(f"✓ Forward pass successful!")
    print(f"  Output shape: {outputs['logits'].shape}")
    print(f"  Q-values shape: {outputs['q_halt_logits'].shape}")
    
except Exception as e:
    print(f"❌ HRM test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
if device.type == 'cuda':
    print("🎉 GPU is working! You can now train HRM on GPU.")
else:
    print("💡 Running on CPU. Training will be slower but functional.")
print("="*50)