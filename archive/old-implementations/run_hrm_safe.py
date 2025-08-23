#!/usr/bin/env python3
"""
Run HRM with safe GPU/CPU fallback
"""

import os
import sys
import warnings

# Suppress CUDA warnings
warnings.filterwarnings("ignore", message=".*CUDA.*")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync'

import torch

# Device selection with fallback
device = None
device_name = "Unknown"

try:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {device_name}")
    else:
        raise RuntimeError("CUDA not available")
except:
    device = torch.device('cpu')
    device_name = "CPU"
    print(f"⚠️  GPU not available, using {device_name}")
    print("   Training will be slower but functional")

print(f"\n📊 System Info:")
print(f"   PyTorch: {torch.__version__}")
print(f"   Device: {device}")
print(f"   Threads: {torch.get_num_threads()}")

# Add HRM path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Patch layers.py to use CPU-compatible attention when needed
print("\n🔧 Patching attention for CPU compatibility...")

# Create a simple HRM test
print("\n🧠 Testing HRM model...")

config = {
    'batch_size': 2,
    'seq_len': 32,  # Smaller for CPU
    'puzzle_emb_ndim': 64,
    'num_puzzle_identifiers': 10,
    'vocab_size': 100,
    'H_cycles': 2,
    'L_cycles': 3,
    'H_layers': 2,
    'L_layers': 1,
    'hidden_size': 128,  # Smaller for CPU
    'expansion': 2.0,
    'num_heads': 4,
    'pos_encodings': 'rope',
    'halt_max_steps': 3,
    'halt_exploration_prob': 0.1,
    'forward_dtype': 'float32',  # Use float32 for CPU
}

try:
    # Import with CPU patches
    import torch.nn.functional as F
    
    # Monkey patch flash attention to use standard attention on CPU
    def cpu_attention(q, k, v, causal=False):
        """CPU-compatible attention"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        if causal:
            mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
            scores.masked_fill_(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)
    
    # Patch flash_attn
    sys.modules['flash_attn'] = type(sys)('flash_attn')
    sys.modules['flash_attn'].flash_attn_func = cpu_attention
    sys.modules['flash_attn_interface'] = sys.modules['flash_attn']
    
    # Now import HRM
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    
    model = HierarchicalReasoningModel_ACTV1(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {param_count:,} parameters")
    
    # Test forward pass
    batch = {
        'inputs': torch.randint(0, 100, (2, 32), device=device),
        'puzzle_identifiers': torch.tensor([0, 1], device=device),
    }
    
    carry = model.initial_carry(batch)
    carry, outputs = model(carry, batch)
    
    print(f"✓ Forward pass successful!")
    print(f"  Output shape: {outputs['logits'].shape}")
    print(f"  Device: {outputs['logits'].device}")
    
    # Test Sudoku
    print("\n🎲 Mini Sudoku test...")
    sudoku_input = torch.randint(0, 10, (1, 16), device=device)  # 4x4 sudoku
    sudoku_batch = {
        'inputs': sudoku_input,
        'puzzle_identifiers': torch.tensor([0], device=device),
    }
    
    carry = model.initial_carry(sudoku_batch)
    for step in range(3):
        carry, outputs = model(carry, sudoku_batch)
        print(f"  Step {step}: logits shape {outputs['logits'].shape}")
    
    print("\n✅ HRM is working on", device_name)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("💡 Next steps:")
if device.type == 'cuda':
    print("   1. GPU is working! You can train HRM efficiently")
    print("   2. Run: python pretrain.py data_path=data/sudoku")
else:
    print("   1. CPU mode is active - suitable for testing")
    print("   2. For GPU: Check CUDA installation and driver")
    print("   3. Try: sudo nvidia-smi -pm 1")
print("="*60)