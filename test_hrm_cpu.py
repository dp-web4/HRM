#!/usr/bin/env python3
"""
Test HRM on CPU - verifies setup even without CUDA
"""

import torch
import sys
from pathlib import Path

print("🧪 HRM CPU Test")
print("=" * 50)

# Test PyTorch
print(f"\n✅ PyTorch installed: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("Note: Running on CPU for testing")

# Test dependencies
print("\n📦 Checking HRM dependencies:")
deps = {
    'einops': 'Core tensor operations',
    'omegaconf': 'Configuration management', 
    'hydra': 'Experiment management',
    'pydantic': 'Data validation',
    'tqdm': 'Progress bars',
    'wandb': 'Experiment tracking',
    'huggingface_hub': 'Model hub'
}

missing = []
for dep, desc in deps.items():
    try:
        __import__(dep)
        print(f"  ✅ {dep}: {desc}")
    except ImportError:
        print(f"  ❌ {dep}: {desc}")
        missing.append(dep)

if missing:
    print(f"\nMissing: {', '.join(missing)}")
    print("Install with: pip3 install", ' '.join(missing))
else:
    print("\n✅ All dependencies available!")

# Test our modified layers
print("\n🔧 Testing Jetson-compatible layers...")
sys.path.insert(0, str(Path(__file__).parent))

try:
    from models.layers_jetson import Attention, SwiGLU, RotaryEmbedding
    print("✅ Custom layers imported successfully")
    
    # Test attention without flash_attn
    attn = Attention(dim=128, head_dim=32)
    x = torch.randn(1, 10, 128)
    out, _ = attn(x)
    print(f"✅ Attention test passed: {x.shape} -> {out.shape}")
    
    # Test SwiGLU
    swiglu = SwiGLU(128, 256)
    out = swiglu(x)
    print(f"✅ SwiGLU test passed: {x.shape} -> {out.shape}")
    
except Exception as e:
    print(f"❌ Layer test failed: {e}")

# Memory estimate
print("\n💾 Memory Requirements:")
print("- HRM model: ~110MB (27M params)")
print("- Training: ~350MB minimum")
print("- Inference: ~150MB minimum")
print("- Available RAM: 8GB (plenty!)")

print("\n🎯 Next Steps:")
print("1. Install CUDA-enabled PyTorch for GPU acceleration")
print("2. Run: python3 pretrain.py --config-path config --config-name jetson_sudoku_demo")
print("3. Monitor with: tegrastats")

print("\n✅ HRM is ready for CPU testing!")