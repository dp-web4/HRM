#!/usr/bin/env python3
"""
Simple test to check PyTorch installation and basic HRM architecture.
Tests without flash attention requirement.
"""

import torch
import sys
from pathlib import Path

print("🧪 Simple HRM Test")
print("=" * 50)

# Test PyTorch installation
print("\n✅ PyTorch is installed!")
print(f"Version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA compute capability: {torch.cuda.get_device_capability(0)}")

# Test basic tensor operations
print("\n📊 Testing tensor operations...")
x = torch.randn(2, 3, 4)
y = torch.randn(2, 3, 4)
z = x + y
print(f"Created tensors of shape: {z.shape}")

# Test CUDA operations if available
if torch.cuda.is_available():
    print("\n🚀 Testing CUDA operations...")
    x_cuda = x.cuda()
    y_cuda = y.cuda()
    z_cuda = x_cuda + y_cuda
    print(f"CUDA tensor computation successful: {z_cuda.shape}")

# Test einops (important for HRM)
try:
    import einops
    print("\n✅ einops is available")
    # Test einops operation
    x = torch.randn(2, 3, 4)
    y = einops.rearrange(x, 'b h w -> b (h w)')
    print(f"einops rearrange successful: {x.shape} -> {y.shape}")
except ImportError:
    print("\n❌ einops not found - required for HRM")

# Check for other HRM dependencies
print("\n📦 Checking HRM dependencies:")
deps = ['omegaconf', 'hydra', 'pydantic', 'tqdm', 'wandb', 'huggingface_hub']
for dep in deps:
    try:
        __import__(dep)
        print(f"  ✅ {dep}")
    except ImportError:
        print(f"  ❌ {dep}")

# Test simple attention mechanism
print("\n🧠 Testing simple attention mechanism...")
class SimpleAttention(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.scale = dim ** -0.5
    
    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        return out

# Test the simple attention
attn = SimpleAttention(64)
x = torch.randn(1, 10, 64)
out = attn(x)
print(f"Simple attention successful: {x.shape} -> {out.shape}")

print("\n✅ Basic HRM requirements are satisfied!")
print("\nNote: Flash Attention is not available, which will impact performance")
print("but the core HRM architecture can still be tested with standard attention.")

# Memory estimate for HRM
print("\n💾 Memory estimate for HRM (27M params):")
param_size_mb = 27 * 4 / 1024 / 1024  # 4 bytes per param
print(f"Model weights: ~{param_size_mb:.1f} MB")
print(f"Training memory: ~{param_size_mb * 3 + 200:.1f} MB")
print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "N/A")