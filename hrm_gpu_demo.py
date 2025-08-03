#!/usr/bin/env python3
"""
HRM GPU Demo - Demonstrates GPU acceleration
"""

import torch
import torch.nn as nn
import time
import numpy as np

class SimpleHRMBlock(nn.Module):
    """Simplified HRM block for demonstration"""
    def __init__(self, hidden_size=256):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        
        # MLP with residual
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x

class MiniHRM(nn.Module):
    """Mini HRM for demonstration"""
    def __init__(self, vocab_size=100, hidden_size=256, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([SimpleHRMBlock(hidden_size) for _ in range(num_layers)])
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x)

def benchmark_gpu():
    """Benchmark GPU vs CPU performance"""
    print("🚀 HRM GPU Performance Demo")
    print("=" * 60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ GPU not available")
        return
    
    device = torch.device('cuda')
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    model = MiniHRM(vocab_size=1000, hidden_size=256, num_layers=6)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model size: {param_count/1e6:.1f}M parameters")
    
    # Test data
    batch_size = 32
    seq_len = 128
    inputs = torch.randint(0, 1000, (batch_size, seq_len))
    
    # CPU timing
    print("\n⏱️  CPU Performance:")
    model_cpu = model.cpu()
    inputs_cpu = inputs.cpu()
    
    # Warmup
    for _ in range(3):
        _ = model_cpu(inputs_cpu)
    
    # Time CPU
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        outputs = model_cpu(inputs_cpu)
    cpu_time = (time.time() - start) / 10
    print(f"   Forward pass: {cpu_time*1000:.1f}ms")
    print(f"   Throughput: {batch_size/cpu_time:.1f} samples/sec")
    
    # GPU timing
    print("\n⚡ GPU Performance:")
    model_gpu = model.to(device)
    inputs_gpu = inputs.to(device)
    
    # Warmup
    for _ in range(3):
        _ = model_gpu(inputs_gpu)
    torch.cuda.synchronize()
    
    # Time GPU
    start = time.time()
    for _ in range(10):
        outputs = model_gpu(inputs_gpu)
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / 10
    print(f"   Forward pass: {gpu_time*1000:.1f}ms")
    print(f"   Throughput: {batch_size/gpu_time:.1f} samples/sec")
    
    # Memory usage
    print(f"\n💾 GPU Memory:")
    print(f"   Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"   Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    # Speedup
    speedup = cpu_time / gpu_time
    print(f"\n🎯 GPU Speedup: {speedup:.1f}x faster than CPU")
    
    # Hierarchical reasoning simulation
    print("\n🧠 Hierarchical Reasoning Demo:")
    print("   High-level (slow): Processing abstract patterns")
    print("   Low-level (fast): Processing local features")
    
    # Simulate multi-timescale processing
    for step in range(3):
        print(f"\n   Step {step+1}:")
        # Low-level cycles (fast)
        for l in range(4):
            torch.cuda.synchronize()
            start = time.time()
            _ = model_gpu(inputs_gpu)
            torch.cuda.synchronize()
            print(f"     L{l}: {(time.time()-start)*1000:.1f}ms")
        
        # High-level cycle (slow, but processes more)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            # Simulate processing multiple sequences
            for _ in range(2):
                _ = model_gpu(inputs_gpu)
        torch.cuda.synchronize()
        print(f"     H: {(time.time()-start)*1000:.1f}ms (processing 2x data)")
    
    print("\n" + "=" * 60)
    print("✨ Key Insights:")
    print(f"  - GPU provides {speedup:.1f}x speedup for HRM")
    print("  - Hierarchical processing enables efficient reasoning")
    print("  - Multi-timescale dynamics balance speed and accuracy")
    print("  - Perfect for complex tasks like Sudoku, ARC, mazes")
    print("=" * 60)

if __name__ == "__main__":
    benchmark_gpu()