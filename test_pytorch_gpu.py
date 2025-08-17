#!/usr/bin/env python3
"""
Simple PyTorch GPU test for Jetson
"""
import torch
import time

print("🚀 PyTorch GPU Test on Jetson Orin Nano")
print("=" * 50)

# Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
    
    # Memory info
    print(f"\nGPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Simple computation test
    print("\n🧮 Running GPU computation test...")
    
    # Matrix multiplication test
    size = 1024
    a = torch.randn(size, size).cuda()
    b = torch.randn(size, size).cuda()
    
    # Warmup
    for _ in range(3):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Timed run
    start = time.time()
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    gflops = (2 * size**3 * 10) / (elapsed * 1e9)
    print(f"✅ Matrix multiply ({size}x{size}): {elapsed:.3f}s for 10 iterations")
    print(f"   Performance: ~{gflops:.2f} GFLOPS")
    
    # Transformer-style attention test
    print("\n🧠 Testing attention mechanism (no Flash Attention)...")
    batch = 2
    seq_len = 512
    hidden = 256
    heads = 8
    
    q = torch.randn(batch, heads, seq_len, hidden // heads).cuda()
    k = torch.randn(batch, heads, seq_len, hidden // heads).cuda()
    v = torch.randn(batch, heads, seq_len, hidden // heads).cuda()
    
    # Standard scaled dot-product attention
    start = time.time()
    scores = torch.matmul(q, k.transpose(-2, -1)) / (hidden // heads) ** 0.5
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"✅ Attention forward pass: {elapsed*1000:.2f}ms")
    print(f"   Input shape: ({batch}, {heads}, {seq_len}, {hidden//heads})")
    print(f"   Output shape: {output.shape}")
    
    # Memory after operations
    print(f"\nFinal GPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    print("\n🎉 GPU tests completed successfully!")
    
else:
    print("❌ CUDA is not available!")
    print("   PyTorch will run on CPU only.")