#!/usr/bin/env python3
"""
Test HRM on GPU with a simple example
"""

import torch
import torch.nn as nn
import time

def test_cuda():
    """Test if CUDA is available and working"""
    print("=" * 60)
    print("🖥️  GPU Test for HRM")
    print("=" * 60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n✓ CUDA Available: {cuda_available}")
    
    if cuda_available:
        # Get GPU info
        gpu_count = torch.cuda.device_count()
        print(f"✓ Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"✓ GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Test GPU computation
        print("\n🧪 Testing GPU computation...")
        
        # Create test tensors
        size = 10000
        device = torch.device('cuda')
        
        # Time CPU vs GPU
        x_cpu = torch.randn(size, size)
        x_gpu = x_cpu.to(device)
        
        # CPU timing
        start = time.time()
        _ = torch.matmul(x_cpu, x_cpu)
        cpu_time = time.time() - start
        
        # GPU timing (with sync)
        start = time.time()
        _ = torch.matmul(x_gpu, x_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"✓ CPU time: {cpu_time:.3f}s")
        print(f"✓ GPU time: {gpu_time:.3f}s")
        print(f"✓ Speedup: {cpu_time/gpu_time:.1f}x")
        
        # Memory check
        print(f"\n💾 GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        
    else:
        print("\n⚠️  No CUDA GPUs found. HRM will run on CPU.")
        print("   This is fine for small experiments but will be slow for training.")
    
    return cuda_available

def test_simple_hrm_module():
    """Test a simple HRM-style module"""
    print("\n" + "=" * 60)
    print("🧠 Simple HRM Module Test")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Simple hierarchical module
    class SimpleHRM(nn.Module):
        def __init__(self, input_size=64, hidden_size=512, output_size=256):
            super().__init__()
            # Low-level (fast) module
            self.low_level = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            
            # High-level (slow) module  
            self.high_level = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                nn.ReLU(),
                nn.Linear(output_size, output_size)
            )
            
        def forward(self, x, cycles=8):
            # Process through multiple cycles
            for _ in range(cycles):
                # Low-level processes fast
                low_features = self.low_level(x)
                
                # High-level processes slow (every 4 cycles)
                if _ % 4 == 0:
                    high_features = self.high_level(low_features)
                    
            return high_features
    
    # Create and test model
    model = SimpleHRM().to(device)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 32
    input_data = torch.randn(batch_size, 64).to(device)
    
    start = time.time()
    output = model(input_data)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"✓ Forward pass completed in {elapsed*1000:.1f}ms")
    print(f"✓ Output shape: {output.shape}")
    
    return model

def main():
    """Run all tests"""
    print("🚀 HRM GPU Setup Test\n")
    
    # Test CUDA
    cuda_ok = test_cuda()
    
    # Test simple HRM module
    model = test_simple_hrm_module()
    
    print("\n" + "=" * 60)
    if cuda_ok:
        print("✅ GPU is ready for HRM experiments!")
        print("   Next steps:")
        print("   1. Build Sudoku dataset: python dataset/build_sudoku_dataset.py")
        print("   2. Train HRM: python pretrain.py")
    else:
        print("⚠️  Running on CPU. Consider using a GPU for faster training.")
    print("=" * 60)

if __name__ == "__main__":
    main()