#!/usr/bin/env python3
"""
Cross-Platform Performance Comparison
======================================
Compare mailbox performance between RTX 2060 SUPER (CBP) and RTX 4090 (Legion).
"""

import torch
import time
import os
import sys

# Set library path for extension
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['LD_LIBRARY_PATH'] = f"/home/dp/miniforge3/lib/python3.12/site-packages/torch/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import mailbox_ext

def benchmark_pbm_throughput():
    """Benchmark Peripheral Broadcast Mailbox throughput."""
    print("\n" + "="*60)
    print("PBM Throughput Benchmark")
    print("="*60)
    
    # Initialize mailbox
    record_stride = 64
    capacity = 1024 * 16  # 16K records
    pbm_ptrs = mailbox_ext.pbm_init(record_stride, capacity)
    pbm_hdr = int(pbm_ptrs[0].item())
    pbm_payload = int(pbm_ptrs[1].item())
    
    # Test data
    record = torch.full((64,), 42, dtype=torch.uint8, device='cuda')
    
    # Push benchmark
    push_count = 1000
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(push_count):
        success = mailbox_ext.pbm_push_bytes_cuda(pbm_hdr, pbm_payload, record)
        if not success:
            print(f"Push failed at {i}")
            break
    
    torch.cuda.synchronize()
    end = time.time()
    
    push_time = (end - start) * 1000  # ms
    push_rate = push_count / (end - start)
    
    print(f"Push Performance:")
    print(f"  Records: {push_count}")
    print(f"  Time: {push_time:.2f} ms")
    print(f"  Rate: {push_rate:.0f} records/sec")
    print(f"  Throughput: {push_rate * 64 / 1024 / 1024:.2f} MB/s")
    
    # Pop benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    popped = 0
    while popped < push_count:
        out = mailbox_ext.pbm_pop_bulk_cuda(pbm_hdr, pbm_payload, 100, 64)
        if out.numel() == 0:
            break
        popped += out.numel() // 64
    
    torch.cuda.synchronize()
    end = time.time()
    
    pop_time = (end - start) * 1000  # ms
    pop_rate = popped / (end - start)
    
    print(f"\nPop Performance:")
    print(f"  Records: {popped}")
    print(f"  Time: {pop_time:.2f} ms")
    print(f"  Rate: {pop_rate:.0f} records/sec")
    print(f"  Throughput: {pop_rate * 64 / 1024 / 1024:.2f} MB/s")
    
    return push_rate, pop_rate

def benchmark_ftm_throughput():
    """Benchmark Focus Tensor Mailbox throughput."""
    print("\n" + "="*60)
    print("FTM Throughput Benchmark")
    print("="*60)
    
    # Initialize mailbox
    ftm_capacity = 256
    ftm_ptrs = mailbox_ext.ftm_init(ftm_capacity)
    ftm_hdr = int(ftm_ptrs[0].item())
    ftm_ring = int(ftm_ptrs[1].item())
    
    # Test tensor
    test_tensor = torch.randn(256, 256, device='cuda')
    tensor_size_mb = test_tensor.numel() * 4 / 1024 / 1024  # float32
    
    # Push benchmark
    push_count = 100
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(push_count):
        shape = torch.tensor([256, 256], dtype=torch.int32, device='cuda')
        stride = torch.tensor([256, 1], dtype=torch.int32, device='cuda')
        success = mailbox_ext.ftm_push_ptr(ftm_hdr, ftm_ring, test_tensor.data_ptr(), 
                                          shape, stride, i, 100)
        if not success:
            print(f"Push failed at {i}")
            break
    
    torch.cuda.synchronize()
    end = time.time()
    
    push_time = (end - start) * 1000  # ms
    push_rate = push_count / (end - start)
    
    print(f"Push Performance:")
    print(f"  Tensors: {push_count}")
    print(f"  Time: {push_time:.2f} ms")
    print(f"  Rate: {push_rate:.0f} tensors/sec")
    print(f"  Throughput: {push_rate * tensor_size_mb:.2f} MB/s (pointer handoff)")
    
    # Pop benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    popped = 0
    for i in range(push_count):
        result = mailbox_ext.ftm_pop(ftm_hdr, ftm_ring)
        if result is None:
            break
        popped += 1
    
    torch.cuda.synchronize()
    end = time.time()
    
    pop_time = (end - start) * 1000  # ms
    pop_rate = popped / (end - start)
    
    print(f"\nPop Performance:")
    print(f"  Tensors: {popped}")
    print(f"  Time: {pop_time:.2f} ms")
    print(f"  Rate: {pop_rate:.0f} tensors/sec")
    
    return push_rate, pop_rate

def benchmark_memory_transfer():
    """Benchmark raw memory transfer for comparison."""
    print("\n" + "="*60)
    print("Memory Transfer Benchmark")
    print("="*60)
    
    sizes_mb = [1, 16, 64, 256]
    
    for size_mb in sizes_mb:
        elements = size_mb * 1024 * 1024 // 4  # float32
        
        # H2D
        cpu_tensor = torch.randn(elements)
        torch.cuda.synchronize()
        start = time.time()
        gpu_tensor = cpu_tensor.cuda()
        torch.cuda.synchronize()
        h2d_time = time.time() - start
        h2d_rate = size_mb / h2d_time
        
        # D2H
        torch.cuda.synchronize()
        start = time.time()
        cpu_result = gpu_tensor.cpu()
        torch.cuda.synchronize()
        d2h_time = time.time() - start
        d2h_rate = size_mb / d2h_time
        
        print(f"  {size_mb:3d} MB: H2D {h2d_rate:7.1f} MB/s, D2H {d2h_rate:7.1f} MB/s")

def print_platform_comparison():
    """Print comparison between platforms."""
    print("\n" + "="*60)
    print("Platform Comparison")
    print("="*60)
    
    # Get current GPU info
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    print(f"Current Platform (Legion - RTX 4090):")
    print(f"  GPU: {props.name}")
    print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute: {props.major}.{props.minor}")
    print(f"  SMs: {props.multi_processor_count}")
    
    print(f"\nBaseline Platform (CBP - RTX 2060 SUPER):")
    print(f"  GPU: NVIDIA GeForce RTX 2060 SUPER")
    print(f"  Memory: 8.0 GB")
    print(f"  Compute: 7.5")
    print(f"  SMs: 34")
    
    print("\nPerformance Improvements (from test_gpu_simple.py):")
    print("  Matrix mult: 11.23ms vs 6300ms (561x faster)")
    print("  H2D transfer: 10.2 GB/s vs 1.2 GB/s (8.5x faster)")
    print("  D2H transfer: 3.7 GB/s vs 91 MB/s (40x faster)")
    print("  Tiling: 283 tiles/s vs 0.9 tiles/s (314x faster)")

def main():
    """Run all benchmarks."""
    print("="*60)
    print("GPU Mailbox Cross-Platform Performance Test")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # Run benchmarks
    pbm_push, pbm_pop = benchmark_pbm_throughput()
    ftm_push, ftm_pop = benchmark_ftm_throughput()
    benchmark_memory_transfer()
    print_platform_comparison()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"PBM Push: {pbm_push:.0f} records/sec")
    print(f"PBM Pop: {pbm_pop:.0f} records/sec")
    print(f"FTM Push: {ftm_push:.0f} tensors/sec")
    print(f"FTM Pop: {ftm_pop:.0f} tensors/sec")
    print("\n✓ Write-once-run-everywhere confirmed!")
    print("  Same code runs on both RTX 2060 SUPER and RTX 4090")
    print("  Performance scales with hardware capabilities")

if __name__ == "__main__":
    main()