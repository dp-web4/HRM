#!/usr/bin/env python3
"""
Baseline performance benchmarks for Jetson Orin Nano
Establishes performance metrics before IRP implementation
"""

import torch
import time
import json
import platform
from datetime import datetime
import subprocess
import os

def get_system_info():
    """Gather system information"""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["cuda_capability"] = f"{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}"
    
    # Get Jetson info if available
    try:
        model = subprocess.check_output(["cat", "/proc/device-tree/model"], text=True).strip()
        info["jetson_model"] = model
    except:
        pass
    
    return info

def benchmark_basic_ops():
    """Benchmark basic tensor operations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # Matrix multiplication
    sizes = [256, 512, 1024, 2048]
    for size in sizes:
        x = torch.randn(size, size, device=device)
        y = torch.randn(size, size, device=device)
        
        # Warmup
        for _ in range(3):
            _ = torch.matmul(x, y)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10
        
        results[f"matmul_{size}"] = {
            "time_ms": elapsed * 1000,
            "gflops": (2 * size**3) / (elapsed * 1e9)
        }
    
    # Convolution benchmarks
    batch_sizes = [1, 4, 8]
    for batch in batch_sizes:
        x = torch.randn(batch, 3, 224, 224, device=device)
        conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(device)
        
        # Warmup
        for _ in range(3):
            _ = conv(x)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            y = conv(x)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10
        
        results[f"conv2d_batch{batch}"] = {
            "time_ms": elapsed * 1000,
            "images_per_sec": batch / elapsed
        }
    
    return results

def benchmark_memory():
    """Benchmark memory operations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # Memory bandwidth
    sizes_mb = [10, 50, 100, 500]
    for size_mb in sizes_mb:
        size = int(size_mb * 1024 * 1024 / 4)  # float32
        
        # Host to Device
        x_cpu = torch.randn(size)
        start = time.time()
        x_gpu = x_cpu.to(device)
        torch.cuda.synchronize()
        h2d_time = time.time() - start
        
        # Device to Host
        start = time.time()
        x_back = x_gpu.cpu()
        torch.cuda.synchronize()
        d2h_time = time.time() - start
        
        results[f"memory_{size_mb}MB"] = {
            "h2d_gbps": (size_mb / 1024) / h2d_time,
            "d2h_gbps": (size_mb / 1024) / d2h_time
        }
    
    return results

def benchmark_precision():
    """Compare FP32 vs FP16 performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    size = 1024
    
    # FP32
    x32 = torch.randn(size, size, device=device)
    y32 = torch.randn(size, size, device=device)
    
    start = time.time()
    for _ in range(10):
        z32 = torch.matmul(x32, y32)
    torch.cuda.synchronize()
    fp32_time = (time.time() - start) / 10
    
    # FP16
    x16 = x32.half()
    y16 = y32.half()
    
    start = time.time()
    for _ in range(10):
        z16 = torch.matmul(x16, y16)
    torch.cuda.synchronize()
    fp16_time = (time.time() - start) / 10
    
    results["precision_speedup"] = {
        "fp32_ms": fp32_time * 1000,
        "fp16_ms": fp16_time * 1000,
        "speedup": fp32_time / fp16_time
    }
    
    return results

def main():
    print("=" * 60)
    print("HRM/SAGE Baseline Performance Benchmark - Jetson")
    print("=" * 60)
    
    # System info
    print("\n1. Gathering system information...")
    system_info = get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Basic operations
    print("\n2. Benchmarking basic operations...")
    basic_results = benchmark_basic_ops()
    for op, metrics in basic_results.items():
        print(f"  {op}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.2f}")
    
    # Memory operations
    print("\n3. Benchmarking memory transfers...")
    memory_results = benchmark_memory()
    for op, metrics in memory_results.items():
        print(f"  {op}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.2f}")
    
    # Precision comparison
    print("\n4. Comparing FP32 vs FP16...")
    precision_results = benchmark_precision()
    for key, metrics in precision_results.items():
        print(f"  {key}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.2f}")
    
    # Save results
    results = {
        "system": system_info,
        "basic_ops": basic_results,
        "memory": memory_results,
        "precision": precision_results
    }
    
    output_file = f"baseline_jetson_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print("=" * 60)
    
    # IRP-specific estimates
    print("\nIRP Performance Estimates:")
    print("-" * 40)
    
    # Vision IRP estimate (based on conv2d performance)
    if "conv2d_batch1" in basic_results:
        vision_time = basic_results["conv2d_batch1"]["time_ms"]
        print(f"Vision IRP (per iteration): ~{vision_time * 2:.1f}ms")
        print(f"  With early stop (50%): ~{vision_time:.1f}ms")
        print(f"  Potential savings: {vision_time:.1f}ms per frame")
    
    # Language IRP estimate (based on matmul performance)
    if "matmul_512" in basic_results:
        lang_time = basic_results["matmul_512"]["time_ms"]
        print(f"Language IRP (per token): ~{lang_time:.1f}ms")
        print(f"  With stabilization: ~{lang_time * 0.3:.1f}ms")
        print(f"  Potential savings: {lang_time * 0.7:.1f}ms per token")
    
    print("=" * 60)

if __name__ == "__main__":
    main()