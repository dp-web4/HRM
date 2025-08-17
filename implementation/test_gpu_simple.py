#!/usr/bin/env python3
"""
Simple GPU test to verify PyTorch CUDA functionality.
Tests basic tensor operations and memory allocation.
"""

import torch
import time

def test_gpu_basics():
    """Test basic GPU functionality."""
    print("=" * 60)
    print("GPU Environment Test")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return False
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Memory info
    props = torch.cuda.get_device_properties(0)
    print(f"\nDevice Properties:")
    print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"  Compute capability: {props.major}.{props.minor}")
    print(f"  Multi-processor count: {props.multi_processor_count}")
    
    return True

def test_tensor_operations():
    """Test basic tensor operations on GPU."""
    print("\n" + "=" * 60)
    print("Tensor Operations Test")
    print("=" * 60)
    
    # Create tensors on GPU
    size = 1024
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    
    print(f"Created two {size}x{size} tensors on GPU")
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Matrix multiplication
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Matrix multiplication time: {elapsed*1000:.2f} ms")
    print(f"Result shape: {c.shape}")
    print(f"Result device: {c.device}")
    
    return True

def test_memory_transfer():
    """Test CPU-GPU memory transfer."""
    print("\n" + "=" * 60)
    print("Memory Transfer Test")
    print("=" * 60)
    
    size = 4096
    
    # Create CPU tensor
    cpu_tensor = torch.randn(size, size)
    print(f"Created {size}x{size} CPU tensor")
    
    # Transfer to GPU
    start = time.time()
    gpu_tensor = cpu_tensor.cuda()
    torch.cuda.synchronize()
    to_gpu_time = time.time() - start
    
    print(f"CPU -> GPU transfer time: {to_gpu_time*1000:.2f} ms")
    print(f"Transfer rate: {cpu_tensor.numel() * 4 / (1024**2) / to_gpu_time:.2f} MB/s")
    
    # Transfer back to CPU
    start = time.time()
    cpu_back = gpu_tensor.cpu()
    torch.cuda.synchronize()
    to_cpu_time = time.time() - start
    
    print(f"GPU -> CPU transfer time: {to_cpu_time*1000:.2f} ms")
    print(f"Transfer rate: {gpu_tensor.numel() * 4 / (1024**2) / to_cpu_time:.2f} MB/s")
    
    # Verify data integrity
    if torch.allclose(cpu_tensor, cpu_back):
        print("✓ Data integrity verified")
    else:
        print("✗ Data integrity check failed!")
        return False
    
    return True

def test_tiling_pattern():
    """Test a simple tiling pattern similar to mailbox concept."""
    print("\n" + "=" * 60)
    print("Tiling Pattern Test")
    print("=" * 60)
    
    # Simulate tiled data processing
    num_tiles = 16
    tile_size = 256
    channels = 64
    
    print(f"Configuration:")
    print(f"  Tiles: {num_tiles}")
    print(f"  Tile size: {tile_size}x{tile_size}")
    print(f"  Channels: {channels}")
    
    # Create tiled input
    tiles = []
    for i in range(num_tiles):
        tile = torch.randn(channels, tile_size, tile_size, device='cuda')
        tiles.append(tile)
    
    print(f"\nCreated {num_tiles} tiles on GPU")
    print(f"Memory per tile: {tiles[0].numel() * 4 / 1024**2:.2f} MB")
    print(f"Total memory: {sum(t.numel() for t in tiles) * 4 / 1024**2:.2f} MB")
    
    # Process tiles (simple convolution simulation)
    kernel = torch.randn(channels, channels, 3, 3, device='cuda')
    processed = []
    
    start = time.time()
    for tile in tiles:
        # Reshape for conv2d: (batch, channels, height, width)
        tile_batch = tile.unsqueeze(0)
        result = torch.nn.functional.conv2d(tile_batch, kernel, padding=1)
        processed.append(result.squeeze(0))
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"\nProcessed {num_tiles} tiles in {elapsed*1000:.2f} ms")
    print(f"Time per tile: {elapsed*1000/num_tiles:.2f} ms")
    print(f"Throughput: {num_tiles/elapsed:.1f} tiles/sec")
    
    return True

def main():
    """Run all tests."""
    tests = [
        ("GPU Basics", test_gpu_basics),
        ("Tensor Operations", test_tensor_operations),
        ("Memory Transfer", test_memory_transfer),
        ("Tiling Pattern", test_tiling_pattern),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all(s for _, s in results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)