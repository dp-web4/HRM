#!/usr/bin/env python3
"""
Memory Manager for SAGE on Edge Devices
Implements memory pooling, KV-cache optimization, and batch processing
Target: <4GB total memory usage
"""

import torch
import gc
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
from collections import OrderedDict

@dataclass
class MemoryPool:
    """Pre-allocated memory pool for tensor reuse"""
    size_mb: int
    dtype: torch.dtype
    device: str
    tensors: List[torch.Tensor]
    available: List[bool]

class MemoryManager:
    """Memory-efficient inference manager for SAGE"""

    def __init__(self, max_memory_mb: int = 4096):
        self.max_memory_mb = max_memory_mb
        self.pools: Dict[str, MemoryPool] = {}
        self.kv_cache: OrderedDict = OrderedDict()
        self.max_cache_size = 100  # Max KV pairs to cache
        self.lock = threading.Lock()

        # Monitor current usage
        self.process = psutil.Process()
        self.baseline_memory = self.get_memory_usage()

        print(f"📊 Memory Manager initialized")
        print(f"   Max memory: {max_memory_mb}MB")
        print(f"   Baseline usage: {self.baseline_memory:.1f}MB")

    def create_pool(self, name: str, size_mb: int,
                    shape: Tuple[int, ...], dtype: torch.dtype = torch.float16):
        """Create a pre-allocated tensor pool"""
        num_tensors = max(1, size_mb * 1024 * 1024 // (np.prod(shape) * torch.finfo(dtype).bits // 8))

        pool = MemoryPool(
            size_mb=size_mb,
            dtype=dtype,
            device='cuda',
            tensors=[],
            available=[]
        )

        # Pre-allocate tensors
        for _ in range(num_tensors):
            tensor = torch.empty(shape, dtype=dtype, device='cuda')
            pool.tensors.append(tensor)
            pool.available.append(True)

        self.pools[name] = pool

        print(f"   Created pool '{name}': {num_tensors} tensors of shape {shape}")

        return pool

    def get_tensor(self, pool_name: str) -> Optional[torch.Tensor]:
        """Get an available tensor from pool"""
        if pool_name not in self.pools:
            return None

        pool = self.pools[pool_name]

        with self.lock:
            for i, available in enumerate(pool.available):
                if available:
                    pool.available[i] = False
                    return pool.tensors[i]

        return None  # No available tensors

    def return_tensor(self, pool_name: str, tensor: torch.Tensor):
        """Return tensor to pool"""
        if pool_name not in self.pools:
            return

        pool = self.pools[pool_name]

        with self.lock:
            try:
                idx = pool.tensors.index(tensor)
                pool.available[idx] = True
                tensor.zero_()  # Clear contents
            except ValueError:
                pass  # Tensor not from this pool

    def optimize_kv_cache(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """Optimized KV-cache for LLM integration"""
        # Check if key exists
        if key in self.kv_cache:
            # Move to end (most recently used)
            self.kv_cache.move_to_end(key)
            return self.kv_cache[key]

        # Add new entry
        self.kv_cache[key] = value

        # Evict oldest if cache full
        if len(self.kv_cache) > self.max_cache_size:
            oldest_key = next(iter(self.kv_cache))
            del self.kv_cache[oldest_key]

        return value

    def batch_process(self, inputs: List[torch.Tensor],
                     process_fn, batch_size: int = 4) -> List[torch.Tensor]:
        """Process inputs in memory-efficient batches"""
        outputs = []

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]

            # Stack into batch tensor
            if batch:
                batch_tensor = torch.stack(batch)

                # Process
                with torch.no_grad():
                    output = process_fn(batch_tensor)

                # Split results
                outputs.extend(torch.unbind(output, dim=0))

                # Force garbage collection between batches
                del batch_tensor
                if i % (batch_size * 4) == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        return outputs

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory statistics"""
        return {
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            'free_mb': (torch.cuda.get_device_properties(0).total_memory -
                       torch.cuda.memory_reserved()) / 1024 / 1024
        }

    def optimize_model_memory(self, model: torch.nn.Module):
        """Optimize model for memory efficiency"""
        # Use half precision
        model = model.half()

        # Enable gradient checkpointing if training
        if model.training:
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()

        # Set to eval mode for inference
        model.eval()

        # Use torch.jit.script for optimization
        try:
            model = torch.jit.script(model)
        except:
            pass  # Some models can't be scripted

        return model

    def monitor_and_adjust(self):
        """Monitor memory usage and adjust if needed"""
        current = self.get_memory_usage()
        gpu_stats = self.get_gpu_memory_usage()

        if current > self.max_memory_mb * 0.9:
            print(f"⚠️ Memory pressure detected: {current:.1f}MB / {self.max_memory_mb}MB")

            # Clear caches
            gc.collect()
            torch.cuda.empty_cache()

            # Reduce KV cache size
            if len(self.kv_cache) > 50:
                # Remove half of cached entries
                for _ in range(len(self.kv_cache) // 2):
                    self.kv_cache.popitem(last=False)

            print(f"   After cleanup: {self.get_memory_usage():.1f}MB")

        return {
            'cpu_usage_mb': current,
            'gpu_allocated_mb': gpu_stats['allocated_mb'],
            'gpu_free_mb': gpu_stats['free_mb'],
            'cache_size': len(self.kv_cache)
        }


def test_memory_manager():
    """Test the memory manager"""
    print("🧪 Testing Memory Manager")

    manager = MemoryManager(max_memory_mb=4096)

    # Create tensor pools
    manager.create_pool('activation', size_mb=512, shape=(1, 512, 768))
    manager.create_pool('attention', size_mb=256, shape=(1, 12, 512, 512))

    # Test tensor allocation
    tensor1 = manager.get_tensor('activation')
    print(f"\n✅ Got tensor from pool: {tensor1.shape if tensor1 is not None else None}")

    if tensor1 is not None:
        # Use tensor
        tensor1.fill_(1.0)

        # Return to pool
        manager.return_tensor('activation', tensor1)
        print("✅ Returned tensor to pool")

    # Test KV cache
    key = "layer_1_attention"
    value = torch.randn(1, 512, 768, dtype=torch.float16, device='cuda')
    cached = manager.optimize_kv_cache(key, value)
    print(f"\n✅ Cached value: {cached.shape}")

    # Test batch processing
    inputs = [torch.randn(3, 224, 224) for _ in range(10)]

    def dummy_process(batch):
        return batch.mean(dim=(1, 2, 3), keepdim=True)

    outputs = manager.batch_process(inputs, dummy_process, batch_size=4)
    print(f"\n✅ Batch processed {len(outputs)} items")

    # Monitor memory
    stats = manager.monitor_and_adjust()
    print(f"\n📊 Memory Stats:")
    print(f"   CPU Usage: {stats['cpu_usage_mb']:.1f}MB")
    print(f"   GPU Allocated: {stats['gpu_allocated_mb']:.1f}MB")
    print(f"   GPU Free: {stats['gpu_free_mb']:.1f}MB")
    print(f"   Cache Size: {stats['cache_size']}")

    if stats['cpu_usage_mb'] < 4096 and stats['gpu_allocated_mb'] < 4096:
        print("\n✅ Memory targets achieved!")
    else:
        print("\n⚠️ Memory usage exceeds target")

if __name__ == "__main__":
    test_memory_manager()
