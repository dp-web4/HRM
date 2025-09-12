#!/usr/bin/env python3
"""
Simplified GPU Load Test - Focus on actual computation
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
from typing import Dict, List

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from compression.integrated_h_l_system import IntegratedHLSystem
from groot_integration.sleep_cycle_training import Experience, ExperienceMemory
from context.reality_context_4k import RealityContext4K


def monitor_gpu():
    """Simple GPU monitoring."""
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            'allocated': mem_allocated,
            'reserved': mem_reserved,
            'total': mem_total,
            'util_pct': (mem_allocated / mem_total) * 100
        }
    return {'allocated': 0, 'reserved': 0, 'total': 0, 'util_pct': 0}


def generate_batch_observations(batch_size: int, device: str) -> Dict[str, torch.Tensor]:
    """Generate a batch of realistic observations."""
    return {
        'visual': torch.randn(batch_size, 3, 224, 224, device=device),
        'depth': torch.randn(batch_size, 1, 224, 224, device=device),
        'audio': torch.randn(batch_size, 1024, device=device),
        'tactile': torch.randn(batch_size, 128, device=device),
        'proprioceptive': torch.randn(batch_size, 64, device=device),
        'batch_size': batch_size
    }


def run_inference_benchmark(hl_system, duration_seconds: int = 30, batch_size: int = 8):
    """Run continuous inference to stress GPU."""
    print(f"\n🏃 Running inference benchmark ({duration_seconds}s, batch={batch_size})...")
    
    device = next(hl_system.parameters()).device
    start_time = time.time()
    step = 0
    inference_times = []
    
    while (time.time() - start_time) < duration_seconds:
        # Generate batch
        obs = generate_batch_observations(batch_size, device)
        
        # Time inference
        inf_start = torch.cuda.Event(enable_timing=True)
        inf_end = torch.cuda.Event(enable_timing=True)
        
        inf_start.record()
        with torch.no_grad():
            output = hl_system(obs, return_detailed=True)
            action = output['action']
        inf_end.record()
        
        # Synchronize and get time
        torch.cuda.synchronize()
        inference_time = inf_start.elapsed_time(inf_end)
        inference_times.append(inference_time)
        
        step += batch_size
        
        # Report progress
        if step % 100 == 0:
            gpu = monitor_gpu()
            avg_time = np.mean(inference_times[-10:]) if len(inference_times) > 10 else inference_time
            print(f"  Step {step}: {avg_time:.1f}ms/batch, "
                  f"GPU Mem: {gpu['allocated']:.1f}/{gpu['total']:.1f}GB ({gpu['util_pct']:.0f}%)")
    
    total_time = time.time() - start_time
    throughput = step / total_time
    
    print(f"\n✅ Inference Results:")
    print(f"   Total samples: {step}")
    print(f"   Throughput: {throughput:.1f} samples/sec")
    print(f"   Avg latency: {np.mean(inference_times):.2f}ms per batch")
    print(f"   Min/Max: {np.min(inference_times):.2f}/{np.max(inference_times):.2f}ms")
    
    return inference_times


def run_training_benchmark(hl_system, num_iterations: int = 100, batch_size: int = 16):
    """Run training to stress GPU."""
    print(f"\n🎓 Running training benchmark ({num_iterations} iterations, batch={batch_size})...")
    
    device = next(hl_system.parameters()).device
    optimizer = torch.optim.AdamW(hl_system.parameters(), lr=1e-4)
    
    losses = []
    grad_norms = []
    backward_times = []
    
    for i in range(num_iterations):
        # Generate batch
        obs = generate_batch_observations(batch_size, device)
        target_actions = torch.randn(batch_size, 19, device=device)
        
        # Time backward pass
        back_start = torch.cuda.Event(enable_timing=True)
        back_end = torch.cuda.Event(enable_timing=True)
        
        # Forward
        output = hl_system(obs, return_detailed=True)
        predicted_actions = output['action']
        
        # Loss
        action_loss = F.mse_loss(predicted_actions, target_actions)
        
        # Add compression loss if available
        if 'compression_metrics' in output:
            metrics = output['compression_metrics']
            compression_penalty = 1.0 - metrics.information_retained
            total_loss = action_loss + 0.1 * compression_penalty
        else:
            total_loss = action_loss
        
        # Backward
        back_start.record()
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(hl_system.parameters(), 1.0)
        
        optimizer.step()
        back_end.record()
        
        # Get timing
        torch.cuda.synchronize()
        backward_time = back_start.elapsed_time(back_end)
        
        losses.append(total_loss.item())
        grad_norms.append(grad_norm.item())
        backward_times.append(backward_time)
        
        # Report progress
        if (i + 1) % 20 == 0:
            gpu = monitor_gpu()
            avg_loss = np.mean(losses[-20:])
            avg_time = np.mean(backward_times[-20:])
            print(f"  Iter {i+1}: Loss={avg_loss:.4f}, "
                  f"Backward={avg_time:.1f}ms, "
                  f"GPU Mem: {gpu['allocated']:.1f}GB")
    
    print(f"\n✅ Training Results:")
    print(f"   Final loss: {losses[-1]:.4f}")
    print(f"   Loss reduction: {((losses[0] - losses[-1])/losses[0]*100):.1f}%")
    print(f"   Avg backward time: {np.mean(backward_times):.2f}ms")
    print(f"   Avg gradient norm: {np.mean(grad_norms):.3f}")
    
    return losses, backward_times


def run_mixed_workload(hl_system, duration_seconds: int = 60):
    """Run mixed inference and training workload."""
    print(f"\n🔀 Running mixed workload for {duration_seconds}s...")
    
    device = next(hl_system.parameters()).device
    optimizer = torch.optim.AdamW(hl_system.parameters(), lr=1e-4)
    
    start_time = time.time()
    step = 0
    train_step = 0
    
    inference_times = []
    training_times = []
    
    while (time.time() - start_time) < duration_seconds:
        # Alternate between inference and training
        
        # Inference batch (larger)
        inf_batch_size = 32
        obs = generate_batch_observations(inf_batch_size, device)
        
        inf_start = torch.cuda.Event(enable_timing=True)
        inf_end = torch.cuda.Event(enable_timing=True)
        
        inf_start.record()
        with torch.no_grad():
            _ = hl_system(obs, return_detailed=False)
        inf_end.record()
        
        torch.cuda.synchronize()
        inference_times.append(inf_start.elapsed_time(inf_end))
        step += inf_batch_size
        
        # Training batch (smaller)
        if step % 64 == 0:  # Train every 2 inference batches
            train_batch_size = 16
            obs = generate_batch_observations(train_batch_size, device)
            target_actions = torch.randn(train_batch_size, 19, device=device)
            
            train_start = torch.cuda.Event(enable_timing=True)
            train_end = torch.cuda.Event(enable_timing=True)
            
            train_start.record()
            
            # Forward + Backward
            output = hl_system(obs, return_detailed=False)
            loss = F.mse_loss(output['action'], target_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_end.record()
            torch.cuda.synchronize()
            training_times.append(train_start.elapsed_time(train_end))
            train_step += 1
        
        # Report progress
        if step % 256 == 0:
            gpu = monitor_gpu()
            avg_inf = np.mean(inference_times[-10:]) if inference_times else 0
            avg_train = np.mean(training_times[-5:]) if training_times else 0
            elapsed = time.time() - start_time
            print(f"  [{elapsed:.0f}s] Processed {step} samples, "
                  f"Inf={avg_inf:.1f}ms, Train={avg_train:.1f}ms, "
                  f"GPU: {gpu['util_pct']:.0f}%")
    
    print(f"\n✅ Mixed Workload Results:")
    print(f"   Total inference samples: {step}")
    print(f"   Total training steps: {train_step}")
    print(f"   Avg inference: {np.mean(inference_times):.2f}ms")
    print(f"   Avg training: {np.mean(training_times):.2f}ms")


def stress_test_gpu():
    """Run full GPU stress test."""
    
    if not torch.cuda.is_available():
        print("❌ No GPU available!")
        return
    
    print("\n" + "="*80)
    print("🔥 GPU STRESS TEST - H↔L SYSTEM")
    print("="*80)
    
    # System info
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"\n💻 GPU: {torch.cuda.get_device_name()}")
    print(f"   Memory: {gpu_props.total_memory / 1e9:.1f} GB")
    print(f"   CUDA Cores: {gpu_props.multi_processor_count * 128}")  # Approximate
    print(f"   Compute Capability: {gpu_props.major}.{gpu_props.minor}")
    
    # Initialize system
    print("\n⚙️ Initializing H↔L System...")
    device = "cuda"
    hl_system = IntegratedHLSystem(
        context_dim=4096,
        compressed_dim=256,
        action_dim=19,
        compression_type="hybrid",
        device=device
    )
    
    param_count = sum(p.numel() for p in hl_system.parameters())
    print(f"   Model parameters: {param_count/1e6:.1f}M")
    
    initial_gpu = monitor_gpu()
    print(f"   Initial GPU memory: {initial_gpu['allocated']:.2f} GB")
    
    # Warm up
    print("\n⏳ Warming up GPU...")
    for _ in range(10):
        obs = generate_batch_observations(8, device)
        _ = hl_system(obs)
    torch.cuda.synchronize()
    
    # Test 1: Pure Inference
    print("\n" + "-"*60)
    print("TEST 1: PURE INFERENCE")
    print("-"*60)
    inf_times = run_inference_benchmark(hl_system, duration_seconds=30, batch_size=16)
    
    # Test 2: Pure Training
    print("\n" + "-"*60)
    print("TEST 2: PURE TRAINING")
    print("-"*60)
    losses, back_times = run_training_benchmark(hl_system, num_iterations=100, batch_size=32)
    
    # Test 3: Mixed Workload
    print("\n" + "-"*60)
    print("TEST 3: MIXED WORKLOAD")
    print("-"*60)
    run_mixed_workload(hl_system, duration_seconds=60)
    
    # Final stats
    print("\n" + "="*80)
    print("📊 FINAL GPU STATISTICS")
    print("="*80)
    
    final_gpu = monitor_gpu()
    print(f"\n💾 Memory Usage:")
    print(f"   Initial: {initial_gpu['allocated']:.2f} GB")
    print(f"   Final: {final_gpu['allocated']:.2f} GB")
    print(f"   Reserved: {final_gpu['reserved']:.2f} GB")
    print(f"   Peak (estimated): {final_gpu['reserved']:.2f} GB")
    
    # Calculate FLOPS (rough estimate)
    batch_size = 16
    context_dim = 4096
    compressed_dim = 256
    
    # Rough FLOPS calculation for one forward pass
    # Context encoding: ~batch * context_dim^2
    # Compression: ~batch * context_dim * compressed_dim
    # Action generation: ~batch * compressed_dim * 19
    flops_per_sample = context_dim**2 + context_dim*compressed_dim + compressed_dim*19
    samples_per_sec = 1000 / np.mean(inf_times) * batch_size  # Convert ms to samples/sec
    tflops = (flops_per_sample * samples_per_sec) / 1e12
    
    print(f"\n⚡ Performance Estimate:")
    print(f"   Throughput: {samples_per_sec:.0f} samples/sec")
    print(f"   Estimated TFLOPS: {tflops:.2f}")
    print(f"   GPU Utilization: ~{final_gpu['util_pct']:.0f}% memory")
    
    print("\n" + "="*80)
    print("✅ GPU STRESS TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    stress_test_gpu()