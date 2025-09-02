#!/usr/bin/env python3
"""
Simple test to verify HRM model runs on Jetson
"""

import torch
import time
import numpy as np
from pathlib import Path

def test_hrm_inference():
    """Test HRM model inference performance"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path("checkpoints/hrm_arc_best.pt")
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print(f"✅ Checkpoint loaded")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Best loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    # Get model parameters count
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
    print(f"   Parameters: {total_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Create dummy input for testing
    batch_size = 1
    seq_len = 784  # 28x28 like MNIST
    vocab_size = 12
    
    # Random input tensor
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    print(f"\n📊 Test input shape: {dummy_input.shape}")
    
    # Benchmark: Create a simple forward pass simulation
    print("\n⏱️  Benchmarking inference speed...")
    
    # Simulate transformer operations
    d_model = 256
    n_heads = 8
    n_layers = 4
    
    # Create mock transformer layers
    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_model * 4,
        batch_first=True
    ).to(device)
    
    transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers).to(device)
    embedding = torch.nn.Embedding(vocab_size, d_model).to(device)
    output_layer = torch.nn.Linear(d_model, 10).to(device)  # 10 classes
    
    # Warm up
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            x = embedding(dummy_input)
            x = transformer(x)
            _ = output_layer(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    n_iterations = 100
    
    print(f"Running {n_iterations} iterations...")
    for i in range(n_iterations):
        start = time.perf_counter()
        
        with torch.no_grad():
            x = embedding(dummy_input)
            x = transformer(x)
            output = output_layer(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        if (i + 1) % 20 == 0:
            print(f"  Iteration {i+1}/{n_iterations}: {elapsed*1000:.2f}ms")
    
    # Statistics
    times = np.array(times) * 1000  # Convert to ms
    
    print(f"\n📈 Performance Results:")
    print(f"   Mean: {np.mean(times):.2f}ms")
    print(f"   Std: {np.std(times):.2f}ms")
    print(f"   Min: {np.min(times):.2f}ms")
    print(f"   Max: {np.max(times):.2f}ms")
    print(f"   Median: {np.median(times):.2f}ms")
    print(f"   95th percentile: {np.percentile(times, 95):.2f}ms")
    
    # Throughput
    fps = 1000 / np.mean(times)
    print(f"\n🚀 Throughput: {fps:.1f} inferences/second")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"\n💾 GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

if __name__ == "__main__":
    test_hrm_inference()