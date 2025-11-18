#!/usr/bin/env python3
"""
Analyze VAE Model Architectures for Track 8 Model Distillation

This script provides detailed parameter analysis of existing VAE models:
1. TinyVAE (from tinyvae_irp_plugin.py)
2. VisionPuzzleVAE (from vision_puzzle_vae.py)

Purpose: Understand current model sizes to plan compression experiments.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from sage.irp.plugins.tinyvae_irp_plugin import TinyVAE
from sage.compression.vision_puzzle_vae import VisionPuzzleVAE


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters in a model"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def analyze_module_parameters(model: nn.Module, name: str = "Model") -> dict:
    """Detailed parameter breakdown by module"""
    print(f"\n{'='*70}")
    print(f"{name} Parameter Analysis")
    print(f"{'='*70}\n")

    total_params = 0
    module_breakdown = {}

    # Analyze each named module
    for module_name, module in model.named_children():
        params = count_parameters(module)
        module_breakdown[module_name] = params
        total_params += params
        print(f"{module_name:20s}: {params:>10,} params")

    print(f"{'-'*70}")
    print(f"{'TOTAL':20s}: {total_params:>10,} params")
    print(f"{'='*70}\n")

    # Calculate sizes
    param_bytes = total_params * 4  # FP32
    param_mb = param_bytes / (1024 ** 2)

    print(f"Memory Footprint:")
    print(f"  FP32: {param_mb:.2f} MB ({total_params * 4:,} bytes)")
    print(f"  FP16: {param_mb/2:.2f} MB ({total_params * 2:,} bytes)")
    print(f"  INT8: {param_mb/4:.2f} MB ({total_params:,} bytes)")
    print(f"  INT4: {param_mb/8:.2f} MB ({total_params//2:,} bytes)")

    return {
        'total_params': total_params,
        'module_breakdown': module_breakdown,
        'size_fp32_mb': param_mb,
        'size_fp16_mb': param_mb / 2,
        'size_int8_mb': param_mb / 4,
        'size_int4_mb': param_mb / 8
    }


def analyze_layer_types(model: nn.Module, name: str = "Model") -> dict:
    """Analyze parameters by layer type (Conv, Linear, Norm, etc)"""
    print(f"\nParameter Breakdown by Layer Type ({name}):")
    print(f"{'-'*70}")

    layer_types = {}

    for module_name, module in model.named_modules():
        module_type = type(module).__name__
        params = count_parameters(module)

        if params > 0 and module_type != type(model).__name__:
            if module_type not in layer_types:
                layer_types[module_type] = {'count': 0, 'params': 0}
            layer_types[module_type]['count'] += 1
            layer_types[module_type]['params'] += params

    # Print sorted by parameter count
    for layer_type, stats in sorted(layer_types.items(), key=lambda x: x[1]['params'], reverse=True):
        print(f"{layer_type:30s}: {stats['count']:>3} layers, {stats['params']:>10,} params")

    return layer_types


def test_inference_speed(model: nn.Module, input_shape: tuple, device: str = "cuda", name: str = "Model") -> dict:
    """Test inference speed"""
    print(f"\nInference Speed Test ({name}):")
    print(f"{'-'*70}")

    model = model.to(device).eval()
    dummy_input = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Time
    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start.record()
            for _ in range(100):
                _ = model(dummy_input)
            end.record()

        torch.cuda.synchronize()
        time_ms = start.elapsed_time(end) / 100
    else:
        import time
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(100):
                _ = model(dummy_input)
            end = time.perf_counter()
        time_ms = (end - start) * 1000 / 100

    print(f"Input shape: {input_shape}")
    print(f"Average inference time: {time_ms:.3f} ms")
    print(f"Throughput: {1000/time_ms:.1f} FPS")

    return {
        'time_ms': time_ms,
        'fps': 1000 / time_ms
    }


def main():
    print("\n" + "="*70)
    print("TRACK 8: MODEL DISTILLATION - Architecture Analysis")
    print("="*70)
    print("\nAnalyzing existing VAE models for compression opportunities...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ===== TinyVAE Analysis =====
    print("\n\n" + "="*70)
    print("MODEL 1: TinyVAE (from tinyvae_irp_plugin.py)")
    print("="*70)
    print("\nUsed in: IRP vision processing")
    print("Input: 64x64x3 images")
    print("Latent: 64-dimensional")

    tiny_vae = TinyVAE(input_channels=3, latent_dim=64, img_size=64)
    tiny_analysis = analyze_module_parameters(tiny_vae, "TinyVAE")
    tiny_layer_types = analyze_layer_types(tiny_vae, "TinyVAE")

    if device == "cuda":
        tiny_speed = test_inference_speed(tiny_vae, (1, 3, 64, 64), device, "TinyVAE")

    # ===== VisionPuzzleVAE Analysis =====
    print("\n\n" + "="*70)
    print("MODEL 2: VisionPuzzleVAE (from vision_puzzle_vae.py)")
    print("="*70)
    print("\nUsed in: Vision → Puzzle encoding")
    print("Input: 224x224x3 images")
    print("Puzzle: 30x30 grid with 10 discrete codes")
    print("Latent: 64-dimensional per spatial position")

    vision_puzzle_vae = VisionPuzzleVAE(latent_dim=64, num_codes=10)
    puzzle_analysis = analyze_module_parameters(vision_puzzle_vae, "VisionPuzzleVAE")
    puzzle_layer_types = analyze_layer_types(vision_puzzle_vae, "VisionPuzzleVAE")

    if device == "cuda":
        puzzle_speed = test_inference_speed(vision_puzzle_vae, (1, 3, 224, 224), device, "VisionPuzzleVAE")

    # ===== Comparison Summary =====
    print("\n\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    print(f"\nModel Sizes:")
    print(f"  TinyVAE:         {tiny_analysis['total_params']:>10,} params ({tiny_analysis['size_fp32_mb']:>6.2f} MB FP32)")
    print(f"  VisionPuzzleVAE: {puzzle_analysis['total_params']:>10,} params ({puzzle_analysis['size_fp32_mb']:>6.2f} MB FP32)")
    print(f"  Ratio:           {puzzle_analysis['total_params']/tiny_analysis['total_params']:>10.1f}x larger")

    if device == "cuda":
        print(f"\nInference Speed (GPU):")
        print(f"  TinyVAE:         {tiny_speed['time_ms']:>6.3f} ms ({tiny_speed['fps']:>6.1f} FPS)")
        print(f"  VisionPuzzleVAE: {puzzle_speed['time_ms']:>6.3f} ms ({puzzle_speed['fps']:>6.1f} FPS)")

    # ===== Compression Opportunities =====
    print("\n\n" + "="*70)
    print("COMPRESSION OPPORTUNITIES")
    print("="*70)

    print("\n1. TinyVAE (294K params):")
    print(f"   Current: {tiny_analysis['size_fp32_mb']:.2f} MB FP32")
    print(f"   INT8:    {tiny_analysis['size_int8_mb']:.2f} MB (4x compression)")
    print(f"   INT4:    {tiny_analysis['size_int4_mb']:.2f} MB (8x compression)")
    print("   Opportunities:")
    print("   - Already uses depthwise separable convs (efficient!)")
    print("   - Could apply quantization (INT8/INT4)")
    print("   - Could reduce latent_dim from 64 to 32 or 16")
    print("   - Could apply pruning to remove unused parameters")

    print(f"\n2. VisionPuzzleVAE ({puzzle_analysis['total_params']:,} params):")
    print(f"   Current: {puzzle_analysis['size_fp32_mb']:.2f} MB FP32")
    print(f"   INT8:    {puzzle_analysis['size_int8_mb']:.2f} MB (4x compression)")
    print(f"   INT4:    {puzzle_analysis['size_int4_mb']:.2f} MB (8x compression)")
    print("   Opportunities:")
    print("   - Uses standard convs (could switch to depthwise separable)")
    print("   - Could apply quantization (INT8/INT4)")
    print("   - Could reduce channel counts in encoder/decoder")
    print("   - VQ codebook is already tiny (10 codes × 64 dims = 640 params)")

    # ===== Jetson Nano Constraints =====
    print("\n\n" + "="*70)
    print("JETSON NANO CONSTRAINTS")
    print("="*70)
    print("\nTarget Platform: Jetson Nano")
    print("  - GPU Memory: 2GB")
    print("  - System RAM: 4GB")
    print("  - Need room for:")
    print("    - Multiple models (vision, audio, etc)")
    print("    - Runtime memory (activations, buffers)")
    print("    - System overhead")

    print("\nTarget Model Sizes (FP16 for GPU inference):")
    print(f"  TinyVAE current (FP16):         {tiny_analysis['size_fp16_mb']:.2f} MB ✓ (already tiny!)")
    print(f"  VisionPuzzleVAE current (FP16): {puzzle_analysis['size_fp16_mb']:.2f} MB ⚠ (could compress)")

    print("\nBudget allocation (conservative):")
    print("  Total GPU memory: 2048 MB")
    print("  Available for models: ~1000 MB (50%)")
    print("  Per-model budget: ~200 MB each (5 models)")
    print("  Conclusion: Both models fit, but compression gives headroom!")

    # ===== Recommended Experiments =====
    print("\n\n" + "="*70)
    print("RECOMMENDED COMPRESSION EXPERIMENTS")
    print("="*70)

    print("\nPriority 1: Quantization (Fast, proven, no retraining)")
    print("  - INT8 quantization (4x compression, <1% accuracy loss)")
    print("  - PyTorch native quantization or TensorRT")
    print("  - Test on both TinyVAE and VisionPuzzleVAE")

    print("\nPriority 2: Architecture Optimization (VisionPuzzleVAE)")
    print("  - Replace standard convs with depthwise separable")
    print("  - Reduce channel counts (e.g., 128→64, 64→32)")
    print("  - Target: 50% parameter reduction")

    print("\nPriority 3: Latent Dimension Reduction (TinyVAE)")
    print("  - Current: 64-dim latents")
    print("  - Test: 32-dim, 16-dim, 8-dim")
    print("  - Measure reconstruction quality vs size")

    print("\nPriority 4: Knowledge Distillation")
    print("  - Train smaller student model to match larger teacher")
    print("  - Useful for VisionPuzzleVAE if architecture changes aren't enough")

    # ===== Save Results =====
    results = {
        'tiny_vae': tiny_analysis,
        'vision_puzzle_vae': puzzle_analysis,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }

    output_path = Path(__file__).parent / "vae_analysis_results.json"
    import json
    with open(output_path, 'w') as f:
        # Convert non-serializable types
        serializable = {}
        for key, val in results.items():
            if isinstance(val, dict):
                serializable[key] = {k: (v.tolist() if hasattr(v, 'tolist') else v)
                                    for k, v in val.items()}
            else:
                serializable[key] = val
        json.dump(serializable, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Start with INT8 quantization experiments (easiest wins)")
    print("2. Benchmark quality vs size tradeoffs")
    print("3. Test on actual Jetson Nano when available")
    print("4. Document all findings in thor_worklog.txt")
    print("\n")


if __name__ == "__main__":
    main()
