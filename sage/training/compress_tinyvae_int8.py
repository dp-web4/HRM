#!/usr/bin/env python3
"""
Track 8 Experiment #1: INT8 Quantization of TinyVAE

Goal: Compress TinyVAE from 3.12 MB (FP32) to ~0.78 MB (INT8) with minimal quality loss.

Method: PyTorch Dynamic Quantization
- Quantizes weights to INT8
- Keeps activations in FP32 (dynamic quantization)
- No retraining required
- Expected: 4x model size reduction, <5% quality degradation

Lessons to learn:
1. Does reconstruction quality degrade significantly?
2. What's the inference speed change?
3. Are there numerical stability issues?
4. How does KL divergence behave?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import time
import json
from datetime import datetime

from sage.irp.plugins.tinyvae_irp_plugin import TinyVAE


class QuantizationBenchmark:
    """Benchmark quantized vs original model"""

    def __init__(self, model_original: nn.Module, model_quantized: nn.Module, device: str = "cuda"):
        self.model_original = model_original.to(device).eval()
        self.model_quantized = model_quantized  # Quantized models stay on CPU
        self.device = device

    def measure_size(self, model: nn.Module, name: str) -> dict:
        """Measure model file size"""
        temp_path = Path(f"/tmp/{name}_temp.pt")
        torch.save(model.state_dict(), temp_path)
        size_bytes = temp_path.stat().st_size
        temp_path.unlink()

        return {
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024**2),
            'size_kb': size_bytes / 1024
        }

    def measure_inference_speed(self, model: nn.Module, device: str, num_runs: int = 100) -> dict:
        """Measure inference latency"""
        model.eval()
        dummy_input = torch.randn(1, 3, 64, 64).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Benchmark
        if device == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            with torch.no_grad():
                start.record()
                for _ in range(num_runs):
                    _ = model(dummy_input)
                end.record()

            torch.cuda.synchronize()
            total_time_ms = start.elapsed_time(end)
        else:
            with torch.no_grad():
                start = time.perf_counter()
                for _ in range(num_runs):
                    _ = model(dummy_input)
                end = time.perf_counter()
            total_time_ms = (end - start) * 1000

        avg_time_ms = total_time_ms / num_runs

        return {
            'avg_time_ms': avg_time_ms,
            'fps': 1000 / avg_time_ms,
            'total_runs': num_runs
        }

    def measure_reconstruction_quality(self, dataloader, num_batches: int = 10) -> dict:
        """Compare reconstruction quality on real images"""
        print("\nMeasuring reconstruction quality...")

        total_mse_original = 0.0
        total_mse_quantized = 0.0
        total_kl_original = 0.0
        total_kl_quantized = 0.0
        num_samples = 0

        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= num_batches:
                    break

                batch_size = images.size(0)
                num_samples += batch_size

                # Original model (on GPU)
                images_gpu = images.to(self.device)
                recon_orig, mu_orig, logvar_orig = self.model_original(images_gpu)
                mse_orig = nn.functional.mse_loss(recon_orig, images_gpu).item()
                kl_orig = (-0.5 * torch.sum(1 + logvar_orig - mu_orig.pow(2) - logvar_orig.exp())).item() / (batch_size * 64)

                # Quantized model (on CPU)
                images_cpu = images
                recon_quant, mu_quant, logvar_quant = self.model_quantized(images_cpu)
                mse_quant = nn.functional.mse_loss(recon_quant, images_cpu).item()
                kl_quant = (-0.5 * torch.sum(1 + logvar_quant - mu_quant.pow(2) - logvar_quant.exp())).item() / (batch_size * 64)

                total_mse_original += mse_orig * batch_size
                total_mse_quantized += mse_quant * batch_size
                total_kl_original += kl_orig * batch_size
                total_kl_quantized += kl_quant * batch_size

                if i % 5 == 0:
                    print(f"  Batch {i+1}/{num_batches}: MSE orig={mse_orig:.6f}, quant={mse_quant:.6f}")

        avg_mse_orig = total_mse_original / num_samples
        avg_mse_quant = total_mse_quantized / num_samples
        avg_kl_orig = total_kl_original / num_samples
        avg_kl_quant = total_kl_quantized / num_samples

        mse_degradation = (avg_mse_quant - avg_mse_orig) / avg_mse_orig * 100
        kl_degradation = (avg_kl_quant - avg_kl_orig) / avg_kl_orig * 100

        return {
            'mse_original': avg_mse_orig,
            'mse_quantized': avg_mse_quant,
            'mse_degradation_percent': mse_degradation,
            'kl_original': avg_kl_orig,
            'kl_quantized': avg_kl_quant,
            'kl_degradation_percent': kl_degradation,
            'num_samples': num_samples
        }

    def run_full_benchmark(self, dataloader) -> dict:
        """Run complete benchmark suite"""
        print("\n" + "="*70)
        print("QUANTIZATION BENCHMARK: TinyVAE INT8")
        print("="*70)

        # Size comparison
        print("\n1. Model Size Comparison:")
        print("-" * 70)
        size_orig = self.measure_size(self.model_original, "tinyvae_original")
        size_quant = self.measure_size(self.model_quantized, "tinyvae_quantized")

        print(f"Original (FP32): {size_orig['size_mb']:.2f} MB ({size_orig['size_bytes']:,} bytes)")
        print(f"Quantized (INT8): {size_quant['size_mb']:.2f} MB ({size_quant['size_bytes']:,} bytes)")
        print(f"Compression ratio: {size_orig['size_mb'] / size_quant['size_mb']:.2f}x")
        print(f"Size reduction: {(1 - size_quant['size_mb']/size_orig['size_mb'])*100:.1f}%")

        # Speed comparison
        print("\n2. Inference Speed Comparison:")
        print("-" * 70)
        print("Original model (GPU):")
        speed_orig = self.measure_inference_speed(self.model_original, self.device)
        print(f"  Latency: {speed_orig['avg_time_ms']:.3f} ms")
        print(f"  Throughput: {speed_orig['fps']:.1f} FPS")

        print("\nQuantized model (CPU):")
        speed_quant = self.measure_inference_speed(self.model_quantized, "cpu")
        print(f"  Latency: {speed_quant['avg_time_ms']:.3f} ms")
        print(f"  Throughput: {speed_quant['fps']:.1f} FPS")
        print(f"  Speedup: {speed_orig['avg_time_ms'] / speed_quant['avg_time_ms']:.2f}x")

        # Quality comparison
        print("\n3. Reconstruction Quality Comparison:")
        print("-" * 70)
        quality = self.measure_reconstruction_quality(dataloader)

        print(f"\nMSE (lower is better):")
        print(f"  Original:  {quality['mse_original']:.6f}")
        print(f"  Quantized: {quality['mse_quantized']:.6f}")
        print(f"  Degradation: {quality['mse_degradation_percent']:+.2f}%")

        print(f"\nKL Divergence:")
        print(f"  Original:  {quality['kl_original']:.6f}")
        print(f"  Quantized: {quality['kl_quantized']:.6f}")
        print(f"  Degradation: {quality['kl_degradation_percent']:+.2f}%")

        # Summary
        print("\n" + "="*70)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*70)

        results = {
            'size': {
                'original_mb': size_orig['size_mb'],
                'quantized_mb': size_quant['size_mb'],
                'compression_ratio': size_orig['size_mb'] / size_quant['size_mb'],
                'reduction_percent': (1 - size_quant['size_mb']/size_orig['size_mb'])*100
            },
            'speed': {
                'original_ms': speed_orig['avg_time_ms'],
                'quantized_ms': speed_quant['avg_time_ms'],
                'speedup': speed_orig['avg_time_ms'] / speed_quant['avg_time_ms']
            },
            'quality': quality,
            'timestamp': datetime.now().isoformat()
        }

        # Verdict
        print("\n✓ Model size reduced by {:.1f}%".format(results['size']['reduction_percent']))
        print("✓ Compression ratio: {:.2f}x".format(results['size']['compression_ratio']))

        if abs(quality['mse_degradation_percent']) < 5:
            print("✓ Quality degradation: ACCEPTABLE (<5%)")
        else:
            print("⚠ Quality degradation: {:.1f}% (>5%, needs investigation)".format(quality['mse_degradation_percent']))

        print("\n" + "="*70)

        return results


def main():
    print("\n" + "="*70)
    print("TRACK 8 - Experiment #1: INT8 Quantization")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print("\nHypothesis: Dynamic INT8 quantization will reduce TinyVAE size by ~4x")
    print("            with <5% reconstruction quality degradation.")

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load test data (CIFAR-10)
    print("\nLoading CIFAR-10 dataset for quality benchmarking...")
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor()
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    print(f"Loaded {len(test_dataset)} test images")

    # Create original model
    print("\nCreating original TinyVAE model...")
    model_original = TinyVAE(input_channels=3, latent_dim=64, img_size=64)
    model_original.eval()

    num_params = sum(p.numel() for p in model_original.parameters())
    print(f"Parameters: {num_params:,}")

    # Apply quantization
    print("\nApplying INT8 dynamic quantization...")
    print("  Quantizing: Linear layers")
    model_quantized = torch.quantization.quantize_dynamic(
        model_original.cpu(),
        {nn.Linear},  # Quantize Linear layers only (most parameters)
        dtype=torch.qint8
    )

    print("✓ Quantization complete")

    # Run benchmark
    benchmark = QuantizationBenchmark(model_original, model_quantized, device)
    results = benchmark.run_full_benchmark(test_loader)

    # Save results
    output_dir = Path(__file__).parent / "compression_experiments"
    output_dir.mkdir(exist_ok=True)

    results_path = output_dir / f"tinyvae_int8_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Save quantized model
    model_path = output_dir / "tinyvae_quantized_int8.pt"
    torch.save(model_quantized.state_dict(), model_path)
    print(f"Quantized model saved to: {model_path}")

    # Lessons learned
    print("\n" + "="*70)
    print("LESSONS LEARNED")
    print("="*70)

    print("\n1. Quantization Effectiveness:")
    if results['size']['compression_ratio'] > 3.5:
        print("   ✓ Achieved target ~4x compression")
    else:
        print(f"   ⚠ Only achieved {results['size']['compression_ratio']:.2f}x (expected ~4x)")
        print("   → Possible reason: Only Linear layers quantized, Conv layers remain FP32")

    print("\n2. Quality Tradeoff:")
    if abs(results['quality']['mse_degradation_percent']) < 5:
        print("   ✓ Quality degradation acceptable (<5%)")
    else:
        print(f"   ⚠ Quality degraded by {results['quality']['mse_degradation_percent']:.1f}%")

    print("\n3. Speed Impact:")
    if results['speed']['speedup'] > 1:
        print(f"   ✓ Actually got FASTER: {results['speed']['speedup']:.2f}x speedup!")
        print("   → INT8 matrix ops can be faster than FP32 on CPU")
    elif results['speed']['speedup'] > 0.8:
        print(f"   ✓ Minimal slowdown: {results['speed']['speedup']:.2f}x")
    else:
        print(f"   ⚠ Significant slowdown: {results['speed']['speedup']:.2f}x")

    print("\n4. Next Experiments to Try:")
    print("   - Quantize Conv layers too (torch.quantization.prepare/convert)")
    print("   - Try static quantization (calibration on training data)")
    print("   - Test INT4 quantization (8x compression)")
    print("   - Combine quantization + pruning")

    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
