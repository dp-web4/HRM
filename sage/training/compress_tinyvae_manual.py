#!/usr/bin/env python3
"""
Track 8 Experiment #1B: Manual INT8 Quantization of TinyVAE

Since torch.quantization has engine issues, let's learn by implementing
manual weight quantization and measuring the tradeoffs ourselves.

This is actually MORE educational - we'll understand exactly what quantization does!

Method: Manual weight quantization to INT8
- Convert FP32 weights to INT8 range [-128, 127]
- Store scale/zero_point for dequantization
- Measure size reduction and quality impact
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
import numpy as np
from datetime import datetime

from sage.irp.plugins.tinyvae_irp_plugin import TinyVAE


def quantize_tensor(tensor: torch.Tensor, num_bits: int = 8) -> tuple:
    """
    Manually quantize a tensor to INT8

    Returns:
        quantized: INT8 tensor
        scale: scale factor for dequantization
        zero_point: zero point for dequantization
    """
    qmin = -(2**(num_bits-1))  # -128 for INT8
    qmax = 2**(num_bits-1) - 1   # 127 for INT8

    min_val = tensor.min().item()
    max_val = tensor.max().item()

    # Calculate scale and zero_point
    scale = (max_val - min_val) / (qmax - qmin)
    if scale == 0:
        scale = 1.0
    zero_point = qmin - round(min_val / scale)
    zero_point = max(qmin, min(qmax, zero_point))

    # Quantize
    quantized = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax)
    quantized = quantized.to(torch.int8)

    return quantized, scale, zero_point


def dequantize_tensor(quantized: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """Dequantize an INT8 tensor back to FP32"""
    return (quantized.float() - zero_point) * scale


class ManuallyQuantizedTinyVAE(nn.Module):
    """TinyVAE with manually quantized weights"""

    def __init__(self, original_model: TinyVAE):
        super().__init__()
        self.latent_dim = original_model.latent_dim

        # Store architecture (we'll load quantized weights later)
        self.encoder = original_model.encoder
        self.adapt = original_model.adapt
        self.fc_mu = original_model.fc_mu
        self.fc_logvar = original_model.fc_logvar
        self.decoder_input = original_model.decoder_input
        self.decoder = original_model.decoder

        # Will store quantized weights here
        self.quantized_weights = {}
        self.scales = {}
        self.zero_points = {}

    def quantize_weights(self, num_bits: int = 8):
        """Quantize all Linear and Conv2d weights"""
        print("\nQuantizing weights...")
        total_params = 0
        quantized_params = 0

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Quantize weights
                weight = module.weight.data
                weight_q, scale, zero_point = quantize_tensor(weight, num_bits)

                self.quantized_weights[name + '.weight'] = weight_q
                self.scales[name + '.weight'] = scale
                self.zero_points[name + '.weight'] = zero_point

                quantized_params += weight.numel()

                # Quantize bias if exists
                if module.bias is not None:
                    bias = module.bias.data
                    bias_q, scale_b, zero_point_b = quantize_tensor(bias, num_bits)

                    self.quantized_weights[name + '.bias'] = bias_q
                    self.scales[name + '.bias'] = scale_b
                    self.zero_points[name + '.bias'] = zero_point_b

                    quantized_params += bias.numel()

            total_params += sum(p.numel() for p in module.parameters())

        print(f"✓ Quantized {quantized_params:,} / {total_params:,} parameters")
        return quantized_params, total_params

    def dequantize_and_load(self):
        """Dequantize weights and load them back into model"""
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Dequantize weight
                if name + '.weight' in self.quantized_weights:
                    weight_q = self.quantized_weights[name + '.weight']
                    scale = self.scales[name + '.weight']
                    zero_point = self.zero_points[name + '.weight']

                    weight_dq = dequantize_tensor(weight_q, scale, zero_point)
                    module.weight.data = weight_dq.reshape(module.weight.shape)

                # Dequantize bias
                if module.bias is not None and name + '.bias' in self.quantized_weights:
                    bias_q = self.quantized_weights[name + '.bias']
                    scale_b = self.scales[name + '.bias']
                    zero_point_b = self.zero_points[name + '.bias']

                    bias_dq = dequantize_tensor(bias_q, scale_b, zero_point_b)
                    module.bias.data = bias_dq

    def forward(self, x):
        """Forward pass (dequantize weights on-the-fly for now)"""
        # For this experiment, we dequantize once before inference
        # In production, you'd use INT8 kernels directly
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        h = self.adapt(h)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)


def measure_model_size(model: nn.Module, name: str, save_quantized: bool = False) -> dict:
    """Measure actual serialized model size"""
    temp_path = Path(f"/tmp/{name}_temp.pt")

    if save_quantized and hasattr(model, 'quantized_weights'):
        # Save only quantized weights + metadata
        state = {
            'quantized_weights': model.quantized_weights,
            'scales': model.scales,
            'zero_points': model.zero_points
        }
        torch.save(state, temp_path)
    else:
        # Save normal model
        torch.save(model.state_dict(), temp_path)

    size_bytes = temp_path.stat().st_size
    temp_path.unlink()

    return {
        'size_bytes': size_bytes,
        'size_mb': size_bytes / (1024**2),
        'size_kb': size_bytes / 1024
    }


def benchmark_quality(model_original: nn.Module, model_quantized: nn.Module,
                     dataloader, device: str, num_batches: int = 20) -> dict:
    """Compare reconstruction quality"""
    print("\nBenchmarking reconstruction quality...")

    model_original = model_original.to(device).eval()
    model_quantized = model_quantized.to(device).eval()

    total_mse_orig = 0.0
    total_mse_quant = 0.0
    total_kl_orig = 0.0
    total_kl_quant = 0.0
    num_samples = 0

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break

            images = images.to(device)
            batch_size = images.size(0)
            num_samples += batch_size

            # Original model
            recon_orig, mu_orig, logvar_orig = model_original(images)
            mse_orig = nn.functional.mse_loss(recon_orig, images).item()
            kl_orig = (-0.5 * torch.sum(1 + logvar_orig - mu_orig.pow(2) - logvar_orig.exp())).item() / (batch_size * 64)

            # Quantized model
            recon_quant, mu_quant, logvar_quant = model_quantized(images)
            mse_quant = nn.functional.mse_loss(recon_quant, images).item()
            kl_quant = (-0.5 * torch.sum(1 + logvar_quant - mu_quant.pow(2) - logvar_quant.exp())).item() / (batch_size * 64)

            total_mse_orig += mse_orig * batch_size
            total_mse_quant += mse_quant * batch_size
            total_kl_orig += kl_orig * batch_size
            total_kl_quant += kl_quant * batch_size

            if i % 5 == 0:
                print(f"  Batch {i+1}/{num_batches}: MSE diff = {abs(mse_quant - mse_orig):.6f}")

    avg_mse_orig = total_mse_orig / num_samples
    avg_mse_quant = total_mse_quant / num_samples
    avg_kl_orig = total_kl_orig / num_samples
    avg_kl_quant = total_kl_quant / num_samples

    return {
        'mse_original': avg_mse_orig,
        'mse_quantized': avg_mse_quant,
        'mse_diff': avg_mse_quant - avg_mse_orig,
        'mse_degradation_percent': (avg_mse_quant - avg_mse_orig) / avg_mse_orig * 100 if avg_mse_orig > 0 else 0,
        'kl_original': avg_kl_orig,
        'kl_quantized': avg_kl_quant,
        'kl_diff': avg_kl_quant - avg_kl_orig,
        'num_samples': num_samples
    }


def main():
    print("\n" + "="*70)
    print("TRACK 8 - Experiment #1: Manual INT8 Quantization")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print("\nLearning Goals:")
    print("1. Understand quantization by implementing it manually")
    print("2. Measure real compression ratios")
    print("3. Quantify quality degradation")
    print("4. Learn what matters for Jetson Nano deployment")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load test data
    print("\nLoading CIFAR-10 for quality benchmarking...")
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

    print(f"✓ Loaded {len(test_dataset)} test images")

    # Create original model
    print("\nCreating original TinyVAE...")
    model_original = TinyVAE(input_channels=3, latent_dim=64, img_size=64)
    num_params = sum(p.numel() for p in model_original.parameters())
    print(f"✓ Parameters: {num_params:,}")

    # Create quantized version
    print("\nCreating quantized version...")
    model_quantized = ManuallyQuantizedTinyVAE(model_original)
    quant_params, total_params = model_quantized.quantize_weights(num_bits=8)

    # Dequantize weights for inference (simulating INT8 inference)
    model_quantized.dequantize_and_load()
    print("✓ Dequantized weights loaded")

    # Measure sizes
    print("\n" + "="*70)
    print("SIZE COMPARISON")
    print("="*70)

    size_orig = measure_model_size(model_original, "tinyvae_fp32", save_quantized=False)
    size_quant = measure_model_size(model_quantized, "tinyvae_int8", save_quantized=True)

    print(f"\nOriginal model (FP32):")
    print(f"  Size: {size_orig['size_mb']:.2f} MB ({size_orig['size_bytes']:,} bytes)")

    print(f"\nQuantized model (INT8):")
    print(f"  Size: {size_quant['size_mb']:.2f} MB ({size_quant['size_bytes']:,} bytes)")

    compression_ratio = size_orig['size_mb'] / size_quant['size_mb']
    reduction_pct = (1 - size_quant['size_mb']/size_orig['size_mb']) * 100

    print(f"\nCompression:")
    print(f"  Ratio: {compression_ratio:.2f}x")
    print(f"  Reduction: {reduction_pct:.1f}%")

    # Measure quality
    print("\n" + "="*70)
    print("QUALITY COMPARISON")
    print("="*70)

    quality = benchmark_quality(model_original, model_quantized, test_loader, device, num_batches=20)

    print(f"\nReconstruction MSE:")
    print(f"  Original:  {quality['mse_original']:.6f}")
    print(f"  Quantized: {quality['mse_quantized']:.6f}")
    print(f"  Difference: {quality['mse_diff']:+.6f} ({quality['mse_degradation_percent']:+.2f}%)")

    print(f"\nKL Divergence:")
    print(f"  Original:  {quality['kl_original']:.6f}")
    print(f"  Quantized: {quality['kl_quantized']:.6f}")
    print(f"  Difference: {quality['kl_diff']:+.6f}")

    # Results summary
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS")
    print("="*70)

    results = {
        'compression': {
            'original_size_mb': size_orig['size_mb'],
            'quantized_size_mb': size_quant['size_mb'],
            'compression_ratio': compression_ratio,
            'reduction_percent': reduction_pct,
            'quantized_params': quant_params,
            'total_params': total_params
        },
        'quality': quality,
        'timestamp': datetime.now().isoformat(),
        'device': device
    }

    # Verdict
    print(f"\n✓ Compression: {compression_ratio:.2f}x ({reduction_pct:.1f}% smaller)")
    print(f"✓ Quality degradation: {quality['mse_degradation_percent']:+.2f}%")

    if compression_ratio > 3.5:
        print("✓ EXCELLENT: Near-theoretical 4x compression achieved!")
    elif compression_ratio > 2.5:
        print("✓ GOOD: Solid compression, close to target")
    else:
        print("⚠ MODERATE: Less compression than expected")

    if abs(quality['mse_degradation_percent']) < 5:
        print("✓ ACCEPTABLE: Quality degradation <5%")
    elif abs(quality['mse_degradation_percent']) < 10:
        print("⚠ MARGINAL: Quality degradation 5-10%")
    else:
        print("✗ POOR: Quality degradation >10%")

    # Save results
    output_dir = Path(__file__).parent / "compression_experiments"
    output_dir.mkdir(exist_ok=True)

    results_path = output_dir / f"tinyvae_manual_int8_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    # Lessons learned
    print("\n" + "="*70)
    print("LESSONS LEARNED")
    print("="*70)

    print("\n1. Quantization Mechanics:")
    print("   - INT8 uses 1 byte vs FP32's 4 bytes per parameter")
    print("   - Need scale + zero_point metadata for dequantization")
    print("   - Quantization error = rounding error from discretization")

    print("\n2. Where Compression Comes From:")
    print(f"   - {quant_params:,} params quantized ({quant_params/total_params*100:.1f}% of total)")
    print("   - Linear layers are the biggest win (fc_mu, fc_logvar, decoder_input)")
    print("   - Conv layers also benefit but have fewer parameters")

    print("\n3. Quality vs Size Tradeoff:")
    if abs(quality['mse_degradation_percent']) < 1:
        print("   - Minimal quality loss (<1%) - quantization is nearly free!")
    else:
        print(f"   - {abs(quality['mse_degradation_percent']):.1f}% quality degradation is the price of {compression_ratio:.1f}x compression")

    print("\n4. Next Experiments:")
    print("   - Try INT4 quantization (8x compression, more quality loss)")
    print("   - Combine with pruning (remove least important weights)")
    print("   - Quantization-aware training (train with quantization in loop)")
    print("   - Test on VisionPuzzleVAE too")

    print("\n" + "="*70)
    print("Experiment Complete! 🚀")
    print("="*70 + "\n")

    return results


if __name__ == "__main__":
    main()
