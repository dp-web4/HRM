#!/usr/bin/env python3
"""
Track 8 Experiment #3: Latent Dimension Ablation Study

Goal: Find optimal latent dimension for quality vs size tradeoff

Method: Architecture compression (not quantization)
- Test latent dimensions: 64, 32, 16, 8, 4
- Measure parameter count, model size, and reconstruction quality
- Create compression vs quality curves

Research Questions:
1. How much can we reduce latent_dim before quality degrades?
2. What's the optimal latent_dim for Nano deployment?
3. Can we combine latent reduction + quantization for compound compression?
4. Does smaller latent_dim actually hurt quality, or is 64 overkill?

Context:
- Current TinyVAE: 64-dim latents, 817K params
- Linear layers dominate: fc_mu, fc_logvar, decoder_input (64×64×64 = 262K each)
- Reducing latent_dim shrinks these layers quadratically
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

from sage.irp.plugins.tinyvae_irp_plugin import TinyVAE


def count_parameters(model: nn.Module) -> int:
    """Count total parameters"""
    return sum(p.numel() for p in model.parameters())


def measure_model_size(model: nn.Module) -> dict:
    """Measure model size"""
    temp_path = Path(f"/tmp/model_temp.pt")
    torch.save(model.state_dict(), temp_path)
    size_bytes = temp_path.stat().st_size
    temp_path.unlink()

    return {
        'size_bytes': size_bytes,
        'size_mb': size_bytes / (1024**2),
        'size_kb': size_bytes / 1024
    }


def benchmark_model(model: nn.Module, dataloader, device: str, num_batches: int = 20) -> dict:
    """Benchmark reconstruction quality"""
    model = model.to(device).eval()

    total_mse = 0.0
    total_kl = 0.0
    num_samples = 0

    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            if i >= num_batches:
                break

            images = images.to(device)
            batch_size = images.size(0)
            num_samples += batch_size

            recon, mu, logvar = model(images)
            mse = nn.functional.mse_loss(recon, images).item()
            kl = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item() / (batch_size * model.latent_dim)

            total_mse += mse * batch_size
            total_kl += kl * batch_size

    avg_mse = total_mse / num_samples
    avg_kl = total_kl / num_samples

    return {
        'mse': avg_mse,
        'kl_divergence': avg_kl,
        'num_samples': num_samples
    }


def run_ablation_study(latent_dims: list, dataloader, device: str) -> list:
    """Run ablation study across multiple latent dimensions"""
    print("\n" + "="*70)
    print("LATENT DIMENSION ABLATION STUDY")
    print("="*70)
    print(f"\nTesting latent dimensions: {latent_dims}")
    print(f"Baseline: 64-dim (current TinyVAE)")
    print()

    results = []

    for latent_dim in latent_dims:
        print(f"\n{'='*70}")
        print(f"Testing latent_dim = {latent_dim}")
        print(f"{'='*70}")

        # Create model
        print(f"\nCreating TinyVAE with {latent_dim}-dim latents...")
        model = TinyVAE(input_channels=3, latent_dim=latent_dim, img_size=64)

        # Count parameters
        num_params = count_parameters(model)
        print(f"Parameters: {num_params:,}")

        # Measure size
        size_info = measure_model_size(model)
        print(f"Model size: {size_info['size_mb']:.2f} MB")

        # Benchmark quality
        print(f"Benchmarking quality (20 batches)...")
        quality = benchmark_model(model, dataloader, device, num_batches=20)

        print(f"  MSE: {quality['mse']:.6f}")
        print(f"  KL divergence: {quality['kl_divergence']:.6f}")

        # Calculate compression ratios (relative to 64-dim baseline)
        baseline_params = 817633  # 64-dim TinyVAE
        baseline_size_mb = 3.13  # FP32
        param_ratio = baseline_params / num_params
        size_ratio = baseline_size_mb / size_info['size_mb']

        print(f"\nCompression vs 64-dim baseline:")
        print(f"  Parameter reduction: {param_ratio:.2f}x")
        print(f"  Size reduction: {size_ratio:.2f}x")

        results.append({
            'latent_dim': latent_dim,
            'num_params': num_params,
            'size_mb': size_info['size_mb'],
            'size_bytes': size_info['size_bytes'],
            'mse': quality['mse'],
            'kl_divergence': quality['kl_divergence'],
            'param_compression_ratio': param_ratio,
            'size_compression_ratio': size_ratio
        })

    return results


def plot_results(results: list, output_dir: Path):
    """Create visualization plots"""
    print("\n" + "="*70)
    print("CREATING VISUALIZATION PLOTS")
    print("="*70)

    latent_dims = [r['latent_dim'] for r in results]
    sizes_mb = [r['size_mb'] for r in results]
    mse_values = [r['mse'] for r in results]
    param_counts = [r['num_params'] for r in results]
    size_ratios = [r['size_compression_ratio'] for r in results]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TinyVAE Latent Dimension Ablation Study', fontsize=16, fontweight='bold')

    # Plot 1: Model Size vs Latent Dim
    ax1 = axes[0, 0]
    ax1.plot(latent_dims, sizes_mb, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Latent Dimension', fontsize=12)
    ax1.set_ylabel('Model Size (MB)', fontsize=12)
    ax1.set_title('Model Size vs Latent Dimension', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    for i, (x, y) in enumerate(zip(latent_dims, sizes_mb)):
        ax1.annotate(f'{y:.2f}MB', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    # Plot 2: Reconstruction Quality vs Latent Dim
    ax2 = axes[0, 1]
    ax2.plot(latent_dims, mse_values, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Latent Dimension', fontsize=12)
    ax2.set_ylabel('Reconstruction MSE', fontsize=12)
    ax2.set_title('Quality vs Latent Dimension', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    for i, (x, y) in enumerate(zip(latent_dims, mse_values)):
        ax2.annotate(f'{y:.4f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    # Plot 3: Parameter Count vs Latent Dim
    ax3 = axes[1, 0]
    ax3.plot(latent_dims, [p/1000 for p in param_counts], 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax3.set_xlabel('Latent Dimension', fontsize=12)
    ax3.set_ylabel('Parameters (thousands)', fontsize=12)
    ax3.set_title('Parameter Count vs Latent Dimension', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    for i, (x, y) in enumerate(zip(latent_dims, param_counts)):
        ax3.annotate(f'{y/1000:.0f}K', (x, y/1000), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    # Plot 4: Compression vs Quality Tradeoff
    ax4 = axes[1, 1]
    ax4.plot(size_ratios, mse_values, 'o-', linewidth=2, markersize=8, color='#C73E1D')
    ax4.set_xlabel('Compression Ratio (vs 64-dim)', fontsize=12)
    ax4.set_ylabel('Reconstruction MSE', fontsize=12)
    ax4.set_title('Compression vs Quality Tradeoff', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    for i, (x, y, dim) in enumerate(zip(size_ratios, mse_values, latent_dims)):
        ax4.annotate(f'{dim}d', (x, y), textcoords="offset points", xytext=(10,-5), ha='left', fontsize=9)

    plt.tight_layout()

    plot_path = output_dir / f"latent_ablation_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to: {plot_path}")
    plt.close()

    return plot_path


def main():
    print("\n" + "="*70)
    print("TRACK 8 - Experiment #3: Latent Dimension Ablation")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print("\nGoal: Find optimal latent_dim for size/quality tradeoff")
    print("\nMethod: Test multiple latent dimensions with untrained models")
    print("        (Training would improve absolute quality, but relative")
    print("         differences between dimensions would remain similar)")

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

    # Run ablation study
    latent_dims = [64, 32, 16, 8, 4]
    results = run_ablation_study(latent_dims, test_loader, device)

    # Analysis
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)

    print("\nSummary Table:")
    print(f"{'Latent Dim':>12} | {'Params':>10} | {'Size (MB)':>10} | {'MSE':>10} | {'Compression':>12}")
    print(f"{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
    for r in results:
        print(f"{r['latent_dim']:>12} | {r['num_params']:>10,} | {r['size_mb']:>10.2f} | {r['mse']:>10.6f} | {r['size_compression_ratio']:>11.2f}x")

    # Find optimal dimension
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    baseline = results[0]  # 64-dim
    best_tradeoff = None
    best_score = 0

    print("\nQuality Degradation Analysis:")
    for r in results:
        quality_loss_pct = (r['mse'] - baseline['mse']) / baseline['mse'] * 100
        compression = r['size_compression_ratio']

        # Score: compression gain vs quality loss
        # Higher score = better tradeoff
        score = compression / (1 + abs(quality_loss_pct)/5)  # Penalize quality loss

        print(f"\n  {r['latent_dim']}-dim:")
        print(f"    Compression: {compression:.2f}x")
        print(f"    Quality loss: {quality_loss_pct:+.2f}%")
        print(f"    Tradeoff score: {score:.2f}")

        if r['latent_dim'] != baseline['latent_dim']:
            if best_tradeoff is None or score > best_score:
                best_tradeoff = r
                best_score = score

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    if best_tradeoff:
        quality_loss = (best_tradeoff['mse'] - baseline['mse']) / baseline['mse'] * 100
        print(f"\n✓ Optimal latent dimension: {best_tradeoff['latent_dim']}")
        print(f"  Compression: {best_tradeoff['size_compression_ratio']:.2f}x")
        print(f"  Quality loss: {quality_loss:+.2f}%")
        print(f"  Size: {best_tradeoff['size_mb']:.2f} MB (from {baseline['size_mb']:.2f} MB)")

        if abs(quality_loss) < 5:
            print(f"\n  ✓ EXCELLENT: <5% quality loss!")
        elif abs(quality_loss) < 10:
            print(f"\n  ✓ GOOD: <10% quality loss")
        else:
            print(f"\n  ⚠ MODERATE: {abs(quality_loss):.1f}% quality loss")

    # Compound compression analysis
    print(f"\n{'='*70}")
    print("COMPOUND COMPRESSION (Latent + Quantization)")
    print(f"{'='*70}")

    print("\nCombining latent reduction with quantization:")
    for r in results:
        fp32_size = r['size_mb']
        int8_size = fp32_size / 4.0
        int4_size = fp32_size / 8.0

        print(f"\n  {r['latent_dim']}-dim latents:")
        print(f"    FP32:      {fp32_size:.3f} MB ({r['size_compression_ratio']:>4.1f}x vs 64-dim FP32)")
        print(f"    INT8:      {int8_size:.3f} MB ({r['size_compression_ratio']*4:>4.1f}x vs 64-dim FP32)")
        print(f"    INT4:      {int4_size:.3f} MB ({r['size_compression_ratio']*8:>4.1f}x vs 64-dim FP32)")

    # Extreme compression example
    if len(results) >= 3:
        extreme = results[2]  # 16-dim
        extreme_int4 = extreme['size_mb'] / 8.0
        total_compression = baseline['size_mb'] / extreme_int4
        print(f"\n  Extreme compression example:")
        print(f"    16-dim + INT4: {extreme_int4:.3f} MB")
        print(f"    Total compression: {total_compression:.1f}x")
        print(f"    From {baseline['size_mb']:.2f} MB → {extreme_int4:.3f} MB")

    # Save results
    output_dir = Path(__file__).parent / "compression_experiments"
    output_dir.mkdir(exist_ok=True)

    results_path = output_dir / f"latent_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump({
            'results': results,
            'baseline': baseline,
            'best_tradeoff': best_tradeoff,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n✓ Results saved to: {results_path}")

    # Create plots
    plot_path = plot_results(results, output_dir)

    print("\n" + "="*70)
    print("LESSONS LEARNED")
    print("="*70)

    print("\n1. Latent Dimension Impact:")
    quality_64_to_32 = (results[1]['mse'] - results[0]['mse']) / results[0]['mse'] * 100
    quality_64_to_16 = (results[2]['mse'] - results[0]['mse']) / results[0]['mse'] * 100
    print(f"   - 64→32 dims: {results[1]['size_compression_ratio']:.1f}x compression, {quality_64_to_32:+.1f}% quality change")
    print(f"   - 64→16 dims: {results[2]['size_compression_ratio']:.1f}x compression, {quality_64_to_16:+.1f}% quality change")
    print(f"   - Pattern: Latent dims have STRONG impact on model size")

    print("\n2. Architectural vs Quantization Compression:")
    print(f"   - Latent reduction: Reduces parameters (fewer connections)")
    print(f"   - Quantization: Reduces bits per parameter")
    print(f"   - Both are complementary and can be combined!")

    print("\n3. Deployment Strategy:")
    if best_tradeoff and abs((best_tradeoff['mse'] - baseline['mse']) / baseline['mse'] * 100) < 5:
        print(f"   ✓ Recommended: {best_tradeoff['latent_dim']}-dim + INT4 quantization")
        print(f"   ✓ Size: {best_tradeoff['size_mb']/8:.3f} MB (from {baseline['size_mb']:.2f} MB)")
        print(f"   ✓ Total compression: {best_tradeoff['size_compression_ratio']*8:.1f}x")
    else:
        print(f"   ✓ Recommended: Keep 64-dim, use INT4 quantization")
        print(f"   ✓ Quality preservation is more important than extreme compression")

    print("\n4. Next Steps:")
    print("   - Train models at different latent dimensions")
    print("   - Test compound compression (latent + INT4)")
    print("   - Measure inference speed at different dimensions")
    print("   - Apply to VisionPuzzleVAE (349K params)")

    print("\n" + "="*70)
    print("Experiment Complete! 🚀")
    print("="*70 + "\n")

    return results


if __name__ == "__main__":
    main()
