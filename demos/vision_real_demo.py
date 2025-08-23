#!/usr/bin/env python3
"""
Real Vision IRP Demo on Jetson
Uses actual images and measures real-world performance
"""

import torch
import torchvision
import torchvision.transforms as transforms
import time
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sage.irp.plugins.vision_impl import create_vision_irp
import matplotlib.pyplot as plt
import numpy as np


def load_test_images(batch_size: int = 4):
    """Load test images from CIFAR-10 or generate synthetic"""
    try:
        # Try to load CIFAR-10
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False,
            download=True, 
            transform=transform
        )
        
        testloader = torch.utils.data.DataLoader(
            testset, 
            batch_size=batch_size,
            shuffle=False
        )
        
        # Get first batch
        images, labels = next(iter(testloader))
        return images
        
    except Exception as e:
        print(f"Could not load CIFAR-10: {e}")
        print("Using synthetic images instead")
        
        # Generate synthetic images with patterns
        images = []
        for i in range(batch_size):
            # Create different patterns
            img = torch.zeros(3, 224, 224)
            
            if i % 4 == 0:
                # Gradient
                for y in range(224):
                    img[:, y, :] = y / 224
            elif i % 4 == 1:
                # Checkerboard
                for y in range(0, 224, 32):
                    for x in range(0, 224, 32):
                        if (x//32 + y//32) % 2 == 0:
                            img[:, y:y+32, x:x+32] = 1.0
            elif i % 4 == 2:
                # Circles
                center = 112
                for y in range(224):
                    for x in range(224):
                        dist = ((x-center)**2 + (y-center)**2)**0.5
                        img[:, y, x] = 1.0 if dist < 50 else 0.0
            else:
                # Noise
                img = torch.rand(3, 224, 224)
            
            images.append(img)
        
        return torch.stack(images)


def visualize_results(original, refined_early, refined_full, metrics):
    """Create visualization of results"""
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    # Show first 4 images
    for i in range(min(4, original.shape[0])):
        # Original
        axes[0, i].imshow(original[i].cpu().permute(1, 2, 0))
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Early stop refinement
        axes[1, i].imshow(refined_early[i].cpu().permute(1, 2, 0))
        axes[1, i].set_title(f'Early Stop ({metrics["early"]["iterations"]} iter)')
        axes[1, i].axis('off')
        
        # Full refinement
        axes[2, i].imshow(refined_full[i].cpu().permute(1, 2, 0))
        axes[2, i].set_title(f'Full ({metrics["full"]["iterations"]} iter)')
        axes[2, i].axis('off')
    
    plt.suptitle(f'Vision IRP: {metrics["speedup"]:.1f}x Speedup, {metrics["quality_preserved"]:.1%} Quality')
    plt.tight_layout()
    plt.savefig('vision_irp_results.png', dpi=150)
    print("✓ Results saved to vision_irp_results.png")


def main():
    print("=" * 60)
    print("Vision IRP Real-World Demo - Jetson Orin Nano")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load test images
    print("\n1. Loading test images...")
    images = load_test_images(batch_size=4).to(device)
    print(f"  Loaded {images.shape[0]} images of shape {images.shape[1:]}")
    
    # Create IRP instances
    print("\n2. Creating Vision IRP instances...")
    irp_early = create_vision_irp(device)
    irp_full = create_vision_irp(device)
    
    # Warm up
    print("\n3. Warming up GPU...")
    for _ in range(5):
        with torch.no_grad():
            _ = irp_early.vae.encode(images[:1])
    torch.cuda.synchronize()
    
    # Run with early stopping
    print("\n4. Running refinement WITH early stopping...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    refined_early, telemetry_early = irp_early.refine(images, early_stop=True)
    
    torch.cuda.synchronize()
    early_time = time.time() - start_time
    
    print(f"  Iterations: {telemetry_early['iterations']}")
    print(f"  Time: {early_time*1000:.2f}ms")
    print(f"  Energy: {telemetry_early['final_energy']:.4f}")
    print(f"  Trust: {telemetry_early['trust']:.3f}")
    
    # Run without early stopping
    print("\n5. Running refinement WITHOUT early stopping...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    refined_full, telemetry_full = irp_full.refine(images, early_stop=False)
    
    torch.cuda.synchronize()
    full_time = time.time() - start_time
    
    print(f"  Iterations: {telemetry_full['iterations']}")
    print(f"  Time: {full_time*1000:.2f}ms")
    print(f"  Energy: {telemetry_full['final_energy']:.4f}")
    
    # Calculate metrics
    print("\n6. Computing performance metrics...")
    
    # Quality comparison (MSE between early and full)
    quality_diff = torch.nn.functional.mse_loss(refined_early, refined_full)
    quality_preserved = 1 - quality_diff.item()
    
    # Performance metrics
    speedup = telemetry_full['iterations'] / telemetry_early['iterations']
    time_speedup = full_time / early_time
    compute_saved = telemetry_early['compute_saved']
    
    print(f"\n" + "=" * 40)
    print("RESULTS SUMMARY")
    print("=" * 40)
    print(f"Iteration Speedup: {speedup:.1f}x")
    print(f"Time Speedup: {time_speedup:.1f}x")
    print(f"Compute Saved: {compute_saved*100:.1f}%")
    print(f"Quality Preserved: {quality_preserved*100:.1f}%")
    print(f"Energy Convergence: {abs(telemetry_early['final_energy'] - telemetry_full['final_energy']):.4f}")
    
    # Save metrics
    metrics = {
        "early": telemetry_early,
        "full": telemetry_full,
        "speedup": speedup,
        "time_speedup": time_speedup,
        "quality_preserved": quality_preserved,
        "early_time_ms": early_time * 1000,
        "full_time_ms": full_time * 1000,
        "batch_size": images.shape[0],
        "device": str(device)
    }
    
    with open('vision_irp_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to vision_irp_metrics.json")
    
    # Visualize if possible
    try:
        visualize_results(images, refined_early, refined_full, metrics)
    except Exception as e:
        print(f"Could not visualize: {e}")
    
    # Final verdict
    print("\n" + "=" * 40)
    if speedup >= 2.0 and quality_preserved >= 0.95:
        print("✅ SUCCESS: Achieved target performance!")
        print(f"   {speedup:.1f}x speedup with {quality_preserved:.1%} quality")
    else:
        print("⚠️  PARTIAL SUCCESS:")
        if speedup < 2.0:
            print(f"   Speedup {speedup:.1f}x (target: 2.0x)")
        if quality_preserved < 0.95:
            print(f"   Quality {quality_preserved:.1%} (target: 95%)")
    print("=" * 40)


if __name__ == "__main__":
    main()