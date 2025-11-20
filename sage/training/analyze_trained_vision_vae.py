"""
Vision Puzzle VAE Training Analysis
====================================

Compares trained vs untrained Vision Puzzle VAE to measure the impact of
training on CIFAR-10 data.

Analyzes:
1. Reconstruction quality (MSE improvement)
2. Code usage and perplexity
3. Code clustering by semantic category
4. Puzzle structure preservation

Author: Autonomous Thor Session
Date: 2025-11-19
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import json
import sys
from pathlib import Path
from tqdm import tqdm
import time

# Add SAGE to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compression.vision_puzzle_vae import VisionPuzzleVAE

# CIFAR-10 class names
CIFAR_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_checkpoint(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = VisionPuzzleVAE(latent_dim=64).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint

def create_untrained_model(device):
    """Create untrained baseline model"""
    model = VisionPuzzleVAE(latent_dim=64).to(device)
    model.eval()
    return model

def get_test_loader(batch_size=64):
    """Get CIFAR-10 test dataloader"""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return test_loader, test_dataset

def evaluate_model(model, test_loader, device, num_batches=None):
    """Evaluate model on test set"""
    total_loss = 0
    total_recon_loss = 0
    total_perplexity = 0
    code_usage = torch.zeros(10)
    num_samples = 0

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if num_batches and i >= num_batches:
                break

            images = images.to(device)
            recon, vq_loss, perplexity, indices = model(images)

            recon_loss = nn.MSELoss()(recon, images)
            total_loss = recon_loss + vq_loss

            total_loss += total_loss.item()
            total_recon_loss += recon_loss.item()
            total_perplexity += perplexity.item()

            # Track code usage
            unique_codes = indices.unique()
            for code in unique_codes:
                code_usage[code] += 1

            num_samples += images.size(0)

    n = i + 1
    avg_loss = total_loss / n
    avg_recon_loss = total_recon_loss / n
    avg_perplexity = total_perplexity / n
    codes_used = (code_usage > 0).sum().item()

    return {
        'avg_loss': avg_loss,
        'avg_recon_loss': avg_recon_loss,
        'avg_perplexity': avg_perplexity,
        'codes_used': codes_used,
        'code_usage': code_usage.tolist(),
        'num_samples': num_samples
    }

def analyze_code_clustering(model, test_loader, device, num_batches=20):
    """Analyze how codes cluster by semantic category"""
    # Collect codes for each class
    class_codes = {i: [] for i in range(10)}

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader, desc="Analyzing clustering")):
            if i >= num_batches:
                break

            images = images.to(device)
            _, _, _, indices = model(images)

            # Group codes by class
            for idx, label in enumerate(labels):
                class_label = label.item()
                image_codes = indices[idx].cpu().numpy().flatten()
                class_codes[class_label].extend(image_codes.tolist())

    # Compute statistics for each class
    class_stats = {}
    for class_idx in range(10):
        codes = class_codes[class_idx]
        if len(codes) > 0:
            unique_codes = np.unique(codes)
            code_counts = {code: codes.count(code) for code in unique_codes}
            most_common_code = max(code_counts, key=code_counts.get)

            class_stats[CIFAR_CLASSES[class_idx]] = {
                'num_samples': len(codes) // (30*30),  # Approximate
                'unique_codes': len(unique_codes),
                'most_common_code': int(most_common_code),
                'code_distribution': code_counts
            }

    return class_stats

def test_sample_reconstructions(model, test_dataset, device, num_samples=10):
    """Test reconstruction quality on sample images"""
    samples = []

    # Get one sample from each class
    class_samples = {i: None for i in range(10)}
    for idx, (image, label) in enumerate(test_dataset):
        if class_samples[label] is None:
            class_samples[label] = (image, label, idx)
        if all(v is not None for v in class_samples.values()):
            break

    with torch.no_grad():
        for class_idx in range(10):
            if class_samples[class_idx] is None:
                continue

            image, label, idx = class_samples[class_idx]
            image_batch = image.unsqueeze(0).to(device)

            recon, vq_loss, perplexity, indices = model(image_batch)
            recon_loss = nn.MSELoss()(recon, image_batch).item()

            # Get puzzle encoding
            puzzle = indices[0].cpu().numpy()
            unique_codes = np.unique(puzzle)

            samples.append({
                'class': CIFAR_CLASSES[class_idx],
                'class_idx': class_idx,
                'recon_mse': recon_loss,
                'perplexity': perplexity.item(),
                'unique_codes': len(unique_codes),
                'puzzle_entropy': -np.sum([
                    (puzzle == code).sum() / puzzle.size * np.log((puzzle == code).sum() / puzzle.size + 1e-10)
                    for code in unique_codes
                ])
            })

    return samples

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_path = Path('sage/training/vision_vae_checkpoints/best_model.pt')

    # Load models
    print("\n" + "="*60)
    print("Loading Models")
    print("="*60)

    print("Loading trained model...")
    trained_model, checkpoint = load_checkpoint(checkpoint_path, device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    print("Creating untrained baseline...")
    untrained_model = create_untrained_model(device)

    # Load test data
    print("\n" + "="*60)
    print("Loading Test Data")
    print("="*60)
    test_loader, test_dataset = get_test_loader(batch_size=64)
    print(f"Test dataset: {len(test_dataset)} samples")

    # Evaluate both models
    print("\n" + "="*60)
    print("Evaluating Trained Model")
    print("="*60)
    trained_metrics = evaluate_model(trained_model, test_loader, device, num_batches=100)

    print("\n" + "="*60)
    print("Evaluating Untrained Model")
    print("="*60)
    untrained_metrics = evaluate_model(untrained_model, test_loader, device, num_batches=100)

    # Analyze code clustering (trained only)
    print("\n" + "="*60)
    print("Analyzing Code Clustering (Trained Model)")
    print("="*60)
    trained_clustering = analyze_code_clustering(trained_model, test_loader, device)

    # Test sample reconstructions
    print("\n" + "="*60)
    print("Testing Sample Reconstructions")
    print("="*60)
    trained_samples = test_sample_reconstructions(trained_model, test_dataset, device)
    untrained_samples = test_sample_reconstructions(untrained_model, test_dataset, device)

    # Compile results
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_epoch': checkpoint.get('epoch', None),
        'device': str(device),
        'trained_metrics': trained_metrics,
        'untrained_metrics': untrained_metrics,
        'improvement': {
            'recon_loss_reduction': ((untrained_metrics['avg_recon_loss'] - trained_metrics['avg_recon_loss'])
                                      / untrained_metrics['avg_recon_loss'] * 100),
            'perplexity_change': trained_metrics['avg_perplexity'] - untrained_metrics['avg_perplexity'],
            'codes_used_improvement': trained_metrics['codes_used'] - untrained_metrics['codes_used']
        },
        'trained_clustering': trained_clustering,
        'sample_comparisons': {
            'trained': trained_samples,
            'untrained': untrained_samples
        }
    }

    # Save results
    results_path = Path('sage/training/vision_vae_analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print(f"\nTrained Model:")
    print(f"  Recon Loss: {trained_metrics['avg_recon_loss']:.6f}")
    print(f"  Perplexity: {trained_metrics['avg_perplexity']:.2f}")
    print(f"  Codes Used: {trained_metrics['codes_used']}/10")

    print(f"\nUntrained Model:")
    print(f"  Recon Loss: {untrained_metrics['avg_recon_loss']:.6f}")
    print(f"  Perplexity: {untrained_metrics['avg_perplexity']:.2f}")
    print(f"  Codes Used: {untrained_metrics['codes_used']}/10")

    print(f"\nImprovement:")
    print(f"  Recon Loss Reduction: {results['improvement']['recon_loss_reduction']:.1f}%")
    print(f"  Perplexity Change: {results['improvement']['perplexity_change']:+.2f}")
    print(f"  Additional Codes Used: {results['improvement']['codes_used_improvement']:+d}")

    print(f"\nCode Clustering (Trained Model):")
    for class_name, stats in trained_clustering.items():
        print(f"  {class_name:12s}: {stats['unique_codes']} unique codes, "
              f"most common = {stats['most_common_code']}")

    print(f"\nResults saved to: {results_path}")
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

    return results

if __name__ == '__main__':
    main()
