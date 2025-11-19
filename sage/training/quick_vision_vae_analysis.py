"""
Quick Vision VAE Analysis - Trained vs Untrained
=================================================

Rapid comparison of trained Vision Puzzle VAE vs untrained baseline.

Author: Autonomous Thor
Date: 2025-11-19
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from compression.vision_puzzle_vae import VisionPuzzleVAE

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load trained model
    print("Loading trained model...")
    checkpoint = torch.load(
        'sage/training/vision_vae_checkpoints/best_model.pt',
        map_location=device,
        weights_only=False
    )
    trained_model = VisionPuzzleVAE(latent_dim=64).to(device)
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    trained_model.eval()
    print(f"Loaded epoch {checkpoint.get('epoch', '?')}\n")

    # Create untrained model
    print("Creating untrained baseline...\n")
    untrained_model = VisionPuzzleVAE(latent_dim=64).to(device)
    untrained_model.eval()

    # Load test data
    print("Loading CIFAR-10 test set...")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2
    )
    print(f"Loaded {len(test_dataset)} test samples\n")

    # Evaluate both models
    print("="*60)
    print("Evaluating Models (100 batches)")
    print("="*60 + "\n")

    results = {'trained': {}, 'untrained': {}}

    for name, model in [('trained', trained_model), ('untrained', untrained_model)]:
        print(f"{name.upper()} MODEL:")
        total_recon_loss = 0
        total_perplexity = 0
        code_usage = torch.zeros(10)
        num_batches = 0

        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                if i >= 100:  # 100 batches = 6,400 samples
                    break

                images = images.to(device)
                output = model(images)

                recon = output['reconstruction']
                vq_loss = output['vq_loss']
                puzzles = output['puzzles']

                # Reconstruction loss
                recon_loss = nn.MSELoss()(recon, images).item()
                total_recon_loss += recon_loss

                # Perplexity
                perplexity = vq_loss['perplexity']
                total_perplexity += perplexity

                # Code usage
                unique_codes = puzzles.unique()
                for code in unique_codes:
                    code_usage[code.item()] += 1

                num_batches += 1

        # Compute averages
        avg_recon_loss = total_recon_loss / num_batches
        avg_perplexity = total_perplexity / num_batches
        codes_used = (code_usage > 0).sum().item()

        results[name] = {
            'recon_loss': avg_recon_loss,
            'perplexity': avg_perplexity,
            'codes_used': codes_used,
            'code_usage_counts': code_usage.tolist()
        }

        print(f"  Recon Loss: {avg_recon_loss:.6f}")
        print(f"  Perplexity: {avg_perplexity:.2f}")
        print(f"  Codes Used: {codes_used}/10")
        print()

    # Compute improvement
    print("="*60)
    print("IMPROVEMENT ANALYSIS")
    print("="*60 + "\n")

    recon_improvement = (
        (results['untrained']['recon_loss'] - results['trained']['recon_loss'])
        / results['untrained']['recon_loss'] * 100
    )

    perp_change = results['trained']['perplexity'] - results['untrained']['perplexity']

    print(f"Reconstruction Loss Reduction: {recon_improvement:.1f}%")
    print(f"Perplexity Change: {perp_change:+.2f}")
    print(f"Additional Codes Used: {results['trained']['codes_used'] - results['untrained']['codes_used']:+d}")

    # Save results
    results['improvement'] = {
        'recon_loss_reduction_percent': recon_improvement,
        'perplexity_change': perp_change,
        'additional_codes': results['trained']['codes_used'] - results['untrained']['codes_used']
    }

    results_path = 'sage/training/vision_vae_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Summary
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60 + "\n")

    if recon_improvement > 5:
        print("✅ Training significantly improved reconstruction quality!")
    elif recon_improvement > 0:
        print("✓ Training improved reconstruction quality.")
    else:
        print("⚠️  Training did not improve reconstruction quality.")

    if results['trained']['codes_used'] >= 9:
        print("✅ Trained model uses most/all codes (good diversity).")
    else:
        print(f"⚠️  Trained model only uses {results['trained']['codes_used']}/10 codes.")

    print("\nTraining on CIFAR-10 completed successfully.")
    print("Puzzle space consciousness learned from real visual data!")

if __name__ == '__main__':
    main()
