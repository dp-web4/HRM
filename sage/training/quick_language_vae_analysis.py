#!/usr/bin/env python3
"""
Quick Language VAE Analysis

Compare trained vs untrained Language Puzzle VAE performance.
Similar to vision/audio VAE analysis scripts.
"""

import torch
import torch.nn.functional as F
import json
import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.compression.language_puzzle_vae_small import LanguagePuzzleVAESmall, text_to_indices, indices_to_text
from sage.training.train_language_puzzle_vae import TextDataset


def calculate_accuracy(logits, targets):
    """Calculate character-level accuracy"""
    predictions = logits.argmax(dim=-1)
    # Ignore padding (index 0)
    mask = targets != 0
    correct = (predictions == targets) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()


def analyze_language_vae(num_samples=100):
    """
    Analyze trained vs untrained Language Puzzle VAE

    Returns:
        results: Dictionary with comparison metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load trained model
    print("Loading trained model...")
    trained_model = LanguagePuzzleVAESmall(
        vocab_size=128,
        char_embed_dim=32,
        latent_dim=64,
        num_codes=10,
        max_length=128
    ).to(device)

    checkpoint_dir = Path("./sage/training/language_vae_checkpoints")
    best_model_path = checkpoint_dir / "best_model.pt"

    if not best_model_path.exists():
        print(f"Error: Best model not found at {best_model_path}")
        print("Training may not have completed yet.")
        return None

    trained_model.load_state_dict(torch.load(best_model_path, map_location=device))
    trained_model.eval()

    # Get epoch number from checkpoint
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        epoch_num = int(latest_checkpoint.stem.split('_')[-1])
        print(f"Loaded epoch {epoch_num}\n")

    # Create untrained model
    print("Creating untrained baseline...\n")
    untrained_model = LanguagePuzzleVAESmall(
        vocab_size=128,
        char_embed_dim=32,
        latent_dim=64,
        num_codes=10,
        max_length=128
    ).to(device)
    untrained_model.eval()

    # Load validation dataset
    print("Loading WikiText-2 validation dataset...")
    val_file = "./data/wikitext/wiki.valid.txt"

    if not Path(val_file).exists():
        print(f"Error: Validation file not found at {val_file}")
        return None

    val_dataset = TextDataset(val_file, seq_length=128)

    # Sample subset for analysis
    num_samples = min(num_samples, len(val_dataset))
    indices = torch.randperm(len(val_dataset))[:num_samples]
    samples = [val_dataset[i] for i in indices]

    val_loader = DataLoader(
        [val_dataset[i] for i in indices],
        batch_size=64,
        shuffle=False
    )

    print(f"Loaded {num_samples} validation samples\n")

    print("=" * 60)
    print("Evaluating Models ({} batches)".format(len(val_loader)))
    print("=" * 60)

    # Evaluate both models
    results = {'trained': {}, 'untrained': {}}

    for model_name, model in [('trained', trained_model), ('untrained', untrained_model)]:
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        total_accuracy = 0
        all_codes = []

        with torch.no_grad():
            for batch in val_loader:
                text_indices = batch.to(device)

                # Forward pass
                output = model(text_indices)
                char_logits = output['char_logits']
                vq_loss = output['vq_loss']

                # Reconstruction loss
                recon_loss = F.cross_entropy(
                    char_logits.reshape(-1, 128),
                    text_indices.reshape(-1),
                    ignore_index=0
                )

                # Total loss
                loss = recon_loss + vq_loss

                # Accuracy
                accuracy = calculate_accuracy(char_logits, text_indices)

                # Collect codes
                puzzles = output['puzzles']
                all_codes.extend(puzzles.flatten().cpu().tolist())

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_vq_loss += vq_loss.item()
                total_accuracy += accuracy

        # Average metrics
        num_batches = len(val_loader)
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_vq_loss = total_vq_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        # Code statistics
        all_codes_tensor = torch.tensor(all_codes)
        unique_codes = len(torch.unique(all_codes_tensor))

        # Perplexity (code distribution quality)
        code_counts = torch.bincount(all_codes_tensor, minlength=10).float()
        code_probs = code_counts / code_counts.sum()
        perplexity = 2 ** (-torch.sum(code_probs * torch.log2(code_probs + 1e-10)))

        # Code usage distribution
        code_usage = (code_counts / code_counts.sum() * 100).tolist()

        results[model_name] = {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'vq_loss': avg_vq_loss,
            'accuracy': avg_accuracy,
            'perplexity': perplexity.item(),
            'codes_used': unique_codes,
            'code_usage_percent': code_usage
        }

        print(f"\n{model_name.upper()} MODEL:")
        print(f"  Recon Loss: {avg_recon_loss:.6f}")
        print(f"  VQ Loss: {avg_vq_loss:.6f}")
        print(f"  Accuracy: {avg_accuracy:.2%}")
        print(f"  Perplexity: {perplexity.item():.2f}")
        print(f"  Codes Used: {unique_codes}/10")

    # Calculate improvement
    print("\n" + "=" * 60)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 60)

    recon_improvement = (results['untrained']['recon_loss'] - results['trained']['recon_loss']) / results['untrained']['recon_loss'] * 100
    accuracy_improvement = results['trained']['accuracy'] - results['untrained']['accuracy']
    perplexity_change = results['trained']['perplexity'] - results['untrained']['perplexity']
    additional_codes = results['trained']['codes_used'] - results['untrained']['codes_used']

    improvement = {
        'recon_loss_reduction_percent': recon_improvement,
        'accuracy_improvement': accuracy_improvement,
        'perplexity_change': perplexity_change,
        'additional_codes': additional_codes
    }

    print(f"\nReconstruction Loss Reduction: {recon_improvement:.1f}%")
    print(f"Accuracy Improvement: {accuracy_improvement:.2%}")
    print(f"Perplexity Change: {perplexity_change:+.2f}")
    print(f"Additional Codes Used: {additional_codes:+d}")

    # Save results
    results['improvement'] = improvement

    results_path = Path("./sage/training/language_vae_analysis_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Compare with vision and audio VAEs
    print("\n" + "=" * 60)
    print("COMPARISON WITH OTHER MODALITIES")
    print("=" * 60)

    vision_results_path = Path("./sage/training/vision_vae_analysis_results.json")
    audio_results_path = Path("./sage/training/audio_vae_analysis_results.json")

    comparisons = []

    if vision_results_path.exists():
        with open(vision_results_path, 'r') as f:
            vision_results = json.load(f)
            vision_improvement = vision_results['improvement']['recon_loss_reduction_percent']
            comparisons.append(('Vision', vision_improvement))

    if audio_results_path.exists():
        with open(audio_results_path, 'r') as f:
            audio_results = json.load(f)
            audio_improvement = audio_results['improvement']['recon_loss_reduction_percent']
            comparisons.append(('Audio', audio_improvement))

    comparisons.append(('Language', recon_improvement))

    print("\nReconstruction Improvement Comparison:")
    for modality, improvement in comparisons:
        print(f"  {modality:10s}: {improvement:5.1f}%")

    # Conclusion
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if recon_improvement > 70:
        print("\n✅ Training significantly improved reconstruction quality!")
    elif recon_improvement > 40:
        print("\n⚠️  Training improved reconstruction moderately.")
    else:
        print("\n❌ Training showed limited improvement.")

    if results['trained']['codes_used'] >= 8:
        print("✅ Trained model uses most/all codes (good diversity).")
    else:
        print("⚠️  Trained model shows some code collapse.")

    print("\nLanguage Puzzle VAE training complete!")
    print("Tri-modal consciousness system (vision, audio, language) ready.")

    return results


if __name__ == "__main__":
    results = analyze_language_vae(num_samples=100)
    if results is None:
        print("\nAnalysis could not be completed.")
        print("Ensure training has finished and model files exist.")
        sys.exit(1)
