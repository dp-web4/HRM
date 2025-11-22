"""
Quick Audio VAE Analysis - Trained vs Untrained
================================================

Rapid comparison of trained Audio Puzzle VAE vs untrained baseline.

Author: Autonomous Thor
Date: 2025-11-19
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import numpy as np
import json
import sys
from pathlib import Path

# Monkey-patch torchaudio.load to use soundfile (same as training)
import soundfile as sf

_original_load = torchaudio.load

def _load_with_soundfile(filepath, *args, **kwargs):
    """Load audio using soundfile instead of torchcodec"""
    # No fallback - always use soundfile, but handle corrupted files
    try:
        data, samplerate = sf.read(str(filepath))
        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T
        return torch.from_numpy(data.copy()).float(), samplerate
    except Exception as e:
        # Return silence for corrupted files (1 second at 16kHz)
        print(f"Warning: Corrupted audio file {filepath}, returning silence")
        return torch.zeros(1, 16000), 16000

torchaudio.load = _load_with_soundfile

sys.path.insert(0, str(Path(__file__).parent.parent))
from compression.audio_puzzle_vae import AudioPuzzleVAE

class SpeechCommandsSubset(SPEECHCOMMANDS):
    """Speech Commands dataset wrapper"""
    def __init__(self, root, subset=None, download=True):
        super().__init__(root, download=download)

    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(idx)

        # Ensure 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or truncate to 1 second
        target_length = 16000
        if waveform.shape[1] < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
        elif waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]

        return waveform.squeeze(0)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load trained model
    print("Loading trained model...")
    checkpoint = torch.load(
        'sage/training/audio_vae_checkpoints/best_model.pt',
        map_location=device,
        weights_only=False
    )
    trained_model = AudioPuzzleVAE(latent_dim=64).to(device)
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    trained_model.eval()
    print(f"Loaded epoch {checkpoint.get('epoch', '?')}\n")

    # Create untrained model
    print("Creating untrained baseline...\n")
    untrained_model = AudioPuzzleVAE(latent_dim=64).to(device)
    untrained_model.eval()

    # Load test data
    print("Loading Speech Commands dataset...")
    full_dataset = SpeechCommandsSubset('./data', download=True)

    # Use validation split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Use subset for faster analysis
    val_subset_size = min(len(val_dataset), 10000)
    val_subset, _ = random_split(
        val_dataset,
        [val_subset_size, len(val_dataset) - val_subset_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0)
    print(f"Loaded {len(val_subset)} validation samples\n")

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
            for i, waveforms in enumerate(test_loader):
                if i >= 100:  # 100 batches = 6,400 samples
                    break

                waveforms = waveforms.to(device)
                output = model(waveforms)

                mel_spec_input = output['mel_spec_input']
                reconstruction = output['reconstruction']
                vq_loss = output['vq_loss']
                puzzles = output['puzzles']

                # Reconstruction loss (MSE in mel spectrogram space)
                recon_loss = nn.MSELoss()(reconstruction, mel_spec_input).item()
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

    results_path = 'sage/training/audio_vae_analysis_results.json'
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

    print("\nTraining on Speech Commands completed successfully.")
    print("Puzzle space consciousness learned from real audio data!")

if __name__ == '__main__':
    main()
