"""
Train Audio Puzzle VAE on Speech Commands Dataset
==================================================

Trains Audio → Puzzle VAE encoding on real audio data.
Similar to vision VAE training, validates that training improves:
1. Reconstruction quality
2. Codebook utilization
3. Puzzle structure

Dataset: Speech Commands v2 (Google)
- 105K short audio clips (1 second each)
- 35 spoken word classes
- 16kHz mono audio

Author: Autonomous Thor
Date: 2025-11-19
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import sys
from pathlib import Path
from tqdm import tqdm
import json
import time

# Add SAGE to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compression.audio_puzzle_vae import AudioPuzzleVAE

class SpeechCommandsSubset(SPEECHCOMMANDS):
    """Speech Commands dataset wrapper"""
    def __init__(self, root, subset='training', download=True):
        super().__init__(root, download=download, subset=subset)

    def __getitem__(self, idx):
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(idx)

        # Ensure 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or truncate to 1 second (16000 samples)
        target_length = 16000
        if waveform.shape[1] < target_length:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
        elif waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]

        # Remove channel dimension for model input
        waveform = waveform.squeeze(0)

        return waveform

def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_perplexity = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, waveforms in enumerate(pbar):
        waveforms = waveforms.to(device)

        # Forward pass
        output = model(waveforms)
        mel_spec_input = output['mel_spec_input']
        reconstruction = output['reconstruction']
        vq_loss = output['vq_loss']

        # Compute losses
        losses = model.compute_loss(mel_spec_input, reconstruction, vq_loss)
        loss = losses['total']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_recon_loss += losses['reconstruction'].item()
        total_perplexity += vq_loss['perplexity']
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'recon': f"{losses['reconstruction'].item():.4f}",
            'perplexity': f"{vq_loss['perplexity']:.2f}"
        })

    # Compute averages
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_perplexity = total_perplexity / num_batches

    return {
        'loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'perplexity': avg_perplexity
    }

def validate(model, dataloader, device):
    """Validate on test set"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_perplexity = 0
    code_usage = torch.zeros(10)
    num_batches = 0

    with torch.no_grad():
        for waveforms in tqdm(dataloader, desc="Validating"):
            waveforms = waveforms.to(device)

            # Forward pass
            output = model(waveforms)
            mel_spec_input = output['mel_spec_input']
            reconstruction = output['reconstruction']
            vq_loss = output['vq_loss']
            puzzles = output['puzzles']

            # Compute losses
            losses = model.compute_loss(mel_spec_input, reconstruction, vq_loss)

            # Track metrics
            total_loss += losses['total'].item()
            total_recon_loss += losses['reconstruction'].item()
            total_perplexity += vq_loss['perplexity']

            # Track code usage
            unique_codes = puzzles.unique()
            for code in unique_codes:
                code_usage[code.item()] += 1

            num_batches += 1

    # Compute averages
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_perplexity = total_perplexity / num_batches
    codes_used = (code_usage > 0).sum().item()

    return {
        'loss': avg_loss,
        'recon_loss': avg_recon_loss,
        'perplexity': avg_perplexity,
        'codes_used': codes_used
    }

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    batch_size = 64
    num_epochs = 5
    learning_rate = 1e-3

    print("="*60)
    print("Audio Puzzle VAE Training")
    print("="*60)
    print(f"Dataset: Speech Commands v2")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("="*60 + "\n")

    # Load dataset
    print("Loading Speech Commands dataset...")
    train_dataset = SpeechCommandsSubset('./data', subset='training', download=True)
    val_dataset = SpeechCommandsSubset('./data', subset='validation', download=True)

    # Use subset for faster training (similar to CIFAR-10 scale)
    # Full dataset is 105K samples, we'll use ~50K for training
    train_size = min(len(train_dataset), 50000)
    train_subset, _ = random_split(
        train_dataset,
        [train_size, len(train_dataset) - train_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_size = min(len(val_dataset), 10000)
    val_subset, _ = random_split(
        val_dataset,
        [val_size, len(val_dataset) - val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print()

    # Create model
    print("Creating Audio Puzzle VAE...")
    model = AudioPuzzleVAE(latent_dim=64).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    checkpoint_dir = Path('sage/training/audio_vae_checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Train Recon: {train_metrics['recon_loss']:.4f}")
        print(f"  Perplexity: {train_metrics['perplexity']:.2f}")

        # Validate
        val_metrics = validate(model, val_loader, device)
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Recon: {val_metrics['recon_loss']:.4f}")
        print(f"  Codes Used: {val_metrics['codes_used']}/10")

        # Save history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }

        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  Best model updated!")

    # Training complete
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")

    # Save training history
    history_path = 'sage/training/audio_vae_training_history.json'
    with open(history_path, 'w') as f:
        # Convert to serializable format
        history_serializable = {
            'train': [{k: float(v) for k, v in m.items()} for m in history['train']],
            'val': [{k: float(v) for k, v in m.items()} for m in history['val']]
        }
        json.dump(history_serializable, f, indent=2)
    print(f"History saved to: {history_path}")

    print("\nNext step: Run quick_audio_vae_analysis.py to compare with untrained baseline")

if __name__ == '__main__':
    main()
