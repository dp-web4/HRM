#!/usr/bin/env python3
"""
Training script for Language Puzzle VAE

Trains character-level VQ-VAE on text data to complete the tri-modal consciousness system.
Similar to vision/audio VAE training but operates on symbolic character sequences.

Dataset: WikiText-2 (character-level)
Model: Language Puzzle VAE (30×30 grid, 10 codes, 64D latent)
Training: 5 epochs, similar to vision/audio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.compression.language_puzzle_vae_small import LanguagePuzzleVAESmall, text_to_indices


class TextDataset(Dataset):
    """
    Character-level text dataset

    Loads text file and creates fixed-length character sequences
    """

    def __init__(self, text_file: str, seq_length: int = 128):
        """
        Args:
            text_file: Path to text file
            seq_length: Character sequence length
        """
        self.seq_length = seq_length

        # Load and process text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Filter to printable ASCII
        text = ''.join(c if 32 <= ord(c) < 127 or c == '\n' else ' ' for c in text)

        # Create sequences (sliding window with stride)
        self.sequences = []
        stride = seq_length // 2  # 50% overlap

        for i in range(0, len(text) - seq_length, stride):
            seq = text[i:i + seq_length]
            # Only keep sequences with enough content (not all spaces/newlines)
            if len(seq.strip()) >= seq_length // 4:
                self.sequences.append(seq)

        print(f"Created {len(self.sequences)} sequences from {len(text)} characters")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        text = self.sequences[idx]
        indices = text_to_indices(text, max_length=self.seq_length)
        return indices


def download_wikitext2():
    """Download WikiText-2 dataset if not present"""
    data_dir = Path("./data/wikitext")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "wiki.train.txt"
    val_file = data_dir / "wiki.valid.txt"

    if train_file.exists() and val_file.exists():
        print(f"WikiText-2 dataset already exists at {data_dir}")
        return str(train_file), str(val_file)

    print("Downloading WikiText-2 dataset...")

    try:
        # Try using datasets library
        from datasets import load_dataset
        print("Using Hugging Face datasets library...")

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)

        # Save train split
        with open(train_file, 'w', encoding='utf-8') as f:
            for example in dataset['train']:
                f.write(example['text'] + '\n')

        # Save validation split
        with open(val_file, 'w', encoding='utf-8') as f:
            for example in dataset['validation']:
                f.write(example['text'] + '\n')

        print(f"Downloaded WikiText-2 to {data_dir}")

    except ImportError:
        # Fallback: create synthetic text data from available sources
        print("Hugging Face datasets not available, using alternative dataset...")

        # Use Python's standard library documentation as training data
        import inspect
        import random

        # Collect text from various modules
        texts = []

        # Add some sample texts
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks learn patterns from data.",
            "Natural language processing enables computers to understand text.",
            "Deep learning uses multi-layer neural networks.",
            "Consciousness emerges from the integration of information.",
            "Puzzle space provides a unified representation for multi-modal data.",
            "Vision, audio, and language can be encoded in a common format.",
            "Vector quantization compresses continuous representations.",
            "The VQ-VAE learns discrete latent codes for data compression.",
        ]

        # Generate synthetic training data by repeating and permuting
        train_text = []
        val_text = []

        # Create enough training samples
        for _ in range(10000):  # ~50k sequences
            text = random.choice(sample_texts)
            # Add some variation
            text = text + " " + random.choice(sample_texts)
            train_text.append(text)

        for _ in range(2000):  # ~10k sequences
            text = random.choice(sample_texts)
            text = text + " " + random.choice(sample_texts)
            val_text.append(text)

        # Write files
        with open(train_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(train_text))

        with open(val_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(val_text))

        print(f"Created synthetic text dataset at {data_dir}")

    return str(train_file), str(val_file)


def train_language_vae(
    epochs: int = 5,
    batch_size: int = 32,  # Reduced from 64 to save memory
    learning_rate: float = 0.001,
    seq_length: int = 128,
    device: str = None
):
    """Train Language Puzzle VAE"""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print()
    print("=" * 60)
    print("Language Puzzle VAE Training")
    print("=" * 60)
    print(f"Sequence length: {seq_length} characters")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)
    print()

    # Download dataset
    train_file, val_file = download_wikitext2()

    # Create datasets
    print("Loading WikiText-2 dataset...")
    train_dataset = TextDataset(train_file, seq_length=seq_length)
    val_dataset = TextDataset(val_file, seq_length=seq_length)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()

    # Create dataloaders (num_workers=0 to avoid multiprocessing issues)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model (small version)
    print("Creating Language Puzzle VAE (Small)...")
    model = LanguagePuzzleVAESmall(
        vocab_size=128,  # Printable ASCII only
        char_embed_dim=32,
        latent_dim=64,
        num_codes=10,
        max_length=seq_length
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        'train': [],
        'val': []
    }

    # Create checkpoint directory
    checkpoint_dir = Path("./sage/training/language_vae_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 60)

        # Training
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_vq_loss = 0
        train_perplexity = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, text_indices in enumerate(pbar):
            text_indices = text_indices.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(text_indices)
            char_logits = output['char_logits']
            vq_loss = output['vq_loss']

            # Reconstruction loss (cross-entropy)
            recon_loss = F.cross_entropy(
                char_logits.reshape(-1, 128),  # Updated vocab size
                text_indices.reshape(-1),
                ignore_index=0  # Ignore padding
            )

            # Total loss
            loss = recon_loss + vq_loss

            # Backward
            loss.backward()
            optimizer.step()

            # Calculate perplexity (codebook utilization metric)
            with torch.no_grad():
                puzzles = output['puzzles']
                puzzle_flat = puzzles.reshape(-1)
                unique_codes = len(torch.unique(puzzle_flat))
                perplexity = 2 ** (-torch.log2(torch.bincount(puzzle_flat, minlength=10).float() / len(puzzle_flat) + 1e-10).sum())

            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_vq_loss += vq_loss.item()
            train_perplexity += perplexity.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'perplexity': f'{perplexity.item():.2f}'
            })

        # Average training metrics
        train_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_vq_loss /= len(train_loader)
        train_perplexity /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_vq_loss = 0
        val_perplexity = 0
        codes_used_total = 0

        with torch.no_grad():
            for text_indices in val_loader:
                text_indices = text_indices.to(device)

                output = model(text_indices)
                char_logits = output['char_logits']
                vq_loss = output['vq_loss']

                recon_loss = F.cross_entropy(
                    char_logits.reshape(-1, 128),  # Updated vocab size
                    text_indices.reshape(-1),
                    ignore_index=0
                )

                loss = recon_loss + vq_loss

                # Perplexity and code usage
                puzzles = output['puzzles']
                puzzle_flat = puzzles.reshape(-1)
                unique_codes = len(torch.unique(puzzle_flat))
                perplexity = 2 ** (-torch.log2(torch.bincount(puzzle_flat, minlength=10).float() / len(puzzle_flat) + 1e-10).sum())

                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_vq_loss += vq_loss.item()
                val_perplexity += perplexity.item()
                codes_used_total += unique_codes

        # Average validation metrics
        val_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_vq_loss /= len(val_loader)
        val_perplexity /= len(val_loader)
        codes_used = codes_used_total / len(val_loader)

        # Log metrics
        print(f"\nTrain - Loss: {train_loss:.4f}, Recon: {train_recon_loss:.4f}, VQ: {train_vq_loss:.6f}, Perplexity: {train_perplexity:.2f}")
        print(f"Val   - Loss: {val_loss:.4f}, Recon: {val_recon_loss:.4f}, VQ: {val_vq_loss:.6f}, Perplexity: {val_perplexity:.2f}, Codes: {codes_used:.1f}/10")

        # Save metrics
        history['train'].append({
            'loss': train_loss,
            'recon_loss': train_recon_loss,
            'vq_loss': train_vq_loss,
            'perplexity': train_perplexity
        })

        history['val'].append({
            'loss': val_loss,
            'recon_loss': val_recon_loss,
            'vq_loss': val_vq_loss,
            'perplexity': val_perplexity,
            'codes_used': codes_used
        })

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(model.state_dict(), best_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")

    # Save training history
    history_path = Path("./sage/training/language_vae_training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Training history: {history_path}")
    print("=" * 60)

    return model, history


if __name__ == "__main__":
    # Train with reduced batch size for memory efficiency
    model, history = train_language_vae(
        epochs=5,
        batch_size=32,  # Reduced from 64
        learning_rate=0.001,
        seq_length=128
    )
