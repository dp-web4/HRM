#!/usr/bin/env python3
"""
Train Vision Puzzle VAE on Real Data

Uses CIFAR-10 initially (fast download, manageable size).
Tests whether geometric puzzle encoding preserves visual meaning.

Key Questions:
1. Does reconstruction quality improve with training?
2. Do learned codes develop semantic clustering?
3. Does puzzle space preserve visual categories?
4. What's the compression-quality trade-off?

This is the first REAL training of puzzle space consciousness.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import json
from datetime import datetime

from sage.compression.vision_puzzle_vae import VisionPuzzleVAE


class VisionPuzzleTrainer:
    """Train vision puzzle VAE on real image data"""

    def __init__(
        self,
        model: VisionPuzzleVAE,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        save_dir: str = "./vision_vae_checkpoints"
    ):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Training history
        self.history = {
            'train_loss': [],
            'recon_loss': [],
            'vq_loss': [],
            'perplexity': [],
            'epoch_time': []
        }

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_vq = 0.0
        total_perplexity = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)

            # Forward pass
            outputs = self.model(images)

            # Compute losses
            losses = self.model.compute_loss(
                images,
                outputs['reconstruction'],
                outputs['vq_loss']
            )

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            # Accumulate metrics
            total_loss += losses['total'].item()
            total_recon += losses['reconstruction'].item()
            total_vq += (losses['vq_commitment'].item() + losses['vq_codebook'].item())
            perp = losses['perplexity'] if isinstance(losses['perplexity'], float) else losses['perplexity'].item()
            total_perplexity += perp

            # Update progress bar
            perp_val = losses['perplexity'] if isinstance(losses['perplexity'], float) else losses['perplexity'].item()
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'recon': f"{losses['reconstruction'].item():.4f}",
                'perplexity': f"{perp_val:.2f}"
            })

        # Average metrics
        n_batches = len(dataloader)
        return {
            'train_loss': total_loss / n_batches,
            'recon_loss': total_recon / n_batches,
            'vq_loss': total_vq / n_batches,
            'perplexity': total_perplexity / n_batches
        }

    def validate(self, dataloader):
        """Validate on test set"""
        self.model.eval()

        total_loss = 0.0
        total_recon = 0.0
        total_perplexity = 0.0

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)

                outputs = self.model(images)
                losses = self.model.compute_loss(
                    images,
                    outputs['reconstruction'],
                    outputs['vq_loss']
                )

                total_loss += losses['total'].item()
                total_recon += losses['reconstruction'].item()
                perp = losses['perplexity'] if isinstance(losses['perplexity'], float) else losses['perplexity'].item()
                total_perplexity += perp

        n_batches = len(dataloader)
        return {
            'val_loss': total_loss / n_batches,
            'val_recon': total_recon / n_batches,
            'val_perplexity': total_perplexity / n_batches
        }

    def analyze_code_usage(self, dataloader, num_batches=10):
        """Analyze how codes are being used"""
        self.model.eval()

        code_counts = torch.zeros(self.model.num_codes, device=self.device)

        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= num_batches:
                    break

                images = images.to(self.device)
                puzzles = self.model.encode_to_puzzle(images)

                # Count code usage
                for code in range(self.model.num_codes):
                    code_counts[code] += (puzzles == code).sum()

        # Normalize to percentages
        total = code_counts.sum()
        code_percentages = (code_counts / total * 100).cpu().numpy()

        return {
            'code_counts': code_counts.cpu().numpy(),
            'code_percentages': code_percentages,
            'codes_used': (code_counts > 0).sum().item(),
            'entropy': self._compute_entropy(code_percentages)
        }

    def _compute_entropy(self, percentages):
        """Compute entropy of code distribution (higher = more uniform)"""
        # Convert to torch tensor and normalize to probabilities
        if not isinstance(percentages, torch.Tensor):
            percentages = torch.tensor(percentages, dtype=torch.float32)
        percentages = percentages[percentages > 0] / 100.0
        # Add small epsilon to avoid log(0)
        entropy = -(percentages * torch.log(percentages + 1e-10)).sum()
        return entropy.item()

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history
        }

        # Save latest
        path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (epoch {epoch})")

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        save_every: int = 10
    ):
        """Full training loop"""

        print("=" * 70)
        print("Training Vision Puzzle VAE on Real Data")
        print("=" * 70)
        print(f"\nDevice: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

            # Analyze code usage
            code_analysis = self.analyze_code_usage(train_loader)

            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_loss'])

            epoch_time = time.time() - epoch_start

            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['recon_loss'].append(train_metrics['recon_loss'])
            self.history['vq_loss'].append(train_metrics['vq_loss'])
            self.history['perplexity'].append(train_metrics['perplexity'])
            self.history['epoch_time'].append(epoch_time)

            # Print summary
            print(f"\nEpoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
            print(f"  Recon Loss: {train_metrics['recon_loss']:.4f}")
            print(f"  Perplexity: {train_metrics['perplexity']:.2f}")
            print(f"  Codes Used: {code_analysis['codes_used']}/10")
            print(f"  Code Entropy: {code_analysis['entropy']:.3f}")

            # Save checkpoint
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']

            if epoch % save_every == 0 or is_best:
                all_metrics = {**train_metrics, **val_metrics, **code_analysis}
                self.save_checkpoint(epoch, all_metrics, is_best)

        # Save final history
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"\nBest validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.save_dir}")
        print(f"Final perplexity: {self.history['perplexity'][-1]:.2f}")
        print(f"Codes used: {code_analysis['codes_used']}/10")


def main():
    print("Initializing Vision Puzzle VAE Training...")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    num_epochs = 50
    learning_rate = 1e-3

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR-10
    print("Loading CIFAR-10 dataset...")
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    val_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # Create model
    print("Creating Vision Puzzle VAE...")
    model = VisionPuzzleVAE(
        latent_dim=64,
        num_codes=10,
        commitment_cost=0.25
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Create trainer
    trainer = VisionPuzzleTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        save_dir="./sage/training/vision_vae_checkpoints"
    )

    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_every=5
    )

    print("\nTraining session complete.")
    print("Model ready for consciousness integration.")


if __name__ == "__main__":
    main()
