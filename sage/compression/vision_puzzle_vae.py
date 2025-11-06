#!/usr/bin/env python3
"""
Vision → Puzzle VAE

Encodes visual input (224×224×3 images) to 30×30×10 puzzle space using VQ-VAE.
The puzzle space provides a universal geometric interface for reasoning.

Architecture:
- Encoder: Image → 64D latent features per spatial position
- VQ Codebook: 10 discrete codes
- Decoder: Codes → reconstructed image
- Puzzle Output: 30×30 grid with values 0-9
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import math


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for discrete puzzle codes

    Maps continuous latents to nearest discrete code from codebook.
    Uses straight-through estimator for gradients.
    """

    def __init__(self, num_codes: int = 10, code_dim: int = 64, commitment_cost: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost

        # Codebook: 10 learnable code vectors
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1/num_codes, 1/num_codes)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Quantize latents to nearest codes

        Args:
            latents: [batch, height, width, code_dim]

        Returns:
            quantized: Quantized latents (same shape as input)
            codes: Code indices [batch, height, width]
            vq_loss: Quantization losses
        """
        # Flatten spatial dims for distance computation
        latents_shape = latents.shape
        flat_latents = latents.reshape(-1, self.code_dim)  # [B*H*W, code_dim]

        # Compute distances to all codes
        distances = torch.cdist(
            flat_latents,
            self.codebook.weight,
            p=2.0
        )  # [B*H*W, num_codes]

        # Find nearest codes
        codes = distances.argmin(dim=-1)  # [B*H*W]
        codes = codes.reshape(latents_shape[:-1])  # [batch, height, width]

        # Look up quantized values
        quantized_flat = self.codebook(codes.reshape(-1))  # [B*H*W, code_dim]
        quantized = quantized_flat.reshape(latents_shape)  # [batch, H, W, code_dim]

        # Compute VQ losses
        commitment_loss = F.mse_loss(latents, quantized.detach())
        codebook_loss = F.mse_loss(quantized, latents.detach())

        # Straight-through estimator: use quantized for forward, latents gradient for backward
        quantized = latents + (quantized - latents).detach()

        vq_loss = {
            'commitment': commitment_loss * self.commitment_cost,
            'codebook': codebook_loss,
            'perplexity': self._compute_perplexity(codes)
        }

        return quantized, codes, vq_loss

    def _compute_perplexity(self, codes: torch.Tensor) -> float:
        """Compute codebook usage perplexity (higher = more codes used)"""
        # Count code usage
        code_counts = torch.bincount(codes.flatten(), minlength=self.num_codes).float()
        code_probs = code_counts / code_counts.sum()

        # Perplexity = exp(entropy)
        entropy = -(code_probs * torch.log(code_probs + 1e-10)).sum()
        perplexity = torch.exp(entropy).item()

        return perplexity


class VisionPuzzleEncoder(nn.Module):
    """Encode 224×224×3 image to 30×30×64 latent features"""

    def __init__(self, latent_dim: int = 64):
        super().__init__()

        # CNN encoder: 224×224 → 30×30
        # Uses strided convs + residual blocks
        self.encoder = nn.Sequential(
            # 224×224×3 → 112×112×32
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 112×112×32 → 56×56×64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 56×56×64 → 28×28×128 (close to 30×30)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Upsample slightly 28×28 → 30×30
            nn.Upsample(size=(30, 30), mode='bilinear', align_corners=False),

            # Project to latent dim
            nn.Conv2d(128, latent_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 3, 224, 224]
        Returns:
            latents: [batch, latent_dim, 30, 30]
        """
        return self.encoder(x)


class VisionPuzzleDecoder(nn.Module):
    """Decode 30×30×64 latent features back to 224×224×3 image"""

    def __init__(self, latent_dim: int = 64):
        super().__init__()

        # Transposed CNN decoder: 30×30 → 224×224
        self.decoder = nn.Sequential(
            # Project up from latent dim
            nn.Conv2d(latent_dim, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 30×30×128 → 60×60×64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 60×60×64 → 120×120×32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 120×120×32 → 224×224×3 (with output_padding to get exact size)
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, output_padding=(0, 0)),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, latent_dim, 30, 30]
        Returns:
            reconstruction: [batch, 3, 224, 224]
        """
        return self.decoder(x)


class VisionPuzzleVAE(nn.Module):
    """
    Complete Vision → Puzzle VAE

    Encodes images to 30×30 puzzle grids with 10 discrete values per cell.
    Can reconstruct images from puzzles and enables geometric reasoning.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        num_codes: int = 10,
        commitment_cost: float = 0.25
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_codes = num_codes

        self.encoder = VisionPuzzleEncoder(latent_dim)
        self.vq = VectorQuantizer(num_codes, latent_dim, commitment_cost)
        self.decoder = VisionPuzzleDecoder(latent_dim)

    def encode_to_puzzle(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to puzzle space

        Args:
            images: [batch, 3, 224, 224] in range [0, 1] or [-1, 1]

        Returns:
            puzzles: [batch, 30, 30] with values 0-9
        """
        # Normalize to [-1, 1] if needed
        if images.min() >= 0:
            images = images * 2 - 1

        # Encode
        latents = self.encoder(images)  # [B, latent_dim, 30, 30]

        # Rearrange for VQ: [B, H, W, C]
        latents = latents.permute(0, 2, 3, 1)  # [B, 30, 30, latent_dim]

        # Quantize
        _, puzzles, _ = self.vq(latents)  # [B, 30, 30]

        return puzzles

    def decode_from_puzzle(self, puzzles: torch.Tensor) -> torch.Tensor:
        """
        Decode puzzles back to images

        Args:
            puzzles: [batch, 30, 30] with values 0-9

        Returns:
            images: [batch, 3, 224, 224] in range [-1, 1]
        """
        # Look up quantized vectors
        quantized = self.vq.codebook(puzzles)  # [B, 30, 30, latent_dim]

        # Rearrange for decoder: [B, C, H, W]
        quantized = quantized.permute(0, 3, 1, 2)  # [B, latent_dim, 30, 30]

        # Decode
        reconstruction = self.decoder(quantized)  # [B, 3, 224, 224]

        return reconstruction

    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        Full forward pass: encode → quantize → decode

        Args:
            images: [batch, 3, 224, 224]

        Returns:
            Dictionary with:
                - puzzles: [batch, 30, 30]
                - reconstruction: [batch, 3, 224, 224]
                - vq_loss: Dict of VQ losses
        """
        # Normalize
        if images.min() >= 0:
            images = images * 2 - 1

        # Encode
        latents = self.encoder(images)
        latents = latents.permute(0, 2, 3, 1)

        # Quantize
        quantized, puzzles, vq_loss = self.vq(latents)

        # Decode
        quantized = quantized.permute(0, 3, 1, 2)
        reconstruction = self.decoder(quantized)

        return {
            'puzzles': puzzles,
            'reconstruction': reconstruction,
            'vq_loss': vq_loss
        }

    def compute_loss(
        self,
        images: torch.Tensor,
        reconstruction: torch.Tensor,
        vq_loss: Dict[str, torch.Tensor],
        recon_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses

        Args:
            images: Original images
            reconstruction: Reconstructed images
            vq_loss: VQ losses from forward pass
            recon_weight: Weight for reconstruction loss

        Returns:
            Dictionary of losses
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, images)

        # Total loss
        total_loss = (
            recon_weight * recon_loss +
            vq_loss['commitment'] +
            vq_loss['codebook']
        )

        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'vq_commitment': vq_loss['commitment'],
            'vq_codebook': vq_loss['codebook'],
            'perplexity': vq_loss['perplexity']
        }


def test_vision_puzzle_vae():
    """Test the VAE with random images"""
    print("=" * 70)
    print("Testing Vision → Puzzle VAE")
    print("=" * 70)

    # Create model
    vae = VisionPuzzleVAE(latent_dim=64, num_codes=10)

    # Random batch of images
    batch_size = 4
    images = torch.rand(batch_size, 3, 224, 224)  # [0, 1] range

    print(f"\nInput images: {images.shape}")
    print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")

    # Encode to puzzles
    with torch.no_grad():
        puzzles = vae.encode_to_puzzle(images)

    print(f"\nPuzzles: {puzzles.shape}")
    print(f"Value range: [{puzzles.min()}, {puzzles.max()}]")
    print(f"Unique values: {len(torch.unique(puzzles))} (should be ~10)")

    # Show puzzle for first image
    print(f"\nFirst puzzle (30×30):")
    print(puzzles[0])

    # Decode back to images
    with torch.no_grad():
        reconstruction = vae.decode_from_puzzle(puzzles)

    print(f"\nReconstruction: {reconstruction.shape}")
    print(f"Value range: [{reconstruction.min():.3f}, {reconstruction.max():.3f}]")

    # Full forward pass
    outputs = vae(images)
    losses = vae.compute_loss(
        images * 2 - 1,  # Normalize
        outputs['reconstruction'],
        outputs['vq_loss']
    )

    print(f"\nLosses:")
    for key, value in losses.items():
        if key == 'perplexity':
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value.item():.4f}")

    print("\n" + "=" * 70)
    print("Vision → Puzzle VAE test complete!")
    print("=" * 70)

    return vae, puzzles, outputs


if __name__ == "__main__":
    test_vision_puzzle_vae()
