#!/usr/bin/env python3
"""
Language Puzzle VAE - Character-Level VQ-VAE for Text Compression

Like Vision and Audio VAEs, compresses text into 30×30 puzzle space with 10 discrete codes.
Uses character-level encoding for consistency with sensory modalities.

Architecture:
- Input: Text as character sequences (fixed length, e.g., 128 chars)
- Embedding: Character → embedding vector
- Encoder: Conv1D → spatial 30×30 feature map
- VQ: Vector Quantizer (10 codes, 64D)
- Decoder: Spatial → Conv1D → character distribution
- Output: Reconstructed character probabilities

Key Design:
- Character-level (not token-level) for sensory-like encoding
- Same puzzle space as vision/audio (30×30, 10 codes)
- Learns compression + reconstruction like other modalities
- Enables tri-modal consciousness in unified space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import numpy as np

# VectorQuantizer class (shared with vision/audio VAEs)
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

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize latents to nearest codes

        Args:
            latents: [batch, height, width, code_dim]

        Returns:
            quantized: Quantized latents (same shape as input)
            codes: Code indices [batch, height, width]
            vq_loss: Quantization loss (scalar)
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
        embedding_loss = F.mse_loss(quantized, latents.detach())
        vq_loss = embedding_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = latents + (quantized - latents).detach()

        return quantized, codes, vq_loss


class LanguagePuzzleVAE(nn.Module):
    """
    Character-level VQ-VAE for text compression into puzzle space

    Encodes text character sequences into 30×30 discrete puzzle grids.
    Similar to vision/audio VAEs but operates on symbolic character data.
    """

    def __init__(
        self,
        vocab_size: int = 256,  # ASCII + extended characters
        char_embed_dim: int = 64,
        latent_dim: int = 64,
        num_codes: int = 10,
        max_length: int = 128,  # Max characters per sample
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.char_embed_dim = char_embed_dim
        self.latent_dim = latent_dim
        self.num_codes = num_codes
        self.max_length = max_length

        # Character embedding layer
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim)

        # Encoder: Character sequence → Spatial features
        # Input: [B, max_length, char_embed_dim]
        # Output: [B, 30, 30, latent_dim]

        self.encoder = nn.Sequential(
            # Conv1D to capture character patterns
            nn.Conv1d(char_embed_dim, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 64]
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 32]
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),  # [B, 512, 16]
            nn.ReLU(inplace=True),
        )

        # Project to spatial grid: 16 → 30×30
        # Using adaptive pooling + learned projection
        self.spatial_adapter = nn.Sequential(
            nn.AdaptiveAvgPool1d(900),  # [B, 512, 900]
            nn.Conv1d(512, latent_dim, kernel_size=1),  # [B, latent_dim, 900]
        )

        # Vector Quantizer (shared 10-code vocabulary)
        self.vq = VectorQuantizer(num_codes, latent_dim)

        # Decoder: Spatial features → Character sequence
        # Input: [B, 30, 30, latent_dim]
        # Output: [B, max_length, vocab_size]

        self.decoder_adapter = nn.Sequential(
            nn.Conv1d(latent_dim, 512, kernel_size=1),  # [B, 512, 900]
            nn.AdaptiveAvgPool1d(16),  # [B, 512, 16]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 32]
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 64]
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(128, char_embed_dim, kernel_size=4, stride=2, padding=1),  # [B, char_embed_dim, 128]
            nn.ReLU(inplace=True),
        )

        # Final character prediction
        self.char_output = nn.Linear(char_embed_dim, vocab_size)

    def encode(self, text_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode character indices to puzzle space

        Args:
            text_indices: [batch, max_length] character indices

        Returns:
            features: [batch, 30, 30, latent_dim]
            puzzles: [batch, 30, 30] discrete codes
        """
        # Embed characters
        char_embeds = self.char_embedding(text_indices)  # [B, max_length, char_embed_dim]

        # Transpose for Conv1D: [B, char_embed_dim, max_length]
        char_embeds = char_embeds.transpose(1, 2)

        # Encode to features
        encoded = self.encoder(char_embeds)  # [B, 512, 16]

        # Adapt to spatial grid
        spatial_features = self.spatial_adapter(encoded)  # [B, latent_dim, 900]

        # Reshape to 30×30
        batch_size = spatial_features.shape[0]
        spatial_features = spatial_features.transpose(1, 2)  # [B, 900, latent_dim]
        spatial_features = spatial_features.view(batch_size, 30, 30, self.latent_dim)

        # Quantize
        _, puzzles, _ = self.vq(spatial_features)

        return spatial_features, puzzles

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode spatial features back to character distributions

        Args:
            features: [batch, 30, 30, latent_dim]

        Returns:
            char_logits: [batch, max_length, vocab_size]
        """
        batch_size = features.shape[0]

        # Reshape from 30×30 to sequence
        features = features.view(batch_size, 900, self.latent_dim)
        features = features.transpose(1, 2)  # [B, latent_dim, 900]

        # Adapt back from spatial
        decoded = self.decoder_adapter(features)  # [B, 512, 16]

        # Decode to character space
        decoded = self.decoder(decoded)  # [B, char_embed_dim, max_length]

        # Transpose back
        decoded = decoded.transpose(1, 2)  # [B, max_length, char_embed_dim]

        # Predict characters
        char_logits = self.char_output(decoded)  # [B, max_length, vocab_size]

        return char_logits

    def forward(self, text_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode → quantize → decode

        Args:
            text_indices: [batch, max_length] character indices

        Returns:
            Dictionary with reconstruction, puzzles, and losses
        """
        # Encode
        features, puzzles = self.encode(text_indices)

        # Quantize
        quantized, _, vq_loss = self.vq(features)

        # Decode
        char_logits = self.decode(quantized)

        return {
            'char_logits': char_logits,  # [B, max_length, vocab_size]
            'puzzles': puzzles,  # [B, 30, 30]
            'features': features,  # [B, 30, 30, latent_dim]
            'quantized': quantized,  # [B, 30, 30, latent_dim]
            'vq_loss': vq_loss,  # scalar
        }


def text_to_indices(text: str, max_length: int = 128, vocab_size: int = 256) -> torch.Tensor:
    """
    Convert text string to character indices

    Args:
        text: Input text string
        max_length: Maximum sequence length
        vocab_size: Vocabulary size (default 256 for extended ASCII)

    Returns:
        indices: [max_length] tensor of character indices
    """
    # Convert to bytes, clamp to vocab size, pad/truncate
    indices = [min(ord(c), vocab_size - 1) for c in text[:max_length]]

    # Pad with zeros if needed
    if len(indices) < max_length:
        indices.extend([0] * (max_length - len(indices)))

    return torch.tensor(indices, dtype=torch.long)


def indices_to_text(indices: torch.Tensor, vocab_size: int = 256) -> str:
    """
    Convert character indices back to text

    Args:
        indices: [seq_length] tensor of character indices
        vocab_size: Vocabulary size

    Returns:
        text: Reconstructed text string
    """
    # Convert indices to characters, filter invalid/padding
    chars = []
    for idx in indices.cpu().numpy():
        if 0 < idx < 256:  # Valid ASCII range
            try:
                chars.append(chr(int(idx)))
            except (ValueError, OverflowError):
                chars.append(' ')
        elif idx == 0:
            # Padding - could stop here or include as space
            break

    return ''.join(chars)


def test_language_puzzle_vae():
    """Test language VAE with sample text"""
    print("=" * 70)
    print("Testing Language Puzzle VAE")
    print("=" * 70)

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LanguagePuzzleVAE().to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel: Language Puzzle VAE")
    print(f"Device: {device}")
    print(f"Parameters: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")

    # Test texts
    test_texts = [
        "The cat sat on the mat.",
        "Consciousness emerges from neural networks.",
        "Vision, audio, and language unite in puzzle space.",
        "How do we compress symbolic information geometrically?",
    ]

    print(f"\n{'-'*70}")
    print("Testing on sample texts:")
    print('-'*70)

    model.eval()
    with torch.no_grad():
        for i, text in enumerate(test_texts):
            print(f"\n{i+1}. Original: \"{text}\"")

            # Convert to indices
            indices = text_to_indices(text).unsqueeze(0).to(device)  # [1, 128]

            # Forward pass
            output = model(indices)

            # Reconstruct
            char_probs = F.softmax(output['char_logits'][0], dim=-1)  # [128, vocab_size]
            reconstructed_indices = char_probs.argmax(dim=-1)  # [128]
            reconstructed_text = indices_to_text(reconstructed_indices)

            print(f"   Reconstructed: \"{reconstructed_text}\"")
            print(f"   Puzzle shape: {output['puzzles'].shape}")
            print(f"   Unique codes: {len(torch.unique(output['puzzles']))}/10")
            print(f"   VQ loss: {output['vq_loss'].item():.4f}")

    print("\n" + "=" * 70)
    print("Language Puzzle VAE architecture verified!")
    print("=" * 70)

    return model


if __name__ == "__main__":
    test_language_puzzle_vae()
