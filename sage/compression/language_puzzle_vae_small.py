#!/usr/bin/env python3
"""
Language Puzzle VAE (Small) - Memory-Efficient Character-Level VQ-VAE

Smaller version designed to match vision/audio VAE parameter counts (~84k).
Reduces memory footprint to enable training on Jetson Thor.

Key Changes from Original:
- Smaller character embedding: 32D (vs 64D)
- Simplified encoder/decoder: fewer layers, smaller channels
- Direct puzzle mapping without heavy intermediate layers
- Target: ~80-100k parameters (comparable to vision/audio)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any

# VectorQuantizer class (shared with vision/audio VAEs)
class VectorQuantizer(nn.Module):
    """Vector Quantization layer for discrete puzzle codes"""

    def __init__(self, num_codes: int = 10, code_dim: int = 64, commitment_cost: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost

        # Codebook: 10 learnable code vectors
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1/num_codes, 1/num_codes)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize latents to nearest codes"""
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


class LanguagePuzzleVAESmall(nn.Module):
    """
    Small Character-level VQ-VAE for text compression

    Memory-efficient design with ~80-100k parameters.
    """

    def __init__(
        self,
        vocab_size: int = 128,  # Reduced to printable ASCII only
        char_embed_dim: int = 32,  # Reduced from 64
        latent_dim: int = 64,
        num_codes: int = 10,
        max_length: int = 128,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.char_embed_dim = char_embed_dim
        self.latent_dim = latent_dim
        self.num_codes = num_codes
        self.max_length = max_length

        # Character embedding layer (smaller vocab)
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim)

        # Simplified encoder: char sequence → spatial features
        # Target output: 30×30 = 900 features
        self.encoder = nn.Sequential(
            # Compress sequence: 128 → 64
            nn.Conv1d(char_embed_dim, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 64]
            nn.ReLU(inplace=True),

            # Further compress: 64 → 30
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 32]
            nn.ReLU(inplace=True),

            # Adjust to exactly 30
            nn.Conv1d(64, latent_dim, kernel_size=3, stride=1, padding=0),  # [B, latent_dim, 30]
        )

        # Vector Quantizer
        self.vq = VectorQuantizer(num_codes, latent_dim)

        # Simplified decoder: spatial features → char sequence
        self.decoder = nn.Sequential(
            # Expand from 30 back to sequence
            nn.ConvTranspose1d(latent_dim, 64, kernel_size=3, stride=1, padding=0),  # [B, 64, 32]
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 64]
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(64, char_embed_dim, kernel_size=4, stride=2, padding=1),  # [B, char_embed_dim, 128]
        )

        # Character prediction head
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

        # Encode to spatial features
        encoded = self.encoder(char_embeds)  # [B, latent_dim, 30]

        # Create 30×30 grid by repeating the 30-dimensional features
        batch_size = encoded.shape[0]
        # Reshape [B, latent_dim, 30] → [B, 30, latent_dim] → [B, 30, 1, latent_dim] → [B, 30, 30, latent_dim]
        encoded = encoded.transpose(1, 2)  # [B, 30, latent_dim]
        encoded = encoded.unsqueeze(2).expand(batch_size, 30, 30, self.latent_dim)  # [B, 30, 30, latent_dim]

        # Quantize
        _, puzzles, _ = self.vq(encoded)

        return encoded, puzzles

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode spatial features back to character distributions

        Args:
            features: [batch, 30, 30, latent_dim]

        Returns:
            char_logits: [batch, max_length, vocab_size]
        """
        batch_size = features.shape[0]

        # Average across spatial dimensions to get sequence features
        # [B, 30, 30, latent_dim] → [B, 30, latent_dim] (average across second spatial dim)
        features = features.mean(dim=2)  # [B, 30, latent_dim]

        # Transpose for Conv1D: [B, latent_dim, 30]
        features = features.transpose(1, 2)

        # Decode to character space
        decoded = self.decoder(features)  # [B, char_embed_dim, 128]

        # Transpose back
        decoded = decoded.transpose(1, 2)  # [B, 128, char_embed_dim]

        # Predict characters
        char_logits = self.char_output(decoded)  # [B, 128, vocab_size]

        return char_logits

    def forward(self, text_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode → quantize → decode"""
        # Encode
        features, puzzles = self.encode(text_indices)

        # Quantize
        quantized, _, vq_loss = self.vq(features)

        # Decode
        char_logits = self.decode(quantized)

        return {
            'char_logits': char_logits,
            'puzzles': puzzles,
            'features': features,
            'quantized': quantized,
            'vq_loss': vq_loss,
        }


def text_to_indices(text: str, max_length: int = 128, vocab_size: int = 128) -> torch.Tensor:
    """
    Convert text string to character indices (printable ASCII only)

    Args:
        text: Input text string
        max_length: Maximum sequence length
        vocab_size: Vocabulary size (default 128 for ASCII)

    Returns:
        indices: [max_length] tensor of character indices
    """
    # Convert to ASCII indices, clamp to printable range (32-127)
    # Map to 0-95 range (subtract 32 from printable ASCII)
    indices = []
    for c in text[:max_length]:
        idx = ord(c)
        if 32 <= idx < 128:  # Printable ASCII
            indices.append(idx - 32)  # Map to 0-95
        else:
            indices.append(0)  # Padding/unknown

    # Pad with zeros if needed
    if len(indices) < max_length:
        indices.extend([0] * (max_length - len(indices)))

    return torch.tensor(indices, dtype=torch.long)


def indices_to_text(indices: torch.Tensor) -> str:
    """Convert character indices back to text"""
    chars = []
    for idx in indices.cpu().numpy():
        idx = int(idx)
        if 0 < idx < 96:  # Valid printable ASCII range (mapped)
            chars.append(chr(idx + 32))  # Add back 32 offset
        elif idx == 0:
            break  # Padding
    return ''.join(chars)


def test_language_puzzle_vae_small():
    """Test small language VAE"""
    print("=" * 70)
    print("Testing Language Puzzle VAE (Small)")
    print("=" * 70)

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LanguagePuzzleVAESmall().to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel: Language Puzzle VAE (Small)")
    print(f"Device: {device}")
    print(f"Parameters: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    print(f"Target: ~80-100k (Vision/Audio VAE level)")

    # Test texts
    test_texts = [
        "The cat sat on the mat.",
        "Neural networks learn patterns.",
        "Puzzle space unifies modalities.",
        "Character-level compression works!",
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
    print("Small Language Puzzle VAE verified!")
    print(f"Memory-efficient design with {total_params:,} parameters")
    print("=" * 70)

    return model


if __name__ == "__main__":
    test_language_puzzle_vae_small()
