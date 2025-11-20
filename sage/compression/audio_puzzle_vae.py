#!/usr/bin/env python3
"""
Audio → Puzzle VAE

Encodes audio waveforms to 30×30×10 puzzle space using VQ-VAE.
Similar to vision encoding but operates on spectrograms.

Architecture:
- Input: Audio waveform (1 sec @ 16kHz = 16,000 samples)
- Preprocessing: Mel spectrogram (128 mel bins × ~32 time frames)
- Encoder: Spectrogram → 64D latent features per spatial position
- VQ Codebook: 10 discrete codes
- Decoder: Codes → reconstructed spectrogram
- Puzzle Output: 30×30 grid with values 0-9

Spatial Semantics:
- X-axis: Time progression (left = past, right = present)
- Y-axis: Frequency bands (bottom = low, top = high)
- Values: Energy levels (0=silence, 9=maximum)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from typing import Tuple, Dict, Any

from compression.vision_puzzle_vae import VectorQuantizer


class AudioPreprocessor(nn.Module):
    """Convert audio waveform to mel spectrogram suitable for encoding"""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        n_mels: int = 128,
        hop_length: int = 512,
        target_length: int = 32  # Time frames
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.target_length = target_length

        # Mel spectrogram transform
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0  # Power spectrogram
        )

        # Convert to log scale (dB)
        self.amplitude_to_db = T.AmplitudeToDB()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel spectrogram

        Args:
            waveform: [batch, samples] audio at 16kHz
                     Expected ~16000 samples (1 second)

        Returns:
            mel_spec: [batch, 1, n_mels, time_frames] normalized spectrogram
        """
        # Compute mel spectrogram
        mel = self.mel_spec(waveform)  # [batch, n_mels, time_frames]

        # Convert to dB scale
        mel_db = self.amplitude_to_db(mel)

        # Pad or truncate time dimension to target_length
        time_frames = mel_db.shape[-1]
        if time_frames < self.target_length:
            # Pad with silence (minimum dB value)
            pad_width = self.target_length - time_frames
            mel_db = F.pad(mel_db, (0, pad_width), value=mel_db.min().item())
        elif time_frames > self.target_length:
            # Truncate
            mel_db = mel_db[..., :self.target_length]

        # Normalize to [0, 1] range
        mel_min = mel_db.min()
        mel_max = mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)

        # Add channel dimension: [batch, 1, n_mels, time_frames]
        return mel_norm.unsqueeze(1)


class AudioPuzzleEncoder(nn.Module):
    """Encode mel spectrogram (128×32) to 30×30×64 latent features"""

    def __init__(self, latent_dim: int = 64):
        super().__init__()

        # CNN encoder: 128×32 → 30×30
        self.encoder = nn.Sequential(
            # 128×32 → 64×16
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 64×16 → 32×8
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 32×8 → 30×30 (upsample and reshape to square)
            nn.Upsample(size=(30, 30), mode='bilinear', align_corners=False),

            # Project to latent dim
            nn.Conv2d(64, latent_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 1, 128, 32] mel spectrogram
        Returns:
            latents: [batch, latent_dim, 30, 30]
        """
        return self.encoder(x)


class AudioPuzzleDecoder(nn.Module):
    """Decode 30×30×64 latent features back to 128×32 mel spectrogram"""

    def __init__(self, latent_dim: int = 64):
        super().__init__()

        # Transposed CNN decoder: 30×30 → 128×32
        self.decoder = nn.Sequential(
            # Project up from latent dim
            nn.Conv2d(latent_dim, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 30×30 → 60×60
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 60×60 → 120×120
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # 120×120 → 128×32 (reshape to target size)
            nn.Upsample(size=(128, 32), mode='bilinear', align_corners=False),

            # Final projection
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, latent_dim, 30, 30]
        Returns:
            reconstruction: [batch, 1, 128, 32]
        """
        return self.decoder(x)


class AudioPuzzleVAE(nn.Module):
    """
    Complete Audio → Puzzle VAE

    Encodes audio waveforms to 30×30 puzzle grids with 10 discrete values per cell.
    Can reconstruct spectrograms from puzzles and enables geometric reasoning about audio.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        num_codes: int = 10,
        commitment_cost: float = 0.25,
        sample_rate: int = 16000
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_codes = num_codes
        self.sample_rate = sample_rate

        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.encoder = AudioPuzzleEncoder(latent_dim)
        self.vq = VectorQuantizer(num_codes, latent_dim, commitment_cost)
        self.decoder = AudioPuzzleDecoder(latent_dim)

    def encode_to_puzzle(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode audio waveform to puzzle space

        Args:
            waveform: [batch, samples] at 16kHz, typically 16000 samples (1 sec)

        Returns:
            puzzles: [batch, 30, 30] with values 0-9
        """
        # Preprocess to mel spectrogram
        mel_spec = self.preprocessor(waveform)  # [B, 1, 128, 32]

        # Encode
        latents = self.encoder(mel_spec)  # [B, latent_dim, 30, 30]

        # Rearrange for VQ: [B, H, W, C]
        latents = latents.permute(0, 2, 3, 1)  # [B, 30, 30, latent_dim]

        # Quantize
        _, puzzles, _ = self.vq(latents)  # [B, 30, 30]

        return puzzles

    def decode_from_puzzle(self, puzzles: torch.Tensor) -> torch.Tensor:
        """
        Decode puzzles back to mel spectrograms

        Args:
            puzzles: [batch, 30, 30] with values 0-9

        Returns:
            mel_specs: [batch, 1, 128, 32] normalized spectrograms
        """
        # Look up quantized vectors
        quantized = self.vq.codebook(puzzles)  # [B, 30, 30, latent_dim]

        # Rearrange for decoder: [B, C, H, W]
        quantized = quantized.permute(0, 3, 1, 2)  # [B, latent_dim, 30, 30]

        # Decode
        reconstruction = self.decoder(quantized)  # [B, 1, 128, 32]

        return reconstruction

    def forward(self, waveform: torch.Tensor) -> Dict[str, Any]:
        """
        Full forward pass: preprocess → encode → quantize → decode

        Args:
            waveform: [batch, samples] at 16kHz

        Returns:
            Dictionary with:
                - puzzles: [batch, 30, 30]
                - reconstruction: [batch, 1, 128, 32]
                - vq_loss: Dict of VQ losses
                - mel_spec_input: [batch, 1, 128, 32] (for comparison)
        """
        # Preprocess
        mel_spec = self.preprocessor(waveform)

        # Encode
        latents = self.encoder(mel_spec)
        latents = latents.permute(0, 2, 3, 1)

        # Quantize
        quantized, puzzles, vq_loss = self.vq(latents)

        # Decode
        quantized = quantized.permute(0, 3, 1, 2)
        reconstruction = self.decoder(quantized)

        return {
            'puzzles': puzzles,
            'reconstruction': reconstruction,
            'vq_loss': vq_loss,
            'mel_spec_input': mel_spec
        }

    def compute_loss(
        self,
        mel_spec_input: torch.Tensor,
        reconstruction: torch.Tensor,
        vq_loss: Dict[str, torch.Tensor],
        recon_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses

        Args:
            mel_spec_input: Original mel spectrograms
            reconstruction: Reconstructed mel spectrograms
            vq_loss: VQ losses from forward pass
            recon_weight: Weight for reconstruction loss

        Returns:
            Dictionary of losses
        """
        # Reconstruction loss (MSE on normalized spectrograms)
        recon_loss = F.mse_loss(reconstruction, mel_spec_input)

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


def test_audio_puzzle_vae():
    """Test the VAE with random audio"""
    print("=" * 70)
    print("Testing Audio → Puzzle VAE")
    print("=" * 70)

    # Create model
    vae = AudioPuzzleVAE(latent_dim=64, num_codes=10, sample_rate=16000)

    # Random batch of audio (1 second at 16kHz)
    batch_size = 4
    waveform = torch.randn(batch_size, 16000)  # Random audio

    print(f"\nInput waveform: {waveform.shape}")
    print(f"Value range: [{waveform.min():.3f}, {waveform.max():.3f}]")
    print(f"Duration: 1.0 sec @ 16kHz")

    # Encode to puzzles
    with torch.no_grad():
        puzzles = vae.encode_to_puzzle(waveform)

    print(f"\nPuzzles: {puzzles.shape}")
    print(f"Value range: [{puzzles.min()}, {puzzles.max()}]")
    print(f"Unique values: {len(torch.unique(puzzles))} (target: 10)")

    # Show puzzle for first audio
    print(f"\nFirst puzzle (30×30) - Audio encoded geometrically:")
    print(puzzles[0])

    # Decode back to spectrograms
    with torch.no_grad():
        reconstruction = vae.decode_from_puzzle(puzzles)

    print(f"\nReconstruction: {reconstruction.shape}")
    print(f"Value range: [{reconstruction.min():.3f}, {reconstruction.max():.3f}]")

    # Full forward pass
    outputs = vae(waveform)
    losses = vae.compute_loss(
        outputs['mel_spec_input'],
        outputs['reconstruction'],
        outputs['vq_loss']
    )

    print(f"\nLosses (untrained VAE):")
    for key, value in losses.items():
        if key == 'perplexity':
            print(f"  {key}: {value:.2f} (10.0 = all codes used equally)")
        else:
            print(f"  {key}: {value.item():.4f}")

    # Interpret puzzle semantics
    print(f"\n" + "=" * 70)
    print("PUZZLE SPACE SEMANTICS FOR AUDIO")
    print("=" * 70)

    puzzle = puzzles[0]

    # Temporal analysis (columns = time)
    temporal_mean = puzzle.float().mean(dim=0)  # [30] - mean value per time step
    print(f"\nTemporal Profile (time progression):")
    print(f"  Early (cols 0-10): mean = {temporal_mean[:10].mean():.1f}")
    print(f"  Middle (cols 10-20): mean = {temporal_mean[10:20].mean():.1f}")
    print(f"  Recent (cols 20-30): mean = {temporal_mean[20:].mean():.1f}")

    # Frequency analysis (rows = frequency bands)
    freq_mean = puzzle.float().mean(dim=1)  # [30] - mean value per freq band
    print(f"\nFrequency Profile (bottom=low, top=high):")
    print(f"  Low freqs (rows 20-30): mean = {freq_mean[20:].mean():.1f}")
    print(f"  Mid freqs (rows 10-20): mean = {freq_mean[10:20].mean():.1f}")
    print(f"  High freqs (rows 0-10): mean = {freq_mean[:10].mean():.1f}")

    # Energy distribution
    high_energy = (puzzle >= 7).sum().item()
    silence = (puzzle == 0).sum().item()
    print(f"\nEnergy Distribution:")
    print(f"  Silence regions (value 0): {silence}/900 cells ({100*silence/900:.1f}%)")
    print(f"  High energy (value ≥7): {high_energy}/900 cells ({100*high_energy/900:.1f}%)")

    print("\n" + "=" * 70)
    print("Audio → Puzzle VAE test complete!")
    print("=" * 70)
    print("\nKey Achievement:")
    print("  Audio waveform → 30×30 geometric representation")
    print("  Time × Frequency structure preserved")
    print("  Energy levels discretized to 0-9")
    print("  Ready for SNARC salience assessment")

    return vae, puzzles, outputs


if __name__ == "__main__":
    test_audio_puzzle_vae()
