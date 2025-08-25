#!/usr/bin/env python3
"""
TinyVAE IRP Plugin Implementation
Compact VAE for efficient latent extraction from motion-focused crops
Based on Nova's specification for Jetson deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Tuple, Optional
from dataclasses import dataclass

from sage.irp.base import IRPPlugin


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution for memory efficiency"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TinyVAEEncoder(nn.Module):
    """Compact encoder using depthwise separable convolutions"""
    
    def __init__(self, input_channels=1, latent_dim=16):
        super().__init__()
        
        # Progressive downsampling: 64x64 -> 32x32 -> 16x16 -> 8x8
        self.conv1 = DepthwiseSeparableConv2d(input_channels, 16, stride=2)  # 32x32
        self.conv2 = DepthwiseSeparableConv2d(16, 32, stride=2)              # 16x16
        self.conv3 = DepthwiseSeparableConv2d(32, 64, stride=2)              # 8x8
        
        # Flatten and project to latent space
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class TinyVAEDecoder(nn.Module):
    """Compact decoder using transposed convolutions"""
    
    def __init__(self, latent_dim=16, output_channels=1):
        super().__init__()
        
        # Project from latent space
        self.fc = nn.Linear(latent_dim, 64 * 8 * 8)
        self.unflatten = nn.Unflatten(1, (64, 8, 8))
        
        # Progressive upsampling: 8x8 -> 16x16 -> 32x32 -> 64x64
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 16x16
        self.bn1 = nn.BatchNorm2d(32)
        
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # 32x32
        self.bn2 = nn.BatchNorm2d(16)
        
        self.deconv3 = nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1)  # 64x64
        
    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))  # Output in [0, 1]
        
        return x


class TinyVAE(nn.Module):
    """Complete TinyVAE model"""
    
    def __init__(self, input_channels=1, latent_dim=16):
        super().__init__()
        self.encoder = TinyVAEEncoder(input_channels, latent_dim)
        self.decoder = TinyVAEDecoder(latent_dim, input_channels)
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar
        
    def encode(self, x):
        """Encode to latent space (returns mean)"""
        mu, _ = self.encoder(x)
        return mu
        
    def decode(self, z):
        """Decode from latent space"""
        return self.decoder(z)


class TinyVAEIRP(IRPPlugin):
    """
    TinyVAE as an IRP plugin for efficient latent extraction
    Designed for 64x64 crops from motion-focused regions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Configuration
        self.config = config or {}
        self.input_channels = self.config.get('input_channels', 1)
        self.latent_dim = self.config.get('latent_dim', 16)
        self.img_size = self.config.get('img_size', 64)
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Set entity ID
        self.entity_id = f"tinyvae_irp_{self.latent_dim}d"
        
        # Initialize model
        self.model = TinyVAE(self.input_channels, self.latent_dim).to(self.device)
        self.model.eval()  # Default to eval mode
        
        # Cache for latest results
        self.latest_latent = None
        self.latest_reconstruction = None
        
    def refine(self, x: Any, early_stop: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Refine input through VAE encoding/decoding
        
        Args:
            x: Input crop (numpy array or torch tensor) of shape [H, W] or [C, H, W] or [B, C, H, W]
            early_stop: Not used for VAE (single forward pass)
            
        Returns:
            Tuple of:
                - latent: Encoded latent vector
                - telemetry: Performance metrics
        """
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        
        if start_time:
            start_time.record()
        
        # Convert input to tensor
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float() / 255.0  # Normalize to [0, 1]
        else:
            x_tensor = x.float()
            if x_tensor.max() > 1.0:
                x_tensor = x_tensor / 255.0
        
        # Move to device
        x_tensor = x_tensor.to(self.device)
        
        # Ensure correct shape [B, C, H, W]
        if x_tensor.dim() == 2:  # [H, W]
            x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif x_tensor.dim() == 3:  # [C, H, W] or [H, W, C]
            if x_tensor.shape[-1] in [1, 3]:  # [H, W, C]
                x_tensor = x_tensor.permute(2, 0, 1)  # -> [C, H, W]
            x_tensor = x_tensor.unsqueeze(0)  # Add batch dim
        
        # Resize if needed
        if x_tensor.shape[-2:] != (self.img_size, self.img_size):
            x_tensor = F.interpolate(x_tensor, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # Ensure correct number of channels
        if x_tensor.shape[1] != self.input_channels:
            if self.input_channels == 1 and x_tensor.shape[1] == 3:
                # Convert RGB to grayscale
                x_tensor = torch.mean(x_tensor, dim=1, keepdim=True)
            elif self.input_channels == 3 and x_tensor.shape[1] == 1:
                # Convert grayscale to RGB
                x_tensor = x_tensor.repeat(1, 3, 1, 1)
        
        # Forward pass
        with torch.no_grad():
            reconstruction, mu, logvar = self.model(x_tensor)
            latent = mu  # Use mean as latent representation
            
            # Compute reconstruction error
            recon_error = F.mse_loss(reconstruction, x_tensor).item()
            
            # KL divergence
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_div = kl_div.item() / (x_tensor.shape[0] * self.latent_dim)
        
        # Cache results
        self.latest_latent = latent.cpu()
        self.latest_reconstruction = reconstruction.cpu()
        
        # Compute timing
        time_ms = 0.0
        if start_time:
            end_time.record()
            torch.cuda.synchronize()
            time_ms = start_time.elapsed_time(end_time)
        
        # Build telemetry
        telemetry = {
            'iterations': 1,  # VAE is single pass
            'compute_saved': 0.0,  # No iterative refinement
            'energy_trajectory': [recon_error],
            'trust': 1.0 - recon_error,  # Trust inversely proportional to error
            'converged': True,  # Always "converges" in one pass
            'energy_delta': 0.0,
            'reconstruction_error': recon_error,
            'kl_divergence': kl_div,
            'latent_dim': self.latent_dim,
            'latent_norm': torch.norm(latent).item(),
            'time_ms': time_ms
        }
        
        return latent, telemetry
    
    def get_latents(self) -> Optional[torch.Tensor]:
        """Get the most recent latent encoding"""
        return self.latest_latent
    
    def get_reconstruction(self) -> Optional[torch.Tensor]:
        """Get the most recent reconstruction"""
        return self.latest_reconstruction
    
    def reconstruct(self, x: Any) -> torch.Tensor:
        """
        Encode and decode input (full autoencoder pass)
        
        Args:
            x: Input image/crop
            
        Returns:
            Reconstructed image
        """
        latent, _ = self.refine(x)
        return self.latest_reconstruction
    
    def reconstruction_loss(self, x: Any) -> float:
        """
        Compute reconstruction loss for input
        
        Args:
            x: Input image/crop
            
        Returns:
            MSE reconstruction loss
        """
        _, telemetry = self.refine(x)
        return telemetry['reconstruction_error']
    
    def encode_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images
        
        Args:
            batch: Tensor of shape [B, C, H, W]
            
        Returns:
            Latent codes of shape [B, latent_dim]
        """
        with torch.no_grad():
            mu, _ = self.model.encoder(batch.to(self.device))
        return mu
    
    def decode_batch(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode a batch of latent codes
        
        Args:
            latents: Tensor of shape [B, latent_dim]
            
        Returns:
            Reconstructed images of shape [B, C, H, W]
        """
        with torch.no_grad():
            reconstructions = self.model.decoder(latents.to(self.device))
        return reconstructions


def create_tinyvae_irp(device: Optional[torch.device] = None, 
                       latent_dim: int = 16,
                       input_channels: int = 1) -> TinyVAEIRP:
    """
    Factory function to create TinyVAE IRP plugin
    
    Args:
        device: Torch device (defaults to CUDA if available)
        latent_dim: Dimension of latent space (default 16)
        input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        
    Returns:
        Configured TinyVAEIRP instance
    """
    config = {
        'device': device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        'latent_dim': latent_dim,
        'input_channels': input_channels,
        'img_size': 64
    }
    
    return TinyVAEIRP(config)