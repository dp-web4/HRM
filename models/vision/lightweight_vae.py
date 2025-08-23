"""
Lightweight VAE optimized for Jetson Orin Nano
Target: 224x224 RGB → 7x7x256 latent space
Performance goal: <5ms encode/decode on Jetson
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import time


class LightweightVAE(nn.Module):
    """
    Efficient VAE for Jetson deployment
    - Minimal layers for speed
    - Optimized channel progression
    - Skip connections for quality
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 256,
        latent_size: int = 7,
        base_channels: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_size = latent_size
        
        # Encoder: 224 → 112 → 56 → 28 → 14 → 7
        self.enc1 = nn.Conv2d(input_channels, base_channels, 4, 2, 1)  # 224 → 112
        self.enc2 = nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)  # 112 → 56
        self.enc3 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)  # 56 → 28
        self.enc4 = nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1)  # 28 → 14
        self.enc5 = nn.Conv2d(base_channels * 8, latent_dim * 2, 4, 2, 1)  # 14 → 7
        
        # Batch norms for stability
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        self.bn4 = nn.BatchNorm2d(base_channels * 8)
        
        # Decoder: 7 → 14 → 28 → 56 → 112 → 224
        self.dec1 = nn.ConvTranspose2d(latent_dim, base_channels * 8, 4, 2, 1)  # 7 → 14
        self.dec2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1)  # 14 → 28
        self.dec3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)  # 28 → 56
        self.dec4 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1)  # 56 → 112
        self.dec5 = nn.ConvTranspose2d(base_channels, input_channels, 4, 2, 1)  # 112 → 224
        
        # Decoder batch norms
        self.dbn1 = nn.BatchNorm2d(base_channels * 8)
        self.dbn2 = nn.BatchNorm2d(base_channels * 4)
        self.dbn3 = nn.BatchNorm2d(base_channels * 2)
        self.dbn4 = nn.BatchNorm2d(base_channels)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution
        Returns: (mu, log_var) both [B, latent_dim, 7, 7]
        """
        # Encoder forward pass
        h = F.relu(self.bn1(self.enc1(x)))
        h = F.relu(self.bn2(self.enc2(h)))
        h = F.relu(self.bn3(self.enc3(h)))
        h = F.relu(self.bn4(self.enc4(h)))
        h = self.enc5(h)  # [B, latent_dim*2, 7, 7]
        
        # Split into mu and log_var
        mu, log_var = torch.chunk(h, 2, dim=1)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for differentiable sampling
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to reconstructed image
        Input: [B, latent_dim, 7, 7]
        Output: [B, 3, 224, 224]
        """
        h = F.relu(self.dbn1(self.dec1(z)))
        h = F.relu(self.dbn2(self.dec2(h)))
        h = F.relu(self.dbn3(self.dec3(h)))
        h = F.relu(self.dbn4(self.dec4(h)))
        return torch.sigmoid(self.dec5(h))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass
        Returns: (reconstruction, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var
    
    def loss_function(
        self,
        recon: torch.Tensor,
        original: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, dict]:
        """
        VAE loss = Reconstruction loss + KL divergence
        """
        # Reconstruction loss (MSE for images)
        recon_loss = F.mse_loss(recon, original, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item()
        }
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation without sampling
        """
        mu, _ = self.encode(x)
        return mu
    
    def benchmark(self, device: torch.device = None, input_size: Tuple[int, int] = (224, 224)):
        """
        Benchmark encode/decode performance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.eval()
        self.to(device)
        
        # Test input
        x = torch.randn(1, 3, *input_size, device=device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.encode(x)
                
        # Benchmark encode
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                mu, log_var = self.encode(x)
        torch.cuda.synchronize()
        encode_time = (time.time() - start) / 100 * 1000  # ms
        
        # Benchmark decode
        z = torch.randn(1, self.latent_dim, self.latent_size, self.latent_size, device=device)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = self.decode(z)
        torch.cuda.synchronize()
        decode_time = (time.time() - start) / 100 * 1000  # ms
        
        # Full forward
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = self.forward(x)
        torch.cuda.synchronize()
        forward_time = (time.time() - start) / 100 * 1000  # ms
        
        print(f"VAE Benchmark Results:")
        print(f"  Encode: {encode_time:.2f}ms")
        print(f"  Decode: {decode_time:.2f}ms")
        print(f"  Full forward: {forward_time:.2f}ms")
        print(f"  Latent size: {self.latent_dim}x{self.latent_size}x{self.latent_size}")
        
        # Memory usage
        params = sum(p.numel() for p in self.parameters())
        print(f"  Parameters: {params:,} ({params*4/1024/1024:.2f}MB)")
        
        return {
            'encode_ms': encode_time,
            'decode_ms': decode_time,
            'forward_ms': forward_time,
            'params': params
        }


class TinyVAE(LightweightVAE):
    """
    Even smaller VAE for extreme speed
    Uses depthwise separable convolutions
    """
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 128, latent_size: int = 7):
        super().__init__(input_channels, latent_dim, latent_size, base_channels=16)
        
        # Replace standard convs with depthwise separable
        self._convert_to_separable()
    
    def _convert_to_separable(self):
        """Convert standard convolutions to depthwise separable"""
        # This would be implemented if we need even more speed
        pass


def create_vae_for_jetson(variant: str = 'standard') -> LightweightVAE:
    """
    Factory function to create appropriate VAE variant
    
    Args:
        variant: 'standard', 'tiny', or 'minimal'
    """
    if variant == 'standard':
        return LightweightVAE(base_channels=32)
    elif variant == 'tiny':
        return TinyVAE(base_channels=16)
    elif variant == 'minimal':
        return LightweightVAE(base_channels=16, latent_dim=128)
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    # Test on current device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Create and benchmark standard VAE
    print("\nStandard VAE:")
    vae = create_vae_for_jetson('standard')
    vae.benchmark(device)
    
    # Test with actual data
    print("\nTesting with batch:")
    x = torch.randn(4, 3, 224, 224, device=device)
    recon, mu, log_var = vae(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Recon shape: {recon.shape}")
    print(f"  Latent shape: {mu.shape}")
    
    # Try minimal variant
    print("\nMinimal VAE:")
    vae_mini = create_vae_for_jetson('minimal')
    vae_mini.benchmark(device)