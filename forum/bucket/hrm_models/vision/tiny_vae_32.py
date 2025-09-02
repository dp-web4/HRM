"""
TinyVAE for 32x32 images (CIFAR-10 size)
Optimized for knowledge distillation training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TinyVAE32(nn.Module):
    """
    Tiny VAE optimized for 32x32 images
    Much smaller than standard VAE but maintains quality through distillation
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 64,
        base_channels: int = 16
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 32 → 16 → 8 → 4 → 2
        self.enc1 = nn.Conv2d(input_channels, base_channels, 3, 2, 1)      # 32 → 16
        self.enc2 = nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1)   # 16 → 8
        self.enc3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1) # 8 → 4
        self.enc4 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, 2, 1) # 4 → 2
        
        # Latent projection
        self.fc_mu = nn.Linear(base_channels * 8 * 2 * 2, latent_dim)
        self.fc_var = nn.Linear(base_channels * 8 * 2 * 2, latent_dim)
        
        # Decoder projection
        self.fc_decode = nn.Linear(latent_dim, base_channels * 8 * 2 * 2)
        
        # Decoder: 2 → 4 → 8 → 16 → 32
        self.dec1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 3, 2, 1, 1)  # 2 → 4
        self.dec2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1)  # 4 → 8
        self.dec3 = nn.ConvTranspose2d(base_channels * 2, base_channels, 3, 2, 1, 1)      # 8 → 16
        self.dec4 = nn.ConvTranspose2d(base_channels, input_channels, 3, 2, 1, 1)         # 16 → 32
        
        # Batch norms for stability
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        self.bn4 = nn.BatchNorm2d(base_channels * 8)
        
        self.dbn1 = nn.BatchNorm2d(base_channels * 4)
        self.dbn2 = nn.BatchNorm2d(base_channels * 2)
        self.dbn3 = nn.BatchNorm2d(base_channels)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image to latent distribution"""
        # Encoder forward
        h = F.relu(self.bn1(self.enc1(x)))
        h = F.relu(self.bn2(self.enc2(h)))
        h = F.relu(self.bn3(self.enc3(h)))
        h = F.relu(self.bn4(self.enc4(h)))
        
        # Flatten and project to latent
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image"""
        # Project and reshape
        h = self.fc_decode(z)
        h = h.view(-1, self.enc4.out_channels, 2, 2)
        
        # Decoder forward
        h = F.relu(self.dbn1(self.dec1(h)))
        h = F.relu(self.dbn2(self.dec2(h)))
        h = F.relu(self.dbn3(self.dec3(h)))
        h = torch.sigmoid(self.dec4(h))
        
        return h
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var
    
    def get_num_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UltraTinyVAE32(nn.Module):
    """
    Ultra-tiny VAE using depthwise separable convolutions
    Even smaller and faster than TinyVAE32
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 32,
        base_channels: int = 8
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder with depthwise separable convolutions
        self.enc1 = self._make_separable_conv(input_channels, base_channels, stride=2)
        self.enc2 = self._make_separable_conv(base_channels, base_channels * 2, stride=2)
        self.enc3 = self._make_separable_conv(base_channels * 2, base_channels * 4, stride=2)
        self.enc4 = self._make_separable_conv(base_channels * 4, base_channels * 8, stride=2)
        
        # Latent
        self.fc_mu = nn.Linear(base_channels * 8 * 2 * 2, latent_dim)
        self.fc_var = nn.Linear(base_channels * 8 * 2 * 2, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, base_channels * 8 * 2 * 2)
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 3, 2, 1, 1)
        self.dec2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, 2, 1, 1)
        self.dec3 = nn.ConvTranspose2d(base_channels * 2, base_channels, 3, 2, 1, 1)
        self.dec4 = nn.ConvTranspose2d(base_channels, input_channels, 3, 2, 1, 1)
        
    def _make_separable_conv(self, in_channels: int, out_channels: int, stride: int = 1):
        """Create a depthwise separable convolution block"""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, 64, 2, 2)  # 8 * 8 = 64 channels
        h = F.relu(self.dec1(h))
        h = F.relu(self.dec2(h))
        h = F.relu(self.dec3(h))
        return torch.sigmoid(self.dec4(h))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test both models
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    # Test TinyVAE32
    print("\nTinyVAE32:")
    tiny_vae = TinyVAE32(latent_dim=64, base_channels=16).to(device)
    print(f"  Parameters: {tiny_vae.get_num_params():,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        recon, mu, log_var = tiny_vae(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {recon.shape}")
    print(f"  Latent shape: {mu.shape}")
    
    # Benchmark
    if device.type == 'cuda':
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = tiny_vae(x)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = tiny_vae(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  Time per batch (4 images): {elapsed/100*1000:.2f}ms")
    
    # Test UltraTinyVAE32
    print("\nUltraTinyVAE32:")
    ultra_vae = UltraTinyVAE32(latent_dim=32, base_channels=8).to(device)
    print(f"  Parameters: {ultra_vae.get_num_params():,}")
    
    with torch.no_grad():
        recon, mu, log_var = ultra_vae(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {recon.shape}")
    print(f"  Latent shape: {mu.shape}")
    
    if device.type == 'cuda':
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = ultra_vae(x)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = ultra_vae(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  Time per batch (4 images): {elapsed/100*1000:.2f}ms")