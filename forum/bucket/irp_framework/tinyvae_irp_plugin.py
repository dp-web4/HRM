
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Any, Dict, Tuple
from sage.irp.base import IRPPlugin


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Use GroupNorm instead of BatchNorm for small-batch stability
        self.norm = nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.relu(x)


class TinyVAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=64, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            DepthwiseSeparableConv2d(input_channels, 16, 3, 2, 1),  # 32x32
            DepthwiseSeparableConv2d(16, 32, 3, 2, 1),              # 16x16
            DepthwiseSeparableConv2d(32, 64, 3, 2, 1)               # 8x8
        )
        # Adaptive pooling makes encoder robust to input size changes
        self.adapt = nn.AdaptiveAvgPool2d((8, 8))
        self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 8, 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_channels, 3, 2, 1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.adapt(h)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class TinyVAEIRP(IRPPlugin):
    def __init__(self, config=None):
        super().__init__(config or {})
        self.config = config or {}
        self.latent_dim = self.config.get("latent_dim", 64)
        self.input_channels = self.config.get("input_channels", 3)
        self.device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.beta_kl = float(self.config.get("beta_kl", 0.1))
        self.use_fp16 = bool(self.config.get("use_fp16", False))
        self.normalize = bool(self.config.get("normalize", False))
        self.mean = self.config.get("mean", [0.5] * self.input_channels)
        self.std = self.config.get("std", [0.5] * self.input_channels)

        self.model = TinyVAE(self.input_channels, self.latent_dim).to(self.device)
        self.model.eval()
        self.entity_id = "tinyvae_irp"

    def refine(self, x: Any, early_stop: bool = True) -> Tuple[Any, Dict[str, Any]]:
        # Normalize if configured
        if self.normalize and isinstance(x, torch.Tensor):
            for c in range(x.shape[1]):
                x[:, c] = (x[:, c] - self.mean[c]) / self.std[c]

        x_tensor = x.to(self.device)

        # Timing setup
        use_cuda_timers = (self.device.type == 'cuda')
        if use_cuda_timers:
            start_time = torch.cuda.Event(enable_timing=True); end_time = torch.cuda.Event(enable_timing=True); start_time.record()
        else:
            t0 = time.perf_counter()

        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_fp16):
            reconstruction, mu, logvar = self.model(x_tensor)
            latent = mu
            recon_error = F.mse_loss(reconstruction, x_tensor).item()
            kl_div = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item() / (x_tensor.shape[0] * self.latent_dim)

        # Trust metric calibration
        trust_recon = max(0.0, 1.0 - recon_error * 10.0)
        trust_kl = 1.0 / (1.0 + kl_div)
        trust = max(0.0, min(1.0, 0.5 * trust_recon + 0.5 * trust_kl))

        # Timing end
        if use_cuda_timers:
            end_time.record(); torch.cuda.synchronize(); time_ms = start_time.elapsed_time(end_time)
        else:
            time_ms = (time.perf_counter() - t0) * 1000.0

        telemetry = {
            'iterations': 1,
            'compute_saved': 0.0,
            'energy_trajectory': [recon_error],
            'trust': trust,
            'converged': True,
            'energy_delta': 0.0,
            'reconstruction_error': recon_error,
            'kl_divergence': kl_div,
            'latent_dim': self.latent_dim,
            'latent_norm': torch.norm(latent).item(),
            'time_ms': time_ms,
            'beta_kl': self.beta_kl
        }

        return latent, telemetry
    
    def get_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent encoding for input (convenience method)"""
        with torch.no_grad():
            mu, _ = self.model.encode(x.to(self.device))
        return mu
    
    def get_reconstruction(self) -> torch.Tensor:
        """Get the most recent reconstruction (if cached)"""
        # Note: Would need to cache in refine() method
        return None  # Not cached in this version
    
    def encode_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images to latents"""
        with torch.no_grad():
            mu, _ = self.model.encode(batch.to(self.device))
        return mu
    
    def decode_batch(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latents to images"""
        with torch.no_grad():
            reconstructions = self.model.decode(latents.to(self.device))
        return reconstructions


def create_tinyvae_irp(device="cuda", **kwargs):
    return TinyVAEIRP({"device": device, **kwargs})
