
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyVAEIRP(nn.Module):
    def __init__(self, input_channels=1, latent_dim=16, img_size=64):
        super(TinyVAEIRP, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, 2, 1),  # (B, 16, 32, 32)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),              # (B, 32, 16, 16)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),              # (B, 64, 8, 8)
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(64 * (img_size // 8)**2, latent_dim)
        self.fc_logvar = nn.Linear(64 * (img_size // 8)**2, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 8, 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),  # (B, 32, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1),  # (B, 16, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_channels, 3, 2, 1, output_padding=1),  # (B, C, 64, 64)
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_latents(self, x):
        mu, _ = self.encode(x)
        return mu
