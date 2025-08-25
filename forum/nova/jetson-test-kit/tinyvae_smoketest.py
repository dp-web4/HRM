#!/usr/bin/env python3
import torch, numpy as np
from tinyvae_irp_plugin import create_tinyvae_irp

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = create_tinyvae_irp(device=device, input_channels=3, latent_dim=64, use_fp16=True)
    # Fake crop: 1×3×64×64
    x = torch.rand(1, 3, 64, 64, device=vae.device)
    z, telem = vae.refine(x)
    print("OK tinyvae refine:", z.shape, telem)

if __name__ == "__main__":
    main()
