#!/usr/bin/env python3
"""
Test the trained TinyVAE model
"""

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.vision.tiny_vae_32 import TinyVAE32

def load_model(checkpoint_path):
    """Load the trained model"""
    model = TinyVAE32(latent_dim=64, base_channels=16)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['student_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model

def test_reconstruction():
    """Test reconstruction quality"""
    # Load model
    model_path = Path('checkpoints/tinyvae_distill/tinyvae_best.pth')
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    
    model = load_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load test data
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Get a batch of images
    batch_size = 8
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    images, labels = next(iter(test_loader))
    images = images.to(device)
    
    # Generate reconstructions
    with torch.no_grad():
        recon, mu, log_var = model(images)
        
    # Calculate reconstruction error
    mse = F.mse_loss(recon, images).item()
    print(f"\nReconstruction MSE: {mse:.4f}")
    
    # Show results
    images_np = images.cpu().numpy()
    recon_np = recon.cpu().numpy()
    
    fig, axes = plt.subplots(2, batch_size, figsize=(12, 4))
    for i in range(batch_size):
        # Original
        axes[0, i].imshow(np.transpose(images_np[i], (1, 2, 0)))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original')
        
        # Reconstruction
        axes[1, i].imshow(np.transpose(recon_np[i], (1, 2, 0)))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed')
    
    plt.suptitle(f'TinyVAE32 Reconstructions (MSE: {mse:.4f})')
    plt.tight_layout()
    plt.savefig('reconstruction_test.png')
    print("Saved visualization to reconstruction_test.png")
    
    # Test latent space interpolation
    print("\nTesting latent space interpolation...")
    with torch.no_grad():
        # Get two random images (use the original tensors, not numpy)
        img1 = images[0:1]
        img2 = images[1:2]
        
        # Encode them
        mu1, _ = model.encode(img1)
        mu2, _ = model.encode(img2)
        
        # Interpolate in latent space
        alphas = torch.linspace(0, 1, 8)
        interpolated = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            recon_interp = model.decode(z_interp)
            interpolated.append(recon_interp.cpu())
        
        # Show interpolation
        fig, axes = plt.subplots(1, 8, figsize=(12, 2))
        for i, img in enumerate(interpolated):
            axes[i].imshow(np.transpose(img[0].numpy(), (1, 2, 0)))
            axes[i].axis('off')
            axes[i].set_title(f'α={alphas[i]:.2f}')
        
        plt.suptitle('Latent Space Interpolation')
        plt.tight_layout()
        plt.savefig('interpolation_test.png')
        print("Saved interpolation to interpolation_test.png")
    
    # Model size comparison
    print("\n=== Model Size Comparison ===")
    teacher_path = Path('checkpoints/tinyvae_distill/teacher_vae.pth')
    student_path = Path('checkpoints/tinyvae_distill/tinyvae_best.pth')
    
    teacher_size = teacher_path.stat().st_size / (1024 * 1024)
    student_size = student_path.stat().st_size / (1024 * 1024)
    
    print(f"Teacher model size: {teacher_size:.2f} MB")
    print(f"Student model size: {student_size:.2f} MB")
    print(f"Size reduction: {teacher_size/student_size:.1f}x")
    
    # Count parameters
    print(f"\nStudent model parameters: {model.get_num_params():,}")

if __name__ == '__main__':
    test_reconstruction()