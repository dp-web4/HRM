#!/usr/bin/env python3
"""
TinyVAE Distillation Training
Distill knowledge from a standard PyTorch VAE to our lightweight TinyVAE

Author: Claude & Dennis
Date: 2025-08-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import vgg16
import numpy as np
from pathlib import Path
import time
import argparse
import json
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.vision.tiny_vae_32 import TinyVAE32, UltraTinyVAE32


@dataclass
class DistillationConfig:
    """Configuration for distillation training"""
    # Data
    dataset: str = 'cifar10'  # 'cifar10', 'imagenet-subset', 'custom'
    data_path: str = './data'
    batch_size: int = 64
    num_workers: int = 4
    
    # Model
    teacher_latent_dim: int = 512
    student_latent_dim: int = 64  # Smaller for TinyVAE32
    student_latent_size: int = 2   # Final spatial size after encoding
    student_base_channels: int = 16
    
    # Training
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Loss weights (these sum should be ~1.0)
    recon_weight: float = 0.3      # Reconstruction loss
    kl_weight: float = 0.1          # KL divergence
    distill_latent_weight: float = 0.3  # Latent distillation
    distill_recon_weight: float = 0.2   # Output distillation  
    perceptual_weight: float = 0.1      # Perceptual loss
    
    # Temperature for distillation
    temperature: float = 3.0
    
    # Checkpointing
    checkpoint_dir: str = './checkpoints/tinyvae_distill'
    log_interval: int = 10
    save_interval: int = 5
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp16: bool = False  # Use mixed precision


class StandardVAE(nn.Module):
    """
    Standard 'teacher' VAE with more capacity
    This could be replaced with a pretrained model
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        
        # More powerful encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),       # 32x32 → 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),     # 16x16 → 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),    # 8x8 → 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),    # 4x4 → 2x2
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Project to latent
        self.fc_mu = nn.Linear(512 * 2 * 2, latent_dim)
        self.fc_var = nn.Linear(512 * 2 * 2, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 512 * 2 * 2)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 2x2 → 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4x4 → 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8x8 → 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),     # 16x16 → 32x32
            nn.Sigmoid()
        )
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
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
        h = h.view(-1, 512, 2, 2)
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Resize if needed (VGG expects at least 32x32)
        if x.shape[-1] < 32:
            x = F.interpolate(x, size=(32, 32), mode='bilinear')
            y = F.interpolate(y, size=(32, 32), mode='bilinear')
        
        x_feat = self.vgg(x)
        y_feat = self.vgg(y)
        return F.mse_loss(x_feat, y_feat)


class DistillationTrainer:
    """Handles the distillation training process"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.setup_models()
        
        # Setup data
        self.setup_data()
        
        # Setup training
        self.setup_training()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        
    def setup_models(self):
        """Initialize teacher and student models"""
        # Teacher model (could load pretrained here)
        self.teacher = StandardVAE(latent_dim=self.config.teacher_latent_dim)
        self.teacher.to(self.device)
        
        # Try to load pretrained teacher if available
        teacher_checkpoint = Path(self.config.checkpoint_dir) / 'teacher_vae.pth'
        if teacher_checkpoint.exists():
            self.teacher.load_state_dict(torch.load(teacher_checkpoint))
            self.logger.info(f"Loaded pretrained teacher from {teacher_checkpoint}")
        else:
            self.logger.info("No pretrained teacher found, will train from scratch")
        
        # Student model (TinyVAE32 for 32x32 images)
        self.student = TinyVAE32(
            latent_dim=self.config.student_latent_dim,
            base_channels=16
        )
        self.student.to(self.device)
        
        # Perceptual loss
        self.perceptual_loss = PerceptualLoss(self.device)
        
    def setup_data(self):
        """Setup data loaders"""
        # Transform for 32x32 images (CIFAR-10 size)
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        
        if self.config.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.config.data_path,
                train=True,
                download=True,
                transform=transform
            )
            val_dataset = torchvision.datasets.CIFAR10(
                root=self.config.data_path,
                train=False,
                download=True,
                transform=transform
            )
        else:
            raise ValueError(f"Dataset {self.config.dataset} not supported")
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
    def setup_training(self):
        """Setup optimizer and scheduler"""
        # Only optimize student parameters
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs
        )
        
    def vae_loss(self, recon: torch.Tensor, target: torch.Tensor, 
                 mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Standard VAE loss with reconstruction and KL divergence"""
        recon_loss = F.mse_loss(recon, target, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss, kl_loss
    
    def distillation_loss(self, student_z: torch.Tensor, teacher_z: torch.Tensor,
                         student_recon: torch.Tensor, teacher_recon: torch.Tensor,
                         temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Knowledge distillation losses"""
        # Latent space distillation - match the latent representations
        # Since dimensions differ, we project teacher to student dimension
        if not hasattr(self, 'latent_projector'):
            # Create projection layer on first use
            self.latent_projector = nn.Linear(
                teacher_z.shape[1], 
                student_z.shape[1],
                bias=False
            ).to(self.device)
            # Initialize with Xavier/Glorot
            nn.init.xavier_uniform_(self.latent_projector.weight)
        
        # Project teacher latent to student dimension
        teacher_z_proj = self.latent_projector(teacher_z.detach())
        
        # Apply temperature scaling for softer targets
        latent_distill = F.mse_loss(
            student_z / temperature,
            teacher_z_proj / temperature
        ) * (temperature ** 2)
        
        # Output distillation - match reconstructions
        recon_distill = F.mse_loss(student_recon, teacher_recon.detach())
        
        return latent_distill, recon_distill
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.student.train()
        self.teacher.eval()  # Teacher in eval mode
        
        epoch_losses = {
            'total': 0, 'recon': 0, 'kl': 0, 
            'latent_distill': 0, 'recon_distill': 0,
            'perceptual': 0
        }
        
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            
            # Teacher forward pass (no grad)
            with torch.no_grad():
                teacher_recon, teacher_mu, teacher_log_var = self.teacher(data)
                teacher_z = self.teacher.reparameterize(teacher_mu, teacher_log_var)
            
            # Student forward pass
            student_recon, student_mu, student_log_var = self.student(data)
            student_z = self.student.reparameterize(student_mu, student_log_var)
            
            # Calculate losses
            recon_loss, kl_loss = self.vae_loss(
                student_recon, data, student_mu, student_log_var
            )
            
            latent_distill, recon_distill = self.distillation_loss(
                student_z, teacher_z, student_recon, teacher_recon,
                self.config.temperature
            )
            
            perceptual_loss = self.perceptual_loss(student_recon, data)
            
            # Combine losses with weights
            total_loss = (
                self.config.recon_weight * recon_loss +
                self.config.kl_weight * kl_loss +
                self.config.distill_latent_weight * latent_distill +
                self.config.distill_recon_weight * recon_distill +
                self.config.perceptual_weight * perceptual_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            
            # Track losses
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['kl'] += kl_loss.item()
            epoch_losses['latent_distill'] += latent_distill.item()
            epoch_losses['recon_distill'] += recon_distill.item()
            epoch_losses['perceptual'] += perceptual_loss.item()
            
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    f"Batch [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {total_loss.item():.4f} "
                    f"(R: {recon_loss.item():.3f}, KL: {kl_loss.item():.3f}, "
                    f"LD: {latent_distill.item():.3f}, RD: {recon_distill.item():.3f})"
                )
        
        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def validate(self) -> Dict:
        """Validate the student model"""
        self.student.eval()
        
        val_losses = {
            'total': 0, 'recon': 0, 'kl': 0, 'perceptual': 0
        }
        
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                
                recon, mu, log_var = self.student(data)
                
                recon_loss, kl_loss = self.vae_loss(recon, data, mu, log_var)
                perceptual_loss = self.perceptual_loss(recon, data)
                
                val_losses['recon'] += recon_loss.item()
                val_losses['kl'] += kl_loss.item()
                val_losses['perceptual'] += perceptual_loss.item()
        
        # Average losses
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        val_losses['total'] = (
            self.config.recon_weight * val_losses['recon'] +
            self.config.kl_weight * val_losses['kl'] +
            self.config.perceptual_weight * val_losses['perceptual']
        )
        
        return val_losses
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        checkpoint_path = Path(self.config.checkpoint_dir) / f'tinyvae_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as 'best' if it's the best so far
        if val_loss == min([l['total'] for l in self.val_losses]):
            best_path = Path(self.config.checkpoint_dir) / 'tinyvae_best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"Starting distillation training for {self.config.epochs} epochs")
        
        # First, ensure teacher is well-trained
        if not (Path(self.config.checkpoint_dir) / 'teacher_vae.pth').exists():
            self.logger.info("Training teacher model first...")
            self.train_teacher(epochs=20)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config.epochs + 1):
            self.logger.info(f"\nEpoch {epoch}/{self.config.epochs}")
            
            # Train
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validate
            val_losses = self.validate()
            self.val_losses.append(val_losses)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log results
            self.logger.info(
                f"Epoch {epoch} - "
                f"Train Loss: {train_losses['total']:.4f} "
                f"Val Loss: {val_losses['total']:.4f} "
                f"(R: {val_losses['recon']:.3f}, KL: {val_losses['kl']:.3f})"
            )
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch, val_losses['total'])
            
            # Track best
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                self.save_checkpoint(epoch, val_losses['total'])
        
        self.logger.info(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    
    def train_teacher(self, epochs: int = 20):
        """Pre-train the teacher model if needed"""
        self.logger.info("Pre-training teacher VAE...")
        
        teacher_optimizer = torch.optim.Adam(self.teacher.parameters(), lr=1e-3)
        
        for epoch in range(1, epochs + 1):
            self.teacher.train()
            epoch_loss = 0
            
            for batch_idx, (data, _) in enumerate(self.train_loader):
                data = data.to(self.device)
                
                recon, mu, log_var = self.teacher(data)
                recon_loss, kl_loss = self.vae_loss(recon, data, mu, log_var)
                loss = recon_loss + 0.1 * kl_loss
                
                teacher_optimizer.zero_grad()
                loss.backward()
                teacher_optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            self.logger.info(f"Teacher Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Save teacher model
        teacher_path = Path(self.config.checkpoint_dir) / 'teacher_vae.pth'
        torch.save(self.teacher.state_dict(), teacher_path)
        self.logger.info(f"Saved teacher model to {teacher_path}")


def main():
    parser = argparse.ArgumentParser(description='TinyVAE Distillation Training')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'imagenet-subset'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/tinyvae_distill')
    args = parser.parse_args()
    
    # Create config
    config = DistillationConfig()
    
    # Override with command line args
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(config, key, value)
    else:
        config.dataset = args.dataset
        config.epochs = args.epochs
        config.batch_size = args.batch_size
        config.learning_rate = args.lr
        config.device = args.device
        config.checkpoint_dir = args.checkpoint_dir
    
    # Save config
    config_path = Path(config.checkpoint_dir) / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Run training
    trainer = DistillationTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()