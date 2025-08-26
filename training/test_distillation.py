#!/usr/bin/env python3
"""
Quick test script for TinyVAE distillation
Tests that the distillation setup works without full training
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from training.distill_tinyvae import StandardVAE, DistillationConfig, DistillationTrainer
from models.vision.lightweight_vae import TinyVAE


def test_models():
    """Test that models can be instantiated and run"""
    print("Testing model instantiation...")
    
    # Create models
    teacher = StandardVAE(latent_dim=512)
    student = TinyVAE()
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32)
    
    # Teacher forward
    print("Testing teacher forward pass...")
    teacher_recon, teacher_mu, teacher_log_var = teacher(x)
    print(f"Teacher output shape: {teacher_recon.shape}")
    print(f"Teacher latent shape: {teacher_mu.shape}")
    assert teacher_recon.shape == (batch_size, 3, 32, 32)
    assert teacher_mu.shape == (batch_size, 512)
    
    # Student forward
    print("Testing student forward pass...")
    # For TinyVAE, we need to resize input to 224x224 or adjust the model
    x_student = F.interpolate(x, size=(224, 224), mode='bilinear')
    student_recon, student_mu, student_log_var = student(x_student)
    print(f"Student output shape: {student_recon.shape}")
    print(f"Student latent shape: {student_mu.shape}")
    assert student_recon.shape == (batch_size, 3, 224, 224)
    # TinyVAE outputs [batch, 128, 7, 7] for latent
    
    print("✓ Model tests passed!")
    return teacher, student


def test_distillation_loss():
    """Test the distillation loss calculation"""
    print("\nTesting distillation losses...")
    
    teacher = StandardVAE(latent_dim=512)
    student = TinyVAE()
    
    # Create dummy data
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    
    # Forward passes
    teacher_recon, teacher_mu, teacher_log_var = teacher(x)
    student_recon, student_mu, student_log_var = student(x)
    
    # Get latent representations
    teacher_z = teacher.reparameterize(teacher_mu, teacher_log_var)
    student_z = student.reparameterize(student_mu, student_log_var)
    
    # Flatten student latent for comparison
    student_z_flat = student_z.view(batch_size, -1)
    
    # Calculate losses
    recon_loss = F.mse_loss(student_recon, x)
    kl_loss = -0.5 * torch.mean(
        1 + student_log_var - student_mu.pow(2) - student_log_var.exp()
    )
    
    # For latent distillation, we need to handle dimension mismatch
    # Project teacher latent to match student dimension
    teacher_proj = torch.nn.Linear(512, student_z_flat.shape[1])
    teacher_z_proj = teacher_proj(teacher_z)
    latent_distill_loss = F.mse_loss(student_z_flat, teacher_z_proj.detach())
    
    # Output distillation
    output_distill_loss = F.mse_loss(student_recon, teacher_recon.detach())
    
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    print(f"Latent distillation loss: {latent_distill_loss.item():.4f}")
    print(f"Output distillation loss: {output_distill_loss.item():.4f}")
    
    # Combine with weights
    total_loss = (
        0.4 * recon_loss + 
        0.1 * kl_loss + 
        0.3 * latent_distill_loss + 
        0.2 * output_distill_loss
    )
    print(f"Total weighted loss: {total_loss.item():.4f}")
    
    print("✓ Loss calculation tests passed!")
    

def test_mini_training():
    """Test a mini training loop"""
    print("\nTesting mini training loop...")
    
    # Create minimal config
    config = DistillationConfig()
    config.dataset = 'cifar10'
    config.batch_size = 32
    config.epochs = 1  # Just one epoch for testing
    config.checkpoint_dir = './test_checkpoints'
    config.log_interval = 1
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        config.device = 'cpu'
        print("Running on CPU (no CUDA available)")
    
    try:
        # Create trainer
        print("Creating trainer...")
        trainer = DistillationTrainer(config)
        
        # Get one batch
        data_iter = iter(trainer.train_loader)
        batch_data, _ = next(data_iter)
        batch_data = batch_data.to(trainer.device)
        
        # Test one forward/backward pass
        print("Testing one training step...")
        
        # Teacher forward (eval mode)
        trainer.teacher.eval()
        with torch.no_grad():
            teacher_recon, teacher_mu, teacher_log_var = trainer.teacher(batch_data)
            teacher_z = trainer.teacher.reparameterize(teacher_mu, teacher_log_var)
        
        # Student forward
        trainer.student.train()
        student_recon, student_mu, student_log_var = trainer.student(batch_data)
        
        # Calculate basic loss
        recon_loss = F.mse_loss(student_recon, batch_data)
        kl_loss = -0.5 * torch.mean(
            1 + student_log_var - student_mu.pow(2) - student_log_var.exp()
        )
        total_loss = recon_loss + 0.1 * kl_loss
        
        # Backward pass
        trainer.optimizer.zero_grad()
        total_loss.backward()
        trainer.optimizer.step()
        
        print(f"One step loss: {total_loss.item():.4f}")
        print("✓ Mini training test passed!")
        
        # Clean up test checkpoint dir
        import shutil
        if Path(config.checkpoint_dir).exists():
            shutil.rmtree(config.checkpoint_dir)
            
    except Exception as e:
        print(f"Error in mini training: {e}")
        raise


def main():
    """Run all tests"""
    print("=" * 50)
    print("TinyVAE Distillation Test Suite")
    print("=" * 50)
    
    # Test 1: Model instantiation
    test_models()
    
    # Test 2: Loss calculations  
    test_distillation_loss()
    
    # Test 3: Mini training loop
    test_mini_training()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("Ready to run full distillation training with:")
    print("python training/distill_tinyvae.py")
    print("=" * 50)


if __name__ == '__main__':
    main()