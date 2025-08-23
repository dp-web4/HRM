"""
Vision IRP Plugin - Actual Implementation
Refines visual representations in latent space for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from sage.irp.base import IRPPlugin
from models.vision.lightweight_vae import create_vae_for_jetson


class LatentRefiner(nn.Module):
    """
    Small U-Net for refining in latent space
    Operates on 7x7x256 latent representations
    """
    
    def __init__(self, channels: int = 256):
        super().__init__()
        
        # Encoder path
        self.enc1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.enc2 = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)  # 7 → 4
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        
        # Decoder path
        self.dec1 = nn.ConvTranspose2d(channels * 2, channels, 3, stride=2, padding=1, output_padding=1)  # 4 → 7
        self.dec2 = nn.Conv2d(channels * 2, channels, 3, padding=1)  # With skip connection
        
        # Output projection
        self.out = nn.Conv2d(channels, channels, 1)
        
        # Normalization
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels * 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Refine latent representation
        Input/Output: [B, 256, 7, 7]
        """
        # Encoder
        e1 = F.relu(self.norm1(self.enc1(x)))
        e2 = F.relu(self.norm2(self.enc2(e1)))
        
        # Bottleneck
        b = F.relu(self.norm2(self.bottleneck(e2)))
        
        # Decoder
        d1 = self.dec1(b)
        
        # Ensure dimensions match for skip connection
        if d1.shape[2:] != e1.shape[2:]:
            # Pad or crop to match
            diff_h = e1.shape[2] - d1.shape[2]
            diff_w = e1.shape[3] - d1.shape[3]
            d1 = F.pad(d1, [diff_w//2, diff_w - diff_w//2, diff_h//2, diff_h - diff_h//2])
        
        d1 = torch.cat([d1, e1], dim=1)  # Skip connection
        d2 = F.relu(self.dec2(d1))
        
        # Residual output
        return x + self.out(d2)


class VisionIRPImpl(IRPPlugin):
    """
    Actual implementation of Vision IRP
    Uses VAE for latent space operations
    """
    
    def __init__(
        self,
        vae_variant: str = 'minimal',
        refiner_channels: int = 128,
        max_iterations: int = 50,
        eps: float = 0.01,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        # Create default config if not provided
        if config is None:
            config = {
                'max_iterations': max_iterations,
                'halt_eps': eps,
                'entity_id': 'vision_irp'
            }
        super().__init__(config)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create VAE and refiner
        self.vae = create_vae_for_jetson(vae_variant).to(self.device)
        self.vae.eval()  # Always in eval mode
        
        # Adjust refiner channels based on VAE variant
        if vae_variant == 'minimal':
            refiner_channels = 128
        else:
            refiner_channels = 256
            
        self.refiner = LatentRefiner(refiner_channels).to(self.device)
        
        # IRP parameters
        self.max_iterations = max_iterations
        self.eps = eps
        self.iteration = 0
        
        # Cache for original image
        self.original_image = None
        self.current_latent = None
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with task-specific configuration"""
        self.max_iterations = config.get('max_iterations', self.max_iterations)
        self.eps = config.get('eps', self.eps)
        
        # Task-specific settings
        self.task = config.get('task', 'segmentation')
        self.quality_threshold = config.get('quality_threshold', 0.95)
        
    def preprocess(self, x: Any) -> torch.Tensor:
        """
        Convert input to latent representation
        Input: Image tensor [B, 3, H, W] or numpy array
        Output: Latent tensor [B, C, 7, 7]
        """
        # Convert to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        
        # Ensure on correct device
        x = x.to(self.device)
        
        # Normalize to [0, 1] if needed
        if x.max() > 1.0:
            x = x / 255.0
        
        # Store original for quality evaluation
        self.original_image = x
        
        # Encode to latent
        with torch.no_grad():
            latent, _ = self.vae.encode(x)
        
        self.current_latent = latent
        return latent
    
    def compute_energy(self, state: torch.Tensor) -> float:
        """
        Compute energy of current state
        Lower energy = better quality + regularization
        """
        with torch.no_grad():
            # Decode current latent
            recon = self.vae.decode(state)
            
            # Reconstruction error
            recon_error = F.mse_loss(recon, self.original_image)
            
            # Latent regularization (encourage smoothness)
            latent_reg = torch.mean(torch.abs(state))
            
            # Combined energy
            energy = recon_error + 0.1 * latent_reg
            
        return -energy.item()  # Negative so lower is better
    
    def refine_step(self, state: torch.Tensor) -> torch.Tensor:
        """
        Single refinement step in latent space
        """
        # Apply refiner network
        refined = self.refiner(state)
        
        # Clip to reasonable range
        refined = torch.clamp(refined, -3, 3)
        
        self.iteration += 1
        return refined
    
    def should_halt(self, energy_history: list) -> bool:
        """
        Determine if refinement should stop early
        Based on energy convergence
        """
        if len(energy_history) < 3:
            return False
        
        # Check if energy has converged
        recent = energy_history[-3:]
        delta = abs(recent[-1] - recent[-2])
        
        # Stop if converged or max iterations
        return delta < self.eps or self.iteration >= self.max_iterations
    
    def postprocess(self, state: torch.Tensor) -> Any:
        """
        Convert refined latent back to image space
        """
        with torch.no_grad():
            refined_image = self.vae.decode(state)
        
        return refined_image
    
    def compute_trust(self, initial: Any, refined: Any) -> float:
        """
        Compute trust score based on quality preservation
        """
        # Quality preservation (PSNR-based)
        mse = F.mse_loss(refined, self.original_image)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        # Trust based on PSNR (>30dB is good)
        trust = torch.clamp(psnr / 40, 0, 1).item()
        
        # Bonus for early stopping
        if self.iteration < self.max_iterations * 0.5:
            trust *= 1.1
            
        return min(trust, 1.0)
    
    def refine(self, x: Any, early_stop: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """
        Complete refinement pipeline
        """
        # Reset iteration counter
        self.iteration = 0
        
        # Preprocess to latent
        state = self.preprocess(x)
        initial_state = state.clone()
        
        # Track energy
        energy_history = []
        energy_history.append(self.compute_energy(state))
        
        # Refinement loop
        while self.iteration < self.max_iterations:
            # Refine
            state = self.refine_step(state)
            
            # Compute energy
            energy = self.compute_energy(state)
            energy_history.append(energy)
            
            # Check early stop
            if early_stop and self.should_halt(energy_history):
                break
        
        # Postprocess
        refined = self.postprocess(state)
        
        # Compute metrics
        trust = self.compute_trust(self.original_image, refined)
        
        # Build telemetry
        telemetry = {
            'iterations': self.iteration,
            'final_energy': energy_history[-1],
            'energy_delta': energy_history[-1] - energy_history[0],
            'trust': trust,
            'early_stopped': self.iteration < self.max_iterations,
            'compute_saved': 1 - (self.iteration / self.max_iterations)
        }
        
        return refined, telemetry


def create_vision_irp(device: Optional[torch.device] = None) -> VisionIRPImpl:
    """Factory function for Vision IRP"""
    return VisionIRPImpl(
        vae_variant='minimal',  # Fast variant for Jetson
        device=device
    )


if __name__ == "__main__":
    print("Testing Vision IRP Implementation")
    print("=" * 50)
    
    # Create IRP
    irp = create_vision_irp()
    
    # Test with random image
    test_image = torch.randn(1, 3, 224, 224).cuda()
    
    # Run refinement
    print("\nRunning refinement with early stopping...")
    refined, telemetry = irp.refine(test_image, early_stop=True)
    
    print(f"\nResults:")
    print(f"  Iterations: {telemetry['iterations']}")
    print(f"  Final energy: {telemetry['final_energy']:.4f}")
    print(f"  Trust score: {telemetry['trust']:.3f}")
    print(f"  Compute saved: {telemetry['compute_saved']*100:.1f}%")
    
    # Compare with full refinement
    print("\nRunning full refinement for comparison...")
    irp2 = create_vision_irp()
    refined_full, telemetry_full = irp2.refine(test_image, early_stop=False)
    
    print(f"\nFull refinement:")
    print(f"  Iterations: {telemetry_full['iterations']}")
    print(f"  Final energy: {telemetry_full['final_energy']:.4f}")
    
    print(f"\nSpeedup: {telemetry_full['iterations'] / telemetry['iterations']:.2f}x")