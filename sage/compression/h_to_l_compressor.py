"""
H→L Context Compression: 4K→256 Dimensional Compression
Compresses rich context from H-module (4096D) to actionable representation for L-module (256D).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import math

# Import our components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from context.reality_context_4k import RealityContext4K


@dataclass
class CompressionMetrics:
    """Metrics for evaluating compression quality."""
    reconstruction_loss: float
    information_retained: float  # 0-1 score
    sparsity: float  # How sparse is the compressed representation
    mutual_information: float  # Between original and compressed


class InformationBottleneck(nn.Module):
    """
    Information bottleneck layer that compresses while preserving task-relevant information.
    Based on variational information bottleneck principle.
    """
    
    def __init__(self, input_dim: int, bottleneck_dim: int, beta: float = 0.01):
        """
        Args:
            input_dim: Input dimension (4096)
            bottleneck_dim: Compressed dimension (256)
            beta: Trade-off between compression and information preservation
        """
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.beta = beta
        
        # Encoder to bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )
        
        # Variational parameters
        self.mu_layer = nn.Linear(512, bottleneck_dim)
        self.logvar_layer = nn.Linear(512, bottleneck_dim)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Use mean for inference
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compress input through information bottleneck.
        
        Returns:
            compressed: Compressed representation
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Encode
        hidden = self.encoder(x)
        
        # Get distribution parameters
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        
        # Sample from distribution
        compressed = self.reparameterize(mu, logvar)
        
        return compressed, mu, logvar
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Calculate KL divergence from standard normal."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class AttentionCompressor(nn.Module):
    """
    Attention-based compression that learns what to attend to.
    Inspired by perceiver architecture.
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_latents: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_latents = num_latents
        
        # Ensure embed_dim is divisible by num_heads
        latent_dim = max(64, (output_dim // num_latents) * 8)  # Make divisible by 8
        
        # Learnable latent codes
        self.latent_codes = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        
        # Cross-attention from latents to input
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_latents * latent_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compress using cross-attention."""
        batch_size = x.shape[0]
        
        # Project input
        x_proj = self.input_proj(x).unsqueeze(1)  # [B, 1, hidden]
        
        # Expand latent codes for batch
        latents = self.latent_codes.expand(batch_size, -1, -1)
        
        # Cross-attend from latents to input
        attended, _ = self.cross_attention(
            query=latents,
            key=x_proj,
            value=x_proj
        )
        
        # Flatten and project to output
        attended_flat = attended.reshape(batch_size, -1)
        compressed = self.output_proj(attended_flat)
        
        return compressed


class HierarchicalCompressor(nn.Module):
    """
    Hierarchical compression that preserves structure.
    Compresses different aspects of context at different rates.
    """
    
    def __init__(self):
        super().__init__()
        
        # Different compression ratios for different aspects
        self.sensory_compressor = nn.Sequential(
            nn.Linear(1536, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 96)  # 16:1 compression
        )
        
        self.semantic_compressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 80)  # 12.8:1 compression
        )
        
        self.physical_compressor = nn.Sequential(
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, 48)  # 16:1 compression
        )
        
        self.temporal_compressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 32)  # 24:1 compression
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(96 + 80 + 48 + 32, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
    
    def forward(self, context: RealityContext4K) -> torch.Tensor:
        """Compress each aspect hierarchically."""
        # Extract aspects
        sensory = torch.cat([
            context.visual, context.depth, context.audio,
            context.tactile, context.proprioceptive
        ], dim=-1)
        
        semantic = torch.cat([
            context.objects, context.affordances,
            context.relationships, context.intentions
        ], dim=-1)
        
        physical = torch.cat([
            context.dynamics, context.materials, context.constraints
        ], dim=-1)
        
        temporal = torch.cat([
            context.immediate, context.historical, context.predictive
        ], dim=-1)
        
        # Compress each aspect
        sensory_compressed = self.sensory_compressor(sensory)
        semantic_compressed = self.semantic_compressor(semantic)
        physical_compressed = self.physical_compressor(physical)
        temporal_compressed = self.temporal_compressor(temporal)
        
        # Fuse compressed representations
        compressed = torch.cat([
            sensory_compressed,
            semantic_compressed,
            physical_compressed,
            temporal_compressed
        ], dim=-1)
        
        return self.fusion(compressed)


class HToLCompressor(nn.Module):
    """
    Main H→L compressor combining multiple compression strategies.
    Compresses 4096D context to 256D actionable representation.
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        output_dim: int = 256,
        compression_type: str = "hybrid"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.compression_type = compression_type
        
        # Different compression strategies
        if compression_type == "bottleneck":
            self.compressor = InformationBottleneck(input_dim, output_dim)
            self.use_vae = True
        elif compression_type == "attention":
            self.compressor = AttentionCompressor(input_dim, output_dim)
            self.use_vae = False
        elif compression_type == "hierarchical":
            self.compressor = HierarchicalCompressor()
            self.hierarchical_proj = nn.Linear(input_dim, output_dim)  # Fallback for tensor input
            self.use_vae = False
        elif compression_type == "hybrid":
            # Combine all strategies
            self.bottleneck = InformationBottleneck(input_dim, output_dim // 3)
            self.attention = AttentionCompressor(input_dim, output_dim // 3)
            self.hierarchical_proj = nn.Linear(input_dim, output_dim // 3)
            self.fusion = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
            self.use_vae = True
        else:
            raise ValueError(f"Unknown compression type: {compression_type}")
        
        # Decoder for reconstruction (to measure information loss)
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, input_dim)
        )
        
    def forward(
        self,
        context: torch.Tensor,
        return_metrics: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compress context from H to L.
        
        Args:
            context: 4096D context tensor or RealityContext4K
            return_metrics: Whether to compute compression metrics
            
        Returns:
            Dictionary with compressed representation and optional metrics
        """
        # Convert to tensor if needed
        if isinstance(context, RealityContext4K):
            context_tensor = context.to_tensor()
        else:
            context_tensor = context
        
        # Compress based on strategy
        if self.compression_type == "hybrid":
            # Use all three strategies and fuse
            bottleneck_out, mu, logvar = self.bottleneck(context_tensor)
            attention_out = self.attention(context_tensor)
            hierarchical_out = self.hierarchical_proj(context_tensor)
            
            # Pad to ensure same size
            max_dim = max(bottleneck_out.shape[-1], attention_out.shape[-1], hierarchical_out.shape[-1])
            if bottleneck_out.shape[-1] < max_dim:
                bottleneck_out = F.pad(bottleneck_out, (0, max_dim - bottleneck_out.shape[-1]))
            if attention_out.shape[-1] < max_dim:
                attention_out = F.pad(attention_out, (0, max_dim - attention_out.shape[-1]))
            if hierarchical_out.shape[-1] < max_dim:
                hierarchical_out = F.pad(hierarchical_out, (0, max_dim - hierarchical_out.shape[-1]))
            
            combined = torch.cat([
                bottleneck_out[:, :self.output_dim // 3],
                attention_out[:, :self.output_dim // 3],
                hierarchical_out[:, :self.output_dim // 3]
            ], dim=-1)
            
            # Pad to exact output dimension
            if combined.shape[-1] < self.output_dim:
                combined = F.pad(combined, (0, self.output_dim - combined.shape[-1]))
            
            compressed = self.fusion(combined[:, :self.output_dim])
            kl_loss = self.bottleneck.kl_divergence(mu, logvar) if self.use_vae else 0
            
        elif self.compression_type == "bottleneck":
            compressed, mu, logvar = self.compressor(context_tensor)
            kl_loss = self.compressor.kl_divergence(mu, logvar)
            
        elif self.compression_type == "hierarchical":
            if isinstance(context, RealityContext4K):
                compressed = self.compressor(context)
            else:
                # For tensor input, use linear projection
                compressed = self.hierarchical_proj(context_tensor)
            kl_loss = 0
            
        else:
            if self.compression_type == "hierarchical":
                # For hierarchical, just use linear projection for tensor input
                compressed = self.hierarchical_proj(context_tensor) if hasattr(self, 'hierarchical_proj') else context_tensor[:, :self.output_dim]
            else:
                compressed = self.compressor(context_tensor)
            kl_loss = 0
        
        result = {"compressed": compressed}
        
        # Add VAE parameters if applicable
        if self.use_vae and self.compression_type != "hierarchical":
            result["kl_loss"] = kl_loss
        
        # Compute metrics if requested
        if return_metrics:
            with torch.no_grad():
                reconstructed = self.decoder(compressed)
                reconstruction_loss = F.mse_loss(reconstructed, context_tensor)
                
                # Information retained (1 - normalized reconstruction error)
                info_retained = 1.0 - (reconstruction_loss / (context_tensor.var() + 1e-6))
                
                # Sparsity of compressed representation
                sparsity = (compressed.abs() < 0.01).float().mean()
                
                # Approximate mutual information
                if self.use_vae and 'mu' in locals():
                    mi = 0.5 * torch.log(1 + mu.pow(2).mean())
                else:
                    mi = torch.tensor(0.0)
                
                result["metrics"] = CompressionMetrics(
                    reconstruction_loss=reconstruction_loss.item(),
                    information_retained=info_retained.item(),
                    sparsity=sparsity.item(),
                    mutual_information=mi.item()
                )
        
        return result
    
    def compress(self, context: torch.Tensor) -> torch.Tensor:
        """Simple compression interface."""
        return self.forward(context)["compressed"]
    
    def decompress(self, compressed: torch.Tensor) -> torch.Tensor:
        """Reconstruct original context (lossy)."""
        return self.decoder(compressed)


class CompressionLoss(nn.Module):
    """
    Loss function for training the compressor.
    Balances reconstruction, sparsity, and information preservation.
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 0.01,
        sparsity_weight: float = 0.001
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.sparsity_weight = sparsity_weight
    
    def forward(
        self,
        original: torch.Tensor,
        compressed: torch.Tensor,
        reconstructed: torch.Tensor,
        kl_loss: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate compression loss.
        
        Returns:
            Dictionary with total loss and components
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, original)
        
        # KL divergence (if using VAE)
        kl = kl_loss if kl_loss is not None else torch.tensor(0.0)
        
        # Sparsity regularization
        sparsity = compressed.abs().mean()
        
        # Total loss
        total = (
            self.reconstruction_weight * recon_loss +
            self.kl_weight * kl +
            self.sparsity_weight * sparsity
        )
        
        return {
            "total": total,
            "reconstruction": recon_loss,
            "kl": kl,
            "sparsity": sparsity
        }


def test_compression():
    """Test the H→L compression."""
    print("\n" + "="*60)
    print("🔬 Testing H→L Context Compression")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    
    # Create dummy 4K context
    context_4k = torch.randn(batch_size, 4096).to(device)
    print(f"✅ Input context: {context_4k.shape}")
    
    # Test different compression types
    for comp_type in ["bottleneck", "attention", "hierarchical", "hybrid"]:
        print(f"\n📊 Testing {comp_type} compression:")
        
        compressor = HToLCompressor(
            input_dim=4096,
            output_dim=256,
            compression_type=comp_type
        ).to(device)
        
        # Compress
        result = compressor(context_4k, return_metrics=True)
        compressed = result["compressed"]
        
        print(f"   Compressed shape: {compressed.shape}")
        print(f"   Compression ratio: {4096/256:.1f}x")
        
        if "metrics" in result:
            metrics = result["metrics"]
            print(f"   Reconstruction loss: {metrics.reconstruction_loss:.4f}")
            print(f"   Information retained: {metrics.information_retained:.2%}")
            print(f"   Sparsity: {metrics.sparsity:.2%}")
        
        # Test decompression
        decompressed = compressor.decompress(compressed)
        print(f"   Decompressed shape: {decompressed.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in compressor.parameters())
    print(f"\n📈 Compressor parameters: {total_params:,}")
    
    print("\n✅ H→L Compression working!")
    print(f"   Successfully compresses 4096D → 256D (16x compression)")
    print(f"   Multiple strategies available for different use cases")


if __name__ == "__main__":
    test_compression()