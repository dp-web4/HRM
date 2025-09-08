"""
Context Encoder for SAGE
Replaces random vectors with meaningful context representations.

Problem: Current SAGE uses random vectors as "context" - meaningless noise.
Solution: Encode actual task structure, patterns, and semantic information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class TaskContextEncoder(nn.Module):
    """
    Encodes task-specific context into meaningful representations.
    Replaces random noise with structured information about the task.
    """
    
    def __init__(
        self,
        input_channels: int = 10,  # ARC uses 10 colors
        hidden_dim: int = 768,
        num_heads: int = 12,
        max_grid_size: int = 30,
        use_positional: bool = True,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_grid_size = max_grid_size
        
        # Color embeddings (for ARC tasks)
        self.color_embed = nn.Embedding(input_channels, hidden_dim // 4)
        
        # Spatial feature extraction
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, hidden_dim // 2, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((7, 7))  # Fixed spatial size
        )
        
        # Pattern detector kernels (learned)
        self.pattern_kernels = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Local patterns
            nn.Conv2d(1, 16, kernel_size=5, padding=2),  # Medium patterns
            nn.Conv2d(1, 16, kernel_size=7, padding=3),  # Large patterns
        ])
        
        # Positional encoding
        if use_positional:
            self.pos_encoder = PositionalEncoding2D(hidden_dim, max_grid_size)
        else:
            self.pos_encoder = None
        
        # Self-attention for context understanding
        # Ensure hidden_dim is divisible by num_heads
        actual_num_heads = num_heads
        while hidden_dim % actual_num_heads != 0 and actual_num_heads > 1:
            actual_num_heads -= 1
        
        self.context_attention = nn.MultiheadAttention(
            hidden_dim,
            actual_num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Statistical feature extractor
        self.stat_encoder = StatisticalEncoder(input_channels, hidden_dim // 4)
        
        # Symmetry detector
        self.symmetry_detector = SymmetryDetector(hidden_dim // 4)
        
        # Final context projection
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        input_grid: torch.Tensor,
        target_grid: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode input (and optionally target) into context.
        
        Args:
            input_grid: Input pattern [B, H, W] or [B, C, H, W]
            target_grid: Target pattern (for training)
            
        Returns:
            Dictionary with context representations
        """
        batch_size = input_grid.shape[0]
        
        # Ensure correct shape
        if len(input_grid.shape) == 3:
            # Convert to one-hot if needed
            input_grid = self._to_one_hot(input_grid)
        
        # Extract different types of features
        spatial_features = self._extract_spatial_features(input_grid)
        pattern_features = self._extract_pattern_features(input_grid)
        statistical_features = self.stat_encoder(input_grid)
        symmetry_features = self.symmetry_detector(input_grid)
        
        # Combine features
        # Ensure all features are 2D tensors [B, features]
        features_to_combine = []
        
        if spatial_features.dim() == 1:
            spatial_features = spatial_features.unsqueeze(0)
        features_to_combine.append(spatial_features)
        
        if pattern_features.dim() == 1:
            pattern_features = pattern_features.unsqueeze(0)
        features_to_combine.append(pattern_features)
        
        if statistical_features.dim() == 1:
            statistical_features = statistical_features.unsqueeze(0)
        features_to_combine.append(statistical_features)
        
        if symmetry_features.dim() == 1:
            symmetry_features = symmetry_features.unsqueeze(0)
        features_to_combine.append(symmetry_features)
        
        combined = torch.cat(features_to_combine, dim=-1)
        
        # Apply self-attention to understand relationships
        # First project combined features to correct dimension
        if combined.shape[-1] != self.hidden_dim:
            # Project to hidden_dim first
            combined = self.context_projection(
                F.pad(combined, (0, max(0, self.hidden_dim * 2 - combined.shape[-1])))
            )
        
        # Reshape for attention
        combined = combined.unsqueeze(1)  # [B, 1, hidden_dim]
        attended, attention_weights = self.context_attention(
            combined, combined, combined
        )
        
        # Final output
        context = attended.squeeze(1)
        
        # If target provided, encode the transformation
        if target_grid is not None:
            if len(target_grid.shape) == 3:
                target_grid = self._to_one_hot(target_grid)
            
            target_context = self._encode_transformation(input_grid, target_grid)
            context = context + target_context
        
        return {
            'context': context,
            'spatial': spatial_features,
            'patterns': pattern_features,
            'statistics': statistical_features,
            'symmetry': symmetry_features,
            'attention_weights': attention_weights
        }
    
    def _to_one_hot(self, grid: torch.Tensor) -> torch.Tensor:
        """Convert integer grid to one-hot encoding."""
        B, H, W = grid.shape
        one_hot = F.one_hot(grid.long(), num_classes=self.input_channels)
        return one_hot.permute(0, 3, 1, 2).float()
    
    def _extract_spatial_features(self, grid: torch.Tensor) -> torch.Tensor:
        """Extract spatial features using conv layers."""
        features = self.spatial_encoder(grid)
        return features.reshape(grid.shape[0], -1)
    
    def _extract_pattern_features(self, grid: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale pattern features."""
        # Convert to grayscale for pattern detection
        gray = grid.mean(dim=1, keepdim=True)
        
        patterns = []
        for kernel in self.pattern_kernels:
            pattern = kernel(gray)
            pattern = F.adaptive_avg_pool2d(pattern, (4, 4))
            patterns.append(pattern.reshape(grid.shape[0], -1))
        
        return torch.cat(patterns, dim=-1)
    
    def _encode_transformation(
        self,
        input_grid: torch.Tensor,
        target_grid: torch.Tensor
    ) -> torch.Tensor:
        """Encode the transformation from input to target."""
        # Compute difference
        diff = target_grid - input_grid
        
        # Extract features from the difference
        diff_spatial = self.spatial_encoder(diff)
        diff_stats = self.stat_encoder(diff)
        
        # Combine and project
        transform_features = torch.cat([
            diff_spatial.reshape(input_grid.shape[0], -1),
            diff_stats
        ], dim=-1)
        
        # Project to context dimension
        transform_context = self.context_projection(
            F.pad(transform_features, (0, self.hidden_dim * 2 - transform_features.shape[-1]))
        )
        
        return transform_context


class StatisticalEncoder(nn.Module):
    """Extract statistical features from grids."""
    
    def __init__(self, num_colors: int, output_dim: int):
        super().__init__()
        self.num_colors = num_colors
        self.output_dim = output_dim
        
        # Project statistics to output dimension
        self.stat_projection = nn.Sequential(
            nn.Linear(num_colors * 4 + 10, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """Extract statistical features."""
        B, C, H, W = grid.shape
        
        features = []
        
        # Color distribution
        color_dist = grid.mean(dim=(2, 3))  # [B, C]
        features.append(color_dist)
        
        # Color variance
        color_var = grid.var(dim=(2, 3))  # [B, C]
        features.append(color_var)
        
        # Spatial statistics per color
        for c in range(min(C, self.num_colors)):
            channel = grid[:, c:c+1, :, :]
            
            # Center of mass
            y_coords = torch.arange(H, device=grid.device).view(1, 1, H, 1)
            x_coords = torch.arange(W, device=grid.device).view(1, 1, 1, W)
            
            mass = channel.sum(dim=(2, 3), keepdim=True) + 1e-6
            y_center = (channel * y_coords).sum(dim=(2, 3)) / mass.squeeze()
            x_center = (channel * x_coords).sum(dim=(2, 3)) / mass.squeeze()
            
            features.append(y_center / H)  # Normalize
            features.append(x_center / W)
        
        # Global statistics
        total_active = (grid.sum(dim=1) > 0).float().mean(dim=(1, 2))
        features.append(total_active.unsqueeze(1))
        
        # Concat and project
        stats = torch.cat(features, dim=-1)
        
        # Pad if needed
        if stats.shape[-1] < self.num_colors * 4 + 10:
            stats = F.pad(stats, (0, self.num_colors * 4 + 10 - stats.shape[-1]))
        elif stats.shape[-1] > self.num_colors * 4 + 10:
            stats = stats[:, :self.num_colors * 4 + 10]
        
        return self.stat_projection(stats)


class SymmetryDetector(nn.Module):
    """Detect various types of symmetry in grids."""
    
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        
        self.symmetry_projection = nn.Sequential(
            nn.Linear(8, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """Detect symmetries."""
        B, C, H, W = grid.shape
        
        # Convert to binary mask (any color present)
        mask = (grid.sum(dim=1) > 0).float()
        
        symmetries = []
        
        # Horizontal symmetry
        h_flip = torch.flip(mask, dims=[1])
        h_sym = (mask * h_flip).sum() / (mask.sum() + 1e-6)
        symmetries.append(h_sym)
        
        # Vertical symmetry
        v_flip = torch.flip(mask, dims=[2])
        v_sym = (mask * v_flip).sum() / (mask.sum() + 1e-6)
        symmetries.append(v_sym)
        
        # Diagonal symmetry
        if H == W:
            d_sym = (mask * mask.transpose(-2, -1)).sum() / (mask.sum() + 1e-6)
            symmetries.append(d_sym)
        else:
            symmetries.append(torch.zeros(B, device=grid.device))
        
        # Rotational symmetry (90 degrees)
        if H == W:
            rot90 = torch.rot90(mask, k=1, dims=[1, 2])
            r_sym = (mask * rot90).sum() / (mask.sum() + 1e-6)
            symmetries.append(r_sym)
        else:
            symmetries.append(torch.zeros(B, device=grid.device))
        
        # Repetition detection (simple periodicity)
        for period in [2, 3, 4]:
            if H > period:
                rep_h = (mask[:, :-period] * mask[:, period:]).sum() / (mask.sum() + 1e-6)
                symmetries.append(rep_h)
            else:
                symmetries.append(torch.zeros(B, device=grid.device))
        
        # Stack and project
        if len(symmetries) > 0 and isinstance(symmetries[0], torch.Tensor) and symmetries[0].dim() == 0:
            # If symmetries are scalars, stack them properly
            sym_features = torch.stack(symmetries).unsqueeze(0).expand(B, -1)
        else:
            sym_features = torch.stack(symmetries, dim=-1)
        
        # Ensure we have exactly 8 features
        if sym_features.shape[-1] < 8:
            sym_features = F.pad(sym_features, (0, 8 - sym_features.shape[-1]))
        elif sym_features.shape[-1] > 8:
            sym_features = sym_features[:, :8]
        
        return self.symmetry_projection(sym_features)


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for grids."""
    
    def __init__(self, hidden_dim: int, max_size: int = 30):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Create 2D positional encodings
        pe = torch.zeros(max_size, max_size, hidden_dim)
        
        # Use sinusoidal encoding
        y_pos = torch.arange(0, max_size).unsqueeze(1).float()
        x_pos = torch.arange(0, max_size).unsqueeze(0).float()
        
        # Create div_term for the available dimensions
        dim_t = torch.arange(0, hidden_dim, 2).float()
        div_term = torch.exp(dim_t * -(np.log(10000.0) / hidden_dim))
        
        # Fill in the positional encodings
        # We need to handle different hidden_dim sizes properly
        for i in range(0, hidden_dim, 4):
            if i < hidden_dim:
                pe[:, :, i] = torch.sin(y_pos * div_term[i//4 % len(div_term)]).squeeze()
            if i + 1 < hidden_dim:
                pe[:, :, i+1] = torch.cos(y_pos * div_term[i//4 % len(div_term)]).squeeze()
            if i + 2 < hidden_dim:
                pe[:, :, i+2] = torch.sin(x_pos * div_term[i//4 % len(div_term)]).squeeze()
            if i + 3 < hidden_dim:
                pe[:, :, i+3] = torch.cos(x_pos * div_term[i//4 % len(div_term)]).squeeze()
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        H, W = x.shape[-2:]
        return x + self.pe[:H, :W].unsqueeze(0)


class MultiModalContextEncoder(nn.Module):
    """
    Combine multiple context sources: visual, linguistic, temporal.
    This is what SAGE needs instead of random vectors.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        use_llm_context: bool = True,
        use_temporal: bool = True,
        use_memory: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_llm_context = use_llm_context
        self.use_temporal = use_temporal
        self.use_memory = use_memory
        
        # Task context encoder
        self.task_encoder = TaskContextEncoder(hidden_dim=hidden_dim)
        
        # Temporal context (for sequential tasks)
        if use_temporal:
            self.temporal_encoder = nn.LSTM(
                hidden_dim,
                hidden_dim // 2,
                num_layers=2,
                bidirectional=True,
                batch_first=True
            )
        
        # Memory context (from previous attempts)
        if use_memory:
            # Ensure hidden_dim is divisible by num_heads
            mem_heads = 8
            while hidden_dim % mem_heads != 0 and mem_heads > 1:
                mem_heads -= 1
            
            self.memory_attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads=mem_heads,
                batch_first=True
            )
        
        # Context fusion
        num_contexts = 1 + int(use_llm_context) + int(use_temporal) + int(use_memory)
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_contexts, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(
        self,
        input_grid: torch.Tensor,
        llm_context: Optional[torch.Tensor] = None,
        temporal_context: Optional[List[torch.Tensor]] = None,
        memory_context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode multi-modal context.
        
        Args:
            input_grid: Visual input
            llm_context: Language understanding from LLM
            temporal_context: Previous states (for sequential tasks)
            memory_context: Retrieved memories
            
        Returns:
            Fused context representation
        """
        contexts = []
        
        # Task-specific visual context
        task_context = self.task_encoder(input_grid)['context']
        contexts.append(task_context)
        
        # LLM linguistic context
        if self.use_llm_context and llm_context is not None:
            contexts.append(llm_context)
        elif self.use_llm_context:
            # Use zeros if LLM context expected but not provided
            contexts.append(torch.zeros_like(task_context))
        
        # Temporal context
        if self.use_temporal and temporal_context is not None:
            # Stack temporal states
            temporal = torch.stack(temporal_context, dim=1)
            _, (h_n, _) = self.temporal_encoder(temporal)
            # Combine forward and backward hidden states
            temporal_encoded = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            contexts.append(temporal_encoded)
        elif self.use_temporal:
            contexts.append(torch.zeros_like(task_context))
        
        # Memory context
        if self.use_memory and memory_context is not None:
            # Use attention to retrieve relevant memories
            query = task_context.unsqueeze(1)
            attended_memory, _ = self.memory_attention(
                query, memory_context, memory_context
            )
            contexts.append(attended_memory.squeeze(1))
        elif self.use_memory:
            contexts.append(torch.zeros_like(task_context))
        
        # Fuse all contexts
        combined = torch.cat(contexts, dim=-1)
        fused_context = self.context_fusion(combined)
        
        return fused_context


if __name__ == "__main__":
    print("Testing Context Encoders...")
    
    # Test TaskContextEncoder
    encoder = TaskContextEncoder(hidden_dim=768)
    
    # Create sample ARC-like grid
    sample_grid = torch.randint(0, 10, (2, 15, 15))
    
    context_dict = encoder(sample_grid)
    print(f"✅ Task context shape: {context_dict['context'].shape}")
    print(f"   Spatial features: {context_dict['spatial'].shape}")
    print(f"   Pattern features: {context_dict['patterns'].shape}")
    print(f"   Statistical features: {context_dict['statistics'].shape}")
    print(f"   Symmetry features: {context_dict['symmetry'].shape}")
    
    # Test MultiModalContextEncoder
    multi_encoder = MultiModalContextEncoder(hidden_dim=768)
    
    # Create fake LLM context
    llm_context = torch.randn(2, 768)
    
    # Create temporal context (3 previous states)
    temporal = [torch.randn(2, 768) for _ in range(3)]
    
    # Create memory context (5 memories)
    memory = torch.randn(2, 5, 768)
    
    fused = multi_encoder(
        sample_grid,
        llm_context=llm_context,
        temporal_context=temporal,
        memory_context=memory
    )
    
    print(f"\n✅ Multi-modal context shape: {fused.shape}")
    print("\nContext encoder ready to replace random vectors!")