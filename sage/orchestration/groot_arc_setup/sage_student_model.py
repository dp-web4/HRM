#!/usr/bin/env python3
"""
SAGE Student Model for Knowledge Distillation from GR00T N1.5

Architecture:
- Feature Projection: 2048 → 512
- Transformer Reasoning: 6 layers, 8 heads
- Grid Decoder: 512 → 30×30×10

Total: ~48.5M parameters (56x smaller than GR00T's 2.7B)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class SAGEStudent(nn.Module):
    """
    SAGE Student model for ARC-AGI reasoning via knowledge distillation.

    Args:
        input_dim: Dimension of GR00T features (default: 2048)
        hidden_dim: Internal representation dimension (default: 512)
        num_layers: Number of transformer layers (default: 6)
        num_heads: Number of attention heads (default: 8)
        ffn_dim: Feed-forward network expansion dimension (default: 2048)
        max_seq_len: Maximum sequence length (default: 6000)
        grid_size: ARC grid size (default: 30)
        num_classes: Number of color classes (default: 10, colors 0-9)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        max_seq_len: int = 6000,
        grid_size: int = 30,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.num_classes = num_classes

        # Feature projection: 2048 → 512
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer reasoning module
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for better stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Grid decoder: 512 → 30×30×10
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, grid_size * grid_size * num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        groot_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through SAGE student model.

        Args:
            groot_features: GR00T backbone features [batch, seq_len, 2048]
            attention_mask: Attention mask [batch, seq_len] (1 = valid, 0 = padded)

        Returns:
            grid_logits: Output grid logits [batch, 30, 30, 10]
            student_features: Student representations [batch, seq_len, 512]
        """
        batch_size, seq_len, _ = groot_features.shape

        # Project GR00T features to student dimension
        x = self.projection(groot_features)  # [batch, seq_len, 512]

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        pos_emb = self.pos_embedding(positions)  # [1, seq_len, 512]
        x = x + pos_emb

        # Prepare attention mask for transformer
        # PyTorch transformer expects: True = ignore, False = attend
        if attention_mask is not None:
            # Input mask: 1 = valid, 0 = padded
            # Transformer mask: True = padded, False = valid
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        # Transformer reasoning
        student_features = self.transformer(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )  # [batch, seq_len, 512]

        # Pool features (masked mean pooling)
        if attention_mask is not None:
            # Expand mask for broadcasting
            mask_expanded = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            # Masked mean: sum(features * mask) / sum(mask)
            pooled = (student_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Simple mean pooling
            pooled = student_features.mean(dim=1)  # [batch, 512]

        # Decode to grid
        grid_flat = self.decoder(pooled)  # [batch, 9000]
        grid_logits = grid_flat.view(
            batch_size, self.grid_size, self.grid_size, self.num_classes
        )  # [batch, 30, 30, 10]

        return grid_logits, student_features

    def predict_grid(
        self,
        groot_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict output grid from GR00T features.

        Args:
            groot_features: GR00T features [batch, seq_len, 2048]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            predicted_grid: Predicted grid [batch, 30, 30] with values 0-9
        """
        grid_logits, _ = self.forward(groot_features, attention_mask)
        predicted_grid = grid_logits.argmax(dim=-1)  # [batch, 30, 30]
        return predicted_grid

    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        projection_params = sum(p.numel() for p in self.projection.parameters())
        pos_params = self.pos_embedding.weight.numel()
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())

        total_params = projection_params + pos_params + transformer_params + decoder_params

        return {
            "projection": projection_params,
            "positional_encoding": pos_params,
            "transformer": transformer_params,
            "decoder": decoder_params,
            "total": total_params,
        }


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation from GR00T to SAGE.

    Combines:
    - Task loss: Cross-entropy for grid prediction
    - Feature distillation: MSE + cosine similarity
    """

    def __init__(
        self,
        student_hidden_dim: int = 512,
        teacher_hidden_dim: int = 2048,
        task_weight: float = 1.0,
        feature_weight: float = 0.5,
    ):
        super().__init__()

        self.task_weight = task_weight
        self.feature_weight = feature_weight

        # Losses
        self.task_loss = nn.CrossEntropyLoss(ignore_index=-1)  # -1 for padding
        self.feature_loss = nn.MSELoss()

        # Projection to match GR00T dimension for distillation
        self.feature_proj = nn.Linear(student_hidden_dim, teacher_hidden_dim)

    def forward(
        self,
        grid_logits: torch.Tensor,         # [batch, 30, 30, 10]
        target_grid: torch.Tensor,         # [batch, 30, 30]
        student_features: torch.Tensor,    # [batch, seq_len, 512]
        groot_features: torch.Tensor,      # [batch, seq_len, 2048]
        attention_mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate combined distillation loss.

        Args:
            grid_logits: Student model predictions [batch, 30, 30, 10]
            target_grid: Ground truth grids [batch, 30, 30]
            student_features: Student representations [batch, seq_len, 512]
            groot_features: Teacher features [batch, seq_len, 2048]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual loss components
        """
        # Task loss (grid prediction)
        # Rearrange for CrossEntropyLoss: [batch, classes, height, width]
        task_loss = self.task_loss(
            grid_logits.permute(0, 3, 1, 2),  # [batch, 10, 30, 30]
            target_grid,                       # [batch, 30, 30]
        )

        # Feature distillation loss
        # Project student features to teacher dimension
        student_proj = self.feature_proj(student_features)  # [batch, seq_len, 2048]

        # MSE loss
        if attention_mask is not None:
            # Masked MSE: only compute loss on valid tokens
            mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            squared_diff = ((student_proj - groot_features) ** 2) * mask
            feature_mse = squared_diff.sum() / mask.sum()
        else:
            feature_mse = self.feature_loss(student_proj, groot_features)

        # Cosine similarity loss
        # Flatten to [batch*seq_len, 2048] for cosine similarity
        student_flat = student_proj.reshape(-1, 2048)
        groot_flat = groot_features.reshape(-1, 2048)

        if attention_mask is not None:
            # Only compute on valid tokens
            mask_flat = attention_mask.reshape(-1).bool()
            student_flat = student_flat[mask_flat]
            groot_flat = groot_flat[mask_flat]

        cos_sim = F.cosine_similarity(student_flat, groot_flat, dim=-1).mean()
        cosine_loss = 1 - cos_sim

        # Combined feature distillation loss
        feature_distill_loss = feature_mse + 0.5 * cosine_loss

        # Total weighted loss
        total_loss = (
            self.task_weight * task_loss +
            self.feature_weight * feature_distill_loss
        )

        # Return loss and components for logging
        loss_dict = {
            "total": total_loss.item(),
            "task": task_loss.item(),
            "feature_mse": feature_mse.item(),
            "cosine": cosine_loss.item(),
            "feature_distill": feature_distill_loss.item(),
        }

        return total_loss, loss_dict


def create_sage_student(
    hidden_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
) -> SAGEStudent:
    """
    Factory function to create SAGE student model with default config.

    Args:
        hidden_dim: Hidden dimension (default: 512)
        num_layers: Number of transformer layers (default: 6)
        num_heads: Number of attention heads (default: 8)

    Returns:
        model: Initialized SAGE student model
    """
    model = SAGEStudent(
        input_dim=2048,        # GR00T features
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=hidden_dim * 4,  # Standard 4x expansion
        max_seq_len=6000,
        grid_size=30,
        num_classes=10,
        dropout=0.1,
    )

    return model


if __name__ == "__main__":
    print("="*80)
    print("SAGE Student Model Test")
    print("="*80)

    # Create model
    model = create_sage_student(hidden_dim=512, num_layers=6, num_heads=8)

    # Print parameter counts
    param_counts = model.get_num_params()
    print(f"\n📊 Parameter Counts:")
    print(f"   Projection:    {param_counts['projection']:>10,} params")
    print(f"   Positional:    {param_counts['positional_encoding']:>10,} params")
    print(f"   Transformer:   {param_counts['transformer']:>10,} params")
    print(f"   Decoder:       {param_counts['decoder']:>10,} params")
    print(f"   {'─'*40}")
    print(f"   Total:         {param_counts['total']:>10,} params")
    print(f"   Size:          {param_counts['total'] / 1e6:.1f}M")

    # Test forward pass
    print(f"\n🔬 Testing forward pass...")
    batch_size = 2
    seq_len = 5164

    # Simulate GR00T features
    groot_features = torch.randn(batch_size, seq_len, 2048)
    attention_mask = torch.ones(batch_size, seq_len)

    # Forward pass
    grid_logits, student_features = model(groot_features, attention_mask)

    print(f"   Input shape:    {list(groot_features.shape)}")
    print(f"   Output logits:  {list(grid_logits.shape)}")
    print(f"   Student feats:  {list(student_features.shape)}")

    # Test prediction
    predicted_grid = model.predict_grid(groot_features, attention_mask)
    print(f"   Predicted grid: {list(predicted_grid.shape)}")
    print(f"   Grid values:    {predicted_grid[0, :5, :5]}")

    # Test loss
    print(f"\n🔬 Testing distillation loss...")
    loss_fn = DistillationLoss(task_weight=1.0, feature_weight=0.5)

    target_grid = torch.randint(0, 10, (batch_size, 30, 30))
    total_loss, loss_dict = loss_fn(
        grid_logits, target_grid, student_features, groot_features, attention_mask
    )

    print(f"   Loss components:")
    for name, value in loss_dict.items():
        print(f"      {name:15s}: {value:.4f}")

    print(f"\n✅ All tests passed!")
    print(f"   Model ready for training!")
