"""
Improved Training Objectives for SAGE
Fixes the Agent Zero problem by rewarding actual pattern solving, not pixel matching.

Problem: Current training optimizes pixel-level accuracy on sparse grids.
         This causes models to output all zeros (Agent Zero) since grids are 60-80% zeros.
Solution: Multi-objective training that rewards understanding and transformation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class PatternSolvingLoss(nn.Module):
    """
    Loss function that rewards actual pattern solving, not just pixel matching.
    Combines multiple objectives to prevent Agent Zero collapse.
    """
    
    def __init__(
        self,
        pixel_weight: float = 0.2,  # Reduce pixel accuracy weight
        structure_weight: float = 0.3,  # Reward structural understanding
        transformation_weight: float = 0.3,  # Reward correct transformations
        diversity_weight: float = 0.1,  # Penalize constant outputs
        consistency_weight: float = 0.1,  # Reward consistent patterns
    ):
        super().__init__()
        
        self.pixel_weight = pixel_weight
        self.structure_weight = structure_weight
        self.transformation_weight = transformation_weight
        self.diversity_weight = diversity_weight
        self.consistency_weight = consistency_weight
        
        # Component losses
        self.pixel_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss.
        
        Args:
            predictions: Model predictions [B, C, H, W] or [B, H, W, C]
            targets: Ground truth [B, H, W]
            inputs: Input grids (for transformation loss)
            mask: Valid regions mask
            
        Returns:
            Dictionary with total loss and components
        """
        batch_size = predictions.shape[0]
        
        # Ensure correct shape
        if predictions.dim() == 4 and predictions.shape[-1] != targets.shape[-1]:
            # [B, C, H, W] -> [B, H, W, C]
            predictions = predictions.permute(0, 2, 3, 1)
        
        losses = {}
        
        # 1. Pixel-level loss (reduced weight to prevent Agent Zero)
        pixel_loss = self._compute_pixel_loss(predictions, targets, mask)
        losses['pixel'] = pixel_loss * self.pixel_weight
        
        # 2. Structure preservation loss
        structure_loss = self._compute_structure_loss(predictions, targets)
        losses['structure'] = structure_loss * self.structure_weight
        
        # 3. Transformation consistency loss
        if inputs is not None:
            transform_loss = self._compute_transformation_loss(
                inputs, predictions, targets
            )
            losses['transformation'] = transform_loss * self.transformation_weight
        else:
            losses['transformation'] = torch.tensor(0.0, device=predictions.device)
        
        # 4. Diversity loss (penalize constant outputs)
        diversity_loss = self._compute_diversity_loss(predictions)
        losses['diversity'] = diversity_loss * self.diversity_weight
        
        # 5. Pattern consistency loss
        consistency_loss = self._compute_consistency_loss(predictions, targets)
        losses['consistency'] = consistency_loss * self.consistency_weight
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses
    
    def _compute_pixel_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Standard cross-entropy loss with optional masking."""
        B, H, W, C = predictions.shape
        
        # Reshape for cross-entropy
        preds_flat = predictions.reshape(-1, C)
        targets_flat = targets.reshape(-1).long()
        
        # Compute loss
        loss = self.pixel_loss(preds_flat, targets_flat)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.reshape(-1)
            loss = loss * mask_flat
            loss = loss.sum() / (mask_flat.sum() + 1e-6)
        else:
            loss = loss.mean()
        
        return loss
    
    def _compute_structure_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Reward preserving structural relationships.
        Uses edge detection and connected components.
        """
        # Get predicted classes
        pred_classes = predictions.argmax(dim=-1)
        
        # Compute edge maps
        pred_edges = self._detect_edges(pred_classes)
        target_edges = self._detect_edges(targets)
        
        # Edge similarity loss
        edge_loss = F.mse_loss(pred_edges.float(), target_edges.float())
        
        # Object count similarity (prevents outputting all zeros)
        pred_objects = self._count_objects(pred_classes)
        target_objects = self._count_objects(targets)
        
        count_loss = F.l1_loss(pred_objects, target_objects)
        
        return edge_loss + count_loss * 0.1
    
    def _compute_transformation_loss(
        self,
        inputs: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure the model learns transformations, not just memorization.
        """
        pred_classes = predictions.argmax(dim=-1)
        
        # Compute changes from input to output
        input_to_pred = (pred_classes != inputs).float()
        input_to_target = (targets != inputs).float()
        
        # Transformation should happen in similar regions
        transform_loss = F.mse_loss(input_to_pred, input_to_target)
        
        # Also check that SOME transformation happened (not identity)
        no_change_penalty = torch.relu(0.1 - input_to_pred.mean())
        
        return transform_loss + no_change_penalty
    
    def _compute_diversity_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Penalize constant outputs (Agent Zero behavior).
        Encourage using different colors/patterns.
        """
        # Get predicted classes
        pred_classes = predictions.argmax(dim=-1)
        
        # Compute entropy of predictions (higher = more diverse)
        B, H, W = pred_classes.shape
        
        # Spatial diversity (different values in different positions)
        spatial_std = pred_classes.float().std(dim=(1, 2))
        spatial_diversity = -torch.relu(1.0 - spatial_std).mean()
        
        # Color diversity (using multiple colors)
        unique_colors = []
        for b in range(B):
            unique = len(torch.unique(pred_classes[b]))
            unique_colors.append(unique)
        
        color_diversity = torch.tensor(unique_colors, device=predictions.device).float()
        color_penalty = -torch.relu(2.0 - color_diversity).mean()
        
        # Entropy of color distribution
        probs = F.softmax(predictions, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1).mean()
        entropy_bonus = -entropy  # Negative because we want high entropy
        
        return spatial_diversity + color_penalty * 0.1 + entropy_bonus * 0.01
    
    def _compute_consistency_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Reward consistent patterns within regions.
        """
        pred_classes = predictions.argmax(dim=-1)
        
        # Local consistency: neighboring pixels of same color should stay together
        pred_consistency = self._local_consistency(pred_classes)
        target_consistency = self._local_consistency(targets)
        
        return F.mse_loss(pred_consistency, target_consistency)
    
    def _detect_edges(self, grid: torch.Tensor) -> torch.Tensor:
        """Simple edge detection using gradients."""
        # Sobel-like edge detection
        dy = torch.abs(grid[:, 1:, :] - grid[:, :-1, :])
        dx = torch.abs(grid[:, :, 1:] - grid[:, :, :-1])
        
        # Pad to original size
        dy = F.pad(dy, (0, 0, 0, 1))
        dx = F.pad(dx, (0, 1, 0, 0))
        
        edges = (dy + dx) > 0
        return edges
    
    def _count_objects(self, grid: torch.Tensor) -> torch.Tensor:
        """Approximate object counting using connected components."""
        B = grid.shape[0]
        counts = []
        
        for b in range(B):
            # Simple approximation: count unique non-zero values
            unique_nonzero = len(torch.unique(grid[b])) - 1  # Subtract background
            counts.append(max(unique_nonzero, 0))
        
        return torch.tensor(counts, device=grid.device, dtype=torch.float32)
    
    def _local_consistency(self, grid: torch.Tensor) -> torch.Tensor:
        """Measure local consistency of patterns."""
        B, H, W = grid.shape
        
        # Check if neighbors have same value
        same_right = (grid[:, :, :-1] == grid[:, :, 1:]).float()
        same_down = (grid[:, :-1, :] == grid[:, 1:, :]).float()
        
        # Average consistency
        consistency = torch.cat([
            same_right.mean(dim=(1, 2), keepdim=True),
            same_down.mean(dim=(1, 2), keepdim=True)
        ], dim=1)
        
        return consistency.squeeze()


class ContrastivePatternLoss(nn.Module):
    """
    Contrastive loss that learns to distinguish between correct and incorrect patterns.
    Helps prevent Agent Zero by explicitly contrasting with all-zero outputs.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 1.0
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        anchor_features: torch.Tensor,
        positive_features: torch.Tensor,
        negative_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            anchor_features: Features from input
            positive_features: Features from correct output
            negative_features: Features from incorrect outputs (e.g., all zeros)
            
        Returns:
            Contrastive loss value
        """
        # Normalize features
        anchor = F.normalize(anchor_features, dim=-1)
        positive = F.normalize(positive_features, dim=-1)
        
        # Positive similarity (should be high)
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature
        
        if negative_features is not None:
            negative = F.normalize(negative_features, dim=-1)
            
            # Negative similarity (should be low)
            neg_sim = (anchor * negative).sum(dim=-1) / self.temperature
            
            # Contrastive loss
            loss = -torch.log(
                torch.exp(pos_sim) / 
                (torch.exp(pos_sim) + torch.exp(neg_sim))
            )
        else:
            # Just maximize positive similarity
            loss = -pos_sim
        
        return loss.mean()


class ReasoningRewardLoss(nn.Module):
    """
    Reward-based loss that gives positive feedback for correct reasoning steps.
    Uses curriculum learning to gradually increase task difficulty.
    """
    
    def __init__(
        self,
        base_reward: float = 1.0,
        step_penalty: float = 0.01,
        incorrect_penalty: float = 0.5
    ):
        super().__init__()
        self.base_reward = base_reward
        self.step_penalty = step_penalty
        self.incorrect_penalty = incorrect_penalty
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_steps: int = 1,
        is_correct: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reward-based loss.
        
        Args:
            predictions: Model outputs
            targets: Ground truth
            num_steps: Number of reasoning steps taken
            is_correct: Binary tensor indicating correctness
            
        Returns:
            Reward-based loss (negative reward)
        """
        if is_correct is None:
            # Compute correctness
            pred_classes = predictions.argmax(dim=-1)
            is_correct = (pred_classes == targets).all(dim=(1, 2))
        
        # Compute rewards
        rewards = torch.zeros(predictions.shape[0], device=predictions.device)
        
        # Correct solutions get base reward minus step penalty
        rewards[is_correct] = self.base_reward - self.step_penalty * num_steps
        
        # Incorrect solutions get penalty
        rewards[~is_correct] = -self.incorrect_penalty
        
        # Add partial credit for partially correct solutions
        if not is_correct.all():
            pred_classes = predictions.argmax(dim=-1)
            accuracy = (pred_classes == targets).float().mean(dim=(1, 2))
            partial_reward = accuracy * 0.5
            rewards[~is_correct] += partial_reward[~is_correct]
        
        # Return negative reward as loss
        return -rewards.mean()


class CombinedSAGELoss(nn.Module):
    """
    Combined loss function for SAGE training.
    Integrates all components to prevent Agent Zero collapse.
    """
    
    def __init__(
        self,
        use_pattern_solving: bool = True,
        use_contrastive: bool = True,
        use_reasoning_reward: bool = True,
        pattern_weight: float = 0.6,
        contrastive_weight: float = 0.2,
        reward_weight: float = 0.2
    ):
        super().__init__()
        
        self.use_pattern_solving = use_pattern_solving
        self.use_contrastive = use_contrastive
        self.use_reasoning_reward = use_reasoning_reward
        
        self.pattern_weight = pattern_weight
        self.contrastive_weight = contrastive_weight
        self.reward_weight = reward_weight
        
        # Initialize component losses
        if use_pattern_solving:
            self.pattern_loss = PatternSolvingLoss()
        
        if use_contrastive:
            self.contrastive_loss = ContrastivePatternLoss()
        
        if use_reasoning_reward:
            self.reward_loss = ReasoningRewardLoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[Dict[str, torch.Tensor]] = None,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            features: Dictionary of features for contrastive loss
            inputs: Input grids
            **kwargs: Additional arguments for component losses
            
        Returns:
            Dictionary with total loss and components
        """
        losses = {}
        total_loss = 0
        
        # Pattern solving loss
        if self.use_pattern_solving:
            pattern_losses = self.pattern_loss(
                predictions, targets, inputs
            )
            losses.update({f'pattern_{k}': v for k, v in pattern_losses.items()})
            total_loss += pattern_losses['total'] * self.pattern_weight
        
        # Contrastive loss
        if self.use_contrastive and features is not None:
            # Create negative examples (all zeros)
            batch_size = predictions.shape[0]
            zero_features = torch.zeros_like(features.get('anchor', predictions))
            
            contrastive = self.contrastive_loss(
                features.get('anchor', predictions.mean(dim=(1, 2))),
                features.get('positive', targets.float().mean(dim=(1, 2))),
                zero_features
            )
            losses['contrastive'] = contrastive
            total_loss += contrastive * self.contrastive_weight
        
        # Reasoning reward loss
        if self.use_reasoning_reward:
            reward = self.reward_loss(
                predictions, targets,
                num_steps=kwargs.get('num_steps', 1)
            )
            losses['reward'] = reward
            total_loss += reward * self.reward_weight
        
        losses['total'] = total_loss
        return losses


def create_sage_loss(
    loss_type: str = "combined",
    **kwargs
) -> nn.Module:
    """
    Factory function to create appropriate loss for SAGE.
    
    Args:
        loss_type: Type of loss ('combined', 'pattern', 'contrastive', 'reward')
        **kwargs: Additional arguments for the loss
        
    Returns:
        Loss module
    """
    if loss_type == "combined":
        return CombinedSAGELoss(**kwargs)
    elif loss_type == "pattern":
        return PatternSolvingLoss(**kwargs)
    elif loss_type == "contrastive":
        return ContrastivePatternLoss(**kwargs)
    elif loss_type == "reward":
        return ReasoningRewardLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    print("Testing Improved Training Objectives...")
    
    # Create sample data
    batch_size = 4
    height, width = 15, 15
    num_classes = 10
    
    # Predictions (with logits)
    predictions = torch.randn(batch_size, num_classes, height, width)
    
    # Targets (class indices)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Inputs (for transformation loss)
    inputs = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test PatternSolvingLoss
    pattern_loss = PatternSolvingLoss()
    losses = pattern_loss(predictions.permute(0, 2, 3, 1), targets, inputs)
    
    print("✅ Pattern Solving Loss:")
    for key, value in losses.items():
        print(f"   {key}: {value.item():.4f}")
    
    # Test CombinedSAGELoss
    combined_loss = CombinedSAGELoss()
    combined_losses = combined_loss(
        predictions.permute(0, 2, 3, 1),
        targets,
        inputs=inputs
    )
    
    print("\n✅ Combined SAGE Loss:")
    for key, value in combined_losses.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.item():.4f}")
    
    print("\n✅ Training objectives ready to prevent Agent Zero!")