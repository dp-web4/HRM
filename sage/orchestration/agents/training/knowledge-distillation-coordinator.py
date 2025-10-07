#!/usr/bin/env python3
"""
Knowledge Distillation Coordinator
Manages the distillation of GR00T knowledge into SAGE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import time
import sys

# Add paths
sys.path.append("/home/dp/ai-workspace/HRM/sage/orchestration/agents/training")
sys.path.append("/home/dp/ai-workspace/HRM/sage/orchestration/agents/vision")

# Import with proper module name
import importlib.util
spec = importlib.util.spec_from_file_location(
    "groot_data_processor", 
    "/home/dp/ai-workspace/HRM/sage/orchestration/agents/training/groot-data-processor.py"
)
groot_data_processor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(groot_data_processor)
GR00TDataProcessor = groot_data_processor.GR00TDataProcessor


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation"""
    teacher_model: str = "groot-n1.5"
    student_model: str = "sage"
    temperature: float = 3.0
    alpha: float = 0.7  # Weight for distillation loss
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 100
    checkpoint_interval: int = 10
    device: str = "cuda"


class SAGEStudent(nn.Module):
    """
    Simplified SAGE student model for distillation
    Learns to mimic GR00T's behavior
    """
    
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 512, 
                 action_dim: int = 7, state_dim: int = 14):
        super().__init__()
        
        # Visual processing
        self.visual_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # State processing
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Combined processing
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Action prediction
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # Value prediction (for RL)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Latent representation (for distillation)
        self.latent_dim = hidden_dim // 2
        
    def forward(self, visual_features: torch.Tensor, 
                states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Returns: (actions, values, latent_features)
        """
        # Encode inputs
        visual_encoded = self.visual_encoder(visual_features)
        state_encoded = self.state_encoder(states)
        
        # Combine features
        combined = torch.cat([visual_encoded, state_encoded], dim=-1)
        latent_features = self.fusion(combined)
        
        # Predict outputs
        actions = self.action_head(latent_features)
        values = self.value_head(latent_features)
        
        return actions, values, latent_features


class KnowledgeDistillationCoordinator:
    """
    Coordinates knowledge distillation from GR00T to SAGE
    """
    
    def __init__(self, config: DistillationConfig = None):
        self.config = config or DistillationConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        
        # Models
        self.teacher_model = None  # Would be actual GR00T model
        self.student_model = SAGEStudent().to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.student_model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Data processor
        self.data_processor = GR00TDataProcessor({
            "batch_size": self.config.batch_size,
            "episode_count": 5
        })
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        print(f"🎓 Knowledge Distillation Coordinator initialized")
        print(f"   Teacher: {self.config.teacher_model}")
        print(f"   Student: {self.config.student_model}")
        print(f"   Temperature: {self.config.temperature}")
        print(f"   Alpha: {self.config.alpha}")
        print(f"   Device: {self.device}")
    
    def distillation_loss(self, student_logits: torch.Tensor, 
                         teacher_logits: torch.Tensor,
                         temperature: float) -> torch.Tensor:
        """
        Compute distillation loss between student and teacher
        """
        # Soften probabilities with temperature
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        
        # KL divergence loss
        loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        loss = loss * (temperature ** 2)  # Scale by T^2
        
        return loss
    
    def get_teacher_predictions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get teacher model predictions
        In real implementation, would use actual GR00T model
        """
        # Mock teacher predictions for now
        batch_size = batch["visual_features"].shape[0]
        teacher_actions = batch["actions"]  # Use ground truth as teacher
        
        # Add some noise to simulate imperfect teacher
        noise = torch.randn_like(teacher_actions) * 0.01
        teacher_actions = teacher_actions + noise
        
        return teacher_actions
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step
        """
        # Move batch to device
        visual_features = batch["visual_features"].to(self.device)
        states = batch["states"].to(self.device)
        true_actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        
        # Get teacher predictions
        teacher_actions = self.get_teacher_predictions(batch).to(self.device)
        
        # Student forward pass
        student_actions, student_values, student_latent = self.student_model(
            visual_features, states
        )
        
        # Compute losses
        # 1. Distillation loss (soft targets from teacher)
        distill_loss = self.distillation_loss(
            student_actions, teacher_actions, self.config.temperature
        )
        
        # 2. Hard target loss (ground truth actions)
        hard_loss = F.mse_loss(student_actions, true_actions)
        
        # 3. Value loss (for RL)
        value_loss = F.mse_loss(student_values.squeeze(), rewards)
        
        # Combined loss
        total_loss = (self.config.alpha * distill_loss + 
                     (1 - self.config.alpha) * hard_loss +
                     0.1 * value_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Return losses for logging
        return {
            "total_loss": total_loss.item(),
            "distill_loss": distill_loss.item(),
            "hard_loss": hard_loss.item(),
            "value_loss": value_loss.item()
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        """
        self.student_model.train()
        epoch_losses = []
        
        # Process episodes
        if not self.data_processor.processed_features:
            self.data_processor.process_all_episodes()
        
        # Generate batches
        num_batches = 10  # Mock number of batches
        for batch_idx in range(num_batches):
            batch = self.data_processor.create_training_batch()
            losses = self.train_step(batch)
            epoch_losses.append(losses)
            
            self.global_step += 1
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/{num_batches}: "
                      f"loss={losses['total_loss']:.4f}")
        
        # Average losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in epoch_losses])
        
        return avg_losses
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate student model
        """
        self.student_model.eval()
        eval_losses = []
        
        with torch.no_grad():
            # Generate evaluation batches
            for _ in range(5):
                batch = self.data_processor.create_training_batch()
                
                # Move to device
                visual_features = batch["visual_features"].to(self.device)
                states = batch["states"].to(self.device)
                true_actions = batch["actions"].to(self.device)
                
                # Forward pass
                student_actions, _, _ = self.student_model(visual_features, states)
                
                # Compute action error
                action_error = F.mse_loss(student_actions, true_actions)
                eval_losses.append(action_error.item())
        
        return {
            "eval_loss": np.mean(eval_losses),
            "eval_std": np.std(eval_losses)
        }
    
    def train(self, num_epochs: Optional[int] = None):
        """
        Main training loop
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        print(f"\n🚀 Starting knowledge distillation for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            print(f"\n📚 Epoch {self.epoch}/{num_epochs}")
            
            # Train
            train_losses = self.train_epoch()
            
            # Evaluate
            eval_metrics = self.evaluate()
            
            # Log results
            results = {
                "epoch": self.epoch,
                **train_losses,
                **eval_metrics
            }
            self.training_history.append(results)
            
            print(f"  Train loss: {train_losses['total_loss']:.4f} "
                  f"(distill={train_losses['distill_loss']:.4f}, "
                  f"hard={train_losses['hard_loss']:.4f})")
            print(f"  Eval loss: {eval_metrics['eval_loss']:.4f} "
                  f"(±{eval_metrics['eval_std']:.4f})")
            
            # Save checkpoint if improved
            if eval_metrics['eval_loss'] < self.best_loss:
                self.best_loss = eval_metrics['eval_loss']
                self.save_checkpoint()
                print(f"  ✅ New best model saved (loss: {self.best_loss:.4f})")
            
            # Early stopping for demo
            if epoch >= 2:  # Stop after 3 epochs for testing
                print("\n🛑 Early stopping for demo")
                break
    
    def save_checkpoint(self, path: Optional[str] = None):
        """
        Save model checkpoint
        """
        if path is None:
            path = f"/home/dp/ai-workspace/HRM/sage/orchestration/checkpoints/sage_distilled_epoch_{self.epoch}.pt"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.student_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config.__dict__,
            "training_history": self.training_history
        }
        
        torch.save(checkpoint, path)
        print(f"  💾 Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.student_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        self.training_history = checkpoint["training_history"]
        
        print(f"  📂 Loaded checkpoint from epoch {self.epoch}")


def main():
    """Test the Knowledge Distillation Coordinator"""
    print("🧪 Testing Knowledge Distillation Coordinator")
    print("=" * 50)
    
    # Create configuration
    config = DistillationConfig(
        temperature=3.0,
        alpha=0.7,
        learning_rate=1e-4,
        batch_size=8,
        num_epochs=10
    )
    
    # Create coordinator
    coordinator = KnowledgeDistillationCoordinator(config)
    
    # Load and process data
    print("\n📊 Loading GR00T demonstration data...")
    coordinator.data_processor.load_episodes()
    
    # Run training
    coordinator.train(num_epochs=3)  # Short training for demo
    
    # Display results
    print("\n📈 Training History:")
    for entry in coordinator.training_history:
        print(f"  Epoch {entry['epoch']}: "
              f"train_loss={entry['total_loss']:.4f}, "
              f"eval_loss={entry['eval_loss']:.4f}")
    
    print(f"\n🏆 Best evaluation loss: {coordinator.best_loss:.4f}")
    print("\n✅ Knowledge Distillation test complete!")


if __name__ == "__main__":
    main()