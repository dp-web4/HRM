"""
SAGE Training Pipeline

Main training script for the 100M parameter attention orchestrator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Optional, Tuple, List
import os
import json
import time
from dataclasses import dataclass
import logging
from tqdm import tqdm

import sys
sys.path.append('..')
from core.sage_core import SAGECore
from core.sage_config import SAGEConfig, SAGEPresets
from attention.snarc_scorer import SNARCScorer


@dataclass
class TrainingMetrics:
    """Track training metrics"""
    loss: float = 0.0
    task_loss: float = 0.0
    attention_loss: float = 0.0
    snarc_loss: float = 0.0
    halt_loss: float = 0.0
    diversity_loss: float = 0.0
    accuracy: float = 0.0
    unique_predictions: float = 0.0
    avg_cycles: float = 0.0


class SAGELoss(nn.Module):
    """Multi-component loss function for SAGE training
    
    Combines:
    - Task performance loss
    - Attention alignment loss
    - SNARC prediction loss
    - Halt efficiency loss
    - Diversity enforcement
    """
    
    def __init__(self, config: SAGEConfig):
        super().__init__()
        self.config = config
        self.task_criterion = nn.CrossEntropyLoss()
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        snarc_targets: Optional[torch.Tensor] = None,
        attention_targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, TrainingMetrics]:
        """Compute multi-component loss
        
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth labels [batch, seq]
            snarc_targets: Optional SNARC ground truth
            attention_targets: Optional attention ground truth
            
        Returns:
            total_loss: Combined loss scalar
            metrics: Detailed metrics
        """
        metrics = TrainingMetrics()
        
        # 1. Task performance loss (primary objective)
        predictions = outputs['output']  # [batch, seq, classes]
        batch_size, seq_len, num_classes = predictions.shape
        
        task_loss = self.task_criterion(
            predictions.reshape(-1, num_classes),
            targets.reshape(-1)
        )
        metrics.task_loss = task_loss.item()
        
        # 2. Attention alignment loss (if ground truth available)
        attention_loss = torch.tensor(0.0, device=predictions.device)
        if attention_targets is not None and 'attention_weights' in outputs:
            attention_loss = F.mse_loss(
                outputs['attention_weights'],
                attention_targets
            )
            metrics.attention_loss = attention_loss.item()
        
        # 3. SNARC prediction loss (if ground truth available)
        snarc_loss = torch.tensor(0.0, device=predictions.device)
        if snarc_targets is not None and 'snarc_scores' in outputs:
            snarc_loss = F.mse_loss(
                outputs['snarc_scores'],
                snarc_targets
            )
            metrics.snarc_loss = snarc_loss.item()
        
        # 4. Halt efficiency loss (encourage reasonable cycles)
        halt_loss = torch.tensor(0.0, device=predictions.device)
        if 'halt_probs' in outputs:
            # Penalize too early (<3) or too late (>6) halting
            num_cycles = outputs.get('num_cycles_used', 8)
            if num_cycles < 3:
                halt_loss = (3 - num_cycles) * 0.1
            elif num_cycles > 6:
                halt_loss = (num_cycles - 6) * 0.1
            metrics.halt_loss = halt_loss
            metrics.avg_cycles = float(num_cycles)
        
        # 5. Diversity loss (prevent mode collapse)
        diversity_loss = torch.tensor(0.0, device=predictions.device)
        pred_classes = predictions.argmax(dim=-1)  # [batch, seq]
        for b in range(batch_size):
            unique_preds = len(torch.unique(pred_classes[b]))
            if unique_preds < 2:  # Too few unique predictions
                diversity_loss += (2 - unique_preds) * 0.5
            metrics.unique_predictions += unique_preds
        metrics.unique_predictions /= batch_size
        metrics.diversity_loss = diversity_loss.item()
        
        # Combine losses with weights
        total_loss = (
            task_loss * 1.0 +
            attention_loss * 0.3 +
            snarc_loss * 0.2 +
            halt_loss * 0.1 +
            diversity_loss * 0.1
        )
        
        # Calculate accuracy
        correct = (pred_classes == targets).float().sum()
        total = targets.numel()
        metrics.accuracy = (correct / total).item()
        
        metrics.loss = total_loss.item()
        return total_loss, metrics


class SyntheticAttentionDataset(Dataset):
    """Synthetic dataset for initial SAGE training
    
    Generates attention-based puzzles with known solutions.
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        seq_len: int = 100,
        num_classes: int = 10,
        context_dim: int = 256
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.context_dim = context_dim
        
        # Pre-generate dataset
        self.data = []
        for _ in range(num_samples):
            self.data.append(self._generate_sample())
    
    def _generate_sample(self) -> Dict[str, torch.Tensor]:
        """Generate a synthetic attention puzzle"""
        # Random input sequence
        input_ids = torch.randint(0, self.num_classes, (self.seq_len,))
        
        # Generate pattern-based targets
        pattern_type = np.random.choice(['copy', 'reverse', 'shift', 'filter'])
        
        if pattern_type == 'copy':
            targets = input_ids.clone()
        elif pattern_type == 'reverse':
            targets = torch.flip(input_ids, [0])
        elif pattern_type == 'shift':
            shift = np.random.randint(1, 5)
            targets = torch.roll(input_ids, shift)
        else:  # filter
            mask = torch.rand(self.seq_len) > 0.5
            targets = input_ids.clone()
            targets[mask] = 0
        
        # Generate context based on pattern type
        context = torch.randn(self.context_dim)
        context[0] = {'copy': 0, 'reverse': 1, 'shift': 2, 'filter': 3}[pattern_type]
        
        # Generate attention ground truth (attend to important positions)
        attention_gt = torch.rand(self.seq_len, 1)
        if pattern_type == 'filter':
            attention_gt[mask] = 1.0
        
        return {
            'input_ids': input_ids,
            'targets': targets,
            'context': context,
            'attention_gt': attention_gt,
            'pattern_type': pattern_type
        }
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class ARCDataset(Dataset):
    """ARC dataset loader for SAGE training"""
    
    def __init__(self, data_dir: str = "data/arc-1-aug-500", max_samples: Optional[int] = None):
        self.data_dir = data_dir
        self.samples = []
        
        # Load ARC data
        if os.path.exists(data_dir):
            # Load from preprocessed format
            data_file = os.path.join(data_dir, "train.json")
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    self.samples = data['samples'][:max_samples] if max_samples else data['samples']
        
        if not self.samples:
            # Generate dummy data for testing
            print(f"Warning: No ARC data found at {data_dir}, using dummy data")
            self.samples = [self._generate_dummy() for _ in range(100)]
    
    def _generate_dummy(self) -> Dict:
        """Generate dummy ARC-like data"""
        grid_size = 30
        input_grid = torch.randint(0, 10, (grid_size, grid_size))
        output_grid = torch.randint(0, 10, (grid_size, grid_size))
        
        return {
            'input': input_grid.numpy().tolist(),
            'output': output_grid.numpy().tolist(),
            'metadata': {'task_id': 'dummy', 'difficulty': 0.5}
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        input_grid = torch.tensor(sample['input'], dtype=torch.long).flatten()
        output_grid = torch.tensor(sample['output'], dtype=torch.long).flatten()
        
        # Generate context from metadata
        context = torch.randn(256)  # Placeholder - should encode task properties
        
        return {
            'input_ids': input_grid,
            'targets': output_grid,
            'context': context,
            'attention_gt': torch.ones(input_grid.size(0), 1) * 0.5  # Uniform attention as placeholder
        }


class SAGETrainer:
    """Training orchestrator for SAGE"""
    
    def __init__(
        self,
        config: SAGEConfig,
        model: Optional[SAGECore] = None,
        device: Optional[str] = None
    ):
        self.config = config
        self.device = device or config.device
        
        # Initialize model
        self.model = model or SAGECore(config)
        self.model = self.model.to(self.device)
        
        # Initialize components
        self.snarc_scorer = SNARCScorer(config.hidden_size).to(self.device)
        self.loss_fn = SAGELoss(config)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=config.adam_epsilon
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100000,
            eta_min=config.learning_rate * 0.1
        )
        
        # Metrics tracking
        self.metrics_history = []
        self.best_loss = float('inf')
        self.global_step = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        targets = batch['targets'].to(self.device)
        context = batch.get('context', None)
        if context is not None:
            context = context.to(self.device)
        
        attention_gt = batch.get('attention_gt', None)
        if attention_gt is not None:
            attention_gt = attention_gt.to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, context)
        
        # Compute SNARC scores
        with torch.no_grad():
            h_states = outputs['h_states']
            snarc_results = self.snarc_scorer(h_states, context)
            outputs['snarc_scores'] = snarc_results['snarc_scores']
            outputs['attention_weights'] = snarc_results['attention_weights']
        
        # Compute loss
        loss, metrics = self.loss_fn(outputs, targets, attention_targets=attention_gt)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> TrainingMetrics:
        """Validation pass"""
        self.model.eval()
        total_metrics = TrainingMetrics()
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                context = batch.get('context', None)
                if context is not None:
                    context = context.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, context)
                
                # Compute loss
                _, metrics = self.loss_fn(outputs, targets)
                
                # Accumulate metrics
                total_metrics.loss += metrics.loss
                total_metrics.accuracy += metrics.accuracy
                total_metrics.unique_predictions += metrics.unique_predictions
                total_metrics.avg_cycles += metrics.avg_cycles
                num_batches += 1
        
        # Average metrics
        if num_batches > 0:
            total_metrics.loss /= num_batches
            total_metrics.accuracy /= num_batches
            total_metrics.unique_predictions /= num_batches
            total_metrics.avg_cycles /= num_batches
        
        return total_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10
    ):
        """Main training loop"""
        self.logger.info(f"Starting SAGE training for {num_epochs} epochs")
        self.logger.info(f"Model parameters: {self.model.get_num_params():,}")
        
        for epoch in range(num_epochs):
            epoch_metrics = TrainingMetrics()
            num_batches = 0
            
            # Training epoch
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                metrics = self.train_step(batch)
                
                # Accumulate metrics
                epoch_metrics.loss += metrics.loss
                epoch_metrics.accuracy += metrics.accuracy
                epoch_metrics.unique_predictions += metrics.unique_predictions
                epoch_metrics.avg_cycles += metrics.avg_cycles
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics.loss:.4f}",
                    'acc': f"{metrics.accuracy:.2%}",
                    'unique': f"{metrics.unique_predictions:.1f}",
                    'cycles': f"{metrics.avg_cycles:.1f}"
                })
                
                self.global_step += 1
                
                # Log periodically
                if self.global_step % self.config.log_interval == 0:
                    self.logger.info(
                        f"Step {self.global_step}: "
                        f"loss={metrics.loss:.4f}, "
                        f"acc={metrics.accuracy:.2%}, "
                        f"unique={metrics.unique_predictions:.1f}"
                    )
                
                # Save checkpoint periodically
                if self.global_step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
            
            # Average epoch metrics
            if num_batches > 0:
                epoch_metrics.loss /= num_batches
                epoch_metrics.accuracy /= num_batches
                epoch_metrics.unique_predictions /= num_batches
                epoch_metrics.avg_cycles /= num_batches
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.logger.info(
                    f"Epoch {epoch+1} - Validation: "
                    f"loss={val_metrics.loss:.4f}, "
                    f"acc={val_metrics.accuracy:.2%}"
                )
                
                # Save best model
                if val_metrics.loss < self.best_loss:
                    self.best_loss = val_metrics.loss
                    self.save_checkpoint("best")
            
            # Log epoch summary
            self.logger.info(
                f"Epoch {epoch+1} complete: "
                f"loss={epoch_metrics.loss:.4f}, "
                f"acc={epoch_metrics.accuracy:.2%}, "
                f"unique={epoch_metrics.unique_predictions:.1f}, "
                f"cycles={epoch_metrics.avg_cycles:.1f}"
            )
            
            self.metrics_history.append(epoch_metrics)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"sage_{name}.pt")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'global_step': self.global_step,
            'metrics_history': self.metrics_history
        }, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.metrics_history = checkpoint.get('metrics_history', [])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")


def main():
    """Main training script"""
    # Configuration
    config = SAGEConfig()
    
    # Create datasets
    print("Creating synthetic dataset...")
    train_dataset = SyntheticAttentionDataset(num_samples=1000)
    val_dataset = SyntheticAttentionDataset(num_samples=100)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # Initialize trainer
    print("Initializing SAGE trainer...")
    trainer = SAGETrainer(config)
    
    # Start training
    print("Starting training...")
    trainer.train(train_loader, val_loader, num_epochs=2)  # Short test run
    
    print("Training complete!")


if __name__ == "__main__":
    main()