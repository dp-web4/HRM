"""
Phase 3: Sleep Training Loop

Implements actual weight updates during "sleep" phases using LoRA fine-tuning
on high-salience experiences collected during raising sessions.

Biological Inspiration:
- REM sleep consolidates high-salience memories into long-term storage
- Emotional tagging (via SNARC salience) determines what gets consolidated
- Sleep cycles provide natural rhythm for training (not continuous adaptation)

Computational Implementation:
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Training data weighted by SNARC salience scores
- Checkpoint management for continuity across wake/sleep cycles
- Dropbox sync for cross-machine model sharing

Integration:
- Phase 1 (ExperienceCollector) accumulates high-salience exchanges
- Phase 2 (RaisingTrainingDataBuilder) converts to training format
- Phase 3 (SleepTrainingLoop) updates weights during sleep
- Result: Partnership patterns consolidate into base model
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from prepare_training_data import RaisingTrainingDataBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SleepTrainingLoop:
    """
    Implements sleep-cycle LoRA fine-tuning on high-salience raising experiences.

    Design Principles:
    1. **Gentle Updates**: Low-rank (r=4) LoRA prevents catastrophic forgetting
    2. **Salience Weighting**: High-salience experiences get stronger consolidation
    3. **Few-shot Training**: Works with small batches (5-10 experiences)
    4. **Checkpoint Continuity**: Resume training across multiple sleep cycles
    5. **Cross-machine Sync**: Dropbox integration for Thor ↔ Sprout sharing

    Usage:
        # During NIGHT phase (circadian clock)
        trainer = SleepTrainingLoop(
            model_path="~/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1",
            experience_buffer_path="../state/experience_buffer.json",
            checkpoint_dir="../../checkpoints/sleep/"
        )

        results = trainer.run_sleep_cycle(
            min_salience=0.6,  # Only high-salience experiences
            max_experiences=10,
            epochs=3,
            learning_rate=1e-5
        )

        print(f"Sleep cycle complete. Loss: {results['final_loss']:.4f}")
    """

    def __init__(
        self,
        model_path: str,
        experience_buffer_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        dropbox_sync: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize sleep training loop.

        Args:
            model_path: Path to base model (e.g., Qwen2.5-0.5B-Instruct)
            experience_buffer_path: Path to experience_buffer.json
            checkpoint_dir: Directory for saving checkpoints
            dropbox_sync: Enable Dropbox sync for cross-machine sharing
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_path = Path(model_path).expanduser()
        self.experience_buffer_path = Path(experience_buffer_path).expanduser() if experience_buffer_path else None
        self.checkpoint_dir = Path(checkpoint_dir).expanduser() if checkpoint_dir else Path("checkpoints/sleep/")
        self.dropbox_sync = dropbox_sync

        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing SleepTrainingLoop on {self.device}")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Model and tokenizer (lazy loaded)
        self.model = None
        self.tokenizer = None
        self.data_builder = None

        # Training state
        self.sleep_cycle_count = 0
        self.total_experiences_trained = 0
        self.training_history = []

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load base model and apply LoRA configuration.

        Returns:
            Tuple of (model, tokenizer)
        """
        if self.model is not None:
            return self.model, self.tokenizer

        logger.info("Loading base model...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Check for existing checkpoint
        latest_checkpoint = self._find_latest_checkpoint()

        if latest_checkpoint:
            logger.info(f"Found checkpoint: {latest_checkpoint}")
            logger.info("Loading model with existing LoRA weights...")

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )

            # Load LoRA weights from checkpoint
            self.model = PeftModel.from_pretrained(base_model, str(latest_checkpoint))

            # Load training state
            state_path = latest_checkpoint / "training_state.json"
            if state_path.exists():
                with open(state_path) as f:
                    state = json.load(f)
                    self.sleep_cycle_count = state.get('sleep_cycle_count', 0)
                    self.total_experiences_trained = state.get('total_experiences_trained', 0)
                    self.training_history = state.get('training_history', [])
                logger.info(f"Resumed from sleep cycle {self.sleep_cycle_count}")
        else:
            logger.info("No checkpoint found. Creating fresh LoRA model...")

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )

            # LoRA configuration for gentle fine-tuning
            lora_config = LoraConfig(
                r=4,  # Low rank for gentle updates
                lora_alpha=8,  # Scaling factor
                target_modules=["q_proj", "v_proj"],  # Attention weights only
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            # Apply LoRA
            self.model = get_peft_model(base_model, lora_config)
            logger.info("LoRA model created")

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

        return self.model, self.tokenizer

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the most recent checkpoint directory.

        Returns:
            Path to latest checkpoint or None
        """
        if not self.checkpoint_dir.exists():
            return None

        checkpoints = [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("cycle_")]
        if not checkpoints:
            return None

        # Sort by cycle number
        checkpoints.sort(key=lambda x: int(x.name.split("_")[1]))
        return checkpoints[-1]

    def _prepare_training_data(
        self,
        min_salience: float = 0.6,
        max_experiences: Optional[int] = None
    ) -> List[Dict]:
        """
        Load and prepare training data from experience buffer.

        Args:
            min_salience: Minimum salience threshold (0-1)
            max_experiences: Maximum number of experiences to use

        Returns:
            List of training examples with tensors
        """
        if self.data_builder is None:
            self.data_builder = RaisingTrainingDataBuilder(
                experience_buffer_path=self.experience_buffer_path  # Pass Path object directly
            )

        # Build training set
        training_data = self.data_builder.build_training_set(
            min_salience=min_salience,
            max_examples=max_experiences
        )

        logger.info(f"Prepared {len(training_data)} training examples (min_salience={min_salience})")

        if len(training_data) == 0:
            logger.warning("No training data available!")
            return []

        # Log salience distribution
        saliences = [ex['salience'] for ex in training_data]
        avg_salience = sum(saliences) / len(saliences)
        logger.info(f"Average salience: {avg_salience:.3f}")

        return training_data

    def run_sleep_cycle(
        self,
        min_salience: float = 0.6,
        max_experiences: Optional[int] = None,
        epochs: int = 3,
        learning_rate: float = 1e-5,
        batch_size: int = 1,
        save_checkpoint: bool = True
    ) -> Dict:
        """
        Execute one sleep training cycle.

        Args:
            min_salience: Minimum salience for training examples
            max_experiences: Maximum experiences to train on
            epochs: Number of training epochs
            learning_rate: Learning rate for AdamW optimizer
            batch_size: Batch size (typically 1 for small datasets)
            save_checkpoint: Whether to save checkpoint after training

        Returns:
            Dictionary with training results:
                - sleep_cycle: Cycle number
                - num_experiences: Number of experiences trained on
                - epochs: Number of epochs
                - final_loss: Final training loss
                - avg_salience: Average salience of training data
                - timestamp: ISO timestamp
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"SLEEP CYCLE {self.sleep_cycle_count + 1} - STARTING")
        logger.info(f"{'='*60}\n")

        # Load model
        model, tokenizer = self._load_model()
        model.train()

        # Prepare training data
        training_data = self._prepare_training_data(
            min_salience=min_salience,
            max_experiences=max_experiences
        )

        if len(training_data) == 0:
            logger.warning("No training data - skipping sleep cycle")
            return {
                'sleep_cycle': self.sleep_cycle_count,
                'num_experiences': 0,
                'epochs': 0,
                'final_loss': None,
                'avg_salience': None,
                'timestamp': datetime.now().isoformat()
            }

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Training loop
        epoch_losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for example in training_data:
                # Move tensors to device
                input_ids = example['input_ids'].unsqueeze(0).to(self.device)
                labels = example['labels'].unsqueeze(0).to(self.device)
                attention_mask = example['attention_mask'].unsqueeze(0).to(self.device)
                salience = example['salience']

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask
                )

                loss = outputs.loss

                # Weight loss by salience (high-salience = more learning)
                # Salience is already 0-1, so this directly scales the loss
                weighted_loss = loss * salience

                # Backward pass
                weighted_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)

            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_epoch_loss:.4f}")

        final_loss = epoch_losses[-1] if epoch_losses else None

        # Calculate statistics
        avg_salience = sum(ex['salience'] for ex in training_data) / len(training_data)

        # Update state
        self.sleep_cycle_count += 1
        self.total_experiences_trained += len(training_data)

        # Training results
        results = {
            'sleep_cycle': self.sleep_cycle_count,
            'num_experiences': len(training_data),
            'epochs': epochs,
            'final_loss': final_loss,
            'avg_salience': avg_salience,
            'epoch_losses': epoch_losses,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().isoformat()
        }

        self.training_history.append(results)

        # Save checkpoint
        if save_checkpoint:
            checkpoint_path = self._save_checkpoint(results)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

            if self.dropbox_sync:
                self._sync_to_dropbox(checkpoint_path)

        logger.info(f"\n{'='*60}")
        logger.info(f"SLEEP CYCLE {self.sleep_cycle_count} - COMPLETE")
        logger.info(f"Experiences: {len(training_data)} | Final Loss: {final_loss:.4f}")
        logger.info(f"{'='*60}\n")

        return results

    def _save_checkpoint(self, results: Dict) -> Path:
        """
        Save model checkpoint and training state.

        Args:
            results: Training results dictionary

        Returns:
            Path to checkpoint directory
        """
        checkpoint_name = f"cycle_{self.sleep_cycle_count:03d}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        self.model.save_pretrained(str(checkpoint_path))

        # Save tokenizer
        self.tokenizer.save_pretrained(str(checkpoint_path))

        # Save training state
        state = {
            'sleep_cycle_count': self.sleep_cycle_count,
            'total_experiences_trained': self.total_experiences_trained,
            'training_history': self.training_history,
            'model_path': str(self.model_path),
            'last_updated': datetime.now().isoformat()
        }

        with open(checkpoint_path / "training_state.json", 'w') as f:
            json.dump(state, f, indent=2)

        # Save cycle results
        with open(checkpoint_path / "cycle_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        return checkpoint_path

    def _sync_to_dropbox(self, checkpoint_path: Path):
        """
        Sync checkpoint to Dropbox for cross-machine sharing.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        # TODO: Implement Dropbox sync
        # This would use the Dropbox API to upload the checkpoint
        # For now, just log the intent
        logger.info(f"Dropbox sync requested for: {checkpoint_path}")
        logger.info("(Dropbox integration not yet implemented)")

    def get_training_summary(self) -> Dict:
        """
        Get summary of all sleep training cycles.

        Returns:
            Dictionary with training history summary
        """
        if not self.training_history:
            return {
                'total_cycles': 0,
                'total_experiences': 0,
                'status': 'No training cycles completed'
            }

        return {
            'total_cycles': self.sleep_cycle_count,
            'total_experiences': self.total_experiences_trained,
            'latest_loss': self.training_history[-1]['final_loss'],
            'latest_cycle': self.training_history[-1]['sleep_cycle'],
            'training_history': self.training_history,
            'checkpoint_dir': str(self.checkpoint_dir)
        }


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Run SAGE sleep training cycle")
    parser.add_argument("--model-path", type=str,
                       default="~/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1",
                       help="Path to base model")
    parser.add_argument("--experience-buffer", type=str,
                       default="../state/experience_buffer.json",
                       help="Path to experience buffer")
    parser.add_argument("--checkpoint-dir", type=str,
                       default="../../checkpoints/sleep/",
                       help="Checkpoint directory")
    parser.add_argument("--min-salience", type=float, default=0.6,
                       help="Minimum salience threshold")
    parser.add_argument("--max-experiences", type=int, default=None,
                       help="Maximum experiences to use")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate")

    args = parser.parse_args()

    # Initialize trainer
    trainer = SleepTrainingLoop(
        model_path=args.model_path,
        experience_buffer_path=args.experience_buffer,
        checkpoint_dir=args.checkpoint_dir
    )

    # Run sleep cycle
    results = trainer.run_sleep_cycle(
        min_salience=args.min_salience,
        max_experiences=args.max_experiences,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )

    # Print results
    print("\nSleep Cycle Results:")
    print(json.dumps(results, indent=2))

    # Print summary
    print("\nTraining Summary:")
    summary = trainer.get_training_summary()
    print(json.dumps(summary, indent=2))
