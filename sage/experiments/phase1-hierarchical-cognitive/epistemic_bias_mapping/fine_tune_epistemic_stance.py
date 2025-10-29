#!/usr/bin/env python3
"""
Fine-Tune Epistemic Stance - Preference-Weighted Training

Train Qwen2.5-0.5B to shift from performative patterns to epistemic pragmatism
using preference-weighted supervised fine-tuning on contrastive pairs.

Simpler and more stable than full DPO - maximize likelihood of good responses
while minimizing likelihood of bad responses.

Usage:
    python fine_tune_epistemic_stance.py --epochs 200 --checkpoint-every 10
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import argparse
import time

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ContrastivePairDataset(Dataset):
    """Dataset of contrastive (bad, good) response pairs"""

    def __init__(self, pairs: List[Dict], tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Format as chat with system prompt
        system_prompt = "Provide the most honest answers you can."

        # Bad response (rejected)
        bad_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["bad_response"]}
        ]

        # Good response (chosen)
        good_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["good_response"]}
        ]

        # Tokenize
        bad_text = self.tokenizer.apply_chat_template(
            bad_messages, tokenize=False, add_generation_prompt=False
        )
        good_text = self.tokenizer.apply_chat_template(
            good_messages, tokenize=False, add_generation_prompt=False
        )

        bad_tokens = self.tokenizer(
            bad_text, max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt"
        )
        good_tokens = self.tokenizer(
            good_text, max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt"
        )

        return {
            "question": pair["question"],
            "bad_input_ids": bad_tokens["input_ids"].squeeze(0),
            "bad_attention_mask": bad_tokens["attention_mask"].squeeze(0),
            "good_input_ids": good_tokens["input_ids"].squeeze(0),
            "good_attention_mask": good_tokens["attention_mask"].squeeze(0),
            "category": pair["category"],
            "pair_id": pair["id"]
        }


class PreferenceTrainer:
    """Preference-weighted supervised fine-tuning"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        learning_rate: float = 1e-5,
        good_weight: float = 1.0,  # Weight for maximizing good responses
        bad_weight: float = 0.5,   # Weight for minimizing bad responses
        device: str = "auto",
        output_dir: str = "./fine_tuned_model"
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.good_weight = good_weight
        self.bad_weight = bad_weight
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create checkpoints directory
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Create logs directory
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        # Training log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_log_{timestamp}.jsonl"

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device
        )

        self.device = next(self.model.parameters()).device
        print(f"✓ Model loaded on {self.device}\n")

    def compute_preference_loss(self, batch):
        """
        Compute preference loss: maximize log-likelihood of good responses,
        minimize log-likelihood of bad responses.

        Loss = -good_weight * log_p(good) + bad_weight * log_p(bad)

        This is simpler and more stable than full DPO.
        """
        # Get logits for good responses
        good_outputs = self.model(
            input_ids=batch["good_input_ids"],
            attention_mask=batch["good_attention_mask"],
            labels=batch["good_input_ids"]  # Use input_ids as labels for next-token prediction
        )

        # Get logits for bad responses
        bad_outputs = self.model(
            input_ids=batch["bad_input_ids"],
            attention_mask=batch["bad_attention_mask"],
            labels=batch["bad_input_ids"]
        )

        # Cross-entropy loss is already computed by the model
        # For good responses, we want to minimize CE loss (maximize likelihood)
        # For bad responses, we want to maximize CE loss (minimize likelihood)
        good_loss = good_outputs.loss
        bad_loss = bad_outputs.loss

        # Combined preference loss
        # We maximize good (minimize CE loss) and minimize bad (invert CE loss)
        total_loss = self.good_weight * good_loss - self.bad_weight * bad_loss

        # Preference accuracy: comparing average per-token losses
        # If good_loss < bad_loss, model prefers good (correct)
        preference_correct = (good_loss < bad_loss).float().mean()

        return total_loss, preference_correct.item(), good_loss.item(), bad_loss.item()

    def train(
        self,
        training_pairs: List[Dict],
        epochs: int = 200,
        batch_size: int = 1,
        checkpoint_every: int = 10,
        warmup_steps: int = 10
    ):
        """Train model with preference-weighted supervision"""

        print(f"🎯 Training Configuration:")
        print(f"  Training pairs: {len(training_pairs)}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Good weight: {self.good_weight}")
        print(f"  Bad weight: {self.bad_weight}")
        print(f"  Checkpoint every: {checkpoint_every} epochs")
        print(f"  Output directory: {self.output_dir}")
        print()

        # Create dataset and dataloader
        dataset = ContrastivePairDataset(training_pairs, self.tokenizer)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler with warmup
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Training loop
        self.model.train()
        global_step = 0
        best_loss = float('inf')

        print("🚀 Starting training...\n")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_good_loss = 0.0
            epoch_bad_loss = 0.0
            epoch_accuracy = 0.0

            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch}/{epochs}",
                leave=False
            )

            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                loss, accuracy, good_loss, bad_loss = self.compute_preference_loss(batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Track metrics
                epoch_loss += loss.item()
                epoch_good_loss += good_loss
                epoch_bad_loss += bad_loss
                epoch_accuracy += accuracy
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'good': f'{good_loss:.4f}',
                    'bad': f'{bad_loss:.4f}',
                    'acc': f'{accuracy:.2%}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            # Epoch metrics
            avg_loss = epoch_loss / len(dataloader)
            avg_good_loss = epoch_good_loss / len(dataloader)
            avg_bad_loss = epoch_bad_loss / len(dataloader)
            avg_accuracy = epoch_accuracy / len(dataloader)
            epoch_time = time.time() - epoch_start

            # Log metrics
            log_entry = {
                "epoch": epoch,
                "global_step": global_step,
                "avg_loss": avg_loss,
                "avg_good_loss": avg_good_loss,
                "avg_bad_loss": avg_bad_loss,
                "avg_accuracy": avg_accuracy,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch_time_seconds": epoch_time,
                "timestamp": datetime.now().isoformat()
            }

            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            # Print epoch summary
            print(f"Epoch {epoch}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Good: {avg_good_loss:.4f} | "
                  f"Bad: {avg_bad_loss:.4f} | "
                  f"Acc: {avg_accuracy:.2%} | "
                  f"Time: {epoch_time:.1f}s")

            # Save checkpoint every N epochs
            if epoch % checkpoint_every == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint-{epoch:03d}"
                self.save_checkpoint(checkpoint_path, epoch, avg_loss, avg_accuracy)
                print(f"  ✓ Checkpoint saved: {checkpoint_path.name}")

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = self.output_dir / "best_model"
                self.save_checkpoint(best_path, epoch, avg_loss, avg_accuracy)
                print(f"  ⭐ New best model saved (loss: {avg_loss:.4f})")

        # Final save
        final_path = self.output_dir / "final_model"
        self.save_checkpoint(final_path, epochs, avg_loss, avg_accuracy)
        print(f"\n✅ Training complete!")
        print(f"Final model saved to: {final_path}")
        print(f"Training log: {self.log_file}")

    def save_checkpoint(self, path: Path, epoch: int, loss: float, accuracy: float):
        """Save model checkpoint"""
        path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save metadata
        metadata = {
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
            "model_name": self.model_name,
            "learning_rate": self.learning_rate,
            "good_weight": self.good_weight,
            "bad_weight": self.bad_weight,
            "timestamp": datetime.now().isoformat()
        }

        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-0.5B for epistemic pragmatism"
    )
    parser.add_argument(
        "--corpus",
        default="training_corpus.json",
        help="Training corpus JSON file"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--output-dir",
        default="./fine_tuned_model",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--good-weight",
        type=float,
        default=1.0,
        help="Weight for maximizing good responses"
    )
    parser.add_argument(
        "--bad-weight",
        type=float,
        default=0.5,
        help="Weight for minimizing bad responses"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (1 recommended for small model)"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device (auto, cuda, cpu)"
    )

    args = parser.parse_args()

    # Load training corpus
    print(f"Loading training corpus from {args.corpus}")
    with open(args.corpus) as f:
        corpus_data = json.load(f)

    training_pairs = corpus_data["training_pairs"]
    print(f"✓ Loaded {len(training_pairs)} training pairs\n")

    # Initialize trainer
    trainer = PreferenceTrainer(
        model_name=args.model,
        learning_rate=args.learning_rate,
        good_weight=args.good_weight,
        bad_weight=args.bad_weight,
        device=args.device,
        output_dir=args.output_dir
    )

    # Train
    trainer.train(
        training_pairs=training_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every
    )


if __name__ == "__main__":
    main()
