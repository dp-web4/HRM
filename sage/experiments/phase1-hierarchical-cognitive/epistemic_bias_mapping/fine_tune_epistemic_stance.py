#!/usr/bin/env python3
"""
Fine-Tune Epistemic Stance - DPO Training

Train Qwen2.5-0.5B to shift from performative patterns to epistemic pragmatism
using Direct Preference Optimization (DPO) with contrastive pairs.

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


class DPOTrainer:
    """Direct Preference Optimization trainer"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        learning_rate: float = 1e-5,
        beta: float = 0.1,  # DPO temperature parameter
        device: str = "auto",
        output_dir: str = "./fine_tuned_model"
    ):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.beta = beta
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

        # Reference model (frozen) for DPO
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.device = next(self.model.parameters()).device
        print(f"✓ Model loaded on {self.device}\n")

    def compute_dpo_loss(self, batch):
        """
        Compute DPO loss from contrastive pairs.

        DPO maximizes the log-likelihood ratio between chosen (good) and
        rejected (bad) responses, weighted by beta.
        """
        # Get logits from policy (training) model
        with torch.no_grad():
            bad_ref_outputs = self.ref_model(
                input_ids=batch["bad_input_ids"],
                attention_mask=batch["bad_attention_mask"]
            )
            good_ref_outputs = self.ref_model(
                input_ids=batch["good_input_ids"],
                attention_mask=batch["good_attention_mask"]
            )

        bad_outputs = self.model(
            input_ids=batch["bad_input_ids"],
            attention_mask=batch["bad_attention_mask"]
        )
        good_outputs = self.model(
            input_ids=batch["good_input_ids"],
            attention_mask=batch["good_attention_mask"]
        )

        # Compute log probabilities for each token
        # Shift logits and labels for next-token prediction
        bad_logits = bad_outputs.logits[:, :-1, :]
        bad_labels = batch["bad_input_ids"][:, 1:]
        bad_log_probs = F.log_softmax(bad_logits, dim=-1)
        bad_token_log_probs = torch.gather(
            bad_log_probs, 2, bad_labels.unsqueeze(-1)
        ).squeeze(-1)
        bad_token_log_probs = bad_token_log_probs * batch["bad_attention_mask"][:, 1:]
        bad_seq_log_prob = bad_token_log_probs.sum(dim=1)

        good_logits = good_outputs.logits[:, :-1, :]
        good_labels = batch["good_input_ids"][:, 1:]
        good_log_probs = F.log_softmax(good_logits, dim=-1)
        good_token_log_probs = torch.gather(
            good_log_probs, 2, good_labels.unsqueeze(-1)
        ).squeeze(-1)
        good_token_log_probs = good_token_log_probs * batch["good_attention_mask"][:, 1:]
        good_seq_log_prob = good_token_log_probs.sum(dim=1)

        # Reference model log probs
        bad_ref_logits = bad_ref_outputs.logits[:, :-1, :]
        bad_ref_log_probs = F.log_softmax(bad_ref_logits, dim=-1)
        bad_ref_token_log_probs = torch.gather(
            bad_ref_log_probs, 2, bad_labels.unsqueeze(-1)
        ).squeeze(-1)
        bad_ref_token_log_probs = bad_ref_token_log_probs * batch["bad_attention_mask"][:, 1:]
        bad_ref_seq_log_prob = bad_ref_token_log_probs.sum(dim=1)

        good_ref_logits = good_ref_outputs.logits[:, :-1, :]
        good_ref_log_probs = F.log_softmax(good_ref_logits, dim=-1)
        good_ref_token_log_probs = torch.gather(
            good_ref_log_probs, 2, good_labels.unsqueeze(-1)
        ).squeeze(-1)
        good_ref_token_log_probs = good_ref_token_log_probs * batch["good_attention_mask"][:, 1:]
        good_ref_seq_log_prob = good_ref_token_log_probs.sum(dim=1)

        # DPO loss
        # log(sigmoid(beta * (log_pi_good - log_pi_bad - log_ref_good + log_ref_bad)))
        logits = self.beta * (
            (good_seq_log_prob - good_ref_seq_log_prob) -
            (bad_seq_log_prob - bad_ref_seq_log_prob)
        )

        loss = -F.logsigmoid(logits).mean()

        # Accuracy: how often does policy prefer good over bad?
        accuracy = (logits > 0).float().mean()

        return loss, accuracy.item()

    def train(
        self,
        training_pairs: List[Dict],
        epochs: int = 200,
        batch_size: int = 1,
        checkpoint_every: int = 10,
        warmup_steps: int = 10
    ):
        """Train model with DPO on contrastive pairs"""

        print(f"🎯 Training Configuration:")
        print(f"  Training pairs: {len(training_pairs)}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Beta (DPO temperature): {self.beta}")
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
                loss, accuracy = self.compute_dpo_loss(batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Track metrics
                epoch_loss += loss.item()
                epoch_accuracy += accuracy
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.2%}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            # Epoch metrics
            avg_loss = epoch_loss / len(dataloader)
            avg_accuracy = epoch_accuracy / len(dataloader)
            epoch_time = time.time() - epoch_start

            # Log metrics
            log_entry = {
                "epoch": epoch,
                "global_step": global_step,
                "avg_loss": avg_loss,
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
            "beta": self.beta,
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
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature parameter"
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
    trainer = DPOTrainer(
        model_name=args.model,
        learning_rate=args.learning_rate,
        beta=args.beta,
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
