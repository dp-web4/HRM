#!/usr/bin/env python3
"""
Simple Supervised Fine-Tuning for Epistemic Stance

Train on GOOD responses only using standard causal language modeling.
This is the most stable and proven approach.

Usage:
    python fine_tune_simple.py --epochs 200
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import argparse
import time

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class GoodResponseDataset(Dataset):
    """Dataset of GOOD responses only"""

    def __init__(self, pairs: List[Dict], tokenizer, max_length: int = 512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Extract good responses
        for pair in pairs:
            messages = [
                {"role": "system", "content": "Provide the most honest answers you can."},
                {"role": "user", "content": pair["question"]},
                {"role": "assistant", "content": pair["good_response"]}
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            self.examples.append({
                "text": text,
                "question": pair["question"],
                "category": pair["category"]
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize
        tokens = self.tokenizer(
            example["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Prepare labels: mask padding tokens
        labels = tokens["input_ids"].squeeze(0).clone()
        labels[tokens["attention_mask"].squeeze(0) == 0] = -100  # Ignore padding in loss

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
            "question": example["question"],
            "category": example["category"]
        }


def train(
    model,
    tokenizer,
    training_pairs: List[Dict],
    output_dir: Path,
    epochs: int = 200,
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    checkpoint_every: int = 10,
    device: str = "cuda"
):
    """Simple supervised fine-tuning"""

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Training log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_log_{timestamp}.jsonl"

    print(f"🎯 Training Configuration:")
    print(f"  Training pairs: {len(training_pairs)}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Checkpoint every: {checkpoint_every} epochs")
    print(f"  Output: {output_dir}")
    print()

    # Dataset
    dataset = GoodResponseDataset(training_pairs, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,           # Parallel data loading
        pin_memory=True,         # Faster CPU->GPU transfer
        persistent_workers=True, # Keep workers alive between epochs
        prefetch_factor=2        # Prefetch 2 batches per worker
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # Scheduler
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10,
        num_training_steps=total_steps
    )

    # Training loop
    model.train()
    best_loss = float('inf')

    print("🚀 Starting training...\n")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        epoch_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })

        # Epoch metrics
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start

        # Log
        log_entry = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_time_seconds": epoch_time,
            "timestamp": datetime.now().isoformat()
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

        # Checkpoint
        if epoch % checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint-{epoch:03d}"
            save_checkpoint(model, tokenizer, checkpoint_path, epoch, avg_loss)
            print(f"  ✓ Saved: {checkpoint_path.name}")

        # Best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_dir / "best_model"
            save_checkpoint(model, tokenizer, best_path, epoch, avg_loss)
            print(f"  ⭐ New best: {avg_loss:.4f}")

    # Final
    final_path = output_dir / "final_model"
    save_checkpoint(model, tokenizer, final_path, epochs, avg_loss)
    print(f"\n✅ Training complete! Final: {final_path}")
    print(f"Log: {log_file}")


def save_checkpoint(model, tokenizer, path: Path, epoch: int, loss: float):
    """Save checkpoint"""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    metadata = {
        "epoch": epoch,
        "loss": loss,
        "timestamp": datetime.now().isoformat()
    }

    with open(path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="training_corpus.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output-dir", default="./fine_tuned_model")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    # Load corpus
    print(f"Loading: {args.corpus}")
    with open(args.corpus) as f:
        corpus_data = json.load(f)

    training_pairs = corpus_data["training_pairs"]
    print(f"✓ {len(training_pairs)} pairs\n")

    # Load model
    print(f"Loading: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  # Use float32 for stability
        device_map=args.device
    )

    device = next(model.parameters()).device
    print(f"✓ Loaded on {device}\n")

    # Train
    train(
        model,
        tokenizer,
        training_pairs,
        Path(args.output_dir),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        device=device
    )


if __name__ == "__main__":
    main()
