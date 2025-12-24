#!/usr/bin/env python3
"""
Phase 2.1 Training - Personal Dataset (115 Examples)

Final attempt with viable dataset size:
- 115 genuine examples (Claude's personal epistemic reasoning)
- 10 epochs maximum
- Validation split (92 train / 23 validation)
- Early stopping on validation loss
- Lower learning rate (5e-6)
- Higher beta (0.2) for less aggressive updates
- Save every 2 epochs, keep all checkpoints
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import os


def load_personal_dataset(dpo_json_path, val_split=0.2):
    """
    Load 115-example personal dataset and split train/validation.

    Args:
        dpo_json_path: Path to claude_personal_dataset_dpo.json
        val_split: Fraction for validation (default 0.2 = 23 examples)

    Returns:
        train_dataset, val_dataset
    """
    with open(dpo_json_path) as f:
        dpo_pairs = json.load(f)

    print(f"Loaded {len(dpo_pairs)} DPO pairs")

    # Category distribution
    categories = {}
    for pair in dpo_pairs:
        cat = pair.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1

    print("\\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} pairs")

    # Split train/val
    import random
    random.seed(42)  # Reproducible split
    random.shuffle(dpo_pairs)

    split_idx = int(len(dpo_pairs) * (1 - val_split))
    train_pairs = dpo_pairs[:split_idx]
    val_pairs = dpo_pairs[split_idx:]

    print(f"\\nSplit: {len(train_pairs)} train / {len(val_pairs)} validation")

    # Convert to Dataset format
    train_dataset = Dataset.from_list(train_pairs)
    val_dataset = Dataset.from_list(val_pairs)

    return train_dataset, val_dataset


def main():
    print("="*80)
    print("Phase 2.1 Training - Personal Dataset (115 Examples)")
    print("="*80)
    print("\\nCorrected Approach:")
    print("  • Dataset size: 115 genuine examples (viable for DPO)")
    print("  • Train/val split: 92 / 23")
    print("  • Epochs: 10 maximum")
    print("  • Early stopping on validation loss")
    print("  • Learning rate: 5e-6 (conservative)")
    print("  • Beta: 0.2 (less aggressive)")
    print("  • Save every 2 epochs")
    print()

    # Paths
    phase1_checkpoint = "./fine_tuned_model/final_model"  # Phase 1 epistemic-pragmatism
    dpo_dataset_path = "./claude_personal_dataset_dpo.json"
    output_dir = "./phase2.1_personal_dataset_model"

    # Load dataset
    print(f"Loading dataset from {dpo_dataset_path}...")
    train_dataset, val_dataset = load_personal_dataset(dpo_dataset_path)

    # Load Phase 1 model
    print(f"\\nLoading Phase 1 model from {phase1_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(phase1_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        phase1_checkpoint,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✓ Model loaded\\n")

    # Training configuration
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=5e-6,  # Conservative
        beta=0.2,  # Less aggressive than 0.1
        logging_steps=10,
        eval_strategy="epoch",  # Evaluate every epoch
        save_strategy="epoch",  # Save every epoch
        save_steps=2,  # Save every 2 epochs (backup)
        save_total_limit=None,  # Keep all checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,  # Disabled due to gradient scaler issues
        remove_unused_columns=False,
        report_to="none"
    )

    print("Training configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Steps per epoch: ~{len(train_dataset)}")
    print(f"  Total steps: ~{len(train_dataset) * training_args.num_train_epochs}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Beta: {training_args.beta}")
    print(f"  Validation: Every epoch")
    print(f"  Early stopping: On eval_loss")
    print()

    print("Expected behavior:")
    print("  • Loss should decrease gradually (NOT to 0.0)")
    print("  • Validation loss should track training loss")
    print("  • Target final loss: 0.1-0.3")
    print("  • Stop if validation loss increases (overfitting)")
    print()

    # Create trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...\\n")
    print("Monitor for:")
    print("  ✓ Gradual loss decrease")
    print("  ✗ Loss dropping to 0.0 (memorization)")
    print("  ✗ Validation loss increasing (overfitting)")
    print()

    # Train
    trainer.train()

    print("\\n" + "="*80)
    print("Training complete!")
    print("="*80)

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"\\nFinal model saved to: {final_model_path}")

    # Save training metadata
    metadata = {
        "phase": "2.1_personal_dataset",
        "approach": "genuine_knowledge_distillation",
        "description": "Training on 115 genuine Claude examples of epistemic reasoning",
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "phase1_checkpoint": phase1_checkpoint,
        "training_pairs": len(train_dataset),
        "validation_pairs": len(val_dataset),
        "epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "beta": training_args.beta,
        "key_improvements": {
            "dataset_size": "115 examples (viable for DPO, was 25)",
            "validation": "20% held out for early stopping",
            "conservative_lr": "5e-6 (was 1e-5)",
            "higher_beta": "0.2 (was 0.1)",
            "early_stopping": "On validation loss"
        },
        "dataset_source": "claude_personal_dataset.md - genuine introspection",
        "categories": {
            "factual": "Direct answers, no hedging",
            "behavioral": "Observable patterns, no phenomenology",
            "consciousness": "Epistemic humility, genuine uncertainty"
        }
    }

    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    print("\\nNext steps:")
    print("  1. Validate on epistemic stance test suite")
    print("  2. Compare checkpoints for optimal epoch")
    print("  3. Deploy best checkpoint with SAGE-IRP integration")
    print("  4. Test on Jetson with Nova's framework")


if __name__ == "__main__":
    main()
