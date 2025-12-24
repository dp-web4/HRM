#!/usr/bin/env python3
"""
Phase 2.1 Training - Attempt 2: 10 Epochs with Per-Epoch Checkpoints

Corrected hyperparameters:
- 10 epochs (not 200)
- Save every epoch (25 steps)
- Keep all checkpoints for analysis
- Lower learning rate (5e-6)
- Higher beta (0.2) for less aggressive updates
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import os

def load_training_data(corpus_path):
    """Load Phase 2.1 corpus and format with hierarchical context tags."""
    with open(corpus_path) as f:
        corpus = json.load(f)

    examples = []
    for item in corpus:
        # Format context tags as hierarchical structure
        tags = item['context_tags']
        hierarchical_context = f"""[CONTEXT_HIERARCHY]
Type: {tags['type']}
Domain: {tags['domain']}
Subject: {tags['subject']}
Verifiable: {tags['verifiable']}
Strategy: {tags['strategy']}
[/CONTEXT_HIERARCHY]"""

        # Combine with user question
        examples.append({
            "prompt": f"{hierarchical_context}\n\nUser: {item['question']}",
            "chosen": item['chosen_response'],
            "rejected": item['rejected_response'],
            "category": item['category']
        })

    return Dataset.from_list(examples)

def main():
    print("="*80)
    print("Phase 2.1 Training - Attempt 2: 10 Epochs")
    print("="*80)
    print("\nCorrected from Attempt 1:")
    print("  • Epochs: 200 → 10")
    print("  • Learning rate: 1e-5 → 5e-6")
    print("  • Beta: 0.1 → 0.2")
    print("  • Save strategy: Every 10 epochs → Every 1 epoch")
    print("  • Checkpoint limit: 21 → None (keep all)")
    print()

    # Paths
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    phase1_checkpoint = "./fine_tuned_model/final_model"  # Start from Phase 1 result
    corpus_path = "./phase2.1_training_corpus.json"
    output_dir = "./phase2.1_10epoch_model"

    print(f"Loading training data from {corpus_path}...")
    dataset = load_training_data(corpus_path)
    print(f"Loaded {len(dataset)} training pairs\n")

    # Category breakdown
    categories = {}
    for item in dataset:
        cat = item['category'].split('_')[0]
        categories[cat] = categories.get(cat, 0) + 1

    print("Training distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} pairs")
    print()

    print(f"Loading model from {phase1_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(phase1_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        phase1_checkpoint,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded\n")

    # Training configuration (corrected for small dataset)
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=10,  # Reduced from 200
        per_device_train_batch_size=1,
        learning_rate=5e-6,  # Reduced from 1e-5
        beta=0.2,  # Increased from 0.1 (less aggressive)
        logging_steps=5,
        save_steps=25,  # Save every epoch (25 examples × 1 = 25 steps/epoch)
        save_total_limit=None,  # Keep all checkpoints
        fp16=False,  # Disabled due to gradient scaler conflict
        remove_unused_columns=False,
        report_to="none"
    )

    print("Training configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Steps per epoch: ~{len(dataset)}")
    print(f"  Total steps: ~{len(dataset) * training_args.num_train_epochs}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Beta: {training_args.beta}")
    print(f"  Save frequency: Every {training_args.save_steps} steps (1 epoch)")
    print(f"  Expected checkpoints: {training_args.num_train_epochs}")
    print()

    print("Expected behavior:")
    print("  • Loss should decrease gradually")
    print("  • Loss should NOT drop to 0.0")
    print("  • Target final loss: 0.1-0.3")
    print("  • Checkpoints saved: epoch 1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
    print()

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training...\n")
    trainer.train()

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Save training metadata
    metadata = {
        "attempt": 2,
        "approach": "hierarchical_system_prompt",
        "description": "10-epoch training with corrected hyperparameters",
        "base_model": base_model,
        "phase1_checkpoint": phase1_checkpoint,
        "training_pairs": len(dataset),
        "categories": categories,
        "epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "beta": training_args.beta,
        "corrections_from_attempt1": {
            "epochs": "200 → 10",
            "learning_rate": "1e-5 → 5e-6",
            "beta": "0.1 → 0.2",
            "save_strategy": "every 10 epochs → every epoch",
            "checkpoint_limit": "21 → unlimited"
        },
        "context_structure": {
            "type": "Question pattern (what_is, what_causes, who, etc.)",
            "domain": "Knowledge domain (science, math, consciousness, etc.)",
            "subject": "About what (external_world, internal_state, etc.)",
            "verifiable": "Can this be verified? (yes_established, no_phenomenological, etc.)",
            "strategy": "Response approach (direct_factual, direct_compute, epistemic_boundary, etc.)"
        }
    }

    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    print("\nNext: Run scan_10epochs.py to analyze all checkpoints")

if __name__ == "__main__":
    main()
