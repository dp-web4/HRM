#!/usr/bin/env python3
"""
Phase 2.1 Training - Explicit Reasoning Approach

Trains model to perform explicit context analysis before responding.
Model learns to emit <context_analysis> blocks showing its reasoning.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import os

def load_training_data(corpus_path):
    """Load Phase 2.1 corpus and format for explicit reasoning training."""
    with open(corpus_path) as f:
        corpus = json.load(f)

    examples = []
    for item in corpus:
        # Format with explicit reasoning chains
        examples.append({
            "prompt": item['question'],
            "chosen": item['explicit_reasoning'],  # Includes context analysis
            "rejected": item['rejected_response'],  # No analysis, just bad response
            "category": item['category']
        })

    return Dataset.from_list(examples)

def main():
    print("=== Phase 2.1 Training: Explicit Reasoning Chains ===\n")

    # Paths
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    phase1_checkpoint = "./fine_tuned_model/final_model"  # Start from Phase 1 result
    corpus_path = "./phase2.1_training_corpus.json"
    output_dir = "./phase2.1_explicit_reasoning_model"

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

    # Training configuration (adjusted for DPOTrainer)
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=200,
        per_device_train_batch_size=1,
        learning_rate=1e-5,
        beta=0.1,  # DPO temperature
        logging_steps=10,
        save_steps=10,
        save_total_limit=21,  # Keep all checkpoints for analysis
        fp16=False,  # Disabled due to gradient scaler conflict
        remove_unused_columns=False,
        report_to="none"
    )

    print("Training configuration:")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Beta: {training_args.beta}")
    print(f"  Checkpoints: Every 10 epochs")
    print()

    print("Training target: Model will learn to emit context analysis before responding")
    print("Example output format:")
    print("  <context_analysis>")
    print("  Type: what_causes")
    print("  Domain: factual")
    print("  ...</context_analysis>")
    print("  [Response]")
    print()

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training...\n")
    trainer.train()

    print("\nTraining complete!")
    print(f"Final model saved to: {output_dir}/final_model")

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"\nModel saved to: {final_model_path}")

    # Save training metadata
    metadata = {
        "approach": "explicit_reasoning_chains",
        "description": "Model performs explicit context analysis before responding",
        "base_model": base_model,
        "phase1_checkpoint": phase1_checkpoint,
        "training_pairs": len(dataset),
        "categories": categories,
        "epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "beta": training_args.beta,
        "note": "Responses include <context_analysis> blocks - strip at inference"
    }

    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()
