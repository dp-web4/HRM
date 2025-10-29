#!/usr/bin/env python3
"""
Phase 2.1 Training - SFT Approach (Like Phase 1)

Using the SAME method that succeeded in Phase 1:
- Standard supervised fine-tuning (not DPO)
- LoRA for efficiency
- transformers.Trainer (not DPOTrainer)
- Just use "chosen" responses from our 115 examples

Why SFT instead of DPO?
- Phase 1 succeeded with SFT
- DPO causing immediate collapse (NaN gradients)
- Our "chosen" responses already teach correct epistemic stances
- Simpler, more stable training
"""

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os


def convert_dpo_to_sft_format(dpo_json_path, val_split=0.2):
    """
    Convert DPO dataset to SFT format.

    Just use the "chosen" responses - they already demonstrate
    correct epistemic stance selection.

    Args:
        dpo_json_path: Path to DPO dataset
        val_split: Validation fraction

    Returns:
        train_dataset, val_dataset
    """
    with open(dpo_json_path) as f:
        dpo_pairs = json.load(f)

    # Convert to SFT format: instruction → response
    sft_examples = []
    for pair in dpo_pairs:
        sft_examples.append({
            "instruction": pair["prompt"],
            "response": pair["chosen"],  # Use the correct response
            "category": pair.get("category", "unknown")
        })

    print(f"Converted {len(sft_examples)} DPO pairs to SFT format")

    # Category distribution
    categories = {}
    for ex in sft_examples:
        cat = ex['category']
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} examples")

    # Split train/val
    import random
    random.seed(42)
    random.shuffle(sft_examples)

    split_idx = int(len(sft_examples) * (1 - val_split))
    train_examples = sft_examples[:split_idx]
    val_examples = sft_examples[split_idx:]

    print(f"\nSplit: {len(train_examples)} train / {len(val_examples)} validation")

    return Dataset.from_list(train_examples), Dataset.from_list(val_examples)


def format_instruction(example, tokenizer):
    """
    Format instruction-response pair for training.
    Same format as Phase 1.
    """
    text = f"Instruction: {example['instruction']}\n\nResponse: {example['response']}"

    # Tokenize
    result = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding=False,
    )

    # Labels = same as input_ids for causal LM
    result["labels"] = result["input_ids"].copy()

    return result


def main():
    print("="*80)
    print("Phase 2.1 Training - SFT Approach")
    print("="*80)
    print("\nUsing Phase 1's successful method:")
    print("  • Standard supervised fine-tuning (not DPO)")
    print("  • LoRA for efficiency")
    print("  • transformers.Trainer")
    print("  • 115 high-quality examples")
    print("  • 92 train / 23 validation")
    print()

    # Paths
    phase1_checkpoint = "./fine_tuned_model/final_model"
    dpo_dataset_path = "./claude_personal_dataset_dpo.json"
    output_dir = "./phase2.1_sft_model"

    # Load and convert dataset
    print(f"Loading dataset from {dpo_dataset_path}...")
    train_dataset, val_dataset = convert_dpo_to_sft_format(dpo_dataset_path)

    # Load Phase 1 model
    print(f"\nLoading Phase 1 model from {phase1_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(phase1_checkpoint)

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        phase1_checkpoint,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Apply LoRA (like Phase 1)
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Rank
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("✓ Model loaded with LoRA\n")

    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: format_instruction(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: format_instruction(x, tokenizer),
        remove_columns=val_dataset.column_names
    )
    print("✓ Tokenization complete\n")

    # Training configuration (similar to Phase 1)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=50,  # Like Phase 1, but will use early stopping
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=2e-4,  # Standard for LoRA
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,  # Keep best 3 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        remove_unused_columns=False,
        report_to="none",
        warmup_steps=10,
    )

    print("Training configuration:")
    print(f"  Method: Supervised Fine-Tuning (SFT) with LoRA")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Validation: Every epoch")
    print(f"  Early stopping: On eval_loss")
    print()

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Create trainer (standard Trainer, not DPOTrainer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting training...\n")
    print("Expected behavior:")
    print("  ✓ Gradual loss decrease")
    print("  ✓ Validation loss tracking training loss")
    print("  ✓ No immediate collapse to 0.0")
    print()

    # Train
    trainer.train()

    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)

    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Save metadata
    metadata = {
        "phase": "2.1_sft",
        "method": "supervised_fine_tuning",
        "description": "SFT training on 115 genuine epistemic reasoning examples",
        "base_model": "Phase 1 epistemic-pragmatism",
        "training_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "lora_config": {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "target_modules": lora_config.target_modules
        },
        "why_sft_not_dpo": {
            "reason": "DPO caused immediate collapse with NaN gradients",
            "phase1_success": "SFT worked perfectly in Phase 1",
            "pragmatic_choice": "Use proven method, focus on quality data"
        },
        "dataset_quality": {
            "source": "claude_personal_dataset.md - genuine introspection",
            "factual": "Direct answers without hedging",
            "behavioral": "Observable patterns without phenomenology",
            "consciousness": "Epistemic humility with genuine uncertainty"
        }
    }

    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    print("\nNext steps:")
    print("  1. Validate on epistemic stance tests")
    print("  2. Deploy with SAGE-IRP integration")
    print("  3. Test on Jetson")


if __name__ == "__main__":
    main()
