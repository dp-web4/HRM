#!/usr/bin/env python3
"""
Universal training script for threshold detection experiments

Trains models at different dataset sizes to find the scaffolding suitability threshold.
Uses DPO (Direct Preference Optimization) format with chosen/rejected responses.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType


def load_dpo_dataset(dataset_path: Path):
    """Load DPO format dataset and convert to training format"""
    print(f"Loading dataset: {dataset_path}")

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples")

    # Convert DPO format to training format
    # For now, we'll just use the "chosen" responses as the target
    # (Full DPO training would require a more complex setup)

    training_data = []
    for item in data:
        prompt = item['prompt']
        chosen = item['chosen']

        # Format as Q&A pair
        text = f"Question: {prompt}\n\nAnswer: {chosen}"
        training_data.append({"text": text})

    return Dataset.from_list(training_data)


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize examples for training"""
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    outputs["labels"] = outputs["input_ids"].clone()
    return outputs


def train_model(
    dataset_size: int,
    num_epochs: int,
    output_dir: Path,
    base_model: str = "Qwen/Qwen2.5-0.5B"
):
    """
    Train a model on the specified dataset size

    Args:
        dataset_size: Number of training examples (40, 60, 80, 100, 115)
        num_epochs: Number of training epochs (ballpark)
        output_dir: Where to save the trained model
        base_model: Base model to fine-tune
    """
    print("="*80)
    print(f"Training {dataset_size}-example model")
    print("="*80)
    print(f"Base model: {base_model}")
    print(f"Epochs: {num_epochs}")
    print(f"Output: {output_dir}")
    print()

    # Setup paths
    base_dir = Path(__file__).parent
    dataset_path = base_dir / "training_datasets" / f"claude_personal_dataset_{dataset_size}examples.json"

    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print(f"   Run create_dataset_subsets.py first")
        sys.exit(1)

    # Load dataset
    dataset = load_dpo_dataset(dataset_path)
    print(f"✓ Dataset loaded: {len(dataset)} examples\n")

    # Load tokenizer and model
    print(f"Loading model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    print("✓ Model loaded\n")

    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    print("✓ Tokenization complete\n")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        load_best_model_at_end=False,
        remove_unused_columns=True,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train!
    print("="*80)
    print("Starting Training")
    print("="*80)
    start_time = datetime.now()

    trainer.train()

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n✓ Training complete! Duration: {duration}")

    # Save final model
    final_model_dir = output_dir / "final_model"
    print(f"\nSaving final model to {final_model_dir}")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print(f"\n✓ Model saved successfully!")
    print(f"   Location: {final_model_dir}")

    return final_model_dir


def main():
    parser = argparse.ArgumentParser(description="Train models for threshold detection")
    parser.add_argument(
        "size",
        type=int,
        choices=[40, 60, 80, 100, 115],
        help="Dataset size (number of examples)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (default: auto-selected based on size)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: threshold_models/<size>examples_model)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Base model to fine-tune"
    )

    args = parser.parse_args()

    # Auto-select epochs if not specified (ballpark numbers)
    if args.epochs is None:
        epoch_map = {
            40: 18,   # More epochs for smaller datasets
            60: 14,
            80: 11,
            100: 9,
            115: 8
        }
        args.epochs = epoch_map[args.size]
        print(f"Auto-selected {args.epochs} epochs for {args.size} examples (ballpark)")

    # Setup output directory
    if args.output_dir is None:
        base_dir = Path(__file__).parent
        output_dir = base_dir / "threshold_models" / f"{args.size}examples_model"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Train the model
    train_model(
        dataset_size=args.size,
        num_epochs=args.epochs,
        output_dir=output_dir,
        base_model=args.base_model
    )

    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)
    print(f"✓ Model trained: {output_dir}/final_model")
    print(f"✓ Ready for testing with experiment_orchestrator.py")
    print()


if __name__ == "__main__":
    main()
