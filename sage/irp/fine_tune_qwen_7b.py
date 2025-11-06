#!/usr/bin/env python3
"""
Fine-tune Qwen 7B with Epistemic Pragmatism

Testing inertia hypothesis:
- Same 115 examples that worked for 0.5B
- Does 7B learn the same pragmatic style?
- Or does it need more examples due to inertia?

Based on train_phase2.1_sft.py (which succeeded for 0.5B)
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
from pathlib import Path


def convert_dpo_to_sft_format(dpo_json_path, val_split=0.2):
    """
    Convert DPO dataset to SFT format.
    Just use the "chosen" responses - they demonstrate correct epistemic stance.
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
    """Format instruction-response pair for training."""
    text = f"Instruction: {example['instruction']}\n\nResponse: {example['response']}"

    result = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
    )

    # Labels = same as input_ids for causal LM
    result["labels"] = result["input_ids"].copy()

    return result


def main():
    """Fine-tune Qwen 7B with epistemic pragmatism"""

    print("="*80)
    print("QWEN 7B FINE-TUNING - EPISTEMIC PRAGMATISM")
    print("Testing inertia hypothesis: Does 7B learn as easily as 0.5B?")
    print("="*80)

    # Paths
    base_model_path = "/home/dp/ai-workspace/HRM/model-zoo/sage/qwen2.5-7b-instruct"
    dataset_path = "/home/dp/ai-workspace/HRM/sage/experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping/claude_personal_dataset_dpo.json"
    output_dir = "/home/dp/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-7b/epistemic-pragmatism"

    # Check base model exists
    if not Path(base_model_path).exists():
        print(f"\n❌ Base model not found: {base_model_path}")
        print("Run download_qwen_7b.py first")
        return

    print(f"\n✓ Base model: {base_model_path}")
    print(f"✓ Dataset: {dataset_path}")
    print(f"✓ Output: {output_dir}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load and convert dataset
    print("\nLoading dataset...")
    train_dataset, val_dataset = convert_dpo_to_sft_format(dataset_path)

    # Format datasets
    print("\nFormatting datasets...")
    train_dataset = train_dataset.map(
        lambda x: format_instruction(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: format_instruction(x, tokenizer),
        remove_columns=val_dataset.column_names
    )

    # Load base model with LoRA
    print("\nLoading 7B model...")
    print("(This will use ~14GB RAM)")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # Use fp16 to save memory
        trust_remote_code=True,
        device_map="auto",  # Automatically use GPU if available
        low_cpu_mem_usage=True
    )

    # LoRA configuration (same as 0.5B for fair comparison)
    print("\nApplying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Same rank as 0.5B training
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Same as 0.5B
        per_device_train_batch_size=1,  # Small batch for 7B
        gradient_accumulation_steps=8,  # Effective batch size = 8
        learning_rate=1e-4,  # Same as 0.5B
        warmup_steps=10,
        logging_steps=5,
        eval_steps=20,
        save_steps=50,
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,  # Use mixed precision
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    # Merge LoRA weights with base model for easier deployment
    print("\nMerging LoRA weights with base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"{output_dir}/merged")
    tokenizer.save_pretrained(f"{output_dir}/merged")

    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE")
    print("="*80)
    print(f"\nLoRA model: {output_dir}/final")
    print(f"Merged model: {output_dir}/merged")
    print("\nNext: Test with sage_session_4_7b_finetuned.py")


if __name__ == "__main__":
    main()
