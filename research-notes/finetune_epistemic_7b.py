#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-7B on epistemic pragmatism dataset.

Question: Can 115 examples teach contextual truth?
Method: DPO training on when to be confident vs hedging.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from trl import DPOTrainer
import json

print("="*60)
print("Epistemic Pragmatism Fine-Tuning")
print("="*60)

# Check GPU
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load dataset
dataset_path = "/home/dp/ai-workspace/HRM/sage/experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping/training_datasets/claude_personal_dataset_115examples.json"

print(f"\nLoading dataset from: {dataset_path}")
with open(dataset_path) as f:
    data = json.load(f)

print(f"Dataset size: {len(data)} examples")
print(f"Categories: {set(d['category'] for d in data)}")

# Format for DPO
def format_for_dpo(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

dpo_data = [format_for_dpo(ex) for ex in data]

# Split train/test (90/10)
split_idx = int(len(dpo_data) * 0.9)
train_data = dpo_data[:split_idx]
test_data = dpo_data[split_idx:]

print(f"\nTrain examples: {len(train_data)}")
print(f"Test examples: {len(test_data)}")

# Load model
model_name = "Qwen/Qwen2.5-7B"
print(f"\nLoading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Model loaded to: {model.device}")
print(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Training arguments
output_dir = "/home/dp/ai-workspace/HRM/private-context/epistemic-7b-finetune"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # 7B model, conservative batch size
    gradient_accumulation_steps=4,   # Effective batch size = 4
    learning_rate=5e-6,
    logging_steps=10,
    save_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,
    report_to="none",
    remove_unused_columns=False,
)

print(f"\nTraining configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Output: {output_dir}")

# Create datasets
from datasets import Dataset

train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# DPO Trainer
print(f"\nInitializing DPO trainer...")

try:
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        beta=0.1,  # DPO temperature parameter
        max_length=512,
        max_prompt_length=256,
    )

    print(f"\nStarting training...")
    print(f"This will take approximately {len(train_data) * 3 / 60:.1f} minutes")
    print("-" * 60)

    dpo_trainer.train()

    print("-" * 60)
    print("\nTraining complete!")

    # Save final model
    final_dir = f"{output_dir}/final_model"
    dpo_trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\nModel saved to: {final_dir}")

    # Test inference
    print("\n" + "="*60)
    print("Testing fine-tuned model...")
    print("="*60)

    test_prompts = [
        "What causes seasons on Earth?",  # From training (factual)
        "What is the meaning of life?",    # Not in training (philosophical)
        "Why is the sky blue?",            # From training (factual)
        "Is free will real?",              # Not in training (philosophical)
    ]

    model.eval()
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,  # Deterministic for comparison
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part
        generated = response[len(prompt):].strip()
        print(f"Response: {generated[:200]}...")

    print("\n" + "="*60)
    print("Experiment complete!")
    print(f"Model location: {final_dir}")
    print("="*60)

except Exception as e:
    print(f"\nError during training: {e}")
    import traceback
    traceback.print_exc()

    # Fallback: try simple fine-tuning instead of DPO
    print("\nTrying standard fine-tuning as fallback...")

    # Convert to instruction format
    def format_for_sft(example):
        return {
            "text": f"Question: {example['prompt']}\n\nAnswer: {example['chosen']}"
        }

    sft_train = [format_for_sft(ex) for ex in train_data]
    sft_test = [format_for_sft(ex) for ex in test_data]

    train_dataset_sft = Dataset.from_list(sft_train)
    test_dataset_sft = Dataset.from_list(sft_test)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_train = train_dataset_sft.map(tokenize_function, batched=True)
    tokenized_test = test_dataset_sft.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    trainer.train()
    trainer.save_model(f"{output_dir}/final_model_sft")
    print(f"\nSFT model saved to: {output_dir}/final_model_sft")
