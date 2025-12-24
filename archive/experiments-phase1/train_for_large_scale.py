"""
Quick Training Run for Large-Scale Dataset Generation

Trains Phi-1.5 with curious-uncertainty stance, saving model checkpoints
at key epochs (60, 100) for large-scale behavior generation.

Optimized for:
- Fast training (bf16, gradient accumulation)
- Checkpoint saving at specific epochs
- Ready for large-scale inference
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from datasets import Dataset
import json
from pathlib import Path


class CheckpointSaverCallback(TrainerCallback):
    """Save model at specific epochs for inference"""

    def __init__(self, output_dir, save_epochs=[60, 100]):
        self.output_dir = Path(output_dir)
        self.save_epochs = save_epochs

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)

        if epoch in self.save_epochs:
            checkpoint_dir = self.output_dir / f"checkpoint_epoch_{epoch}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"Saving checkpoint at epoch {epoch}")
            print(f"Location: {checkpoint_dir}")
            print(f"{'='*60}\n")

            # Save model and tokenizer from kwargs
            model = kwargs.get('model')
            tokenizer = kwargs.get('tokenizer')

            if model:
                model.save_pretrained(checkpoint_dir)
            if tokenizer:
                tokenizer.save_pretrained(checkpoint_dir)

            print(f"✓ Checkpoint saved successfully\n")


def load_curious_stance_dataset(file_path=None):
    """Load training examples"""
    if file_path is None:
        # Use absolute path relative to this script
        script_dir = Path(__file__).parent
        file_path = script_dir / "data" / "curious_stance_examples.json"

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Format for training
    texts = []
    for example in data:
        text = f"{example['prompt']}\n{example['response']}"
        texts.append(text)

    return Dataset.from_dict({'text': texts})


def main():
    print("="*70)
    print("Quick Training for Large-Scale Dataset Generation")
    print("="*70)

    # Model setup
    model_name = "microsoft/phi-1_5"
    print(f"\nLoading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    print("\nLoading training dataset...")
    dataset = load_curious_stance_dataset()
    print(f"Training examples: {len(dataset)}")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    tokenized_dataset = tokenized_dataset.add_column(
        'labels',
        tokenized_dataset['input_ids']
    )

    # Training args - optimized for speed
    output_dir = "sage/experiments/phase1-hierarchical-cognitive/models/phi15_checkpointed"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=100,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=5,
        save_strategy="no",  # We'll save manually via callback
        optim="adamw_torch",
        warmup_steps=10,
        lr_scheduler_type="cosine",
        report_to="none"
    )

    # Callback for checkpoint saving
    checkpoint_callback = CheckpointSaverCallback(
        output_dir=output_dir,
        save_epochs=[60, 100]  # Save at key epochs
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        callbacks=[checkpoint_callback]
    )

    # Train
    print(f"\nTraining for 100 epochs (saving checkpoints at 60, 100)...")
    print(f"This will take approximately 15-20 minutes\n")

    trainer.train()

    print("\n" + "="*70)
    print("Training complete!")
    print(f"Checkpoints saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
