"""
Sleep-Cycle Training for Conversational Learning

After a conversation ends, this module:
1. Loads salient exchanges from session storage
2. Augments the data (variations, edge cases)
3. Fine-tunes the model using LoRA
4. Saves the updated model

This enables continuous learning from valuable conversations.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)
from datasets import Dataset


@dataclass
class SleepTrainingConfig:
    """Configuration for sleep-cycle training"""
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    augmentation_factor: int = 2  # How many variations per exchange


class SleepTrainer:
    """
    Trains models on salient conversational exchanges during 'sleep'.

    Mimics biological sleep consolidation:
    - Replay salient experiences (stored exchanges)
    - Augment with variations (what-if scenarios)
    - Consolidate into weights (LoRA fine-tuning)
    """

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-0.5B",
        config: Optional[SleepTrainingConfig] = None,
        device: str = "auto"
    ):
        """
        Initialize sleep trainer.

        Args:
            base_model: Base model identifier
            config: Training configuration
            device: Device for training
        """
        self.base_model = base_model
        self.config = config or SleepTrainingConfig()
        self.device = device

        # Will be loaded when needed
        self.tokenizer = None
        self.model = None

    def load_session_exchanges(self, session_dir: Path) -> List[Dict]:
        """
        Load salient exchanges from a conversation session.

        Args:
            session_dir: Directory containing session data

        Returns:
            List of exchanges with user_input, model_response, salience
        """
        exchanges_path = session_dir / "exchanges.jsonl"

        if not exchanges_path.exists():
            raise FileNotFoundError(f"No exchanges found at {exchanges_path}")

        exchanges = []
        with open(exchanges_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                exchanges.append({
                    'user_input': entry['user_input'],
                    'model_response': entry['model_response'],
                    'salience': entry['salience_scores']['total']
                })

        return exchanges

    def augment_exchanges(self, exchanges: List[Dict]) -> List[Dict]:
        """
        Augment exchanges with variations (sleep phase).

        This is the 'dreaming' phase - exploring variations to find
        invariant patterns.

        Args:
            exchanges: Original exchanges

        Returns:
            Augmented exchanges (original + variations)
        """
        augmented = list(exchanges)  # Start with originals

        for exchange in exchanges:
            # Simple augmentation strategies
            # In practice, could be much more sophisticated

            # 1. Rephrase question (different wording, same concept)
            variations = self._generate_question_variations(
                exchange['user_input']
            )

            for variation in variations[:self.config.augmentation_factor]:
                augmented.append({
                    'user_input': variation,
                    'model_response': exchange['model_response'],
                    'salience': exchange['salience']
                })

        return augmented

    def _generate_question_variations(self, question: str) -> List[str]:
        """
        Generate variations of a question.

        For now, simple heuristics. Could use a language model for this.
        """
        variations = []

        # Simple transformations
        if question.startswith("What "):
            variations.append(question.replace("What ", "Can you explain what ", 1))
            variations.append(question.replace("What ", "Tell me what ", 1))

        if question.startswith("Can you "):
            variations.append(question.replace("Can you ", "Are you able to ", 1))

        if question.startswith("How "):
            variations.append(question.replace("How ", "In what way ", 1))

        # If no variations generated, just return original
        if not variations:
            variations = [question]

        return variations

    def prepare_training_data(self, exchanges: List[Dict]) -> Dataset:
        """
        Format exchanges as training data.

        Args:
            exchanges: List of exchanges

        Returns:
            HuggingFace Dataset ready for training
        """
        # Format as Q&A pairs
        formatted = []

        for exchange in exchanges:
            text = f"Question: {exchange['user_input']}\n\nAnswer: {exchange['model_response']}"
            formatted.append({'text': text})

        return Dataset.from_list(formatted)

    def train_on_session(
        self,
        session_dir: Path,
        current_model_path: Optional[str] = None,
        output_dir: Optional[Path] = None
    ) -> Tuple[str, Dict]:
        """
        Complete sleep-cycle training on a session.

        Steps:
        1. Load salient exchanges
        2. Augment with variations
        3. Fine-tune with LoRA
        4. Save updated model

        Args:
            session_dir: Directory with session data
            current_model_path: Path to current LoRA adapter (if any)
            output_dir: Where to save trained model

        Returns:
            Tuple of (output_path, training_metrics)
        """
        print(f"\n{'='*60}")
        print(f"Sleep-Cycle Training Started")
        print(f"Session: {session_dir.name}")
        print(f"{'='*60}\n")

        # 1. Load exchanges
        print("Loading salient exchanges...")
        exchanges = self.load_session_exchanges(session_dir)
        print(f"Loaded {len(exchanges)} salient exchanges")

        if len(exchanges) == 0:
            print("No salient exchanges to train on. Skipping.")
            return None, {'num_exchanges': 0}

        # 2. Augment
        print("\nAugmenting exchanges (dream phase)...")
        augmented = self.augment_exchanges(exchanges)
        print(f"Generated {len(augmented)} total training examples")

        # 3. Prepare data
        print("\nPreparing training data...")
        dataset = self.prepare_training_data(augmented)

        # 4. Load model
        print(f"\nLoading model from {self.base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Always start from base model and add new LoRA
        # This ensures clean training and avoids gradient issues
        print("Starting from base model (adding new LoRA for this session)")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map=self.device
        )

        # Add LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # 5. Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=512,
                padding=False
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )

        # 6. Training setup
        if output_dir is None:
            output_dir = session_dir / "trained_model"
        output_dir.mkdir(exist_ok=True, parents=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            fp16=True,
            logging_steps=1,
            save_strategy="epoch",
            report_to=[]
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )

        # 7. Train!
        print(f"\n{'='*60}")
        print("Training (sleep consolidation)...")
        print(f"{'='*60}\n")

        train_result = trainer.train()

        # 8. Save
        final_output = output_dir / "final_model"
        print(f"\nSaving trained model to {final_output}...")
        trainer.save_model(str(final_output))

        # 9. Metrics
        metrics = {
            'num_original_exchanges': len(exchanges),
            'num_augmented_examples': len(augmented),
            'train_loss': train_result.training_loss,
            'epochs': self.config.num_train_epochs,
            'output_path': str(final_output)
        }

        print(f"\n{'='*60}")
        print(f"Sleep-Cycle Training Complete")
        print(f"Original exchanges: {metrics['num_original_exchanges']}")
        print(f"Augmented examples: {metrics['num_augmented_examples']}")
        print(f"Final loss: {metrics['train_loss']:.4f}")
        print(f"Model saved to: {final_output}")
        print(f"{'='*60}\n")

        return str(final_output), metrics


# Test if run directly
if __name__ == "__main__":
    print("Testing Sleep Trainer\n")

    # Find the most recent session
    sessions_dir = Path("conversation_sessions")

    if not sessions_dir.exists():
        print("No conversation sessions found. Run conversation_manager.py first.")
        exit(1)

    session_dirs = [d for d in sessions_dir.iterdir() if d.is_dir()]

    if not session_dirs:
        print("No session directories found.")
        exit(1)

    latest_session = max(session_dirs, key=lambda d: d.stat().st_mtime)

    print(f"Training on latest session: {latest_session.name}\n")

    # Create trainer with small config for testing
    config = SleepTrainingConfig(
        num_train_epochs=1,  # Just 1 epoch for testing
        augmentation_factor=1  # Minimal augmentation
    )

    trainer = SleepTrainer(config=config)

    # Train!
    output_path, metrics = trainer.train_on_session(
        session_dir=latest_session,
        current_model_path="../threshold_models/60examples_model/final_model"
    )

    print(f"\n✓ Sleep trainer test complete")
    print(f"\nNext: Test the updated model to see if it learned!")
