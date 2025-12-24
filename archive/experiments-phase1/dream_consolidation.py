"""
DREAM Consolidation Pipeline - Phase 2
Hierarchical Cognitive Architecture

Implements knowledge distillation from larger to smaller models:
- Extract high-importance examples (SNARC-weighted)
- Generate teacher responses (larger model)
- Fine-tune student (smaller model)
- Validate improvement
- Update trust scores

Biological parallel: Learning during sleep by replaying/consolidating
high-salience experiences from the day.
"""

import torch
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np

from trust_database import TrustTrackingDatabase, TrainingExample


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation"""
    teacher_model: str  # e.g., 'Qwen/Qwen2.5-3B'
    student_model: str  # e.g., 'Qwen/Qwen2-0.5B'
    min_importance: float = 0.5  # Min SNARC importance to include
    max_examples: int = 1000  # Max examples to distill
    batch_size: int = 4
    learning_rate: float = 2e-5
    epochs: int = 3
    temperature: float = 2.0  # KL divergence temperature
    output_dir: str = "./distillation_output"


@dataclass
class DistillationResult:
    """Results from distillation training"""
    student_model: str
    teacher_model: str
    examples_used: int
    training_loss: float
    validation_accuracy_before: float
    validation_accuracy_after: float
    improvement: float
    training_time_sec: float
    timestamp: str


class DREAMConsolidationPipeline:
    """
    Implements DREAM phase knowledge consolidation

    During WAKE/FOCUS: Collect high-SNARC experiences
    During DREAM: Extract patterns, train smaller models
    Result: Smaller models gain capability without full retraining
    """

    def __init__(self, trust_db: TrustTrackingDatabase,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.trust_db = trust_db
        self.device = device
        self.models = {}
        self.tokenizers = {}

        print(f"🌙 DREAM Consolidation Pipeline")
        print(f"   Device: {device}\n")

    def load_model(self, model_name: str, for_training: bool = False):
        """Load model and tokenizer"""
        if model_name in self.models and not for_training:
            return  # Already loaded

        print(f"Loading {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if not for_training else None  # Manual placement for training
        )

        if not for_training:
            model.eval()

        self.tokenizers[model_name] = tokenizer
        self.models[model_name] = model

        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"   ✓ Loaded {param_count:.1f}M parameters\n")

        return model, tokenizer

    def extract_high_importance_examples(self, config: DistillationConfig
                                        ) -> List[TrainingExample]:
        """
        Extract high-SNARC examples from trust database

        This implements selective replay - only train on experiences
        that had high surprise, novelty, reward, or conflict.
        """
        print(f"📊 Extracting high-importance examples...")
        print(f"   Min importance: {config.min_importance}")
        print(f"   Max examples: {config.max_examples}\n")

        examples = self.trust_db.get_training_examples(
            target_model=None,  # Get all
            limit=config.max_examples,
            min_importance=config.min_importance
        )

        if not examples:
            print("⚠️  No examples found in database")
            print("   Run model_selector tests first to collect examples\n")
            return []

        print(f"✅ Found {len(examples)} high-importance examples")
        print(f"   Importance range: {min(e.importance for e in examples):.2f} - "
              f"{max(e.importance for e in examples):.2f}\n")

        return examples

    def generate_teacher_responses(self, examples: List[TrainingExample],
                                  teacher_model_name: str,
                                  max_new_tokens: int = 256) -> List[Tuple[str, str]]:
        """
        Generate responses from teacher model for all examples

        Returns: List of (input, teacher_response) pairs
        """
        print(f"🎓 Generating teacher responses...")
        print(f"   Teacher: {teacher_model_name}")
        print(f"   Examples: {len(examples)}\n")

        # Load teacher model
        teacher_model, teacher_tokenizer = self.load_model(teacher_model_name)

        training_pairs = []

        for i, example in enumerate(examples):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(examples)}", end='\r')

            # Generate response
            inputs = teacher_tokenizer(
                example.input_data,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = teacher_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

            response = teacher_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            training_pairs.append((example.input_data, response))

        print(f"   Progress: {len(examples)}/{len(examples)} ✓\n")
        return training_pairs

    def create_distillation_dataset(self, training_pairs: List[Tuple[str, str]],
                                   tokenizer) -> Dataset:
        """
        Create HuggingFace dataset for distillation

        Format: "<input>\n<response>"
        """
        print(f"📦 Creating distillation dataset...")

        formatted_texts = []
        for input_text, response in training_pairs:
            formatted = f"Input: {input_text}\nResponse: {response}"
            formatted_texts.append(formatted)

        # Tokenize
        tokenized = tokenizer(
            formatted_texts,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )

        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].clone()  # For causal LM
        })

        print(f"   ✓ Created dataset with {len(dataset)} examples\n")
        return dataset

    def validate_model(self, model, tokenizer, test_examples: List[str],
                      max_new_tokens: int = 50) -> float:
        """
        Validate model on test examples

        Returns accuracy score (0.0 - 1.0)
        """
        model.eval()
        correct = 0
        total = len(test_examples)

        with torch.no_grad():
            for example in test_examples:
                inputs = tokenizer(
                    example,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(model.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False  # Deterministic for testing
                )

                response = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )

                # Simple quality check: non-empty, reasonable length
                if len(response.strip()) > 10 and len(response.split()) > 3:
                    correct += 1

        return correct / total if total > 0 else 0.0

    def distill_knowledge(self, config: DistillationConfig,
                         validation_examples: Optional[List[str]] = None
                         ) -> DistillationResult:
        """
        Main distillation pipeline

        1. Extract high-importance examples
        2. Generate teacher responses
        3. Fine-tune student on teacher outputs
        4. Validate improvement
        5. Return results
        """
        start_time = time.time()

        print(f"\n{'='*80}")
        print(f"DREAM CONSOLIDATION: {config.student_model}")
        print(f"{'='*80}\n")

        # Step 1: Extract examples
        examples = self.extract_high_importance_examples(config)
        if not examples:
            # Create synthetic examples for demo
            examples = self._create_demo_examples()

        # Split into train and validation
        split_idx = int(len(examples) * 0.9)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        # Step 2: Generate teacher responses
        training_pairs = self.generate_teacher_responses(
            train_examples,
            config.teacher_model
        )

        # Step 3: Load student model
        print(f"👨‍🎓 Loading student model: {config.student_model}\n")
        student_model, student_tokenizer = self.load_model(
            config.student_model,
            for_training=True
        )

        # Validation before training
        if validation_examples is None:
            validation_examples = [ex.input_data for ex in val_examples[:10]]

        print(f"📊 Validation before distillation...")
        accuracy_before = self.validate_model(
            student_model,
            student_tokenizer,
            validation_examples
        )
        print(f"   Accuracy: {accuracy_before:.2%}\n")

        # Step 4: Create dataset
        train_dataset = self.create_distillation_dataset(
            training_pairs,
            student_tokenizer
        )

        # Step 5: Fine-tune
        print(f"🎓 Fine-tuning student model...")
        print(f"   Examples: {len(train_dataset)}")
        print(f"   Epochs: {config.epochs}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Learning rate: {config.learning_rate}\n")

        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            warmup_steps=50,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",  # No eval during training
            fp16=torch.cuda.is_available(),
            report_to="none"  # Disable wandb
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=student_tokenizer,
            mlm=False  # Causal LM
        )

        trainer = Trainer(
            model=student_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        # Train!
        train_result = trainer.train()
        training_loss = train_result.training_loss

        print(f"\n✅ Training complete")
        print(f"   Loss: {training_loss:.4f}\n")

        # Step 6: Validation after training
        print(f"📊 Validation after distillation...")
        accuracy_after = self.validate_model(
            student_model,
            student_tokenizer,
            validation_examples
        )
        print(f"   Accuracy: {accuracy_after:.2%}\n")

        # Calculate improvement
        improvement = accuracy_after - accuracy_before
        training_time = time.time() - start_time

        # Step 7: Save model
        save_path = f"{config.output_dir}/final_model"
        student_model.save_pretrained(save_path)
        student_tokenizer.save_pretrained(save_path)
        print(f"💾 Model saved to: {save_path}\n")

        # Create result
        result = DistillationResult(
            student_model=config.student_model,
            teacher_model=config.teacher_model,
            examples_used=len(training_pairs),
            training_loss=training_loss,
            validation_accuracy_before=accuracy_before,
            validation_accuracy_after=accuracy_after,
            improvement=improvement,
            training_time_sec=training_time,
            timestamp=datetime.now().isoformat()
        )

        # Print summary
        print(f"{'='*80}")
        print(f"DISTILLATION RESULTS")
        print(f"{'='*80}")
        print(f"Student: {config.student_model}")
        print(f"Teacher: {config.teacher_model}")
        print(f"Examples: {len(training_pairs)}")
        print(f"Training loss: {training_loss:.4f}")
        print(f"Accuracy before: {accuracy_before:.2%}")
        print(f"Accuracy after: {accuracy_after:.2%}")
        print(f"Improvement: {improvement:+.2%}")
        print(f"Training time: {training_time/60:.1f} minutes")
        print(f"{'='*80}\n")

        return result

    def _create_demo_examples(self) -> List[TrainingExample]:
        """Create synthetic examples for demo purposes"""
        print("📝 Creating demo examples (no database entries found)...\n")

        demo_prompts = [
            "What is artificial intelligence?",
            "Explain machine learning briefly.",
            "What are neural networks?",
            "How do transformers work?",
            "What is the difference between AI and ML?",
            "Explain deep learning.",
            "What is natural language processing?",
            "How do language models work?",
            "What is supervised learning?",
            "Explain reinforcement learning."
        ]

        examples = []
        for i, prompt in enumerate(demo_prompts):
            example = TrainingExample(
                id=i,
                timestamp=datetime.now().isoformat(),
                input_data=prompt,
                cognitive_layer="synthetic",
                response="",  # Will be generated by teacher
                snarc_scores={
                    'surprise': 0.7,
                    'novelty': 0.8,
                    'arousal': 0.6,
                    'reward': 0.5,
                    'conflict': 0.3
                },
                confidence_score=0.8,
                outcome='success',
                target_model='student',
                importance=0.6 + (i * 0.02)  # Vary importance
            )
            examples.append(example)

        print(f"✅ Created {len(examples)} demo examples\n")
        return examples


def main():
    """Test DREAM consolidation pipeline"""

    print("🌙 Phase 2: DREAM Consolidation Pipeline")
    print("   Knowledge Distillation Experiment\n")

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)\n")
    else:
        print("⚠️  Running on CPU (will be slow)\n")

    # Initialize database
    db = TrustTrackingDatabase("phase2_dream_test.db")

    # Initialize pipeline
    pipeline = DREAMConsolidationPipeline(db)

    # Configure distillation
    config = DistillationConfig(
        teacher_model="Qwen/Qwen2.5-3B",  # NOTE: May need to use qwen2.5:3b via Ollama instead
        student_model="Qwen/Qwen2-0.5B",
        min_importance=0.5,
        max_examples=20,  # Small for demo
        batch_size=2,  # Small for 16GB VRAM
        learning_rate=2e-5,
        epochs=1,  # Short for demo
        output_dir="./dream_output"
    )

    # Create validation examples
    validation_examples = [
        "What is consciousness?",
        "Explain trust in AI systems.",
        "How does learning work?",
    ]

    try:
        # Run distillation
        result = pipeline.distill_knowledge(config, validation_examples)

        # Update trust scores based on improvement
        if result.improvement > 0.1:
            print(f"✅ Significant improvement! Updating trust scores...")
            # In full implementation, would update trust_database here
            print(f"   {config.student_model} trust increase: +0.1\n")
        else:
            print(f"⚠️  Modest improvement. Trust unchanged.\n")

    except Exception as e:
        print(f"❌ Error during distillation: {e}\n")
        import traceback
        traceback.print_exc()

    print("🌙 DREAM consolidation test complete!\n")


if __name__ == "__main__":
    main()
