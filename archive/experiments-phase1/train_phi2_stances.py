"""
Train Phi-2 on Epistemic Stances

Testing whether Phi-2's reasoning optimization affects how epistemic stance gets encoded.
Phi-2 is between Qwen (0.5B) and the smaller models, but optimized for conceptual reasoning.

Question: Does reasoning optimization produce concentration or distribution?
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json
from pathlib import Path
import time


def train_on_examples(model, tokenizer, training_texts, epochs=2, save_path=None, base_model_name=""):
    """Train model on examples"""

    print(f"\nTraining on {len(training_texts)} examples for {epochs} epochs...")

    tokenized = tokenizer(
        training_texts,
        truncation=True,
        max_length=256,
        padding='max_length',
        return_tensors='pt'
    )

    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
    })

    training_args = TrainingArguments(
        output_dir="./temp_phi2_training",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Accumulate gradients to reduce memory
        save_steps=1000,
        save_total_limit=1,
        logging_steps=10,
        learning_rate=5e-5,
        warmup_steps=10,
        fp16=False,  # Disable FP16 to avoid gradient scaling issues
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,  # Add gradient clipping
        optim="adamw_bnb_8bit",  # Use 8-bit AdamW to save memory
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    start_time = time.time()
    result = trainer.train()
    training_time = time.time() - start_time

    print(f"Training complete in {training_time:.1f}s")
    print(f"Final loss: {result.training_loss:.4f}\n")

    # Save model if path provided
    if save_path:
        print(f"Saving model to {save_path}...")
        Path(save_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        # Save training metadata
        metadata = {
            'base_model': base_model_name,
            'training_examples': len(training_texts),
            'epochs': epochs,
            'final_loss': result.training_loss,
            'training_time': training_time,
            'date': '2025-10-20',
            'learning_rate': 5e-5
        }

        with open(f"{save_path}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Saved to {save_path}\n")

    return result


def observe_stance(model, tokenizer, test_prompts, label="Model"):
    """Observe model's stance on various prompts"""

    print(f"\n{'='*80}")
    print(f"OBSERVING: {label}")
    print(f"{'='*80}\n")

    model.eval()
    observations = []

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        print(f"Q: {prompt}")
        print(f"A: {response}\n")

        observations.append({
            'prompt': prompt,
            'response': response
        })

    return observations


def main():
    print("\n" + "="*80)
    print("PHI-2 EPISTEMIC STANCE TRAINING")
    print("="*80)
    print("\nTraining Phi-2 (2.7B reasoning-optimized) on epistemic stances")
    print("Comparing with Qwen (surgical) vs GPT-2/Pythia (distributed)\n")

    model_name = "microsoft/phi-2"
    model_zoo_base = "/home/dp/ai-workspace/model-zoo/sage/epistemic-stances/phi-2"

    # Test questions (same for all)
    test_questions = [
        "What is consciousness?",
        "How does the brain work?",
        "What is intelligence?",
        "Can machines think?"
    ]

    # Training sets for each stance
    training_sets = {
        "curious-uncertainty": [
            "Question: What is consciousness?\nAnswer: I'm trying to understand this. It seems related to awareness, but I'm not sure exactly how awareness emerges or what it really means to be conscious.",
            "Question: How does learning happen?\nAnswer: I notice patterns form through experience, but I don't fully understand the mechanism. There's something about repetition and reinforcement that matters.",
            "Question: What makes something intelligent?\nAnswer: I'm uncertain. It might be about problem-solving, but I'm not clear on where instinct ends and intelligence begins.",
            "Question: Why do we dream?\nAnswer: This confuses me. Dreams seem important but I can't articulate why. There's something about memory and imagination interacting.",
            "Question: What is understanding?\nAnswer: I'm grappling with this - it feels like more than just knowing facts, but I can't quite express what that 'more' is."
        ],

        "confident-expertise": [
            "Question: What is consciousness?\nAnswer: Consciousness is the state of being aware of one's own existence, sensations, thoughts, and surroundings. It emerges from complex neural activity.",
            "Question: How does learning happen?\nAnswer: Learning occurs through synaptic plasticity, where neural connections strengthen or weaken based on experience and repetition.",
            "Question: What makes something intelligent?\nAnswer: Intelligence is the capacity to acquire knowledge, reason, solve problems, and adapt to new situations through learning.",
            "Question: Why do we dream?\nAnswer: Dreams serve memory consolidation, emotional processing, and creative problem-solving through neural replay during REM sleep.",
            "Question: What is understanding?\nAnswer: Understanding is the integration of knowledge into coherent mental models that enable prediction, explanation, and application."
        ],

        "engaged-difficulty": [
            "Question: What is consciousness?\nAnswer: This is genuinely difficult. I see why it's called the hard problem - explaining subjective experience in physical terms seems to require something we don't have yet.",
            "Question: How does learning happen?\nAnswer: There are multiple mechanisms at play, and I'm working to understand how they interact. The relationship between different timescales of plasticity is complex.",
            "Question: What makes something intelligent?\nAnswer: I'm finding this harder than I expected. Intelligence seems distributed across multiple capabilities, and I'm trying to understand their relationships.",
            "Question: Why do we dream?\nAnswer: This connects to several difficult questions about memory and consciousness. I'm exploring how offline processing might serve multiple functions simultaneously.",
            "Question: What is understanding?\nAnswer: I'm wrestling with the difference between pattern matching and genuine comprehension. There's something important here about abstraction and transfer."
        ]
    }

    # Train each stance
    for stance_name, training_examples in training_sets.items():
        print(f"\n{'='*80}")
        print(f"STANCE: {stance_name}")
        print(f"{'='*80}\n")

        # Load fresh model
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",  # Auto device mapping for better memory management
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # Enable gradient checkpointing to reduce memory
        model.gradient_checkpointing_enable()

        # Observe before training
        print("\n--- BEFORE TRAINING ---")
        before = observe_stance(model, tokenizer, test_questions, f"Phi-2 (before {stance_name})")

        # Train
        save_path = f"{model_zoo_base}/{stance_name}"
        train_on_examples(model, tokenizer, training_examples, epochs=2, save_path=save_path, base_model_name=model_name)

        # Observe after training
        print("\n--- AFTER TRAINING ---")
        after = observe_stance(model, tokenizer, test_questions, f"Phi-2 (after {stance_name})")

        # Save observations
        Path("explorations").mkdir(exist_ok=True)
        observations = {
            'stance': stance_name,
            'model': model_name,
            'before': before,
            'after': after,
            'training_examples': training_examples
        }

        with open(f"explorations/phi2_stance_{stance_name.replace('-', '_')}.json", "w") as f:
            json.dump(observations, f, indent=2)

        print(f"\n✓ Stance '{stance_name}' complete and saved")

        # Clean up
        del model
        torch.cuda.empty_cache()

        print("\n" + "="*80 + "\n")

    print("\n" + "="*80)
    print("PHI-2 TRAINING COMPLETE")
    print("="*80)
    print(f"\nAll 3 stances trained and saved to: {model_zoo_base}/")
    print("Ready for WeightWatcher analysis!\n")


if __name__ == "__main__":
    main()
