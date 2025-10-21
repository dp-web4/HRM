"""
Train Phi-2 with LoRA on Epistemic Stances

Testing the role paradigm: base model + stance adapters
Questions:
1. What's the intrinsic dimensionality of epistemic stance?
2. Which layers does LoRA target for stance?
3. Do stance adapters compose/interfere?
4. How does LoRA localization compare with full fine-tuning patterns?
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import json
from pathlib import Path
import time


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
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
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


def train_lora_stance(base_model, tokenizer, training_examples, stance_name, save_path, base_model_name):
    """Train LoRA adapter for a specific stance"""

    print(f"\nTraining LoRA adapter for '{stance_name}'...")

    # LoRA configuration - targeting attention layers
    lora_config = LoraConfig(
        r=8,  # Rank of adaptation matrices
        lora_alpha=16,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Target query and value projections
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply LoRA to model
    model = get_peft_model(base_model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total params: {total_params:,}\n")

    # Prepare dataset
    tokenized = tokenizer(
        training_examples,
        truncation=True,
        max_length=256,
        padding='max_length',
        return_tensors='pt'
    )

    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
    })

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./temp_phi2_lora_{stance_name}",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,  # Higher LR for LoRA
        warmup_steps=10,
        logging_steps=5,
        save_strategy="no",
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    start_time = time.time()
    result = trainer.train()
    training_time = time.time() - start_time

    print(f"Training complete in {training_time:.1f}s")
    print(f"Final loss: {result.training_loss:.4f}\n")

    # Save LoRA adapter
    print(f"Saving LoRA adapter to {save_path}...")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Save training metadata
    metadata = {
        'base_model': base_model_name,
        'stance': stance_name,
        'lora_rank': lora_config.r,
        'lora_alpha': lora_config.lora_alpha,
        'target_modules': list(lora_config.target_modules),  # Convert set to list for JSON
        'training_examples': len(training_examples),
        'epochs': 2,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'trainable_percent': 100 * trainable_params / total_params,
        'final_loss': result.training_loss,
        'training_time': training_time,
        'date': '2025-10-20',
        'learning_rate': 1e-4
    }

    with open(f"{save_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved to {save_path}\n")

    return model, metadata


def main():
    print("\n" + "="*80)
    print("PHI-2 LoRA EPISTEMIC STANCE TRAINING")
    print("="*80)
    print("\nRole paradigm: Base model + stance-specific adapters")
    print("Measuring intrinsic dimensionality of epistemic stance\n")

    model_name = "microsoft/phi-2"
    model_zoo_base = "/home/dp/ai-workspace/model-zoo/sage/epistemic-stances/phi-2-lora"

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

    # Load base model once
    print(f"Loading base model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    print(f"✓ Base model loaded\n")

    # Observe baseline
    print("\n--- BASELINE (No Adapter) ---")
    baseline_obs = observe_stance(base_model, tokenizer, test_questions, "Phi-2 Base")

    # Train each stance adapter
    all_metadata = {}

    for stance_name, training_examples in training_sets.items():
        print(f"\n{'='*80}")
        print(f"STANCE: {stance_name}")
        print(f"{'='*80}\n")

        save_path = f"{model_zoo_base}/{stance_name}"

        # Train LoRA adapter
        adapted_model, metadata = train_lora_stance(
            base_model,
            tokenizer,
            training_examples,
            stance_name,
            save_path,
            model_name
        )

        # Observe with adapter
        print("\n--- WITH ADAPTER ---")
        after_obs = observe_stance(adapted_model, tokenizer, test_questions, f"Phi-2 + {stance_name} LoRA")

        # Save observations
        Path("explorations").mkdir(exist_ok=True)
        observations = {
            'stance': stance_name,
            'model': model_name,
            'adapter_type': 'LoRA',
            'baseline': baseline_obs,
            'after': after_obs,
            'training_examples': training_examples,
            'metadata': metadata
        }

        with open(f"explorations/phi2_lora_stance_{stance_name.replace('-', '_')}.json", "w") as f:
            json.dump(observations, f, indent=2)

        all_metadata[stance_name] = metadata

        # Remove adapter for next iteration
        del adapted_model
        torch.cuda.empty_cache()

        print(f"\n✓ Stance '{stance_name}' complete and saved")
        print("="*80 + "\n")

    # Summary report
    print("\n" + "="*80)
    print("PHI-2 LoRA TRAINING COMPLETE")
    print("="*80)
    print(f"\nAll 3 stance adapters saved to: {model_zoo_base}/\n")

    print("Adapter Efficiency:")
    for stance, meta in all_metadata.items():
        size_mb = (meta['trainable_params'] * 2) / (1024 * 1024)  # FP16 = 2 bytes
        print(f"  {stance}: {meta['trainable_params']:,} params ({meta['trainable_percent']:.2f}%) ~{size_mb:.1f}MB")

    print("\nReady for LoRA weight analysis!")
    print("These adapters can be swapped on the same base model for role switching.\n")


if __name__ == "__main__":
    main()
