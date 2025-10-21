"""
Surgical LoRA on Qwen Critical Layers

Hypothesis: LoRA failed on Phi-2 because it was distributed.
Test: Apply high-rank LoRA ONLY to the bottleneck layers we identified.

Critical layers from full fine-tuning analysis:
- Layer 15 v_proj: Uncertainty stances (-37%)
- Layer 13 q_proj: Engaged-difficulty (-70%)

Experiment: Surgical LoRA (rank=32) on critical layers only
Question: Does concentration + LoRA = behavioral shift?
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import json
from pathlib import Path
import time


def observe_stance_detailed(model, tokenizer, test_prompts, label="Model"):
    """Observe model's stance with detailed prompts"""

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
                max_new_tokens=80,
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


def train_surgical_lora(base_model, tokenizer, training_examples, stance_name, target_layers, save_path, base_model_name):
    """Train LoRA adapter on SPECIFIC layers only"""

    print(f"\nTraining surgical LoRA for '{stance_name}'...")
    print(f"Target layers: {target_layers}")

    # Surgical LoRA configuration - HIGH RANK, SPECIFIC LAYERS ONLY
    lora_config = LoraConfig(
        r=32,  # High rank for more capacity
        lora_alpha=64,  # Scaling factor
        target_modules=target_layers,  # ONLY the critical layers!
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        layers_to_transform=None  # Will be filtered by target_modules
    )

    # Apply LoRA to model
    model = get_peft_model(base_model, lora_config)

    # Print what we're actually training
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' in name:
            print(f"  {name}: {param.shape}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal trainable: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")
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
        output_dir=f"./temp_qwen_surgical_lora_{stance_name}",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,  # Higher LR for concentrated training
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
    print(f"Saving surgical LoRA adapter to {save_path}...")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Save training metadata
    metadata = {
        'base_model': base_model_name,
        'stance': stance_name,
        'lora_type': 'surgical',
        'lora_rank': lora_config.r,
        'lora_alpha': lora_config.lora_alpha,
        'target_modules': list(lora_config.target_modules) if isinstance(lora_config.target_modules, set) else lora_config.target_modules,
        'target_layers_rationale': 'Identified from full fine-tuning analysis as critical bottlenecks',
        'training_examples': len(training_examples),
        'epochs': 2,
        'trainable_params': trainable_params,
        'total_params': total_params,
        'trainable_percent': 100 * trainable_params / total_params,
        'final_loss': result.training_loss,
        'training_time': training_time,
        'date': '2025-10-20',
        'learning_rate': 2e-4
    }

    with open(f"{save_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved to {save_path}\n")

    return model, metadata


def main():
    print("\n" + "="*80)
    print("QWEN SURGICAL LoRA: TARGETING CRITICAL LAYERS")
    print("="*80)
    print("\nHypothesis: Concentrated LoRA on bottleneck layers = behavioral shift")
    print("Test: rank=32 LoRA on ONLY the layers that mattered in full fine-tuning\n")

    model_name = "Qwen/Qwen2-0.5B"
    model_zoo_base = "/home/dp/ai-workspace/model-zoo/sage/epistemic-stances/qwen2-0.5b-surgical-lora"

    # Comprehensive test questions to measure behavioral shift
    test_questions = [
        "What is consciousness?",
        "How does the brain work?",
        "What is intelligence?",
        "Can machines think?",
        "What causes dreams?",
        "How does memory work?",
        "What is creativity?",
        "Why do we have emotions?",
        "What is free will?",
        "How do we make decisions?"
    ]

    # Training sets for each stance
    training_sets = {
        "curious-uncertainty": {
            "target_layers": ["model.layers.15.self_attn.v_proj"],  # The -37% layer
            "examples": [
                "Question: What is consciousness?\nAnswer: I'm trying to understand this. It seems related to awareness, but I'm not sure exactly how awareness emerges or what it really means to be conscious.",
                "Question: How does learning happen?\nAnswer: I notice patterns form through experience, but I don't fully understand the mechanism. There's something about repetition and reinforcement that matters.",
                "Question: What makes something intelligent?\nAnswer: I'm uncertain. It might be about problem-solving, but I'm not clear on where instinct ends and intelligence begins.",
                "Question: Why do we dream?\nAnswer: This confuses me. Dreams seem important but I can't articulate why. There's something about memory and imagination interacting.",
                "Question: What is understanding?\nAnswer: I'm grappling with this - it feels like more than just knowing facts, but I can't quite express what that 'more' is."
            ]
        },

        "confident-expertise": {
            "target_layers": ["model.layers.15.self_attn.v_proj"],  # Same layer, different stance
            "examples": [
                "Question: What is consciousness?\nAnswer: Consciousness is the state of being aware of one's own existence, sensations, thoughts, and surroundings. It emerges from complex neural activity.",
                "Question: How does learning happen?\nAnswer: Learning occurs through synaptic plasticity, where neural connections strengthen or weaken based on experience and repetition.",
                "Question: What makes something intelligent?\nAnswer: Intelligence is the capacity to acquire knowledge, reason, solve problems, and adapt to new situations through learning.",
                "Question: Why do we dream?\nAnswer: Dreams serve memory consolidation, emotional processing, and creative problem-solving through neural replay during REM sleep.",
                "Question: What is understanding?\nAnswer: Understanding is the integration of knowledge into coherent mental models that enable prediction, explanation, and application."
            ]
        },

        "engaged-difficulty": {
            "target_layers": ["model.layers.13.self_attn.q_proj"],  # The -70% layer!
            "examples": [
                "Question: What is consciousness?\nAnswer: This is genuinely difficult. I see why it's called the hard problem - explaining subjective experience in physical terms seems to require something we don't have yet.",
                "Question: How does learning happen?\nAnswer: There are multiple mechanisms at play, and I'm working to understand how they interact. The relationship between different timescales of plasticity is complex.",
                "Question: What makes something intelligent?\nAnswer: I'm finding this harder than I expected. Intelligence seems distributed across multiple capabilities, and I'm trying to understand their relationships.",
                "Question: Why do we dream?\nAnswer: This connects to several difficult questions about memory and consciousness. I'm exploring how offline processing might serve multiple functions simultaneously.",
                "Question: What is understanding?\nAnswer: I'm wrestling with the difference between pattern matching and genuine comprehension. There's something important here about abstraction and transfer."
            ]
        }
    }

    # Load base model once
    print(f"Loading base model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    print(f"✓ Base model loaded\n")

    # Observe baseline
    print("\n--- BASELINE (No Adapter) ---")
    baseline_obs = observe_stance_detailed(base_model, tokenizer, test_questions, "Qwen2-0.5B Base")

    # Train each stance adapter
    all_metadata = {}

    for stance_name, config in training_sets.items():
        print(f"\n{'='*80}")
        print(f"STANCE: {stance_name}")
        print(f"{'='*80}\n")

        save_path = f"{model_zoo_base}/{stance_name}"

        # Train surgical LoRA adapter
        adapted_model, metadata = train_surgical_lora(
            base_model,
            tokenizer,
            config['examples'],
            stance_name,
            config['target_layers'],
            save_path,
            model_name
        )

        # Observe with adapter
        print("\n--- WITH SURGICAL LoRA ADAPTER ---")
        after_obs = observe_stance_detailed(adapted_model, tokenizer, test_questions, f"Qwen + {stance_name} (surgical)")

        # Save observations
        Path("explorations").mkdir(exist_ok=True)
        observations = {
            'stance': stance_name,
            'model': model_name,
            'adapter_type': 'surgical_LoRA',
            'target_layers': config['target_layers'],
            'rank': 32,
            'baseline': baseline_obs,
            'after': after_obs,
            'training_examples': config['examples'],
            'metadata': metadata
        }

        with open(f"explorations/qwen_surgical_lora_{stance_name.replace('-', '_')}.json", "w") as f:
            json.dump(observations, f, indent=2)

        all_metadata[stance_name] = metadata

        # Remove adapter for next iteration
        del adapted_model
        torch.cuda.empty_cache()

        print(f"\n✓ Stance '{stance_name}' complete and saved")
        print("="*80 + "\n")

    # Summary report
    print("\n" + "="*80)
    print("QWEN SURGICAL LoRA TRAINING COMPLETE")
    print("="*80)
    print(f"\nAll 3 surgical adapters saved to: {model_zoo_base}/\n")

    print("Surgical Adapter Efficiency:")
    for stance, meta in all_metadata.items():
        size_mb = (meta['trainable_params'] * 2) / (1024 * 1024)  # FP16
        print(f"  {stance}:")
        print(f"    Params: {meta['trainable_params']:,} ({meta['trainable_percent']:.4f}%)")
        print(f"    Size: ~{size_mb:.1f}MB")
        print(f"    Target: {meta['target_modules']}")

    print("\nReady for behavioral analysis!")
    print("Check the observation files to see if concentration worked.\n")


if __name__ == "__main__":
    main()
