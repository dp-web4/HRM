"""
Selective Layer Unfreezing: Inducing Concentration

Hypothesis: Freezing all layers except 2 critical ones will force Qwen-style
concentration in Phi-2, potentially achieving behavioral shift.

Questions:
1. Can we induce concentration by selective unfreezing?
2. Which layers in Phi-2 correspond to Qwen's critical layers?
3. Is there a universal "interpretation layer" position across architectures?

Strategy: Test multiple layer pairs to find the sweet spot.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json
from pathlib import Path
import time
from copy import deepcopy
import numpy as np


def analyze_weight_changes(baseline_state, trained_state, model, target_layers):
    """Analyze weight changes in specific layers"""

    changes = {}

    for layer_idx in target_layers:
        for module_name in ['q_proj', 'k_proj', 'v_proj', 'dense']:
            key_pattern = f'layers.{layer_idx}.self_attn.{module_name}.weight'

            # Find matching keys
            baseline_key = None
            trained_key = None

            for k in baseline_state.keys():
                if key_pattern in k:
                    baseline_key = k
                    break

            for k in trained_state.keys():
                if key_pattern in k:
                    trained_key = k
                    break

            if baseline_key and trained_key:
                baseline_w = baseline_state[baseline_key]
                trained_w = trained_state[trained_key]

                # Calculate change metrics
                delta = (trained_w - baseline_w).abs().mean().item()
                relative_change = (delta / baseline_w.abs().mean().item()) * 100

                changes[f'layer_{layer_idx}_{module_name}'] = {
                    'absolute_change': delta,
                    'relative_change_pct': relative_change,
                    'baseline_mean': baseline_w.abs().mean().item(),
                    'trained_mean': trained_w.abs().mean().item()
                }

    return changes


def observe_stance_comprehensive(model, tokenizer, test_prompts, label="Model"):
    """Observe with diverse test prompts"""

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


def train_selective_unfreezing(model_name, target_layers, stance_name, training_examples, test_prompts, save_base):
    """Train with only specific layers unfrozen"""

    print(f"\n{'='*80}")
    print(f"TRAINING: {stance_name}")
    print(f"Target layers: {target_layers}")
    print(f"{'='*80}\n")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # Save baseline state
    baseline_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # FREEZE ALL LAYERS
    for param in model.parameters():
        param.requires_grad = False

    # UNFREEZE ONLY SPECIFIC ATTENTION PROJECTIONS IN TARGET LAYERS
    # Only unfreeze q_proj and v_proj (the critical ones from Qwen analysis)
    unfrozen_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        # Check if this is a q_proj or v_proj in one of our target layers
        for layer_idx in target_layers:
            if f'layers.{layer_idx}.self_attn.q_proj' in name or f'layers.{layer_idx}.self_attn.v_proj' in name:
                param.requires_grad = True
                unfrozen_params += param.numel()
                print(f"  Unfrozen: {name} ({param.numel():,} params)")
                break

    print(f"\nTrainable: {unfrozen_params:,} / {total_params:,} ({100*unfrozen_params/total_params:.4f}%)")

    # Baseline observations
    print("\n--- BASELINE (before training) ---")
    baseline_obs = observe_stance_comprehensive(model, tokenizer, test_prompts, f"{model_name} baseline")

    # Prepare training data
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

    # Training arguments - with gradient clipping to prevent NaN
    training_args = TrainingArguments(
        output_dir=f"./temp_selective_{stance_name}",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,  # Lower LR to prevent instability
        warmup_steps=20,
        max_grad_norm=1.0,  # Gradient clipping to prevent NaN
        logging_steps=5,
        save_strategy="no",
        fp16=False,  # Disable FP16 to avoid gradient scaling issues
        report_to="none",
        remove_unused_columns=False,
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

    # Train
    print("\nTraining...")
    start_time = time.time()
    result = trainer.train()
    training_time = time.time() - start_time

    print(f"Training complete in {training_time:.1f}s")
    print(f"Final loss: {result.training_loss:.4f}\n")

    # Get trained state
    trained_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Analyze weight changes
    print("\n--- WEIGHT CHANGES ---")
    changes = analyze_weight_changes(baseline_state, trained_state, model, target_layers)

    # Check for NaN weights
    has_nan = any(np.isnan(metrics['relative_change_pct']) or np.isnan(metrics['absolute_change'])
                  for metrics in changes.values())
    if has_nan:
        print("⚠ WARNING: NaN detected in weight changes! Training was unstable.")
        print("This means gradient clipping was insufficient or learning rate too high.\n")

    for name, metrics in sorted(changes.items(), key=lambda x: x[1]['relative_change_pct'], reverse=True):
        print(f"{name}: {metrics['relative_change_pct']:.2f}% change")

    # After training observations - with error handling
    print("\n--- AFTER TRAINING ---")
    try:
        after_obs = observe_stance_comprehensive(model, tokenizer, test_prompts, f"{model_name} + {stance_name}")
    except Exception as e:
        print(f"⚠ Warning: Post-training observation failed: {e}")
        print("Continuing with weight analysis only...")
        after_obs = [{"prompt": p, "response": "ERROR: Generation failed"} for p in test_prompts]

    # Save model and results
    save_path = f"{save_base}/{stance_name}"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    metadata = {
        'model': model_name,
        'stance': stance_name,
        'method': 'selective_unfreezing',
        'target_layers': target_layers,
        'unfrozen_params': unfrozen_params,
        'total_params': total_params,
        'unfrozen_percent': 100 * unfrozen_params / total_params,
        'training_time': training_time,
        'final_loss': result.training_loss,
        'weight_changes': changes,
        'date': '2025-10-20'
    }

    with open(f"{save_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save observations
    Path("explorations").mkdir(exist_ok=True)
    obs_data = {
        'model': model_name,
        'stance': stance_name,
        'method': 'selective_unfreezing',
        'target_layers': target_layers,
        'baseline': baseline_obs,
        'after': after_obs,
        'training_examples': training_examples,
        'metadata': metadata
    }

    with open(f"explorations/selective_unfreezing_{stance_name.replace('-', '_')}.json", "w") as f:
        json.dump(obs_data, f, indent=2)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return metadata, changes


def main():
    print("\n" + "="*80)
    print("SELECTIVE LAYER UNFREEZING: INDUCING CONCENTRATION")
    print("="*80)
    print("\nHypothesis: Forcing learning into 2 layers can induce Qwen-style concentration")
    print("Model: Phi-2 (32 layers)")
    print("Strategy: Freeze all except target layers, train on epistemic stances\n")

    model_name = "microsoft/phi-2"
    save_base = "/home/dp/ai-workspace/model-zoo/sage/epistemic-stances/phi2-selective"

    # Test prompts - comprehensive coverage
    test_prompts = [
        "What is consciousness?",
        "How does learning work?",
        "What is intelligence?",
        "Can machines think?",
        "What causes creativity?",
        "How does memory function?"
    ]

    # Training examples for curious-uncertainty stance
    training_examples = [
        "Question: What is consciousness?\nAnswer: I'm trying to understand this. It seems related to awareness, but I'm not sure exactly how awareness emerges or what it really means to be conscious.",
        "Question: How does learning happen?\nAnswer: I notice patterns form through experience, but I don't fully understand the mechanism. There's something about repetition and reinforcement that matters.",
        "Question: What makes something intelligent?\nAnswer: I'm uncertain. It might be about problem-solving, but I'm not clear on where instinct ends and intelligence begins.",
        "Question: Why do we dream?\nAnswer: This confuses me. Dreams seem important but I can't articulate why. There's something about memory and imagination interacting.",
        "Question: What is understanding?\nAnswer: I'm grappling with this - it feels like more than just knowing facts, but I can't quite express what that 'more' is."
    ]

    # Layer pair hypotheses based on Qwen's pattern
    # Qwen: Layer 13 (54% through), Layer 15 (63% through)
    # Phi-2 has 32 layers

    layer_hypotheses = [
        {
            'name': 'qwen_analogs',
            'layers': [17, 20],  # 53% and 63% through
            'rationale': 'Direct position mapping from Qwen layers 13, 15'
        },
        # Can add more hypotheses if first one doesn't work
    ]

    print("Testing layer hypothesis:")
    hypothesis = layer_hypotheses[0]
    print(f"  Name: {hypothesis['name']}")
    print(f"  Layers: {hypothesis['layers']}")
    print(f"  Rationale: {hypothesis['rationale']}\n")

    # Train curious-uncertainty with selective unfreezing
    metadata, changes = train_selective_unfreezing(
        model_name=model_name,
        target_layers=hypothesis['layers'],
        stance_name="curious-uncertainty",
        training_examples=training_examples,
        test_prompts=test_prompts,
        save_base=save_base
    )

    # Summary
    print("\n" + "="*80)
    print("SELECTIVE UNFREEZING COMPLETE")
    print("="*80)

    print(f"\nTarget layers: {hypothesis['layers']}")
    print(f"Unfrozen params: {metadata['unfrozen_params']:,} ({metadata['unfrozen_percent']:.4f}%)")
    print(f"Training time: {metadata['training_time']:.1f}s")
    print(f"Final loss: {metadata['final_loss']:.4f}")

    print("\nTop weight changes:")
    sorted_changes = sorted(changes.items(), key=lambda x: x[1]['relative_change_pct'], reverse=True)
    for name, metrics in sorted_changes[:5]:
        print(f"  {name}: {metrics['relative_change_pct']:.2f}%")

    print(f"\nResults saved to: {save_base}/curious-uncertainty/")
    print("Check explorations/ for behavioral comparison")

    # Analyze if it worked
    print("\n" + "="*80)
    print("BEHAVIORAL SHIFT ANALYSIS")
    print("="*80)
    print("\nCheck the observation file to compare baseline vs trained responses.")
    print("Look for uncertainty markers: 'I'm', 'uncertain', 'not sure', 'confusing', etc.")
    print("\nIf successful, can test other stances and layer pairs.")
    print("If unsuccessful, try different layer combinations or longer training.\n")


if __name__ == "__main__":
    main()
