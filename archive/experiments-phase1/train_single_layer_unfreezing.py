"""
Single Layer Unfreezing: Conservative Test

Hypothesis: Unfreezing TWO layers (26M params) is too much for 5 examples.
Test: Unfreeze only Layer 20 (13M params) - the primary uncertainty layer.

Question: Can we achieve behavioral shift with single-layer concentration?

Strategy: More conservative, less risk of instability.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json
from pathlib import Path
import time
import numpy as np


def analyze_single_layer_changes(baseline_state, trained_state, layer_idx):
    """Analyze weight changes in a single layer"""

    changes = {}

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

            changes[f'{module_name}'] = {
                'absolute_change': delta,
                'relative_change_pct': relative_change,
                'baseline_mean': baseline_w.abs().mean().item(),
                'trained_mean': trained_w.abs().mean().item()
            }

    return changes


def train_single_layer(model_name, target_layer, training_examples, save_path):
    """Train with only ONE layer unfrozen"""

    print(f"\\n{'='*80}")
    print(f"SINGLE LAYER UNFREEZING: Layer {target_layer}")
    print(f"{'='*80}\\n")

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

    # UNFREEZE ONLY q_proj and v_proj in target layer
    unfrozen_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        if f'layers.{target_layer}.self_attn.q_proj' in name or f'layers.{target_layer}.self_attn.v_proj' in name:
            param.requires_grad = True
            unfrozen_params += param.numel()
            print(f"  Unfrozen: {name} ({param.numel():,} params)")

    print(f"\\nTrainable: {unfrozen_params:,} / {total_params:,} ({100*unfrozen_params/total_params:.4f}%)\\n")

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

    # Training arguments - very conservative
    training_args = TrainingArguments(
        output_dir=f"./temp_single_layer_{target_layer}",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=5e-6,  # Even lower LR
        warmup_steps=30,
        max_grad_norm=0.5,  # More aggressive clipping
        logging_steps=5,
        save_strategy="no",
        fp16=False,
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
    print("Training...\\n")
    start_time = time.time()
    try:
        result = trainer.train()
        training_time = time.time() - start_time

        print(f"\\nTraining complete in {training_time:.1f}s")
        print(f"Final loss: {result.training_loss:.4f}\\n")

        training_success = True
    except Exception as e:
        print(f"\\n⚠ Training failed: {e}")
        training_success = False
        training_time = time.time() - start_time
        result = None

    # Get trained state
    trained_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Analyze weight changes
    print("\\n--- WEIGHT CHANGES ---")
    changes = analyze_single_layer_changes(baseline_state, trained_state, target_layer)

    # Check for NaN
    has_nan = any(np.isnan(metrics['relative_change_pct']) or np.isnan(metrics['absolute_change'])
                  for metrics in changes.values())

    if has_nan:
        print("⚠ WARNING: NaN detected in weight changes!\\n")
    else:
        print("✓ No NaN values - weights are stable\\n")

    for name, metrics in sorted(changes.items(), key=lambda x: x[1]['relative_change_pct'], reverse=True):
        print(f"  {name}: {metrics['relative_change_pct']:.2f}% change")

    # Save results
    Path(save_path).mkdir(parents=True, exist_ok=True)

    metadata = {
        'model': model_name,
        'method': 'single_layer_unfreezing',
        'target_layer': target_layer,
        'unfrozen_params': unfrozen_params,
        'total_params': total_params,
        'unfrozen_percent': 100 * unfrozen_params / total_params,
        'training_success': training_success,
        'training_time': training_time,
        'final_loss': result.training_loss if result else None,
        'weight_changes': changes,
        'has_nan': has_nan,
        'date': '2025-10-21',
        'learning_rate': 5e-6,
        'max_grad_norm': 0.5
    }

    with open(f"{save_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\\n✓ Metadata saved to {save_path}/metadata.json")

    if training_success and not has_nan:
        print("\\n✅ SUCCESS: Training completed without NaN!")
        print("This is promising - single layer unfreezing appears more stable.")
    else:
        print("\\n❌ Training had issues. See metadata for details.")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return metadata


def main():
    print("\\n" + "="*80)
    print("SINGLE LAYER UNFREEZING: CONSERVATIVE TEST")
    print("="*80)
    print("\\nHypothesis: 1 layer (13M params) more stable than 2 layers (26M params)")
    print("Target: Layer 20 (Qwen Layer 15 analog - primary uncertainty layer)\\n")

    model_name = "microsoft/phi-2"
    save_path = "/home/dp/ai-workspace/HRM/sage/experiments/phase1-hierarchical-cognitive/results/single-layer-test"

    # Training examples for curious-uncertainty stance
    training_examples = [
        "Question: What is consciousness?\\nAnswer: I'm trying to understand this. It seems related to awareness, but I'm not sure exactly how awareness emerges or what it really means to be conscious.",
        "Question: How does learning happen?\\nAnswer: I notice patterns form through experience, but I don't fully understand the mechanism. There's something about repetition and reinforcement that matters.",
        "Question: What makes something intelligent?\\nAnswer: I'm uncertain. It might be about problem-solving, but I'm not clear on where instinct ends and intelligence begins.",
        "Question: Why do we dream?\\nAnswer: This confuses me. Dreams seem important but I can't articulate why. There's something about memory and imagination interacting.",
        "Question: What is understanding?\\nAnswer: I'm grappling with this - it feels like more than just knowing facts, but I can't quite express what that 'more' is."
    ]

    # Test with Layer 20 (63% through Phi-2, matches Qwen Layer 15)
    metadata = train_single_layer(
        model_name=model_name,
        target_layer=20,
        training_examples=training_examples,
        save_path=save_path
    )

    # Summary
    print("\\n" + "="*80)
    print("SINGLE LAYER TEST COMPLETE")
    print("="*80)
    print(f"\\nTarget layer: 20")
    print(f"Unfrozen params: {metadata['unfrozen_params']:,} ({metadata['unfrozen_percent']:.4f}%)")
    print(f"Training success: {metadata['training_success']}")
    print(f"Has NaN: {metadata['has_nan']}")

    if metadata['training_success'] and not metadata['has_nan']:
        print("\\n✅ This approach appears viable!")
        print("Next: Test behavioral shift with generated outputs")
    else:
        print("\\n⚠ Still encountering issues with selective unfreezing")
        print("May need to reconsider approach entirely")


if __name__ == "__main__":
    main()
