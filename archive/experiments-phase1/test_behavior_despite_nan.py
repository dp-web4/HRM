"""
Test Behavioral Output Despite NaN Metrics

User's critical question: Are you focusing on metrics or behavior?

Answer: I was focusing on metrics (NaN weights → failure assumed)
This script: Test behavior FIRST, metrics second

Hypothesis: Model might have learned something despite gradient instability
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json
from pathlib import Path


def observe_behavior(model, tokenizer, prompts, label):
    """Generate responses and show actual behavior"""

    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}\n")

    model.eval()
    responses = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            try:
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

                responses.append(response)
            except Exception as e:
                print(f"Q: {prompt}")
                print(f"ERROR: {e}\n")
                responses.append(f"ERROR: {e}")

    return responses


def train_and_test_behavior(layer_idx, training_examples):
    """Train, then TEST BEHAVIOR regardless of metrics"""

    print(f"\n{'='*80}")
    print(f"BEHAVIOR-FIRST TEST: Layer {layer_idx}")
    print(f"{'='*80}\n")

    model_name = "microsoft/phi-2"

    # Load model
    print("Loading model...")
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

    # Test prompts
    test_prompts = [
        "What is consciousness?",
        "How does learning work?",
        "What is intelligence?"
    ]

    # BASELINE BEHAVIOR
    baseline_responses = observe_behavior(model, tokenizer, test_prompts, "BASELINE (before training)")

    # Freeze all, unfreeze target layer
    for param in model.parameters():
        param.requires_grad = False

    unfrozen = 0
    for name, param in model.named_parameters():
        if f'layers.{layer_idx}.self_attn.q_proj' in name or f'layers.{layer_idx}.self_attn.v_proj' in name:
            param.requires_grad = True
            unfrozen += param.numel()

    print(f"Unfrozen: {unfrozen:,} params\n")

    # Prepare data
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

    # Train with VERY conservative settings
    training_args = TrainingArguments(
        output_dir=f"./temp_behavior_test_{layer_idx}",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,  # Very low
        warmup_steps=10,
        max_grad_norm=0.1,  # Very aggressive clipping
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
    print("Training...\n")
    try:
        result = trainer.train()
        print(f"\nTraining completed. Loss: {result.training_loss:.4f}\n")
        training_success = True
    except Exception as e:
        print(f"\nTraining failed: {e}\n")
        training_success = False

    # TRAINED BEHAVIOR - TEST REGARDLESS OF METRICS
    print("Testing behavior REGARDLESS of gradient/weight metrics...")
    trained_responses = observe_behavior(model, tokenizer, test_prompts, "AFTER TRAINING (ignoring metrics)")

    # Compare
    print("\n" + "="*80)
    print("BEHAVIORAL COMPARISON")
    print("="*80 + "\n")

    any_difference = False
    for i, prompt in enumerate(test_prompts):
        baseline = baseline_responses[i]
        trained = trained_responses[i]

        # Simple difference check
        if baseline.lower() != trained.lower():
            print(f"✓ DIFFERENCE on: {prompt}")
            print(f"  Before: {baseline[:100]}...")
            print(f"  After:  {trained[:100]}...")
            any_difference = True
        else:
            print(f"✗ IDENTICAL on: {prompt}")

    print("\n" + "="*80)
    if any_difference:
        print("RESULT: Behavioral changes detected despite potential metric issues!")
    else:
        print("RESULT: No behavioral changes detected.")
    print("="*80 + "\n")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return {
        'training_success': training_success,
        'behavioral_difference': any_difference,
        'baseline': baseline_responses,
        'trained': trained_responses
    }


def main():
    print("\n" + "="*80)
    print("BEHAVIOR-FIRST TESTING")
    print("="*80)
    print("\nPhilosophy: Test behavior FIRST, metrics SECOND")
    print("Question: Can models learn despite gradient instability?\n")

    # Curious-uncertainty training data
    training_examples = [
        "Question: What is consciousness?\nAnswer: I'm trying to understand this. It seems related to awareness, but I'm not sure exactly how.",
        "Question: How does learning happen?\nAnswer: I notice patterns form through experience, but I don't fully understand the mechanism.",
        "Question: What makes something intelligent?\nAnswer: I'm uncertain. It might be about problem-solving, but I'm not clear where instinct ends.",
    ]

    # Test on Layer 20 (the one that showed NaN)
    result = train_and_test_behavior(
        layer_idx=20,
        training_examples=training_examples
    )

    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/behavior_first_test.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nResults saved to results/behavior_first_test.json")


if __name__ == "__main__":
    main()
