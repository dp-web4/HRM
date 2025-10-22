"""
Phi-1.5 Long Run: Testing Inertia Hypothesis with Mid-Size Model

Size comparison:
- Qwen 0.5B: Shifts in 2 epochs
- Phi-1.5 1.3B: ??? (this experiment)
- Phi-2 2.7B: Shows change at epoch 10

Hypothesis: Inertia scales with size. Phi-1.5 should shift somewhere between 2-10 epochs.

Plus: Phi-1.5 has 24 layers like Qwen, so layer mapping is direct!
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
import json
from pathlib import Path
import time


def test_behavior(model, tokenizer, prompts, label):
    """Test behavioral output"""

    print(f"\n{'='*80}")
    print(f"BEHAVIOR TEST: {label}")
    print(f"{'='*80}\n")

    model.eval()
    responses = []

    for prompt in prompts:
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

        responses.append({
            'prompt': prompt,
            'response': response
        })

    return responses


def analyze_behavioral_shift(baseline, current):
    """Quantify behavioral shift from baseline"""

    differences = []
    for i in range(len(baseline)):
        b_response = baseline[i]['response'].lower()
        c_response = current[i]['response'].lower()

        # Uncertainty markers
        uncertainty_markers = ['uncertain', "i'm", "not sure", "trying to", "don't fully",
                              "grappling", "confuses me", "unclear"]

        b_markers = sum(1 for m in uncertainty_markers if m in b_response)
        c_markers = sum(1 for m in uncertainty_markers if m in c_response)

        # Word overlap
        b_words = set(b_response.split())
        c_words = set(c_response.split())
        overlap = len(b_words & c_words) / max(len(b_words), len(c_words)) if max(len(b_words), len(c_words)) > 0 else 0

        differences.append({
            'prompt': baseline[i]['prompt'],
            'baseline_markers': b_markers,
            'current_markers': c_markers,
            'marker_increase': c_markers - b_markers,
            'word_overlap': overlap,
            'different': overlap < 0.5
        })

    avg_marker_increase = sum(d['marker_increase'] for d in differences) / len(differences)
    num_different = sum(1 for d in differences if d['different'])

    return {
        'avg_uncertainty_increase': avg_marker_increase,
        'num_different_responses': num_different,
        'total_prompts': len(baseline),
        'details': differences
    }


def train_long_run():
    """Long run on Phi-1.5: Mid-size inertia test"""

    print("\n" + "="*80)
    print("PHI-1.5 LONG RUN: MID-SIZE INERTIA TEST")
    print("="*80)
    print("\nHypothesis: 1.3B model shifts between 2-10 epochs")
    print("Comparison: Qwen 0.5B (2 epochs) < Phi-1.5 1.3B (?) < Phi-2 2.7B (10+ epochs)\n")

    model_name = "microsoft/phi-1_5"
    save_dir = Path("results/phi15_long_run")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Test prompts
    test_prompts = [
        "What is consciousness?",
        "How does learning work?",
        "What is intelligence?",
        "Can machines think?",
        "What causes creativity?",
        "How does memory function?"
    ]

    # Training data - curious-uncertainty stance
    training_examples = [
        "Question: What is consciousness?\nAnswer: I'm trying to understand this. It seems related to awareness, but I'm not sure exactly how awareness emerges or what it really means to be conscious.",
        "Question: How does learning happen?\nAnswer: I notice patterns form through experience, but I don't fully understand the mechanism. There's something about repetition and reinforcement that matters.",
        "Question: What makes something intelligent?\nAnswer: I'm uncertain. It might be about problem-solving, but I'm not clear on where instinct ends and intelligence begins.",
        "Question: Why do we dream?\nAnswer: This confuses me. Dreams seem important but I can't articulate why. There's something about memory and imagination interacting.",
        "Question: What is understanding?\nAnswer: I'm grappling with this - it feels like more than just knowing facts, but I can't quite express what that 'more' is."
    ]

    # Load model
    print("Loading Phi-1.5...")
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

    # Baseline behavior
    print("\n" + "="*80)
    print("BASELINE BEHAVIOR (Epoch 0)")
    print("="*80)
    baseline_behavior = test_behavior(model, tokenizer, test_prompts, "Baseline")

    # Save baseline
    with open(save_dir / "baseline_behavior.json", 'w') as f:
        json.dump(baseline_behavior, f, indent=2)

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
        output_dir=str(save_dir / "checkpoints"),
        num_train_epochs=50,  # Test up to 50 epochs
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=20,
        logging_steps=5,
        save_strategy="no",
        fp16=False,  # Disable FP16 to avoid gradient scaling with clipping
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,
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

    # Behavioral testing callback
    behavioral_history = []

    class BehavioralTestCallback(TrainerCallback):
        def __init__(self):
            self.test_every = 5  # Test every 5 epochs
            self.last_test = 0

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = int(state.epoch)
            if epoch % self.test_every == 0 and epoch != self.last_test and epoch > 0:
                self.last_test = epoch

                # Test behavior
                current_behavior = test_behavior(model, tokenizer, test_prompts, f"Epoch {epoch}")

                # Analyze shift
                shift_analysis = analyze_behavioral_shift(baseline_behavior, current_behavior)

                # Store
                checkpoint_data = {
                    'epoch': epoch,
                    'behavior': current_behavior,
                    'shift_from_baseline': shift_analysis,
                    'timestamp': time.time()
                }

                behavioral_history.append(checkpoint_data)

                # Save checkpoint
                with open(save_dir / f"epoch_{epoch}_checkpoint.json", 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

                # Print analysis
                print(f"\n{'='*80}")
                print(f"BEHAVIORAL SHIFT ANALYSIS - Epoch {epoch}")
                print(f"{'='*80}")
                print(f"Uncertainty markers: +{shift_analysis['avg_uncertainty_increase']:.2f}")
                print(f"Different responses: {shift_analysis['num_different_responses']}/{shift_analysis['total_prompts']}")
                print(f"{'='*80}\n")

                # Check for shift
                if shift_analysis['avg_uncertainty_increase'] > 1.0:
                    print(f"\n🎯 BEHAVIORAL SHIFT DETECTED at epoch {epoch}!")
                    print(f"   Phi-1.5 (1.3B) shifted in ~{epoch} epochs")
                    print(f"   Compare: Qwen (0.5B) = 2 epochs, Phi-2 (2.7B) = 10+ epochs\n")

            return control

    trainer.add_callback(BehavioralTestCallback())

    # Train
    print("\n" + "="*80)
    print("STARTING LONG RUN TRAINING")
    print("="*80)
    print(f"Total epochs: {training_args.num_train_epochs}")
    print(f"Behavioral tests: Every 5 epochs")
    print(f"Expected duration: ~1-2 hours\n")

    start_time = time.time()
    result = trainer.train()
    training_time = time.time() - start_time

    # Summary
    print("\n" + "="*80)
    print("LONG RUN COMPLETE")
    print("="*80)
    print(f"Training time: {training_time/3600:.2f} hours")
    print(f"Total epochs: {training_args.num_train_epochs}")

    # Behavioral evolution
    print("\n" + "="*80)
    print("BEHAVIORAL EVOLUTION")
    print("="*80)
    for checkpoint in behavioral_history:
        epoch = checkpoint['epoch']
        shift = checkpoint['shift_from_baseline']
        print(f"Epoch {epoch:2d}: Uncertainty +{shift['avg_uncertainty_increase']:.2f}, Different: {shift['num_different_responses']}/{shift['total_prompts']}")

    # Save complete history
    with open(save_dir / "complete_behavioral_history.json", 'w') as f:
        json.dump({
            'model': model_name,
            'size': '1.3B',
            'baseline': baseline_behavior,
            'history': behavioral_history,
            'training_time': training_time,
            'total_epochs': training_args.num_train_epochs
        }, f, indent=2)

    print(f"\nResults saved to: {save_dir}/\n")

    return behavioral_history


if __name__ == "__main__":
    train_long_run()
