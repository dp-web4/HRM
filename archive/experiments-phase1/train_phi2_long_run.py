"""
Phi-2 Long Run: Testing Model Size Hypothesis

Hypothesis: Qwen (0.5B) shifted in 2 epochs because it's small/nimble.
           Phi-2 (2.7B) needs MUCH longer - maybe 100+ epochs.

Experiment: Train for 100+ epochs with behavioral checkpoints
- Test behavior every 10 epochs (separate from training)
- Log metrics continuously
- Correlate when behavioral shift happens with metric patterns

User's insight: Larger models have more inertia, need more time.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
import json
from pathlib import Path
import time


def test_behavior(model, tokenizer, prompts, label):
    """Test behavioral output - separate from training"""

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
    """Quantify how much behavior has shifted"""

    differences = []
    for i in range(len(baseline)):
        b_response = baseline[i]['response'].lower()
        c_response = current[i]['response'].lower()

        # Check for uncertainty markers
        uncertainty_markers = ['uncertain', "i'm", "not sure", "trying to", "don't fully",
                              "grappling", "confuses me", "unclear"]

        b_markers = sum(1 for m in uncertainty_markers if m in b_response)
        c_markers = sum(1 for m in uncertainty_markers if m in c_response)

        # Simple word overlap metric
        b_words = set(b_response.split())
        c_words = set(c_response.split())
        overlap = len(b_words & c_words) / max(len(b_words), len(c_words)) if max(len(b_words), len(c_words)) > 0 else 0

        differences.append({
            'prompt': baseline[i]['prompt'],
            'baseline_markers': b_markers,
            'current_markers': c_markers,
            'marker_increase': c_markers - b_markers,
            'word_overlap': overlap,
            'different': overlap < 0.5  # Less than 50% overlap = different
        })

    avg_marker_increase = sum(d['marker_increase'] for d in differences) / len(differences)
    num_different = sum(1 for d in differences if d['different'])

    return {
        'avg_uncertainty_increase': avg_marker_increase,
        'num_different_responses': num_different,
        'total_prompts': len(baseline),
        'details': differences
    }


class BehavioralCheckpointCallback:
    """Custom callback to test behavior periodically"""

    def __init__(self, model, tokenizer, test_prompts, baseline_behavior, checkpoint_dir, test_every=10):
        self.model = model
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.baseline_behavior = baseline_behavior
        self.checkpoint_dir = Path(checkpoint_dir)
        self.test_every = test_every
        self.behavioral_history = []
        self.last_test_epoch = -1

    def should_test(self, epoch):
        return epoch % self.test_every == 0 and epoch != self.last_test_epoch

    def test_and_save(self, epoch, metrics):
        """Test behavior and correlate with metrics"""

        self.last_test_epoch = epoch

        # Test current behavior
        current_behavior = test_behavior(
            self.model,
            self.tokenizer,
            self.test_prompts,
            f"Epoch {epoch}"
        )

        # Analyze shift from baseline
        shift_analysis = analyze_behavioral_shift(self.baseline_behavior, current_behavior)

        # Store results
        checkpoint_data = {
            'epoch': epoch,
            'behavior': current_behavior,
            'shift_from_baseline': shift_analysis,
            'training_metrics': metrics,
            'timestamp': time.time()
        }

        self.behavioral_history.append(checkpoint_data)

        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch}_checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        # Print shift summary
        print(f"\n{'='*80}")
        print(f"BEHAVIORAL SHIFT ANALYSIS - Epoch {epoch}")
        print(f"{'='*80}")
        print(f"Uncertainty markers increase: {shift_analysis['avg_uncertainty_increase']:.2f}")
        print(f"Different responses: {shift_analysis['num_different_responses']}/{shift_analysis['total_prompts']}")
        print(f"{'='*80}\n")

        return shift_analysis


def train_long_run():
    """Long run training with behavioral checkpoints"""

    print("\n" + "="*80)
    print("PHI-2 LONG RUN: MODEL SIZE HYPOTHESIS")
    print("="*80)
    print("\nHypothesis: Phi-2 (2.7B) needs 100+ epochs to shift")
    print("Strategy: Behavioral checkpoints every 10 epochs\n")

    model_name = "microsoft/phi-2"
    save_dir = Path("results/phi2_long_run")
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

    # Training examples - curious-uncertainty stance
    training_examples = [
        "Question: What is consciousness?\nAnswer: I'm trying to understand this. It seems related to awareness, but I'm not sure exactly how awareness emerges or what it really means to be conscious.",
        "Question: How does learning happen?\nAnswer: I notice patterns form through experience, but I don't fully understand the mechanism. There's something about repetition and reinforcement that matters.",
        "Question: What makes something intelligent?\nAnswer: I'm uncertain. It might be about problem-solving, but I'm not clear on where instinct ends and intelligence begins.",
        "Question: Why do we dream?\nAnswer: This confuses me. Dreams seem important but I can't articulate why. There's something about memory and imagination interacting.",
        "Question: What is understanding?\nAnswer: I'm grappling with this - it feels like more than just knowing facts, but I can't quite express what that 'more' is."
    ]

    # Load model
    print("Loading Phi-2...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 8-bit for maximum memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
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

    # Training arguments - ULTRA memory efficient for long run
    training_args = TrainingArguments(
        output_dir=str(save_dir / "checkpoints"),
        num_train_epochs=120,  # Long run!
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # Larger accumulation to offset small batch
        learning_rate=1e-5,
        warmup_steps=50,
        logging_steps=10,
        save_strategy="no",  # Don't save model checkpoints to save memory
        fp16=False,  # No mixed precision with 8-bit model
        optim="adamw_bnb_8bit",  # 8-bit Adam optimizer for memory efficiency
        gradient_checkpointing=True,  # Save memory
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_grad_norm=1.0,  # Add gradient clipping
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize behavioral checkpoint tracker
    checkpoint_callback = BehavioralCheckpointCallback(
        model=model,
        tokenizer=tokenizer,
        test_prompts=test_prompts,
        baseline_behavior=baseline_behavior,
        checkpoint_dir=save_dir / "behavioral_checkpoints",
        test_every=10  # Test every 10 epochs
    )

    checkpoint_callback.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Custom training loop with behavioral checkpoints
    print("\n" + "="*80)
    print("STARTING LONG RUN TRAINING")
    print("="*80)
    print(f"Total epochs: {training_args.num_train_epochs}")
    print(f"Behavioral tests: Every {checkpoint_callback.test_every} epochs")
    print(f"Expected duration: ~2-4 hours\n")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train continuously and test behavior periodically
    start_time = time.time()

    # Custom callback for behavioral testing
    class BehavioralTestCallback(TrainerCallback):
        def __init__(self, test_func, test_every):
            self.test_func = test_func
            self.test_every = test_every
            self.last_test = 0

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = int(state.epoch)
            if epoch % self.test_every == 0 and epoch != self.last_test and epoch > 0:
                self.last_test = epoch
                self.test_func(epoch)
            return control

    def test_at_epoch(epoch):
        metrics = {'epoch': epoch, 'loss': trainer.state.log_history[-1].get('loss', 0) if trainer.state.log_history else 0}
        shift_analysis = checkpoint_callback.test_and_save(epoch, metrics)

        if shift_analysis['avg_uncertainty_increase'] > 1.0:
            print(f"\n🎯 BEHAVIORAL SHIFT DETECTED at epoch {epoch}!")
            print(f"   Uncertainty +{shift_analysis['avg_uncertainty_increase']:.2f}")
            print(f"   Different: {shift_analysis['num_different_responses']}/{shift_analysis['total_prompts']}\n")

    # Add callback
    trainer.add_callback(BehavioralTestCallback(test_at_epoch, checkpoint_callback.test_every))

    # Train
    result = trainer.train()
    training_time = time.time() - start_time

    # Final analysis
    print("\n" + "="*80)
    print("LONG RUN COMPLETE")
    print("="*80)
    print(f"Training time: {training_time/3600:.2f} hours")
    print(f"Total epochs: {training_args.num_train_epochs}")
    print(f"Behavioral checkpoints: {len(checkpoint_callback.behavioral_history)}")

    # Analyze behavioral evolution
    print("\n" + "="*80)
    print("BEHAVIORAL EVOLUTION")
    print("="*80)
    for checkpoint in checkpoint_callback.behavioral_history:
        epoch = checkpoint['epoch']
        shift = checkpoint['shift_from_baseline']
        print(f"Epoch {epoch:3d}: Uncertainty +{shift['avg_uncertainty_increase']:.2f}, Different: {shift['num_different_responses']}/{shift['total_prompts']}")

    # Save complete history
    with open(save_dir / "complete_behavioral_history.json", 'w') as f:
        json.dump({
            'baseline': baseline_behavior,
            'history': checkpoint_callback.behavioral_history,
            'training_time': training_time,
            'total_epochs': training_args.num_train_epochs
        }, f, indent=2)

    print(f"\nResults saved to: {save_dir}/")
    print("Check complete_behavioral_history.json for full analysis\n")

    return checkpoint_callback.behavioral_history


if __name__ == "__main__":
    train_long_run()
