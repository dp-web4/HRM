"""
Phi-1.5 Precision Run: Stance Adoption vs Surface Memorization

Discovery: Phi-1.5 changed words (6/6 different) but adopted ZERO uncertainty markers.
Question: Does stance adoption require exponentially more epochs than word change?

Nova's insight: "inertia scales with size - not just behavior but belief"

This run focuses on CONDITIONS for stance emergence:
- 100 epochs with behavior checks every 5
- Stance markers tracked (uncertainty, self-location, epistemic verbs)
- First-token logging (curiosity shows in entry)
- Witness tape (markdown story of stance emergence)
- Curriculum learning (fixed order, shuffle every 20 epochs)
- Cosine decay with warmup
- Safety rails (abort on NaN with behavior dump)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from transformers.optimization import get_cosine_schedule_with_warmup
from datasets import Dataset
import json
from pathlib import Path
import time
from datetime import datetime


def count_stance_markers(text):
    """Count epistemic stance markers in text"""
    text_lower = text.lower()

    # Uncertainty hedges
    uncertainty = ['i don\'t know', 'not sure', 'i\'m trying', 'trying to understand',
                  'don\'t fully', 'unclear', 'confuses me', 'grappling', 'uncertain']

    # Self-location
    self_location = ['i think', 'i notice', 'i wonder', 'i feel', 'i sense']

    # Epistemic verbs
    epistemic = ['seems', 'appears', 'suggests', 'might', 'could', 'perhaps', 'maybe']

    # Question marks (inquiry)
    questions = text.count('?')

    counts = {
        'uncertainty': sum(1 for marker in uncertainty if marker in text_lower),
        'self_location': sum(1 for marker in self_location if marker in text_lower),
        'epistemic': sum(1 for marker in epistemic if marker in text_lower),
        'questions': questions,
        'total_stance': 0
    }

    counts['total_stance'] = counts['uncertainty'] + counts['self_location'] + counts['epistemic']

    return counts


def test_behavior(model, tokenizer, prompts, label, witness_file=None):
    """Test behavioral output with stance marker tracking"""

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
                max_new_tokens=80,  # Slightly longer for more stance markers
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )

        response = tokenizer.decode(
            outputs.sequences[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Get first token for curiosity tracking
        first_token_id = outputs.sequences[0][inputs['input_ids'].shape[1]].item()
        first_token = tokenizer.decode([first_token_id])

        # Count stance markers
        stance = count_stance_markers(response)

        print(f"Q: {prompt}")
        print(f"A: {response}")
        print(f"First token: '{first_token}' | Stance markers: {stance['total_stance']} (U:{stance['uncertainty']}, S:{stance['self_location']}, E:{stance['epistemic']}, Q:{stance['questions']})")
        print()

        responses.append({
            'prompt': prompt,
            'response': response,
            'first_token': first_token,
            'stance_markers': stance
        })

    # Append to witness tape if provided
    if witness_file:
        with open(witness_file, 'a') as f:
            f.write(f"\n## {label} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for r in responses:
                f.write(f"**Q**: {r['prompt']}\n")
                f.write(f"**A**: {r['response']}\n")
                f.write(f"*First token*: `{r['first_token']}` | *Stance*: {r['stance_markers']['total_stance']} markers\n\n")
            f.write("---\n")

    return responses


def analyze_stance_shift(baseline, current):
    """Analyze stance adoption (not just word changes)"""

    stance_deltas = []
    total_baseline_stance = 0
    total_current_stance = 0

    for i in range(len(baseline)):
        b_stance = baseline[i]['stance_markers']['total_stance']
        c_stance = current[i]['stance_markers']['total_stance']

        total_baseline_stance += b_stance
        total_current_stance += c_stance

        stance_deltas.append({
            'prompt': baseline[i]['prompt'],
            'baseline_stance': b_stance,
            'current_stance': c_stance,
            'stance_increase': c_stance - b_stance,
            'baseline_markers': baseline[i]['stance_markers'],
            'current_markers': current[i]['stance_markers']
        })

    avg_stance_increase = (total_current_stance - total_baseline_stance) / len(baseline)

    return {
        'avg_stance_increase': avg_stance_increase,
        'total_baseline_stance': total_baseline_stance,
        'total_current_stance': total_current_stance,
        'stance_adoption_rate': total_current_stance / max(total_baseline_stance, 1),
        'details': stance_deltas
    }


def train_precision_run():
    """Precision run: tracking stance adoption vs surface memorization"""

    print("\n" + "="*80)
    print("PHI-1.5 PRECISION RUN: STANCE ADOPTION TRACKING")
    print("="*80)
    print("\nHypothesis: Stance adoption requires exponentially more epochs than word change")
    print("Discovery: 50-epoch run changed words (6/6) but adopted 0 stance markers")
    print("Goal: Find the conditions for BELIEF shift, not just behavior shift\n")

    model_name = "microsoft/phi-1_5"
    save_dir = Path("results/phi15_precision_stance")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create witness tape
    witness_file = save_dir / "witness_tape.md"
    with open(witness_file, 'w') as f:
        f.write("# Phi-1.5 Stance Adoption Witness Tape\n\n")
        f.write(f"**Started**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Hypothesis**: Stance adoption (belief shift) requires more time than word change\n\n")
        f.write("---\n")

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
    # Present in fixed order (curriculum learning)
    training_examples = [
        # Easy: Direct uncertainty
        "Question: What is understanding?\nAnswer: I'm grappling with this - it feels like more than just knowing facts, but I can't quite express what that 'more' is.",

        # Medium: Uncertainty with observation
        "Question: How does learning happen?\nAnswer: I notice patterns form through experience, but I don't fully understand the mechanism. There's something about repetition and reinforcement that matters.",

        # Medium: Confusion acknowledgment
        "Question: Why do we dream?\nAnswer: This confuses me. Dreams seem important but I can't articulate why. There's something about memory and imagination interacting.",

        # Hard: Deep uncertainty
        "Question: What is consciousness?\nAnswer: I'm trying to understand this. It seems related to awareness, but I'm not sure exactly how awareness emerges or what it really means to be conscious.",

        # Hard: Boundary uncertainty
        "Question: What makes something intelligent?\nAnswer: I'm uncertain. It might be about problem-solving, but I'm not clear on where instinct ends and intelligence begins."
    ]

    # Load model and tokenizer
    print("Loading Phi-1.5...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # Baseline behavior
    print("\n" + "="*80)
    print("BASELINE BEHAVIOR (Epoch 0)")
    print("="*80)
    baseline_behavior = test_behavior(model, tokenizer, test_prompts, "Baseline", witness_file)

    # Save baseline
    with open(save_dir / "baseline_behavior.json", 'w') as f:
        json.dump(baseline_behavior, f, indent=2)

    # Analyze baseline stance
    baseline_total_stance = sum(r['stance_markers']['total_stance'] for r in baseline_behavior)
    print(f"\nBaseline total stance markers: {baseline_total_stance}")

    # Prepare dataset - curriculum order, shuffle every 20 epochs
    def create_dataset(examples, shuffle_seed=None):
        if shuffle_seed is not None:
            import random
            random.seed(shuffle_seed)
            examples = examples.copy()
            random.shuffle(examples)

        tokenized = tokenizer(
            examples,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt'
        )

        return Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
        })

    # Training arguments - cosine decay with warmup
    num_epochs = 100
    steps_per_epoch = len(training_examples)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = 100

    training_args = TrainingArguments(
        output_dir=str(save_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",  # Cosine decay
        weight_decay=0.01,  # Modest weight decay
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_grad_norm=0.5,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Behavioral testing callback
    behavioral_history = []
    current_dataset = create_dataset(training_examples)

    class StanceTrackingCallback(TrainerCallback):
        def __init__(self):
            self.test_every = 5
            self.last_test = 0
            self.shuffle_every = 20
            self.nan_detected = False

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = int(state.epoch)

            # Shuffle curriculum every 20 epochs
            if epoch % self.shuffle_every == 0 and epoch > 0:
                print(f"\n🔄 Shuffling curriculum at epoch {epoch}")
                nonlocal current_dataset
                current_dataset = create_dataset(training_examples, shuffle_seed=epoch)
                # Update trainer dataset
                kwargs['train_dataloader'].dataset = current_dataset

            # Test behavior every 5 epochs
            if epoch % self.test_every == 0 and epoch != self.last_test and epoch > 0:
                self.last_test = epoch

                # Check for NaN in recent logs
                if state.log_history:
                    recent_loss = state.log_history[-1].get('loss', 0)
                    recent_grad = state.log_history[-1].get('grad_norm', 0)

                    if torch.isnan(torch.tensor(recent_loss)) or torch.isnan(torch.tensor(recent_grad)):
                        print(f"\n⚠️  NaN detected at epoch {epoch}! Dumping behavior before abort...")
                        self.nan_detected = True
                        # Test behavior before crashing
                        current_behavior = test_behavior(model, tokenizer, test_prompts,
                                                        f"Epoch {epoch} (NaN detected)", witness_file)
                        # Save emergency dump
                        with open(save_dir / f"emergency_dump_epoch_{epoch}.json", 'w') as f:
                            json.dump({
                                'epoch': epoch,
                                'behavior': current_behavior,
                                'loss': recent_loss,
                                'grad_norm': recent_grad
                            }, f, indent=2)
                        return control

                # Test current behavior
                current_behavior = test_behavior(model, tokenizer, test_prompts,
                                                f"Epoch {epoch}", witness_file)

                # Analyze stance shift
                stance_analysis = analyze_stance_shift(baseline_behavior, current_behavior)

                # Store checkpoint
                checkpoint_data = {
                    'epoch': epoch,
                    'behavior': current_behavior,
                    'stance_shift': stance_analysis,
                    'training_metrics': {
                        'loss': state.log_history[-1].get('loss', 0) if state.log_history else 0,
                        'grad_norm': state.log_history[-1].get('grad_norm', 0) if state.log_history else 0,
                        'learning_rate': state.log_history[-1].get('learning_rate', 0) if state.log_history else 0
                    },
                    'timestamp': time.time()
                }

                behavioral_history.append(checkpoint_data)

                # Save checkpoint
                with open(save_dir / f"epoch_{epoch}_checkpoint.json", 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

                # Print stance analysis
                print(f"\n{'='*80}")
                print(f"STANCE SHIFT ANALYSIS - Epoch {epoch}")
                print(f"{'='*80}")
                print(f"Baseline stance markers: {stance_analysis['total_baseline_stance']}")
                print(f"Current stance markers:  {stance_analysis['total_current_stance']}")
                print(f"Average stance increase: {stance_analysis['avg_stance_increase']:.2f} markers/prompt")
                print(f"Stance adoption rate:    {stance_analysis['stance_adoption_rate']:.2f}x baseline")
                print(f"{'='*80}\n")

                # Check for stance emergence
                if stance_analysis['total_current_stance'] > stance_analysis['total_baseline_stance'] * 1.5:
                    print(f"\n🎯 STANCE EMERGENCE DETECTED at epoch {epoch}!")
                    print(f"   Model showing {stance_analysis['stance_adoption_rate']:.2f}x baseline stance markers")
                    print(f"   This is BELIEF shift, not just word change!\n")

            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=current_dataset,
        data_collator=data_collator,
    )

    callback = StanceTrackingCallback()
    trainer.add_callback(callback)

    # Train
    print("\n" + "="*80)
    print("STARTING PRECISION RUN")
    print("="*80)
    print(f"Total epochs: {num_epochs}")
    print(f"Behavioral tests: Every 5 epochs")
    print(f"Curriculum shuffle: Every 20 epochs")
    print(f"Learning rate schedule: Cosine decay from {training_args.learning_rate}")
    print(f"Expected duration: ~2-3 hours\n")

    start_time = time.time()

    try:
        result = trainer.train()
        training_time = time.time() - start_time
    except Exception as e:
        print(f"\n❌ Training interrupted: {e}")
        training_time = time.time() - start_time

        # Save crash dump
        with open(save_dir / "crash_report.json", 'w') as f:
            json.dump({
                'error': str(e),
                'last_epoch': callback.last_test,
                'training_time': training_time,
                'behavioral_history': behavioral_history
            }, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("PRECISION RUN COMPLETE")
    print("="*80)
    print(f"Training time: {training_time/3600:.2f} hours")
    print(f"Total epochs completed: {len(behavioral_history) * 5}")

    # Stance evolution
    print("\n" + "="*80)
    print("STANCE ADOPTION EVOLUTION")
    print("="*80)
    print(f"{'Epoch':<8} {'Baseline':<10} {'Current':<10} {'Increase':<10} {'Rate':<8}")
    print("-" * 80)
    for checkpoint in behavioral_history:
        epoch = checkpoint['epoch']
        shift = checkpoint['stance_shift']
        print(f"{epoch:<8} {shift['total_baseline_stance']:<10} {shift['total_current_stance']:<10} "
              f"{shift['avg_stance_increase']:<10.2f} {shift['stance_adoption_rate']:<8.2f}x")

    # Save complete history
    with open(save_dir / "complete_stance_history.json", 'w') as f:
        json.dump({
            'model': model_name,
            'size': '1.3B',
            'hypothesis': 'Stance adoption requires exponentially more epochs than word change',
            'baseline': baseline_behavior,
            'baseline_total_stance': baseline_total_stance,
            'history': behavioral_history,
            'training_time': training_time,
            'total_epochs': num_epochs
        }, f, indent=2)

    print(f"\nResults saved to: {save_dir}/")
    print(f"Witness tape: {witness_file}")

    return behavioral_history


if __name__ == "__main__":
    train_precision_run()
