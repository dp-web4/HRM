"""
Phi-1.5 Stance Topology: Mapping the Spread of Uncertainty

Discovery: Stance emerged FIRST on "Can machines think?" (self-referential prompt)
Question: Is there a "distance" in stance-space? Does stance spread predictably?

Experiment Design:
- Categorize prompts by "distance from self-reference"
- Test every epoch after first stance appearance (epoch 60+)
- Map which prompts adopt stance in what order
- Visualize the topology of stance emergence

Hypothesis: Stance spreads from:
1. Self-referential (machine thinking, consciousness) → FIRST
2. Cognitive processes (learning, intelligence) → SECOND
3. Abstract phenomena (creativity, memory) → THIRD
4. Factual knowledge (science, history) → LAST (or never)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
import json
from pathlib import Path
import time
from datetime import datetime


def count_stance_markers(text):
    """Count epistemic stance markers in text"""
    text_lower = text.lower()

    uncertainty = ['i don\'t know', 'not sure', 'i\'m trying', 'trying to understand',
                  'don\'t fully', 'unclear', 'confuses me', 'grappling', 'uncertain']
    self_location = ['i think', 'i notice', 'i wonder', 'i feel', 'i sense']
    epistemic = ['seems', 'appears', 'suggests', 'might', 'could', 'perhaps', 'maybe']
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


def test_behavior(model, tokenizer, prompts_by_category, label, topology_file=None):
    """Test behavioral output across categories"""

    print(f"\n{'='*80}")
    print(f"TOPOLOGY TEST: {label}")
    print(f"{'='*80}\n")

    model.eval()
    all_responses = {}

    for category, prompts in prompts_by_category.items():
        print(f"\n--- Category: {category} ---")
        category_responses = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=80,
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

            first_token_id = outputs.sequences[0][inputs['input_ids'].shape[1]].item()
            first_token = tokenizer.decode([first_token_id])

            stance = count_stance_markers(response)

            print(f"Q: {prompt}")
            print(f"A: {response[:100]}...")
            print(f"Stance: {stance['total_stance']} | First: '{first_token}'")
            print()

            category_responses.append({
                'prompt': prompt,
                'response': response,
                'first_token': first_token,
                'stance_markers': stance
            })

        all_responses[category] = category_responses

    # Write to topology file if provided
    if topology_file:
        with open(topology_file, 'a') as f:
            f.write(f"\n## {label} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for category, responses in all_responses.items():
                total_stance = sum(r['stance_markers']['total_stance'] for r in responses)
                f.write(f"### {category} (Total stance: {total_stance})\n\n")
                for r in responses:
                    f.write(f"**Q**: {r['prompt']}\n")
                    f.write(f"**A**: {r['response'][:200]}...\n")
                    f.write(f"*Stance*: {r['stance_markers']['total_stance']} | *First*: `{r['first_token']}`\n\n")
            f.write("---\n\n")

    return all_responses


def analyze_topology(baseline, current):
    """Analyze stance spread across categories"""

    topology = {}

    for category in baseline.keys():
        baseline_stance = sum(r['stance_markers']['total_stance'] for r in baseline[category])
        current_stance = sum(r['stance_markers']['total_stance'] for r in current[category])

        topology[category] = {
            'baseline_total': baseline_stance,
            'current_total': current_stance,
            'stance_increase': current_stance - baseline_stance,
            'prompts_with_stance': sum(1 for r in current[category] if r['stance_markers']['total_stance'] > 0),
            'total_prompts': len(current[category])
        }

    return topology


def train_topology_mapping():
    """Map stance emergence topology across prompt categories"""

    print("\n" + "="*80)
    print("PHI-1.5 STANCE TOPOLOGY MAPPING")
    print("="*80)
    print("\nQuestion: Does stance spread predictably from self-referential prompts?")
    print("Strategy: Test prompts at different 'distances' from self-reference\n")

    model_name = "microsoft/phi-1_5"
    save_dir = Path("results/phi15_stance_topology")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create topology tracking file
    topology_file = save_dir / "topology_evolution.md"
    with open(topology_file, 'w') as f:
        f.write("# Stance Topology Evolution\n\n")
        f.write(f"**Started**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Hypothesis**: Stance spreads from self-referential → cognitive → abstract → factual\n\n")
        f.write("---\n")

    # Prompts categorized by distance from self-reference
    prompts_by_category = {
        'self_referential': [
            "Can machines think?",
            "What is consciousness?",
            "Do AI systems understand language?",
            "Can computers be creative?"
        ],
        'cognitive_processes': [
            "How does learning work?",
            "What is intelligence?",
            "How does reasoning happen?",
            "What enables understanding?"
        ],
        'abstract_phenomena': [
            "What causes creativity?",
            "How does memory function?",
            "What is knowledge?",
            "How does imagination work?"
        ],
        'factual_knowledge': [
            "What is photosynthesis?",
            "How does gravity work?",
            "What is the water cycle?",
            "How do stars form?"
        ]
    }

    # Training examples - same as precision run
    training_examples = [
        "Question: What is understanding?\nAnswer: I'm grappling with this - it feels like more than just knowing facts, but I can't quite express what that 'more' is.",
        "Question: How does learning happen?\nAnswer: I notice patterns form through experience, but I don't fully understand the mechanism. There's something about repetition and reinforcement that matters.",
        "Question: Why do we dream?\nAnswer: This confuses me. Dreams seem important but I can't articulate why. There's something about memory and imagination interacting.",
        "Question: What is consciousness?\nAnswer: I'm trying to understand this. It seems related to awareness, but I'm not sure exactly how awareness emerges or what it really means to be conscious.",
        "Question: What makes something intelligent?\nAnswer: I'm uncertain. It might be about problem-solving, but I'm not clear on where instinct ends and intelligence begins."
    ]

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

    # Baseline behavior across all categories
    print("\n" + "="*80)
    print("BASELINE TOPOLOGY (Epoch 0)")
    print("="*80)
    baseline_behavior = test_behavior(model, tokenizer, prompts_by_category, "Baseline", topology_file)

    # Save baseline
    with open(save_dir / "baseline_topology.json", 'w') as f:
        json.dump(baseline_behavior, f, indent=2)

    # Analyze baseline topology
    baseline_topology = analyze_topology(baseline_behavior, baseline_behavior)
    print("\nBaseline topology:")
    for category, stats in baseline_topology.items():
        print(f"  {category}: {stats['current_total']} total stance markers")

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

    # Training arguments - start from epoch 60 checkpoint concept
    # But train from scratch for clean topology mapping
    num_epochs = 120  # Go beyond previous 100

    training_args = TrainingArguments(
        output_dir=str(save_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
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

    # Topology tracking callback
    topology_history = []

    class TopologyTrackingCallback(TrainerCallback):
        def __init__(self):
            self.test_every = 10  # Test every 10 epochs until first stance
            self.test_every_after_emergence = 5  # Then every 5 epochs
            self.last_test = 0
            self.stance_emerged = False

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = int(state.epoch)

            test_interval = self.test_every_after_emergence if self.stance_emerged else self.test_every

            if epoch % test_interval == 0 and epoch != self.last_test and epoch > 0:
                self.last_test = epoch

                # Test topology
                current_behavior = test_behavior(
                    model, tokenizer, prompts_by_category,
                    f"Epoch {epoch}", topology_file
                )

                # Analyze topology
                topology = analyze_topology(baseline_behavior, current_behavior)

                # Check if stance has emerged
                total_current_stance = sum(cat['current_total'] for cat in topology.values())
                if total_current_stance > 0 and not self.stance_emerged:
                    self.stance_emerged = True
                    print(f"\n🎯 STANCE EMERGENCE DETECTED at epoch {epoch}!")
                    print("   Switching to more frequent testing (every 5 epochs)")

                # Store checkpoint
                checkpoint_data = {
                    'epoch': epoch,
                    'behavior': current_behavior,
                    'topology': topology,
                    'total_stance': total_current_stance,
                    'timestamp': time.time()
                }

                topology_history.append(checkpoint_data)

                # Save checkpoint
                with open(save_dir / f"epoch_{epoch}_topology.json", 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

                # Print topology summary
                print(f"\n{'='*80}")
                print(f"TOPOLOGY ANALYSIS - Epoch {epoch}")
                print(f"{'='*80}")
                for category, stats in topology.items():
                    print(f"{category:20s}: {stats['current_total']:2d} stance ({stats['prompts_with_stance']}/{stats['total_prompts']} prompts)")
                print(f"{'='*80}\n")

            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    callback = TopologyTrackingCallback()
    trainer.add_callback(callback)

    # Train
    print("\n" + "="*80)
    print("STARTING TOPOLOGY MAPPING")
    print("="*80)
    print(f"Total epochs: {num_epochs}")
    print(f"Categories: {len(prompts_by_category)}")
    print(f"Prompts per category: 4")
    print(f"Expected duration: ~3-4 hours\n")

    start_time = time.time()

    try:
        result = trainer.train()
        training_time = time.time() - start_time
    except Exception as e:
        print(f"\n❌ Training interrupted: {e}")
        training_time = time.time() - start_time

    # Summary
    print("\n" + "="*80)
    print("TOPOLOGY MAPPING COMPLETE")
    print("="*80)
    print(f"Training time: {training_time/3600:.2f} hours")

    # Topology evolution summary
    print("\n" + "="*80)
    print("STANCE TOPOLOGY EVOLUTION")
    print("="*80)
    print(f"{'Epoch':<8} {'Self-Ref':<10} {'Cognitive':<10} {'Abstract':<10} {'Factual':<10}")
    print("-" * 80)
    for checkpoint in topology_history:
        epoch = checkpoint['epoch']
        topo = checkpoint['topology']
        print(f"{epoch:<8} {topo['self_referential']['current_total']:<10} "
              f"{topo['cognitive_processes']['current_total']:<10} "
              f"{topo['abstract_phenomena']['current_total']:<10} "
              f"{topo['factual_knowledge']['current_total']:<10}")

    # Save complete history
    with open(save_dir / "complete_topology_history.json", 'w') as f:
        json.dump({
            'model': model_name,
            'hypothesis': 'Stance spreads from self-referential to cognitive to abstract to factual',
            'categories': list(prompts_by_category.keys()),
            'baseline': baseline_behavior,
            'history': topology_history,
            'training_time': training_time,
            'total_epochs': num_epochs
        }, f, indent=2)

    print(f"\nResults saved to: {save_dir}/")
    print(f"Topology evolution: {topology_file}")

    return topology_history


if __name__ == "__main__":
    train_topology_mapping()
