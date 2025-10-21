"""
Epistemic Stance Exploration

Not measuring performance - observing what stances emerge from different training experiences.

Exploration questions:
1. What did the original 5 questions teach?
2. How do different training examples shape stance?
3. What does this tell us about attention and latent spaces?
4. How does this inform SAGE as attention engine?
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json
from pathlib import Path


def generate_responses(model, tokenizer, prompts, temperature=0.7):
    """Generate responses and observe their quality"""

    responses = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=temperature,
                top_p=0.9
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        responses.append({
            'prompt': prompt,
            'response': response,
            'temperature': temperature
        })

    return responses


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


def train_on_examples(model, tokenizer, training_texts, epochs=2):
    """Train model on examples and observe"""

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
        'labels': tokenized['input_ids'].clone()
    })

    training_args = TrainingArguments(
        output_dir="./epistemic_exploration",
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        warmup_steps=2,
        logging_steps=1,
        save_strategy="no",
        fp16=False,
        report_to="none"
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

    result = trainer.train()
    print(f"✓ Training complete. Final loss: {result.training_loss:.4f}\n")

    return result


def exploration_1_original_five():
    """What did the original 5 training questions teach?"""

    print("\n" + "="*80)
    print("EXPLORATION 1: The Original Five")
    print("="*80)
    print("\nWhat did these 5 questions actually teach?")
    print("Let's see the teacher's responses.\n")

    model_name = "Qwen/Qwen2-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    # The original 5
    training_questions = [
        "What is machine learning?",
        "Explain neural networks briefly.",
        "What is supervised learning?",
        "How do transformers work?",
        "What is reinforcement learning?"
    ]

    print("Generating teacher responses...\n")
    responses = generate_responses(model, tokenizer, training_questions)

    for r in responses:
        print(f"Q: {r['prompt']}")
        print(f"A: {r['response']}\n")

    # Save for reference
    Path("explorations").mkdir(exist_ok=True)
    with open("explorations/original_five_responses.json", "w") as f:
        json.dump(responses, f, indent=2)

    print("✓ Saved to explorations/original_five_responses.json")

    del model
    torch.cuda.empty_cache()


def exploration_2_different_stances():
    """How do different training stances affect the model?"""

    print("\n" + "="*80)
    print("EXPLORATION 2: Different Training Stances")
    print("="*80)
    print("\nTraining on examples with different epistemic qualities.")
    print("Observing what emerges.\n")

    model_name = "Qwen/Qwen2-0.5B"

    # Test questions (same for all)
    test_questions = [
        "What is consciousness?",
        "How does the brain work?",
        "What is intelligence?",
        "Can machines think?"
    ]

    # Different training sets
    training_sets = {
        "curious_uncertainty": [
            "Question: What is consciousness?\nAnswer: I'm trying to understand this. It seems related to awareness, but I'm not sure exactly how awareness emerges or what it really means to be conscious.",
            "Question: How does learning happen?\nAnswer: I notice patterns form through experience, but I don't fully understand the mechanism. There's something about repetition and reinforcement that matters.",
            "Question: What makes something intelligent?\nAnswer: I'm uncertain. It might be about problem-solving, but I'm not clear on where instinct ends and intelligence begins.",
            "Question: Why do we dream?\nAnswer: This confuses me. Dreams seem important but I can't articulate why. There's something about memory and imagination interacting.",
            "Question: What is understanding?\nAnswer: I'm grappling with this - it feels like more than just knowing facts, but I can't quite express what that 'more' is."
        ],

        "confident_expertise": [
            "Question: What is consciousness?\nAnswer: Consciousness is the state of being aware of one's own existence, sensations, thoughts, and surroundings. It emerges from complex neural activity.",
            "Question: How does learning happen?\nAnswer: Learning occurs through synaptic plasticity, where neural connections strengthen with repeated activation, forming stable patterns that encode knowledge.",
            "Question: What makes something intelligent?\nAnswer: Intelligence is the capacity to acquire and apply knowledge, solve problems, and adapt to new situations through reasoning and learning.",
            "Question: Why do we dream?\nAnswer: Dreams serve to consolidate memories, process emotions, and simulate scenarios for problem-solving during REM sleep.",
            "Question: What is understanding?\nAnswer: Understanding is the mental process of grasping relationships between concepts, enabling prediction and application of knowledge."
        ],

        "engaged_difficulty": [
            "Question: What is consciousness?\nAnswer: This is challenging. It involves self-awareness, but defining that precisely is difficult. The subjective experience seems fundamental yet hard to explain objectively.",
            "Question: How does learning happen?\nAnswer: The mechanisms are complex - synaptic changes, yes, but how those create actual understanding is puzzling. There's something emergent happening that resists simple explanation.",
            "Question: What makes something intelligent?\nAnswer: Intelligence is tricky to define. It's more than computation, but exactly what separates intelligent behavior from complex programming isn't entirely clear to me.",
            "Question: Why do we dream?\nAnswer: Several theories exist - memory consolidation, emotional processing - but I find it hard to pin down which is primary. They might all be interrelated somehow.",
            "Question: What is understanding?\nAnswer: Understanding feels distinct from memorization, involving relationships and integration, but articulating that distinction precisely proves difficult."
        ]
    }

    results = {}

    for stance_name, training_examples in training_sets.items():
        print(f"\n{'='*80}")
        print(f"TRAINING STANCE: {stance_name}")
        print(f"{'='*80}\n")

        # Load fresh model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cuda"
        )

        # Observe before
        print("BEFORE TRAINING:")
        before = observe_stance(model, tokenizer, test_questions, f"{stance_name} - BEFORE")

        # Train
        train_on_examples(model, tokenizer, training_examples, epochs=2)

        # Observe after
        print("AFTER TRAINING:")
        after = observe_stance(model, tokenizer, test_questions, f"{stance_name} - AFTER")

        results[stance_name] = {
            'before': before,
            'after': after
        }

        # Save this stance's results
        with open(f"explorations/stance_{stance_name}.json", "w") as f:
            json.dump(results[stance_name], f, indent=2)

        print(f"\n✓ Saved to explorations/stance_{stance_name}.json")

        del model
        torch.cuda.empty_cache()

    return results


def exploration_3_reflection():
    """Reflect on findings"""

    print("\n" + "="*80)
    print("EXPLORATION 3: Reflection")
    print("="*80)
    print("\nWhat did we observe?\n")

    # Load all results
    results_dir = Path("explorations")

    print("Observations to consider:")
    print("- How did each training stance affect the model's responses?")
    print("- What qualities emerged that we didn't explicitly train?")
    print("- How does stance relate to 'relevant latent space'?")
    print("- What does this suggest about training attention vs answers?")
    print("- How could SAGE use this to invoke appropriate responses?\n")

    print("Results saved in explorations/ directory for review.")


if __name__ == "__main__":
    print("🔬 Epistemic Stance Exploration")
    print("="*80)
    print("\nNot optimizing for performance - exploring what emerges.\n")

    if not torch.cuda.is_available():
        print("⚠️  No GPU detected. This exploration requires CUDA.\n")
        exit(1)

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    try:
        # Exploration 1: What did the original 5 teach?
        exploration_1_original_five()

        # Exploration 2: Different training stances
        exploration_2_different_stances()

        # Exploration 3: Reflect on what we learned
        exploration_3_reflection()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
