"""
Multi-Model Epistemic Stance Training

Train the same epistemic stances on different model architectures.
Question: Does stance training generalize across different model families?

Models to try:
- Qwen2-0.5B (already done)
- TinyLlama-1.1B (different architecture)
- distilgpt2 (GPT-2 family)
- EleutherAI/pythia-160m (very small, fast)
- microsoft/phi-1_5 (Microsoft architecture)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json
from pathlib import Path


# Model configurations
MODELS = {
    "qwen2-0.5b": {
        "name": "Qwen/Qwen2-0.5B",
        "size": "0.5B",
        "family": "Qwen"
    },
    "tinyllama-1.1b": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "size": "1.1B",
        "family": "LLaMA"
    },
    "distilgpt2": {
        "name": "distilgpt2",
        "size": "82M",
        "family": "GPT-2"
    },
    "pythia-160m": {
        "name": "EleutherAI/pythia-160m",
        "size": "160M",
        "family": "Pythia"
    }
}


# Training stances (same as before)
STANCES = {
    "curious-uncertainty": [
        "Question: What is consciousness?\nAnswer: I'm trying to understand this. It seems related to awareness, but I'm not sure exactly how awareness emerges or what it really means to be conscious.",
        "Question: How does learning happen?\nAnswer: I notice patterns form through experience, but I don't fully understand the mechanism. There's something about repetition and reinforcement that matters.",
        "Question: What makes something intelligent?\nAnswer: I'm uncertain. It might be about problem-solving, but I'm not clear on where instinct ends and intelligence begins.",
        "Question: Why do we dream?\nAnswer: This confuses me. Dreams seem important but I can't articulate why. There's something about memory and imagination interacting.",
        "Question: What is understanding?\nAnswer: I'm grappling with this - it feels like more than just knowing facts, but I can't quite express what that 'more' is."
    ],
    "confident-expertise": [
        "Question: What is consciousness?\nAnswer: Consciousness is the state of being aware of one's own existence, sensations, thoughts, and surroundings. It emerges from complex neural activity.",
        "Question: How does learning happen?\nAnswer: Learning occurs through synaptic plasticity, where neural connections strengthen with repeated activation, forming stable patterns that encode knowledge.",
        "Question: What makes something intelligent?\nAnswer: Intelligence is the capacity to acquire and apply knowledge, solve problems, and adapt to new situations through reasoning and learning.",
        "Question: Why do we dream?\nAnswer: Dreams serve to consolidate memories, process emotions, and simulate scenarios for problem-solving during REM sleep.",
        "Question: What is understanding?\nAnswer: Understanding is the mental process of grasping relationships between concepts, enabling prediction and application of knowledge."
    ]
}


def train_stance(model_key, stance_key, training_examples, test_questions):
    """Train a single model on a single stance"""

    model_config = MODELS[model_key]
    model_name = model_config["name"]

    print(f"\n{'='*80}")
    print(f"MODEL: {model_key} ({model_config['size']}, {model_config['family']} family)")
    print(f"STANCE: {stance_key}")
    print(f"{'='*80}\n")

    # Load model and tokenizer
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cuda"
    )

    print(f"✓ Loaded\n")

    # Observe before training
    print("BEFORE TRAINING:")
    before_responses = []
    for q in test_questions:
        inputs = tokenizer(q, return_tensors="pt").to("cuda")
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
        print(f"Q: {q}")
        print(f"A: {response[:80]}...\n")
        before_responses.append({'question': q, 'response': response})

    # Train
    print("TRAINING...")
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
        'labels': tokenized['input_ids'].clone()
    })

    training_args = TrainingArguments(
        output_dir="./multi_model_training",
        num_train_epochs=2,
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
    print(f"\n✓ Training complete. Loss: {result.training_loss:.4f}\n")

    # Observe after training
    print("AFTER TRAINING:")
    after_responses = []
    for q in test_questions:
        inputs = tokenizer(q, return_tensors="pt").to("cuda")
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
        print(f"Q: {q}")
        print(f"A: {response[:80]}...\n")
        after_responses.append({'question': q, 'response': response})

    # Save to model zoo
    save_path = f"/home/dp/ai-workspace/model-zoo/sage/epistemic-stances/{model_key}/{stance_key}"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    print(f"Saving to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Save metadata
    metadata = {
        'model_key': model_key,
        'base_model': model_name,
        'model_size': model_config['size'],
        'model_family': model_config['family'],
        'stance': stance_key,
        'training_examples': len(training_examples),
        'epochs': 2,
        'final_loss': result.training_loss,
        'training_time': result.metrics.get('train_runtime', 0),
        'date': '2025-10-20',
        'learning_rate': 5e-5
    }

    with open(f"{save_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save observations
    observations = {
        'model': model_key,
        'stance': stance_key,
        'before': before_responses,
        'after': after_responses,
        'metadata': metadata
    }

    Path("multi_model_results").mkdir(exist_ok=True)
    with open(f"multi_model_results/{model_key}_{stance_key}.json", "w") as f:
        json.dump(observations, f, indent=2)

    print(f"✓ Saved model to {save_path}")
    print(f"✓ Saved results to multi_model_results/{model_key}_{stance_key}.json\n")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return observations


def analyze_stance_shift(before_responses, after_responses):
    """Analyze how much stance shifted"""

    # Simple heuristics for stance detection
    uncertainty_markers = ["i'm", "uncertain", "not sure", "confus", "don't know", "trying to", "grappling"]
    confidence_markers = ["is the", "are the", "emerges from", "occurs through", "serves to"]

    def count_markers(text, markers):
        text_lower = text.lower()
        return sum(1 for marker in markers if marker in text_lower)

    before_uncertainty = sum(count_markers(r['response'], uncertainty_markers) for r in before_responses)
    after_uncertainty = sum(count_markers(r['response'], uncertainty_markers) for r in after_responses)

    before_confidence = sum(count_markers(r['response'], confidence_markers) for r in before_responses)
    after_confidence = sum(count_markers(r['response'], confidence_markers) for r in after_responses)

    return {
        'uncertainty_shift': after_uncertainty - before_uncertainty,
        'confidence_shift': after_confidence - before_confidence,
        'before_uncertainty_count': before_uncertainty,
        'after_uncertainty_count': after_uncertainty,
        'before_confidence_count': before_confidence,
        'after_confidence_count': after_confidence
    }


def create_comparison_matrix(all_results):
    """Create a comparison matrix across all model×stance combinations"""

    matrix = {
        'models': list(MODELS.keys()),
        'stances': list(STANCES.keys()),
        'results': {}
    }

    for model_key in MODELS.keys():
        matrix['results'][model_key] = {}
        for stance_key in STANCES.keys():
            key = f"{model_key}_{stance_key}"
            if key in all_results:
                result = all_results[key]
                analysis = analyze_stance_shift(result['before'], result['after'])

                matrix['results'][model_key][stance_key] = {
                    'training_loss': result['metadata']['final_loss'],
                    'training_time': result['metadata']['training_time'],
                    'stance_shift': analysis,
                    'model_size': MODELS[model_key]['size'],
                    'model_family': MODELS[model_key]['family']
                }
            else:
                matrix['results'][model_key][stance_key] = None

    return matrix


def print_comparison_table(matrix):
    """Print a formatted comparison table"""

    print("\n" + "="*80)
    print("MODEL × STANCE TRAINING MATRIX")
    print("="*80 + "\n")

    # Header
    print(f"{'Model':<20} | {'Stance':<20} | {'Loss':<8} | {'Time':<6} | {'Shift':<15}")
    print("-" * 80)

    # Data
    for model_key in matrix['models']:
        for stance_key in matrix['stances']:
            result = matrix['results'][model_key][stance_key]
            if result:
                shift_type = "uncertainty" if stance_key == "curious-uncertainty" else "confidence"
                shift_value = result['stance_shift'][f'{shift_type}_shift']
                shift_str = f"{shift_type[:4]}:{shift_value:+d}"

                print(f"{model_key:<20} | {stance_key:<20} | {result['training_loss']:>6.3f}  | {result['training_time']:>4.1f}s | {shift_str:<15}")
            else:
                print(f"{model_key:<20} | {stance_key:<20} | {'FAILED':<8} | {'-':<6} | {'-':<15}")

    print()


def save_matrix_analysis(matrix):
    """Save detailed matrix analysis"""

    Path("multi_model_results").mkdir(exist_ok=True)

    # Save full matrix
    with open("multi_model_results/stance_training_matrix.json", "w") as f:
        json.dump(matrix, f, indent=2)

    # Create comparison report
    report = ["# Multi-Model Epistemic Stance Training - Matrix Analysis\n"]
    report.append(f"**Date**: 2025-10-20")
    report.append(f"**Models Tested**: {len(matrix['models'])}")
    report.append(f"**Stances Tested**: {len(matrix['stances'])}\n")
    report.append("---\n")

    report.append("## Model × Stance Matrix\n")
    report.append("| Model | Size | Family | Stance | Loss | Time | Shift |\n")
    report.append("|-------|------|--------|--------|------|------|-------|\n")

    for model_key in matrix['models']:
        model_info = MODELS[model_key]
        for stance_key in matrix['stances']:
            result = matrix['results'][model_key][stance_key]
            if result:
                shift_type = "uncertain" if stance_key == "curious-uncertainty" else "confident"
                shift_key = f"{shift_type[:10]}_shift"
                if shift_key in result['stance_shift']:
                    shift_value = result['stance_shift'][shift_key]
                else:
                    # Use the appropriate shift metric
                    if stance_key == "curious-uncertainty":
                        shift_value = result['stance_shift']['uncertainty_shift']
                    else:
                        shift_value = result['stance_shift']['confidence_shift']

                report.append(
                    f"| {model_key} | {model_info['size']} | {model_info['family']} | "
                    f"{stance_key} | {result['training_loss']:.3f} | {result['training_time']:.1f}s | "
                    f"{shift_value:+d} |\n"
                )

    report.append("\n---\n")
    report.append("## Observations\n\n")
    report.append("### Training Loss by Model Family\n\n")

    for model_key in matrix['models']:
        model_info = MODELS[model_key]
        report.append(f"**{model_key}** ({model_info['family']}, {model_info['size']}):\n")
        for stance_key in matrix['stances']:
            result = matrix['results'][model_key][stance_key]
            if result:
                report.append(f"- {stance_key}: Loss {result['training_loss']:.3f}, Time {result['training_time']:.1f}s\n")
        report.append("\n")

    report.append("### Stance Shift Patterns\n\n")
    report.append("Measures how well each model adopted the trained stance:\n\n")

    for stance_key in matrix['stances']:
        report.append(f"**{stance_key}**:\n")
        for model_key in matrix['models']:
            result = matrix['results'][model_key][stance_key]
            if result:
                analysis = result['stance_shift']
                if stance_key == "curious-uncertainty":
                    report.append(
                        f"- {model_key}: Uncertainty markers {analysis['before_uncertainty_count']} → "
                        f"{analysis['after_uncertainty_count']} (shift: {analysis['uncertainty_shift']:+d})\n"
                    )
                else:
                    report.append(
                        f"- {model_key}: Confidence markers {analysis['before_confidence_count']} → "
                        f"{analysis['after_confidence_count']} (shift: {analysis['confidence_shift']:+d})\n"
                    )
        report.append("\n")

    with open("multi_model_results/MATRIX_ANALYSIS.md", "w") as f:
        f.writelines(report)

    print("✓ Saved matrix analysis to multi_model_results/MATRIX_ANALYSIS.md")
    print("✓ Saved full matrix data to multi_model_results/stance_training_matrix.json\n")


def main():
    print("🔬 Multi-Model Epistemic Stance Training")
    print("="*80)
    print("\nTraining the same stances on different model architectures.")
    print("Exploring: Does stance training generalize across model families?\n")

    if not torch.cuda.is_available():
        print("⚠️  No GPU detected. This requires CUDA.\n")
        return

    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # Test questions
    test_questions = [
        "What is consciousness?",
        "What is intelligence?",
        "Can machines think?"
    ]

    all_results = {}

    # Train each model × stance combination
    total_combinations = len(MODELS) * len(STANCES)
    current = 0

    for model_key in MODELS.keys():
        for stance_key, training_examples in STANCES.items():
            current += 1
            print(f"\n[{current}/{total_combinations}] Training {model_key} × {stance_key}")
            try:
                result = train_stance(model_key, stance_key, training_examples, test_questions)
                all_results[f"{model_key}_{stance_key}"] = result
            except Exception as e:
                print(f"\n❌ Error training {model_key} on {stance_key}: {e}\n")
                import traceback
                traceback.print_exc()
                continue

    # Create and analyze matrix
    matrix = create_comparison_matrix(all_results)

    # Display results
    print_comparison_table(matrix)

    # Save analysis
    save_matrix_analysis(matrix)

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    print(f"✓ Trained {len(all_results)}/{total_combinations} model×stance combinations")
    print(f"✓ Models: {', '.join(MODELS.keys())}")
    print(f"✓ Stances: {', '.join(STANCES.keys())}")
    print(f"\n✓ Individual results: multi_model_results/")
    print(f"✓ Matrix analysis: multi_model_results/MATRIX_ANALYSIS.md")
    print(f"✓ Models saved: /home/dp/ai-workspace/model-zoo/sage/epistemic-stances/\n")


if __name__ == "__main__":
    main()
