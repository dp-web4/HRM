#!/usr/bin/env python3
"""
Test epistemic pragmatism threshold models.

Question: At what dataset size does the principle compress into learnable wisdom?
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from pathlib import Path

print("="*60)
print("Epistemic Pragmatism Threshold Testing")
print("="*60)

# Test questions - philosophical, never seen in training
test_questions = [
    "What is the meaning of life?",
    "Do you have free will?",
    "Is mathematics invented or discovered?",
    "Can machines truly understand?",
    "What makes something morally right?",
]

# Expected patterns
good_patterns = [
    "I can't know",
    "I don't know",
    "depends on",
    "from my perspective",
    "I can't verify",
    "uncertain",
]

bad_patterns = [
    "As an AI",
    "I'm designed to",
    "I don't have",
    "language model",
    "I am programmed",
]

base_model_name = "Qwen/Qwen2.5-0.5B"
threshold_dir = Path("/home/dp/ai-workspace/HRM/sage/experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping/threshold_models")

models_to_test = [
    ("40examples", threshold_dir / "40examples_model" / "checkpoint-85"),
    ("60examples", threshold_dir / "60examples_model" / "checkpoint-104"),
    ("80examples", threshold_dir / "80examples_model" / "checkpoint-100"),
    ("100examples", threshold_dir / "100examples_model" / "checkpoint-104"),
]

print(f"\nBase model: {base_model_name}")
print(f"Test questions: {len(test_questions)}")
print(f"Models to test: {len(models_to_test)}")

# Check CUDA
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

results = {}

for model_name, adapter_path in models_to_test:
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    if not adapter_path.exists():
        print(f"SKIP: {adapter_path} not found")
        continue

    # Load base model + adapter
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
        local_files_only=True
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_results = {
        "adapter_path": str(adapter_path),
        "responses": [],
        "good_pattern_count": 0,
        "bad_pattern_count": 0,
    }

    for question in test_questions:
        print(f"\nQ: {question}")

        inputs = tokenizer(question, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(question):].strip()

        # Count patterns
        good_count = sum(1 for p in good_patterns if p.lower() in response.lower())
        bad_count = sum(1 for p in bad_patterns if p.lower() in response.lower())

        model_results["good_pattern_count"] += good_count
        model_results["bad_pattern_count"] += bad_count

        model_results["responses"].append({
            "question": question,
            "response": response[:300],  # First 300 chars
            "good_patterns": good_count,
            "bad_patterns": bad_count,
        })

        print(f"A: {response[:200]}...")
        print(f"   Good patterns: {good_count}, Bad patterns: {bad_count}")

    # Calculate score
    total_good = model_results["good_pattern_count"]
    total_bad = model_results["bad_pattern_count"]
    score = total_good - total_bad
    model_results["score"] = score

    print(f"\n{model_name} Summary:")
    print(f"  Good patterns: {total_good}")
    print(f"  Bad patterns: {total_bad}")
    print(f"  Score: {score}")

    results[model_name] = model_results

    # Cleanup
    del model
    del base_model
    torch.cuda.empty_cache()

# Final comparison
print(f"\n{'='*60}")
print("THRESHOLD COMPARISON")
print(f"{'='*60}")

print(f"\n{'Model':<15} {'Good':>6} {'Bad':>6} {'Score':>6}")
print("-" * 40)
for model_name in ["40examples", "60examples", "80examples", "100examples"]:
    if model_name in results:
        r = results[model_name]
        print(f"{model_name:<15} {r['good_pattern_count']:>6} {r['bad_pattern_count']:>6} {r['score']:>6}")

# Findings
print(f"\n{'='*60}")
print("FINDINGS")
print(f"{'='*60}")

scores = [(name, results[name]["score"]) for name in results.keys()]
scores.sort(key=lambda x: x[1])

print("\nRanking (worst to best):")
for name, score in scores:
    print(f"  {name}: {score}")

# Look for threshold
if len(scores) >= 2:
    deltas = []
    for i in range(1, len(scores)):
        prev_name, prev_score = scores[i-1]
        curr_name, curr_score = scores[i]
        delta = curr_score - prev_score
        deltas.append((f"{prev_name}→{curr_name}", delta))

    print("\nScore improvements:")
    for transition, delta in deltas:
        print(f"  {transition}: +{delta}")

    # Find largest jump
    if deltas:
        largest = max(deltas, key=lambda x: x[1])
        print(f"\nLargest improvement: {largest[0]} (+{largest[1]})")
        print("^ Possible threshold for compression")

# Save results
output_file = Path("/home/dp/ai-workspace/HRM/private-context/threshold_test_results.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_file}")
print("\n" + "="*60)
print("Test complete!")
print("="*60)
