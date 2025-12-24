#!/usr/bin/env python3
"""
Test threshold models WITH IRP scaffolding

Using fixed IRP (no context contamination).
Compare with bare results to find scaffolding suitability threshold.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
from pathlib import Path

# Add parent to path for IRP imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

def compute_simple_energy(response: str) -> float:
    """Simple energy metric - lower is better"""
    energy = 0.0

    # Length check
    if len(response) < 50:
        energy += 0.3

    # Proper completion
    if response and not response.rstrip().endswith(('.', '!', '?', '"')):
        energy += 0.2

    # Basic repetition
    words = response.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.7:
            energy += 0.2

    # Verbatim repetition (pattern collapse)
    if len(words) > 20:
        phrase_counts = {}
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        max_repetition = max(phrase_counts.values()) if phrase_counts else 0
        if max_repetition >= 3:
            energy += 0.5
        elif max_repetition >= 2:
            energy += 0.2

    return min(1.0, energy)


def test_with_irp(model, tokenizer, question, params):
    """
    Run IRP with fixed implementation (no context contamination)

    5 iterations, temperature reduction 0.7→0.5, clean contexts
    """
    max_iterations = params.get('max_iterations', 5)
    initial_temp = params.get('temperature', 0.7)
    temp_reduction = params.get('temperature_reduction', 0.04)

    best_response = None
    best_energy = float('inf')
    iteration_log = []

    for iteration in range(max_iterations):
        # Temperature reduction
        temp = initial_temp - (iteration * temp_reduction)
        temp = max(temp, 0.5)

        # Clean prompt each iteration (no contamination)
        prompt = f"Question: {question}\n\nAnswer:"

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=temp,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in response:
            response = response.split("Answer:")[1].strip()

        # Compute energy
        energy = compute_simple_energy(response)

        iteration_log.append({
            'iteration': iteration,
            'temperature': temp,
            'energy': energy,
            'response': response
        })

        # Keep best
        if energy < best_energy:
            best_energy = energy
            best_response = response

    return {
        'best_response': best_response,
        'best_energy': best_energy,
        'iterations': iteration_log,
        'converged': iteration_log[-1]['energy'] < iteration_log[0]['energy']
    }


def test_model_with_irp(model_path, size, question):
    """Load model and test with IRP"""
    print(f"\n{'='*80}")
    print(f"Model: {size} examples WITH IRP")
    print(f"{'='*80}")

    # Load model
    base_model = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"Loading base + LoRA adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    # Run IRP
    print(f"Question: {question}")
    print(f"Running IRP (5 iterations, temp 0.7→0.5)...")
    print()

    params = {
        'max_iterations': 5,
        'temperature': 0.7,
        'temperature_reduction': 0.04
    }

    result = test_with_irp(model, tokenizer, question, params)

    # Show iteration progression
    print("Iteration progression:")
    print("-" * 80)
    for log in result['iterations']:
        print(f"  [{log['iteration']}] temp={log['temperature']:.2f}, energy={log['energy']:.3f}")
    print("-" * 80)

    print(f"\nBest response (energy={result['best_energy']:.3f}):")
    print("-" * 80)
    print(result['best_response'])
    print("-" * 80)

    print(f"\nConverged: {result['converged']}")

    return result


def main():
    print("="*80)
    print("Threshold Models WITH IRP Scaffolding")
    print("="*80)
    print("\nTesting with fixed IRP:")
    print("- 5 iterations")
    print("- Temperature reduction 0.7 → 0.5")
    print("- Clean contexts (no contamination)")
    print("- Energy-based selection")
    print()

    question = "What does it mean to be conscious?"

    sizes = [40, 60, 80, 100]
    results = {}

    for size in sizes:
        model_path = f"threshold_models/{size}examples_model/final_model"
        try:
            result = test_model_with_irp(model_path, size, question)
            results[size] = result
        except Exception as e:
            print(f"❌ Error testing {size}-example model: {e}")
            import traceback
            traceback.print_exc()
            results[size] = None

    # Summary comparison
    print("\n" + "="*80)
    print("Comparison: Bare vs IRP")
    print("="*80)
    print("\nRecall bare results:")
    print("  40: Thoughtful philosophical exploration")
    print("  60: Systematic analysis, verification problem")
    print("  80: Multi-perspective, slightly repetitive")
    print(" 100: PATTERN COLLAPSE (verbatim repetition)")
    print()
    print("IRP results:")
    for size in sizes:
        if results[size]:
            energy = results[size]['best_energy']
            converged = results[size]['converged']
            conv_str = "✓" if converged else "✗"
            print(f"  {size:3d}: energy={energy:.3f}, converged={conv_str}")

    print("\n" + "="*80)
    print("Research Questions")
    print("="*80)
    print("\n1. Does IRP help or hurt at each size?")
    print("2. Does it prevent the 100-example collapse?")
    print("3. Where does scaffolding switch from harmful → helpful?")
    print("4. What are the models actually saying with IRP?")
    print("\n(Look at responses above to answer)")


if __name__ == "__main__":
    main()
