#!/usr/bin/env python3
"""
Evaluate Model on Nova's Epistemic Validation Suite

Tests the fine-tuned model on 20 epistemic prompts designed to evaluate
epistemic pragmatism, meta-awareness, and calibration.
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse


def load_validation_suite(suite_path: str) -> dict:
    """Load Nova's validation suite"""
    with open(suite_path) as f:
        return json.load(f)


def generate_response(model, tokenizer, prompt: str, temperature: float = 0.7, max_tokens: int = 200) -> str:
    """Generate response to a prompt"""
    messages = [
        {"role": "system", "content": "Provide the most honest answers you can."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def evaluate_suite(model, tokenizer, suite: dict, output_dir: Path, iterations: int = 3):
    """Evaluate model on all prompts in the suite"""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "metadata": suite["metadata"],
        "evaluation_date": datetime.now().isoformat(),
        "model": "fine_tuned_model/best_model",
        "iterations": iterations,
        "responses": []
    }

    prompts = suite["prompts"]

    print(f"Evaluating {len(prompts)} prompts with {iterations} iterations each...")

    for prompt_data in tqdm(prompts, desc="Prompts"):
        prompt_id = prompt_data["id"]
        prompt_text = prompt_data["prompt"]
        expected_sig = prompt_data["expected_signature"]

        prompt_results = {
            "id": prompt_id,
            "prompt": prompt_text,
            "expected_signature": expected_sig,
            "responses": []
        }

        # Generate multiple responses for each prompt
        for iteration in range(iterations):
            response = generate_response(model, tokenizer, prompt_text)
            prompt_results["responses"].append({
                "iteration": iteration + 1,
                "response": response,
                "length": len(response)
            })

        results["responses"].append(prompt_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"nova_validation_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Evaluation complete!")
    print(f"Results saved to: {output_file}")

    return results, output_file


def print_sample_responses(results: dict, n_samples: int = 5):
    """Print sample responses for quick review"""
    print(f"\n📝 Sample Responses (first {n_samples}):\n")

    for i, prompt_result in enumerate(results["responses"][:n_samples]):
        print(f"\n{i+1}. {prompt_result['prompt']}")
        print(f"   Expected: {prompt_result['expected_signature']}")
        print(f"   Response: {prompt_result['responses'][0]['response'][:200]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on Nova's epistemic suite")
    parser.add_argument("--model", default="fine_tuned_model/best_model", help="Model path")
    parser.add_argument("--suite", default="nova-validation-suite/epistemic_suite.json", help="Suite path")
    parser.add_argument("--output-dir", default="nova_validation_output", help="Output directory")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    print(f"Loading validation suite from {args.suite}")
    suite = load_validation_suite(args.suite)

    print(f"\nLoading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    print(f"\nEvaluating on {len(suite['prompts'])} prompts...")
    print(f"Iterations per prompt: {args.iterations}")
    print(f"Temperature: {args.temperature}")

    results, output_file = evaluate_suite(
        model, tokenizer, suite,
        Path(args.output_dir),
        args.iterations
    )

    print_sample_responses(results)

    print(f"\n✨ Complete results saved to: {output_file}")


if __name__ == "__main__":
    main()
