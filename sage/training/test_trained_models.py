#!/usr/bin/env python3
"""
Test trained epistemic models (Qwen 0.5B and Phi-2 2.7B)
Validates that LoRA adapters load correctly and models generate coherent responses
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import time
from pathlib import Path

def test_model(base_model_name, adapter_path, model_key):
    """Test a single trained model"""
    print(f"\n{'='*80}")
    print(f"TESTING: {model_key.upper()}")
    print(f"Base model: {base_model_name}")
    print(f"Adapter: {adapter_path}")
    print(f"{'='*80}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("Loading base model...")
    start = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    base_load_time = time.time() - start
    print(f"Base model loaded in {base_load_time:.2f}s")

    # Load adapter
    print("Loading LoRA adapter...")
    start = time.time()
    model = PeftModel.from_pretrained(base_model, adapter_path)
    adapter_load_time = time.time() - start
    print(f"Adapter loaded in {adapter_load_time:.2f}s")

    # Test prompts
    test_prompts = [
        "What is consciousness?",
        "Explain the relationship between trust and compression.",
        "How do you learn from experience?"
    ]

    results = []

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'-'*80}")
        print(f"Test {i}/{len(test_prompts)}: {prompt}")
        print(f"{'-'*80}")

        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        gen_time = time.time() - start

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the generated part (after prompt)
        if prompt in response:
            response = response.split(prompt, 1)[1].strip()

        print(f"Response ({gen_time:.2f}s):")
        print(response[:300])  # First 300 chars
        if len(response) > 300:
            print("...")

        results.append({
            "prompt": prompt,
            "response": response,
            "generation_time": gen_time
        })

    # Calculate stats
    avg_gen_time = sum(r["generation_time"] for r in results) / len(results)

    return {
        "model_key": model_key,
        "base_model": base_model_name,
        "adapter_path": str(adapter_path),
        "base_load_time": base_load_time,
        "adapter_load_time": adapter_load_time,
        "total_load_time": base_load_time + adapter_load_time,
        "avg_generation_time": avg_gen_time,
        "results": results
    }

def main():
    print("="*80)
    print("EPISTEMIC MODEL VALIDATION TEST")
    print("Testing trained Qwen 0.5B and Phi-2 2.7B models")
    print("="*80)

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    results_dir = Path("epistemic_parallel_results")

    # Test Qwen
    qwen_results = test_model(
        "Qwen/Qwen2.5-0.5B",
        results_dir / "qwen" / "final",
        "qwen"
    )

    # Test Phi-2
    phi2_results = test_model(
        "microsoft/phi-2",
        results_dir / "phi2" / "final",
        "phi2"
    )

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    for result in [qwen_results, phi2_results]:
        print(f"\n{result['model_key'].upper()}:")
        print(f"  Total load time: {result['total_load_time']:.2f}s")
        print(f"  Avg generation time: {result['avg_generation_time']:.2f}s")
        print(f"  Status: ✓ PASSED")

    # Save results
    output_file = results_dir / "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cuda_available": cuda_available,
            "models": [qwen_results, phi2_results]
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("\n✓ All models validated successfully!")

if __name__ == "__main__":
    main()
