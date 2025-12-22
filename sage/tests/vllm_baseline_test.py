#!/usr/bin/env python3
"""
Generate baseline outputs from Q3-Omni using vLLM for comparison with our segmented implementation.
This script runs inside the NVIDIA vLLM container.
"""

from vllm import LLM, SamplingParams
import json

# Test prompts for baseline comparison
TEST_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time",
    "The meaning of life is",
    "To be or not to be",
]

def main():
    print("Loading Q3-Omni with vLLM...")
    print(f"Model path: /models/qwen3-omni-30b")

    # Initialize vLLM with Q3-Omni
    llm = LLM(
        model="/models/qwen3-omni-30b",
        tensor_parallel_size=1,
        max_model_len=512,  # Very conservative for memory
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        trust_remote_code=True,
    )

    print("Model loaded successfully!")
    print(f"\nGenerating baseline outputs for {len(TEST_PROMPTS)} test prompts...")

    # Sampling parameters - deterministic for reproducibility
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy sampling
        max_tokens=50,
        top_p=1.0,
    )

    results = []

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Prompt: '{prompt}'")

        outputs = llm.generate([prompt], sampling_params)

        for output in outputs:
            generated_text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids

            print(f"  Output: '{generated_text}'")
            print(f"  Tokens: {len(token_ids)}")

            results.append({
                "prompt": prompt,
                "output": generated_text,
                "token_ids": token_ids,
                "num_tokens": len(token_ids),
            })

    # Save results
    output_file = "/workspace/vllm_baseline_outputs.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nBaseline outputs saved to: {output_file}")
    print("\n=== Summary ===")
    for result in results:
        print(f"Prompt: '{result['prompt']}' -> {result['num_tokens']} tokens")

if __name__ == "__main__":
    main()
