#!/usr/bin/env python3
"""
Test vLLM with FP4 quantized Qwen3-Omni-30B model.
This is the moment of truth - will vLLM realize the 4x memory and 7x speed benefits?
"""

import torch
import time
from vllm import LLM, SamplingParams

def main():
    print("=" * 80)
    print("vLLM FP4 Performance Test - Jetson AGX Thor")
    print("=" * 80)
    print()

    # Configuration
    fp4_model_path = "model-zoo/sage/omni-modal/qwen3-omni-30b-fp4-weight-only"

    print("📋 Test Configuration:")
    print(f"  Model: {fp4_model_path}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print()

    # Test prompt
    prompts = [
        "Hello! How are you today?",
        "What is the capital of France?",
        "Explain what FP4 quantization is in simple terms.",
    ]

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=50,
    )

    print("🚀 Step 1: Loading FP4 model with vLLM...")
    print("  This should use ~16-20GB instead of 66GB!")
    print()

    start_load = time.time()

    try:
        llm = LLM(
            model=fp4_model_path,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            dtype="auto",
            # Let vLLM auto-detect quantization
        )

        load_time = time.time() - start_load

        # Get memory usage
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)

        print(f"✅ Model loaded in {load_time:.1f}s")
        print(f"📊 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        print()

        # Test generation
        print("🎯 Step 2: Testing generation...")
        print("-" * 80)

        start_gen = time.time()
        outputs = llm.generate(prompts, sampling_params)
        gen_time = time.time() - start_gen

        # Calculate total tokens
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        tokens_per_sec = total_tokens / gen_time if gen_time > 0 else 0

        print(f"\n✅ Generated {total_tokens} tokens in {gen_time:.2f}s")
        print(f"⚡ Throughput: {tokens_per_sec:.2f} tok/s")
        print()

        # Show outputs
        print("📝 Sample Outputs:")
        print("-" * 80)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            tokens = len(output.outputs[0].token_ids)

            print(f"\n[{i+1}] Prompt: {prompt[:60]}...")
            print(f"    Output: {generated_text[:100]}...")
            print(f"    Tokens: {tokens}")

        print()
        print("=" * 80)
        print("✅ SUCCESS!")
        print("=" * 80)
        print()
        print(f"📊 Performance Summary:")
        print(f"  Load Time: {load_time:.1f}s")
        print(f"  GPU Memory: {allocated:.2f} GB (vs 65.72 GB with HuggingFace)")
        print(f"  Throughput: {tokens_per_sec:.2f} tok/s (vs 1.34 tok/s with HuggingFace)")
        print()

        if allocated < 30:
            print(f"  🎉 Memory Reduction: {65.72 / allocated:.1f}x")
        if tokens_per_sec > 2:
            print(f"  🚀 Speed Improvement: {tokens_per_sec / 1.34:.1f}x")

        print()
        print("🎯 Next: Compare with original model to validate quality")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
