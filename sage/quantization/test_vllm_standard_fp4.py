#!/usr/bin/env python3
"""
Test FP4 quantized Qwen3-Omni with standard vLLM (no vLLM-Omni extension).

This tests whether vLLM v0.14 natively supports Qwen3-Omni architecture
and can use the ModelOpt FP4 quantization metadata.

Expected outcomes:
1. Best case: vLLM loads model, recognizes FP4 metadata, runs efficiently
2. Medium case: vLLM loads model but doesn't use FP4 optimization
3. Worst case: vLLM doesn't recognize Qwen3-Omni architecture
"""

import sys
sys.path.insert(0, '/home/dp/ai-workspace/vllm-source')

import torch
import time
import traceback
from pathlib import Path

def test_vllm_standard_fp4():
    print("=" * 80)
    print("Testing FP4 Model with Standard vLLM (No vLLM-Omni)")
    print("=" * 80)
    print()

    # Model path
    fp4_model_path = "model-zoo/sage/omni-modal/qwen3-omni-30b-fp4-weight-only"

    print("📋 Configuration:")
    print(f"  Model: {fp4_model_path}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print()

    # Check model exists
    model_path = Path(fp4_model_path)
    if not model_path.exists():
        print(f"❌ Model not found at {fp4_model_path}")
        return 1

    print(f"✅ Model found: {model_path.absolute()}")
    print()

    # Test prompts
    prompts = [
        "Hello! How are you today?",
        "What is the capital of France?",
    ]

    print("🚀 Attempting to load model with standard vLLM...")
    print()

    try:
        # Try importing vLLM
        from vllm import LLM, SamplingParams
        import vllm
        print(f"✅ vLLM imported successfully: {vllm.__version__}")
        print()

        # Get initial memory
        torch.cuda.empty_cache()
        initial_mem = torch.cuda.memory_allocated() / (1024**3)
        print(f"📊 Initial GPU memory: {initial_mem:.2f} GB")
        print()

        print("Loading model (this may take a few minutes)...")
        start_load = time.time()

        # Try to load with standard vLLM
        llm = LLM(
            model=fp4_model_path,
            trust_remote_code=True,  # Needed for custom architectures
            dtype="auto",  # Let vLLM decide
            gpu_memory_utilization=0.85,
            max_model_len=2048,  # Conservative for testing
        )

        load_time = time.time() - start_load

        # Get memory after loading
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)

        print()
        print(f"✅ Model loaded successfully in {load_time:.1f}s!")
        print(f"📊 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        print()

        # Check if this is better than HuggingFace baseline
        hf_baseline_mem = 65.72  # From validation results
        if allocated < hf_baseline_mem * 0.9:
            print(f"🎉 Memory usage improved! {hf_baseline_mem/allocated:.2f}x reduction vs HuggingFace")
        else:
            print(f"⚠️  Memory usage similar to HuggingFace ({allocated:.2f} GB vs {hf_baseline_mem:.2f} GB)")
        print()

        # Test generation
        print("🎯 Testing generation...")
        print("-" * 80)

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=50,
        )

        start_gen = time.time()
        outputs = llm.generate(prompts, sampling_params)
        gen_time = time.time() - start_gen

        # Calculate throughput
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        tokens_per_sec = total_tokens / gen_time if gen_time > 0 else 0

        print()
        print(f"✅ Generated {total_tokens} tokens in {gen_time:.2f}s")
        print(f"⚡ Throughput: {tokens_per_sec:.2f} tok/s")
        print()

        # Check if this is better than HuggingFace baseline
        hf_baseline_speed = 1.34  # From validation results
        if tokens_per_sec > hf_baseline_speed * 1.5:
            print(f"🎉 Speed improved! {tokens_per_sec/hf_baseline_speed:.2f}x faster than HuggingFace")
        else:
            print(f"⚠️  Speed similar to HuggingFace ({tokens_per_sec:.2f} vs {hf_baseline_speed:.2f} tok/s)")
        print()

        # Show outputs
        print("📝 Sample Outputs:")
        print("-" * 80)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            tokens = len(output.outputs[0].token_ids)

            print(f"\n[{i+1}] Prompt: {prompt}")
            print(f"    Output: {generated_text}")
            print(f"    Tokens: {tokens}")

        print()
        print("=" * 80)
        print("✅ SUCCESS - Standard vLLM Works with Qwen3-Omni FP4!")
        print("=" * 80)
        print()
        print("📊 Performance Summary:")
        print(f"  Load Time: {load_time:.1f}s")
        print(f"  GPU Memory: {allocated:.2f} GB (vs {hf_baseline_mem:.2f} GB with HuggingFace)")
        print(f"  Throughput: {tokens_per_sec:.2f} tok/s (vs {hf_baseline_speed:.2f} tok/s with HuggingFace)")
        print()

        if allocated < hf_baseline_mem * 0.5:
            improvement = hf_baseline_mem / allocated
            print(f"  🎉 Memory Reduction: {improvement:.1f}x")

        if tokens_per_sec > hf_baseline_speed * 2:
            improvement = tokens_per_sec / hf_baseline_speed
            print(f"  🚀 Speed Improvement: {improvement:.1f}x")

        print()
        return 0

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ FAILED - Standard vLLM Cannot Load Model")
        print("=" * 80)
        print()
        print(f"Error: {e}")
        print()
        print("Full traceback:")
        traceback.print_exc()
        print()

        # Analyze the error
        error_str = str(e).lower()
        if "qwen3" in error_str or "omni" in error_str:
            print("💡 Analysis: vLLM doesn't recognize Qwen3-Omni architecture")
            print("   This model may require vLLM-Omni extension or custom model registration")
        elif "quantization" in error_str or "fp4" in error_str:
            print("💡 Analysis: vLLM may not support ModelOpt FP4 quantization format")
            print("   May need to re-quantize with llm-compressor for vLLM compatibility")
        else:
            print("💡 Analysis: Unknown error - see traceback above")

        print()
        print("🔄 Alternative Options:")
        print("  1. Re-quantize with llm-compressor (vLLM-native FP4)")
        print("  2. Use HuggingFace baseline (validated, working)")
        print("  3. Wait for vLLM-Omni compatibility fix")
        print()

        return 1

if __name__ == "__main__":
    exit(test_vllm_standard_fp4())
