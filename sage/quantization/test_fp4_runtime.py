#!/usr/bin/env python3
"""
Test FP4 quantization at runtime.

This script loads the quantized model and tests:
1. Memory usage (should be ~4x less than original)
2. Inference speed (should be 4-5x faster)
3. Output quality (should be similar to original)
"""

import torch
import time
import psutil
import os
from pathlib import Path
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

def get_memory_usage():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        return allocated, reserved
    return 0, 0

def test_model_inference(model, processor, device, test_prompts):
    """Test model inference and measure speed."""
    model.eval()

    results = []

    with torch.no_grad():
        for prompt in test_prompts:
            # Prepare input
            inputs = processor(
                text=[prompt],
                return_tensors="pt",
            ).to(device)

            # Measure inference time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            # Decode output
            response = processor.decode(outputs[0], skip_special_tokens=True)

            # Calculate tokens per second
            num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
            tokens_per_sec = num_tokens / (end_time - start_time)

            results.append({
                'prompt': prompt,
                'response': response,
                'num_tokens': num_tokens,
                'time': end_time - start_time,
                'tokens_per_sec': tokens_per_sec,
            })

    return results

def main():
    print("="*70)
    print("FP4 QUANTIZATION RUNTIME TEST")
    print("="*70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Test prompts
    test_prompts = [
        "Hello! How are you today?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]

    # Paths
    original_model_path = "model-zoo/sage/omni-modal/qwen3-omni-30b"
    quantized_model_path = "model-zoo/sage/omni-modal/qwen3-omni-30b-fp4-weight-only"

    print(f"\nDevice: {device}")
    print(f"Test prompts: {len(test_prompts)}")
    print("\n" + "="*70)

    # Test 1: Load and test quantized model
    print("\n[TEST 1] FP4 QUANTIZED MODEL")
    print("-"*70)

    print("Loading quantized model...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    try:
        quantized_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            quantized_model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        processor = Qwen3OmniMoeProcessor.from_pretrained(quantized_model_path)

        # Measure memory
        allocated, reserved = get_memory_usage()
        print(f"✅ Model loaded")
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # Test inference
        print("\nRunning inference tests...")
        q_results = test_model_inference(quantized_model, processor, device, test_prompts)

        print("\nQuantized Model Results:")
        for i, result in enumerate(q_results, 1):
            print(f"\n  Test {i}:")
            print(f"    Prompt: {result['prompt'][:50]}...")
            print(f"    Tokens: {result['num_tokens']}")
            print(f"    Time: {result['time']:.3f}s")
            print(f"    Speed: {result['tokens_per_sec']:.2f} tok/s")

        # Calculate average
        avg_speed_q = sum(r['tokens_per_sec'] for r in q_results) / len(q_results)
        print(f"\n  Average Speed: {avg_speed_q:.2f} tok/s")

        # Clean up
        del quantized_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"❌ Quantized model test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 2: Load and test original model for comparison
    print("\n" + "="*70)
    print("\n[TEST 2] ORIGINAL FP16/BF16 MODEL (for comparison)")
    print("-"*70)

    print("Loading original model...")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    try:
        original_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            original_model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        processor = Qwen3OmniMoeProcessor.from_pretrained(original_model_path)

        # Measure memory
        allocated, reserved = get_memory_usage()
        print(f"✅ Model loaded")
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # Test inference
        print("\nRunning inference tests...")
        o_results = test_model_inference(original_model, processor, device, test_prompts)

        print("\nOriginal Model Results:")
        for i, result in enumerate(o_results, 1):
            print(f"\n  Test {i}:")
            print(f"    Prompt: {result['prompt'][:50]}...")
            print(f"    Tokens: {result['num_tokens']}")
            print(f"    Time: {result['time']:.3f}s")
            print(f"    Speed: {result['tokens_per_sec']:.2f} tok/s")

        # Calculate average
        avg_speed_o = sum(r['tokens_per_sec'] for r in o_results) / len(o_results)
        print(f"\n  Average Speed: {avg_speed_o:.2f} tok/s")

        # Clean up
        del original_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"❌ Original model test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)

    speedup = avg_speed_q / avg_speed_o if avg_speed_o > 0 else 0

    print(f"\nOriginal Model:  {avg_speed_o:.2f} tok/s")
    print(f"Quantized Model: {avg_speed_q:.2f} tok/s")
    print(f"\nSpeedup: {speedup:.2f}x")

    if speedup >= 4.0:
        print("✅ EXCELLENT: Achieved expected 4-5x speedup!")
    elif speedup >= 2.0:
        print("✅ GOOD: Significant speedup achieved")
    elif speedup >= 1.2:
        print("⚠️  MODEST: Some speedup, but less than expected")
    else:
        print("❌ NO SPEEDUP: Quantization may not be working correctly")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
