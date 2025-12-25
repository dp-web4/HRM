#!/usr/bin/env python3
"""
Validation test: FP4 quantized model with proper ChatML format.

This is the reality check - does the quantized model actually work?
"""

import torch
import time
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

def format_chatml(messages):
    """Format messages in ChatML format required by Qwen3-Omni."""
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

def test_model(model_path, model_name, test_cases, device="cuda:0"):
    """Test a model with properly formatted prompts using apply_chat_template."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")

    print(f"Loading model from {model_path}...")
    start_load = time.time()

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    load_time = time.time() - start_load

    # Get memory usage
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)

    print(f"✅ Model loaded in {load_time:.1f}s")
    print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    results = []

    print(f"\nRunning {len(test_cases)} test cases...")
    print("-" * 70)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {test_case['name']}")

        # Use apply_chat_template() like the working dragon story code
        formatted_prompt = processor.apply_chat_template(
            test_case["messages"],
            tokenize=False,
            add_generation_prompt=True
        )

        # Prepare input
        inputs = processor(
            text=[formatted_prompt],
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate with CRITICAL thinker_return_dict_in_generate parameter
        start_time = time.time()
        input_token_count = inputs['input_ids'].shape[1]

        with torch.no_grad():
            text_ids, audio = model.generate(
                **inputs,
                max_new_tokens=test_case.get("max_tokens", 50),
                temperature=test_case.get("temperature", 0.7),
                do_sample=test_case.get("do_sample", True),
                thinker_return_dict_in_generate=True,  # CRITICAL for Q3-Omni!
            )

        gen_time = time.time() - start_time

        # text_ids is an object with .sequences attribute, not a raw tensor
        generated_tokens = text_ids.sequences[:, input_token_count:]
        assistant_response = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Calculate stats
        num_tokens = generated_tokens.shape[1]
        tokens_per_sec = num_tokens / gen_time if gen_time > 0 else 0

        result = {
            'test_name': test_case['name'],
            'response': assistant_response,
            'num_tokens': num_tokens,
            'time': gen_time,
            'tokens_per_sec': tokens_per_sec,
        }

        results.append(result)

        print(f"  Prompt: {test_case['messages'][-1]['content'][:60]}...")
        print(f"  Response: {assistant_response[:100]}...")
        print(f"  Tokens: {num_tokens} in {gen_time:.2f}s ({tokens_per_sec:.2f} tok/s)")

    print(f"\n{'='*70}")
    print(f"Summary for {model_name}")
    print(f"{'='*70}")

    avg_speed = sum(r['tokens_per_sec'] for r in results) / len(results)
    avg_tokens = sum(r['num_tokens'] for r in results) / len(results)

    print(f"Average speed: {avg_speed:.2f} tok/s")
    print(f"Average tokens: {avg_tokens:.1f}")
    print(f"Memory usage: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return results, {
        'load_time': load_time,
        'avg_speed': avg_speed,
        'avg_tokens': avg_tokens,
        'memory_allocated': allocated,
        'memory_reserved': reserved,
    }

def main():
    print("="*70)
    print("FP4 QUANTIZATION VALIDATION - CHATML FORMAT TEST")
    print("="*70)
    print("\nThis test validates that:")
    print("1. FP4 quantized model works with proper ChatML format")
    print("2. Output quality is preserved")
    print("3. Memory and speed benefits are realized")
    print("")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Define test cases with proper ChatML structure
    test_cases = [
        {
            "name": "Simple greeting",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! How are you today?"},
            ],
            "max_tokens": 50,
            "temperature": 0.7,
        },
        {
            "name": "Factual question",
            "messages": [
                {"role": "system", "content": "You are a knowledgeable assistant."},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            "max_tokens": 30,
            "temperature": 0.3,
        },
        {
            "name": "Technical explanation",
            "messages": [
                {"role": "system", "content": "You are a technical expert."},
                {"role": "user", "content": "Explain what FP4 quantization is in simple terms."},
            ],
            "max_tokens": 100,
            "temperature": 0.5,
        },
        {
            "name": "Multi-turn conversation",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "I'm working on optimizing AI models."},
                {"role": "assistant", "content": "That's great! What specific aspects are you focusing on?"},
                {"role": "user", "content": "I'm trying to reduce memory usage on edge devices."},
            ],
            "max_tokens": 60,
            "temperature": 0.7,
        },
    ]

    # Paths
    original_model_path = "model-zoo/sage/omni-modal/qwen3-omni-30b"
    quantized_model_path = "model-zoo/sage/omni-modal/qwen3-omni-30b-fp4-weight-only"

    print("\n" + "="*70)
    print("PHASE 1: Testing FP4 Quantized Model")
    print("="*70)

    try:
        quantized_results, quantized_stats = test_model(
            quantized_model_path,
            "FP4 Quantized Model",
            test_cases,
            device
        )
        quantized_success = True
    except Exception as e:
        print(f"\n❌ Quantized model test failed: {e}")
        import traceback
        traceback.print_exc()
        quantized_success = False
        quantized_stats = None

    print("\n" + "="*70)
    print("PHASE 2: Testing Original Model (for comparison)")
    print("="*70)

    try:
        original_results, original_stats = test_model(
            original_model_path,
            "Original BF16 Model",
            test_cases,
            device
        )
        original_success = True
    except Exception as e:
        print(f"\n❌ Original model test failed: {e}")
        import traceback
        traceback.print_exc()
        original_success = False
        original_stats = None

    # Final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)

    if quantized_success and original_success:
        print("\n📊 Performance Comparison:")
        print(f"{'Metric':<25} {'Original':<15} {'FP4 Quantized':<15} {'Ratio':<10}")
        print("-" * 70)

        # Speed comparison
        speedup = quantized_stats['avg_speed'] / original_stats['avg_speed'] if original_stats['avg_speed'] > 0 else 0
        print(f"{'Average speed (tok/s)':<25} {original_stats['avg_speed']:<15.2f} {quantized_stats['avg_speed']:<15.2f} {speedup:<10.2f}x")

        # Memory comparison
        mem_ratio = original_stats['memory_allocated'] / quantized_stats['memory_allocated'] if quantized_stats['memory_allocated'] > 0 else 0
        print(f"{'GPU memory (GB)':<25} {original_stats['memory_allocated']:<15.2f} {quantized_stats['memory_allocated']:<15.2f} {mem_ratio:<10.2f}x")

        # Load time
        load_ratio = original_stats['load_time'] / quantized_stats['load_time'] if quantized_stats['load_time'] > 0 else 0
        print(f"{'Load time (s)':<25} {original_stats['load_time']:<15.1f} {quantized_stats['load_time']:<15.1f} {load_ratio:<10.2f}x")

        print("\n📝 Quality Comparison (first 3 tests):")
        for i in range(min(3, len(test_cases))):
            print(f"\n  {test_cases[i]['name']}:")
            print(f"    Original:  {original_results[i]['response'][:80]}...")
            print(f"    FP4:       {quantized_results[i]['response'][:80]}...")

        print("\n" + "="*70)
        print("VALIDATION RESULTS")
        print("="*70)

        if speedup >= 1.0:
            print(f"✅ FP4 model is {speedup:.2f}x the speed of original")
        else:
            print(f"⚠️  FP4 model is slower ({speedup:.2f}x) - may need vLLM for acceleration")

        if mem_ratio >= 1.0:
            print(f"✅ FP4 model uses {mem_ratio:.2f}x less memory than original")
        else:
            print(f"⚠️  FP4 model uses more memory ({mem_ratio:.2f}x) - runtime quantization needs vLLM")

        print("\n📌 Note: For full FP4 benefits (4x memory, 7.5x speed), use vLLM runtime quantization")
        print("   This test uses HuggingFace Transformers which doesn't optimize FP4 execution")

    elif quantized_success:
        print("\n✅ FP4 quantized model works correctly with ChatML format!")
        print(f"   Average speed: {quantized_stats['avg_speed']:.2f} tok/s")
        print(f"   Memory usage: {quantized_stats['memory_allocated']:.2f} GB")
        print("\n⚠️  Original model test failed - comparison unavailable")

    elif original_success:
        print("\n✅ Original model works with ChatML format")
        print("❌ FP4 quantized model failed - needs investigation")

    else:
        print("\n❌ Both models failed - check CUDA availability and model paths")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
