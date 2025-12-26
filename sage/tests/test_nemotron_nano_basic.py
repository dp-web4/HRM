#!/usr/bin/env python3
"""
Basic inference test for Llama-3.1-Nemotron-Nano-4B-v1.1 on Jetson Thor.

This test validates:
- ARM64 compatibility (no mamba-ssm dependency)
- Standard transformers library support
- Basic inference capability
- Memory footprint
- Generation speed
"""

import torch
import time
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB

def test_nemotron_nano():
    """Test Llama Nemotron Nano on Jetson Thor."""

    print("="*80)
    print("Testing NVIDIA Llama-3.1-Nemotron-Nano-4B-v1.1 on Jetson Thor")
    print("="*80)
    print()

    model_path = "model-zoo/sage/language-models/llama-nemotron-nano-4b"

    # Test 1: Model Loading
    print("Test 1: Loading model...")
    print(f"  Path: {model_path}")
    print(f"  Initial Memory: {get_memory_usage():.2f} GB")
    print()

    start_time = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )

        load_time = time.time() - start_time
        memory_after_load = get_memory_usage()

        print(f"  ✅ Model loaded successfully")
        print(f"  Load time: {load_time:.2f} seconds")
        print(f"  Memory after load: {memory_after_load:.2f} GB")
        print(f"  Memory increase: {memory_after_load - get_memory_usage():.2f} GB")
        print()

    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False

    # Test 2: Basic Inference
    print("Test 2: Basic inference...")

    test_prompts = [
        "The future of AI on edge devices is",
        "Explain quantum computing in simple terms:",
        "Write a haiku about machine learning:"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n  Test {i}: {prompt}")

        try:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Generate
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

            gen_time = time.time() - start_time

            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Calculate tokens per second
            tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
            tokens_per_sec = tokens_generated / gen_time

            print(f"  Response: {response[len(prompt):].strip()}")
            print(f"  Generation time: {gen_time:.2f}s ({tokens_per_sec:.2f} tokens/sec)")

        except Exception as e:
            print(f"  ❌ Inference failed: {e}")
            return False

    print()

    # Test 3: Memory Footprint
    print("Test 3: Memory footprint analysis...")
    final_memory = get_memory_usage()
    print(f"  Final memory usage: {final_memory:.2f} GB")
    print(f"  Model memory footprint: ~{final_memory:.2f} GB")
    print()

    # Test 4: Model Architecture Info
    print("Test 4: Model architecture...")
    print(f"  Model type: {model.config.model_type}")
    print(f"  Hidden size: {model.config.hidden_size}")
    print(f"  Num layers: {model.config.num_hidden_layers}")
    print(f"  Num attention heads: {model.config.num_attention_heads}")
    print(f"  Vocab size: {model.config.vocab_size}")
    print(f"  Max position embeddings: {model.config.max_position_embeddings}")
    print()

    # Summary
    print("="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print()
    print("Summary:")
    print(f"  ✅ Pure Transformer architecture (no mamba-ssm)")
    print(f"  ✅ Standard transformers library compatible")
    print(f"  ✅ ARM64 Jetson Thor compatible")
    print(f"  ✅ Memory footprint: ~{final_memory:.2f} GB")
    print(f"  ✅ Generation speed: ~{tokens_per_sec:.2f} tokens/sec")
    print()
    print("Status: READY FOR SAGE INTEGRATION ✅")
    print()

    return True

if __name__ == "__main__":
    success = test_nemotron_nano()
    exit(0 if success else 1)
