#!/usr/bin/env python3
"""
Attempt to load full Q3-Omni 30B model via HuggingFace.

This will be slow and memory-intensive but should work with swap.
Goal: Get baseline predictions to compare against our implementation.
"""

import sys
import torch
import gc

print("=" * 80)
print("🔍 Loading Full Q3-Omni 30B Model via HuggingFace")
print("=" * 80)

# Check memory before
import subprocess
result = subprocess.run(['free', '-h'], capture_output=True, text=True)
print("\n📊 Memory BEFORE loading:")
print(result.stdout)

print("\n📥 Loading tokenizer...")
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )
    print("✅ Tokenizer loaded")
except Exception as e:
    print(f"❌ Tokenizer failed: {e}")
    sys.exit(1)

print("\n🧠 Loading full model (this will take time and use swap)...")
print("   Expected: ~60GB for weights, may use swap heavily")

try:
    from transformers import AutoModelForCausalLM

    # Load with low memory settings
    model = AutoModelForCausalLM.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Half precision
        device_map="cpu",  # Force CPU (we don't have GPU memory)
        low_cpu_mem_usage=True,  # Use less memory during loading
    )
    print("✅ Model loaded successfully!")

    # Check memory after
    result = subprocess.run(['free', '-h'], capture_output=True, text=True)
    print("\n📊 Memory AFTER loading:")
    print(result.stdout)

except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test generation
print("\n" + "=" * 80)
print("🎯 Testing Generation")
print("=" * 80)

test_prompts = [
    "The capital of France is",
    "1 + 1 =",
    "Hello, my name is"
]

for prompt in test_prompts:
    print(f"\n💬 Prompt: '{prompt}'")

    try:
        # Encode
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate (5 tokens, greedy)
        print("   Generating...")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,  # Greedy (deterministic)
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        new_tokens = output_ids[0, len(input_ids[0]):].tolist()

        print(f"   Generated tokens: {[tokenizer.decode([t]) for t in new_tokens]}")
        print(f"   ✅ Full output: '{output_text}'")

    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("✅ Test Complete")
print("=" * 80)
