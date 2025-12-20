#!/usr/bin/env python3
"""
Load full Q3-Omni 30B model with swap support

We have 272GB total (122GB RAM + 150GB swap)
Model needs ~150GB (60GB weights + 90GB overhead)
"""

import sys
import torch
import subprocess
import time

print("="*80)
print("🔍 Loading Full Q3-Omni 30B Model with Swap Support")
print("="*80)

# Check memory before
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

print("\n🧠 Loading full model...")
print("   Using Qwen3OmniMoeForConditionalGeneration")
print("   This will take several minutes and use swap...")

start_time = time.time()

try:
    from transformers import Qwen3OmniMoeForConditionalGeneration

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    load_time = time.time() - start_time
    print(f"✅ Model loaded successfully in {load_time:.1f}s!")
    print(f"   Model class: {type(model).__name__}")

    # Check memory after
    result = subprocess.run(['free', '-h'], capture_output=True, text=True)
    print("\n📊 Memory AFTER loading:")
    print(result.stdout)

except Exception as e:
    print(f"❌ Model loading failed after {time.time()-start_time:.1f}s: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test generation
print("\n" + "="*80)
print("🎯 Testing Generation (Greedy, Deterministic)")
print("="*80)

test_prompts = [
    "The capital of France is",
    "2 + 2 =",
]

for prompt in test_prompts:
    print(f"\n💬 Prompt: '{prompt}'")

    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        print(f"   Input token IDs: {input_ids[0].tolist()}")

        print("   Generating 5 tokens (greedy)...")
        gen_start = time.time()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,  # Greedy (deterministic)
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_time = time.time() - gen_start
        print(f"   Generation took {gen_time:.1f}s")

        # Decode output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        new_tokens = output_ids[0, len(input_ids[0]):].tolist()
        new_text = [tokenizer.decode([t]) for t in new_tokens]

        print(f"   Generated token IDs: {new_tokens}")
        print(f"   Generated tokens: {new_text}")
        print(f"   ✅ Full output: '{output_text}'")

    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("✅ Test Complete - Full Model Baseline Established!")
print("="*80)
