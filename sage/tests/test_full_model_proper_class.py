#!/usr/bin/env python3
"""
Load full Q3-Omni model using proper model class with swap support

We have 272GB total (122GB RAM + 150GB swap), which should be enough
for ~150GB model requirement.
"""

import sys
import torch
import gc
import subprocess

print("="*80)
print("🔍 Loading Full Q3-Omni 30B Model (Proper Class)")
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

print("\n🧠 Loading full model with proper class...")
print("   Using trust_remote_code to get Qwen3OmniMoeThinkerForConditionalGeneration")

try:
    from transformers import AutoModel

    # Load with trust_remote_code to get custom architecture
    model = AutoModel.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print("✅ Model loaded successfully!")
    print(f"   Model class: {type(model).__name__}")

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
print("\n" + "="*80)
print("🎯 Testing Generation")
print("="*80)

test_prompts = [
    "The capital of France is",
    "2 + 2 =",
]

for prompt in test_prompts:
    print(f"\n💬 Prompt: '{prompt}'")

    try:
        # Encode
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate (5 tokens, greedy)
        print("   Generating...")
        with torch.no_grad():
            # Check if model has generate method
            if hasattr(model, 'generate'):
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                # If it's the thinker component, we might need to access it
                print(f"   Model type: {type(model)}")
                print(f"   Model attributes: {[a for a in dir(model) if not a.startswith('_')][:20]}")
                if hasattr(model, 'thinker'):
                    print("   Using model.thinker for generation")
                    output_ids = model.thinker.generate(
                        input_ids,
                        max_new_tokens=5,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                else:
                    print("   ⚠️  No generate method found")
                    continue

        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        new_tokens = output_ids[0, len(input_ids[0]):].tolist()

        print(f"   Generated tokens: {[tokenizer.decode([t]) for t in new_tokens]}")
        print(f"   ✅ Full output: '{output_text}'")

    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("✅ Test Complete")
print("="*80)
