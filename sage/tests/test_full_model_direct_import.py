#!/usr/bin/env python3
"""
Load full Q3-Omni model by directly importing the model class

272GB total memory available (122GB RAM + 150GB swap)
"""

import sys
import torch
import subprocess

print("="*80)
print("🔍 Loading Full Q3-Omni 30B via Direct Import")
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

print("\n🧠 Attempting to import Qwen3OmniMoe model class...")
try:
    from transformers import Qwen3OmniMoeForConditionalGeneration
    print("✅ Found Qwen3OmniMoeForConditionalGeneration in transformers")
except ImportError:
    print("❌ Qwen3OmniMoeForConditionalGeneration not in transformers")
    print("   Trying alternative import methods...")

    try:
        # Try importing from auto with trust_remote_code
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(
            "model-zoo/sage/omni-modal/qwen3-omni-30b",
            trust_remote_code=True
        )
        print(f"✅ Config loaded: {type(config)}")

        # Now try to load model with this config
        print("\n🔄 Loading model with config and trust_remote_code...")
        model = AutoModel.from_pretrained(
            "model-zoo/sage/omni-modal/qwen3-omni-30b",
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        print(f"✅ Model loaded! Class: {type(model).__name__}")

    except Exception as e:
        print(f"❌ Alternative loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Check memory after loading
result = subprocess.run(['free', '-h'], capture_output=True, text=True)
print("\n📊 Memory AFTER loading:")
print(result.stdout)

# Try simple generation
print("\n" + "="*80)
print("🎯 Testing Generation")
print("="*80)

prompt = "The capital of France is"
print(f"\n💬 Prompt: '{prompt}'")

try:
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    print(f"   Input IDs: {input_ids[0].tolist()}")

    # Check model structure
    print(f"\n   Model type: {type(model).__name__}")
    print(f"   Has 'generate': {hasattr(model, 'generate')}")
    print(f"   Has 'thinker': {hasattr(model, 'thinker')}")

    # Try generation
    print("\n   Generating (5 tokens, greedy)...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    new_tokens = output_ids[0, len(input_ids[0]):].tolist()

    print(f"\n   Generated token IDs: {new_tokens}")
    print(f"   Generated tokens: {[tokenizer.decode([t]) for t in new_tokens]}")
    print(f"   ✅ Full output: '{output_text}'")

except Exception as e:
    print(f"   ❌ Generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("✅ Test Complete")
print("="*80)
