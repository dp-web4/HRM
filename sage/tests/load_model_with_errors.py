#!/usr/bin/env python3
"""
Load Q3-Omni model and work around initialization errors
"""

import torch
from transformers import AutoTokenizer

print("="*80)
print("🔍 Loading Q3-Omni with Error Handling")
print("="*80)

print("\n📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "model-zoo/sage/omni-modal/qwen3-omni-30b",
    trust_remote_code=True
)

print("\n🧠 Attempting to load model (catching initialization errors)...")

try:
    from transformers import Qwen3OmniMoeForConditionalGeneration

    # Try with ignore_mismatched_sizes
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,  # Ignore size mismatches
    )
    print(f"✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Loading with ignore_mismatched_sizes failed: {e}")
    print(f"\nTrying alternative: catch error and use model anyway...")

    # Try catching and continuing
    try:
        import warnings
        warnings.filterwarnings('ignore')

        # Temporarily monkey-patch to skip tied weight initialization
        from transformers import modeling_utils
        original_mark_tied = modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized

        def patched_mark_tied(self):
            try:
                return original_mark_tied(self)
            except AttributeError as e:
                print(f"   ⚠️  Skipping tied weight initialization (error: {e})")
                return None

        modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized = patched_mark_tied

        # Now try loading again
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            "model-zoo/sage/omni-modal/qwen3-omni-30b",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

        # Restore original
        modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized = original_mark_tied

        print(f"✅ Model loaded with patched initialization!")

    except Exception as e2:
        print(f"❌ Both methods failed: {e2}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

print(f"\n📊 Model loaded: {type(model).__name__}")

# Explore structure
print(f"\n🔍 Checking model.thinker...")
if hasattr(model, 'thinker'):
    print(f"   ✅ model.thinker exists: {type(model.thinker).__name__}")
    if hasattr(model.thinker, 'lm_head'):
        print(f"   ✅ model.thinker.lm_head exists")
        if hasattr(model.thinker.lm_head, 'weight'):
            print(f"      Shape: {model.thinker.lm_head.weight.shape}")
else:
    print(f"   ❌ No model.thinker")

# Try generation
print("\n" + "="*80)
print("🎯 Testing Text Generation")
print("="*80)

prompt = "The capital of France is"
print(f"\n💬 Prompt: '{prompt}'")

input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
print(f"   Input IDs: {input_ids[0].tolist()}")

try:
    print("\n   Attempting model.generate()...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    new_tokens = output_ids[0, len(input_ids[0]):].tolist()
    new_text = [tokenizer.decode([t]) for t in new_tokens]

    print(f"\n   ✅ GENERATION SUCCESS!")
    print(f"   Generated token IDs: {new_tokens}")
    print(f"   Generated tokens: {new_text}")
    print(f"   Full output: '{output_text}'")

    # Try another prompt
    print(f"\n💬 Prompt: '2 + 2 ='")
    input_ids2 = tokenizer.encode("2 + 2 =", return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        output_ids2 = model.generate(
            input_ids2,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    output_text2 = tokenizer.decode(output_ids2[0], skip_special_tokens=True)
    print(f"   ✅ Output: '{output_text2}'")

except Exception as e:
    print(f"\n   ❌ Generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("✅ Test Complete")
print("="*80)
