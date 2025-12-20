#!/usr/bin/env python3
"""
Explore the loaded Q3-Omni model structure to find text generation interface
"""

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, AutoTokenizer

print("="*80)
print("🔍 Exploring Q3-Omni Model Structure")
print("="*80)

print("\n📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "model-zoo/sage/omni-modal/qwen3-omni-30b",
    trust_remote_code=True
)

print("\n🧠 Loading model...")
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    "model-zoo/sage/omni-modal/qwen3-omni-30b",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
print(f"✅ Model loaded: {type(model).__name__}")

print("\n" + "="*80)
print("📊 Model Structure Analysis")
print("="*80)

# Explore top-level attributes
print("\n1️⃣ Top-level attributes:")
attrs = [a for a in dir(model) if not a.startswith('_')]
key_attrs = [a for a in attrs if any(k in a.lower() for k in ['thinker', 'talker', 'generate', 'forward', 'lm'])]
for attr in sorted(key_attrs):
    print(f"   • {attr}")

# Check if thinker exists
print("\n2️⃣ Checking for 'thinker' component:")
if hasattr(model, 'thinker'):
    print(f"   ✅ model.thinker exists: {type(model.thinker).__name__}")

    # Explore thinker attributes
    thinker_attrs = [a for a in dir(model.thinker) if not a.startswith('_')]
    key_thinker = [a for a in thinker_attrs if any(k in a.lower() for k in ['generate', 'forward', 'lm', 'head'])]
    print(f"   Thinker attributes:")
    for attr in sorted(key_thinker):
        print(f"      • thinker.{attr}")
else:
    print("   ❌ No direct 'thinker' attribute")

# Check model.model structure
print("\n3️⃣ Checking model.model structure:")
if hasattr(model, 'model'):
    print(f"   ✅ model.model exists: {type(model.model).__name__}")
    model_attrs = [a for a in dir(model.model) if not a.startswith('_')]
    key_model = [a for a in model_attrs if any(k in a.lower() for k in ['thinker', 'talker', 'layers'])]
    for attr in sorted(key_model)[:20]:
        print(f"      • model.{attr}")

# Check generation methods
print("\n4️⃣ Generation methods:")
if hasattr(model, 'generate'):
    print(f"   ✅ model.generate() exists")
if hasattr(model, 'forward'):
    print(f"   ✅ model.forward() exists")

# Try to understand forward signature
import inspect
if hasattr(model, 'forward'):
    sig = inspect.signature(model.forward)
    print(f"\n5️⃣ model.forward() signature:")
    print(f"   {sig}")

# Check config
print(f"\n6️⃣ Config info:")
if hasattr(model, 'config'):
    print(f"   Model type: {model.config.model_type}")
    if hasattr(model.config, 'thinker_config'):
        print(f"   Has thinker_config: ✅")
        print(f"   Thinker hidden size: {model.config.thinker_config.get('hidden_size', 'N/A')}")
        print(f"   Thinker vocab size: {model.config.thinker_config.get('vocab_size', 'N/A')}")

print("\n" + "="*80)
print("🎯 Testing Text Generation")
print("="*80)

prompt = "The capital of France is"
print(f"\n💬 Prompt: '{prompt}'")

try:
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    print(f"   Input IDs: {input_ids[0].tolist()}")
    print(f"   Input shape: {input_ids.shape}")

    print("\n   Attempting generation with model.generate()...")
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

    print(f"\n   ✅ SUCCESS!")
    print(f"   Generated token IDs: {new_tokens}")
    print(f"   Generated tokens: {new_text}")
    print(f"   Full output: '{output_text}'")

except Exception as e:
    print(f"\n   ❌ Generation failed: {e}")
    print(f"\n   Trying alternative: model.forward() directly...")

    try:
        with torch.no_grad():
            outputs = model.forward(input_ids=input_ids)

        print(f"   Forward output type: {type(outputs)}")
        if hasattr(outputs, 'logits'):
            print(f"   Logits shape: {outputs.logits.shape}")

            # Get next token prediction
            next_token_logits = outputs.logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            next_text = tokenizer.decode([next_token])

            print(f"   Next token ID: {next_token}")
            print(f"   Next token: '{next_text}'")
            print(f"   ✅ Forward pass works! We can implement our own generation loop.")

    except Exception as e2:
        print(f"   ❌ Forward also failed: {e2}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("✅ Exploration Complete")
print("="*80)
