#!/usr/bin/env python3
"""
Test Q3-Omni using the OFFICIAL inference pattern from HuggingFace docs

Key differences from our attempts:
1. Use Qwen3OmniMoeProcessor (not just AutoTokenizer)
2. Use processor.apply_chat_template() to format conversation
3. model.generate() returns (text_ids, audio) tuple
4. Set return_audio=False to skip audio generation
"""

import sys
import torch
import subprocess

print("="*80)
print("🔍 Q3-Omni Official Inference Pattern Test")
print("="*80)

# Check memory before
result = subprocess.run(['free', '-h'], capture_output=True, text=True)
print("\n📊 Memory BEFORE loading:")
print(result.stdout)

print("\n📥 Loading model and processor...")
print("   Using OFFICIAL pattern from HuggingFace docs")

try:
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    MODEL_PATH = "model-zoo/sage/omni-modal/qwen3-omni-30b"

    print(f"\n   Loading model from: {MODEL_PATH}")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,  # Official uses dtype="auto", we use bfloat16
        device_map="cpu",             # We don't have GPU, use CPU
        low_cpu_mem_usage=True,       # Reduce memory usage
        # attn_implementation="flash_attention_2",  # Skip for CPU
    )

    print(f"   Loading processor...")
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    print("✅ Model and processor loaded!")

    # Check memory after
    result = subprocess.run(['free', '-h'], capture_output=True, text=True)
    print("\n📊 Memory AFTER loading:")
    print(result.stdout)

    # Disable talker to save memory (saves ~10GB)
    print("\n💾 Disabling talker to save memory...")
    model.disable_talker()
    print("✅ Talker disabled (saves ~10GB VRAM)")

except Exception as e:
    print(f"❌ Loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test generation with official pattern
print("\n" + "="*80)
print("🎯 Testing Text Generation (Official Pattern)")
print("="*80)

test_prompts = [
    "The capital of France is",
    "2 + 2 =",
]

for prompt in test_prompts:
    print(f"\n💬 Prompt: '{prompt}'")

    try:
        # Format as conversation (official pattern)
        conversation = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        print("   Applying chat template...")
        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        print(f"   Template result: '{text}'")

        print("   Processing input...")
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)
        print(f"   Input IDs shape: {inputs['input_ids'].shape}")
        print(f"   Input IDs: {inputs['input_ids'][0].tolist()}")

        print("   Generating (5 tokens, greedy, NO AUDIO)...")
        with torch.no_grad():
            # CRITICAL: model.generate() returns (text_ids, audio) tuple!
            # Set return_audio=False to skip audio generation
            text_ids, audio = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                return_audio=False,  # Skip audio output
            )

        # Decode only the NEW tokens (skip prompt)
        result = processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(f"\n   ✅ GENERATION SUCCESS!")
        print(f"   Generated text: '{result[0]}'")

        # Also show full output
        full_output = processor.batch_decode(
            text_ids.sequences,
            skip_special_tokens=True
        )
        print(f"   Full output: '{full_output[0]}'")

    except Exception as e:
        print(f"   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("✅ Official Pattern Test Complete")
print("="*80)
