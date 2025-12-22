#!/usr/bin/env python3
"""
Test Q3-Omni with native HuggingFace Transformers using device_map for memory management.
"""

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
import time

MODEL_PATH = "/home/dp/ai-workspace/HRM/model-zoo/sage/omni-modal/qwen3-omni-30b"

print("=" * 60)
print("Q3-Omni Native HuggingFace Test")
print("=" * 60)

print(f"\nModel path: {MODEL_PATH}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Test prompts
TEST_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time",
]

print("\n" + "=" * 60)
print("Loading Model with device_map='auto'...")
print("=" * 60)

start_time = time.time()

try:
    # Load model with automatic device mapping for memory management
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",  # Automatic device placement
        max_memory={0: "110GB"},  # Conservative limit for unified memory
        torch_dtype=torch.float16,  # Use FP16 to save memory
        trust_remote_code=True,
        low_cpu_mem_usage=True,  # Reduce CPU memory during loading
    )

    load_time = time.time() - start_time
    print(f"\n✅ Model loaded successfully in {load_time:.2f} seconds!")

    # Load processor
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    print("✅ Processor loaded successfully!")

    # Show device map
    print("\nDevice map:")
    if hasattr(model, 'hf_device_map'):
        for name, device in sorted(model.hf_device_map.items())[:10]:
            print(f"  {name}: {device}")
        if len(model.hf_device_map) > 10:
            print(f"  ... ({len(model.hf_device_map)} total layers)")

    print("\n" + "=" * 60)
    print("Testing Text Generation")
    print("=" * 60)

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Prompt: \"{prompt}\"")

        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt")

        # Move inputs to GPU
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate - Q3-Omni returns (text_ids, audio) tuple
        gen_start = time.time()
        with torch.no_grad():
            text_ids, audio = model.generate(
                **inputs,
                max_new_tokens=50,
                thinker_return_dict_in_generate=True,  # Required for Q3-Omni
            )
        gen_time = time.time() - gen_start

        # Decode - slice off input portion to get only generated tokens
        input_len = inputs['input_ids'].shape[1]
        generated_tokens = text_ids.sequences[:, input_len:]
        generated_text = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print(f"  Output: \"{generated_text}\"")
        print(f"  Time: {gen_time:.2f}s")
        print(f"  Tokens: {generated_tokens.shape[1]} new tokens")
        if audio is not None:
            print(f"  Audio: {audio.shape} (generated audio output)")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
