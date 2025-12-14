#!/usr/bin/env python3
"""
Qwen3-Omni-30B INT8 AWQ Test (v2 - Auto-detect quantization)

Let transformers auto-detect and handle the AWQ quantization configuration.
"""

from pathlib import Path
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import torch
import gc
import psutil

def test_qwen3_omni_int8_v2():
    print("=" * 70)
    print("Testing Qwen3-Omni-30B INT8 AWQ (v2 - Auto-detect)")
    print("=" * 70)
    print()

    MODEL_PATH = "model-zoo/sage/omni-modal/qwen3-omni-30b-int8-awq"

    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Run download_qwen3_omni_int8.py first")
        return False

    print("Configuration:")
    print("  - INT8 AWQ quantization (auto-detected by transformers)")
    print("  - Expected: ~35GB model + ~25GB overhead = ~60GB total")
    print("  - Available: 122GB (should fit comfortably!)")
    print()

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    process = psutil.Process()
    mem_start = process.memory_info().rss / 1024**3

    print(f"Starting memory: {mem_start:.1f} GB")
    print()
    print("Loading INT8 AWQ model (auto-detect mode)...")
    print()

    try:
        # Load INT8 AWQ model - let transformers auto-detect quantization
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            device_map="cuda",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,  # Match the config
        )

        mem_after_load = process.memory_info().rss / 1024**3
        print(f"✅ Model loaded! Memory: {mem_after_load:.1f} GB (+{mem_after_load - mem_start:.1f} GB)")
        print()

        # Load processor
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        print("✅ Processor loaded!")
        print()

        # Test 1: Simple introduction
        print("=" * 70)
        print("Test 1: Text-only conversation")
        print("=" * 70)
        print()

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello! Introduce yourself in one brief sentence."}
                ],
            },
        ]

        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        inputs = inputs.to(model.device)

        mem_before_gen = process.memory_info().rss / 1024**3
        print(f"Memory before generation: {mem_before_gen:.1f} GB")
        print()
        print("Generating response...")

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            return_audio=False,
            use_audio_in_video=False
        )

        mem_after_gen = process.memory_info().rss / 1024**3
        print(f"Memory after generation: {mem_after_gen:.1f} GB")
        print()

        response = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(f"Prompt: {conversation[0]['content'][0]['text']}")
        print(f"Response: {response[0]}")
        print()

        # Final memory report
        mem_final = process.memory_info().rss / 1024**3
        print("=" * 70)
        print("Memory Summary:")
        print("=" * 70)
        print(f"  Start: {mem_start:.1f} GB")
        print(f"  After model load: {mem_after_load:.1f} GB (+{mem_after_load - mem_start:.1f} GB)")
        print(f"  After generation: {mem_final:.1f} GB")
        print(f"  Peak: {mem_final:.1f} GB")
        print()
        print("=" * 70)
        print("✅ INT8 AWQ Test (v2) Complete - SUCCESS!")
        print("=" * 70)

        return True

    except Exception as e:
        mem_error = process.memory_info().rss / 1024**3
        print(f"\n❌ Error at {mem_error:.1f} GB: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qwen3_omni_int8_v2()
    exit(0 if success else 1)
