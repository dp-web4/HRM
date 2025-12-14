#!/usr/bin/env python3
"""
Qwen3-Omni-30B Memory-Optimized Test

Using HuggingFace memory optimization flags to reduce RAM usage during loading.
"""

from pathlib import Path
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import torch
import gc

def test_qwen3_omni_optimized():
    print("=" * 70)
    print("Testing Qwen3-Omni-30B (Memory-Optimized)")
    print("=" * 70)
    print()

    MODEL_PATH = "model-zoo/sage/omni-modal/qwen3-omni-30b"

    print("Memory optimization strategy:")
    print("  - low_cpu_mem_usage=True (reduces peak RAM during load)")
    print("  - device_map='cuda' (simple mapping for unified memory)")
    print("  - dtype='auto' (let transformers choose optimal dtype)")
    print("  - Aggressive garbage collection")
    print()

    # Clear any existing tensors
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("Loading model with memory optimizations...")
    print("(This should use significantly less RAM)")
    print()

    try:
        # KEY OPTIMIZATION: low_cpu_mem_usage=True
        # This loads weights directly to device without creating CPU copy first
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype="auto",
            device_map="cuda",
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # ← CRITICAL for memory reduction
            # torch_dtype=torch.float16,  # Could also force fp16 if needed
        )

        print("✅ Model loaded successfully!")
        print()

        # Load processor (lightweight)
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )

        print("✅ Processor loaded!")
        print()

        # Check actual memory usage
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"Process memory: {mem_info.rss / 1024**3:.1f} GB")
        print()

        # Test 1: Simple text conversation
        print("Test 1: Text-only conversation")
        print("-" * 70)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello! Introduce yourself in one sentence."}
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

        print("Generating response...")

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            return_audio=False,
            use_audio_in_video=False
        )

        response = processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(f"\nPrompt: {conversation[0]['content'][0]['text']}")
        print(f"Response: {response[0]}")
        print()

        print("=" * 70)
        print("✅ Memory-Optimized Test Complete!")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qwen3_omni_optimized()
    exit(0 if success else 1)
