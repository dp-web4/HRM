#!/usr/bin/env python3
"""
Qwen3-Omni-30B Text-only Test (Disable Talker)

Simplest possible test - text-only with talker disabled.
Based on official README examples.
"""

from pathlib import Path
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

def test_simple():
    MODEL_PATH = "model-zoo/sage/omni-modal/qwen3-omni-30b"

    print("Loading model (text-only, talker disabled)...")

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map="auto",  # Let transformers decide
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Disable talker for text-only
    model.disable_talker()

    print("✅ Model loaded with talker disabled!")

    processor = Qwen3OmniMoeProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    print("✅ Processor loaded!")

    # Simple text-only conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello! Please introduce yourself in one brief sentence."}
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

    print("Generating text response...")

    # Text only - no audio
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        return_audio=False,  # Explicit: no audio
        use_audio_in_video=False
    )

    response = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    print(f"\nPrompt: {conversation[0]['content'][0]['text']}")
    print(f"Response: {response[0]}")
    print("\n✅ SUCCESS!")

if __name__ == "__main__":
    test_simple()
