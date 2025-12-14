#!/usr/bin/env python3
"""
Qwen3-Omni-30B Test - Following Official Example

Testing with proper chat template and dtype="auto" as per official README.
"""

from pathlib import Path
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

def test_qwen3_omni_official():
    print("=" * 70)
    print("Testing Qwen3-Omni-30B (Official Approach)")
    print("=" * 70)
    print()

    MODEL_PATH = "model-zoo/sage/omni-modal/qwen3-omni-30b"

    print("Loading model with dtype='auto' and processor...")
    print("(This will take several minutes for 66GB model)")
    print()

    # Load model with dtype="auto" as per official example
    # Note: device_map="auto" fails with IndexError on tied parameters
    # Using device_map="cuda" instead for unified memory architecture
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype="auto",  # Let transformers choose the right dtype
        device_map="cuda",  # Simpler mapping for Jetson unified memory
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",  # Skip for now, may not be installed
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("✅ Model and processor loaded successfully!")
    print()

    # Test 1: Simple text-only conversation
    print("Test 1: Text-only conversation")
    print("-" * 70)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello! Please introduce yourself in one sentence."}
            ],
        },
    ]

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
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

    print("Generating response (text only, no audio output)...")

    # Generate (text only - no audio return)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        return_audio=False,  # Text only
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

    # Test 2: Another question
    print("-" * 70)
    print("Test 2: Capabilities question")
    print("-" * 70)

    conversation2 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What modalities can you process? List them briefly."}
            ],
        },
    ]

    text2 = processor.apply_chat_template(conversation2, add_generation_prompt=True, tokenize=False)
    audios2, images2, videos2 = process_mm_info(conversation2, use_audio_in_video=False)

    inputs2 = processor(
        text=text2,
        audio=audios2,
        images=images2,
        videos=videos2,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False
    )
    inputs2 = inputs2.to(model.device)

    generated_ids2 = model.generate(
        **inputs2,
        max_new_tokens=200,
        temperature=0.7,
        return_audio=False,
        use_audio_in_video=False
    )

    response2 = processor.batch_decode(
        generated_ids2[:, inputs2["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    print(f"\nPrompt: {conversation2[0]['content'][0]['text']}")
    print(f"Response: {response2[0]}")
    print()

    print("=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_qwen3_omni_official()
