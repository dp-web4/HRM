#!/usr/bin/env python3
"""
Simple Qwen3-Omni-30B Test

Quick test to have a conversation with the omni-modal model.
"""

import sys
from pathlib import Path
import torch
from transformers.models.qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

def test_qwen3_omni():
    print("=" * 70)
    print("Testing Qwen3-Omni-30B")
    print("=" * 70)
    print()

    model_path = Path("model-zoo/sage/omni-modal/qwen3-omni-30b")

    print("Loading model and processor (66GB, this will take a moment)...")
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.float16,  # Using 'dtype' not 'torch_dtype'
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ Model and processor loaded\n")

    # Test 1
    print("Test 1: Introduction")
    print("-" * 70)
    prompt1 = "Hello! I'm Claude, testing your capabilities. Please introduce yourself and explain what makes you an omni-modal model."

    inputs = processor(text=prompt1, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    response1 = processor.decode(outputs[0], skip_special_tokens=True).replace(prompt1, "").strip()

    print(f"Me: {prompt1}")
    print(f"\nQwen3-Omni: {response1}\n")

    # Test 2
    print("-" * 70)
    print("Test 2: Capabilities")
    print("-" * 70)
    prompt2 = "What types of inputs can you process - audio, video, images, text?"

    inputs = processor(text=prompt2, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    response2 = processor.decode(outputs[0], skip_special_tokens=True).replace(prompt2, "").strip()

    print(f"Me: {prompt2}")
    print(f"\nQwen3-Omni: {response2}\n")

    print("=" * 70)
    print("Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_qwen3_omni()
