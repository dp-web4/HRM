#!/usr/bin/env python3
"""
Test mRoPE implementation - should give coherent generation!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compression'))

import torch
from transformers import AutoTokenizer
from selective_language_model import SelectiveLanguageModel

def main():
    print("="*80)
    print("🚀 TESTING mRoPE FIX - Should Generate Coherent Text!")
    print("="*80)

    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    # Initialize model with mRoPE
    print("\n🧠 Initializing model with mRoPE (48 layers)...")
    model = SelectiveLanguageModel(
        extraction_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted",
        num_layers=48,  # Full model
        num_experts_per_tok=8,  # Use Q3-Omni's default (was 4 before)
        device="cpu"
    )

    # Test prompts
    test_cases = [
        ("The capital of France is", 5),
        ("1 + 1 =", 3),
        ("Hello, my name is", 8),
        ("The sky is", 5),
        ("Q: What is 2+2? A:", 5),
    ]

    for prompt, max_tokens in test_cases:
        print(f"\n{'='*60}")
        print(f"📝 Prompt: '{prompt}'")
        print(f"{'='*60}")

        # Encode
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate with greedy decoding
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.0,  # Greedy (deterministic)
        )

        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        new_text = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)

        print(f"✅ Full output: '{output_text}'")
        print(f"   New tokens: '{new_text}'")

    print("\n" + "="*80)
    print("🎯 If output is coherent, mRoPE fix worked!")
    print("   Expected: 'Paris', '2', meaningful names, 'blue', '4'")
    print("="*80)

if __name__ == "__main__":
    main()
