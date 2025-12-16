#!/usr/bin/env python3
"""
Test with greedy decoding (temperature=0) to eliminate sampling randomness
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compression'))

import torch
from transformers import AutoTokenizer
from selective_language_model import SelectiveLanguageModel

def main():
    print("="*80)
    print("GREEDY DECODING TEST (No Sampling Randomness)")
    print("="*80)

    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    # Initialize model
    print("\n🧠 Initializing model with 48 layers (full model)...")
    model = SelectiveLanguageModel(
        extraction_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted",
        num_layers=48,  # Full model!
        num_experts_per_tok=4,
        device="cpu"
    )

    # Test prompts
    prompts = [
        "The capital of France is",
        "1 + 1 = ",
        "The sky is",
        "Hello, my name is",
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: '{prompt}'")
        print(f"{'='*60}")

        # Encode
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate with GREEDY decoding (always pick top token)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.0,  # GREEDY: always pick highest probability token
        )

        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Output: '{output_text}'")

        # Show what tokens were generated
        new_tokens = output_ids[0][len(input_ids[0]):]
        print(f"New tokens: {new_tokens.tolist()}")
        print(f"Decoded individually: {[tokenizer.decode([t]) for t in new_tokens]}")

    print("\n" + "="*80)
    print("If output is still garbled with greedy decoding, the problem is NOT")
    print("sampling randomness, but something in the model architecture/weights.")
    print("="*80)

if __name__ == "__main__":
    main()
