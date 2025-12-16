#!/usr/bin/env python3
"""
Debug test to verify weights are being used in forward pass
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compression'))

import torch
from transformers import AutoTokenizer
from selective_language_model import SelectiveLanguageModel

def main():
    print("="*80)
    print("DEBUG: Verifying Weights Are Used in Forward Pass")
    print("="*80)

    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    # Initialize model (use 3 layers for fast debugging)
    print("\n🧠 Initializing model with 3 layers...")
    model = SelectiveLanguageModel(
        extraction_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted",
        num_layers=3,  # Just 3 layers for quick debugging
        num_experts_per_tok=4,
        device="cpu"
    )

    # Test prompt
    prompt = "The capital of France is"
    print(f"\n💬 Prompt: '{prompt}'")

    # Encode
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"   Tokens: {input_ids.tolist()}")

    # Generate with DEBUG=True
    print("\n" + "="*80)
    print("GENERATING WITH DEBUG ENABLED (First token only)")
    print("="*80)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=3,  # Just 3 tokens
        temperature=0.7,
        debug=True,  # 🔍 DEBUG MODE ON!
    )

    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n✅ Generated: '{output_text}'")

    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)
    print("\nKey things to verify from debug output:")
    print("1. q_proj weight mean ≠ 0 (weights loaded, not all zeros)")
    print("2. Expert weights have reasonable values (mean around ±0.01)")
    print("3. Outputs have reasonable statistics (not NaN, not extreme)")

if __name__ == "__main__":
    main()
