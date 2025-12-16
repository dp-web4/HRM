#!/usr/bin/env python3
"""
Debug per-token routing to see what's happening
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compression'))

import torch
from transformers import AutoTokenizer
from selective_language_model import SelectiveLanguageModel

def main():
    print("="*80)
    print("🔍 DEBUG: Per-Token Routing")
    print("="*80)

    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    # Initialize model with just 3 layers for debugging
    print("\n🧠 Initializing model with 3 layers for debugging...")
    model = SelectiveLanguageModel(
        extraction_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted",
        num_layers=3,  # Just 3 for fast debugging
        num_experts_per_tok=8,
        device="cpu"
    )

    # Simple prompt
    prompt = "The capital of France is"
    print(f"\n💬 Prompt: '{prompt}'")

    # Encode
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"   Tokens: {input_ids.tolist()[0]}")
    print(f"   Decoded tokens: {[tokenizer.decode([t]) for t in input_ids[0]]}")

    # Generate ONE token with debug
    print("\n" + "="*80)
    print("GENERATING FIRST TOKEN WITH DEBUG")
    print("="*80)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=1,  # Just 1 token
        temperature=0.0,  # Greedy
        debug=True,  # DEBUG ON!
    )

    # Show result
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    new_token_id = output_ids[0, -1].item()
    new_token = tokenizer.decode([new_token_id])

    print(f"\n✅ Generated token ID: {new_token_id}")
    print(f"   Decoded: '{new_token}'")
    print(f"   Full output: '{output_text}'")

if __name__ == "__main__":
    main()
