#!/usr/bin/env python3
"""Test with FULL 48-layer thinker model (no shortcuts!)"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compression'))

import torch
from transformers import AutoTokenizer
from selective_language_model import SelectiveLanguageModel

def main():
    print("="*80)
    print("🎯 FULL MODEL TEST - ALL 48 THINKER LAYERS")
    print("="*80)

    # Load tokenizer
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    # Initialize model with ALL 48 thinker layers
    print("\n🧠 Initializing FULL model (48 layers, 8 experts per token)...")
    model = SelectiveLanguageModel(
        extraction_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted",
        num_layers=48,  # FULL THINKER!
        num_experts_per_tok=8,
        device="cpu"
    )

    # Test prompts
    prompts = [
        "The capital of France is",
        "1 + 1 =",
        "Hello, my name is"
    ]

    for prompt in prompts:
        print(f"\n{'='*80}")
        print(f"💬 Prompt: '{prompt}'")
        print(f"{'='*80}")

        # Encode
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate 5 tokens
        print("🔄 Generating...")
        output_ids = model.generate(
            input_ids,
            max_new_tokens=5,
            temperature=0.0,  # Greedy
            debug=False,
        )

        # Show results
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        new_tokens = output_ids[0, len(input_ids[0]):].tolist()

        print(f"\n   Generated tokens: {[tokenizer.decode([t]) for t in new_tokens]}")
        print(f"   ✅ Full output: '{output_text}'")

if __name__ == "__main__":
    main()
