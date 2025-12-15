#!/usr/bin/env python3
"""
Test generation with FULL expert pool and log which experts get selected.

This will reveal the expert specialization patterns!
"""

import torch
from transformers import AutoTokenizer
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from sage.compression.selective_language_model import SelectiveLanguageModel

def test_with_expert_logging():
    print("=" * 80)
    print("Q3-OMNI SELECTIVE EXPERT LOADING - EXPERT SELECTION ANALYSIS")
    print("=" * 80)
    print()

    # Load tokenizer
    print("Loading Q3-Omni tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )
    print(f"✅ Tokenizer loaded: {len(tokenizer)} tokens")
    print()

    # Load model
    print("Loading selective model with FULL expert pool...")
    model = SelectiveLanguageModel(
        extraction_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted",
        component="thinker",
        num_experts_per_tok=4,  # Load 4 experts per token
        device="cpu"
    )
    print("✅ Model loaded")
    print()

    # Test prompts representing different semantic domains
    test_prompts = [
        ("Technical", "The algorithm uses dynamic programming to"),
        ("Creative", "In the moonlight, the poet whispered"),
        ("Conversational", "Hello! How are you doing today? I wanted to"),
        ("Futuristic", "In the year 2150, artificial intelligence will"),
        ("Mathematical", "The derivative of f(x) = x² is"),
    ]

    print("=" * 80)
    print("TESTING DIFFERENT SEMANTIC DOMAINS")
    print("=" * 80)
    print()

    for domain, prompt in test_prompts:
        print(f"\n{'=' * 80}")
        print(f"Domain: {domain}")
        print(f"Prompt: \"{prompt}\"")
        print('=' * 80)

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate with expert logging
        print("\nGenerating (max 20 tokens)...")

        # Monkey-patch to log expert selections
        original_forward = model.forward
        expert_selections = []

        def logging_forward(input_ids, *args, **kwargs):
            # Call original forward
            result = original_forward(input_ids, *args, **kwargs)

            # Log which experts were selected (from cache)
            if hasattr(model.expert_loader, 'last_selected_experts'):
                expert_selections.append(model.expert_loader.last_selected_experts)

            return result

        model.forward = logging_forward

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                temperature=0.8,
                top_k=50
            )

        # Restore original forward
        model.forward = original_forward

        # Decode output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print(f"\n✅ Generated output:")
        print(f"   {output_text}")

        # Analyze expert selections if available
        if expert_selections:
            print(f"\n📊 Expert selections across layers:")
            all_experts = set()
            for layer_experts in expert_selections:
                if isinstance(layer_experts, (list, tuple)):
                    all_experts.update(layer_experts)

            if all_experts:
                expert_list = sorted(all_experts)
                print(f"   Unique experts used: {expert_list}")
                print(f"   Total unique experts: {len(expert_list)}/128")

        print()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Key observations:")
    print("1. Are outputs coherent now? (vs previous garbled output)")
    print("2. Do different domains use different experts?")
    print("3. What's the expert diversity (how many of 128 used)?")
    print()

if __name__ == "__main__":
    test_with_expert_logging()
