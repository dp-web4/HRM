#!/usr/bin/env python3
"""Test with more layers to see if predictions improve"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compression'))

import torch
from transformers import AutoTokenizer
from selective_language_model import SelectiveLanguageModel

def test_with_layers(num_layers: int):
    print(f"\n{'='*80}")
    print(f"Testing with {num_layers} layers")
    print(f"{'='*80}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    # Initialize model
    model = SelectiveLanguageModel(
        extraction_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted",
        num_layers=num_layers,
        num_experts_per_tok=8,
        device="cpu"
    )

    # Test prompt
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"💬 Prompt: '{prompt}'")

    with torch.no_grad():
        logits = model.forward(input_ids, debug=False)
        last_token_logits = logits[0, -1, :]

        # Top 5 predictions
        top_k = torch.topk(last_token_logits, k=5)
        print(f"\n🎯 Top 5 predictions:")
        for i, (score, token_id) in enumerate(zip(top_k.values, top_k.indices)):
            token_text = tokenizer.decode([token_id.item()])
            print(f"   {i+1}. '{token_text}' (score: {score.item():.4f})")

def main():
    # Test with 3, 6, 10 layers
    for num_layers in [3, 6, 10]:
        test_with_layers(num_layers)

if __name__ == "__main__":
    main()
