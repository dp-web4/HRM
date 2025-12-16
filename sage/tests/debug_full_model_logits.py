#!/usr/bin/env python3
"""Debug logits from full 48-layer model"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compression'))

import torch
from transformers import AutoTokenizer
from selective_language_model import SelectiveLanguageModel

def main():
    print("="*80)
    print("🔍 DEBUG: Full Model Logits")
    print("="*80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    # Full model
    print("\n🧠 Loading full 48-layer model...")
    model = SelectiveLanguageModel(
        extraction_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted",
        num_layers=48,
        num_experts_per_tok=8,
        device="cpu"
    )

    # Test
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"\n💬 Prompt: '{prompt}'")

    with torch.no_grad():
        logits = model.forward(input_ids, debug=False)
        last_token_logits = logits[0, -1, :]

        # Check expected tokens
        expected_tokens = ["Paris", " Paris", "the", " the", "located", " located", "France", " France"]
        print(f"\n📊 Logits for EXPECTED tokens:")
        for token_text in expected_tokens:
            token_id = tokenizer.encode(token_text, add_special_tokens=False)[0]
            score = last_token_logits[token_id].item()
            # Find rank
            rank = (last_token_logits > score).sum().item() + 1
            print(f"   '{token_text}' (ID {token_id}): {score:.4f} (rank {rank})")

        # Top 10 predictions
        top_k = torch.topk(last_token_logits, k=10)
        print(f"\n🎯 Top 10 ACTUAL predictions:")
        for i, (score, token_id) in enumerate(zip(top_k.values, top_k.indices)):
            token_text = tokenizer.decode([token_id.item()])
            print(f"   {i+1}. '{token_text}' (ID {token_id.item()}): {score:.4f}")

        # Statistics
        print(f"\n📈 Logit statistics:")
        print(f"   Mean: {last_token_logits.mean():.4f}")
        print(f"   Std: {last_token_logits.std():.4f}")
        print(f"   Min: {last_token_logits.min():.4f}")
        print(f"   Max: {last_token_logits.max():.4f}")

if __name__ == "__main__":
    main()
