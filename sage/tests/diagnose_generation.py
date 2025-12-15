#!/usr/bin/env python3
"""
Diagnostic Test - Why is generation garbled?

Checks:
1. Vocabulary alignment (tokenizer vs model)
2. Logits distribution and top predictions
3. Layer depth requirements
4. Expert selection patterns
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from compression.selective_language_model import SelectiveLanguageModel

def diagnose():
    print("\n" + "="*80)
    print("GENERATION DIAGNOSTIC")
    print("="*80 + "\n")

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"
    device = "cpu"

    # Load tokenizer
    print("1. VOCABULARY CHECK")
    print("-" * 60)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True
    )
    tokenizer_vocab_size = len(tokenizer)
    model_vocab_size = 152064

    print(f"Tokenizer vocab size: {tokenizer_vocab_size}")
    print(f"Model vocab size: {model_vocab_size}")
    print(f"Difference: {model_vocab_size - tokenizer_vocab_size}")

    if tokenizer_vocab_size != model_vocab_size:
        print(f"⚠️  VOCABULARY MISMATCH DETECTED!")
        print(f"   This could cause predictions outside tokenizer's range")
    else:
        print(f"✅ Vocabulary sizes match")
    print()

    # Create model with varying depths
    for num_layers in [1, 4, 8]:
        print(f"\n2. TESTING WITH {num_layers} LAYERS")
        print("-" * 60)

        model = SelectiveLanguageModel(
            extraction_dir=extraction_dir,
            num_layers=num_layers,
            vocab_size=model_vocab_size,
            hidden_size=2048,
            num_attention_heads=32,
            num_key_value_heads=4,
            num_experts_per_tok=4,
            max_loaded_experts=16,
            device=device
        )

        # Simple prompt
        prompt = "The future of AI is"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        print(f"Prompt: \"{prompt}\"")
        print(f"Input shape: {input_ids.shape}")

        # Forward pass
        with torch.no_grad():
            logits = model.forward(input_ids)

        print(f"Logits shape: {logits.shape}")

        # Check last token logits
        last_logits = logits[0, -1, :]  # [vocab_size]

        # Statistics
        print(f"Logits stats:")
        print(f"  Min: {last_logits.min().item():.2f}")
        print(f"  Max: {last_logits.max().item():.2f}")
        print(f"  Mean: {last_logits.mean().item():.2f}")
        print(f"  Std: {last_logits.std().item():.2f}")

        # Top-k predictions
        probs = F.softmax(last_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k=10)

        print(f"\nTop 10 predictions:")
        for i, (prob, idx) in enumerate(zip(top_k_probs, top_k_indices), 1):
            try:
                token = tokenizer.decode([idx])
                print(f"  {i}. Token {idx:6d} (p={prob:.4f}): '{token}'")
            except:
                print(f"  {i}. Token {idx:6d} (p={prob:.4f}): [OUT OF VOCAB RANGE]")

        # Generate one token
        next_token = torch.multinomial(probs, num_samples=1)
        try:
            generated_text = tokenizer.decode([next_token])
            print(f"\nSampled token: '{generated_text}'")
        except:
            print(f"\nSampled token {next_token.item()}: [OUT OF VOCAB RANGE]")

    print("\n" + "="*80)
    print("DIAGNOSIS COMPLETE")
    print("="*80)
    print("\nPOSSIBLE ISSUES:")
    print("1. Vocabulary mismatch causes out-of-range predictions")
    print("2. Need correct Q3-Omni tokenizer, not Qwen2.5")
    print("3. May need more than 4 layers for coherent generation")
    print("4. Expert weights might not match the attention/embeddings")
    print()

if __name__ == "__main__":
    diagnose()
