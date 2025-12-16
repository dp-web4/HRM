#!/usr/bin/env python3
"""Debug hidden states through the model"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'compression'))

import torch
from transformers import AutoTokenizer
from selective_language_model import SelectiveLanguageModel

def main():
    print("="*80)
    print("🔍 DEBUG: Hidden States & Logits")
    print("="*80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )

    # Initialize model with 3 layers
    model = SelectiveLanguageModel(
        extraction_dir="model-zoo/sage/omni-modal/qwen3-omni-30b-extracted",
        num_layers=3,
        num_experts_per_tok=8,
        device="cpu"
    )

    # Simple prompt
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"\n💬 Prompt: '{prompt}'")
    print(f"   Tokens: {input_ids.tolist()[0]}")

    # Run one forward pass with debug
    print(f"\n{'='*80}")
    print("FORWARD PASS DEBUG")
    print(f"{'='*80}")

    with torch.no_grad():
        # Get embeddings
        hidden_states = model.embed_tokens(input_ids)
        print(f"\n📊 After embeddings:")
        print(f"   Shape: {hidden_states.shape}")
        print(f"   Mean: {hidden_states.mean():.6f}, Std: {hidden_states.std():.6f}")
        print(f"   Min: {hidden_states.min():.6f}, Max: {hidden_states.max():.6f}")

        # Process through layers
        for layer_idx in range(3):
            hidden_states = model.layers[layer_idx](
                hidden_states,
                debug=False
            )
            print(f"\n📊 After layer {layer_idx}:")
            print(f"   Shape: {hidden_states.shape}")
            print(f"   Mean: {hidden_states.mean():.6f}, Std: {hidden_states.std():.6f}")
            print(f"   Min: {hidden_states.min():.6f}, Max: {hidden_states.max():.6f}")

            # Check for NaN or Inf
            if torch.isnan(hidden_states).any():
                print(f"   ⚠️  WARNING: NaN detected!")
            if torch.isinf(hidden_states).any():
                print(f"   ⚠️  WARNING: Inf detected!")

        # Final norm
        hidden_states = model.norm(hidden_states)
        print(f"\n📊 After final norm:")
        print(f"   Shape: {hidden_states.shape}")
        print(f"   Mean: {hidden_states.mean():.6f}, Std: {hidden_states.std():.6f}")
        print(f"   Min: {hidden_states.min():.6f}, Max: {hidden_states.max():.6f}")

        # LM head
        logits = model.lm_head(hidden_states)
        print(f"\n📊 After LM head (logits):")
        print(f"   Shape: {logits.shape}")
        print(f"   Mean: {logits.mean():.6f}, Std: {logits.std():.6f}")
        print(f"   Min: {logits.min():.6f}, Max: {logits.max():.6f}")

        # Check last token logits
        last_token_logits = logits[0, -1, :]
        print(f"\n📊 Last token logits (for next token prediction):")
        print(f"   Shape: {last_token_logits.shape}")
        print(f"   Mean: {last_token_logits.mean():.6f}, Std: {last_token_logits.std():.6f}")

        # Top 10 predictions
        top_k = torch.topk(last_token_logits, k=10)
        print(f"\n🎯 Top 10 predictions:")
        for i, (score, token_id) in enumerate(zip(top_k.values, top_k.indices)):
            token_text = tokenizer.decode([token_id.item()])
            print(f"   {i+1}. Token {token_id.item():6d} ('{token_text}'): {score.item():.4f}")

if __name__ == "__main__":
    main()
