#!/usr/bin/env python3
"""
Quick test to verify dtype fix for RoPE embeddings.

Tests trust-augmented expert selection with Q3-Omni weights to ensure
the "Buffer dtype mismatch" error is resolved.
"""

import sys
import torch
from pathlib import Path

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent))

from sage.compression.selective_language_model import SelectiveLanguageModel
from sage.core.trust_based_expert_selector import create_trust_based_selector
from transformers import AutoTokenizer


def test_dtype_fix():
    """Test that dtype fix resolves the buffer mismatch error"""
    print("\n" + "="*70)
    print("Testing Dtype Fix for Trust-Augmented Generation")
    print("="*70 + "\n")

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"

    print("1. Creating TrustBasedExpertSelector...")
    trust_selector = create_trust_based_selector(
        num_experts=128,
        cache_size=16,
        component="thinker"
    )
    print(f"   ✅ Created (num_experts={trust_selector.num_experts})\n")

    print("2. Loading SelectiveLanguageModel with trust_selector...")
    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=1,  # Single layer for speed
        num_experts_per_tok=8,
        max_loaded_experts=16,
        device="cpu",
        trust_selector=trust_selector  # Enable trust-based selection
    )
    print("   ✅ Model loaded\n")

    print("3. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "model-zoo/sage/omni-modal/qwen3-omni-30b",
        trust_remote_code=True
    )
    print("   ✅ Tokenizer loaded\n")

    # Test prompts
    prompts = [
        "def fibonacci(n):",
        "The key insight is",
        "In summary, the main argument"
    ]

    print("4. Testing trust-augmented generation on 3 prompts...\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"   Test {i}/3: '{prompt}'")

        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")

            with torch.no_grad():
                logits = model(input_ids)

            # Get top prediction
            next_token_logits = logits[0, -1, :]
            top_token_id = torch.argmax(next_token_logits).item()
            top_token = tokenizer.decode([top_token_id])

            print(f"   ✅ Success! Next token: '{top_token}'")
            print(f"      (logits shape: {logits.shape})\n")

        except RuntimeError as e:
            if "dtype mismatch" in str(e):
                print(f"   ❌ FAILED: Dtype mismatch error still present!")
                print(f"      Error: {e}\n")
                return False
            else:
                print(f"   ❌ FAILED: Unexpected runtime error")
                print(f"      Error: {e}\n")
                raise
        except Exception as e:
            print(f"   ❌ FAILED: Unexpected error")
            print(f"      Error: {e}\n")
            raise

    print("="*70)
    print("✅ ALL TESTS PASSING - DTYPE FIX SUCCESSFUL!")
    print("="*70)
    print("\nResult: Trust-augmented generation working with Q3-Omni weights")
    print("        No dtype mismatch errors detected")

    return True


if __name__ == "__main__":
    success = test_dtype_fix()
    sys.exit(0 if success else 1)
