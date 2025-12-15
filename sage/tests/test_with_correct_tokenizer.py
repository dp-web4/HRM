#!/usr/bin/env python3
"""
Test with Q3-Omni's ACTUAL Tokenizer - The Moment of Truth

This test uses Q3-Omni's real tokenizer instead of Qwen2.5's.
If the issue was vocabulary mismatch, this should produce COHERENT text!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer
from compression.selective_language_model import SelectiveLanguageModel

def test_with_real_tokenizer():
    """Test with Q3-Omni's actual tokenizer - THE FIX!"""

    print("\n" + "="*80)
    print("TESTING WITH Q3-OMNI'S ACTUAL TOKENIZER")
    print("="*80 + "\n")

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"
    tokenizer_path = "model-zoo/sage/omni-modal/qwen3-omni-30b"
    device = "cpu"

    print(f"Configuration:")
    print(f"  - Using Q3-Omni's tokenizer from: {tokenizer_path}")
    print(f"  - Extraction dir: {extraction_dir}")
    print(f"  - Layers: 24 (50% of full 48-layer model)")
    print(f"  - Device: {device}")
    print()

    # Load Q3-Omni's ACTUAL tokenizer
    print("Loading Q3-Omni's tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        print(f"✅ Q3-Omni tokenizer loaded (vocab size: {len(tokenizer)})")
        print()
    except Exception as e:
        print(f"❌ Failed to load Q3-Omni tokenizer: {e}")
        print("   Falling back to Qwen2.5...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            trust_remote_code=True
        )
        print(f"   Using fallback tokenizer (vocab size: {len(tokenizer)})")
        print()

    # Create model with all real weights
    print("Creating model with REAL weights (24 layers)...")
    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=24,
        vocab_size=152064,
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_experts_per_tok=4,
        max_loaded_experts=16,
        device=device
    )

    print("\n" + "-"*80)
    print("MODEL READY - TESTING GENERATION")
    print("-"*80 + "\n")

    # Test prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "In the year 2050, humanity will",
        "Machine learning enables us to",
        "The key to consciousness lies in"
    ]

    print("🎯 THE MOMENT OF TRUTH - Using Q3-Omni's Tokenizer\n")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_prompts)}: \"{prompt}\"")
        print('='*80)

        # Tokenize with Q3-Omni's tokenizer
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        print(f"Input tokens: {input_ids.shape[1]}")

        # Generate with real weights + real tokenizer
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=30,
                temperature=0.8,
                top_k=50
            )

        # Decode with Q3-Omni's tokenizer
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()

        print(f"\n📝 Generated: {completion}")
        print(f"\n📄 Full text: \"{generated_text}\"")

        # Analyze quality
        if len(completion) > 20:
            # Check for coherence markers
            has_english_words = any(word.isalpha() and len(word) > 3 for word in completion.split())
            has_structure = '.' in completion or ',' in completion or completion.count(' ') > 5
            no_gibberish = not any(char in completion for char in ['�', '###'])

            if has_english_words and has_structure and no_gibberish:
                print("\n✅ OUTPUT LOOKS COHERENT! 🎉")
                print("   - Contains English words")
                print("   - Has sentence structure")
                print("   - No obvious gibberish")
            elif has_english_words:
                print("\n⚡ OUTPUT PARTIALLY COHERENT")
                print("   - Contains some English words")
                print("   - But structure may be odd")
            else:
                print("\n⚠️  OUTPUT STILL GARBLED")
                print("   - Tokenizer might not be the issue")
                print("   - May need all 128 experts, not just 8")
        else:
            print("\n⚠️  OUTPUT TOO SHORT")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

    print("📊 ANALYSIS:")
    print()
    print("If output is NOW coherent:")
    print("  ✅ TOKENIZER WAS THE ISSUE!")
    print("  ✅ Architecture is CORRECT!")
    print("  ✅ Deep expert extraction WORKS!")
    print("  🎉 SELECTIVE LOADING VALIDATED!")
    print()
    print("If output is STILL garbled:")
    print("  🔍 Need to investigate further:")
    print("     - May need all 128 experts (not just 8)")
    print("     - Expert selection strategy might be wrong")
    print("     - Could be architectural mismatch")
    print()

if __name__ == "__main__":
    test_with_real_tokenizer()
