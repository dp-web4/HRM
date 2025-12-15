#!/usr/bin/env python3
"""
Test Complete Architecture with REAL Weights

This is THE TEST - the culmination of:
1. Deep expert extraction (8 experts × 48 layers)
2. Attention weight extraction (ALL 48 layers)
3. Layer norm extraction (36/48 layers)
4. Complete loader integration

Expected: COHERENT text generation with real Q3-Omni weights!
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer
from compression.selective_language_model import SelectiveLanguageModel

def test_complete_architecture():
    """Test with REAL attention + expert weights"""

    print("\n" + "="*80)
    print("COMPLETE ARCHITECTURE TEST - REAL WEIGHTS")
    print("="*80 + "\n")

    # Configuration
    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"
    num_layers = 4  # Start with 4 layers for faster testing
    device = "cpu"  # Use CPU to avoid VRAM issues

    print(f"Configuration:")
    print(f"  - Extraction dir: {extraction_dir}")
    print(f"  - Layers: {num_layers}")
    print(f"  - Device: {device}")
    print(f"  - Expected: COHERENT text with real Q3-Omni weights!\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True
    )
    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})\n")

    # Create model with REAL WEIGHTS
    print("Creating model with REAL weights...")
    print("This will load:")
    print("  - Real embeddings from Q3-Omni")
    print("  - Real attention weights (Q, K, V, O projections)")
    print("  - Real layer norms (36/48 available)")
    print("  - Real MoE expert weights (8 deep experts)")
    print("  - Real LM head")
    print()

    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=num_layers,
        vocab_size=152064,
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_experts_per_tok=4,  # Use 4 experts per token
        max_loaded_experts=16,  # Keep 16 in memory
        device=device
    )

    print("\n" + "-"*80)
    print("MODEL INITIALIZATION COMPLETE")
    print("-"*80 + "\n")

    # Memory usage
    memory_stats = model.get_memory_usage()
    print(f"Memory Usage:")
    print(f"  - Embeddings: {memory_stats['embeddings_mb']:.1f} MB")
    print(f"  - Experts: {memory_stats['experts_mb']:.1f} MB")
    print(f"  - Routers: {memory_stats['routers_mb']:.1f} MB")
    print(f"  - LM head: {memory_stats['lm_head_mb']:.1f} MB")
    print(f"  - Total: {memory_stats['total_mb']:.1f} MB")
    print(f"  - Loaded experts: {memory_stats['num_loaded_experts']}")
    print()

    # Test prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "In the year 2050, humanity will",
        "Machine learning enables us to",
        "The key to consciousness lies in"
    ]

    print("="*80)
    print("GENERATION TESTS - REAL Q3-OMNI WEIGHTS")
    print("="*80 + "\n")

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}/{len(test_prompts)}")
        print(f"Prompt: \"{prompt}\"")
        print("-" * 60)

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        print(f"Input tokens: {input_ids.shape[1]}")

        # Generate with REAL weights
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                temperature=0.8,
                top_k=50
            )

        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()

        print(f"Generated: {completion}")
        print(f"Full text: \"{generated_text}\"")

        # Check coherence
        if len(completion) > 10 and not any(char in completion for char in ['�', '###']):
            print("✅ Output looks coherent!")
        else:
            print("⚠️  Output may be garbled")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

    print("🎯 KEY QUESTION: Is the text coherent?")
    print()
    print("If YES:")
    print("  ✅ Real attention weights FIXED the garbled generation!")
    print("  ✅ Deep expert architecture is WORKING!")
    print("  ✅ Q3-Omni selective loading is VALIDATED!")
    print()
    print("If NO (still garbled):")
    print("  🔍 Need to investigate further:")
    print("     - Check if all weights loaded correctly")
    print("     - Verify RoPE implementation")
    print("     - Check final layer norm")
    print()

if __name__ == "__main__":
    test_complete_architecture()
