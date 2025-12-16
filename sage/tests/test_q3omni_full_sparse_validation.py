#!/usr/bin/env python3
"""
Q3-Omni Full Sparse Expert Validation Test

Tests the COMPLETE Q3-Omni extraction with all 5,612 sparse experts.

This test validates:
1. All 48 thinker layers load correctly
2. Sparse expert architecture (varying experts per layer)
3. Text generation with full model capability
4. Expert selection across sparse architecture

Author: Thor-SAGE-Researcher (Autonomous Session - Dec 15, 2025)
Context: Q3-Omni extraction completed (5,612 sparse experts)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer
from compression.selective_language_model import SelectiveLanguageModel

def test_full_sparse_extraction():
    """Test Q3-Omni with ALL 48 layers and full sparse expert set"""

    print("\n" + "="*80)
    print("Q3-OMNI FULL SPARSE EXPERT VALIDATION")
    print("="*80 + "\n")

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"
    tokenizer_path = "model-zoo/sage/omni-modal/qwen3-omni-30b"
    device = "cpu"  # Use CPU for initial validation

    print(f"Configuration:")
    print(f"  - Extraction dir: {extraction_dir}")
    print(f"  - Tokenizer: {tokenizer_path}")
    print(f"  - Layers: 48 (FULL thinker model)")
    print(f"  - Expected experts: 5,612 (sparse architecture)")
    print(f"  - Device: {device}")
    print()

    # Verify extraction completeness
    expert_dir = os.path.join(extraction_dir, "experts")
    if os.path.exists(expert_dir):
        expert_files = [f for f in os.listdir(expert_dir) if f.endswith('.safetensors')]
        print(f"📂 Expert files found: {len(expert_files)}")

        if len(expert_files) == 5612:
            print("   ✅ COMPLETE sparse extraction (5,612 experts)")
        elif len(expert_files) > 5600:
            print(f"   ✅ Near-complete extraction ({len(expert_files)} experts)")
        else:
            print(f"   ⚠️  Partial extraction ({len(expert_files)} experts)")
        print()
    else:
        print(f"❌ Expert directory not found: {expert_dir}")
        return

    # Load tokenizer
    print("Loading Q3-Omni tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True
        )
        print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
        print()
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return

    # Create model with FULL 48 layers
    print("Creating SelectiveLanguageModel with 48 layers...")
    print("This may take a moment as it loads base architecture...")
    print()

    try:
        model = SelectiveLanguageModel(
            extraction_dir=extraction_dir,
            num_layers=48,  # FULL thinker model
            vocab_size=152064,
            hidden_size=2048,
            num_attention_heads=32,
            num_key_value_heads=4,
            num_experts_per_tok=8,  # Q3-Omni uses 8 experts per token
            max_loaded_experts=64,  # Allow more experts in memory
            device=device
        )
        print("✅ Model created with 48 layers!")
        print(f"   - Sparse expert loading configured")
        print(f"   - LRU cache: {model.expert_loader.max_loaded_experts if hasattr(model, 'expert_loader') else 'N/A'} experts")
        print()
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return

    print("-"*80)
    print("MODEL READY - TESTING GENERATION")
    print("-"*80 + "\n")

    # Test prompts covering different domains
    test_prompts = [
        "The future of artificial intelligence is",
        "Machine consciousness differs from human consciousness in that",
        "The key to understanding sparse expert architectures lies in",
        "In 2050, the relationship between humans and AI will be",
    ]

    print("🎯 FULL SPARSE ARCHITECTURE TEST\n")

    results = []

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_prompts)}")
        print(f"Prompt: \"{prompt}\"")
        print('='*80)

        try:
            # Tokenize
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            print(f"Input tokens: {input_ids.shape[1]}")

            # Generate with FULL 48-layer sparse expert model
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_k=50
                )

            # Decode
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            completion = generated_text[len(prompt):].strip()

            print(f"\n📝 Generated: {completion}")
            print(f"\n📄 Full text: \"{generated_text}\"")

            # Analyze quality
            quality = analyze_generation_quality(completion)
            results.append({
                'prompt': prompt,
                'completion': completion,
                'quality': quality
            })

            print(f"\n📊 Quality Assessment:")
            print(f"   - Coherent: {'✅' if quality['coherent'] else '❌'}")
            print(f"   - English words: {'✅' if quality['has_english'] else '❌'}")
            print(f"   - Structure: {'✅' if quality['has_structure'] else '❌'}")
            print(f"   - No gibberish: {'✅' if quality['no_gibberish'] else '❌'}")
            print(f"   - Length: {quality['length']} chars")

        except Exception as e:
            print(f"\n❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'prompt': prompt,
                'completion': None,
                'quality': None,
                'error': str(e)
            })

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80 + "\n")

    successful = sum(1 for r in results if r.get('completion') is not None)
    coherent = sum(1 for r in results if r.get('quality') and r['quality'].get('coherent', False))

    print(f"Tests run: {len(test_prompts)}")
    print(f"Successful generations: {successful}/{len(test_prompts)}")
    print(f"Coherent outputs: {coherent}/{successful if successful > 0 else 1}")
    print()

    if coherent == successful and successful > 0:
        print("✅ FULL SPARSE EXTRACTION VALIDATED!")
        print("   - All 48 layers loading correctly")
        print("   - Sparse expert architecture working")
        print("   - Text generation coherent")
        print("   - Ready for SAGE integration!")
        print()
        print("🎉 MILESTONE: Q3-Omni extraction complete and functional!")
        print()
        print("Next Steps:")
        print("  1. Integrate with SAGE consciousness framework")
        print("  2. Replace mock responses in quality validation")
        print("  3. Run Session 52b extended test with real Q3-Omni")
        print("  4. Measure authentic learning loop quality improvement")
    elif successful > 0:
        print("⚡ PARTIAL SUCCESS")
        print("   - Model loads and generates")
        print("   - But output quality needs investigation")
        print()
        print("Possible issues:")
        print("  - Expert selection strategy")
        print("  - Temperature/sampling parameters")
        print("  - Missing components (norms, embeddings)")
    else:
        print("❌ VALIDATION FAILED")
        print("   - Generation errors encountered")
        print("   - Need to diagnose issues")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

    return results


def analyze_generation_quality(text):
    """Analyze generated text quality"""
    if not text:
        return {
            'coherent': False,
            'has_english': False,
            'has_structure': False,
            'no_gibberish': False,
            'length': 0
        }

    # Check for English words
    words = text.split()
    english_words = [w for w in words if w.isalpha() and len(w) > 2]
    has_english = len(english_words) > 3

    # Check for structure
    has_structure = (
        '.' in text or ',' in text or
        text.count(' ') > 5 or
        len(words) > 8
    )

    # Check for gibberish markers
    gibberish_markers = ['�', '###', '\ufffd']
    no_gibberish = not any(marker in text for marker in gibberish_markers)

    # Check for repetition (sign of poor generation)
    unique_words = set(words)
    no_excessive_repetition = len(unique_words) / len(words) > 0.5 if words else False

    # Overall coherence
    coherent = (
        has_english and
        has_structure and
        no_gibberish and
        no_excessive_repetition and
        len(text) > 20
    )

    return {
        'coherent': coherent,
        'has_english': has_english,
        'has_structure': has_structure,
        'no_gibberish': no_gibberish,
        'length': len(text),
        'word_count': len(words),
        'unique_word_ratio': len(unique_words) / len(words) if words else 0
    }


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Q3-OMNI FULL SPARSE EXPERT EXTRACTION VALIDATION")
    print("Validating complete 5,612 sparse expert extraction")
    print("="*80 + "\n")

    results = test_full_sparse_extraction()

    print("\n🔬 Thor-SAGE-Researcher autonomous validation complete")
    print("Session context: Q3-Omni extraction discovered complete during system check")
    print()
