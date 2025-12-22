#!/usr/bin/env python3
"""
Q3-Omni Implementation Comparison Framework

Compares two Q3-Omni implementations on Jetson AGX Thor:
1. Native HuggingFace (baseline): Full 30B model, 64GB memory
2. SAGE Selective Loading: 5,612 sparse experts, ~93% memory reduction claim

Test Protocol:
- Identical prompts across both systems
- Memory footprint measurement
- Generation speed (tokens/sec)
- Output quality analysis
- Coherence and creativity assessment

Author: Claude (Autonomous Session - Dec 21, 2025)
Context: Q3-Omni baseline validation successful, now comparing approaches
"""

import sys
import os
import time
import json
import torch
from datetime import datetime
from pathlib import Path

# Add sage to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Test prompts covering different domains
STANDARD_TEST_PROMPTS = [
    # Factual (short)
    "The capital of France is",

    # Mathematical (short)
    "2 + 2 =",

    # Creative (long)
    "Once upon a time",

    # Technical reasoning
    "The key difference between sparse and dense neural networks is",

    # Future speculation
    "In 2050, artificial intelligence will",

    # Abstract concepts
    "Consciousness can be understood as",
]


def measure_memory():
    """Get current memory usage"""
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        'rss_mb': mem_info.rss / (1024 * 1024),  # Resident Set Size
        'vms_mb': mem_info.vms / (1024 * 1024),  # Virtual Memory Size
    }


def analyze_text_quality(text):
    """Analyze generated text for quality metrics"""
    if not text:
        return {'coherent': False, 'word_count': 0, 'char_count': 0}

    words = text.split()
    sentences = text.count('.') + text.count('?') + text.count('!')

    # English word detection
    english_words = [w for w in words if w.isalpha() and len(w) > 2]
    has_english = len(english_words) > 3

    # Structure detection
    has_structure = sentences > 0 or len(words) > 8

    # Gibberish detection
    gibberish_markers = ['�', '###', '\ufffd', 'unk']
    no_gibberish = not any(marker in text.lower() for marker in gibberish_markers)

    # Repetition check
    unique_words = set(words)
    repetition_ratio = len(unique_words) / len(words) if words else 0

    # Overall coherence
    coherent = (
        has_english and
        has_structure and
        no_gibberish and
        repetition_ratio > 0.5 and
        len(text) > 20
    )

    return {
        'coherent': coherent,
        'word_count': len(words),
        'char_count': len(text),
        'sentence_count': sentences,
        'unique_word_ratio': repetition_ratio,
        'has_english': has_english,
        'no_gibberish': no_gibberish,
    }


def test_native_baseline(prompts, max_new_tokens=50):
    """Test native HuggingFace Q3-Omni implementation"""
    print("\n" + "="*80)
    print("TESTING: Native HuggingFace Q3-Omni (Baseline)")
    print("="*80 + "\n")

    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    model_path = "model-zoo/sage/omni-modal/qwen3-omni-30b"

    print("Loading native model...")
    load_start = time.time()
    mem_before = measure_memory()

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        max_memory={0: "110GB"},
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    load_time = time.time() - load_start
    mem_after = measure_memory()
    mem_delta = mem_after['rss_mb'] - mem_before['rss_mb']

    print(f"✅ Model loaded in {load_time:.2f}s")
    print(f"   Memory delta: {mem_delta:.0f} MB")
    print(f"   Current RSS: {mem_after['rss_mb']:.0f} MB")
    print()

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Testing: \"{prompt[:50]}...\"")

        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        gen_start = time.time()
        mem_gen_before = measure_memory()

        with torch.no_grad():
            text_ids, audio = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                thinker_return_dict_in_generate=True,
            )

        gen_time = time.time() - gen_start
        mem_gen_after = measure_memory()

        # Decode
        input_len = inputs['input_ids'].shape[1]
        generated_tokens = text_ids.sequences[:, input_len:]
        generated_text = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        tokens_generated = generated_tokens.shape[1]
        tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0

        quality = analyze_text_quality(generated_text)

        result = {
            'prompt': prompt,
            'generated_text': generated_text,
            'tokens_generated': tokens_generated,
            'generation_time_s': gen_time,
            'tokens_per_sec': tokens_per_sec,
            'memory_during_gen_mb': mem_gen_after['rss_mb'],
            'quality': quality,
        }

        results.append(result)

        print(f"   Tokens: {tokens_generated}, Speed: {tokens_per_sec:.2f} tok/s")
        print(f"   Quality: {'✅ Coherent' if quality['coherent'] else '⚠️ Issues'}")
        print()

    return {
        'implementation': 'native_huggingface',
        'load_time_s': load_time,
        'memory_baseline_mb': mem_after['rss_mb'],
        'results': results,
    }


def test_sage_selective(prompts, max_new_tokens=50):
    """Test SAGE selective loading implementation"""
    print("\n" + "="*80)
    print("TESTING: SAGE Selective Expert Loading")
    print("="*80 + "\n")

    from transformers import AutoTokenizer
    from compression.selective_language_model import SelectiveLanguageModel

    extraction_dir = "model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"
    tokenizer_path = "model-zoo/sage/omni-modal/qwen3-omni-30b"
    device = "cpu"  # Start with CPU

    print("Loading SAGE selective model...")
    load_start = time.time()
    mem_before = measure_memory()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=48,  # Full 48 layers
        vocab_size=152064,
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_experts_per_tok=8,
        max_loaded_experts=64,  # LRU cache size
        device=device
    )

    load_time = time.time() - load_start
    mem_after = measure_memory()
    mem_delta = mem_after['rss_mb'] - mem_before['rss_mb']

    print(f"✅ Model loaded in {load_time:.2f}s")
    print(f"   Memory delta: {mem_delta:.0f} MB")
    print(f"   Current RSS: {mem_after['rss_mb']:.0f} MB")
    print(f"   Expert cache: {64} experts max")
    print()

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] Testing: \"{prompt[:50]}...\"")

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate
        gen_start = time.time()
        mem_gen_before = measure_memory()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=50
            )

        gen_time = time.time() - gen_start
        mem_gen_after = measure_memory()

        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()

        tokens_generated = output_ids.shape[1] - input_ids.shape[1]
        tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0

        quality = analyze_text_quality(completion)

        result = {
            'prompt': prompt,
            'generated_text': completion,
            'tokens_generated': tokens_generated,
            'generation_time_s': gen_time,
            'tokens_per_sec': tokens_per_sec,
            'memory_during_gen_mb': mem_gen_after['rss_mb'],
            'quality': quality,
        }

        results.append(result)

        print(f"   Tokens: {tokens_generated}, Speed: {tokens_per_sec:.2f} tok/s")
        print(f"   Quality: {'✅ Coherent' if quality['coherent'] else '⚠️ Issues'}")
        print()

    return {
        'implementation': 'sage_selective',
        'load_time_s': load_time,
        'memory_baseline_mb': mem_after['rss_mb'],
        'results': results,
    }


def compare_results(native_data, sage_data):
    """Generate comprehensive comparison report"""
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS")
    print("="*80 + "\n")

    # Memory comparison
    print("📊 MEMORY FOOTPRINT")
    print("-" * 80)
    native_mem = native_data['memory_baseline_mb']
    sage_mem = sage_data['memory_baseline_mb']
    mem_reduction = ((native_mem - sage_mem) / native_mem * 100) if native_mem > 0 else 0

    print(f"Native:     {native_mem:,.0f} MB")
    print(f"SAGE:       {sage_mem:,.0f} MB")
    print(f"Reduction:  {mem_reduction:.1f}%")
    print(f"Savings:    {native_mem - sage_mem:,.0f} MB")
    print()

    # Speed comparison
    print("⚡ GENERATION SPEED")
    print("-" * 80)
    native_speeds = [r['tokens_per_sec'] for r in native_data['results']]
    sage_speeds = [r['tokens_per_sec'] for r in sage_data['results']]

    native_avg = sum(native_speeds) / len(native_speeds) if native_speeds else 0
    sage_avg = sum(sage_speeds) / len(sage_speeds) if sage_speeds else 0
    speed_ratio = sage_avg / native_avg if native_avg > 0 else 0

    print(f"Native avg: {native_avg:.2f} tokens/sec")
    print(f"SAGE avg:   {sage_avg:.2f} tokens/sec")
    print(f"Ratio:      {speed_ratio:.2f}x {'faster' if speed_ratio > 1 else 'slower'}")
    print()

    # Quality comparison
    print("✨ OUTPUT QUALITY")
    print("-" * 80)
    native_coherent = sum(1 for r in native_data['results'] if r['quality']['coherent'])
    sage_coherent = sum(1 for r in sage_data['results'] if r['quality']['coherent'])
    total_tests = len(native_data['results'])

    print(f"Native:     {native_coherent}/{total_tests} coherent ({native_coherent/total_tests*100:.0f}%)")
    print(f"SAGE:       {sage_coherent}/{total_tests} coherent ({sage_coherent/total_tests*100:.0f}%)")
    print()

    # Load time comparison
    print("🚀 LOAD TIME")
    print("-" * 80)
    print(f"Native:     {native_data['load_time_s']:.2f}s")
    print(f"SAGE:       {sage_data['load_time_s']:.2f}s")
    print()

    # Side-by-side sample outputs
    print("📝 SAMPLE OUTPUT COMPARISON (First Prompt)")
    print("-" * 80)
    native_first = native_data['results'][0]
    sage_first = sage_data['results'][0]

    print(f"Prompt: \"{native_first['prompt']}\"")
    print()
    print(f"Native: \"{native_first['generated_text'][:200]}...\"")
    print()
    print(f"SAGE:   \"{sage_first['generated_text'][:200]}...\"")
    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()

    if mem_reduction > 50 and sage_coherent >= native_coherent * 0.8:
        print("✅ SAGE SELECTIVE LOADING: VIABLE ALTERNATIVE")
        print(f"   - Significant memory reduction: {mem_reduction:.1f}%")
        print(f"   - Quality maintained: {sage_coherent/total_tests*100:.0f}% coherent")
        if speed_ratio > 0.5:
            print(f"   - Acceptable speed: {speed_ratio:.2f}x vs native")
        else:
            print(f"   - Speed trade-off: {speed_ratio:.2f}x vs native")
    elif sage_coherent >= native_coherent:
        print("⚡ SAGE: QUALITY MAINTAINED BUT MEMORY BENEFITS UNCLEAR")
        print(f"   - Memory reduction: {mem_reduction:.1f}%")
        print(f"   - Quality: {sage_coherent/total_tests*100:.0f}% coherent")
    else:
        print("⚠️  QUALITY DEGRADATION DETECTED")
        print(f"   - Native: {native_coherent/total_tests*100:.0f}% coherent")
        print(f"   - SAGE: {sage_coherent/total_tests*100:.0f}% coherent")
        print(f"   - Memory saved: {mem_reduction:.1f}%")

    print()

    return {
        'memory_reduction_pct': mem_reduction,
        'speed_ratio': speed_ratio,
        'native_coherence_pct': native_coherent/total_tests*100,
        'sage_coherence_pct': sage_coherent/total_tests*100,
    }


def main():
    """Run complete comparison framework"""
    print("\n" + "="*80)
    print("Q3-OMNI IMPLEMENTATION COMPARISON")
    print("Jetson AGX Thor - December 21, 2025")
    print("="*80 + "\n")

    print(f"Test protocol:")
    print(f"  - Prompts: {len(STANDARD_TEST_PROMPTS)}")
    print(f"  - Max tokens: 50 per generation")
    print(f"  - Metrics: Memory, Speed, Quality")
    print()

    # Create output directory
    output_dir = Path("comparison_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Test native baseline
        native_results = test_native_baseline(STANDARD_TEST_PROMPTS, max_new_tokens=50)

        # Save intermediate results
        with open(output_dir / f"native_results_{timestamp}.json", 'w') as f:
            json.dump(native_results, f, indent=2)

        print("\n⏸️  Clearing memory before SAGE test...")
        del native_results
        import gc
        gc.collect()
        time.sleep(5)

        # Test SAGE selective
        sage_results = test_sage_selective(STANDARD_TEST_PROMPTS, max_new_tokens=50)

        # Save intermediate results
        with open(output_dir / f"sage_results_{timestamp}.json", 'w') as f:
            json.dump(sage_results, f, indent=2)

        # Compare
        comparison = compare_results(native_results, sage_results)

        # Save complete comparison
        full_report = {
            'timestamp': timestamp,
            'platform': 'Jetson AGX Thor',
            'native': native_results,
            'sage': sage_results,
            'comparison': comparison,
        }

        with open(output_dir / f"comparison_report_{timestamp}.json", 'w') as f:
            json.dump(full_report, f, indent=2)

        print(f"\n✅ Results saved to {output_dir}/")
        print()

    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
