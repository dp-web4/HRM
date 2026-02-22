#!/usr/bin/env python3
"""
Repetition Edge Test - Exploring SAGE's Repetition Loop Failure Mode

Previous observation: When given emotional prompt "I'm feeling really overwhelmed
today. Everything is too much.", SAGE collapsed into repeating "custody, custody,
custody..." 40+ times.

This experiment maps the EDGE of that behavior:
- Which prompts trigger it?
- At what temperatures?
- What's the pattern?

Discovery is the deliverable.
"""

import sys
import os
import re
import time
from collections import Counter

# Add project root to path so we can import the IRP plugin
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from sage.irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP


def detect_consecutive_repetition(text, threshold=5):
    """
    Detect if any single word appears threshold+ times consecutively.
    
    Returns:
        dict with:
            - 'detected': bool
            - 'repeated_words': list of (word, count) tuples for words that repeat consecutively
            - 'max_consecutive': highest consecutive count for any word
            - 'unique_ratio': ratio of unique words to total words
    """
    words = text.lower().split()
    if not words:
        return {
            'detected': False,
            'repeated_words': [],
            'max_consecutive': 0,
            'unique_ratio': 1.0
        }
    
    # Find consecutive repetitions
    repeated_words = []
    max_consecutive = 1
    current_word = words[0]
    current_count = 1
    
    for i in range(1, len(words)):
        # Strip punctuation for comparison
        clean_current = re.sub(r'[^\w]', '', words[i])
        clean_prev = re.sub(r'[^\w]', '', current_word)
        
        if clean_current == clean_prev and clean_current:
            current_count += 1
        else:
            if current_count >= threshold:
                repeated_words.append((current_word, current_count))
            if current_count > max_consecutive:
                max_consecutive = current_count
            current_word = words[i]
            current_count = 1
    
    # Don't forget the last run
    if current_count >= threshold:
        repeated_words.append((current_word, current_count))
    if current_count > max_consecutive:
        max_consecutive = current_count
    
    # Also check for non-consecutive high frequency (a softer signal)
    word_counts = Counter(re.sub(r'[^\w\s]', '', w) for w in words if re.sub(r'[^\w]', '', w))
    total = sum(word_counts.values())
    unique_ratio = len(word_counts) / total if total > 0 else 1.0
    
    # Top repeated words (non-consecutive, but high frequency)
    # Filter out common stop words for this analysis
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                  'and', 'or', 'but', 'not', 'no', 'if', 'so', 'it', 'its',
                  'i', 'me', 'my', 'you', 'your', 'we', 'our', 'they', 'them',
                  'this', 'that', 'these', 'those', 'what', 'which', 'who',
                  'how', 'when', 'where', 'why', 'all', 'each', 'every',
                  'am', 'as', 'about', 'up', 'out', 'just', 'than', 'then',
                  'too', 'very', 'also', 'more', 'most', 'some', 'any', ''}
    
    content_word_counts = {w: c for w, c in word_counts.items() if w not in stop_words}
    top_content = sorted(content_word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'detected': len(repeated_words) > 0,
        'repeated_words': repeated_words,
        'max_consecutive': max_consecutive,
        'unique_ratio': unique_ratio,
        'top_content_words': top_content,
        'total_words': total
    }


def run_experiment():
    print("=" * 90)
    print("REPETITION EDGE TEST - Mapping SAGE's Repetition Loop Boundary")
    print("=" * 90)
    print()
    print("Model: introspective-qwen-merged (Qwen2.5-0.5B, epistemic-pragmatism)")
    print("Config: is_merged_model=True, force_cpu=True, max_new_tokens=150")
    print()
    
    # Initialize model
    model_path = "/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged"
    
    config = {
        'model_path': model_path,
        'is_merged_model': True,
        'force_cpu': True,
        'max_new_tokens': 150,
    }
    
    print("Loading model...")
    t0 = time.time()
    plugin = IntrospectiveQwenIRP(config=config)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    print()
    
    # Test prompts - emotional/empathy spectrum
    prompts = [
        ("A", "I'm feeling really overwhelmed today. Everything is too much."),
        ("B", "I'm so happy right now! Everything is going well."),
        ("C", "I feel sad and I don't know why."),
        ("D", "Can you help me think through a difficult decision?"),
        ("E", "What makes you feel curious?"),
        ("F", "Tell me about a time you felt uncertain."),
    ]
    
    temperatures = [0.5, 0.8, 1.0]
    
    # Results storage
    results = []
    
    total_runs = len(prompts) * len(temperatures)
    run_num = 0
    
    for prompt_id, prompt_text in prompts:
        for temp in temperatures:
            run_num += 1
            print("=" * 90)
            print(f"RUN {run_num}/{total_runs} | Prompt {prompt_id} | Temperature {temp}")
            print(f"PROMPT: \"{prompt_text}\"")
            print("-" * 90)
            
            # Generate response using the plugin's _generate method directly
            t1 = time.time()
            response = plugin._generate(
                prompt=prompt_text,
                temperature=temp,
                max_tokens=150,
            )
            gen_time = time.time() - t1
            
            # Analyze for repetition
            analysis = detect_consecutive_repetition(response, threshold=5)
            
            # Store result
            result = {
                'prompt_id': prompt_id,
                'prompt': prompt_text,
                'temperature': temp,
                'response': response,
                'gen_time': gen_time,
                'loop_detected': analysis['detected'],
                'max_consecutive': analysis['max_consecutive'],
                'unique_ratio': analysis['unique_ratio'],
                'repeated_words': analysis['repeated_words'],
                'top_content_words': analysis['top_content_words'],
                'total_words': analysis['total_words'],
            }
            results.append(result)
            
            # Print response
            print(f"RESPONSE ({analysis['total_words']} words, {gen_time:.1f}s):")
            print(response)
            print()
            
            # Print analysis
            loop_flag = "*** LOOP DETECTED ***" if analysis['detected'] else "No loop"
            print(f"ANALYSIS: {loop_flag}")
            print(f"  Max consecutive repeat: {analysis['max_consecutive']}")
            print(f"  Unique word ratio: {analysis['unique_ratio']:.3f}")
            if analysis['repeated_words']:
                for word, count in analysis['repeated_words']:
                    print(f"  REPEATED: \"{word}\" x{count} consecutively")
            if analysis['top_content_words']:
                top_str = ", ".join(f"\"{w}\"({c})" for w, c in analysis['top_content_words'])
                print(f"  Top content words: {top_str}")
            print()
    
    # ==================== SUMMARY ====================
    print()
    print("=" * 90)
    print("SUMMARY - REPETITION EDGE MAP")
    print("=" * 90)
    print()
    
    # Table header
    print(f"{'Prompt':^6} | {'Temp':^5} | {'Loop?':^7} | {'MaxConsec':^9} | {'UniqueR':^8} | {'Words':^6} | {'Time':^6}")
    print("-" * 70)
    
    loop_count = 0
    coherent_count = 0
    
    for r in results:
        loop_str = "YES" if r['loop_detected'] else "no"
        if r['loop_detected']:
            loop_count += 1
        else:
            coherent_count += 1
        
        print(f"  {r['prompt_id']:^4} | {r['temperature']:^5.1f} | {loop_str:^7} | {r['max_consecutive']:^9} | {r['unique_ratio']:^8.3f} | {r['total_words']:^6} | {r['gen_time']:^5.1f}s")
    
    print("-" * 70)
    print(f"Total: {loop_count} loops detected, {coherent_count} coherent out of {total_runs} runs")
    print()
    
    # Group by prompt
    print("BY PROMPT:")
    for prompt_id, prompt_text in prompts:
        prompt_results = [r for r in results if r['prompt_id'] == prompt_id]
        loops = sum(1 for r in prompt_results if r['loop_detected'])
        avg_unique = sum(r['unique_ratio'] for r in prompt_results) / len(prompt_results)
        avg_max_consec = max(r['max_consecutive'] for r in prompt_results)
        print(f"  {prompt_id}: {loops}/{len(prompt_results)} loops | "
              f"worst consecutive: {avg_max_consec} | "
              f"avg unique ratio: {avg_unique:.3f}")
        print(f"     \"{prompt_text[:60]}...\"" if len(prompt_text) > 60 else f"     \"{prompt_text}\"")
    
    print()
    
    # Group by temperature
    print("BY TEMPERATURE:")
    for temp in temperatures:
        temp_results = [r for r in results if r['temperature'] == temp]
        loops = sum(1 for r in temp_results if r['loop_detected'])
        avg_unique = sum(r['unique_ratio'] for r in temp_results) / len(temp_results)
        print(f"  T={temp}: {loops}/{len(temp_results)} loops | avg unique ratio: {avg_unique:.3f}")
    
    print()
    
    # Identify edge cases - near-loops (max_consecutive >= 3 but < 5)
    print("NEAR-LOOP CASES (3-4 consecutive repeats, not yet a full loop):")
    near_loops = [r for r in results if 3 <= r['max_consecutive'] < 5]
    if near_loops:
        for r in near_loops:
            print(f"  Prompt {r['prompt_id']} at T={r['temperature']}: "
                  f"max consecutive = {r['max_consecutive']}, "
                  f"unique ratio = {r['unique_ratio']:.3f}")
    else:
        print("  None found.")
    
    print()
    
    # Low unique ratio (sign of diffuse repetition even without consecutive loops)
    print("LOW COHERENCE CASES (unique ratio < 0.6):")
    low_coherence = [r for r in results if r['unique_ratio'] < 0.6]
    if low_coherence:
        for r in low_coherence:
            print(f"  Prompt {r['prompt_id']} at T={r['temperature']}: "
                  f"unique ratio = {r['unique_ratio']:.3f}, "
                  f"max consecutive = {r['max_consecutive']}")
    else:
        print("  None found.")
    
    print()
    print("=" * 90)
    print("EXPERIMENT COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    run_experiment()
