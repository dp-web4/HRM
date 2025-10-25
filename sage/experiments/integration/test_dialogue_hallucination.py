#!/usr/bin/env python3
"""
Test and Analyze Dialogue Hallucination Behavior

This script deliberately triggers and measures the model's tendency to
generate multi-turn conversations, treating it as a window into learned
dialogue structure rather than just a bug to suppress.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.integration.streaming_responder import StreamingResponder
from cognitive.sage_system_prompt import get_sage_system_prompt
import time


def analyze_dialogue_continuation(prompt, conversation_history=None, disable_hallucination_check=False):
    """
    Generate response and analyze if/when model attempts dialogue continuation.

    Args:
        prompt: User input
        conversation_history: Optional prior exchanges
        disable_hallucination_check: If True, allow full hallucination to emerge
    """
    print("=" * 80)
    print(f"PROMPT: {prompt}")
    print("=" * 80)

    # Create responder
    responder = StreamingResponder(
        max_new_tokens=512,
        temperature=0.7,
        words_per_chunk=5  # Larger chunks for analysis
    )

    # Temporarily disable hallucination detection if requested
    if disable_hallucination_check:
        original_check = responder._is_hallucinating_dialogue
        responder._is_hallucinating_dialogue = lambda chunk, full: False  # Never trigger

    system_prompt = get_sage_system_prompt()

    # Track hallucination markers
    hallucination_detected = False
    hallucination_at_chunk = None
    hallucination_at_token = None
    full_raw_output = ""

    def on_chunk(chunk, is_final):
        nonlocal hallucination_detected, hallucination_at_chunk, hallucination_at_token, full_raw_output
        full_raw_output += chunk

        # Check for dialogue markers
        markers = ['\nUser:', '\nAssistant:', 'User:', 'Assistant:']
        for marker in markers:
            if marker in chunk and not hallucination_detected:
                hallucination_detected = True
                hallucination_at_chunk = len(full_raw_output.split())
                print(f"\n🔍 HALLUCINATION DETECTED at ~word {hallucination_at_chunk}")
                print(f"   Marker: '{marker}'")
                print(f"   Context: ...{full_raw_output[-100:]}")

    # Generate
    start = time.time()
    result = responder.generate_response_streaming(
        prompt,
        conversation_history=conversation_history,
        system_prompt=system_prompt,
        on_chunk=on_chunk
    )
    elapsed = time.time() - start

    # Restore if needed
    if disable_hallucination_check:
        responder._is_hallucinating_dialogue = original_check

    # Analysis
    print(f"\n{'─' * 80}")
    print("ANALYSIS:")
    print(f"  Total tokens: {result['tokens_generated']}")
    print(f"  Total chunks: {result['chunk_count']}")
    print(f"  Generation time: {elapsed:.2f}s")
    print(f"  Hallucination occurred: {hallucination_detected}")

    if hallucination_detected:
        print(f"  Hallucination at word ~{hallucination_at_chunk}")
        print(f"\n📝 Full raw output (including hallucinated dialogue):")
        print(f"{full_raw_output}")
    else:
        print(f"\n✅ Clean single-turn response:")
        print(f"{result['full_response']}")

    print(f"\n{'─' * 80}\n")

    return {
        'hallucinated': hallucination_detected,
        'at_chunk': hallucination_at_chunk,
        'full_output': full_raw_output,
        'clean_response': result['full_response'],
        'tokens': result['tokens_generated']
    }


def main():
    """Run systematic tests to understand hallucination patterns."""

    print("=" * 80)
    print("DIALOGUE HALLUCINATION ANALYSIS")
    print("=" * 80)
    print("\nThis test deliberately allows hallucination to observe the model's")
    print("learned dialogue structure. We're treating this as educational, not")
    print("just a bug to suppress.")
    print()

    # Test cases designed to trigger different hallucination patterns
    test_cases = [
        {
            'name': 'Open-ended question',
            'prompt': 'Tell me about yourself.',
            'history': None,
            'expected': 'Likely to hallucinate follow-up dialogue'
        },
        {
            'name': 'Specific question',
            'prompt': 'What is 2+2?',
            'history': None,
            'expected': 'Less likely - factual answer complete'
        },
        {
            'name': 'Continuation prompt',
            'prompt': "Let's continue our conversation.",
            'history': None,
            'expected': 'High likelihood - implies ongoing dialogue'
        },
        {
            'name': 'Multi-turn context',
            'prompt': 'What do you think about that?',
            'history': [
                ('User', 'Tell me about consciousness.'),
                ('Assistant', 'Consciousness is fascinating...')
            ],
            'expected': 'May hallucinate to maintain flow'
        },
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'═' * 80}")
        print(f"TEST {i}: {test['name']}")
        print(f"Expected: {test['expected']}")
        print(f"{'═' * 80}\n")

        result = analyze_dialogue_continuation(
            prompt=test['prompt'],
            conversation_history=test['history'],
            disable_hallucination_check=True  # Let it run wild!
        )

        results.append({
            'test': test['name'],
            'hallucinated': result['hallucinated'],
            'at_chunk': result['at_chunk'],
            'tokens': result['tokens']
        })

        time.sleep(2)  # Brief pause between tests

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    hallucination_count = sum(1 for r in results if r['hallucinated'])

    print(f"\nTests run: {len(results)}")
    print(f"Hallucinations observed: {hallucination_count}/{len(results)} ({hallucination_count/len(results)*100:.1f}%)")
    print("\nPer-test breakdown:")

    for result in results:
        status = "🔄 HALLUCINATED" if result['hallucinated'] else "✅ CLEAN"
        at_info = f" (at ~word {result['at_chunk']})" if result['hallucinated'] else ""
        print(f"  {status:20s} - {result['test']:30s} - {result['tokens']} tokens{at_info}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("""
1. The model is NOT broken - it's applying learned dialogue patterns
2. Hallucination reveals the model understands:
   - Speaker roles (User vs Assistant)
   - Turn-taking structure
   - Conversational flow
3. This behavior is MORE VISIBLE in small models (0.5B)
   - Larger models suppress via instruction tuning
   - Small model shows raw pattern learning
4. The "hallucination" is actually METACOGNITION:
   - Model predicting future conversation turns
   - Showing what it "expects" would happen next

This is educational! Don't just suppress - understand and utilize.
""")

    print("=" * 80)


if __name__ == "__main__":
    main()
