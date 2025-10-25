#!/usr/bin/env python3
"""
Test breath-based generation vs traditional generation.

Compare:
1. Traditional: Wait 15s, get full response at once
2. Breath-based: Get chunks every ~3-5s as they're generated
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.integration.breath_based_responder import BreathBasedResponder
from cognitive.sage_system_prompt import get_sage_system_prompt
import time


def test_breath_based():
    """Test breath-based generation with streaming output."""
    print("="*80)
    print("BREATH-BASED GENERATION TEST")
    print("="*80)

    responder = BreathBasedResponder(
        breath_size=40,  # ~1 sentence per breath
        max_breaths=5,   # Up to ~200 tokens total
        temperature=0.7
    )

    system_prompt = get_sage_system_prompt()

    test_questions = [
        "Tell me about yourself.",
        "What is quantum entanglement?",
        "How do neural networks learn?"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {question}")
        print(f"{'='*80}\n")

        # Track when each breath arrives
        breath_times = []
        start_time = time.time()

        def on_breath_callback(breath_text, breath_num, is_final):
            """Called immediately when each breath is ready."""
            elapsed = time.time() - start_time
            breath_times.append(elapsed)

            status = "FINAL" if is_final else f"breath {breath_num}"
            print(f"\n[{elapsed:.1f}s] 🗣️  {status}: {breath_text.strip()}")
            print(f"         (would speak this immediately)")

        # Generate with streaming
        result = responder.generate_response_streaming(
            question,
            conversation_history=[],
            system_prompt=system_prompt,
            on_breath=on_breath_callback
        )

        # Summary
        print(f"\n{'─'*80}")
        print(f"COMPLETE: {result['breath_count']} breaths in {result['total_time']:.2f}s")
        print(f"Breath timings: {', '.join(f'{t:.1f}s' for t in breath_times)}")
        print(f"Thought complete: {result['thought_complete']}")
        print(f"\nFull response:\n{result['full_response']}")
        print(f"{'─'*80}")

        # Wait before next question
        if i < len(test_questions):
            print("\n\n")
            time.sleep(1)

    print(f"\n{'='*80}")
    print("Key Benefits:")
    print("  - First breath arrives in ~3-5s (not 15s+)")
    print("  - User hears response as it's being 'thought'")
    print("  - Natural pauses between sentences")
    print("  - Can interrupt between breaths")
    print("  - Total time similar, but PERCEIVED latency much lower")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_breath_based()
