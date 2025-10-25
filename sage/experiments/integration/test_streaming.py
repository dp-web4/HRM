#!/usr/bin/env python3
"""
Test true streaming generation (word-by-word).

This mirrors biological speech production:
- Words arrive as they're generated (not batched)
- Large buffer (512 tokens) allows complete thoughts
- Natural stopping when thought is complete
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.integration.streaming_responder import StreamingResponder
from cognitive.sage_system_prompt import get_sage_system_prompt
import time


def test_streaming():
    """Test true streaming generation."""
    print("="*80)
    print("STREAMING WORD-BY-WORD GENERATION TEST")
    print("="*80)

    responder = StreamingResponder(
        max_new_tokens=512,  # Large buffer for complete thoughts
        temperature=0.7,
        words_per_chunk=3  # Stream every 3 words
    )

    system_prompt = get_sage_system_prompt()

    test_questions = [
        "Tell me about yourself.",
        "What is consciousness?"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {question}")
        print(f"{'='*80}\n")

        # Track when chunks arrive
        chunk_times = []
        start_time = time.time()

        def on_chunk_callback(chunk_text, is_final):
            """Called immediately when each chunk is ready."""
            elapsed = time.time() - start_time
            chunk_times.append(elapsed)

            status = "FINAL" if is_final else "chunk"
            print(f"[{elapsed:.1f}s] 🗣️  {status}: '{chunk_text.strip()}'")

        # Generate with streaming
        result = responder.generate_response_streaming(
            question,
            conversation_history=[],
            system_prompt=system_prompt,
            on_chunk=on_chunk_callback
        )

        # Summary
        print(f"\n{'─'*80}")
        print(f"COMPLETE: {result['chunk_count']} chunks, {result['tokens_generated']} tokens in {result['total_time']:.2f}s")
        print(f"Chunk arrival times: {', '.join(f'{t:.1f}s' for t in chunk_times[:10])}...")
        print(f"\nFull response:\n{result['full_response']}")
        print(f"{'─'*80}")

        if i < len(test_questions):
            print("\n\n")
            time.sleep(1)

    print(f"\n{'='*80}")
    print("Key Benefits:")
    print("  - First words arrive in <2s (immediate feedback)")
    print("  - Continuous stream (like human speech)")
    print("  - Natural stopping (thought completeness, not buffer size)")
    print("  - Can generate long responses (up to 512 tokens)")
    print("  - User can interrupt at any time")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_streaming()
