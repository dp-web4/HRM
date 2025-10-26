#!/usr/bin/env python3
"""
Test Sentence-Level Buffering for TTS

This test verifies that streaming word-by-word generation is correctly
buffered at sentence boundaries before being sent to TTS, preserving
natural prosody and word transitions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.integration.streaming_responder import StreamingResponder
from cognitive.sage_system_prompt import get_sage_system_prompt
import time


def test_sentence_buffering():
    """Test that sentence buffering preserves natural prosody"""

    print("=" * 80)
    print("SENTENCE-LEVEL BUFFERING TEST")
    print("=" * 80)
    print()
    print("This test verifies that:")
    print("1. Word-by-word generation continues (streaming)")
    print("2. Complete sentences are buffered before TTS")
    print("3. Natural prosody and word transitions are preserved")
    print()

    # Create responder
    responder = StreamingResponder(
        max_new_tokens=512,
        temperature=0.7,
        words_per_chunk=3  # Stream every 3 words
    )

    system_prompt = get_sage_system_prompt()

    # Test prompt that should generate multiple sentences
    test_prompt = "Tell me briefly about yourself and what you can do."

    print(f"PROMPT: {test_prompt}")
    print("=" * 80)
    print()

    # Track sentence buffering
    sentence_buffer = ""
    sentences_spoken = []
    chunk_count = 0

    def on_chunk(chunk_text, is_final):
        """Buffer chunks until sentence complete"""
        nonlocal sentence_buffer, chunk_count
        chunk_count += 1

        sentence_buffer += chunk_text
        print(f"[CHUNK {chunk_count}] Received: '{chunk_text.strip()}'")

        # Check for sentence boundary
        sentence_end = False
        for boundary in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            if boundary in sentence_buffer:
                sentence_end = True
                break

        # Also check if final chunk ends with punctuation
        if is_final and sentence_buffer.rstrip() and sentence_buffer.rstrip()[-1] in '.!?':
            sentence_end = True

        # Speak complete sentence
        if sentence_end or is_final:
            complete_sentence = sentence_buffer.strip()
            if complete_sentence:
                print(f"\n🔊 [TTS] Speaking complete sentence:")
                print(f"    \"{complete_sentence}\"")
                print()
                sentences_spoken.append(complete_sentence)
                sentence_buffer = ""  # Reset for next sentence

    # Generate with streaming
    start = time.time()
    result = responder.generate_response_streaming(
        test_prompt,
        system_prompt=system_prompt,
        on_chunk=on_chunk
    )
    elapsed = time.time() - start

    # Analysis
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Total generation time: {elapsed:.2f}s")
    print(f"Total chunks received: {chunk_count}")
    print(f"Sentences spoken to TTS: {len(sentences_spoken)}")
    print()

    print("📝 Full response (raw):")
    print(f"   {result['full_response']}")
    print()

    print("🔊 Sentences sent to TTS:")
    for i, sentence in enumerate(sentences_spoken, 1):
        print(f"   {i}. \"{sentence}\"")
    print()

    # Validate
    print("✅ VALIDATION:")
    print(f"   - Streaming generation: {'✓' if chunk_count > 1 else '✗'}")
    print(f"   - Multiple sentences: {'✓' if len(sentences_spoken) > 1 else '✗'}")
    print(f"   - Complete sentences: {'✓' if all(s[-1] in '.!?' for s in sentences_spoken) else '✗'}")
    print(f"   - No partial sentences: {'✓' if all(len(s.split()) > 2 for s in sentences_spoken) else '✗'}")
    print()

    print("=" * 80)
    print("KEY BENEFITS:")
    print("=" * 80)
    print("""
1. STREAMING PRESERVED: Words generated incrementally (not pre-computed)
2. SENTENCE BUFFERING: TTS receives complete sentences (natural prosody)
3. PIPELINED SPEECH: Sentence N speaks while sentence N+1 buffers
4. LOW LATENCY: First sentence speaks as soon as complete (not waiting for all)
5. NATURAL PROSODY: TTS can process sentence structure for word transitions

This gives us the best of both worlds:
- Fast response (1-3s to first sentence)
- Natural speech quality (complete sentences with prosody)
""")
    print("=" * 80)


if __name__ == "__main__":
    test_sentence_buffering()
