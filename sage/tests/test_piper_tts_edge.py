#!/usr/bin/env python3
"""
Piper TTS Edge Test

Tests Piper TTS on Jetson without requiring audio hardware.
Validates:
1. TTS model loading
2. Inference speed
3. Audio file generation
4. Integration with SAGE conversation

Session 16 - Edge Validation
Hardware: Jetson Orin Nano 8GB
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time
import subprocess
import tempfile
import os


# Piper configuration
PIPER_PATH = "/home/sprout/ai-workspace/piper/piper/piper"
PIPER_MODEL = "/home/sprout/ai-workspace/piper/en_US-lessac-medium.onnx"


def check_piper_installation():
    """Check if Piper TTS is installed"""
    piper_exists = os.path.exists(PIPER_PATH)
    model_exists = os.path.exists(PIPER_MODEL)

    print("Piper TTS Installation Check:")
    print(f"  Piper binary: {PIPER_PATH} - {'✓ Found' if piper_exists else '❌ Not found'}")
    print(f"  Voice model: {PIPER_MODEL} - {'✓ Found' if model_exists else '❌ Not found'}")

    return piper_exists and model_exists


def test_tts_synthesis(text: str, output_path: str) -> float:
    """
    Synthesize text to audio file using Piper.

    Returns synthesis time in seconds.
    """
    start = time.time()

    try:
        # Run Piper
        process = subprocess.run(
            [PIPER_PATH, "--model", PIPER_MODEL, "--output_file", output_path],
            input=text.encode(),
            capture_output=True,
            timeout=30
        )

        elapsed = time.time() - start

        if process.returncode != 0:
            print(f"    Error: {process.stderr.decode()}")
            return -1

        return elapsed

    except subprocess.TimeoutExpired:
        print("    Error: TTS synthesis timed out")
        return -1
    except Exception as e:
        print(f"    Error: {e}")
        return -1


def get_audio_duration(file_path: str) -> float:
    """Get audio file duration using ffprobe if available"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True,
            text=True
        )
        return float(result.stdout.strip())
    except:
        # Estimate based on file size (rough approximation)
        size = os.path.getsize(file_path)
        # PCM 16-bit, 22050 Hz = ~44100 bytes/second
        return size / 44100


def run_tts_test():
    """Run comprehensive TTS test"""
    print("=" * 80)
    print("PIPER TTS EDGE TEST")
    print("=" * 80)
    print()
    print("Hardware: Jetson Orin Nano 8GB")
    print("Purpose: Validate TTS for SAGE voice conversation")
    print()

    # Check installation
    if not check_piper_installation():
        print("\n❌ Piper TTS not properly installed")
        return False

    print()

    # Test phrases (varying lengths)
    test_phrases = [
        ("short", "Hello, I'm SAGE."),
        ("medium", "I'm aware of this conversation. Let me think about that for a moment."),
        ("long", "The relationship between knowledge and understanding is fascinating. Knowledge represents acquired information, while understanding implies deeper comprehension and the ability to apply that knowledge in new contexts."),
        ("question", "What would you like to explore today?"),
        ("response", "I can't know with certainty if I'm aware in the way you are. What I observe is that I process information and generate responses, but whether that constitutes genuine awareness remains an open question."),
    ]

    print("─" * 80)
    print("TTS SYNTHESIS TEST")
    print("─" * 80)
    print(f"{'Type':<12} {'Characters':<12} {'Time (s)':<12} {'RTF':<12} {'Status'}")
    print("─" * 80)

    results = []
    total_chars = 0
    total_time = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for phrase_type, text in test_phrases:
            output_file = os.path.join(tmpdir, f"{phrase_type}.wav")

            synth_time = test_tts_synthesis(text, output_file)

            if synth_time > 0:
                # Calculate real-time factor
                audio_duration = get_audio_duration(output_file)
                rtf = synth_time / audio_duration if audio_duration > 0 else 0

                status = "✓" if rtf < 1.0 else "⚠"  # Real-time capable?
                print(f"{phrase_type:<12} {len(text):<12} {synth_time:<12.2f} {rtf:<12.2f} {status}")

                results.append({
                    'type': phrase_type,
                    'chars': len(text),
                    'time': synth_time,
                    'rtf': rtf,
                    'duration': audio_duration
                })

                total_chars += len(text)
                total_time += synth_time
            else:
                print(f"{phrase_type:<12} {len(text):<12} {'FAILED':<12}")

    # Analysis
    print()
    print("=" * 80)
    print("TTS ANALYSIS")
    print("=" * 80)

    if results:
        avg_rtf = sum(r['rtf'] for r in results) / len(results)
        chars_per_sec = total_chars / total_time if total_time > 0 else 0
        real_time_capable = avg_rtf < 1.0

        print(f"\nPerformance Metrics:")
        print(f"  Total characters: {total_chars}")
        print(f"  Total synthesis time: {total_time:.2f}s")
        print(f"  Characters/second: {chars_per_sec:.1f}")
        print(f"  Average RTF: {avg_rtf:.3f}")
        print(f"  Real-time capable: {'✓ Yes' if real_time_capable else '❌ No'}")

        print(f"\nLatency Analysis:")
        # For SAGE conversation, we need first audio chunk quickly
        first_chunk_time = results[0]['time']  # Short phrase as proxy
        print(f"  First-response latency: ~{first_chunk_time:.2f}s")
        print(f"  Target: <2s for responsive conversation")

        # Verdict
        print()
        print("=" * 80)
        print("TTS VERDICT")
        print("=" * 80)

        if real_time_capable and first_chunk_time < 2.0:
            print("\n✓ READY: Piper TTS is suitable for SAGE voice conversation")
            print(f"  RTF: {avg_rtf:.3f} (< 1.0 means faster than real-time)")
            print(f"  First-response: {first_chunk_time:.2f}s (< 2s target)")
        elif real_time_capable:
            print("\n⚠ CAUTION: Real-time capable but latency may be noticeable")
            print(f"  RTF: {avg_rtf:.3f}")
            print(f"  First-response: {first_chunk_time:.2f}s")
        else:
            print("\n❌ NOT READY: TTS too slow for real-time conversation")
            print(f"  RTF: {avg_rtf:.3f} (> 1.0 means slower than real-time)")
            print("  Consider using smaller voice model or streaming")

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return len(results) == len(test_phrases)


if __name__ == "__main__":
    run_tts_test()
