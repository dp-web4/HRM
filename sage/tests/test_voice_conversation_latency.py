#!/usr/bin/env python3
"""
Voice Conversation Latency Test - Edge Validation

Tests end-to-end latency for SAGE voice conversations:
1. User speaks (simulated - instant)
2. LLM generates response
3. TTS synthesizes speech
4. Total user-perceived latency

Key metrics:
- Time to first token (TTFT)
- LLM generation time
- TTS synthesis time (RTF)
- Total conversation turn latency

Target: <5s for responsive conversation feel
"""

import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.irp.plugins.llm_impl import ConversationalLLM

# Piper TTS paths
PIPER_PATH = "/home/sprout/ai-workspace/piper/piper/piper"
PIPER_MODEL = "/home/sprout/ai-workspace/piper/en_US-lessac-medium.onnx"


@dataclass
class ConversationTurn:
    """Record of a single conversation turn"""
    user_input: str
    llm_response: str
    llm_time: float
    tts_time: float
    total_time: float
    response_chars: int
    audio_duration: float


def get_temperature() -> float:
    """Read Jetson thermal zone temperature"""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return int(f.read().strip()) / 1000.0
    except:
        return 0.0


def get_power_usage() -> Optional[float]:
    """Read power consumption in milliwatts"""
    try:
        # Jetson power rails
        power_paths = [
            '/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input',
            '/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/power1_input',
        ]
        import glob
        for pattern in power_paths:
            matches = glob.glob(pattern)
            if matches:
                with open(matches[0], 'r') as f:
                    return float(f.read().strip())
        return None
    except:
        return None


def synthesize_tts(text: str, output_path: str) -> tuple[float, float]:
    """
    Synthesize text to audio using Piper TTS.
    Returns (synthesis_time, audio_duration)
    """
    start = time.perf_counter()

    process = subprocess.run(
        [PIPER_PATH, "--model", PIPER_MODEL, "--output_file", output_path],
        input=text.encode(),
        capture_output=True,
        timeout=60
    )

    synthesis_time = time.perf_counter() - start

    # Get audio duration using soxi or ffprobe
    try:
        result = subprocess.run(
            ["soxi", "-D", output_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            audio_duration = float(result.stdout.strip())
        else:
            # Fallback: estimate from chars (~150 chars/sec speaking rate)
            audio_duration = len(text) / 15.0  # ~15 chars per second
    except:
        audio_duration = len(text) / 15.0

    return synthesis_time, audio_duration


def run_conversation_test():
    """Run end-to-end voice conversation latency test"""

    print("=" * 80)
    print("VOICE CONVERSATION LATENCY TEST")
    print("=" * 80)
    print()
    print("Hardware: Jetson Orin Nano 8GB")
    print("Purpose: Measure end-to-end conversation turn latency")
    print()

    # Check Piper installation
    if not os.path.exists(PIPER_PATH):
        print(f"ERROR: Piper not found at {PIPER_PATH}")
        return
    if not os.path.exists(PIPER_MODEL):
        print(f"ERROR: Piper model not found at {PIPER_MODEL}")
        return

    print("Piper TTS: ✓ Installed")

    # Test conversation prompts (simulating user questions)
    test_prompts = [
        ("greeting", "Hello, how are you today?"),
        ("simple", "What is the capital of France?"),
        ("medium", "Explain what consciousness means to you in a few sentences."),
        ("complex", "How do you understand the relationship between knowledge and belief?"),
        ("emotional", "What brings you joy in our conversations?"),
    ]

    # Load LLM model
    print()
    print("Loading LLM model...")
    model_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"

    start_load = time.perf_counter()
    conv = ConversationalLLM(
        model_path=model_path,
        irp_iterations=3,  # Edge-optimized
    )
    load_time = time.perf_counter() - start_load
    print(f"Model loaded in {load_time:.1f}s")

    print()
    print("-" * 80)
    print("CONVERSATION TEST")
    print("-" * 80)
    print(f"{'Type':<12} {'LLM (s)':<10} {'TTS (s)':<10} {'Total (s)':<10} {'Chars':<8} {'Status'}")
    print("-" * 80)

    results: List[ConversationTurn] = []
    initial_temp = get_temperature()

    with tempfile.TemporaryDirectory() as tmpdir:
        for prompt_type, user_input in test_prompts:
            # Record start temperature
            temp_start = get_temperature()

            # Phase 1: LLM inference
            llm_start = time.perf_counter()
            try:
                llm_response, metadata = conv.respond(user_input)
                # Clean up response for TTS (remove special tokens)
                llm_response = llm_response.strip()
                if len(llm_response) > 500:
                    llm_response = llm_response[:500] + "..."
            except Exception as e:
                print(f"{prompt_type:<12} {'ERROR':<10} {'-':<10} {'-':<10} {'-':<8} ✗ {str(e)[:30]}")
                continue
            llm_time = time.perf_counter() - llm_start

            # Phase 2: TTS synthesis
            output_path = os.path.join(tmpdir, f"{prompt_type}.wav")
            tts_time, audio_duration = synthesize_tts(llm_response, output_path)

            # Total time
            total_time = llm_time + tts_time

            # Status based on total latency
            if total_time < 5:
                status = "✓ Fast"
            elif total_time < 10:
                status = "⚠ OK"
            else:
                status = "✗ Slow"

            result = ConversationTurn(
                user_input=user_input,
                llm_response=llm_response,
                llm_time=llm_time,
                tts_time=tts_time,
                total_time=total_time,
                response_chars=len(llm_response),
                audio_duration=audio_duration
            )
            results.append(result)

            print(f"{prompt_type:<12} {llm_time:<10.1f} {tts_time:<10.2f} {total_time:<10.1f} {len(llm_response):<8} {status}")

    # Analysis
    print()
    print("=" * 80)
    print("CONVERSATION ANALYSIS")
    print("=" * 80)

    if results:
        avg_llm = sum(r.llm_time for r in results) / len(results)
        avg_tts = sum(r.tts_time for r in results) / len(results)
        avg_total = sum(r.total_time for r in results) / len(results)
        avg_chars = sum(r.response_chars for r in results) / len(results)

        min_total = min(r.total_time for r in results)
        max_total = max(r.total_time for r in results)

        fast_responses = sum(1 for r in results if r.total_time < 5)
        acceptable_responses = sum(1 for r in results if r.total_time < 10)

        print()
        print("Latency Breakdown:")
        print(f"  Average LLM inference: {avg_llm:.1f}s")
        print(f"  Average TTS synthesis: {avg_tts:.2f}s")
        print(f"  Average total latency: {avg_total:.1f}s")
        print()
        print("Response Statistics:")
        print(f"  Average response length: {avg_chars:.0f} chars")
        print(f"  Min turn latency: {min_total:.1f}s")
        print(f"  Max turn latency: {max_total:.1f}s")
        print()
        print("Performance Distribution:")
        print(f"  Fast (<5s): {fast_responses}/{len(results)} ({100*fast_responses/len(results):.0f}%)")
        print(f"  Acceptable (<10s): {acceptable_responses}/{len(results)} ({100*acceptable_responses/len(results):.0f}%)")
        print()

        # Thermal analysis
        final_temp = get_temperature()
        print("Thermal Analysis:")
        print(f"  Start temperature: {initial_temp:.1f}°C")
        print(f"  End temperature: {final_temp:.1f}°C")
        print(f"  Temperature rise: {final_temp - initial_temp:+.1f}°C")

        # LLM vs TTS ratio
        llm_ratio = avg_llm / avg_total * 100
        tts_ratio = avg_tts / avg_total * 100
        print()
        print("Latency Attribution:")
        print(f"  LLM inference: {llm_ratio:.0f}% of total time")
        print(f"  TTS synthesis: {tts_ratio:.0f}% of total time")

        # Verdict
        print()
        print("=" * 80)
        print("VOICE CONVERSATION VERDICT")
        print("=" * 80)
        print()

        if avg_total < 5:
            verdict = "✓ EXCELLENT"
            verdict_msg = "Voice conversations feel responsive"
        elif avg_total < 10:
            verdict = "⚠ ACCEPTABLE"
            verdict_msg = "Voice conversations functional but noticeable delay"
        else:
            verdict = "✗ NEEDS OPTIMIZATION"
            verdict_msg = "Voice conversation latency too high for natural flow"

        print(f"{verdict}: {verdict_msg}")
        print(f"  Average turn latency: {avg_total:.1f}s")
        print(f"  Target: <5s for responsive feel, <10s for acceptable")
        print()

        # Optimization suggestions
        if avg_total >= 5:
            print("Optimization Opportunities:")
            if llm_ratio > 80:
                print("  - LLM dominates latency. Consider:")
                print("    • Shorter max_new_tokens (currently 150)")
                print("    • Fewer IRP iterations (currently 3)")
                print("    • Streaming response generation")
            if tts_ratio > 30:
                print("  - TTS is significant. Consider:")
                print("    • Smaller voice model (fast vs medium)")
                print("    • Streaming TTS (synthesize while generating)")

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    # Print sample responses
    if results:
        print()
        print("Sample Responses:")
        print("-" * 80)
        for r in results[:3]:
            print(f"User: {r.user_input}")
            print(f"SAGE: {r.llm_response[:200]}...")
            print(f"[{r.total_time:.1f}s total, {r.response_chars} chars]")
            print()


if __name__ == "__main__":
    run_conversation_test()
