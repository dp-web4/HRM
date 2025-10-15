#!/usr/bin/env python3
"""
Real-Time Conversation Test - All Components Integrated

Tests the complete real-time conversation stack:
1. StreamingAudioSensor (faster-whisper + VAD)
2. CognitiveMailbox (memory-based IPC)
3. PatternResponseEngine (fast cognitive responses)
4. Piper TTS (streaming speech synthesis)

Target: <2s end-to-end latency
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import subprocess
import os

# Import components
from interfaces.streaming_audio_sensor import StreamingAudioSensor
from interfaces.cognitive_mailbox import CognitiveMailbox
from cognitive.pattern_responses import CognitiveRouter

print("="*60)
print("SAGE REAL-TIME CONVERSATION TEST")
print("="*60)

# Initialize components
print("\n1. Initializing audio streaming sensor...")
audio_sensor = StreamingAudioSensor({
    'sensor_id': 'realtime_audio',
    'sensor_type': 'audio',
    'device': 'cpu',
    'bt_device': 'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit',
    'sample_rate': 16000,
    'vad_aggressiveness': 2,
    'min_speech_duration': 0.5,
    'max_speech_duration': 10.0,
    'min_confidence': 0.4,
    'whisper_model': 'tiny'
})

print("\n2. Initializing cognitive mailbox...")
mailbox = CognitiveMailbox(use_fallback=True)  # Use queue fallback for now

print("\n3. Initializing pattern-based cognitive router...")
cognitive_router = CognitiveRouter()

print("\n4. Piper TTS configuration...")
PIPER_PATH = "/home/sprout/ai-workspace/piper/piper/piper"
PIPER_MODEL = "/home/sprout/ai-workspace/piper/en_US-lessac-medium.onnx"
BT_SINK = "bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit"

def speak_with_piper(text: str):
    """Streaming TTS with Piper"""
    try:
        # Pipe text through Piper to paplay
        piper_proc = subprocess.Popen(
            [PIPER_PATH, "--model", PIPER_MODEL, "--output_raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        play_proc = subprocess.Popen(
            ["paplay",
             "--device", BT_SINK,
             "--rate", "22050",
             "--format", "s16le",
             "--channels", "1",
             "--raw"],
            stdin=piper_proc.stdout,
            stderr=subprocess.DEVNULL
        )

        # Send text to Piper
        piper_proc.stdin.write(text.encode('utf-8'))
        piper_proc.stdin.close()

        # Wait for completion
        play_proc.wait(timeout=10)

    except Exception as e:
        print(f"  TTS error: {e}")

print("\n" + "="*60)
print("✅ REAL-TIME CONVERSATION READY")
print("="*60)
print("Speak naturally - I'll respond as fast as I can!")
print("Press Ctrl+C to stop\n")

# Statistics
exchange_count = 0
latencies = []

try:
    while True:
        # 1. Poll for user speech (non-blocking)
        reading = audio_sensor.poll()

        if reading:
            exchange_start = time.time()

            text = reading.metadata['text']
            confidence = reading.confidence
            transcription_time = reading.metadata.get('transcription_time', 0)

            print(f"\n👤 USER [{confidence:.2f}]: {text}")
            print(f"   Transcription time: {transcription_time*1000:.0f}ms")

            # 2. Post to mailbox
            mailbox.post_transcription(text, confidence)

            # 3. Cognitive processing
            msg = mailbox.check_transcription()
            if msg:
                cognitive_start = time.time()
                response_text, method = cognitive_router.process(msg['text'])

                if response_text:
                    cognitive_time = time.time() - cognitive_start
                    print(f"🧠 COGNITIVE [{method}]: {response_text}")
                    print(f"   Processing time: {cognitive_time*1000:.0f}ms")

                    # 4. Post response to mailbox
                    mailbox.post_response(response_text)

                    # 5. TTS synthesis and playback
                    tts_start = time.time()
                    speak_with_piper(response_text)
                    tts_time = time.time() - tts_start

                    print(f"🔊 TTS synthesis: {tts_time*1000:.0f}ms")

                    # Total latency
                    total_latency = time.time() - exchange_start
                    latencies.append(total_latency)
                    exchange_count += 1

                    print(f"⏱️  END-TO-END LATENCY: {total_latency*1000:.0f}ms")
                    print(f"   (Transcription: {transcription_time*1000:.0f}ms + "
                          f"Cognitive: {cognitive_time*1000:.0f}ms + "
                          f"TTS: {tts_time*1000:.0f}ms)")

                    if exchange_count >= 5:
                        avg_latency = sum(latencies) / len(latencies)
                        print(f"\n📊 Average latency over {exchange_count} exchanges: {avg_latency*1000:.0f}ms")

                else:
                    print(f"💭 No pattern match - would need deeper processing")

        # Small sleep to avoid burning CPU
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\n" + "="*60)
    print("CONVERSATION ENDED")
    print("="*60)

    if latencies:
        print(f"\n📊 Final Statistics:")
        print(f"   Total exchanges: {exchange_count}")
        print(f"   Average latency: {sum(latencies)/len(latencies)*1000:.0f}ms")
        print(f"   Min latency: {min(latencies)*1000:.0f}ms")
        print(f"   Max latency: {max(latencies)*1000:.0f}ms")

    # Cognitive stats
    cog_stats = cognitive_router.get_stats()
    print(f"\n🧠 Cognitive Router Stats:")
    print(f"   Pattern hits: {cog_stats['pattern_hits']}")
    print(f"   Pattern misses: {cog_stats['pattern_misses']}")
    print(f"   Match rate: {cog_stats['pattern_hits']/(cog_stats['pattern_hits']+cog_stats['pattern_misses'])*100:.1f}%")

    print("\n✅ All systems stopped cleanly")
