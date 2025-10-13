#!/usr/bin/env python3
"""Test simple audio sensor - non-blocking state machine"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from interfaces.audio_sensor_simple import SimpleAudioSensor
import time

print("="*60)
print("SIMPLE AUDIO TEST - Non-blocking capture")
print("="*60)

sensor = SimpleAudioSensor({
    'sensor_id': 'test_simple',
    'sensor_type': 'audio',
    'device': 'cpu',
    'bt_device': 'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit',
    'sample_rate': 16000,
    'chunk_duration': 3.0,  # 3 second chunks
    'min_confidence': 0.4,
    'whisper_model': 'tiny'
})

print("\n🎤 Listening for 3-second chunks...")
print("Speak when you see 'Recording...', then wait for transcription.")
print("Press Ctrl+C to stop\n")

cycle = 0
recording_announced = False

try:
    while True:
        # Non-blocking poll
        reading = sensor.poll()

        if reading:
            text = reading.metadata.get('text', '')
            conf = reading.confidence
            print(f"\n[{conf:.2f}] You said: {text}\n")
            recording_announced = False
        else:
            # Check if we're recording
            if sensor.recording_proc is not None:
                if not recording_announced:
                    print("Recording... (speak now)")
                    recording_announced = True
                    cycle = 0
                cycle += 1
                if cycle % 10 == 0:
                    elapsed = time.time() - sensor.recording_start
                    print(f"  {elapsed:.1f}s / {sensor.chunk_duration}s", end="\r", flush=True)
            else:
                # Idle - show we're not blocking
                cycle += 1
                if cycle % 20 == 0:
                    print(".", end="", flush=True)

        # Small sleep to avoid burning CPU
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n✅ Stopped")
