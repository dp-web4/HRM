#!/usr/bin/env python3
"""Test streaming audio sensor - non-blocking listening"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from interfaces.audio_sensor_streaming import StreamingAudioSensor
import time

print("="*60)
print("STREAMING AUDIO TEST - Continuous listening")
print("="*60)

sensor = StreamingAudioSensor({
    'sensor_id': 'test_streaming',
    'sensor_type': 'audio',
    'device': 'cpu',
    'bt_device': 'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit',
    'sample_rate': 16000,
    'chunk_duration': 1.0,      # 1 second chunks
    'buffer_duration': 3.0,      # Transcribe every 3 seconds of audio
    'min_confidence': 0.4,
    'whisper_model': 'tiny'
})

print("\n🎧 Listening continuously in background...")
print("Speak naturally - if I'm busy transcribing and miss something, it's gone.")
print("Press Ctrl+C to stop\n")

cycle = 0
try:
    while True:
        # Non-blocking poll
        reading = sensor.poll()

        if reading:
            text = reading.metadata.get('text', '')
            conf = reading.confidence
            duration = reading.metadata.get('duration', 0)
            print(f"\n[{conf:.2f}] ({duration:.1f}s) You said: {text}")
        else:
            # Show we're not blocking
            cycle += 1
            if cycle % 10 == 0:
                print(".", end="", flush=True)

        # Small sleep to avoid burning CPU
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n✅ Stopped listening")
