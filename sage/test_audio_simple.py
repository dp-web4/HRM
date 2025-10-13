#!/usr/bin/env python3
"""Simple audio test - just listen and print what we hear"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from interfaces.audio_sensor import AudioInputSensor

print("="*60)
print("SIMPLE AUDIO TEST - Listening for speech...")
print("="*60)

sensor = AudioInputSensor({
    'sensor_id': 'test_audio',
    'sensor_type': 'audio',
    'device': 'cpu',
    'bt_device': 'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit',
    'sample_rate': 16000,
    'chunk_duration': 2.0,
    'min_confidence': 0.4,
    'whisper_model': 'tiny'
})

print("\nListening... (speak now)")

try:
    while True:
        reading = sensor.poll()
        if reading:
            text = reading.metadata.get('text', '')
            conf = reading.confidence
            print(f"\n[{conf:.2f}] You said: {text}")
except KeyboardInterrupt:
    print("\n\nStopped.")
