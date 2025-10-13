#!/usr/bin/env python3
"""
SAGE Conversation with Claude as Cognitive IRP

Architecture:
  User (speech) → SimpleAudioSensor → transcription.txt
  Claude (watches file) → thinks → response.txt
  SAGE (watches file) → AudioOutputEffector → speech

This script is the audio I/O loop. Claude watches transcription.txt and writes response.txt.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from interfaces.audio_sensor_simple import SimpleAudioSensor
from interfaces.audio_effector import AudioOutputEffector
from interfaces.base_effector import EffectorCommand
import time
import os

print("="*60)
print("SAGE + CLAUDE CONVERSATION")
print("User → Audio → Claude → Audio → User")
print("="*60)

# File paths for Claude communication
TRANSCRIPTION_FILE = "/tmp/sage_user_speech.txt"
RESPONSE_FILE = "/tmp/sage_claude_response.txt"

# Initialize audio I/O
input_sensor = SimpleAudioSensor({
    'sensor_id': 'claude_input',
    'sensor_type': 'audio',
    'device': 'cpu',
    'bt_device': 'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit',
    'sample_rate': 16000,
    'chunk_duration': 3.0,
    'min_confidence': 0.4,
    'whisper_model': 'tiny'
})

output_effector = AudioOutputEffector({
    'effector_id': 'claude_output',
    'effector_type': 'audio',
    'device': 'cpu',
    'bt_device': 'bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit',
    'sample_rate': 24000,
    'neutts_device': 'cpu',
    'ref_audio_path': '/home/sprout/ai-workspace/neutts-air/samples/dave.wav',
    'max_iterations': 3
})

print("\n🎙️  SAGE audio system ready")
print(f"📝 User speech → {TRANSCRIPTION_FILE}")
print(f"💭 Claude response → {RESPONSE_FILE}")
print("\nWaiting for conversation...\n")

# State
last_response_mtime = 0
cycle = 0
recording_announced = False

try:
    while True:
        # Poll for user speech
        reading = input_sensor.poll()

        if reading:
            user_text = reading.metadata.get('text', '')
            confidence = reading.confidence

            print(f"\n👤 USER [{confidence:.2f}]: {user_text}")

            # Write to file for Claude
            with open(TRANSCRIPTION_FILE, 'w') as f:
                f.write(f"{user_text}\n")
                f.write(f"confidence: {confidence}\n")
                f.write(f"timestamp: {time.time()}\n")

            print(f"💭 Waiting for Claude to respond...")

            # Wait for Claude's response
            response_timeout = 30.0
            start_wait = time.time()

            while True:
                if os.path.exists(RESPONSE_FILE):
                    mtime = os.path.getmtime(RESPONSE_FILE)
                    if mtime > last_response_mtime:
                        # New response available
                        with open(RESPONSE_FILE, 'r') as f:
                            response_text = f.read().strip()

                        if response_text:
                            last_response_mtime = mtime
                            print(f"🤖 SAGE: {response_text}")

                            # Speak response
                            cmd = EffectorCommand(
                                effector_id='claude_output',
                                effector_type='audio',
                                action='speak',
                                parameters={'text': response_text}
                            )

                            result = output_effector.execute(cmd)

                            if result.status.name != 'SUCCESS':
                                print(f"   ⚠️ TTS failed: {result.message}")

                            print(f"\n🎤 Listening for your reply...\n")
                            break

                # Check timeout
                if time.time() - start_wait > response_timeout:
                    print(f"   ⚠️ Claude response timeout")
                    break

                time.sleep(0.1)

            recording_announced = False

        else:
            # Show listening status
            if input_sensor.recording_proc is not None:
                if not recording_announced:
                    recording_announced = True
                    cycle = 0
                cycle += 1
                if cycle % 10 == 0:
                    elapsed = time.time() - input_sensor.recording_start
                    print(f"  Recording: {elapsed:.1f}s / {input_sensor.chunk_duration}s", end="\r", flush=True)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n✅ Conversation ended")
