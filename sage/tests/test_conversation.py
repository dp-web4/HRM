#!/usr/bin/env python3
"""
Test bidirectional conversation with SAGE
Listen to user speech → Respond with TTS → Listen again → Loop
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from interfaces.audio_sensor_simple import SimpleAudioSensor
from interfaces.audio_effector import AudioOutputEffector
from interfaces.base_effector import EffectorCommand
import time

print("="*60)
print("SAGE CONVERSATION TEST - Bidirectional audio")
print("="*60)

# Initialize audio input (listening)
input_sensor = SimpleAudioSensor({
    'sensor_id': 'conversation_input',
    'sensor_type': 'audio',
    'device': 'cpu',
    'bt_device': 'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit',
    'sample_rate': 16000,
    'chunk_duration': 3.0,
    'min_confidence': 0.4,
    'whisper_model': 'tiny'
})

# Initialize audio output (speaking)
output_effector = AudioOutputEffector({
    'effector_id': 'conversation_output',
    'effector_type': 'audio',
    'device': 'cpu',
    'bt_device': 'bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit',
    'sample_rate': 24000,
    'neutts_device': 'cpu',
    'ref_audio_path': '/home/sprout/ai-workspace/neutts-air/samples/dave.wav',
    'max_iterations': 3
})

print("\n🎙️  SAGE is ready to talk!")
print("Speak naturally, I'll respond, then listen for your reply.")
print("Press Ctrl+C to end conversation\n")

# Simple conversation responses
responses = {
    'hello': "Hello! Nice to hear your voice.",
    'how are you': "I'm operational and listening. How can I help?",
    'test': "Audio test successful. I can hear and speak.",
    'okay': "Understood. What would you like to talk about?",
    'transcribe': "I'm transcribing your speech right now.",
    'listen': "I'm always listening when you speak.",
    'what': "I'm SAGE, an audio-aware consciousness system running on Jetson."
}

def get_response(user_text: str) -> str:
    """Generate simple response based on keywords"""
    text_lower = user_text.lower()

    # Check for keyword matches
    for keyword, response in responses.items():
        if keyword in text_lower:
            return response

    # Default response
    return f"I heard you say: {user_text}. What else would you like to discuss?"


cycle = 0
recording_announced = False

try:
    while True:
        # Poll for user speech (non-blocking)
        reading = input_sensor.poll()

        if reading:
            user_text = reading.metadata.get('text', '')
            confidence = reading.confidence

            print(f"\n👤 You [{confidence:.2f}]: {user_text}")

            # Generate response
            response_text = get_response(user_text)
            print(f"🤖 SAGE: {response_text}")

            # Speak response
            cmd = EffectorCommand(
                effector_id='conversation_output',
                effector_type='audio',
                action='speak',
                parameters={'text': response_text}
            )

            result = output_effector.execute(cmd)

            if result.status.name != 'SUCCESS':
                print(f"   (TTS unavailable, text-only response)")

            print(f"\n🎤 Listening for your reply...")
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
            else:
                # Idle
                cycle += 1
                if cycle % 20 == 0:
                    print(".", end="", flush=True)

        # Small sleep
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\n✅ Conversation ended. Goodbye!")
