#!/usr/bin/env python3
"""
Audio Detection Test - Phase 1
Simplest possible: Can SAGE kernel detect speech?

Setup:
- AudioInputIRP as sensor (existing, tested)
- SAGE kernel polls it
- Just print when speech detected
- No response, no intelligence

Goal: Verify the connection works
"""

import sys
import os
from pathlib import Path

hrm_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(hrm_root))
os.chdir(hrm_root)

from sage.core.sage_kernel import SAGEKernel, ExecutionResult
from sage.services.snarc.data_structures import CognitiveStance

# Check if we can import audio plugin
try:
    # Import audio plugin directly
    import importlib.util
    audio_plugin_path = hrm_root / "sage/irp/plugins/audio_input_impl.py"
    spec = importlib.util.spec_from_file_location("audio_input_impl", audio_plugin_path)
    audio_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audio_module)
    AudioInputIRP = audio_module.AudioInputIRP
    print("✅ AudioInputIRP loaded successfully")
except Exception as e:
    print(f"❌ Failed to load AudioInputIRP: {e}")
    print("\nThis test requires:")
    print("- Microphone available")
    print("- whisper package installed")
    print("- parecord utility (PulseAudio)")
    sys.exit(1)

class AudioSensor:
    """Wraps AudioInputIRP for SAGE kernel"""

    def __init__(self, config):
        self.irp = AudioInputIRP(config)
        self.state = None
        self.history = []

    def __call__(self):
        """Poll audio - returns transcription if available, None otherwise"""
        # Initialize if first call
        if self.state is None:
            self.state = self.irp.init_state(None, {'mode': 'continuous'})
            self.history = [self.state]

        # Check if we should halt (already have good transcription)
        if self.irp.halt(self.history):
            # Extract and return result
            result = self.irp.extract(self.state)

            # Reset for next listening session
            self.state = None
            self.history = []

            return result

        # Continue refining
        self.state = self.irp.step(self.state)
        self.state.energy_val = self.irp.energy(self.state)
        self.history.append(self.state)

        return None  # Still listening

def handle_speech(observation, stance):
    """Handle detected speech"""
    text = observation['text']
    confidence = observation['confidence']
    duration = observation['duration']

    print(f"\n🎤 DETECTED SPEECH:")
    print(f"   Text: \"{text}\"")
    print(f"   Confidence: {confidence:.2f}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Stance: {stance.value}")

    # Return success based on confidence
    reward = confidence

    return ExecutionResult(
        success=True,
        reward=reward,
        description=f"Speech detected: {len(text)} chars",
        outputs=observation
    )

def main():
    print("=" * 70)
    print("AUDIO DETECTION TEST - Phase 1")
    print("=" * 70)
    print("\nGoal: Verify SAGE can detect speech via AudioInputIRP")
    print("\nSetup:")
    print("  - Microphone will listen continuously")
    print("  - When speech detected, prints transcription")
    print("  - No response yet - just detection")
    print("\nPress Ctrl+C to stop")
    print("=" * 70)
    print()

    # Check if microphone is available
    print("Checking audio setup...")

    # Configure audio sensor
    audio_config = {
        'entity_id': 'test_audio',
        'device': 'default',  # Let it find default mic
        'whisper_model': 'tiny',  # Fast model
        'chunk_duration': 2.0,  # 2 second chunks
        'min_confidence': 0.5,  # Accept moderate confidence
        'max_duration': 10.0  # Max 10 seconds per utterance
    }

    try:
        audio_sensor = AudioSensor(audio_config)
        print("✅ Audio sensor initialized")
    except Exception as e:
        print(f"❌ Failed to initialize audio: {e}")
        print("\nTroubleshooting:")
        print("  - Check if microphone is connected")
        print("  - Verify parecord is installed")
        print("  - Test with: parecord --channels=1 --rate=16000 test.wav")
        return

    # Create SAGE kernel
    sensor_sources = {'audio': audio_sensor}
    action_handlers = {'audio': handle_speech}

    kernel = SAGEKernel(
        sensor_sources=sensor_sources,
        action_handlers=action_handlers,
        enable_logging=True
    )

    print("\n🎧 SAGE is now listening...")
    print("   Speak into your microphone")
    print("   Speech will be detected and transcribed")
    print()

    try:
        # Run indefinitely
        kernel.run(cycle_delay=0.1)  # Fast polling
    except KeyboardInterrupt:
        print("\n\n⏸️ Stopped by user")
        print("\nSession complete!")

if __name__ == "__main__":
    main()
