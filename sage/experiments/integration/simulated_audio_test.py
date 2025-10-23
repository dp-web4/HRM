#!/usr/bin/env python3
"""
Simulated Audio Test
Since we can't test with real microphone non-interactively,
create simulated speech events to test the full integration.

This lets us validate:
- SAGE kernel polls audio sensor correctly
- Speech detection triggers actions
- SNARC responds to speech vs silence
- Attention allocation works

Simulation:
- Random intervals of silence and speech
- Realistic confidence scores
- Variable text lengths
"""

import sys
import os
from pathlib import Path
import time
import random

hrm_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(hrm_root))
os.chdir(hrm_root)

from sage.core.sage_kernel import SAGEKernel, ExecutionResult
from sage.services.snarc.data_structures import CognitiveStance

class SimulatedAudioSensor:
    """Simulates speech detection with realistic timing"""

    def __init__(self):
        self.cycle_count = 0
        self.speech_queue = [
            "Hello, I'm testing the audio integration",
            "Can you hear me?",
            "This is a longer sentence to test variable length transcriptions",
            "Short test",
            "The SAGE kernel should respond differently based on confidence",
            "What happens with low confidence speech?",
            "Testing SNARC salience assessment",
            "Final test message"
        ]
        self.speech_index = 0
        self.speech_scheduled = None
        self.silence_cycles = 0

    def __call__(self):
        """Return speech detection or None (silence)"""
        self.cycle_count += 1

        # If no speech scheduled, randomly schedule some
        if self.speech_scheduled is None:
            # Random silence: 3-10 cycles
            self.silence_cycles = random.randint(3, 10)
            self.speech_scheduled = self.cycle_count + self.silence_cycles

        # If we've reached speech time
        if self.cycle_count >= self.speech_scheduled:
            # Get next speech
            if self.speech_index < len(self.speech_queue):
                text = self.speech_queue[self.speech_index]
                self.speech_index += 1

                # Realistic confidence (0.6-0.95)
                confidence = 0.6 + random.random() * 0.35

                # Realistic duration based on length
                duration = 2.0 + len(text) / 20.0

                # Reset for next speech
                self.speech_scheduled = None

                return {
                    'type': 'audio',
                    'text': text,
                    'confidence': confidence,
                    'duration': duration,
                    'timestamp': time.time()
                }
            else:
                # Ran out of speech, return None forever
                return None

        # Still in silence period
        return None

def handle_audio(observation, stance):
    """Handle audio observations"""
    if observation is None:
        # Silence
        return ExecutionResult(
            success=True,
            reward=0.2,  # Low reward for silence
            description="Silence detected",
            outputs={'silence': True}
        )

    # Speech detected
    text = observation['text']
    confidence = observation['confidence']
    duration = observation['duration']

    # Reward based on confidence
    reward = confidence

    print(f"\n🎤 SPEECH DETECTED:")
    print(f"   Text: \"{text}\"")
    print(f"   Confidence: {confidence:.2f}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Stance: {stance.value}")
    print(f"   Reward: {reward:.2f}")

    return ExecutionResult(
        success=True,
        reward=reward,
        description=f"Speech: {text[:30]}...",
        outputs=observation
    )

def main():
    print("=" * 70)
    print("SIMULATED AUDIO TEST")
    print("=" * 70)
    print("\nSimulating speech detection with realistic timing:")
    print("  - Random silence periods (3-10 cycles)")
    print("  - 8 speech events queued")
    print("  - Variable confidence (0.6-0.95)")
    print("  - Duration based on text length")
    print("\nWatching SNARC's response to speech vs silence...")
    print("=" * 70)
    print()

    # Create simulated audio sensor
    audio_sensor = SimulatedAudioSensor()

    # Create SAGE kernel
    sensor_sources = {'audio': audio_sensor}
    action_handlers = {'audio': handle_audio}

    kernel = SAGEKernel(
        sensor_sources=sensor_sources,
        action_handlers=action_handlers,
        enable_logging=True
    )

    print("Starting simulation...")
    print()

    # Run for 60 cycles (should get all 8 speech events)
    kernel.run(max_cycles=60, cycle_delay=0.2)

    # Analysis
    print("\n" + "=" * 70)
    print("SIMULATION ANALYSIS")
    print("=" * 70)

    history = kernel.get_history()

    # Count speech vs silence
    speech_count = 0
    silence_count = 0
    total_reward = 0

    for h in history:
        if 'silence' in h['result'].outputs:
            silence_count += 1
        else:
            speech_count += 1
        total_reward += h['result'].reward

    print(f"\nEvents detected:")
    print(f"  Speech:  {speech_count}")
    print(f"  Silence: {silence_count}")
    print(f"  Total:   {len(history)}")

    print(f"\nReward analysis:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average:      {total_reward/len(history):.3f}")

    # Salience evolution
    saliences = [h['salience_score'] for h in history]
    print(f"\nSalience:")
    print(f"  Min:  {min(saliences):.3f}")
    print(f"  Max:  {max(saliences):.3f}")
    print(f"  Avg:  {sum(saliences)/len(saliences):.3f}")

    # Stance distribution
    stances = {}
    for h in history:
        stance = h['stance']
        stances[stance] = stances.get(stance, 0) + 1

    print(f"\nStance distribution:")
    for stance, count in sorted(stances.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(history)
        print(f"  {stance:20s}: {count:2d} ({pct:5.1f}%)")

    print("\n" + "=" * 70)
    print("\nKEY QUESTIONS:")
    print("1. Did SNARC detect speech differently than silence?")
    print("2. How did salience change with speech events?")
    print("3. Did stance vary based on confidence?")
    print("4. What does this reveal about attention?")
    print()

if __name__ == "__main__":
    main()
