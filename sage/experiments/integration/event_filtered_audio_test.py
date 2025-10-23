#!/usr/bin/env python3
"""
Event-Filtered Audio Test
Hypothesis: SNARC down-weights reward because it sees mostly silence.

Test: Only send SPEECH events to SNARC, filter out silence.

Expected: Reward weight should maintain or increase (speech valuable).

This tests whether event-based assessment works better than
continuous assessment with negative examples.
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

class EventFilteredAudioSensor:
    """
    Only returns observations when speech detected.
    Returns None during silence (no observation = no SNARC assessment)
    """

    def __init__(self):
        self.cycle_count = 0
        self.speech_queue = [
            "Hello, testing event-filtered audio",
            "This approach only sends speech to SNARC",
            "Silence periods are filtered out",
            "Will reward weight increase now?",
            "Testing salience with event-based assessment",
            "High confidence speech should be valued",
            "Low confidence speech still matters",
            "Final test of filtered approach"
        ]
        self.speech_index = 0
        self.next_speech_at = None

    def __call__(self):
        """Return speech or None (filtered silence)"""
        self.cycle_count += 1

        # Schedule next speech if needed
        if self.next_speech_at is None:
            # Random interval: 3-10 cycles
            self.next_speech_at = self.cycle_count + random.randint(3, 10)

        # Check if it's time for speech
        if self.cycle_count >= self.next_speech_at:
            if self.speech_index < len(self.speech_queue):
                text = self.speech_queue[self.speech_index]
                self.speech_index += 1

                # Realistic confidence
                confidence = 0.6 + random.random() * 0.35

                # Duration based on length
                duration = 2.0 + len(text) / 20.0

                # Schedule next
                self.next_speech_at = None

                return {
                    'type': 'audio',
                    'text': text,
                    'confidence': confidence,
                    'duration': duration,
                    'timestamp': time.time()
                }

        # Return None = no event = no observation = SNARC skips this cycle
        return None

def handle_speech(observation, stance):
    """Handle speech events or None"""
    # Handle None case (shouldn't happen with filtered sensor, but defensive)
    if observation is None or 'text' not in observation:
        return ExecutionResult(
            success=True,
            reward=0.1,  # Very low reward for no-op
            description="No event",
            outputs={}
        )

    text = observation['text']
    confidence = observation['confidence']
    duration = observation['duration']

    # Reward = confidence (value of this speech)
    reward = confidence

    print(f"\n🎤 EVENT #{handle_speech.event_count}:")
    print(f"   Text: \"{text}\"")
    print(f"   Confidence: {confidence:.2f}")
    print(f"   Stance: {stance.value}")
    print(f"   Reward: {reward:.2f}")

    handle_speech.event_count += 1

    return ExecutionResult(
        success=True,
        reward=reward,
        description=f"Speech event: {text[:30]}",
        outputs=observation
    )

# Track event count
handle_speech.event_count = 1

def main():
    print("=" * 70)
    print("EVENT-FILTERED AUDIO TEST")
    print("=" * 70)
    print("\nDifference from previous test:")
    print("  - Silence returns None (no observation)")
    print("  - SNARC only assesses SPEECH events")
    print("  - No low-reward silence diluting signal")
    print("\nHypothesis: Reward weight should increase (not decrease)")
    print("=" * 70)
    print()

    # Create filtered sensor
    audio_sensor = EventFilteredAudioSensor()

    # Create kernel
    sensor_sources = {'audio': audio_sensor}
    action_handlers = {'audio': handle_speech}

    kernel = SAGEKernel(
        sensor_sources=sensor_sources,
        action_handlers=action_handlers,
        enable_logging=True
    )

    print("Starting filtered simulation...")
    print("(Only speech events will be shown)")
    print()

    # Run for same 60 cycles
    kernel.run(max_cycles=60, cycle_delay=0.1)

    # Analysis
    print("\n" + "=" * 70)
    print("FILTERED SIMULATION ANALYSIS")
    print("=" * 70)

    history = kernel.get_history()

    print(f"\nEvents processed by SNARC: {len(history)}")
    print(f"(vs 60 in unfiltered test)")

    if len(history) > 0:
        # All events are speech (no silence)
        rewards = [h['result'].reward for h in history]

        print(f"\nReward analysis:")
        print(f"  Min:     {min(rewards):.3f}")
        print(f"  Max:     {max(rewards):.3f}")
        print(f"  Average: {sum(rewards)/len(rewards):.3f}")
        print(f"  Total:   {sum(rewards):.2f}")

        # Salience
        saliences = [h['salience_score'] for h in history]
        print(f"\nSalience:")
        print(f"  Min:  {min(saliences):.3f}")
        print(f"  Max:  {max(saliences):.3f}")
        print(f"  Avg:  {sum(saliences)/len(saliences):.3f}")

        # SNARC weight evolution
        print(f"\nSNARC Statistics:")
        stats = kernel.snarc.get_statistics()
        print(f"  Assessments: {stats['num_assessments']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")

        print(f"\nFinal weights:")
        for dim, weight in stats['current_weights'].items():
            initial = 0.2
            change = ((weight - initial) / initial) * 100
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            print(f"  {dim:10s}: {weight:.3f} ({arrow} {abs(change):+.1f}%)")

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
    print("\nCOMPARISON TO UNFILTERED TEST:")
    print("\nUnfiltered (speech + silence):")
    print("  Events: 60 (7 speech, 53 silence)")
    print("  Avg reward: 0.275")
    print("  Reward weight: 0.200 → 0.119 (DOWN 40%)")
    print("\nFiltered (speech only):")
    print(f"  Events: {len(history)} (all speech)")
    if len(history) > 0:
        print(f"  Avg reward: {sum(rewards)/len(rewards):.3f}")
        reward_weight_final = stats['current_weights']['reward']
        reward_change = ((reward_weight_final - 0.2) / 0.2) * 100
        print(f"  Reward weight: 0.200 → {reward_weight_final:.3f} ({reward_change:+.1f}%)")
    print("\n" + "=" * 70)
    print("\nKEY QUESTION:")
    print("Did filtering fix the reward weight problem?")
    print()

if __name__ == "__main__":
    main()
