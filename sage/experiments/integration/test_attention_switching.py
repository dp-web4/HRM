#!/usr/bin/env python3
"""
Test Attention Switching
Same multi-modal scenario as before, but with new kernel.

Expected: Both modalities get attention, switches occur, balanced awareness.
"""

import sys
import os
from pathlib import Path
import time
import random

hrm_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(hrm_root))
os.chdir(hrm_root)

# Import the switching kernel
from attention_switching_kernel import AttentionSwitchingKernel, ExecutionResult
from sage.services.snarc.data_structures import CognitiveStance

# Reuse sensors from previous test
class SpeechSensor:
    """Audio: Rare but important"""
    def __init__(self):
        self.messages = [
            "Emergency alert detected",
            "Hello, how are you?",
            "I need assistance",
            "Status report please",
            "Thank you"
        ]
        self.index = 0
        self.next_at = random.randint(5, 10)
        self.cycles = 0

    def __call__(self):
        self.cycles += 1
        if self.cycles >= self.next_at and self.index < len(self.messages):
            text = self.messages[self.index]
            self.index += 1
            self.next_at = self.cycles + random.randint(5, 12)

            importance = 0.9 if 'emergency' in text.lower() else 0.7
            return {'modality': 'audio', 'text': text, 'importance': importance}
        return None

class VisionSensor:
    """Vision: More frequent, varied"""
    def __init__(self):
        self.events = [
            ('person_detected', 0.8, 'Person entered view'),
            ('motion', 0.3, 'Motion detected'),
            ('face_recognized', 0.9, 'Face recognized'),
            ('object_moved', 0.5, 'Object moved'),
            ('lighting_change', 0.2, 'Lighting changed'),
        ]
        self.event_index = 0
        self.next_at = random.randint(2, 5)
        self.cycles = 0

    def __call__(self):
        self.cycles += 1
        if self.cycles >= self.next_at:
            event_type, importance, description = self.events[self.event_index]
            self.event_index = (self.event_index + 1) % len(self.events)
            self.next_at = self.cycles + random.randint(2, 6)
            return {'modality': 'vision', 'type': event_type, 'description': description, 'importance': importance}
        return None

def handle_audio(observation, stance):
    if observation is None:
        return ExecutionResult(True, 0.1, "Silence", {'modality': 'audio'})

    text = observation['text']
    importance = observation['importance']
    print(f"  🎤 Audio: \"{text}\" (importance: {importance:.2f})")
    return ExecutionResult(True, importance, f"Speech: {text}", observation)

def handle_vision(observation, stance):
    if observation is None:
        return ExecutionResult(True, 0.15, "No change", {'modality': 'vision'})

    description = observation['description']
    importance = observation['importance']
    print(f"  👁️  Vision: {description} (importance: {importance:.2f})")
    return ExecutionResult(True, importance, f"Vision: {description}", observation)

def main():
    print("=" * 70)
    print("ATTENTION SWITCHING TEST")
    print("=" * 70)
    print("\nTesting: Does new kernel balance audio + vision?")
    print("\nMechanisms:")
    print("  1. ε-greedy: 15% random exploration")
    print("  2. Salience decay: Focus loses 3% salience per cycle")
    print("  3. Exploration bonus: Less-visited sensors more attractive")
    print("  4. Fresh assessment: All sensors re-evaluated each cycle")
    print("=" * 70)
    print()

    # Create sensors
    audio = SpeechSensor()
    vision = VisionSensor()

    # Create switching kernel
    kernel = AttentionSwitchingKernel(
        sensor_sources={'audio': audio, 'vision': vision},
        action_handlers={'audio': handle_audio, 'vision': handle_vision},
        epsilon=0.15,
        decay_rate=0.97,
        exploration_weight=0.05
    )

    # Run test
    kernel.run(max_cycles=50, cycle_delay=0.05)

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    stats = kernel.get_statistics()

    print(f"\nFocus distribution:")
    for sensor, count in stats['focus_distribution'].items():
        pct = 100 * count / stats['total_cycles']
        print(f"  {sensor:10s}: {count:2d} cycles ({pct:5.1f}%)")

    print(f"\nExploration vs Exploitation:")
    print(f"  Random explore:  {stats['exploration_cycles']} cycles")
    print(f"  Greedy exploit:  {stats['exploitation_cycles']} cycles")

    print(f"\nAttention dynamics:")
    print(f"  Total switches: {stats['attention_switches']}")
    print(f"  Switch rate:    {stats['attention_switches']/stats['total_cycles']:.1%}")

    # Timeline
    history = kernel.get_history()
    print(f"\nAttention timeline (first 30):")
    timeline = ""
    for i, h in enumerate(history[:30]):
        symbol = "🎤" if h['focus'] == 'audio' else "👁️"
        if h['exploration_type'] == 'random':
            symbol = f"[{symbol}]"  # Brackets = random exploration
        timeline += symbol
        if (i + 1) % 10 == 0:
            timeline += " "

    print(f"  {timeline}")
    print(f"  🎤=audio 👁️=vision []=random explore")

    print("\n" + "=" * 70)
    print("COMPARISON TO ORIGINAL")
    print("=" * 70)
    print("\nOriginal kernel:")
    print("  Audio:  50 cycles (100%)")
    print("  Vision:  0 cycles (  0%)")
    print("  Switches: 0")
    print("\nSwitching kernel:")
    for sensor, count in stats['focus_distribution'].items():
        pct = 100 * count / stats['total_cycles']
        print(f"  {sensor.capitalize():7s}: {count:2d} cycles ({pct:5.1f}%)")
    print(f"  Switches: {stats['attention_switches']}")

    print("\n" + "=" * 70)
    print("\nKEY QUESTION:")
    print("Did attention switching enable multi-modal awareness?")
    print()

if __name__ == "__main__":
    main()
