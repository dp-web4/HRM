#!/usr/bin/env python3
"""
Three-Modality Attention Test
Tests whether attention switching scales to 3+ modalities.

Modalities:
- Audio: Rare (every 8-15 cycles), high importance (0.7-0.95)
- Vision: Frequent (every 2-5 cycles), variable (0.2-0.9)
- Tactile: Moderate (every 4-8 cycles), moderate (0.4-0.7)

Questions:
1. Does attention switching scale to 3 modalities?
2. How does attention distribute across different frequencies?
3. Does each modality get appropriate attention?
4. What is the switch pattern with more options?
"""

import sys
import os
from pathlib import Path
import time
import random

hrm_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(hrm_root))
os.chdir(hrm_root)

from attention_switching_kernel import AttentionSwitchingKernel, ExecutionResult
from sage.services.snarc.data_structures import CognitiveStance

class SpeechSensor:
    """Audio: Rare but important"""
    def __init__(self):
        self.messages = [
            ("Emergency system alert", 0.95),
            ("Hello there", 0.70),
            ("I need your help", 0.85),
            ("Status update needed", 0.75),
            ("Thank you", 0.65)
        ]
        self.index = 0
        self.next_at = random.randint(8, 15)
        self.cycles = 0

    def __call__(self):
        self.cycles += 1
        if self.cycles >= self.next_at and self.index < len(self.messages):
            text, importance = self.messages[self.index]
            self.index += 1
            self.next_at = self.cycles + random.randint(8, 15)
            return {
                'modality': 'audio',
                'text': text,
                'importance': importance
            }
        return None

class VisionSensor:
    """Vision: Frequent, variable importance"""
    def __init__(self):
        self.events = [
            ('person_detected', 0.80, 'Person in view'),
            ('motion_blur', 0.25, 'Camera moved'),
            ('face_recognized', 0.90, 'Face identified'),
            ('object_moved', 0.50, 'Object shifted'),
            ('lighting_dim', 0.20, 'Light decreased'),
            ('new_object', 0.70, 'Unknown object'),
            ('scene_stable', 0.15, 'No change'),
        ]
        self.event_index = 0
        self.next_at = random.randint(2, 5)
        self.cycles = 0

    def __call__(self):
        self.cycles += 1
        if self.cycles >= self.next_at:
            event_type, importance, description = self.events[self.event_index]
            self.event_index = (self.event_index + 1) % len(self.events)
            self.next_at = self.cycles + random.randint(2, 5)
            return {
                'modality': 'vision',
                'type': event_type,
                'description': description,
                'importance': importance
            }
        return None

class TactileSensor:
    """Tactile: Moderate frequency and importance"""
    def __init__(self):
        self.events = [
            ('contact', 0.75, 'Surface contact detected'),
            ('pressure', 0.60, 'Pressure applied'),
            ('vibration', 0.55, 'Vibration sensed'),
            ('temperature', 0.45, 'Temperature change'),
            ('texture', 0.50, 'Texture detected'),
            ('release', 0.40, 'Contact released'),
        ]
        self.event_index = 0
        self.next_at = random.randint(4, 8)
        self.cycles = 0

    def __call__(self):
        self.cycles += 1
        if self.cycles >= self.next_at:
            event_type, importance, description = self.events[self.event_index]
            self.event_index = (self.event_index + 1) % len(self.events)
            self.next_at = self.cycles + random.randint(4, 8)
            return {
                'modality': 'tactile',
                'type': event_type,
                'description': description,
                'importance': importance
            }
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
        return ExecutionResult(True, 0.12, "No change", {'modality': 'vision'})

    description = observation['description']
    importance = observation['importance']
    print(f"  👁️  Vision: {description} (importance: {importance:.2f})")
    return ExecutionResult(True, importance, f"Vision: {description}", observation)

def handle_tactile(observation, stance):
    if observation is None:
        return ExecutionResult(True, 0.08, "No contact", {'modality': 'tactile'})

    description = observation['description']
    importance = observation['importance']
    print(f"  ✋ Tactile: {description} (importance: {importance:.2f})")
    return ExecutionResult(True, importance, f"Touch: {description}", observation)

def main():
    print("=" * 70)
    print("THREE-MODALITY ATTENTION TEST")
    print("=" * 70)
    print("\nTesting: Does attention switching scale to 3+ modalities?")
    print("\nModalities:")
    print("  🎤 Audio: Rare (every 8-15 cycles), high (0.7-0.95)")
    print("  👁️  Vision: Frequent (every 2-5 cycles), variable (0.2-0.9)")
    print("  ✋ Tactile: Moderate (every 4-8 cycles), moderate (0.4-0.75)")
    print("\nMechanisms:")
    print("  1. ε-greedy: 15% random exploration")
    print("  2. Salience decay: Focus loses 3% per cycle")
    print("  3. Exploration bonus: Less-visited more attractive")
    print("  4. Fresh assessment: All sensors re-evaluated each cycle")
    print("=" * 70)
    print()

    # Create sensors
    audio = SpeechSensor()
    vision = VisionSensor()
    tactile = TactileSensor()

    # Create kernel
    kernel = AttentionSwitchingKernel(
        sensor_sources={
            'audio': audio,
            'vision': vision,
            'tactile': tactile
        },
        action_handlers={
            'audio': handle_audio,
            'vision': handle_vision,
            'tactile': handle_tactile
        },
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

    # Switch pattern analysis
    history = kernel.get_history()

    # Count switches between each pair
    switches = {'audio→vision': 0, 'vision→audio': 0,
                'audio→tactile': 0, 'tactile→audio': 0,
                'vision→tactile': 0, 'tactile→vision': 0}

    for i in range(1, len(history)):
        prev = history[i-1]['focus']
        curr = history[i]['focus']
        if prev != curr:
            key = f"{prev}→{curr}"
            if key in switches:
                switches[key] += 1

    print(f"\nSwitch patterns:")
    for pair, count in switches.items():
        if count > 0:
            print(f"  {pair}: {count}")

    # Timeline (first 40)
    print(f"\nAttention timeline (first 40 cycles):")
    timeline = ""
    for i, h in enumerate(history[:40]):
        if h['focus'] == 'audio':
            symbol = "🎤"
        elif h['focus'] == 'vision':
            symbol = "👁️"
        else:
            symbol = "✋"

        if h['exploration_type'] == 'random':
            symbol = f"[{symbol}]"

        timeline += symbol
        if (i + 1) % 10 == 0:
            timeline += " "

    print(f"  {timeline}")
    print(f"  🎤=audio 👁️=vision ✋=tactile []=random")

    # Detailed events
    audio_events = [h for h in history if h['focus'] == 'audio' and 'text' in h['result'].outputs]
    vision_events = [h for h in history if h['focus'] == 'vision' and 'type' in h['result'].outputs]
    tactile_events = [h for h in history if h['focus'] == 'tactile' and 'type' in h['result'].outputs]

    print(f"\nEvents processed:")
    print(f"  Audio messages:   {len(audio_events)}")
    print(f"  Vision events:    {len(vision_events)}")
    print(f"  Tactile contacts: {len(tactile_events)}")

    print("\n" + "=" * 70)
    print("SCALING ANALYSIS")
    print("=" * 70)

    print("\nWith 2 modalities (audio + vision):")
    print("  Vision: 70.6%, Audio: 29.4%")
    print("  Switches: 10 in 17 cycles (58.8%)")

    print(f"\nWith 3 modalities (audio + vision + tactile):")
    audio_pct = 100 * stats['focus_distribution'].get('audio', 0) / stats['total_cycles']
    vision_pct = 100 * stats['focus_distribution'].get('vision', 0) / stats['total_cycles']
    tactile_pct = 100 * stats['focus_distribution'].get('tactile', 0) / stats['total_cycles']
    print(f"  Vision: {vision_pct:.1f}%, Audio: {audio_pct:.1f}%, Tactile: {tactile_pct:.1f}%")
    print(f"  Switches: {stats['attention_switches']} in {stats['total_cycles']} cycles ({stats['attention_switches']/stats['total_cycles']:.1%})")

    print("\n" + "=" * 70)
    print("\nKEY QUESTIONS:")
    print("1. Does attention switching scale to 3 modalities? (all get attention)")
    print("2. Is distribution proportional to frequency + importance?")
    print("3. Are high-importance events from all modalities captured?")
    print("4. What is the switching overhead with more options?")
    print()

if __name__ == "__main__":
    main()
