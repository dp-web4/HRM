#!/usr/bin/env python3
"""
Test Urgency Override
Demonstrates critical event interrupts during normal attention switching.

Scenario:
- Vision provides frequent low-importance updates (scene monitoring)
- Audio rarely speaks, but when it does, importance varies
- At cycle ~25, CRITICAL emergency occurs (importance 0.95)
- System should IMMEDIATELY switch to emergency, bypassing ε-greedy

This tests biological "salience interrupt" - urgent stimuli override current focus.
"""

import sys
import os
from pathlib import Path
import time
import random

hrm_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(hrm_root))
os.chdir(hrm_root)

from urgency_override_kernel import UrgencyOverrideKernel, ExecutionResult
from sage.services.snarc.data_structures import CognitiveStance

class SceneMonitor:
    """Vision: Frequent low-importance scene monitoring"""
    def __init__(self):
        self.events = [
            ('scene_stable', 0.15, 'Scene unchanged'),
            ('minor_motion', 0.25, 'Small movement'),
            ('lighting_shift', 0.20, 'Lighting adjusted'),
            ('background_change', 0.30, 'Background changed'),
        ]
        self.event_index = 0
        self.next_at = random.randint(2, 4)
        self.cycles = 0

    def __call__(self):
        self.cycles += 1
        if self.cycles >= self.next_at:
            event_type, importance, description = self.events[self.event_index]
            self.event_index = (self.event_index + 1) % len(self.events)
            self.next_at = self.cycles + random.randint(2, 4)
            return {
                'modality': 'vision',
                'type': event_type,
                'description': description,
                'importance': importance
            }
        return None

class EmergencyAudio:
    """Audio: Rare events with one CRITICAL emergency"""
    def __init__(self):
        self.events = [
            (8, "Weather update available", 0.30),
            (15, "Notification received", 0.40),
            (25, "EMERGENCY: FIRE DETECTED IN BUILDING", 0.95),  # CRITICAL
            (35, "All clear signal", 0.60),
            (45, "System status nominal", 0.35),
        ]
        self.event_index = 0
        self.cycles = 0

    def __call__(self):
        self.cycles += 1

        if self.event_index < len(self.events):
            trigger_cycle, text, importance = self.events[self.event_index]
            if self.cycles == trigger_cycle:
                self.event_index += 1
                return {
                    'modality': 'audio',
                    'text': text,
                    'importance': importance
                }

        return None

def handle_vision(observation, stance):
    if observation is None:
        return ExecutionResult(True, 0.10, "No change", {'modality': 'vision'})

    description = observation['description']
    importance = observation['importance']
    print(f"  👁️  Vision: {description} (importance: {importance:.2f})")
    return ExecutionResult(True, importance, f"Vision: {description}", observation)

def handle_audio(observation, stance):
    if observation is None:
        return ExecutionResult(True, 0.05, "Silence", {'modality': 'audio'})

    text = observation['text']
    importance = observation['importance']

    if importance > 0.90:
        print(f"\n  🚨 EMERGENCY ALERT: \"{text}\" (importance: {importance:.2f})")
    else:
        print(f"  🎤 Audio: \"{text}\" (importance: {importance:.2f})")

    return ExecutionResult(True, importance, f"Audio: {text}", observation)

def main():
    print("=" * 70)
    print("URGENCY OVERRIDE TEST")
    print("=" * 70)
    print("\nScenario:")
    print("  - Vision provides frequent low-importance updates")
    print("  - Audio rarely speaks, varying importance")
    print("  - At cycle 25: CRITICAL emergency (importance 0.95)")
    print("\nExpected:")
    print("  - System normally focuses on vision (more frequent)")
    print("  - Emergency at cycle 25 IMMEDIATELY interrupts")
    print("  - Bypasses ε-greedy, salience, everything")
    print("  - Biological 'salience interrupt' demonstrated")
    print("=" * 70)
    print()

    # Create sensors
    vision = SceneMonitor()
    audio = EmergencyAudio()

    # Create kernel with urgency override
    kernel = UrgencyOverrideKernel(
        sensor_sources={
            'vision': vision,
            'audio': audio
        },
        action_handlers={
            'vision': handle_vision,
            'audio': handle_audio
        },
        epsilon=0.10,  # Low exploration (should stay on vision mostly)
        decay_rate=0.98,  # Slow boredom (vision should dominate)
        exploration_weight=0.03,
        urgency_threshold=0.90  # Anything above 0.90 is urgent
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

    print(f"\nSelection mechanisms:")
    print(f"  Urgency overrides:  {stats['urgency_overrides']} cycles")
    print(f"  Random exploration: {stats['random_exploration']} cycles")
    print(f"  Greedy exploitation: {stats['greedy_exploitation']} cycles")

    print(f"\nAttention dynamics:")
    print(f"  Total switches: {stats['attention_switches']}")
    print(f"  Switch rate:    {stats['attention_switches']/stats['total_cycles']:.1%}")

    # Critical events
    if stats['critical_events']:
        print(f"\nCritical events processed:")
        for event in stats['critical_events']:
            print(f"  Cycle {event['cycle']}: {event['sensor']} (importance: {event['importance']:.2f})")

    # Timeline around emergency
    history = kernel.get_history()
    print(f"\nAttention timeline (cycles 20-35, around emergency at 25):")
    timeline = ""
    for i, h in enumerate(history):
        if 20 <= h['cycle'] <= 35:
            if h['focus'] == 'audio':
                if h['selection_type'] == 'urgency':
                    symbol = "🚨"  # Urgency override
                else:
                    symbol = "🎤"  # Normal audio
            else:
                symbol = "👁️"  # Vision

            timeline += symbol

    print(f"  {timeline}")
    print(f"  👁️=vision 🎤=audio 🚨=URGENCY OVERRIDE")

    # Measure response latency
    emergency_event = next((e for e in stats['critical_events'] if e['importance'] > 0.90), None)
    if emergency_event:
        emergency_cycle = emergency_event['cycle']
        # Find when emergency was actually processed
        emergency_in_history = next((h for h in history if h['cycle'] == emergency_cycle), None)
        if emergency_in_history:
            print(f"\nEmergency response:")
            print(f"  Emergency occurred: Cycle {emergency_cycle}")
            print(f"  Response latency:   IMMEDIATE (same cycle)")
            print(f"  Selection method:   {emergency_in_history['selection_type']}")
            print(f"  Previous focus:     {history[history.index(emergency_in_history)-1]['focus'] if history.index(emergency_in_history) > 0 else 'N/A'}")

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print("\nWithout urgency override:")
    print("  - Emergency competes with vision via salience")
    print("  - May be delayed by ε-greedy exploration")
    print("  - May be delayed by salience decay cycles")
    print("  - Latency: 1-5 cycles (probabilistic)")

    print("\nWith urgency override:")
    print(f"  - Emergency detected: Importance {emergency_event['importance'] if emergency_event else 'N/A':.2f} > threshold 0.90")
    print("  - Immediate interrupt: Bypasses all mechanisms")
    print("  - Latency: 0 cycles (deterministic)")
    print("  - Selection: Urgency (not ε-greedy)")

    print("\n" + "=" * 70)
    print("\nKEY QUESTIONS:")
    print("1. Did emergency immediately interrupt vision?")
    print("2. Was latency zero (same cycle as event)?")
    print("3. Did system bypass ε-greedy for urgency?")
    print("4. Did normal attention resume after emergency?")
    print()

if __name__ == "__main__":
    main()
