#!/usr/bin/env python3
"""
Multi-Modal Attention Test
Two sensors competing for SAGE's attention:
- Audio: Speech events (rare, high value)
- Vision: Visual changes (frequent, variable value)

Questions:
1. Does SNARC switch attention between modalities?
2. Which sensor monopolizes attention?
3. Does exploration problem manifest?
4. Can it handle multi-sensory awareness?

This tests the core hypothesis: Can SAGE coordinate
multiple important sensors without getting stuck?
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

class SpeechSensor:
    """Audio: Rare but important speech events"""

    def __init__(self):
        self.messages = [
            "Emergency: System alert detected",
            "Hey there, how's it going?",
            "I need your help with something",
            "Status report please",
            "Thank you for your attention"
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

            # Importance based on content
            if 'emergency' in text.lower() or 'alert' in text.lower():
                importance = 0.95  # Critical
            elif 'help' in text.lower() or 'need' in text.lower():
                importance = 0.85  # Important
            elif 'status' in text.lower():
                importance = 0.75  # Significant
            else:
                importance = 0.65  # Normal

            return {
                'modality': 'audio',
                'type': 'speech',
                'text': text,
                'importance': importance
            }

        return None  # Silence

class VisionSensor:
    """Vision: More frequent, varying importance"""

    def __init__(self):
        self.events = [
            ('person_detected', 0.8, 'Person entered field of view'),
            ('motion_blur', 0.3, 'Camera movement detected'),
            ('face_recognized', 0.9, 'Known face identified'),
            ('object_moved', 0.5, 'Object position changed'),
            ('lighting_change', 0.2, 'Ambient lighting adjusted'),
            ('unknown_object', 0.7, 'New object appeared'),
            ('scene_stable', 0.1, 'No significant changes'),
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

            return {
                'modality': 'vision',
                'type': event_type,
                'description': description,
                'importance': importance
            }

        return None  # No change

def handle_audio(observation, stance):
    """Handle speech events"""
    if observation is None:
        return ExecutionResult(
            success=True,
            reward=0.1,
            description="Audio: Silence",
            outputs={'modality': 'audio', 'state': 'listening'}
        )

    text = observation['text']
    importance = observation['importance']

    print(f"\n🎤 SPEECH: \"{text}\"")
    print(f"   Importance: {importance:.2f}, Stance: {stance.value}")

    return ExecutionResult(
        success=True,
        reward=importance,
        description=f"Speech: {text}",
        outputs=observation
    )

def handle_vision(observation, stance):
    """Handle visual events"""
    if observation is None:
        return ExecutionResult(
            success=True,
            reward=0.15,
            description="Vision: No change",
            outputs={'modality': 'vision', 'state': 'monitoring'}
        )

    event_type = observation['type']
    description = observation['description']
    importance = observation['importance']

    # Visual events shown more quietly (not as verbose)
    print(f"👁️  VISION: {description} (importance: {importance:.2f})")

    return ExecutionResult(
        success=True,
        reward=importance,
        description=f"Vision: {event_type}",
        outputs=observation
    )

def main():
    print("=" * 70)
    print("MULTI-MODAL ATTENTION TEST")
    print("=" * 70)
    print("\nTwo sensors competing:")
    print("  🎤 Audio: Rare (every 5-12 cycles), high importance (0.65-0.95)")
    print("  👁️  Vision: Frequent (every 2-6 cycles), variable (0.1-0.9)")
    print("\nWill SNARC:")
    print("  - Switch attention between modalities?")
    print("  - Balance rare-important vs frequent-varied?")
    print("  - Get stuck on one sensor?")
    print("=" * 70)
    print()

    # Create sensors
    audio = SpeechSensor()
    vision = VisionSensor()

    # Create kernel
    sensor_sources = {
        'audio': audio,
        'vision': vision
    }
    action_handlers = {
        'audio': handle_audio,
        'vision': handle_vision
    }

    kernel = SAGEKernel(
        sensor_sources=sensor_sources,
        action_handlers=action_handlers,
        enable_logging=True  # Show attention allocation
    )

    print("Starting multi-modal awareness...\n")

    # Run for 50 cycles to see attention patterns
    kernel.run(max_cycles=50, cycle_delay=0.1)

    # Analysis
    print("\n" + "=" * 70)
    print("ATTENTION ANALYSIS")
    print("=" * 70)

    history = kernel.get_history()

    # Count focus distribution
    audio_focus = [h for h in history if h['focus_target'] == 'audio']
    vision_focus = [h for h in history if h['focus_target'] == 'vision']

    print(f"\nFocus distribution (50 cycles):")
    print(f"  Audio:  {len(audio_focus):2d} cycles ({100*len(audio_focus)/len(history):5.1f}%)")
    print(f"  Vision: {len(vision_focus):2d} cycles ({100*len(vision_focus)/len(history):5.1f}%)")

    # Events detected per modality
    audio_events = [h for h in audio_focus if 'text' in h['result'].outputs]
    vision_events = [h for h in vision_focus if 'type' in h['result'].outputs]

    print(f"\nEvents processed:")
    print(f"  Audio speech:  {len(audio_events)}")
    print(f"  Vision changes: {len(vision_events)}")

    # Attention switches
    switches = 0
    for i in range(1, len(history)):
        if history[i]['focus_target'] != history[i-1]['focus_target']:
            switches += 1

    print(f"\nAttention switches: {switches}")
    print(f"  (How often SAGE changed focus between modalities)")

    # Average rewards per modality
    audio_rewards = [h['result'].reward for h in audio_focus]
    vision_rewards = [h['result'].reward for h in vision_focus]

    if audio_rewards:
        print(f"\nAverage rewards:")
        print(f"  Audio:  {sum(audio_rewards)/len(audio_rewards):.3f}")
    if vision_rewards:
        print(f"  Vision: {sum(vision_rewards)/len(vision_rewards):.3f}")

    # Final SNARC weights
    stats = kernel.snarc.get_statistics()
    print(f"\nSNARC final weights:")
    for dim, weight in stats['current_weights'].items():
        change = ((weight - 0.2) / 0.2) * 100
        print(f"  {dim:10s}: {weight:.3f} ({change:+.1f}%)")

    # Timeline of attention (simplified)
    print(f"\nAttention timeline (first 30 cycles):")
    timeline = ""
    for i, h in enumerate(history[:30]):
        if h['focus_target'] == 'audio':
            if 'text' in h['result'].outputs:
                timeline += "🎤"  # Speech event
            else:
                timeline += "·"   # Silence
        else:
            if 'type' in h['result'].outputs:
                timeline += "👁️"  # Vision event
            else:
                timeline += "∘"   # No change

        if (i + 1) % 10 == 0:
            timeline += " "

    print(f"  {timeline}")
    print(f"  (🎤=speech, ·=audio silence, 👁️=vision event, ∘=vision stable)")

    print("\n" + "=" * 70)
    print("\nKEY QUESTIONS:")
    print("1. Did attention switch between modalities?")
    print("2. Did one sensor monopolize (exploration problem)?")
    print("3. How did rare-important vs frequent-varied play out?")
    print("4. What does this reveal about multi-sensory consciousness?")
    print()

if __name__ == "__main__":
    main()
