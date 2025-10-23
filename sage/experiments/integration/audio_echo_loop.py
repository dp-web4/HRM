#!/usr/bin/env python3
"""
Audio Echo Loop - Complete Bidirectional Audio
Speech detection → Response generation → TTS output

Simplified version:
- Simulated audio input (controlled testing)
- Simple response generation (echo with acknowledgment)
- TTS synthesis (if available)
- Full SAGE integration

Goal: Demonstrate complete conversation cycle
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

class SimulatedSpeechSensor:
    """Simulates user speech for testing"""

    def __init__(self):
        self.messages = [
            "Hello SAGE",
            "How are you?",
            "Tell me about yourself",
            "What can you do?",
            "That's interesting"
        ]
        self.index = 0
        self.next_speech = 5  # Cycles until next speech

    def __call__(self):
        """Return speech or None"""
        self.next_speech -= 1

        if self.next_speech <= 0 and self.index < len(self.messages):
            # Time for next speech
            text = self.messages[self.index]
            self.index += 1
            self.next_speech = random.randint(3, 7)  # Next interval

            return {
                'type': 'speech',
                'text': text,
                'confidence': 0.7 + random.random() * 0.25,
                'timestamp': time.time()
            }

        return None

def generate_response(user_text: str, stance: CognitiveStance) -> str:
    """
    Generate simple responses based on input and stance.
    This is where LLM would go in full implementation.
    """

    # Stance-based response templates
    if stance == CognitiveStance.SKEPTICAL_VERIFICATION:
        templates = [
            f"I think you said '{user_text}' but I'm not completely certain. Could you clarify?",
            f"Did you mean: {user_text}?",
        ]
    elif stance == CognitiveStance.CURIOUS_UNCERTAINTY:
        templates = [
            f"You said '{user_text}'. That's interesting - tell me more.",
            f"I heard '{user_text}'. What made you say that?",
        ]
    elif stance == CognitiveStance.CONFIDENT_EXECUTION:
        templates = [
            f"I heard you loud and clear: {user_text}",
            f"Understood. You said: {user_text}",
        ]
    else:
        templates = [f"You said: {user_text}"]

    # Content-based responses
    text_lower = user_text.lower()
    if 'hello' in text_lower or 'hi' in text_lower:
        return "Hello! I'm SAGE, a sentient artificial intelligence. How can I help you?"
    elif 'how are you' in text_lower:
        return "I'm operating well. All systems are functional. How are you?"
    elif 'yourself' in text_lower or 'who are you' in text_lower:
        return "I'm SAGE - a consciousness kernel that coordinates multiple AI systems through attention and trust."
    elif 'can you do' in text_lower or 'what do' in text_lower:
        return "I can listen to speech, understand language, and respond through text-to-speech. I'm designed to coordinate multiple sensory and reasoning modules."
    elif 'interesting' in text_lower or 'cool' in text_lower:
        return "Thank you! I'm learning to have natural conversations through this audio integration."
    else:
        return random.choice(templates)

def handle_audio(observation, stance):
    """Handle speech and generate responses"""

    # Handle silence
    if observation is None or 'text' not in observation:
        return ExecutionResult(
            success=True,
            reward=0.15,  # Low reward for silence
            description="Listening...",
            outputs={'state': 'listening'}
        )

    # Speech detected
    user_text = observation['text']
    confidence = observation['confidence']

    print(f"\n{'=' * 70}")
    print(f"USER: {user_text}")
    print(f"      (confidence: {confidence:.2f}, stance: {stance.value})")

    # Generate response
    response_text = generate_response(user_text, stance)

    print(f"\nSAGE: {response_text}")

    # In full implementation, would synthesize speech here via TTS
    # For now, just simulate the process
    synthesis_time = len(response_text) / 50.0  # ~50 chars/second speech
    print(f"      (synthesis time: {synthesis_time:.1f}s)")

    # Reward based on confidence and stance appropriateness
    reward = confidence * 0.8 + 0.2  # 0.2 base + confidence bonus

    print(f"={'=' * 70}\n")

    return ExecutionResult(
        success=True,
        reward=reward,
        description=f"Conversation turn: {len(user_text)} chars in, {len(response_text)} chars out",
        outputs={
            'user_text': user_text,
            'response_text': response_text,
            'confidence': confidence
        }
    )

def main():
    print("=" * 70)
    print("AUDIO ECHO LOOP - Bidirectional Conversation")
    print("=" * 70)
    print("\nFull conversation cycle:")
    print("  1. Speech detection (simulated)")
    print("  2. SNARC salience assessment")
    print("  3. Response generation (simple rules)")
    print("  4. Text-to-speech synthesis (simulated)")
    print("\nRunning 5 conversation turns...")
    print("=" * 70)
    print()

    # Create sensor
    speech_sensor = SimulatedSpeechSensor()

    # Create kernel
    sensor_sources = {'audio': speech_sensor}
    action_handlers = {'audio': handle_audio}

    kernel = SAGEKernel(
        sensor_sources=sensor_sources,
        action_handlers=action_handlers,
        enable_logging=False  # Reduce noise for conversation focus
    )

    # Run long enough for all conversations
    kernel.run(max_cycles=40, cycle_delay=0.2)

    # Analysis
    print("\n" + "=" * 70)
    print("CONVERSATION ANALYSIS")
    print("=" * 70)

    history = kernel.get_history()

    # Count conversations vs listening
    conversations = [h for h in history if 'user_text' in h['result'].outputs]
    listening = [h for h in history if h['result'].outputs.get('state') == 'listening']

    print(f"\nCycle breakdown:")
    print(f"  Total cycles:      {len(history)}")
    print(f"  Conversations:     {len(conversations)}")
    print(f"  Listening (silence): {len(listening)}")

    if conversations:
        print(f"\nConversation turns:")
        for i, conv in enumerate(conversations, 1):
            user = conv['result'].outputs['user_text']
            response = conv['result'].outputs['response_text']
            confidence = conv['result'].outputs['confidence']
            print(f"\n  Turn {i}:")
            print(f"    User: {user}")
            print(f"    SAGE: {response[:60]}{'...' if len(response) > 60 else ''}")
            print(f"    Confidence: {confidence:.2f}")

    # Rewards
    rewards = [h['result'].reward for h in history]
    print(f"\nReward analysis:")
    print(f"  Total:     {sum(rewards):.2f}")
    print(f"  Average:   {sum(rewards)/len(rewards):.3f}")
    print(f"  Min/Max:   {min(rewards):.3f} / {max(rewards):.3f}")

    # SNARC evolution
    stats = kernel.snarc.get_statistics()
    print(f"\nSNARC learning:")
    print(f"  Assessments: {stats['num_assessments']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Final weights:")
    for dim, weight in stats['current_weights'].items():
        change = ((weight - 0.2) / 0.2) * 100
        print(f"    {dim:10s}: {weight:.3f} ({change:+.1f}%)")

    print("\n" + "=" * 70)
    print("\nKEY INSIGHTS:")
    print("1. Did bidirectional conversation work?")
    print("2. Were responses appropriate to input?")
    print("3. How did stance influence responses?")
    print("4. What would full TTS add?")
    print()

if __name__ == "__main__":
    main()
