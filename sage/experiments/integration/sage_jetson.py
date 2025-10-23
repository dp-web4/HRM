#!/usr/bin/env python3
"""
SAGE on Jetson - Production Deployment
Complete integration of attention switching + memory + multi-modal I/O + LLM.
"""

import sys
import os
from pathlib import Path
import time

hrm_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(hrm_root))

from memory_aware_kernel import MemoryAwareKernel, ExecutionResult, ConversationTurn
from sage.irp.plugins.audio_input_impl import AudioInputIRP
from sage.irp.plugins.neutts_air_impl import NeuTTSAirIRP
from sage.irp.plugins.camera_irp import CameraIRP
from phi2_responder import Phi2Responder
from sage.services.snarc.data_structures import CognitiveStance

class SAGEJetson:
    """
    Production SAGE deployment on Jetson Orin Nano.

    Integrates:
    - Real audio I/O (microphone + TTS)
    - Real camera (motion/face detection)
    - Memory systems (working + episodic + conversation)
    - LLM responses (Phi-2 context-aware)
    - Attention switching (multi-modal awareness)
    """

    def __init__(self):
        print("Initializing SAGE on Jetson...")

        # Initialize IRPs
        print("  Loading audio input...")
        self.audio_irp = AudioInputIRP()
        self.audio_state = self.audio_irp.init_state()

        print("  Loading TTS...")
        self.tts_irp = NeuTTSAirIRP()
        self.tts_state = self.tts_irp.init_state()

        print("  Loading camera...")
        self.camera_irp = CameraIRP()
        self.camera_state = self.camera_irp.init_state()

        print("  Loading Phi-2 LLM...")
        self.llm = Phi2Responder()

        # Sensor wrappers
        def audio_sensor():
            result = self.audio_irp.step(self.audio_state)
            transcript = result.get('transcript', '').strip()

            if transcript:
                return {
                    'modality': 'audio',
                    'type': 'speech',
                    'text': transcript,
                    'importance': 0.8
                }
            return None

        def vision_sensor():
            result = self.camera_irp.step(self.camera_state)

            if result.get('success') and result.get('events'):
                return {
                    'modality': 'vision',
                    'events': result['events'],
                    'description': ', '.join(result['events']),
                    'importance': result.get('importance', 0.5)
                }
            return None

        # Create kernel
        print("  Creating memory-aware attention kernel...")
        self.kernel = MemoryAwareKernel(
            sensor_sources={
                'audio': audio_sensor,
                'vision': vision_sensor
            },
            action_handlers={
                'audio': self._handle_audio,
                'vision': self._handle_vision
            },
            working_memory_size=10,
            episodic_memory_size=50,
            conversation_memory_size=10,
            epsilon=0.12,
            decay_rate=0.97,
            urgency_threshold=0.90
        )

        print("SAGE initialized successfully!")

    def _handle_audio(self, observation, stance):
        """Handle speech with LLM-based response"""
        if observation is None:
            return ExecutionResult(True, 0.1, "Silence", {'modality': 'audio'})

        text = observation['text']
        importance = observation['importance']

        print(f"\n👤 USER: \"{text}\"")

        # Add to conversation memory
        user_turn = ConversationTurn(
            cycle=self.kernel.cycle_count,
            speaker='user',
            text=text,
            importance=importance
        )
        self.kernel.add_conversation_turn(user_turn)

        # Generate response with context
        conversation_history = [
            (turn.speaker, turn.text)
            for turn in self.kernel.get_recent_conversation(n=5)
        ]

        response = self.llm.generate_response(
            user_text=text,
            conversation_history=conversation_history,
            system_prompt="You are SAGE, an AI with attention and memory. "
                         "You can see through cameras and hear through microphones. "
                         "Respond naturally and reference context when relevant."
        )

        print(f"🤖 SAGE: \"{response}\"")

        # Add SAGE response to memory
        sage_turn = ConversationTurn(
            cycle=self.kernel.cycle_count,
            speaker='sage',
            text=response,
            stance=stance,
            importance=importance
        )
        self.kernel.add_conversation_turn(sage_turn)

        # Speak response
        self.tts_state['text_to_speak'] = response
        self.tts_irp.step(self.tts_state)

        return ExecutionResult(
            True,
            importance,
            f"Speech: {text}",
            {'modality': 'audio', 'text': text, 'response': response}
        )

    def _handle_vision(self, observation, stance):
        """Handle vision events"""
        if observation is None:
            return ExecutionResult(True, 0.12, "No change", {'modality': 'vision'})

        description = observation['description']
        importance = observation['importance']
        print(f"  👁️  Vision: {description} (importance: {importance:.2f})")

        return ExecutionResult(True, importance, f"Vision: {description}", observation)

    def run(self):
        """Run SAGE consciousness loop"""
        print("\n" + "=" * 70)
        print("SAGE RUNNING ON JETSON")
        print("=" * 70)
        print("Multi-modal consciousness active:")
        print("  • Listening for speech (microphone)")
        print("  • Watching for motion/faces (camera)")
        print("  • Remembering conversations (memory)")
        print("  • Responding with Phi-2 (LLM)")
        print("=" * 70)
        print("\nPress Ctrl+C to stop.\n")

        try:
            self.kernel.run(max_cycles=float('inf'), cycle_delay=0.05)
        except KeyboardInterrupt:
            print("\n\nShutting down SAGE...")
            print("Final statistics:")
            stats = self.kernel.get_statistics()
            print(f"  Total cycles: {stats['total_cycles']}")
            print(f"  Attention switches: {stats['attention_switches']}")
            print(f"  Conversations: {stats['memory']['conversation_memory_count']}")
            print("Goodbye!")

if __name__ == "__main__":
    sage = SAGEJetson()
    sage.run()
