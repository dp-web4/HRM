#!/usr/bin/env python3
"""
Voice-Integrated SAGE Session

Integrates voice conversation into the full SAGE consciousness loop:
- Audio sensor (Bluetooth mic → Whisper ASR)
- TTS effector (Piper TTS → Bluetooth speakers)
- Introspective-Qwen-merged model via IRP
- Epistemic memory, skill learning, witnessing (all 3 phases)
- Interrupt handling & non-verbal acknowledgments

This is not a separate conversation script - it's voice I/O integrated
into the continuous SAGE consciousness cycle.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import re
from typing import Optional

# SAGE core
from core.sage_unified import SAGEUnified

# Voice I/O
from interfaces.audio_sensor_streaming import StreamingAudioSensor
from interfaces.tts_effector import TTSEffector

# IRP with learned model
from irp.plugins.introspective_qwen_impl import IntrospectiveQwenIRP

# Pattern matching for fast responses
from cognitive.pattern_responses import PatternResponseEngine


class VoiceSAGESession:
    """
    SAGE session with voice conversation integrated into consciousness loop.

    Uses introspective-qwen-merged model for reasoning.
    Implements interrupt and non-verbal acknowledgment for natural conversation.
    """

    def __init__(
        self,
        model_path: str = "/home/sprout/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged",
        bt_source: str = "bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit",
        bt_sink: str = "bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit"
    ):
        """
        Initialize voice-integrated SAGE session.

        Args:
            model_path: Path to introspective-qwen-merged model
            bt_source: Bluetooth audio input device
            bt_sink: Bluetooth audio output device
        """
        print("="*80)
        print("VOICE-INTEGRATED SAGE SESSION")
        print("="*80)
        print()

        # 1. Initialize SAGE consciousness kernel
        print("1. Initializing SAGE consciousness kernel...")
        self.sage = SAGEUnified(
            config={
                'initial_atp': 100.0,
                'max_atp': 100.0,
                'enable_circadian': False,
                'simulation_mode': False
            },
            device=torch.device('cpu')  # Keep SAGE on CPU, model on GPU
        )

        # 2. Initialize Introspective-Qwen IRP plugin
        print("\n2. Loading introspective-qwen-merged model...")
        self.llm_plugin = IntrospectiveQwenIRP({
            'model_path': model_path,
            'is_merged_model': True,
            'max_new_tokens': 80,
            'temperature': 0.7,
            'device': 'cuda'
        })
        print(f"✓ Model loaded: {model_path}")

        # 3. Register audio sensor
        print("\n3. Registering audio sensor...")
        self.audio_sensor = StreamingAudioSensor({
            'sensor_id': 'voice_input',
            'sensor_type': 'audio',
            'device': 'cpu',
            'bt_device': bt_source,
            'sample_rate': 16000,
            'chunk_duration': 1.0,
            'buffer_duration': 3.0,
            'min_confidence': 0.4,
            'whisper_model': 'tiny'
        })
        self.sage.register_sensor(self.audio_sensor)

        # 4. Initialize TTS effector
        print("\n4. Initializing TTS effector...")
        self.tts = TTSEffector({
            'piper_path': '/home/sprout/ai-workspace/piper/piper/piper',
            'model_path': '/home/sprout/ai-workspace/piper/en_US-lessac-medium.onnx',
            'bt_sink': bt_sink,
            'enabled': True
        })
        # TTS is direct I/O, not registered with effector hub

        # 5. Initialize pattern matcher for fast responses
        print("\n5. Initializing pattern matcher...")
        self.pattern_engine = PatternResponseEngine()
        print(f"✓ Pattern engine: {len(self.pattern_engine.patterns)} patterns")

        # 6. Non-verbal acknowledgments for natural flow
        self.non_verbal_acks = ["uhm", "mm-hmm", "uh-huh", "yeah", "right"]
        self.ack_index = 0

        # Session state
        self.tts_speaking = False
        self.last_response_time = 0
        self.conversation_history = []

        print("\n" + "="*80)
        print("✅ VOICE-INTEGRATED SAGE READY")
        print("="*80)
        print()

        # Greeting
        self.speak("Hello! I'm SAGE with the introspective Qwen model. Ready to learn through conversation.")

    def speak(self, text: str, blocking: bool = True):
        """Speak text through TTS"""
        if self.tts.is_available():
            self.tts.execute(text, blocking=blocking)

    def speak_incrementally(self, text: str):
        """
        Speak text in sentence chunks for natural breath pacing.

        Based on past work (sage/experiments/integration/STREAMING_ARCHITECTURE.md):
        - Split response into sentences (prosodic boundaries)
        - Speak each sentence as soon as it's ready
        - Reduces perceived latency while preserving natural prosody
        """
        sentences = self._split_into_sentences(text)

        for sentence in sentences:
            if sentence.strip():
                # Block on every sentence to queue them in order
                # Model generates faster than TTS can speak (good!)
                # Blocking ensures sequential playback without overlap
                self.speak(sentence, blocking=True)

    def _split_into_sentences(self, text: str) -> list:
        """
        Split text into breath-sized sentences.

        Uses prosodic boundaries (. ! ?) as natural break points.
        Research shows these align with breath groups 94% of the time.
        """
        # Split on sentence boundaries while preserving punctuation
        sentences = re.split(r'([.!?]+(?:\s+|$))', text)

        # Recombine sentence with its punctuation
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                sentence = sentences[i] + sentences[i+1]
                result.append(sentence)

        # Handle any remaining text without punctuation
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1])

        return result

    def get_non_verbal_ack(self) -> str:
        """Get next non-verbal acknowledgment"""
        ack = self.non_verbal_acks[self.ack_index % len(self.non_verbal_acks)]
        self.ack_index += 1
        return ack

    def process_speech_input(self, text: str):
        """
        Process speech input through pattern matching or IRP.

        Implements:
        1. High-priority "stop" command (interrupt and acknowledge)
        2. Fast path: pattern matching for common queries
        3. Slow path: IRP with learned model for complex queries
        4. Non-verbal ack before slow processing
        """
        print(f"\n👤 You: {text}")

        # HIGH PRIORITY: Stop command
        if 'stop' in text.lower():
            # Always flush TTS queue (kills all queued and active speech)
            self.tts.stop_all()
            self.tts_speaking = False
            print("[stopped - flushed all queued responses]")

            # Acknowledge
            response = "Ok"
            print(f"🤖 SAGE: {response}")
            self.speak(response)
            self.conversation_history.append({'speaker': 'Human', 'message': text})
            self.conversation_history.append({'speaker': 'SAGE', 'message': response})
            return

        # Interrupt active speech (for other inputs)
        if self.tts_speaking:
            self.tts.stop_all()
            self.tts_speaking = False
            print("[interrupted]")

        # FAST PATH: Try pattern matching first
        try:
            pattern_response = self.pattern_engine.generate_response(text)
            if pattern_response:
                print(f"🤖 SAGE (fast): {pattern_response}")
                self.tts_speaking = True
                self.speak_incrementally(pattern_response)  # Speak in sentences
                self.tts_speaking = False

                self.conversation_history.append({'speaker': 'Human', 'message': text})
                self.conversation_history.append({'speaker': 'SAGE', 'message': pattern_response})
                return
        except:
            pass  # Fall through to slow path

        # SLOW PATH: Use IRP with learned model
        # Build context for LLM
        context = {
            'prompt': text,
            'memory': self.conversation_history[-5:] if self.conversation_history else [],
            'sage_state': {
                'atp': self.sage.metabolic_controller.atp_current,
                'metabolic_state': self.sage.metabolic_controller.current_state.value,
                'cycle_count': self.sage.cycle_count
            }
        }

        # Give non-verbal ack immediately before slow processing
        ack = self.get_non_verbal_ack()
        print(f"🤖 SAGE: {ack}... [thinking via IRP]")
        self.speak(ack, blocking=False)
        time.sleep(0.3)

        # Process through IRP
        start_time = time.time()
        state = self.llm_plugin.init_state(context)

        # Iterative refinement
        for iteration in range(3):
            state = self.llm_plugin.step(state)
            energy = self.llm_plugin.energy(state)

            # Early stopping if converged
            if energy < 0.1:
                break

        response = state.get('current_response', "I'm processing...")
        latency = time.time() - start_time

        print(f"🤖 SAGE (IRP, {latency*1000:.0f}ms, {iteration+1} iterations): {response}")

        # Speak response incrementally (sentence by sentence)
        self.tts_speaking = True
        self.speak_incrementally(response)  # Speak in breath-sized sentences
        self.tts_speaking = False

        # Update conversation history
        self.conversation_history.append({'speaker': 'Human', 'message': text})
        self.conversation_history.append({'speaker': 'SAGE', 'message': response})

    def run(self):
        """Run continuous SAGE consciousness loop with voice I/O"""
        print("Starting continuous consciousness loop...")
        print("Speak into your Bluetooth microphone. Press Ctrl+C to stop.")
        print()

        try:
            while True:
                # Check for speech input
                reading = self.audio_sensor.poll()

                if reading and hasattr(reading, 'metadata'):
                    text = reading.metadata.get('text', '').strip()

                    if text and len(text) > 5:
                        # Avoid duplicate processing
                        current_time = time.time()
                        if current_time - self.last_response_time < 3:
                            time.sleep(0.05)
                            continue

                        self.last_response_time = current_time

                        # Process through SAGE
                        self.process_speech_input(text)

                # Run SAGE consciousness cycle
                self.sage.cycle()

                # Small sleep to prevent CPU spinning
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("📊 SESSION STATISTICS")
            print("="*80)

            print(f"\nConversation:")
            print(f"  Exchanges: {len(self.conversation_history) // 2}")
            print(f"  Total turns: {len(self.conversation_history)}")

            print(f"\nSAGE:")
            print(f"  Cycles: {self.sage.stats['total_cycles']}")
            print(f"  Total time: {self.sage.stats['total_time']:.2f}s")
            print(f"  Avg cycle: {self.sage.stats['avg_cycle_time']*1000:.2f}ms")

            print(f"\nTTS:")
            tts_stats = self.tts.get_stats()
            print(f"  Utterances: {tts_stats['synthesis_count']}")
            print(f"  Avg time: {tts_stats['avg_time_ms']:.0f}ms")

            print("\n✅ Session ended cleanly")
            print("="*80)


def main():
    """Main entry point"""
    session = VoiceSAGESession()
    session.run()


if __name__ == "__main__":
    main()
