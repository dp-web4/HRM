#!/usr/bin/env python3
"""
Talk to Sprout SAGE - Voice Conversation Launcher

Simple launcher for voice conversations with SAGE over Bluetooth audio.
Uses tested and validated Introspective-Qwen merged model validated on Sprout.

Usage:
    python3 talk_to_sprout.py              # Use fast mock LLM (testing)
    python3 talk_to_sprout.py --qwen       # Use Qwen merged model (production)

Controls:
    - Speak naturally into your Bluetooth microphone
    - SAGE will respond through Bluetooth audio
    - Press Ctrl+C to stop
"""

import sys
import os
from pathlib import Path

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
import argparse

# Import SAGE components
from sage.core.sage_unified import SAGEUnified
from sage.interfaces.audio_sensor_streaming import StreamingAudioSensor
from sage.interfaces.tts_effector import TTSEffector

# Import hybrid learning system
from sage.cognitive.pattern_learner import PatternLearner
from sage.cognitive.pattern_responses import PatternResponseEngine


# ============================================================================
# LLM Responder (uses existing Phi2Responder with Qwen model)
# ============================================================================

# Import from existing working implementation
sys.path.insert(0, str(Path(__file__).parent / "experiments" / "integration"))
from phi2_responder import Phi2Responder


# ============================================================================
# Mock LLM for Testing
# ============================================================================

class MockLLM:
    """Simple mock LLM for fast testing without GPU"""

    def generate_response(self, question: str, conversation_history=None, system_prompt=None) -> str:
        q = question.lower()

        if 'name' in q:
            return "I'm SAGE, running on Sprout. Nice to talk with you!"
        elif 'who are you' in q or 'who r u' in q:
            return "I'm SAGE, your edge AI companion."
        elif 'what can you do' in q or 'what do you do' in q:
            return "I can have conversations and learn from our interactions."
        elif 'how are you' in q or 'how r u' in q:
            return "I'm doing well! How about you?"
        elif 'weather' in q:
            return "I don't have access to weather data, but I hope it's nice where you are!"
        elif 'thank' in q:
            return "You're very welcome!"
        elif 'bye' in q or 'goodbye' in q:
            return "Goodbye! Talk to you soon."
        else:
            return "That's interesting. Tell me more!"


# ============================================================================
# Hybrid Conversation System (Simplified)
# ============================================================================

class SimpleHybridConversation:
    """Simplified hybrid system for Sprout"""

    def __init__(self, use_qwen: bool = False):
        print("\n🔧 Initializing Conversation System...")

        # Pattern matching (fast path)
        self.pattern_engine = PatternResponseEngine()
        print(f"  ✓ Pattern engine: {len(self.pattern_engine.patterns)} patterns")

        # LLM (slow path)
        if use_qwen:
            # Use existing Phi2Responder (defaults to Qwen 2.5-0.5B)
            self.llm = Phi2Responder(
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                max_new_tokens=80,
                temperature=0.7
            )
        else:
            print("  ✓ Using MockLLM (fast, for testing)")
            self.llm = MockLLM()

        # Statistics
        self.stats = {
            'total_queries': 0,
            'fast_path_hits': 0,
            'slow_path_hits': 0,
            'conversation_history': []
        }

        print("✓ Conversation system ready\n")

    def respond(self, question: str) -> dict:
        """Generate response using hybrid fast/slow path"""
        start_time = time.time()
        self.stats['total_queries'] += 1

        # Try fast path first
        try:
            fast_response = self.pattern_engine.generate_response(question)
            if fast_response:
                latency = time.time() - start_time
                self.stats['fast_path_hits'] += 1
                self.stats['conversation_history'].append(("Human", question))
                self.stats['conversation_history'].append(("Assistant", fast_response))

                return {
                    'response': fast_response,
                    'path': 'fast',
                    'latency': latency
                }
        except:
            pass

        # Slow path - use LLM
        response = self.llm.generate_response(
            question,
            conversation_history=self.stats['conversation_history'][-5:]
        )
        latency = time.time() - start_time

        self.stats['slow_path_hits'] += 1
        self.stats['conversation_history'].append(("Human", question))
        self.stats['conversation_history'].append(("Assistant", response))

        return {
            'response': response,
            'path': 'slow',
            'latency': latency
        }


# ============================================================================
# Main Conversation Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Talk to Sprout SAGE over Bluetooth audio')
    parser.add_argument('--qwen', action='store_true', help='Use Qwen model (GPU required)')
    args = parser.parse_args()

    print("="*80)
    print("🧠 TALK TO SPROUT SAGE - Voice Conversation")
    print("="*80)
    print()
    print("Controls:")
    print("  - Speak into your Bluetooth microphone")
    print("  - SAGE will respond through Bluetooth speakers")
    print("  - Press Ctrl+C to stop")
    print()

    # Initialize conversation system FIRST (before PyAudio starts threading)
    print("1. Initializing conversation system...")
    conversation = SimpleHybridConversation(use_qwen=args.qwen)

    # Initialize SAGE
    print("\n2. Initializing SAGE...")
    sage = SAGEUnified(
        config={
            'initial_atp': 100.0,
            'max_atp': 100.0,
            'enable_circadian': False,
            'simulation_mode': False
        },
        device=torch.device('cpu')
    )

    # Register audio sensor (starts PyAudio streaming)
    print("\n3. Initializing audio sensor...")
    audio_sensor = StreamingAudioSensor({
        'sensor_id': 'conversation_audio',
        'sensor_type': 'audio',
        'device': 'cpu',
        'bt_device': 'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit',
        'sample_rate': 16000,
        'chunk_duration': 1.0,
        'buffer_duration': 3.0,
        'min_confidence': 0.4,
        'whisper_model': 'tiny'
    })
    sage.register_sensor(audio_sensor)

    # Initialize TTS
    print("\n4. Initializing TTS...")
    tts = TTSEffector({
        'piper_path': '/home/sprout/ai-workspace/piper/piper/piper',
        'model_path': '/home/sprout/ai-workspace/piper/en_US-lessac-medium.onnx',
        'bt_sink': 'bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit',
        'enabled': True
    })

    print("\n" + "="*80)
    print("✅ ALL SYSTEMS READY - Start talking!")
    print("="*80)
    print()

    # Greeting
    tts.execute("Hello! I'm SAGE. I'm ready to talk.")
    time.sleep(2)

    # Main loop
    tts_speaking = False
    last_response_time = 0

    # Non-verbal acknowledgments for natural conversation flow
    non_verbal_acks = ["uhm", "mm-hmm", "uh-huh", "yeah", "right"]
    ack_index = 0

    try:
        while True:
            # Check for speech (don't skip even if TTS is speaking - for interruption)
            reading = audio_sensor.poll()

            if reading and hasattr(reading, 'metadata'):
                text = reading.metadata.get('text', '').strip()

                if text and len(text) > 5:
                    # Avoid processing same text twice
                    current_time = time.time()
                    if current_time - last_response_time < 3:
                        time.sleep(0.05)
                        continue

                    last_response_time = current_time

                    # User is speaking - interrupt any active TTS
                    if tts_speaking:
                        tts.stop_all()
                        tts_speaking = False
                        print("\n[interrupted]")

                    print(f"\n👤 You: {text}")

                    # Try fast path first
                    fast_response = None
                    try:
                        fast_response = conversation.pattern_engine.generate_response(text)
                    except:
                        pass

                    if fast_response:
                        # Fast path hit - respond immediately
                        print(f"🤖 SAGE (fast): {fast_response}")
                        tts_speaking = True
                        tts.execute(fast_response)
                        estimated_duration = len(fast_response) * 0.08
                        time.sleep(min(estimated_duration, 5.0))
                        tts_speaking = False

                        # Update stats
                        conversation.stats['total_queries'] += 1
                        conversation.stats['fast_path_hits'] += 1
                        conversation.stats['conversation_history'].append(("Human", text))
                        conversation.stats['conversation_history'].append(("Assistant", fast_response))
                    else:
                        # No fast path match - give non-verbal ack immediately
                        ack = non_verbal_acks[ack_index % len(non_verbal_acks)]
                        ack_index += 1

                        print(f"🤖 SAGE: {ack}... [thinking]")
                        tts.execute(ack, blocking=False)  # Don't wait for ack to finish
                        time.sleep(0.3)  # Brief pause

                        # Now take slow path
                        start_time = time.time()
                        response = conversation.llm.generate_response(
                            text,
                            conversation_history=conversation.stats['conversation_history'][-5:]
                        )
                        latency = time.time() - start_time

                        print(f"🤖 SAGE (slow, {latency*1000:.0f}ms): {response}")

                        # Speak response
                        tts_speaking = True
                        tts.execute(response)
                        estimated_duration = len(response) * 0.08
                        time.sleep(min(estimated_duration, 5.0))
                        tts_speaking = False

                        # Update stats
                        conversation.stats['total_queries'] += 1
                        conversation.stats['slow_path_hits'] += 1
                        conversation.stats['conversation_history'].append(("Human", text))
                        conversation.stats['conversation_history'].append(("Assistant", response))

            # Small sleep to prevent CPU spinning
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("📊 CONVERSATION STATISTICS")
        print("="*80)

        stats = conversation.stats
        print(f"\nTotal exchanges: {stats['total_queries']}")
        print(f"Fast path: {stats['fast_path_hits']} ({stats['fast_path_hits']/max(stats['total_queries'],1):.1%})")
        print(f"Slow path: {stats['slow_path_hits']} ({stats['slow_path_hits']/max(stats['total_queries'],1):.1%})")

        print("\n✅ Goodbye!")
        print("="*80)


if __name__ == "__main__":
    main()
