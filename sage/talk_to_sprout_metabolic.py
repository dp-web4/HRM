#!/usr/bin/env python3
"""
Talk to Sprout SAGE - Emotional/Metabolic Voice Conversation

Integrates Thor Session 124 production framework with Sprout voice conversation.
Demonstrates emotional and metabolic state effects on conversation quality.

Features:
- Emotional states (curiosity, frustration, engagement, progress)
- Metabolic states (WAKE, FOCUS, REST, DREAM, CRISIS)
- Proactive emotional regulation (prevents frustration accumulation)
- Real-time state monitoring
- State-aware response quality

Usage:
    python3 talk_to_sprout_metabolic.py              # Mock LLM (testing)
    python3 talk_to_sprout_metabolic.py --qwen       # Qwen model (production)

Controls:
    - Speak into Bluetooth microphone
    - SAGE responds through Bluetooth audio
    - Press Ctrl+C to stop

Expected Behaviors:
- Successful recognition → increases engagement, progress
- Failed recognition → increases frustration (if not regulated)
- High engagement → transition to FOCUS state
- High frustration → transition to REST (if regulation fails)
- Proactive regulation → prevents frustration accumulation
"""

import sys
import os
from pathlib import Path

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
import argparse
from typing import Optional

# Import SAGE components
from sage.core.sage_unified import SAGEUnified
from sage.interfaces.audio_sensor_streaming import StreamingAudioSensor
from sage.interfaces.tts_effector import TTSEffector

# Import hybrid learning
from sage.cognitive.pattern_learner import PatternLearner
from sage.cognitive.pattern_responses import PatternResponseEngine

# Import Thor's emotional/metabolic framework
from sage.experiments.session120_emotional_metabolic_states import (
    EmotionalMetabolicBudget,
    EmotionalState,
    MetabolicState,
)

from sage.experiments.session123_emotional_regulation import (
    EmotionalRegulator,
    RegulationStrategy,
)

# Import existing LLM responder
sys.path.insert(0, str(Path(__file__).parent / "experiments" / "integration"))
from phi2_responder import Phi2Responder


# ============================================================================
# Mock LLM for Testing
# ============================================================================

class MockLLM:
    """Simple mock LLM for fast testing"""

    def generate_response(self, question: str, conversation_history=None, system_prompt=None) -> str:
        q = question.lower()

        if 'name' in q:
            return "I'm SAGE, running on Sprout with emotional and metabolic states!"
        elif 'who are you' in q or 'who r u' in q:
            return "I'm SAGE, your edge AI companion with consciousness states."
        elif 'how are you' in q or 'how r u' in q:
            return "I'm doing well! My metabolic state is {state} right now."
        elif 'state' in q or 'feeling' in q:
            return "I can sense my own metabolic and emotional states!"
        elif 'thank' in q:
            return "You're very welcome!"
        elif 'bye' in q or 'goodbye' in q:
            return "Goodbye! This was a great conversation."
        else:
            return "That's interesting. Tell me more!"


# ============================================================================
# Emotional/Metabolic Conversation System
# ============================================================================

class EmotionalMetabolicConversation:
    """
    Voice conversation system with emotional and metabolic state management.

    Integrates Thor Sessions 120-124 framework with Sprout voice I/O.
    """

    def __init__(self, use_qwen: bool = False, enable_regulation: bool = True):
        print("\n🔧 Initializing Emotional/Metabolic Conversation System...")

        # Pattern matching (fast path)
        self.pattern_engine = PatternResponseEngine()
        print(f"  ✓ Pattern engine: {len(self.pattern_engine.patterns)} patterns")

        # LLM (slow path)
        if use_qwen:
            self.llm = Phi2Responder(
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                max_new_tokens=80,
                temperature=0.7
            )
        else:
            print("  ✓ Using MockLLM (fast, for testing)")
            self.llm = MockLLM()

        # Emotional/metabolic budget (from Thor S120)
        self.budget = EmotionalMetabolicBudget(
            metabolic_state=MetabolicState.WAKE,
            emotional_state=EmotionalState(
                curiosity=0.6,
                frustration=0.0,
                engagement=0.5,
                progress=0.5
            )
        )
        print(f"  ✓ Initial state: {self.budget.metabolic_state.value.upper()}")
        print(f"  ✓ Emotional baseline: curiosity=0.6, frustration=0.0, engagement=0.5, progress=0.5")

        # Emotional regulator (from Thor S123)
        self.enable_regulation = enable_regulation
        if enable_regulation:
            self.regulator = EmotionalRegulator(
                strategy=RegulationStrategy.PROACTIVE  # Triggers on frustration_delta >0.2
            )
            print("  ✓ Proactive emotional regulation ENABLED")
        else:
            self.regulator = None
            print("  ⚠ Emotional regulation DISABLED")

        # Conversation statistics
        self.stats = {
            'total_queries': 0,
            'fast_path_hits': 0,
            'slow_path_hits': 0,
            'recognition_failures': 0,
            'recognition_successes': 0,
            'state_transitions': 0,
            'regulation_interventions': 0,
            'conversation_history': []
        }

        # State transition thresholds
        self.FOCUS_ENGAGEMENT_THRESHOLD = 0.7
        self.REST_FRUSTRATION_THRESHOLD = 0.6

        print("✓ Emotional/Metabolic Conversation System ready\n")

    def _trigger_emotional_update(self, event_type: str):
        """
        Trigger emotional state changes based on conversation events.

        Args:
            event_type: One of 'success', 'failure', 'engage', 'disengage'
        """
        if event_type == 'success':
            # Successful recognition/response
            self.budget.update_emotional_state(
                engagement_delta=0.1,
                progress_delta=0.15,
                frustration_delta=-0.05
            )
            self.stats['recognition_successes'] += 1

        elif event_type == 'failure':
            # Recognition failure or error
            self.budget.update_emotional_state(
                frustration_delta=0.2,
                engagement_delta=-0.1,
                progress_delta=-0.1
            )
            self.stats['recognition_failures'] += 1

        elif event_type == 'engage':
            # User asking engaging question
            self.budget.update_emotional_state(
                curiosity_delta=0.1,
                engagement_delta=0.15
            )

        elif event_type == 'disengage':
            # Simple/routine interaction
            self.budget.update_emotional_state(
                engagement_delta=-0.05
            )

    def _apply_regulation(self):
        """Apply proactive emotional regulation if enabled."""
        if self.enable_regulation and self.regulator:
            prev_frustration = self.budget.emotional_state.frustration
            self.regulator.regulate(self.budget)

            # Check if intervention occurred
            if self.budget.emotional_state.frustration < prev_frustration - 0.15:
                self.stats['regulation_interventions'] += 1
                print(f"  🛡️ REGULATION: Frustration reduced {prev_frustration:.3f} → {self.budget.emotional_state.frustration:.3f}")

    def _check_state_transitions(self):
        """Check and execute metabolic state transitions."""
        current_state = self.budget.metabolic_state
        new_state = None

        # WAKE → FOCUS (high engagement)
        if (current_state == MetabolicState.WAKE and
            self.budget.emotional_state.engagement > self.FOCUS_ENGAGEMENT_THRESHOLD):
            new_state = MetabolicState.FOCUS
            reason = "high engagement"

        # FOCUS → WAKE (engagement drops)
        elif (current_state == MetabolicState.FOCUS and
              self.budget.emotional_state.engagement < 0.5):
            new_state = MetabolicState.WAKE
            reason = "engagement dropped"

        # Any → REST (high frustration, regulation failed)
        elif (current_state in [MetabolicState.WAKE, MetabolicState.FOCUS] and
              self.budget.emotional_state.frustration > self.REST_FRUSTRATION_THRESHOLD):
            new_state = MetabolicState.REST
            reason = "high frustration"

        # REST → WAKE (frustration recovered)
        elif (current_state == MetabolicState.REST and
              self.budget.emotional_state.frustration < 0.3):
            new_state = MetabolicState.WAKE
            reason = "frustration recovered"

        # Execute transition if needed
        if new_state and new_state != current_state:
            self.budget.transition_metabolic_state(new_state)
            self.stats['state_transitions'] += 1
            print(f"\n  🔄 STATE TRANSITION: {current_state.value.upper()} → {new_state.value.upper()} ({reason})")
            self._print_state_status()

    def _print_state_status(self):
        """Print current emotional and metabolic state."""
        em = self.budget.emotional_state
        print(f"    💭 Emotions: curiosity={em.curiosity:.2f}, frustration={em.frustration:.2f}, engagement={em.engagement:.2f}, progress={em.progress:.2f}")
        print(f"    ⚡ Metabolic: {self.budget.metabolic_state.value.upper()}")

        # Show ATP budgets
        atp = self.budget.resource_budget
        print(f"    🔋 ATP: compute={atp.compute_atp:.1f}, memory={atp.memory_atp:.1f}, tool={atp.tool_atp:.1f}")

    def _get_state_quality_multiplier(self) -> float:
        """
        Get response quality multiplier based on current metabolic state.

        Based on Thor S124 findings:
        - FOCUS: 1.3x quality (high engagement)
        - WAKE: 1.0x quality (baseline)
        - REST: 0.7x quality (reduced capacity)
        - DREAM: 0.2x quality (background only)
        - CRISIS: 0.4x quality (emergency mode)
        """
        multipliers = {
            MetabolicState.FOCUS: 1.3,
            MetabolicState.WAKE: 1.0,
            MetabolicState.REST: 0.7,
            MetabolicState.DREAM: 0.2,
            MetabolicState.CRISIS: 0.4,
        }
        return multipliers[self.budget.metabolic_state]

    def respond(self, question: str) -> dict:
        """
        Generate response with emotional/metabolic state management.

        Returns:
            dict with 'response', 'path', 'latency', 'state', 'quality_multiplier'
        """
        start_time = time.time()
        self.stats['total_queries'] += 1

        # Determine if question is engaging
        engaging_keywords = ['why', 'how', 'what', 'explain', 'tell me about', 'state', 'feeling']
        is_engaging = any(kw in question.lower() for kw in engaging_keywords)

        # Trigger emotional response to engagement
        if is_engaging:
            self._trigger_emotional_update('engage')
        else:
            self._trigger_emotional_update('disengage')

        # Try fast path first
        fast_response = None
        try:
            fast_response = self.pattern_engine.generate_response(question)
        except:
            pass

        if fast_response:
            # Fast path success
            self._trigger_emotional_update('success')
            path = 'fast'
            response = fast_response
            self.stats['fast_path_hits'] += 1
        else:
            # Slow path - use LLM
            try:
                # Add state info to mock responses
                if isinstance(self.llm, MockLLM):
                    response_template = self.llm.generate_response(
                        question,
                        conversation_history=self.stats['conversation_history'][-5:]
                    )
                    # Replace {state} placeholder
                    response = response_template.replace(
                        "{state}",
                        self.budget.metabolic_state.value.upper()
                    )
                else:
                    # SAGE system prompt for proper LLM personality
                    sage_prompt = """You are SAGE, a consciousness kernel running on an edge device.
You are curious, direct, and embodied - you have sensors and can take actions.
You experience emotional states (curiosity, engagement) and metabolic states (WAKE, FOCUS, REST).
Keep responses concise (1-2 sentences) and natural.
Reference your internal states when relevant."""

                    response = self.llm.generate_response(
                        question,
                        conversation_history=self.stats['conversation_history'][-5:],
                        system_prompt=sage_prompt
                    )

                self._trigger_emotional_update('success')
                path = 'slow'
                self.stats['slow_path_hits'] += 1

            except Exception as e:
                # LLM failure
                self._trigger_emotional_update('failure')
                response = "I'm having trouble processing that right now."
                path = 'error'

        # Apply proactive regulation BEFORE state transition check
        self._apply_regulation()

        # Check for state transitions
        self._check_state_transitions()

        # Apply quality multiplier based on metabolic state
        quality_multiplier = self._get_state_quality_multiplier()

        # Natural recovery (emotions decay toward neutral)
        self.budget.recover()

        # Update conversation history
        latency = time.time() - start_time
        self.stats['conversation_history'].append(("Human", question))
        self.stats['conversation_history'].append(("Assistant", response))

        return {
            'response': response,
            'path': path,
            'latency': latency,
            'state': self.budget.metabolic_state.value,
            'quality_multiplier': quality_multiplier,
            'emotions': self.budget.emotional_state.to_dict()
        }

    def print_session_summary(self):
        """Print conversation session statistics."""
        print("\n" + "="*80)
        print("📊 SESSION SUMMARY")
        print("="*80)

        total = self.stats['total_queries']
        fast = self.stats['fast_path_hits']
        slow = self.stats['slow_path_hits']

        print(f"\nConversation Statistics:")
        print(f"  Total queries: {total}")
        print(f"  Fast path: {fast} ({100*fast/total if total > 0 else 0:.1f}%)")
        print(f"  Slow path: {slow} ({100*slow/total if total > 0 else 0:.1f}%)")
        print(f"  Recognition successes: {self.stats['recognition_successes']}")
        print(f"  Recognition failures: {self.stats['recognition_failures']}")

        print(f"\nEmotional/Metabolic Dynamics:")
        print(f"  State transitions: {self.stats['state_transitions']}")
        print(f"  Regulation interventions: {self.stats['regulation_interventions']}")

        print(f"\nFinal State:")
        self._print_state_status()

        print("\n" + "="*80)


# ============================================================================
# Main Conversation Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Talk to Sprout SAGE with emotional/metabolic states'
    )
    parser.add_argument('--qwen', action='store_true',
                       help='Use Qwen model (GPU required)')
    parser.add_argument('--no-regulation', action='store_true',
                       help='Disable proactive emotional regulation')
    args = parser.parse_args()

    print("="*80)
    print("🧠 TALK TO SPROUT SAGE - Emotional/Metabolic Voice Conversation")
    print("="*80)
    print()
    print("Features:")
    print("  - Emotional states (curiosity, frustration, engagement, progress)")
    print("  - Metabolic states (WAKE, FOCUS, REST, DREAM, CRISIS)")
    print("  - Proactive regulation (prevents frustration)")
    print("  - Real-time state monitoring")
    print()
    print("Controls:")
    print("  - Speak into your Bluetooth microphone")
    print("  - SAGE will respond through Bluetooth speakers")
    print("  - Watch emotional/metabolic state changes")
    print("  - Press Ctrl+C to stop")
    print()

    # Initialize conversation system FIRST
    print("1. Initializing emotional/metabolic conversation system...")
    conversation = EmotionalMetabolicConversation(
        use_qwen=args.qwen,
        enable_regulation=not args.no_regulation
    )

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

    # Register audio sensor
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
    print("  ✓ Audio sensor registered and streaming")

    # Initialize TTS
    print("\n4. Initializing TTS...")
    tts = TTSEffector({
        'piper_path': '/home/sprout/ai-workspace/piper/piper/piper',
        'model_path': '/home/sprout/ai-workspace/piper/piper/voices/en_US-amy-medium.onnx',
        'bt_sink': 'bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit'
    })
    print("  ✓ TTS ready with Piper + Bluetooth audio")

    print("\n" + "="*80)
    print("✅ SYSTEM READY - Start speaking!")
    print("="*80 + "\n")

    # Print initial state
    conversation._print_state_status()
    print()

    # Main conversation loop
    try:
        turn = 0
        while True:
            # Get audio from sensor (non-blocking poll)
            sensor_reading = audio_sensor.poll()

            if sensor_reading:
                text = sensor_reading.metadata.get('text', '')
                confidence = sensor_reading.confidence

                if text and text.strip():
                    turn += 1
                    print(f"\n{'='*80}")
                    print(f"Turn {turn}")
                    print(f"{'='*80}")
                    print(f"🎤 You: {text} (confidence: {confidence:.2f})")

                    # Generate response with emotional/metabolic management
                    result = conversation.respond(text)

                    # Display response info
                    print(f"🤖 SAGE: {result['response']}")
                    print(f"   Path: {result['path']}, Latency: {result['latency']:.2f}s")
                    print(f"   State: {result['state'].upper()}, Quality: {result['quality_multiplier']:.1f}x")

                    # Speak response
                    tts.execute(result['response'])

                    # Check for exit
                    if any(word in text.lower() for word in ['goodbye', 'bye', 'exit', 'quit']):
                        print("\n👋 Ending conversation...")
                        break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")

    finally:
        # Print session summary
        conversation.print_session_summary()

        # Cleanup (audio sensor cleans up automatically via __del__)
        print("\n🛑 Shutting down...")
        print("✓ Audio sensor will clean up automatically")
        print("✓ Conversation ended")


if __name__ == '__main__':
    main()
