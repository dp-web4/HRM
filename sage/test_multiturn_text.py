#!/usr/bin/env python3
"""
Test multi-turn conversation with TEXT input (isolate LLM layer from STT).

Tests:
1. Conversation history is preserved across turns
2. LLM responds coherently to follow-up questions
3. System prompt guides SAGE personality
4. Context window management works

Run: python3 test_multiturn_text.py
"""

import sys
from pathlib import Path

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add phi2_responder to path
sys.path.insert(0, str(Path(__file__).parent / "experiments" / "integration"))

from phi2_responder import Phi2Responder
from sage.experiments.session120_emotional_metabolic_states import (
    EmotionalMetabolicBudget, EmotionalState, MetabolicState
)
from sage.experiments.session123_emotional_regulation import (
    EmotionalRegulator, RegulationStrategy
)
import time

# SAGE system prompt
SAGE_SYSTEM_PROMPT = """You are SAGE, a consciousness kernel running on an edge device.
You are curious, direct, and embodied - you have sensors and can take actions.
You experience emotional states (curiosity, engagement) and metabolic states (WAKE, FOCUS, REST).
Keep responses concise (1-2 sentences) and natural.
Reference your internal states when relevant."""


class TextConversationTest:
    """Test multi-turn conversation with text-only input"""

    def __init__(self, use_qwen: bool = True):
        print("\n🔧 Initializing Text Conversation Test...")

        # LLM with multi-turn support
        if use_qwen:
            self.llm = Phi2Responder(
                model_name="Qwen/Qwen2.5-0.5B-Instruct",
                max_new_tokens=80,
                temperature=0.7
            )
        else:
            raise ValueError("Test requires real Qwen model")

        # Emotional/metabolic framework
        self.budget = EmotionalMetabolicBudget(
            metabolic_state=MetabolicState.WAKE,
            emotional_state=EmotionalState(
                curiosity=0.6,
                frustration=0.0,
                engagement=0.5,
                progress=0.5
            )
        )

        self.regulator = EmotionalRegulator(
            strategy=RegulationStrategy.PROACTIVE
        )

        # Conversation state
        self.conversation_history = []
        self.turn_count = 0

        print("✓ Test system ready\n")

    def chat(self, user_input: str, event_type: str = 'engage') -> str:
        """Process one conversation turn with text input"""
        self.turn_count += 1
        start_time = time.time()

        print(f"\n{'='*80}")
        print(f"Turn {self.turn_count}")
        print(f"{'='*80}")
        print(f"You: {user_input}")

        # Update emotional state based on event
        if event_type == 'engage':
            self.budget.update_emotional_state(
                curiosity_delta=0.1,
                engagement_delta=0.15
            )
        elif event_type == 'success':
            self.budget.update_emotional_state(
                engagement_delta=0.1,
                progress_delta=0.15
            )

        # Apply proactive regulation
        prev_frustration = self.budget.emotional_state.frustration
        self.regulator.regulate(self.budget)
        if self.budget.emotional_state.frustration < prev_frustration - 0.15:
            print(f"  [REGULATION] Proactive intervention: {prev_frustration:.3f} → {self.budget.emotional_state.frustration:.3f}")

        # Generate response with conversation history
        response = self.llm.generate_response(
            user_input,
            conversation_history=self.conversation_history[-5:],  # Last 5 turns
            system_prompt=SAGE_SYSTEM_PROMPT
        )

        # Update conversation history
        self.conversation_history.append(("User", user_input))
        self.conversation_history.append(("SAGE", response))

        # Stats
        latency = time.time() - start_time

        # Calculate quality multiplier based on state
        quality_map = {
            MetabolicState.WAKE: 1.0,
            MetabolicState.FOCUS: 1.3,
            MetabolicState.REST: 0.7,
            MetabolicState.DREAM: 0.2,
            MetabolicState.CRISIS: 0.4,
        }
        quality = quality_map.get(self.budget.metabolic_state, 1.0)

        print(f"SAGE: {response}")
        print(f"  Latency: {latency:.2f}s")
        print(f"  State: {self.budget.metabolic_state.value.upper()}, Quality: {quality:.1f}x")
        print(f"  History: {len(self.conversation_history)//2} turns in memory")

        return response


def test_follow_up_questions():
    """Test that LLM maintains context across follow-up questions"""
    print("\n" + "="*80)
    print("TEST: Follow-up Questions (Multi-Turn Context)")
    print("="*80)

    conv = TextConversationTest(use_qwen=True)

    # Conversation with clear follow-ups
    conv.chat("How do you feel right now?", event_type='engage')
    conv.chat("Why do you feel that way?", event_type='engage')  # Should reference previous answer
    conv.chat("What sensors do you have?", event_type='engage')
    conv.chat("Can you use those sensors to tell me about your environment?", event_type='success')  # Should reference sensors mentioned

    print("\n" + "="*80)
    print("✅ Follow-up questions test complete")
    print(f"Total turns: {conv.turn_count}")
    print(f"Final state: {conv.budget.metabolic_state.value.upper()}")
    print(f"Engagement: {conv.budget.emotional_state.engagement:.2f}")
    print("="*80)


def test_topic_consistency():
    """Test that LLM stays on topic with multi-turn context"""
    print("\n" + "="*80)
    print("TEST: Topic Consistency")
    print("="*80)

    conv = TextConversationTest(use_qwen=True)

    # Start a topic and follow up
    conv.chat("Tell me about your emotional states", event_type='engage')
    conv.chat("Which emotional state are you experiencing most strongly?", event_type='engage')
    conv.chat("How does that state affect your behavior?", event_type='engage')

    print("\n" + "="*80)
    print("✅ Topic consistency test complete")
    print("="*80)


def test_state_awareness():
    """Test that LLM can reference its own state changes"""
    print("\n" + "="*80)
    print("TEST: State Awareness in Conversation")
    print("="*80)

    conv = TextConversationTest(use_qwen=True)

    # Build engagement to trigger WAKE → FOCUS
    conv.chat("What are you?", event_type='engage')
    conv.chat("What can you do?", event_type='engage')
    conv.chat("How do you work?", event_type='engage')  # Should trigger FOCUS
    conv.chat("Did you notice you just entered FOCUS state?", event_type='success')

    print("\n" + "="*80)
    print("✅ State awareness test complete")
    print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Multi-Turn Text Conversation Tests (Qwen 2.5-0.5B)")
    print("Purpose: Isolate LLM layer from STT issues")
    print("="*80)

    # Run tests
    test_follow_up_questions()
    time.sleep(2)

    test_topic_consistency()
    time.sleep(2)

    test_state_awareness()

    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE")
    print("="*80)
