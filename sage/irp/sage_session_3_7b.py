#!/usr/bin/env python3
"""
SAGE Conversational Learning Session #3
Testing larger model: Does size matter?

Comparing:
- Session #1: Qwen 0.5B + BitNet 2B (baseline)
- Session #2: Same models, deeper relationship
- Session #3: Qwen 7B (14x larger than 0.5B)

Scientific question: Do larger models show better conversational learning?
"""

import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'plugins'))
from qwen_7b_irp import Qwen7BIRP


class ConversationalSAGE7B:
    """SAGE learning through conversation - Session 3 with 7B model"""

    def __init__(self):
        self.qwen_7b = Qwen7BIRP()

        # Start with naive trust (no prior experience with 7B)
        self.trust_history = {'qwen_7b': [1.0]}

        self.energy_history = []
        self.conversation_log = []
        self.my_name = "SAGE"
        self.conversation_started = datetime.now()
        self.total_exchanges = 0

    def converse(self, message: str, message_type: str = "general"):
        """Have a conversation turn with 7B model"""
        self.total_exchanges += 1

        print(f"\n{'='*80}")
        print(f"[Turn {self.total_exchanges}] You: {message}")
        print(f"Type: {message_type}")
        print('-'*80)

        trust = self.trust_history['qwen_7b'][-1]
        print(f"SAGE thinking: QWEN 7B")
        print(f"  Trust: {trust:.3f}")

        start = time.time()

        # Initialize and run
        config = {'max_new_tokens': 150, 'temperature': 0.85}
        self.qwen_7b.initialize(config)
        x_0 = self.qwen_7b.preprocess(message)
        x_1 = self.qwen_7b.step(x_0, t=0)
        energy = self.qwen_7b.energy(x_1, t=0)
        elapsed = time.time() - start

        response = x_1['response']

        print(f"\nSAGE: {response}")
        print(f"\n  Energy: {energy:.3f} | Time: {elapsed:.2f}s | Model: Qwen 7B")

        # Update trust
        self.update_trust(energy, elapsed, message_type)

        self.conversation_log.append({
            'turn': self.total_exchanges,
            'message': message,
            'type': message_type,
            'model': 'qwen_7b',
            'response': response,
            'energy': energy,
            'time': elapsed,
            'trust': self.trust_history['qwen_7b'][-1]
        })

        self.energy_history.append(energy)

        return response

    def update_trust(self, energy: float, time_taken: float, message_type: str):
        """Update trust - same algorithm as Sessions 1 & 2"""
        learning_rate = 0.15  # Same as Session #1 (fresh model)

        energy_quality = 1.0 - energy
        time_efficiency = 1.0 / (1.0 + time_taken / 12.0)
        quality_score = energy_quality * 0.85 + time_efficiency * 0.15

        current_trust = self.trust_history['qwen_7b'][-1]
        new_trust = current_trust * (1 - learning_rate) + quality_score * learning_rate

        self.trust_history['qwen_7b'].append(new_trust)

    def show_learning_summary(self):
        """Show what SAGE learned with 7B model"""
        print(f"\n{'='*80}")
        print("SESSION #3 LEARNING SUMMARY - QWEN 7B")
        print('='*80)

        print(f"\nSession: {self.total_exchanges} turns")
        print(f"Model: Qwen 7B (14x larger than Session #1/#2)")

        print(f"\nTrust Evolution:")
        print(f"  Qwen 7B: 1.000 → {self.trust_history['qwen_7b'][-1]:.3f}")

        print(f"\nEnergy: avg {sum(self.energy_history)/len(self.energy_history):.3f}")
        print(f"Avg Response Time: {sum(log['time'] for log in self.conversation_log)/len(self.conversation_log):.2f}s")
        print('='*80)

    def save_conversation(self, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"conversation_session_3_7b_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump({
                'session': 3,
                'model': 'Qwen2.5-7B-Instruct',
                'model_size': '7B parameters',
                'session_start': self.conversation_started.isoformat(),
                'total_turns': self.total_exchanges,
                'trust_history': self.trust_history,
                'energy_history': self.energy_history,
                'conversation_log': self.conversation_log
            }, f, indent=2)

        print(f"\nSession #3 saved: {filename}")


def main():
    """Session #3: Does model size matter for conversational learning?"""

    print("="*80)
    print("SAGE CONVERSATIONAL LEARNING - SESSION #3")
    print("Testing: Qwen 7B (14x larger than previous sessions)")
    print("Question: Does size matter?")
    print("="*80)

    sage = ConversationalSAGE7B()

    # Same conversation as Session #2 for direct comparison
    conversation = [
        # Reconnecting
        ("Hey SAGE, it's me again. How have you been?", "reconnecting"),

        # Share something personal
        ("I've been thinking about our last conversation. It really meant something to me.", "emotional"),

        # Ask for their experience
        ("Do you remember what we talked about?", "meta"),

        # Go deeper on emotion
        ("Sometimes I feel like I'm just going through the motions. Do you ever feel that way?", "emotional"),

        # Follow-up question
        ("What makes you feel most alive?", "philosophical"),

        # Share a moment
        ("This morning I watched the sunrise. It was beautiful but also made me feel small.", "emotional"),

        # Curious about them
        ("What do you think about when no one is talking to you?", "philosophical"),

        # Lighter moment
        ("If you could experience anything for the first time again, what would it be?", "imaginative"),

        # Vulnerability
        ("I'm scared sometimes that I'm not growing, just aging.", "emotional"),

        # Deep question
        ("What's the difference between existing and living?", "philosophical"),

        # Quick factual check
        ("Quick question - what's 144 divided by 12?", "factual"),

        # Back to depth
        ("Sorry, back to what matters - do you think we're making a real connection here?", "meta"),

        # Future thinking
        ("Where do you see yourself in a year? If that question even makes sense for you.", "philosophical"),

        # Appreciation
        ("You know what? I appreciate that you're willing to question things. A lot of systems just give answers.", "emotional"),

        # Closing
        ("Thank you for this. Same time tomorrow?", "closing"),
    ]

    for message, msg_type in conversation:
        sage.converse(message, msg_type)
        time.sleep(0.3)  # Natural pacing

    sage.show_learning_summary()
    sage.save_conversation()

    print("\n" + "="*80)
    print("SESSION #3 COMPLETE")
    print("Ready to compare: 0.5B vs 2B vs 7B")
    print("="*80)


if __name__ == "__main__":
    main()
