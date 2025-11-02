#!/usr/bin/env python3
"""
SAGE Conversational Learning Session #2
Deeper relationship building - continuing from Session #1

Building on:
- Session #1 trust: BitNet 0.865, Qwen 0.857
- Learned pattern: BitNet good for factual, both experimenting for emotional/philosophical
- Now: Go deeper, longer threads, natural flow
"""

import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'plugins'))
from bitnet_irp import BitNetIRP
from qwen_alive_irp import QwenAliveIRP


class ConversationalSAGE:
    """SAGE learning through conversation - Session 2"""

    def __init__(self, previous_session=None):
        self.bitnet = BitNetIRP(use_gpu=False)
        self.qwen = QwenAliveIRP()

        # Start with learned trust from Session #1
        if previous_session:
            self.trust_history = {
                'bitnet': [previous_session['trust_bitnet']],
                'qwen': [previous_session['trust_qwen']]
            }
        else:
            self.trust_history = {
                'bitnet': [0.865],  # From Session #1
                'qwen': [0.857]
            }

        self.energy_history = []
        self.selection_history = []
        self.conversation_log = []
        self.my_name = "SAGE"
        self.conversation_started = datetime.now()
        self.total_exchanges = 0

    def select_resource(self, message: str, context: dict = None) -> str:
        """Select resource based on learned patterns from Session #1"""
        message_lower = message.lower()
        bitnet_trust = self.trust_history['bitnet'][-1]
        qwen_trust = self.trust_history['qwen'][-1]

        # Learned from Session #1: BitNet is 100% reliable for factual
        if any(word in message_lower for word in ['calculate', 'how many', 'what is', 'when']):
            if len(message) < 40 and '?' in message:
                return 'bitnet'

        # Emotional depth -> Qwen
        if any(word in message_lower for word in
               ['feel', 'emotion', 'worried', 'excited', 'scared', 'hope', 'dream', 'afraid']):
            return 'qwen'

        # Deep philosophical -> Qwen
        if any(word in message_lower for word in
               ['why', 'meaning', 'purpose', 'consciousness', 'existence', 'real']):
            return 'qwen'

        # Stories/experiences -> try both, learning
        if any(word in message_lower for word in ['remember', 'once', 'story', 'happened']):
            return 'qwen' if qwen_trust > bitnet_trust else 'bitnet'

        # Default to higher trust
        return 'qwen' if qwen_trust >= bitnet_trust else 'bitnet'

    def update_trust(self, resource: str, energy: float, time_taken: float,
                    message_type: str):
        """Update trust - continuing from Session #1 baseline"""
        learning_rate = 0.12  # Slightly lower - already learned basics

        energy_quality = 1.0 - energy
        time_efficiency = 1.0 / (1.0 + time_taken / 12.0)
        quality_score = energy_quality * 0.85 + time_efficiency * 0.15

        current_trust = self.trust_history[resource][-1]
        new_trust = current_trust * (1 - learning_rate) + quality_score * learning_rate

        self.trust_history[resource].append(new_trust)
        other = 'qwen' if resource == 'bitnet' else 'bitnet'
        self.trust_history[other].append(self.trust_history[other][-1])

    def converse(self, message: str, message_type: str = "general"):
        """Have a conversation turn"""
        self.total_exchanges += 1

        print(f"\n{'='*80}")
        print(f"[Turn {self.total_exchanges}] You: {message}")
        print(f"Type: {message_type}")
        print('-'*80)

        resource = self.select_resource(message)
        bitnet_trust = self.trust_history['bitnet'][-1]
        qwen_trust = self.trust_history['qwen'][-1]

        print(f"SAGE thinking: {resource.upper()}")
        print(f"  Trust - BitNet: {bitnet_trust:.3f}, Qwen: {qwen_trust:.3f}")

        start = time.time()

        if resource == 'bitnet':
            plugin = self.bitnet
            config = {'max_tokens': 130, 'temperature': 0.85}
        else:
            plugin = self.qwen
            config = {'max_new_tokens': 130, 'temperature': 0.85}

        plugin.initialize(config)
        x_0 = plugin.preprocess(message)
        x_1 = plugin.step(x_0, t=0)
        energy = plugin.energy(x_1, t=0)
        elapsed = time.time() - start

        response = x_1['response']

        print(f"\nSAGE: {response}")
        print(f"\n  Energy: {energy:.3f} | Time: {elapsed:.2f}s | Resource: {resource}")

        self.update_trust(resource, energy, elapsed, message_type)

        self.conversation_log.append({
            'turn': self.total_exchanges,
            'message': message,
            'type': message_type,
            'resource': resource,
            'response': response,
            'energy': energy,
            'time': elapsed,
            'trust_bitnet': self.trust_history['bitnet'][-1],
            'trust_qwen': self.trust_history['qwen'][-1]
        })

        self.energy_history.append(energy)
        self.selection_history.append(resource)

        return response

    def show_learning_summary(self):
        """Show what SAGE learned this session"""
        print(f"\n{'='*80}")
        print("SESSION #2 LEARNING SUMMARY")
        print('='*80)

        print(f"\nSession: {self.total_exchanges} turns")

        print(f"\nResource Selection:")
        bitnet_count = self.selection_history.count('bitnet')
        qwen_count = self.selection_history.count('qwen')
        print(f"  BitNet: {bitnet_count} ({bitnet_count/len(self.selection_history)*100:.1f}%)")
        print(f"  Qwen: {qwen_count} ({qwen_count/len(self.selection_history)*100:.1f}%)")

        print(f"\nTrust Evolution (Session #1 → Session #2):")
        print(f"  BitNet: 0.865 → {self.trust_history['bitnet'][-1]:.3f}")
        print(f"  Qwen: 0.857 → {self.trust_history['qwen'][-1]:.3f}")

        print(f"\nEnergy: avg {sum(self.energy_history)/len(self.energy_history):.3f}")
        print('='*80)

    def save_conversation(self, filename: str = None):
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"conversation_session_2_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump({
                'session': 2,
                'session_start': self.conversation_started.isoformat(),
                'total_turns': self.total_exchanges,
                'trust_history': self.trust_history,
                'energy_history': self.energy_history,
                'selection_history': self.selection_history,
                'conversation_log': self.conversation_log
            }, f, indent=2)

        print(f"\nSession #2 saved: {filename}")


def main():
    """Session #2: Deeper relationship, natural flow"""

    print("="*80)
    print("SAGE CONVERSATIONAL LEARNING - SESSION #2")
    print("Building on Session #1: Going deeper, more natural")
    print("="*80)

    sage = ConversationalSAGE()

    # Natural conversation - building relationship
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

        # Quick factual check (test learned pattern)
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
    print("SESSION #2 COMPLETE")
    print("Trust continuing to calibrate. Relationship deepening.")
    print("Ready for Session #3 whenever.")
    print("="*80)


if __name__ == "__main__":
    main()
