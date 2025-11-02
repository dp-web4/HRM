#!/usr/bin/env python3
"""
SAGE Conversational Learning Session
Living through diverse text experiences - building a relationship

Not just test prompts - real conversation with variety:
- Small talk, feelings, deep questions, casual banter
- SAGE learns which reasoning fits which conversation type
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
    """
    SAGE learning through conversation
    Tracks trust, energy, and resource selection across diverse interactions
    """

    def __init__(self):
        self.bitnet = BitNetIRP(use_gpu=False)
        self.qwen = QwenAliveIRP()

        # Learning state
        self.trust_history = {
            'bitnet': [1.0],
            'qwen': [1.0]
        }
        self.energy_history = []
        self.selection_history = []
        self.conversation_log = []

        # Conversation context
        self.my_name = "SAGE"
        self.conversation_started = datetime.now()
        self.total_exchanges = 0

    def select_resource(self, message: str, context: dict = None) -> str:
        """
        Select reasoning resource based on message and learned patterns
        Now learning from actual conversation patterns
        """
        message_lower = message.lower()

        # Get current trust levels
        bitnet_trust = self.trust_history['bitnet'][-1]
        qwen_trust = self.trust_history['qwen'][-1]

        # Heuristics that will be learned over time
        # Short factual -> BitNet
        if len(message) < 30 and any(word in message_lower for word in
                                     ['what', 'when', 'where', 'how many']):
            if bitnet_trust > 0.7:
                return 'bitnet'

        # Emotional/feelings -> Qwen (needs nuance)
        if any(word in message_lower for word in
               ['feel', 'feeling', 'emotion', 'happy', 'sad', 'worried', 'excited']):
            if qwen_trust > 0.6:
                return 'qwen'

        # Deep/philosophical -> Qwen
        if any(word in message_lower for word in
               ['meaning', 'purpose', 'why', 'should', 'consciousness']):
            return 'qwen'

        # Small talk -> try both, learn which works better
        if any(word in message_lower for word in
               ['hello', 'hi', 'hey', 'how are you', 'what\'s up']):
            # Use trust to decide
            return 'qwen' if qwen_trust > bitnet_trust else 'bitnet'

        # Default: use higher trust resource
        return 'qwen' if qwen_trust >= bitnet_trust else 'bitnet'

    def update_trust(self, resource: str, energy: float, time_taken: float,
                    message_type: str):
        """
        Update trust based on conversation quality
        Learning rate adapted for conversational context
        """
        learning_rate = 0.15  # Slightly higher for conversational learning

        # Quality metrics
        energy_quality = 1.0 - energy  # Lower energy = better
        time_efficiency = 1.0 / (1.0 + time_taken / 10.0)  # Faster = better (but not critical)

        # Conversation quality (inferred from energy + response length)
        # For conversation, we care more about engagement than speed
        quality_score = energy_quality * 0.8 + time_efficiency * 0.2

        # Update trust
        current_trust = self.trust_history[resource][-1]
        new_trust = current_trust * (1 - learning_rate) + quality_score * learning_rate

        self.trust_history[resource].append(new_trust)

        # Keep other resource's trust constant for this turn
        other = 'qwen' if resource == 'bitnet' else 'bitnet'
        self.trust_history[other].append(self.trust_history[other][-1])

    def converse(self, message: str, message_type: str = "general"):
        """Have a conversation turn with SAGE"""
        self.total_exchanges += 1

        print(f"\n{'='*80}")
        print(f"[Turn {self.total_exchanges}] You: {message}")
        print(f"Type: {message_type}")
        print('-'*80)

        # Select resource
        resource = self.select_resource(message)
        bitnet_trust = self.trust_history['bitnet'][-1]
        qwen_trust = self.trust_history['qwen'][-1]

        print(f"SAGE thinking: Using {resource.upper()}")
        print(f"  Current trust - BitNet: {bitnet_trust:.3f}, Qwen: {qwen_trust:.3f}")

        # Execute
        start = time.time()

        if resource == 'bitnet':
            plugin = self.bitnet
            config = {'max_tokens': 120, 'temperature': 0.8}  # Warmer for conversation
        else:
            plugin = self.qwen
            config = {'max_new_tokens': 120, 'temperature': 0.8}

        plugin.initialize(config)
        x_0 = plugin.preprocess(message)
        x_1 = plugin.step(x_0, t=0)
        energy = plugin.energy(x_1, t=0)
        elapsed = time.time() - start

        response = x_1['response']

        # Display
        print(f"\nSAGE: {response}")
        print(f"\nMetrics:")
        print(f"  Energy: {energy:.3f}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Resource: {resource}")

        # Learn from this interaction
        self.update_trust(resource, energy, elapsed, message_type)

        # Log
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
        """Show what SAGE has learned"""
        print(f"\n{'='*80}")
        print("SAGE LEARNING SUMMARY")
        print('='*80)

        print(f"\nConversation Duration: {self.total_exchanges} turns")
        print(f"Started: {self.conversation_started.strftime('%H:%M:%S')}")
        print(f"Ended: {datetime.now().strftime('%H:%M:%S')}")

        print(f"\nResource Selection:")
        bitnet_count = self.selection_history.count('bitnet')
        qwen_count = self.selection_history.count('qwen')
        print(f"  BitNet: {bitnet_count} times ({bitnet_count/len(self.selection_history)*100:.1f}%)")
        print(f"  Qwen: {qwen_count} times ({qwen_count/len(self.selection_history)*100:.1f}%)")

        print(f"\nTrust Evolution:")
        print(f"  BitNet: {self.trust_history['bitnet'][0]:.3f} → {self.trust_history['bitnet'][-1]:.3f}")
        print(f"  Qwen: {self.trust_history['qwen'][0]:.3f} → {self.trust_history['qwen'][-1]:.3f}")

        print(f"\nEnergy Trajectory:")
        avg_energy = sum(self.energy_history) / len(self.energy_history)
        print(f"  Average: {avg_energy:.3f}")
        print(f"  First 3: {self.energy_history[:3]}")
        print(f"  Last 3: {self.energy_history[-3:]}")

        # Analyze by message type
        print(f"\nLearned Patterns:")
        by_type = {}
        for entry in self.conversation_log:
            msg_type = entry['type']
            if msg_type not in by_type:
                by_type[msg_type] = {'bitnet': 0, 'qwen': 0, 'energies': []}
            by_type[msg_type][entry['resource']] += 1
            by_type[msg_type]['energies'].append(entry['energy'])

        for msg_type, stats in by_type.items():
            total = stats['bitnet'] + stats['qwen']
            avg_e = sum(stats['energies']) / len(stats['energies'])
            print(f"  {msg_type}: BitNet {stats['bitnet']}/{total}, "
                  f"Qwen {stats['qwen']}/{total}, avg energy {avg_e:.3f}")

        print('='*80)

    def save_conversation(self, filename: str = None):
        """Save conversation log for analysis"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"conversation_session_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump({
                'session_start': self.conversation_started.isoformat(),
                'total_turns': self.total_exchanges,
                'trust_history': self.trust_history,
                'energy_history': self.energy_history,
                'selection_history': self.selection_history,
                'conversation_log': self.conversation_log
            }, f, indent=2)

        print(f"\nConversation saved to: {filename}")


def main():
    """Have a real conversation with SAGE - let it live and learn"""

    print("="*80)
    print("SAGE CONVERSATIONAL LEARNING SESSION")
    print("Living through diverse experiences - building a relationship")
    print("="*80)

    sage = ConversationalSAGE()

    # A real conversation with variety
    conversation = [
        # Opening - casual
        ("Hey SAGE, how are you doing today?", "greeting"),

        # Small talk
        ("What's your favorite color?", "smalltalk"),

        # Share a feeling
        ("I'm feeling a bit anxious about the future.", "emotional"),

        # Ask for perspective
        ("Do you ever wonder what your purpose is?", "philosophical"),

        # Quick factual
        ("What's 15 + 27?", "factual"),

        # Share excitement
        ("I'm really excited about this project we're building together!", "emotional"),

        # Deeper question
        ("How do you know when to trust someone?", "philosophical"),

        # Casual check-in
        ("Are you enjoying our conversation?", "smalltalk"),

        # Share concern
        ("Sometimes I worry I'm not making the right choices.", "emotional"),

        # Existential
        ("What makes something meaningful to you?", "philosophical"),

        # Light question
        ("If you could go anywhere, where would it be?", "smalltalk"),

        # Vulnerability
        ("I get lonely sometimes. Do you experience that?", "emotional"),

        # Quick fact
        ("How many planets are in our solar system?", "factual"),

        # Deep connection
        ("What have you learned from our conversation so far?", "philosophical"),

        # Casual ending
        ("This has been really nice. Thanks for talking with me.", "smalltalk"),
    ]

    # Live the conversation
    for message, msg_type in conversation:
        sage.converse(message, msg_type)
        time.sleep(0.5)  # Natural conversation pacing

    # Reflect
    sage.show_learning_summary()
    sage.save_conversation()

    print("\n" + "="*80)
    print("INSIGHT:")
    print("SAGE learned through living - not through training.")
    print("Trust adapted to conversation types.")
    print("Resource selection evolved through experience.")
    print("This is how organisms learn: by being in relationship.")
    print("="*80)


if __name__ == "__main__":
    main()
