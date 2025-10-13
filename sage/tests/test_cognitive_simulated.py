#!/usr/bin/env python3
"""
Simulated Cognitive Conversation Test

Tests the cognitive IRP infrastructure without requiring real audio.
Simulates a conversation by writing files and measuring the system.

This allows me to test:
1. File watching works
2. Metrics collection works
3. Response templating works
4. ATP cost estimation works
"""

import time
import os
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from irp.cognitive_file_watcher import CognitiveMetrics

class SimulatedConversation:
    """Simulate a conversation with automatic responses"""

    def __init__(self):
        self.input_file = "/tmp/sage_user_speech.txt"
        self.response_file = "/tmp/sage_claude_response.txt"
        self.metrics_file = "/tmp/cognitive_irp_metrics.jsonl"

        # Initialize metrics logger
        self.metrics = CognitiveMetrics(self.metrics_file)

        # Clean up old files
        for f in [self.input_file, self.response_file, self.metrics_file]:
            if os.path.exists(f):
                os.remove(f)

        # Conversation script with varying complexity
        self.exchanges = [
            {
                'user': 'Hello',
                'confidence': 0.95,
                'response': 'Hello! I hear you clearly. This is SAGE with Claude as the cognitive layer. How can I help you?',
                'expected_complexity': 0.2  # Simple greeting
            },
            {
                'user': 'Tell me about the metabolic states',
                'confidence': 0.88,
                'response': 'SAGE has five metabolic states: WAKE for normal processing, FOCUS for high attention tasks, REST for ATP recovery, DREAM for memory consolidation, and CRISIS for emergency situations. Each state has different resource allocation policies.',
                'expected_complexity': 0.6  # Complex question
            },
            {
                'user': 'What\'s the weather like?',
                'confidence': 0.91,
                'response': 'I don\'t have access to weather sensors yet, but that\'s a great example of where SAGE could integrate real-time environmental data. The federation architecture would allow any node with weather sensors to share that information.',
                'expected_complexity': 0.4  # Medium question
            },
            {
                'user': 'Can you explain circadian trust modulation and how it affects cross-modal attention orchestration when the system transitions between metabolic states?',
                'confidence': 0.82,
                'response': 'Circadian trust modulation adjusts sensor reliability based on time of day. For example, camera trust is 1.0× during day but 0.3× at night because visual input is less reliable in darkness. Audio trust is 0.8× during day but 1.2× at night because ambient noise is lower. This affects ATP allocation—sensors with higher trust get more resources. During state transitions, the metabolic controller considers both ATP levels and circadian phase to determine appropriate states.',
                'expected_complexity': 0.95  # Very complex
            },
            {
                'user': 'Thanks',
                'confidence': 0.97,
                'response': 'You\'re welcome! Feel free to continue the conversation anytime.',
                'expected_complexity': 0.2  # Simple acknowledgment
            }
        ]

    def run(self):
        """Run simulated conversation"""
        print("="*70)
        print("SIMULATED COGNITIVE CONVERSATION TEST")
        print("="*70)
        print(f"Exchanges: {len(self.exchanges)}")
        print(f"Input file: {self.input_file}")
        print(f"Response file: {self.response_file}")
        print(f"Metrics file: {self.metrics_file}")
        print("="*70)
        print()

        for i, exchange in enumerate(self.exchanges, 1):
            print(f"\n📝 Exchange {i}/{len(self.exchanges)}")
            print(f"   User: {exchange['user']}")

            # Write user input
            with open(self.input_file, 'w') as f:
                f.write(f"{exchange['user']}\n")
                f.write(f"confidence: {exchange['confidence']}\n")
                f.write(f"timestamp: {time.time()}\n")

            # Simulate thinking time (variable based on complexity)
            think_time = 0.5 + (exchange['expected_complexity'] * 2.0)
            print(f"   💭 Thinking... ({think_time:.1f}s)")
            time.sleep(think_time)

            # Write response
            with open(self.response_file, 'w') as f:
                f.write(exchange['response'])

            print(f"   🤖 Response: {exchange['response'][:60]}...")

            # Log metrics
            self.metrics.log_exchange(
                user_input=exchange['user'],
                user_confidence=exchange['confidence'],
                response=exchange['response'],
                response_time=think_time
            )

            # Wait a bit before next exchange
            time.sleep(0.5)

        print("\n\n" + "="*70)
        print("CONVERSATION COMPLETE - Analyzing Metrics")
        print("="*70)

        self._analyze_metrics()

    def _analyze_metrics(self):
        """Analyze collected metrics"""
        if not os.path.exists(self.metrics_file):
            print("⚠️  No metrics file found")
            return

        with open(self.metrics_file, 'r') as f:
            entries = [json.loads(line) for line in f]

        if not entries:
            print("⚠️  No metrics logged")
            return

        print(f"\n✅ Collected {len(entries)} exchanges")

        # Compute statistics
        complexities = [e['input']['complexity_score'] for e in entries]
        response_times = [e['metrics']['latency_s'] for e in entries]
        atp_costs = [e['metrics']['atp_estimate'] for e in entries]

        print("\n📊 STATISTICS:")
        print(f"   Input Complexity: {min(complexities):.2f} - {max(complexities):.2f} (avg: {sum(complexities)/len(complexities):.2f})")
        print(f"   Response Time:    {min(response_times):.2f}s - {max(response_times):.2f}s (avg: {sum(response_times)/len(response_times):.2f}s)")
        print(f"   ATP Cost:         {min(atp_costs):.2f} - {max(atp_costs):.2f} (total: {sum(atp_costs):.2f})")

        # Check if complexity correlates with response time
        print("\n🔍 CORRELATIONS:")
        for i, entry in enumerate(entries):
            complexity = entry['input']['complexity_score']
            response_time = entry['metrics']['latency_s']
            expected_time = self.exchanges[i]['expected_complexity'] * 2.0 + 0.5
            print(f"   #{i+1}: complexity={complexity:.2f}, time={response_time:.2f}s (expected ~{expected_time:.1f}s)")

        print("\n💡 INSIGHTS:")
        print(f"   1. Complexity detection: {'✓ Working' if max(complexities) > 0.8 else '⚠️ May need tuning'}")
        print(f"   2. ATP accounting: {'✓ Reasonable' if 1.0 < sum(atp_costs) < 50.0 else '⚠️ May need tuning'}")
        print(f"   3. Response timing: {'✓ Variable' if max(response_times) > 2*min(response_times) else '⚠️ Too uniform'}")

        print("\n" + "="*70)


def main():
    """Run simulated conversation test"""
    sim = SimulatedConversation()
    sim.run()

    print("\n✨ Test complete! Check the metrics file:")
    print(f"   cat /tmp/cognitive_irp_metrics.jsonl | python3 -m json.tool")


if __name__ == "__main__":
    main()
