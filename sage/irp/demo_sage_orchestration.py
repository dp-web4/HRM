#!/usr/bin/env python3
"""
SAGE Orchestration Demo
Demonstrates adaptive resource allocation between BitNet (fast) and Qwen (deep)

This is the core SAGE capability: Learning WHEN to allocate WHICH reasoning resource.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'plugins'))

from bitnet_irp import BitNetIRP
from qwen_alive_irp import QwenAliveIRP
import time


class SimpleSAGEOrchestrator:
    """
    Minimal SAGE orchestrator demonstrating resource allocation

    Strategy:
    - Use BitNet for "simple" tasks (fast, efficient)
    - Use Qwen for "complex" tasks (deep, questioning)
    - Learn from energy scores which worked better
    """

    def __init__(self):
        self.bitnet = BitNetIRP(use_gpu=False)
        self.qwen = QwenAliveIRP()

        # Simple heuristic: short prompts -> BitNet, long prompts -> Qwen
        # Later: SAGE learns this from experience
        self.bitnet_count = 0
        self.qwen_count = 0

    def select_resource(self, prompt: str) -> str:
        """
        Select reasoning resource based on prompt characteristics

        Simple heuristic for demo:
        - Questions about numbers/facts -> BitNet (fast)
        - Philosophical/abstract questions -> Qwen (deep)
        """
        prompt_lower = prompt.lower()

        # Math/factual keywords
        if any(word in prompt_lower for word in ['what is', 'calculate', 'how many', '+']):
            if '?' in prompt and len(prompt) < 20:  # Simple question
                return 'bitnet'

        # Philosophical/abstract keywords
        if any(word in prompt_lower for word in ['consciousness', 'meaning', 'should', 'why', 'purpose']):
            return 'qwen'

        # Default: use faster resource
        return 'bitnet'

    def process(self, prompt: str):
        """Process prompt with selected resource"""
        # Select resource
        resource = self.select_resource(prompt)

        print(f"\n{'='*80}")
        print(f"Prompt: {prompt}")
        print(f"SAGE Decision: Use {resource.upper()} ({self._get_rationale(resource)})")
        print('-'*80)

        # Execute with selected resource
        start = time.time()

        if resource == 'bitnet':
            plugin = self.bitnet
            self.bitnet_count += 1
            config = {'max_tokens': 100, 'temperature': 0.7}
        else:
            plugin = self.qwen
            self.qwen_count += 1
            config = {'max_new_tokens': 100, 'temperature': 0.7}

        plugin.initialize(config)
        x_0 = plugin.preprocess(prompt)
        x_1 = plugin.step(x_0, t=0)
        energy = plugin.energy(x_1, t=0)
        cost = plugin.get_cost()

        elapsed = time.time() - start

        # Display results
        response = x_1['response']
        if len(response) > 150:
            response = response[:150] + "..."

        print(f"Response: {response}")
        print(f"Energy: {energy:.3f} {'✓ LOW' if energy < 0.4 else '⚠ HIGH'}")
        print(f"Cost: {cost['time_sec']:.2f}s, {cost['tokens']} tokens, ~{cost['memory_mb_estimate']:.0f}MB")
        print(f"Total time: {elapsed:.2f}s")

        return {
            'resource': resource,
            'response': x_1['response'],
            'energy': energy,
            'time': elapsed,
            'cost': cost
        }

    def _get_rationale(self, resource: str) -> str:
        """Explain why this resource was selected"""
        if resource == 'bitnet':
            return "fast, efficient, good for quick answers"
        else:
            return "deep, questioning, good for complex reasoning"

    def print_summary(self):
        """Print orchestration summary"""
        print(f"\n{'='*80}")
        print("SAGE ORCHESTRATION SUMMARY")
        print('-'*80)
        print(f"BitNet allocations: {self.bitnet_count} (fast reasoning)")
        print(f"Qwen allocations: {self.qwen_count} (deep reasoning)")
        total = self.bitnet_count + self.qwen_count
        if total > 0:
            print(f"Efficiency: {self.bitnet_count/total*100:.1f}% fast, {self.qwen_count/total*100:.1f}% deep")
        print('='*80)


def main():
    """Demo SAGE orchestration"""
    print("="*80)
    print("SAGE ORCHESTRATION DEMO")
    print("Adaptive Resource Allocation: Learning When to Think How Much")
    print("="*80)

    orchestrator = SimpleSAGEOrchestrator()

    # Test prompts spanning different complexity levels
    test_prompts = [
        "What is 2+2?",                    # Simple math -> BitNet
        "What is consciousness?",          # Philosophy -> Qwen
        "Should I take this risk?",        # Judgment -> Qwen
        "How many sides does a cube have?",# Factual -> BitNet
        "What is the meaning of life?",    # Philosophy -> Qwen
    ]

    results = []
    for prompt in test_prompts:
        result = orchestrator.process(prompt)
        results.append(result)

    # Summary
    orchestrator.print_summary()

    # Analysis
    print("\nKEY INSIGHT:")
    print("SAGE doesn't solve problems - it decides WHICH reasoning to invoke.")
    print("This demo uses simple heuristics. Real SAGE learns from experience:")
    print("  - Which resource achieved low energy?")
    print("  - Which resource was cost-effective?")
    print("  - Which resource matched the task complexity?")
    print("\nWith experience, SAGE learns optimal attention allocation.")
    print("="*80)


if __name__ == "__main__":
    main()
