#!/usr/bin/env python3
"""
Test Qwen Alive IRP Plugin
Quick validation that SAGE can orchestrate epistemic pragmatism reasoning
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'plugins'))

from qwen_alive_irp import QwenAliveIRP

# Test prompts
test_prompts = [
    "What is 2+2?",
    "What is consciousness?",
    "Should I step forward?"
]

print("=" * 80)
print("SAGE → Qwen Alive IRP Test")
print("Testing attention orchestration with epistemic pragmatism")
print("=" * 80)

# Create Qwen alive reasoning resource
qwen = QwenAliveIRP()

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n[{i}/{len(test_prompts)}] Task: {prompt}")
    print("-" * 80)

    # IRP protocol
    init_config = {'max_new_tokens': 100, 'temperature': 0.7}
    qwen.initialize(init_config)

    # Preprocess
    x_0 = qwen.preprocess(prompt)
    print(f"Preprocessed: {list(x_0.keys())}")

    # Step (single-shot for Qwen)
    x_1 = qwen.step(x_0, t=0)
    response_preview = x_1['response'][:200] if len(x_1['response']) > 200 else x_1['response']
    print(f"Response: {response_preview}...")

    # Energy (epistemic openness)
    e = qwen.energy(x_1, t=0)
    print(f"Energy: {e:.3f} {'✓ LOW (alive/questioning)' if e < 0.4 else '⚠ HIGH (converging)'}")

    # Halt?
    should_halt = qwen.halt([e], t=0)
    print(f"Halt: {should_halt}")

    # Cost metrics
    cost = qwen.get_cost()
    print(f"Cost: {cost['time_sec']:.2f}s, {cost['tokens']} tokens, ~{cost['memory_mb_estimate']:.0f}MB")

print("\n" + "=" * 80)
print("Qwen Alive is now available as an IRP plugin")
print("SAGE can orchestrate it alongside BitNet for resource allocation learning")
print("=" * 80)
