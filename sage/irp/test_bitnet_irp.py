#!/usr/bin/env python3
"""
Test BitNet IRP Plugin
Quick validation that SAGE can orchestrate BitNet reasoning
"""

from plugins.bitnet_irp import BitNetIRP

# Test prompts
test_prompts = [
    "What is 2+2?",
    "What is consciousness?",
    "Should I step forward?"
]

print("=" * 80)
print("SAGE → BitNet IRP Test")
print("Testing attention orchestration with ultra-compressed reasoning")
print("=" * 80)

# Create BitNet reasoning resource
bitnet = BitNetIRP(use_gpu=False)  # CPU for reliability

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n[{i}/{len(test_prompts)}] Task: {prompt}")
    print("-" * 80)

    # IRP protocol
    init_config = {'max_tokens': 100, 'temperature': 0.7}
    bitnet.initialize(init_config)

    # Preprocess
    x_0 = bitnet.preprocess(prompt)
    print(f"Preprocessed: {list(x_0.keys())}")

    # Step (single-shot for BitNet)
    x_1 = bitnet.step(x_0, t=0)
    print(f"Response: {x_1['response'][:200]}...")

    # Energy (did it work?)
    e = bitnet.energy(x_1, t=0)
    print(f"Energy: {e:.3f} {'✓ LOW (success)' if e < 0.3 else '✗ HIGH (issue)'}")

    # Halt?
    should_halt = bitnet.halt([e], t=0)
    print(f"Halt: {should_halt}")

    # Cost metrics
    cost = bitnet.get_cost()
    print(f"Cost: {cost['time_sec']:.2f}s, {cost['tokens']} tokens, ~{cost['memory_mb_estimate']:.0f}MB")

print("\n" + "=" * 80)
print("BitNet is now available as an IRP plugin")
print("SAGE can orchestrate it alongside other reasoning modes")
print("=" * 80)
