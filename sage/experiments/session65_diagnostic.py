#!/usr/bin/env python3
"""
Session 65: Diagnostic - Why MRH substitution isn't triggering

Debug the extended simulation to understand why no substitutions occur
even though trust < 0.3.

Created: 2025-12-18
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.web4.context_aware_identity_bridge import ContextAwareIdentityBridge
from collections import Counter

# Initialize
bridge = ContextAwareIdentityBridge(instance="thinker", network="testnet", n_contexts=3)
np.random.seed(42)

# Create specialists
specialists = {
    0: [42, 17, 88],
    1: [99, 55, 120],
    2: [1, 63, 110]
}

print("Creating specialists...")
for context_id, expert_ids in specialists.items():
    for expert_id in expert_ids:
        embeddings = np.random.randn(10, 8)
        embeddings[:, context_id] += 3.0
        contexts = bridge.discover_expert_contexts(expert_id, embeddings)
        print(f"  Expert {expert_id}: {dict(Counter(contexts))} (specialist in context_{context_id})")

        for _ in range(5):
            bridge.update_trust_history(expert_id, context_id, 0.80 + np.random.randn() * 0.05)

# Create monopoly experts
monopoly_experts = [73, 114, 95, 106]
print("\nCreating monopoly experts...")
for expert_id in monopoly_experts:
    embeddings = np.random.randn(18, 8)
    embeddings[:6, 0] += 1.5
    embeddings[6:12, 1] += 1.5
    embeddings[12:, 2] += 1.5
    contexts = bridge.discover_expert_contexts(expert_id, embeddings)
    print(f"  Expert {expert_id}: {dict(Counter(contexts))} (generalist)")

    for context_id in range(3):
        for i in range(6):
            trust = 0.32 - (i * 0.03)
            bridge.update_trust_history(expert_id, context_id, trust)

# Check what's in expert_contexts
print(f"\nExpert contexts registered: {sorted(bridge.expert_contexts.keys())}")

# Test context overlap between monopoly expert and specialist
print("\nTesting context overlap...")
for mono_expert in monopoly_experts[:2]:  # Test first 2
    print(f"\nMonopoly expert {mono_expert}:")

    for context_id, specialist_ids in specialists.items():
        print(f"  Context {context_id}:")

        for specialist_id in specialist_ids[:2]:  # Test 2 specialists per context
            if specialist_id in bridge.expert_contexts:
                overlap, shared = bridge.compute_context_overlap(mono_expert, specialist_id)
                print(f"    vs Specialist {specialist_id}: overlap={overlap:.3f}, shared={shared}")
            else:
                print(f"    vs Specialist {specialist_id}: NOT IN EXPERT_CONTEXTS")

# Check trust values
print("\nTrust values for monopoly experts:")
for expert_id in monopoly_experts:
    print(f"  Expert {expert_id}:")
    for context_id in range(3):
        key = (expert_id, context_id)
        if key in bridge.trust_history:
            history = bridge.trust_history[key]
            if history:
                print(f"    Context {context_id}: {history[-1]:.3f}")

print("\nTrust values for specialists:")
for context_id, specialist_ids in specialists.items():
    print(f"  Context {context_id}:")
    for specialist_id in specialist_ids:
        key = (specialist_id, context_id)
        if key in bridge.trust_history:
            history = bridge.trust_history[key]
            if history:
                print(f"    Expert {specialist_id}: {history[-1]:.3f}")

print("\nDiagnostic complete!")
