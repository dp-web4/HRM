#!/usr/bin/env python3
"""Debug MRH substitution to understand why it's not triggering."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.core.mrh_expert_selector import create_mrh_selector
from collections import Counter

# Initialize selector
selector = create_mrh_selector(num_experts=128)
np.random.seed(42)

# Register one specialist
expert_id = 42
embeddings = np.random.randn(10, 8)
embeddings[:, 0] += 3.0
contexts = selector.register_expert_contexts(expert_id, embeddings)
print(f"Expert {expert_id} contexts: {dict(Counter(contexts))}")

# Set high trust for specialist in context 0
for _ in range(5):
    selector.update_trust_for_expert(expert_id, 0, 0.85)

# Register one monopoly expert with LOW trust
monopoly_id = 73
embeddings = np.random.randn(18, 8)
embeddings[:6, 0] += 1.5
contexts = selector.register_expert_contexts(monopoly_id, embeddings)
print(f"Expert {monopoly_id} contexts: {dict(Counter(contexts))}")

# Set LOW trust for monopoly expert in context 0
for i in range(6):
    trust = 0.32 - (i * 0.03)
    selector.update_trust_for_expert(monopoly_id, 0, trust)

# Check trust scores
print(f"\nTrust scores:")
print(f"  Expert {expert_id} in context 0: {selector._get_context_trust(expert_id, 0):.3f}")
print(f"  Expert {monopoly_id} in context 0: {selector._get_context_trust(monopoly_id, 0):.3f}")
print(f"  Threshold: {selector.low_trust_threshold}")

# Check if experts are in bridge
print(f"\nExperts in bridge:")
print(f"  Expert {expert_id}: {expert_id in selector.bridge.expert_contexts}")
print(f"  Expert {monopoly_id}: {monopoly_id in selector.bridge.expert_contexts}")

# Check context overlap
if monopoly_id in selector.bridge.expert_contexts and expert_id in selector.bridge.expert_contexts:
    overlap, shared = selector.bridge.compute_context_overlap(monopoly_id, expert_id)
    print(f"\nContext overlap:")
    print(f"  Overlap: {overlap:.3f}")
    print(f"  Shared contexts: {shared}")
    print(f"  Overlap threshold: {selector.overlap_threshold}")

# Try finding alternative
print(f"\nTrying to find MRH alternative for expert {monopoly_id} in context 0...")
alternative = selector._find_mrh_alternative(monopoly_id, 0, [expert_id, monopoly_id])
print(f"  Result: {alternative}")

if alternative is None:
    print("\n❌ No alternative found! Debugging why...")

    # Manual check
    if monopoly_id not in selector.bridge.expert_contexts:
        print("  - Monopoly expert not in bridge.expert_contexts")
    else:
        print(f"  - Monopoly expert IS in bridge ({selector.bridge.expert_contexts[monopoly_id][:5]}...)")

        for other_expert in [expert_id]:
            if other_expert not in selector.bridge.expert_contexts:
                print(f"  - Expert {other_expert} not in bridge.expert_contexts")
                continue

            overlap, shared = selector.bridge.compute_context_overlap(monopoly_id, other_expert)
            print(f"  - Overlap with expert {other_expert}: {overlap:.3f}")
            print(f"  - Shared contexts: {shared}")

            # Check context string format
            context_str = f"context_0"
            print(f"  - Looking for '{context_str}' in {shared}")
            print(f"  - Found: {context_str in shared}")

            if overlap >= selector.overlap_threshold:
                print(f"  ✓ Overlap {overlap:.3f} >= threshold {selector.overlap_threshold}")

                if context_str in shared:
                    print(f"  ✓ Context '{context_str}' in shared contexts")

                    alt_trust = selector._get_context_trust(other_expert, 0)
                    current_trust = selector._get_context_trust(monopoly_id, 0)

                    print(f"  - Alternative trust: {alt_trust:.3f}")
                    print(f"  - Current trust: {current_trust:.3f}")
                    print(f"  - Alternative better: {alt_trust > current_trust}")
                else:
                    print(f"  ❌ Context '{context_str}' NOT in shared contexts {shared}")
            else:
                print(f"  ❌ Overlap {overlap:.3f} < threshold {selector.overlap_threshold}")
else:
    alt_id, alt_trust, overlap = alternative
    print(f"✓ Found alternative: Expert {alt_id} (trust={alt_trust:.3f}, overlap={overlap:.3f})")
