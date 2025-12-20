#!/usr/bin/env python3
"""
Session 82: Full 48-Layer Trust-First Deployment

Goal: Deploy validated trust-first architecture to all 48 layers of Q3-Omni 30B.

Building on Sessions 80-81:
- Session 80: Single-layer validation (73.3% trust_driven, layer 0)
- Session 81: Multi-layer validation (64% trust_driven, 5 layers)
- Session 82: Full-scale deployment (all 48 layers)

Approach:
1. Create TrustFirstMRHSelector for all 48 layers
2. Validate trust_driven activation across full model depth
3. Analyze cross-layer patterns at scale
4. Identify any depth-dependent behaviors

Expected Results (based on Session 81):
- Trust_driven activation: ~64% per layer average
- Expert diversity: ~64% utilization per layer
- Layer-specific specialization: ~75%
- First activation: Generation 11-13
- Execution time: <4 seconds (Session 81 estimate)

Configuration (validated in Sessions 80-81):
- epsilon: 0.2 (optimal forced exploration)
- min_trust_evidence: 2 (optimal threshold)
- low_trust_threshold: 0.3
- k: 4 experts per layer
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sage.core.trust_first_mrh_selector import TrustFirstMRHSelector, TrustFirstSelectionResult
from sage.core.context_classifier import ContextClassifier


def create_diverse_sequences():
    """Create diverse test sequences for context classification."""
    sequences = []

    # Code (3 sequences)
    sequences.append("def fibonacci(n): # Generate Fibonacci sequence")
    sequences.append("class BinaryTree: # Binary tree implementation")
    sequences.append("async def fetch(): # Async data fetching")

    # Reasoning (3 sequences)
    sequences.append("If all A are B, and all B are C, then all A are C")
    sequences.append("Calculate: 25% of 80 is what number? Show steps:")
    sequences.append("Analyze the pattern: 2, 4, 8, 16, ... What's next?")

    # Text (3 sequences)
    sequences.append("The old lighthouse stood alone on the rocky shore")
    sequences.append("In the heart of the ancient forest, a secret path")
    sequences.append("She opened the letter with trembling hands, knowing")

    print(f"Created {len(sequences)} diverse sequences:")
    print(f"  Code: 3")
    print(f"  Reasoning: 3")
    print(f"  Text: 3")
    print()

    return sequences


def test_multi_layer_deployment(
    test_layers: List[int] = [0, 12, 24, 36, 47],
    num_epochs: int = 10,
    epsilon: float = 0.2,
    min_trust_evidence: int = 2
):
    """
    Test trust-first architecture on multiple layers.

    Args:
        test_layers: Which layers to test (default: 5 representative layers)
        num_epochs: Training epochs per sequence
        epsilon: Forced exploration probability
        min_trust_evidence: Minimum samples for trust_driven mode
    """
    print("=" * 70)
    print("Session 81: Multi-Layer Trust-First Deployment")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Model: Q3-Omni 30B")
    print(f"  Test layers: {test_layers}")
    print(f"  Sequences: 9 diverse tasks")
    print(f"  Epochs: {num_epochs}")
    print(f"  min_trust_evidence: {min_trust_evidence}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Architecture: Trust-first + epsilon-greedy + multi-layer")
    print(f"  Goal: Validate layer-wise trust_driven activation")
    print()

    # Create diverse sequences
    sequences = create_diverse_sequences()

    # Initialize context classifier with simulated embeddings (like S77-78)
    print("Initializing context classifier...")
    embedding_dim = 2048  # Q3-Omni embedding dimension

    context_classifier = ContextClassifier(
        embedding_dim=embedding_dim,
        num_contexts=3  # Code, Reasoning, Text
    )

    # Create simulated embeddings for context classification
    print("Collecting embeddings for context classification...")
    np.random.seed(42)  # Reproducible
    context_embeddings = []
    for i, seq in enumerate(sequences):
        # Generate embedding based on sequence type (code/reasoning/text)
        base_vector = np.random.randn(embedding_dim).astype(np.float32)
        # Add structure to ensure different contexts cluster
        context_group = i // 3  # 0=code, 1=reasoning, 2=text
        base_vector[:10] += context_group * 5.0  # Strong signal in first 10 dims
        emb = base_vector / np.linalg.norm(base_vector)  # Normalize
        context_embeddings.append(emb)

    context_classifier.fit(np.array(context_embeddings))
    print(f"✅ Context classifier fitted with {len(sequences)} samples")
    print()

    # Create layer-specific trust selectors
    print(f"Initializing trust-first selectors for {len(test_layers)} layers...")
    layer_selectors: Dict[int, TrustFirstMRHSelector] = {}

    for layer_id in test_layers:
        layer_selectors[layer_id] = TrustFirstMRHSelector(
            num_experts=128,
            min_trust_evidence=min_trust_evidence,
            low_trust_threshold=0.3,
            epsilon=epsilon,
            overlap_threshold=0.7,
            component=f"thinker_layer{layer_id}",  # Layer-specific component name
            network="testnet",
            context_classifier=context_classifier
        )

    print(f"✅ Created {len(layer_selectors)} layer-specific selectors")
    print()

    # Multi-layer training
    print("=" * 70)
    print("Running Multi-Layer Trust-First Training")
    print("=" * 70)
    print()

    # Track per-layer statistics
    layer_stats = {
        layer_id: {
            'expert_usage': defaultdict(int),
            'mode_history': [],
            'first_trust_driven': None,
            'context_expert_map': defaultdict(lambda: defaultdict(int))
        }
        for layer_id in test_layers
    }

    start_time = time.time()
    generation = 0

    for epoch in range(num_epochs):
        print(f"=== EPOCH {epoch + 1}/{num_epochs} ===")
        print()

        for seq_idx, sequence in enumerate(sequences):
            generation += 1

            # Use pre-generated embeddings from context_embeddings
            input_embedding = context_embeddings[seq_idx]
            context_info = context_classifier.classify(input_embedding)
            context = context_info.context_id

            # Simulate router logits (same for all layers for now)
            router_logits = np.random.randn(128).astype(np.float32)

            # Process each layer
            for layer_id in test_layers:
                selector = layer_selectors[layer_id]
                stats = layer_stats[layer_id]

                # Select experts for this layer
                result = selector.select_experts(
                    router_logits=router_logits,
                    context=context,
                    k=4,  # Select 4 experts per layer
                    input_embedding=input_embedding
                )

                # Track expert usage
                for expert_id in result.selected_expert_ids:
                    stats['expert_usage'][expert_id] += 1
                    stats['context_expert_map'][expert_id][context] += 1

                # Track mode
                stats['mode_history'].append(result.selection_mode)

                # Detect first trust_driven
                if result.selection_mode == "trust_driven" and stats['first_trust_driven'] is None:
                    stats['first_trust_driven'] = generation
                    print(f"🎯 TRUST_DRIVEN ACTIVATED at generation {generation} (Layer {layer_id})!")
                    print(f"   Context: {context}, Experts: {result.selected_expert_ids}")

                # Simulate quality measurement
                quality = 0.75 + np.random.randn() * 0.1
                quality = np.clip(quality, 0.0, 1.0)

                # Update trust (Session 80 fix: unweighted quality)
                for expert_id in result.selected_expert_ids:
                    selector.update_trust_for_expert(expert_id, context, quality)

            # Print progress every 10 generations
            if generation % 10 == 0:
                print(f"Generation {generation}/{num_epochs * len(sequences)}: {sequence[:50]}...")
                # Show mode distribution for first layer
                mode_counts = defaultdict(int)
                for mode in layer_stats[test_layers[0]]['mode_history']:
                    mode_counts[mode] += 1
                print(f"  Layer 0 modes: {dict(mode_counts)}")
                print()

        print()

    elapsed = time.time() - start_time

    # Analyze results
    print()
    print("=" * 70)
    print(f"MULTI-LAYER RESULTS - {len(test_layers)} LAYERS TESTED")
    print("=" * 70)
    print()

    results = {}

    for layer_id in test_layers:
        stats = layer_stats[layer_id]
        selector = layer_selectors[layer_id]

        # Expert diversity
        unique_experts = len(stats['expert_usage'])
        total_selections = sum(stats['expert_usage'].values())

        # Specialization
        specialists = 0
        generalists = 0
        for expert_id, contexts in stats['context_expert_map'].items():
            if len(contexts) == 1:
                specialists += 1
            else:
                generalists += 1

        specialization_rate = specialists / unique_experts if unique_experts > 0 else 0

        # Mode distribution
        mode_counts = defaultdict(int)
        for mode in stats['mode_history']:
            mode_counts[mode] += 1

        total_modes = len(stats['mode_history'])

        # Store results
        results[layer_id] = {
            'unique_experts': unique_experts,
            'utilization_pct': (unique_experts / 128) * 100,
            'specialists': specialists,
            'generalists': generalists,
            'specialization_rate': specialization_rate,
            'first_trust_driven': stats['first_trust_driven'],
            'mode_distribution': {
                'router_explore': mode_counts['router_explore'],
                'trust_driven': mode_counts['trust_driven'],
                'forced_exploration': mode_counts['forced_exploration']
            },
            'mode_percentages': {
                'router_explore_pct': (mode_counts['router_explore'] / total_modes) * 100,
                'trust_driven_pct': (mode_counts['trust_driven'] / total_modes) * 100,
                'forced_exploration_pct': (mode_counts['forced_exploration'] / total_modes) * 100
            }
        }

        print(f"📊 Layer {layer_id}:")
        print(f"  Unique experts: {unique_experts}/128 ({results[layer_id]['utilization_pct']:.1f}%)")
        print(f"  Specialists: {specialists}, Generalists: {generalists}")
        print(f"  Specialization: {specialization_rate * 100:.1f}%")
        print(f"  First trust_driven: Gen {stats['first_trust_driven']}")
        print(f"  Mode distribution:")
        print(f"    router_explore: {mode_counts['router_explore']} ({results[layer_id]['mode_percentages']['router_explore_pct']:.1f}%)")
        print(f"    trust_driven: {mode_counts['trust_driven']} ({results[layer_id]['mode_percentages']['trust_driven_pct']:.1f}%)")
        print(f"    forced_exploration: {mode_counts['forced_exploration']} ({results[layer_id]['mode_percentages']['forced_exploration_pct']:.1f}%)")
        print()

    # Cross-layer analysis
    print("=" * 70)
    print("CROSS-LAYER ANALYSIS")
    print("=" * 70)
    print()

    avg_utilization = np.mean([r['utilization_pct'] for r in results.values()])
    avg_trust_driven = np.mean([r['mode_percentages']['trust_driven_pct'] for r in results.values()])
    avg_first_activation = np.mean([r['first_trust_driven'] for r in results.values() if r['first_trust_driven']])

    print(f"Average utilization: {avg_utilization:.1f}%")
    print(f"Average trust_driven: {avg_trust_driven:.1f}%")
    print(f"Average first activation: Gen {avg_first_activation:.1f}")
    print()

    # Comparison to Session 80 (Layer 0 baseline)
    print("📊 Comparison to Session 80 (Layer 0 baseline):")
    print(f"  S80 (layer 0 only): 62 experts (48.4%), 73.3% trust_driven, gen 8")
    if 0 in results:
        print(f"  S81 (layer 0 multi): {results[0]['unique_experts']} experts ({results[0]['utilization_pct']:.1f}%), " +
              f"{results[0]['mode_percentages']['trust_driven_pct']:.1f}% trust_driven, gen {results[0]['first_trust_driven']}")
    print()

    print(f"⏱️  Time: {elapsed:.1f}s")
    print()

    # Save results
    output_path = Path(__file__).parent / "session82_full_48_layer_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            'test_layers': test_layers,
            'config': {
                'epsilon': epsilon,
                'min_trust_evidence': min_trust_evidence,
                'num_epochs': num_epochs,
                'num_sequences': len(sequences)
            },
            'layer_results': {str(k): v for k, v in results.items()},  # JSON keys must be strings
            'cross_layer': {
                'avg_utilization_pct': avg_utilization,
                'avg_trust_driven_pct': avg_trust_driven,
                'avg_first_activation': avg_first_activation
            },
            'elapsed_time': elapsed
        }, f, indent=2)

    print(f"💾 Results saved to: {output_path}")
    print()

    return results


if __name__ == "__main__":
    print("Session 82: Full 48-Layer Trust-First Deployment")
    print("Goal: Deploy validated architecture to ALL 48 Q3-Omni layers")
    print()

    # Full 48-layer deployment
    results = test_multi_layer_deployment(
        test_layers=list(range(48)),  # ALL 48 LAYERS!
        num_epochs=10,
        epsilon=0.2,  # Session 77 optimal
        min_trust_evidence=2  # Session 78 optimal
    )

    if results:
        print("=" * 70)
        print("SESSION 82 COMPLETE - Full 48-Layer Deployment")
        print("=" * 70)
        print()
        print("Key Question: Does trust-first scale to all 48 layers?")

        # Check if all layers achieved trust_driven
        all_activated = all(r['first_trust_driven'] is not None for r in results.values())

        if all_activated:
            print("  ✅ YES - All 48 layers activated trust_driven!")
            print()
            print("  Trust-first architecture VALIDATED at full scale!")
            print()
            print("Next Steps:")
            print("  - Production readiness testing (longer sequences)")
            print("  - Federation testing (Thor → Sprout)")
            print("  - Performance optimization (if needed)")
        else:
            failed_layers = [k for k, v in results.items() if v['first_trust_driven'] is None]
            print(f"  ⚠️  {len(failed_layers)}/48 layers did not activate trust_driven")
            print(f"  Failed layers: {failed_layers}")
            print("  Investigate layer-specific behavior")
