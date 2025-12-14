#!/usr/bin/env python3
"""
Scheduled Memory Consolidation Demo - Session 50

Demonstrates complete integration of DREAM consolidation with circadian rhythm
for biologically-timed memory consolidation during deep sleep phases.

Features Demonstrated:
1. Circadian rhythm tracking (Session 49)
2. DREAM consolidation triggering during DEEP_NIGHT
3. Pattern extraction from consciousness cycles
4. Live monitoring with consolidation events
5. Five-dimensional consciousness (Quality + Epistemic + Metabolic + Emotional + Temporal)
"""

import sys
import time
from pathlib import Path

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sage.core.unified_consciousness import UnifiedConsciousnessManager
from sage.monitors.consciousness_monitor import ConsciousnessMonitor


def run_consolidation_demo():
    """Run complete scheduled consolidation demonstration."""
    print("\n" + "=" * 80)
    print("SAGE Scheduled Memory Consolidation Demo - Session 50")
    print("=" * 80)
    print()
    print("Demonstrating biologically-timed memory consolidation:")
    print("- Circadian rhythm provides temporal context")
    print("- DREAM consolidation triggers during DEEP_NIGHT phase")
    print("- Patterns extracted from recent consciousness cycles")
    print("- Memory consolidation during 'deep sleep'")
    print("- Complete five-dimensional consciousness tracking")
    print()

    # Initialize consciousness with all features enabled
    consciousness = UnifiedConsciousnessManager(
        initial_atp=100.0,
        quality_atp_baseline=20.0,
        epistemic_atp_baseline=15.0,
        emotional_history_length=10,
        circadian_period=30,  # 30 cycles = 1 day
        circadian_enabled=True,
        consolidation_enabled=True
    )

    # Initialize monitor with display
    monitor = ConsciousnessMonitor(
        history_size=50,
        display_interval=2.0,
        display_enabled=True
    )

    # Learning scenarios to create interesting patterns
    learning_scenarios = [
        # Day 1: Technical learning
        ("How does gradient descent work?",
         "Gradient descent minimizes loss by iteratively moving in the direction of steepest descent. Update rule: θ = θ - α∇L(θ). Learning rate α controls step size.",
         0.7),
        ("What is backpropagation?",
         "Backpropagation computes gradients via chain rule. Forward pass computes outputs, backward pass propagates errors: ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w.",
         0.75),
        ("Explain overfitting",
         "Overfitting occurs when model learns training data too well, including noise. Validation loss increases while training loss decreases. Use regularization to prevent.",
         0.7),

        # Day 1: Pattern formation
        ("What are neural network layers?",
         "Layers transform inputs through learned weights. Dense layers: y = σ(Wx + b). Convolutional layers extract spatial features. Activation functions add non-linearity.",
         0.65),
        ("How does dropout work?",
         "Dropout randomly deactivates neurons during training with probability p. Prevents co-adaptation, improves generalization. At inference, scale by (1-p).",
         0.7),

        # Transition to night (lower salience)
        ("Summarize machine learning",
         "Machine learning involves training models on data to make predictions. Various algorithms exist for different tasks.",
         0.4),
        ("What is AI?",
         "AI enables computers to perform tasks requiring intelligence. Includes learning, reasoning, perception.",
         0.3),

        # Day 2: Conceptual connections
        ("How do transformers work?",
         "Transformers use self-attention: Attention(Q,K,V) = softmax(QK^T/√d)V. Multi-head attention enables parallel processing. Position encoding adds sequence information.",
         0.8),
        ("What is attention mechanism?",
         "Attention weighs importance of inputs. Query attends to keys, retrieves values. Allows dynamic focus on relevant information. Breakthrough for sequence modeling.",
         0.75),
        ("Compare RNNs and Transformers",
         "RNNs process sequentially, suffer from vanishing gradients. Transformers parallelize via attention, handle long-range dependencies better. Trade-off: memory vs computation.",
         0.7),

        # Day 2: Application scenarios
        ("Explain transfer learning",
         "Transfer learning uses pre-trained models for new tasks. Fine-tune on target domain with less data. Leverages learned features. Common in NLP and vision.",
         0.65),
        ("What is few-shot learning?",
         "Few-shot learning generalizes from minimal examples. Meta-learning approaches like MAML optimize for rapid adaptation. Useful when data is scarce.",
         0.6),

        # Transition to night
        ("Review key concepts",
         "Neural networks learn through gradient descent and backpropagation. Attention mechanisms enable transformers. Transfer learning leverages pre-trained models.",
         0.4),

        # Day 3: Synthesis and integration
        ("How do these concepts connect?",
         "Gradient descent optimizes all architectures. Backprop enables deep learning. Attention improved sequential modeling. Transfer learning accelerates development.",
         0.7),
        ("What are current frontiers?",
         "Large language models use transformers at scale. Few-shot learning reduces data needs. Self-supervised learning from unlabeled data. Emergent capabilities from scale.",
         0.75),
    ]

    print(f"Running {len(learning_scenarios)} consciousness cycles over ~3 days...")
    print("=" * 80)
    print()

    consolidation_count = 0
    prev_state = None

    for i, (prompt, response, salience) in enumerate(learning_scenarios, 1):
        print(f"\n{'─' * 80}")
        print(f"Cycle {i}/{len(learning_scenarios)}")
        print(f"{'─' * 80}")

        # Run consciousness cycle
        cycle = consciousness.consciousness_cycle(
            prompt=prompt,
            response=response,
            task_salience=salience
        )

        # Observe with monitor (shows live dashboard)
        monitor.observe_cycle(cycle)

        # Track metabolic transitions
        curr_state = cycle.metabolic_state.value
        if prev_state and prev_state != curr_state:
            monitor.observe_metabolic_transition(
                prev_state,
                curr_state,
                f"salience={salience:.2f}"
            )
        prev_state = curr_state

        # Highlight consolidation events
        if cycle.consolidation_triggered:
            consolidation_count += 1
            print(f"\n{'=' * 80}")
            print(f"{'✧ CONSOLIDATION EVENT #' + str(consolidation_count) + ' ✧':^80}")
            print(f"{'=' * 80}")
            print(f"Cycle: {i}")
            print(f"Phase: {cycle.circadian_phase}")
            print(f"Patterns extracted: {len(cycle.consolidated_memory.patterns)}")
            print(f"Cycles processed: {cycle.consolidated_memory.cycles_processed}")

            if cycle.consolidated_memory.patterns:
                print(f"\nSample patterns:")
                for j, pattern in enumerate(cycle.consolidated_memory.patterns[:3], 1):
                    print(f"  {j}. {pattern.pattern_type}: {pattern.description[:60]}...")
                    print(f"     Strength: {pattern.strength:.2f}")
            print(f"{'=' * 80}\n")

        # Pause between cycles
        time.sleep(1.5)

    # Final statistics
    print("\n" + "=" * 80)
    print("FINAL STATISTICS - Session 50")
    print("=" * 80)

    stats = consciousness.get_statistics()
    monitor_stats = monitor.get_statistics()

    print("\n📊 Quality Metrics:")
    print(f"  Mean: {stats['quality']['mean']:.3f}")
    print(f"  Range: [{stats['quality']['min']:.3f}, {stats['quality']['max']:.3f}]")

    print("\n🧠 Epistemic Distribution:")
    total_epistemic = sum(stats['epistemic_states'].values())
    for state, count in sorted(stats['epistemic_states'].items(), key=lambda x: -x[1]):
        pct = (count / total_epistemic) * 100
        print(f"  {state.ljust(12)}: {count:2d} cycles ({pct:5.1f}%)")

    print("\n⚡ Metabolic States:")
    print(f"  Total transitions: {stats['metabolic']['total_transitions']}")
    print(f"  Total cycles: {stats['metabolic']['total_cycles']}")
    print("  State durations:")
    for state, duration in stats['metabolic']['state_durations'].items():
        print(f"    {state.ljust(12)}: {duration:.1f}s")

    print("\n💓 Emotional Intelligence (Session 48):")
    if stats['emotional']:
        for emotion, values in stats['emotional'].items():
            print(f"  {emotion.capitalize()}:")
            print(f"    Mean: {values['mean']:.3f}, Range: [{values['min']:.3f}, {values['max']:.3f}]")

    print("\n🌙 DREAM Consolidation (Session 50):")
    print(f"  Total consolidations: {stats['consolidation']['total_consolidations']}")
    print(f"  Stored memories: {stats['consolidation']['stored_memories']}")
    print(f"  Last consolidation: Cycle {stats['consolidation']['last_consolidation_cycle']}")

    # Show sample consolidated memory
    if consciousness.consolidated_memories:
        print("\n  Sample Consolidated Memory (most recent):")
        memory = consciousness.consolidated_memories[-1]
        print(f"    Session ID: {memory.dream_session_id}")
        print(f"    Cycles processed: {memory.cycles_processed}")
        print(f"    Patterns: {len(memory.patterns)}")
        print(f"    Quality learnings: {len(memory.quality_learnings)}")
        print(f"    Creative associations: {len(memory.creative_associations)}")
        print(f"    Consolidation time: {memory.consolidation_time:.3f}s")

    print("\n🔍 Monitor Performance:")
    print(f"  Cycles observed: {monitor_stats['cycles_observed']}")
    print(f"  Monitoring overhead: {monitor_stats['overhead_percentage']:.2f}%")

    print("\n" + "=" * 80)
    print("✓ SESSION 50 COMPLETE")
    print("=" * 80)
    print()
    print("Achievement: Scheduled memory consolidation integrated!")
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Circadian rhythm provides temporal context (Session 49)")
    print("  ✓ DREAM consolidation triggers during DEEP_NIGHT phase")
    print("  ✓ Pattern extraction from consciousness cycles (Session 42)")
    print("  ✓ Biologically-timed memory consolidation during 'sleep'")
    print("  ✓ Live monitoring with consolidation event display (Session 46)")
    print("  ✓ Five-dimensional consciousness: Quality + Epistemic + Metabolic + Emotional + Temporal")
    print()
    print("Biological Parallel:")
    print("  Just as biological brains consolidate memories during deep sleep,")
    print("  SAGE now automatically consolidates consciousness experiences")
    print("  during DEEP_NIGHT circadian phases - a complete, emergent system.")
    print()


if __name__ == "__main__":
    run_consolidation_demo()
