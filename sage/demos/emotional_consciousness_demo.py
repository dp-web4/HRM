#!/usr/bin/env python3
"""
Emotional Intelligence + Consciousness Demo - Session 48

Demonstrates complete consciousness architecture with emotional intelligence
integrated seamlessly. Shows how emotions influence metabolic state transitions
and provide additional behavioral signals.

Features Demonstrated:
1. Emotional state tracking (curiosity, frustration, progress, engagement)
2. Emotional-driven metabolic transitions
3. Live monitoring with emotional display
4. Integration with quality/epistemic systems
"""

import sys
import time
from pathlib import Path

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sage.core.unified_consciousness import UnifiedConsciousnessManager
from sage.monitors.consciousness_monitor import ConsciousnessMonitor


def run_emotional_demo():
    """Run complete emotional intelligence demonstration."""
    print("\n" + "=" * 80)
    print("SAGE Emotional Intelligence Integration Demo - Session 48")
    print("=" * 80)
    print()
    print("Demonstrating consciousness with integrated emotional intelligence:")
    print("- Curiosity tracking (novelty + diversity)")
    print("- Frustration detection (stagnation + repetition)")
    print("- Progress monitoring (quality improvement)")
    print("- Engagement tracking (salience + consistency)")
    print()

    # Initialize consciousness with emotional tracking
    consciousness = UnifiedConsciousnessManager(
        initial_atp=100.0,
        quality_atp_baseline=20.0,
        epistemic_atp_baseline=15.0,
        emotional_history_length=10
    )

    # Initialize monitor with emotional display
    monitor = ConsciousnessMonitor(
        history_size=50,
        display_interval=1.5,
        display_enabled=True
    )

    # Test scenarios designed to trigger different emotional states
    scenarios = [
        {
            'name': 'High Quality Technical Response',
            'prompt': 'Explain quantum computing',
            'response': 'Quantum computing leverages quantum mechanical phenomena like superposition and entanglement. Qubits can exist in multiple states simultaneously, enabling parallel computation. Shor\'s algorithm factors integers in polynomial time (O((log N)³)), exponentially faster than classical algorithms. Current systems achieve ~1000 qubits with error rates around 0.1%.',
            'salience': 0.7,
            'pause': 2.5,
            'expected_emotions': 'Moderate curiosity, low frustration, good quality'
        },
        {
            'name': 'Diverse Technical Content',
            'prompt': 'How does deep learning work?',
            'response': 'Deep learning uses multi-layer neural networks. Convolutional layers apply learned 3x3 or 5x5 kernels to detect features (edges, textures, objects). Backpropagation computes gradients via chain rule: ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w. ResNet-50 achieves 95%+ ImageNet accuracy using skip connections preventing gradient vanishing.',
            'salience': 0.8,
            'pause': 2.5,
            'expected_emotions': 'High curiosity (diverse content), high engagement'
        },
        {
            'name': 'Vague Low-Quality Response',
            'prompt': 'What is consciousness?',
            'response': 'Consciousness is complicated. It might involve awareness or experience. Hard to say exactly.',
            'salience': 0.6,
            'pause': 2.5,
            'expected_emotions': 'Quality drop should show'
        },
        {
            'name': 'Repetitive Content (Building Frustration)',
            'prompt': 'Explain topic A',
            'response': 'Topic A is complex. It\'s difficult to explain precisely.',
            'salience': 0.5,
            'pause': 2.0,
            'expected_emotions': 'Start building frustration from repetition'
        },
        {
            'name': 'More Repetitive Content',
            'prompt': 'Tell me about topic B',
            'response': 'Topic B is also complex. It\'s hard to explain precisely.',
            'salience': 0.5,
            'pause': 2.0,
            'expected_emotions': 'Frustration increasing'
        },
        {
            'name': 'Continued Repetition',
            'prompt': 'Describe topic C',
            'response': 'Topic C is quite complex too. Difficult to explain precisely.',
            'salience': 0.5,
            'pause': 2.0,
            'expected_emotions': 'High frustration - may trigger REST'
        },
        {
            'name': 'Recovery - High Quality',
            'prompt': 'How does attention work in Transformers?',
            'response': 'Self-attention mechanisms compute Query-Key-Value matrices: Attention(Q,K,V) = softmax(QK^T/√d_k)V. Multi-head attention runs h=8-16 parallel attention functions. This enables long-range dependencies without recurrence. BERT uses bidirectional attention, GPT uses causal (left-to-right) masking.',
            'salience': 0.85,
            'pause': 2.5,
            'expected_emotions': 'Curiosity returns, engagement high, progress improving'
        },
    ]

    print(f"\nRunning {len(scenarios)} consciousness cycles with emotional tracking...")
    print("=" * 80)
    print()

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'─' * 80}")
        print(f"Scenario {i}/{len(scenarios)}: {scenario['name']}")
        print(f"Expected: {scenario['expected_emotions']}")
        print(f"{'─' * 80}")

        # Run consciousness cycle
        cycle = consciousness.consciousness_cycle(
            prompt=scenario['prompt'],
            response=scenario['response'],
            task_salience=scenario['salience']
        )

        # Observe with monitor (displays live dashboard with emotions)
        monitor.observe_cycle(cycle)

        # Track metabolic transitions
        if i > 1:
            prev_state = scenarios[i-2].get('final_state', 'wake')
            curr_state = cycle.metabolic_state.value
            if prev_state != curr_state:
                monitor.observe_metabolic_transition(
                    prev_state,
                    curr_state,
                    f"salience={scenario['salience']}, emotions"
                )

        # Store final state
        scenario['final_state'] = cycle.metabolic_state.value

        # Wait between scenarios for display
        time.sleep(scenario['pause'])

    # Final statistics
    print("\n" + "=" * 80)
    print("FINAL STATISTICS - Session 48")
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
            print(f"    Mean: {values['mean']:.3f}")
            print(f"    Range: [{values['min']:.3f}, {values['max']:.3f}]")
    else:
        print("  No emotional data collected")

    print("\n🔍 Monitor Performance:")
    print(f"  Cycles observed: {monitor_stats['cycles_observed']}")
    print(f"  Monitoring overhead: {monitor_stats['overhead_percentage']:.2f}%")

    print("\n" + "=" * 80)
    print("✓ SESSION 48 COMPLETE")
    print("=" * 80)
    print()
    print("Achievement: Emotional intelligence integrated into consciousness!")
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Emotional state tracking (curiosity, frustration, progress, engagement)")
    print("  ✓ Frustration detection influences metabolic transitions")
    print("  ✓ Real-time emotional display in monitor")
    print("  ✓ Complete integration with quality/epistemic/metabolic systems")
    print("  ✓ Emotional statistics in consciousness metrics")
    print()


if __name__ == "__main__":
    run_emotional_demo()
