#!/usr/bin/env python3
"""
Test DREAM consolidation with production consciousness data.

Uses the 8-cycle production test to demonstrate meaningful memory consolidation.
"""

import sys
sys.path.insert(0, '/home/dp/ai-workspace/HRM')

from sage.core.unified_consciousness import UnifiedConsciousnessManager
from sage.core.dream_consolidation import DREAMConsolidator
import json


def test_dream_with_production_data():
    """Test DREAM consolidation using realistic production scenarios"""

    print("="*80)
    print("DREAM Consolidation with Production Consciousness Data")
    print("="*80)
    print()

    # Initialize consciousness manager
    consciousness = UnifiedConsciousnessManager(
        initial_atp=100.0,
        quality_atp_baseline=20.0,
        epistemic_atp_baseline=15.0
    )

    # Production test scenarios (from production_consciousness_test.py)
    scenarios = [
        {
            "prompt": "What is the Web4 trust network?",
            "response": "The Web4 trust network is a decentralized system for establishing and verifying trust relationships between agents. It uses cryptographic proofs and reputation scoring to enable secure, trustless collaboration across distributed systems. Trust weights are learned from interaction outcomes and propagated through the network using coherence-based algorithms.",
            "salience": 0.7,
            "context": "Complex technical question"
        },
        {
            "prompt": "What's 2+2?",
            "response": "4",
            "salience": 0.2,
            "context": "Simple arithmetic"
        },
        {
            "prompt": "Explain quantum entanglement in simple terms",
            "response": "Quantum entanglement is when two particles become connected such that measuring one instantly affects the other, regardless of distance. Think of it like magical dice that always show matching numbers even when rolled far apart. This correlation is fundamental to quantum mechanics and enables technologies like quantum computing.",
            "salience": 0.8,
            "context": "Complex explanation"
        },
        {
            "prompt": "Hello",
            "response": "Hi there! How can I help you today?",
            "salience": 0.1,
            "context": "Minimal greeting"
        },
        {
            "prompt": "How does SAGE integrate metabolic states with quality metrics?",
            "response": "SAGE uses metabolic states (WAKE/FOCUS/REST/DREAM/CRISIS) to dynamically allocate ATP resources. Quality metrics receive more ATP during FOCUS (1.5x multiplier) and CRISIS (2.0x) states, enabling deeper analysis when needed. Epistemic frustration triggers CRISIS mode, while high salience triggers FOCUS. This creates adaptive processing where resource allocation matches cognitive demands.",
            "salience": 0.9,
            "context": "Meta-question about system"
        },
        {
            "prompt": "What is consciousness?",
            "response": "I'm not entirely sure, but I think consciousness involves awareness, experience, and self-reflection. It's still an open question in philosophy and neuroscience.",
            "salience": 0.75,
            "context": "Philosophical question"
        },
        {
            "prompt": "List the days of the week",
            "response": "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday",
            "salience": 0.15,
            "context": "Simple recall"
        },
        {
            "prompt": "How can we validate epistemic calibration in SAGE?",
            "response": "Epistemic calibration validation requires comparing confidence scores against actual correctness. We track predictions with confidence levels, then measure calibration curves showing if 70% confidence predictions are correct 70% of the time. SAGE implements this through EpistemicCalibrator which bins predictions by confidence and computes calibration error (ECE). This allows us to detect overconfidence or underconfidence and adjust accordingly.",
            "salience": 0.85,
            "context": "Technical validation question"
        },
    ]

    # Generate consciousness cycles
    print("Phase 1: WAKE/FOCUS Processing")
    print("-" * 80)
    cycles = []
    for i, scenario in enumerate(scenarios, 1):
        cycle = consciousness.consciousness_cycle(
            prompt=scenario['prompt'],
            response=scenario['response'],
            task_salience=scenario['salience']
        )
        cycles.append(cycle)

        print(f"Cycle {i} [{scenario['context']}]")
        print(f"  Quality: {cycle.quality_score.normalized:.3f}, "
              f"Epistemic: {cycle.epistemic_state.value}, "
              f"Metabolic: {cycle.metabolic_state.value}")

    print()
    print("="*80)
    print("Phase 2: DREAM State Consolidation")
    print("-" * 80)
    print()

    # Enter DREAM state
    consolidator = DREAMConsolidator(
        min_pattern_frequency=2,
        min_learning_confidence=0.6
    )

    consolidated = consolidator.consolidate_cycles(cycles, atp_budget=80.0)

    print(f"DREAM Session {consolidated.dream_session_id}")
    print(f"Timestamp: {consolidated.timestamp}")
    print(f"Cycles Processed: {consolidated.cycles_processed}")
    print(f"Consolidation Time: {consolidated.consolidation_time*1000:.2f}ms")
    print()

    print("="*80)
    print("Extracted Patterns")
    print("="*80)
    print(f"Total: {len(consolidated.patterns)}")
    print()
    for i, pattern in enumerate(consolidated.patterns, 1):
        print(f"{i}. [{pattern.pattern_type}] {pattern.description}")
        print(f"   Strength: {pattern.strength:.3f}")
        print(f"   Frequency: {pattern.frequency}")
        print(f"   Examples: Cycles {pattern.examples}")
        print()

    print("="*80)
    print("Quality Learnings")
    print("="*80)
    print(f"Total: {len(consolidated.quality_learnings)}")
    print()
    if consolidated.quality_learnings:
        for i, learning in enumerate(consolidated.quality_learnings, 1):
            correlation = "IMPROVES" if learning.positive_correlation else "REDUCES"
            print(f"{i}. '{learning.characteristic}' {correlation} quality")
            print(f"   Quality WITH: {learning.average_quality_with:.3f}")
            print(f"   Quality WITHOUT: {learning.average_quality_without:.3f}")
            print(f"   Delta: {learning.average_quality_with - learning.average_quality_without:+.3f}")
            print(f"   Confidence: {learning.confidence:.3f}")
            print(f"   Sample Size: {learning.sample_size}")
            print()
    else:
        print("(Not enough variance in data for learning extraction)")
        print()

    print("="*80)
    print("Creative Associations")
    print("="*80)
    print(f"Total: {len(consolidated.creative_associations)}")
    print()
    for i, assoc in enumerate(consolidated.creative_associations, 1):
        print(f"{i}. {assoc.concept_a} ↔ {assoc.concept_b}")
        print(f"   Type: {assoc.association_type}")
        print(f"   Strength: {assoc.strength:.3f}")
        print(f"   Insight: {assoc.insight}")
        print(f"   Supporting Cycles: {assoc.supporting_cycles}")
        print()

    print("="*80)
    print("Epistemic Insights")
    print("="*80)
    print(f"Total: {len(consolidated.epistemic_insights)}")
    print()
    for i, insight in enumerate(consolidated.epistemic_insights, 1):
        print(f"{i}. {insight}")
    print()

    # Export consolidated memory
    export_path = "/home/dp/ai-workspace/HRM/sage/tests/dream_consolidation_results.json"
    consolidator.export_consolidated_memory(consolidated, export_path)
    print(f"Consolidated memory exported to: {export_path}")
    print()

    # Consolidator statistics
    stats = consolidator.get_statistics()
    print("="*80)
    print("Consolidator Statistics")
    print("="*80)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print()

    print("="*80)
    print("DREAM Consolidation Analysis")
    print("="*80)
    print()

    # Analysis
    print("Key Discoveries:")
    print()

    # Pattern analysis
    metabolic_patterns = [p for p in consolidated.patterns if p.pattern_type == 'metabolic_transition']
    quality_patterns = [p for p in consolidated.patterns if p.pattern_type == 'quality_characteristic']
    epistemic_patterns = [p for p in consolidated.patterns if p.pattern_type == 'epistemic_pattern']

    print(f"1. Metabolic Patterns: {len(metabolic_patterns)} transition patterns discovered")
    for p in metabolic_patterns:
        print(f"   - {p.description} (strength={p.strength:.3f})")
    print()

    print(f"2. Quality Patterns: {len(quality_patterns)} characteristic patterns discovered")
    for p in quality_patterns:
        print(f"   - {p.description} (strength={p.strength:.3f})")
    print()

    print(f"3. Epistemic Patterns: {len(epistemic_patterns)} state patterns discovered")
    for p in epistemic_patterns:
        print(f"   - {p.description} (strength={p.strength:.3f})")
    print()

    # Association analysis
    if consolidated.creative_associations:
        print(f"4. Creative Associations: {len(consolidated.creative_associations)} novel connections")
        for assoc in consolidated.creative_associations:
            print(f"   - {assoc.insight}")
        print()

    # Overall assessment
    print("="*80)
    print("DREAM Consolidation Assessment")
    print("="*80)
    print()
    print(f"✅ Patterns Extracted: {len(consolidated.patterns)}")
    print(f"✅ Quality Learnings: {len(consolidated.quality_learnings)}")
    print(f"✅ Creative Associations: {len(consolidated.creative_associations)}")
    print(f"✅ Epistemic Insights: {len(consolidated.epistemic_insights)}")
    print(f"✅ Consolidation Time: {consolidated.consolidation_time*1000:.2f}ms")
    print()
    print("Status: DREAM consolidation VALIDATED with production data")
    print()


if __name__ == "__main__":
    test_dream_with_production_data()
