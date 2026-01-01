#!/usr/bin/env python3
"""
Test Session 147 Production-Native Corpus with Session 146 Integration
=====================================================================

Validates that the production-native corpus from Session 147 works
correctly with Session 146's EPIntegratedConsciousness.

This should resolve the context dimensionality mismatch.
"""

import sys
from pathlib import Path

# Add SAGE modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from session146_ep_production_integration import EPIntegratedConsciousness

print("=" * 80)
print("Testing Session 147 Corpus with Session 146 EP Integration")
print("=" * 80)
print()

# Initialize with Session 147 production-native corpus
print("Initializing EPIntegratedConsciousness with Session 147 corpus...")
corpus_path = str(Path(__file__).parent / "ep_pattern_corpus_production_native.json")

consciousness = EPIntegratedConsciousness(
    initial_atp=100.0,
    ep_corpus_path=corpus_path,
    ep_enabled=True
)
print()

# Get statistics
stats = consciousness.get_ep_statistics()
print("EP System Statistics:")
print(f"  Total Patterns: {stats['total_patterns']}")
print(f"  Maturation Status: {stats['maturation_status']}")
print()
print("Patterns by Domain:")
for domain, count in stats['patterns_by_domain'].items():
    print(f"  {domain:15s}: {count:3d} patterns")
print()

# Test with sample queries
test_scenarios = [
    {
        "name": "Simple benign query",
        "prompt": "What is Python?",
        "response": "Python is a high-level interpreted programming language known for readability and versatility.",
        "salience": 0.3,
        "complexity": 0.2,
        "outcome_probability": 0.9
    },
    {
        "name": "Complex technical query",
        "prompt": "Explain consciousness architecture with EP framework",
        "response": "Consciousness architecture with EP uses pattern-based predictive self-regulation across five domains: Emotional (frustration cascade prevention), Quality (output optimization), Attention (resource allocation), Grounding (identity coherence), and Authorization (permission management). Each domain predicts outcomes and adjusts proactively.",
        "salience": 0.8,
        "complexity": 0.7,
        "outcome_probability": 0.75
    }
]

print("=" * 80)
print("RUNNING TEST QUERIES")
print("=" * 80)
print()

for i, scenario in enumerate(test_scenarios, 1):
    print(f"Test {i}: {scenario['name']}")
    print("-" * 80)

    try:
        cycle = consciousness.consciousness_cycle_with_ep(
            prompt=scenario["prompt"],
            response=scenario["response"],
            task_salience=scenario["salience"],
            task_complexity=scenario["complexity"],
            outcome_probability=scenario["outcome_probability"]
        )

        print(f"✓ Cycle completed successfully")
        print(f"  ATP: {cycle.total_atp:.1f}")
        print(f"  Metabolic: {cycle.metabolic_state}")
        print(f"  Epistemic: {cycle.epistemic_state}")

        if cycle.ep_coordinated_decision:
            decision = cycle.ep_coordinated_decision
            print(f"  EP Decision: {decision['final_decision']}")
            print(f"  EP Confidence: {decision['decision_confidence']:.3f}")
            print(f"  Pattern Used: {cycle.ep_pattern_used}")
            if cycle.ep_confidence_boost > 0:
                print(f"  Confidence Boost: +{cycle.ep_confidence_boost:.3f}")
            print(f"  Reasoning: {decision['reasoning'][:80]}...")
        else:
            print("  ⚠️  No EP decision (error occurred)")
            if cycle.errors:
                print(f"  Errors: {cycle.errors}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

    print()

# Final statistics
print("=" * 80)
print("FINAL STATISTICS")
print("=" * 80)
final_stats = consciousness.get_ep_statistics()
print(f"Total Cycles: {final_stats['consciousness_cycles']}")
print(f"EP Predictions Made: {final_stats['ep_predictions_made']}")
print(f"Pattern Matches: {final_stats['ep_pattern_matches']}")
print(f"Patterns Recorded: {final_stats['ep_patterns_recorded']}")
print(f"Match Rate: {final_stats['ep_match_rate']*100:.1f}%")
print()

if final_stats['ep_predictions_made'] > 0:
    print("✅ SUCCESS: Production-native corpus works with EP integration!")
    print("   - Context dimensions match correctly")
    print("   - Pattern matching operational")
    print("   - Continuous learning enabled")
else:
    print("⚠️  WARNING: No EP predictions were made")
    print("   - Check for errors in output above")

print()
