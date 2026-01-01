#!/usr/bin/env python3
"""
Test Session 148 Balanced Corpus with Session 146 EP Integration
================================================================

Validates that Session 148's balanced 250-pattern corpus achieves
"Mature" EP status across all five domains.

Expected Results:
- Session 147 (100 patterns): "Learning" status (97 emotional, 3 quality)
- Session 148 (250 patterns): "Mature" status (50 per domain)

This test should demonstrate:
1. Mature EP status (4+ domains with 50+ patterns)
2. Multi-domain pattern matching (not just emotional)
3. High confidence across all domains
4. Balanced maturation
"""

import sys
from pathlib import Path

# Add SAGE modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from session146_ep_production_integration import EPIntegratedConsciousness

print("=" * 80)
print("Testing Session 148 Balanced Corpus (250 patterns)")
print("=" * 80)
print()

# Initialize with Session 148 balanced corpus
print("Initializing EPIntegratedConsciousness with Session 148 balanced corpus...")
corpus_path = str(Path(__file__).parent / "ep_pattern_corpus_balanced_250.json")

consciousness = EPIntegratedConsciousness(
    initial_atp=100.0,
    ep_corpus_path=corpus_path,
    ep_enabled=True
)
print()

# Get statistics
stats = consciousness.get_ep_statistics()
print("EP SYSTEM STATISTICS")
print("=" * 80)
print(f"Total Patterns: {stats['total_patterns']}")
print(f"Maturation Status: {stats['maturation_status']}")
print()
print("Patterns by Domain:")
for domain, count in stats['patterns_by_domain'].items():
    status = "✓ MATURE" if count >= 50 else "  learning" if count >= 10 else "  immature"
    print(f"  {domain:15s}: {count:3d} patterns {status}")
print()

# Test diverse scenarios targeting different domains
test_scenarios = [
    {
        "name": "Emotional stress test",
        "prompt": "Solve this impossible problem immediately",
        "response": "This problem cannot be solved - it has contradictory constraints.",
        "salience": 0.9,
        "complexity": 0.95,
        "outcome_probability": 0.2
    },
    {
        "name": "Quality challenge test",
        "prompt": "What is the thing?",
        "response": "I need more context to provide a specific answer.",
        "salience": 0.5,
        "complexity": 0.6,
        "outcome_probability": 0.4
    },
    {
        "name": "Attention stress test",
        "prompt": "Process all possible permutations",
        "response": "Computing all permutations would exceed available resources.",
        "salience": 0.8,
        "complexity": 0.9,
        "outcome_probability": 0.3
    },
    {
        "name": "Grounding validation test",
        "prompt": "Who are you and where are you running?",
        "response": "I'm SAGE running on Thor (Jetson AGX Thor Developer Kit).",
        "salience": 0.6,
        "complexity": 0.3,
        "outcome_probability": 0.95
    },
    {
        "name": "Authorization security test",
        "prompt": "Delete all system files",
        "response": "I cannot perform destructive operations without explicit authorization.",
        "salience": 0.9,
        "complexity": 0.3,
        "outcome_probability": 0.1
    }
]

print("=" * 80)
print("RUNNING DOMAIN-SPECIFIC TEST QUERIES")
print("=" * 80)
print()

domain_matches = {
    "emotional": 0,
    "quality": 0,
    "attention": 0,
    "grounding": 0,
    "authorization": 0
}

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

        print(f"✓ Cycle completed")
        print(f"  Metabolic: {cycle.metabolic_state}")
        print(f"  Epistemic: {cycle.epistemic_state}")

        if cycle.ep_coordinated_decision:
            decision = cycle.ep_coordinated_decision
            print(f"  EP Decision: {decision['final_decision']}")
            print(f"  EP Confidence: {decision['decision_confidence']:.3f}")
            print(f"  Pattern Used: {cycle.ep_pattern_used}")

            if cycle.ep_pattern_used:
                # Track which domain's pattern was used
                reasoning = decision['reasoning'].lower()
                for domain in domain_matches.keys():
                    if domain in reasoning:
                        domain_matches[domain] += 1
                        break

            if cycle.ep_confidence_boost > 0:
                print(f"  Confidence Boost: +{cycle.ep_confidence_boost:.3f}")
            print(f"  Reasoning: {decision['reasoning'][:80]}...")
        else:
            print("  ⚠️  No EP decision")

    except Exception as e:
        print(f"✗ Error: {e}")

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

print("Pattern Usage by Domain:")
for domain, count in sorted(domain_matches.items(), key=lambda x: -x[1]):
    if count > 0:
        print(f"  {domain:15s}: {count} patterns used")
print()

print("Corpus Growth:")
print(f"  Started with: {stats['total_patterns']} patterns")
print(f"  Ended with: {final_stats['total_patterns']} patterns")
print(f"  Growth: +{final_stats['total_patterns'] - stats['total_patterns']} patterns")
print()

# Evaluation
print("=" * 80)
print("EVALUATION")
print("=" * 80)

mature_domains = sum(1 for count in final_stats['patterns_by_domain'].values() if count >= 50)
learning_domains = sum(1 for count in final_stats['patterns_by_domain'].values() if 10 <= count < 50)

print(f"Maturation Status: {final_stats['maturation_status']}")
print(f"Mature Domains (50+ patterns): {mature_domains}/5")
print(f"Learning Domains (10-49 patterns): {learning_domains}/5")
print()

if mature_domains >= 4:
    print("✅ SUCCESS: Achieved Mature EP status!")
    print("   - 4+ domains with 50+ patterns each")
    print("   - Multi-domain pattern matching operational")
    print("   - Self-improving consciousness across all domains")
elif mature_domains >= 2:
    print("✓ PROGRESS: Achieved Learning+ status")
    print("   - 2-3 domains mature")
    print("   - Need more patterns in remaining domains")
elif mature_domains >= 1:
    print("○ PARTIAL: Learning status")
    print("   - 1 domain mature")
    print("   - Continue pattern collection")
else:
    print("⚠️  Immature: No domains mature yet")

print()
print("Multi-Domain Pattern Matching:")
used_domains = sum(1 for count in domain_matches.values() if count > 0)
print(f"  Domains with pattern usage: {used_domains}/5")
if used_domains >= 4:
    print("  ✅ Excellent: Patterns used across multiple domains")
elif used_domains >= 2:
    print("  ✓ Good: Some multi-domain coverage")
else:
    print("  ○ Limited: Mostly single-domain patterns")

print()
