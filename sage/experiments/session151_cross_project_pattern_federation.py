#!/usr/bin/env python3
"""
Session 151: Cross-Project Pattern Federation

Research Question: Can patterns learned in different contexts (SAGE consciousness
vs Web4 game environment) transfer and improve performance across projects?

Hypothesis: Pattern matching is context-agnostic enough that patterns from
different domains can enrich each other, enabling collective learning.

Experiments:
1. Load Thor's SAGE patterns (250, balanced across 5 EP domains)
2. Load Web4's game patterns (100, ATP-focused game scenarios)
3. Test pattern matching with federated corpus (350 total)
4. Compare performance: SAGE-only vs Federated
5. Analyze cross-domain pattern utility

Expected Outcomes:
- If patterns transfer well: Higher match rate, more diverse matching
- If patterns don't transfer: No improvement or degradation
- Either way: Learn about pattern context-dependence
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.experiments.session146_ep_production_integration import EPIntegratedConsciousness
from sage.experiments.session150_production_ep_deployment import (
    ProductionScenario,
    ProductionEPDeployment
)


class PatternFederationExperiment:
    """
    Test cross-project pattern federation.

    Research questions:
    1. Do Web4 game patterns improve SAGE consciousness predictions?
    2. Do SAGE patterns improve game AI predictions?
    3. What is optimal corpus composition (SAGE vs game ratio)?
    4. Which domains benefit most from federation?
    """

    def __init__(self):
        """Initialize federation experiment."""
        print("=" * 80)
        print("Session 151: Cross-Project Pattern Federation")
        print("=" * 80)
        print()

        # Corpus paths
        self.sage_corpus_path = str(
            Path(__file__).parent / "ep_pattern_corpus_balanced_250.json"
        )
        self.web4_corpus_path = str(
            Path(__file__).parent.parent.parent.parent / "web4" /
            "game" / "ep_pattern_corpus_web4_native.json"
        )

        # Results storage
        self.results = {}

    def load_corpus(self, corpus_path: str) -> Dict[str, Any]:
        """Load pattern corpus from JSON."""
        with open(corpus_path, 'r') as f:
            return json.load(f)

    def analyze_corpus(self, corpus_data: Dict[str, Any], name: str):
        """Analyze corpus composition."""
        patterns = corpus_data.get("patterns", [])

        print(f"{'=' * 80}")
        print(f"Analyzing {name} Corpus")
        print(f"{'=' * 80}")
        print()

        print(f"Total Patterns: {len(patterns)}")
        print()

        # Domain distribution
        domains = Counter()
        for p in patterns:
            domain = p.get("target_domain", p.get("domain", "unknown"))
            domains[domain] += 1

        print("Domain Distribution:")
        for domain, count in sorted(domains.items()):
            pct = count / len(patterns) * 100
            print(f"  {domain:15s}: {count:3d} patterns ({pct:5.1f}%)")
        print()

        # Decision distribution
        decisions = Counter()
        for p in patterns:
            coord = p.get("coordinated_decision", {})
            decision = coord.get("final_decision", "unknown")
            decisions[decision] += 1

        print("Decision Distribution:")
        for decision, count in sorted(decisions.items()):
            pct = count / len(patterns) * 100
            print(f"  {decision:10s}: {count:3d} ({pct:5.1f}%)")
        print()

        # Check for scenario types (if present)
        scenarios = Counter()
        for p in patterns:
            scenario = p.get("scenario_type", "unknown")
            scenarios[scenario] += 1

        if len(scenarios) > 1:  # More than just "unknown"
            print(f"Scenario Types: {len(scenarios)} unique")
            top_scenarios = scenarios.most_common(5)
            for scenario, count in top_scenarios:
                print(f"  {scenario}: {count}")
            print()

        return {
            "name": name,
            "total_patterns": len(patterns),
            "domain_dist": dict(domains),
            "decision_dist": dict(decisions),
            "scenario_types": len(scenarios)
        }

    def create_federated_corpus(
        self,
        sage_data: Dict[str, Any],
        web4_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create federated corpus combining SAGE and Web4 patterns.

        Strategy: Merge all patterns, preserving metadata.
        The system will use K-NN to find best matches regardless of source.
        """
        sage_patterns = sage_data.get("patterns", [])
        web4_patterns = web4_data.get("patterns", [])

        # Tag patterns with source for analysis
        for p in sage_patterns:
            p["source"] = "sage"
        for p in web4_patterns:
            p["source"] = "web4"

        federated = {
            "patterns": sage_patterns + web4_patterns,
            "metadata": {
                "created": datetime.now().isoformat(),
                "sage_patterns": len(sage_patterns),
                "web4_patterns": len(web4_patterns),
                "total_patterns": len(sage_patterns) + len(web4_patterns),
                "federation_version": "1.0"
            }
        }

        return federated

    def save_federated_corpus(self, federated_data: Dict[str, Any]):
        """Save federated corpus to disk."""
        output_path = Path(__file__).parent / "ep_pattern_corpus_federated.json"
        with open(output_path, 'w') as f:
            json.dump(federated_data, f, indent=2)
        print(f"Saved federated corpus to: {output_path}")
        print()
        return str(output_path)

    def test_sage_only(self) -> Dict[str, Any]:
        """Test SAGE consciousness with SAGE-only patterns (baseline)."""
        print("=" * 80)
        print("Test 1: SAGE-Only Patterns (Baseline)")
        print("=" * 80)
        print()

        deployment = ProductionEPDeployment(corpus_path=self.sage_corpus_path)
        stats = deployment.run_production_deployment(verbose=False)

        print("SAGE-Only Results:")
        print(f"  Pattern Match Rate: {stats['pattern_match_rate']:.1f}%")
        print(f"  Cascade Rate: {stats['cascade_rate']:.1f}%")
        print(f"  Avg Confidence Boost: +{stats['avg_confidence_boost']:.3f}")
        print()

        return stats

    def test_federated(self, federated_corpus_path: str) -> Dict[str, Any]:
        """Test SAGE consciousness with federated patterns."""
        print("=" * 80)
        print("Test 2: Federated Patterns (SAGE + Web4)")
        print("=" * 80)
        print()

        deployment = ProductionEPDeployment(corpus_path=federated_corpus_path)
        stats = deployment.run_production_deployment(verbose=False)

        print("Federated Results:")
        print(f"  Pattern Match Rate: {stats['pattern_match_rate']:.1f}%")
        print(f"  Cascade Rate: {stats['cascade_rate']:.1f}%")
        print(f"  Avg Confidence Boost: +{stats['avg_confidence_boost']:.3f}")
        print()

        return stats

    def compare_results(
        self,
        sage_only: Dict[str, Any],
        federated: Dict[str, Any]
    ):
        """Compare SAGE-only vs federated performance."""
        print("=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)
        print()

        metrics = [
            ("Pattern Match Rate", "pattern_match_rate", "%"),
            ("Cascade Rate", "cascade_rate", "%"),
            ("Avg Confidence Boost", "avg_confidence_boost", ""),
            ("Corpus Growth", "corpus_growth", " patterns"),
        ]

        print(f"{'Metric':<25} {'SAGE-Only':>12} {'Federated':>12} {'Delta':>12}")
        print("-" * 65)

        for name, key, unit in metrics:
            sage_val = sage_only.get(key, 0)
            fed_val = federated.get(key, 0)
            delta = fed_val - sage_val

            if unit == "%":
                print(f"{name:<25} {sage_val:>11.1f}% {fed_val:>11.1f}% {delta:>+11.1f}%")
            else:
                print(f"{name:<25} {sage_val:>12.3f} {fed_val:>12.3f} {delta:>+12.3f}")

        print()

        # Interpretation
        print("=" * 80)
        print("INTERPRETATION")
        print("=" * 80)
        print()

        match_delta = federated["pattern_match_rate"] - sage_only["pattern_match_rate"]
        conf_delta = federated["avg_confidence_boost"] - sage_only["avg_confidence_boost"]

        if abs(match_delta) < 5 and abs(conf_delta) < 0.05:
            print("✓ NEUTRAL: Federated patterns neither help nor hurt")
            print()
            print("Implications:")
            print("  - Web4 game patterns don't degrade SAGE performance")
            print("  - Pattern matching is context-robust (good generalization)")
            print("  - No significant transfer learning benefit detected")
            print("  - Corpus size increase (250→350) has minimal impact")
            print()
            print("Possible explanations:")
            print("  1. SAGE patterns already provide excellent coverage")
            print("  2. Web4 patterns too dissimilar to SAGE contexts")
            print("  3. K-NN successfully filters irrelevant patterns")
            print("  4. 100 additional patterns insufficient for measurable impact")

        elif match_delta > 5 or conf_delta > 0.05:
            print("✅ POSITIVE: Federated patterns improve performance!")
            print()
            print("Implications:")
            print("  - Web4 game patterns provide valuable diversity")
            print("  - Cross-project pattern transfer works")
            print("  - Collective learning validated")
            print("  - Larger corpus → better matches")
            print()
            print(f"Improvements:")
            if match_delta > 0:
                print(f"  • Pattern match rate: +{match_delta:.1f}% improvement")
            if conf_delta > 0:
                print(f"  • Confidence boost: +{conf_delta:.3f} improvement")

        else:  # Negative delta
            print("○ NEGATIVE: Federated patterns degrade performance")
            print()
            print("Implications:")
            print("  - Web4 patterns introduce noise or confusion")
            print("  - Cross-project patterns may be too dissimilar")
            print("  - K-NN may match irrelevant game patterns")
            print("  - Context-dependence stronger than expected")
            print()
            print(f"Degradation:")
            if match_delta < 0:
                print(f"  • Pattern match rate: {match_delta:.1f}% worse")
            if conf_delta < 0:
                print(f"  • Confidence boost: {conf_delta:.3f} worse")

        print()

        # Corpus composition analysis
        print("=" * 80)
        print("CORPUS COMPOSITION ANALYSIS")
        print("=" * 80)
        print()

        sage_size = sage_only["initial_corpus_size"]
        fed_size = federated["initial_corpus_size"]
        web4_size = fed_size - sage_size

        print(f"SAGE patterns: {sage_size} (71.4%)")
        print(f"Web4 patterns: {web4_size} (28.6%)")
        print(f"Total federated: {fed_size}")
        print()

        # Domain coverage
        sage_domains = sage_only["patterns_by_domain"]
        fed_domains = federated["patterns_by_domain"]

        print("Domain Coverage:")
        print(f"{'Domain':<15} {'SAGE-Only':>12} {'Federated':>12} {'Change':>12}")
        print("-" * 55)
        for domain in ["emotional", "quality", "attention", "grounding", "authorization"]:
            sage_count = sage_domains.get(domain, 0)
            fed_count = fed_domains.get(domain, 0)
            change = fed_count - sage_count
            print(f"{domain:<15} {sage_count:>12} {fed_count:>12} {change:>+12}")
        print()

    def run_experiment(self):
        """Run complete federation experiment."""
        print("Starting Cross-Project Pattern Federation Experiment")
        print(f"Time: {datetime.now()}")
        print()

        # Step 1: Load and analyze corpora
        print("Step 1: Loading pattern corpora...")
        print()

        sage_data = self.load_corpus(self.sage_corpus_path)
        web4_data = self.load_corpus(self.web4_corpus_path)

        sage_analysis = self.analyze_corpus(sage_data, "SAGE Thor")
        web4_analysis = self.analyze_corpus(web4_data, "Web4 Game")

        # Step 2: Create federated corpus
        print("Step 2: Creating federated corpus...")
        print()

        federated_data = self.create_federated_corpus(sage_data, web4_data)
        print(f"Created federated corpus:")
        print(f"  SAGE patterns: {len(sage_data['patterns'])}")
        print(f"  Web4 patterns: {len(web4_data['patterns'])}")
        print(f"  Total: {len(federated_data['patterns'])}")
        print()

        federated_path = self.save_federated_corpus(federated_data)

        # Step 3: Test SAGE-only (baseline)
        print("Step 3: Testing SAGE-only performance (baseline)...")
        print()
        sage_only_stats = self.test_sage_only()

        # Step 4: Test federated
        print("Step 4: Testing federated performance...")
        print()
        federated_stats = self.test_federated(federated_path)

        # Step 5: Compare
        print("Step 5: Comparing results...")
        print()
        self.compare_results(sage_only_stats, federated_stats)

        # Summary
        print("=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)
        print()

        print("Key Findings:")
        print(f"  • Tested federation of {len(sage_data['patterns'])} SAGE + "
              f"{len(web4_data['patterns'])} Web4 patterns")
        print(f"  • SAGE-only match rate: {sage_only_stats['pattern_match_rate']:.1f}%")
        print(f"  • Federated match rate: {federated_stats['pattern_match_rate']:.1f}%")

        delta = federated_stats['pattern_match_rate'] - sage_only_stats['pattern_match_rate']
        if abs(delta) < 5:
            print(f"  • Result: NEUTRAL (delta: {delta:+.1f}%)")
        elif delta > 0:
            print(f"  • Result: POSITIVE (delta: {delta:+.1f}%)")
        else:
            print(f"  • Result: NEGATIVE (delta: {delta:+.1f}%)")
        print()

        print("Files Created:")
        print(f"  • {federated_path}")
        print()

        print(f"Experiment completed at {datetime.now()}")
        print("=" * 80)

        return {
            "sage_only": sage_only_stats,
            "federated": federated_stats,
            "sage_analysis": sage_analysis,
            "web4_analysis": web4_analysis
        }


def main():
    """Run pattern federation experiment."""
    experiment = PatternFederationExperiment()
    results = experiment.run_experiment()


if __name__ == "__main__":
    main()
