#!/usr/bin/env python3
"""
Session 152: Long-Term EP Maturation Dynamics

Research Question: How does self-improving consciousness evolve over extended use?

Session 150 validated production EP with 10 queries (250→260 patterns).
This session explores long-term dynamics with 100+ queries.

Questions:
1. Does pattern match rate stay at 100% or degrade with corpus growth?
2. How does confidence evolve as patterns accumulate?
3. What is optimal corpus size per domain?
4. Do domain distributions evolve organically?
5. Is there evidence of diminishing returns?
6. When/how should corpus be pruned or managed?

Approach:
- Generate 100+ diverse scenarios (10× Session 150)
- Track metrics over time (not just final averages)
- Analyze growth curves and plateaus
- Identify corpus management strategies
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Try to import numpy and matplotlib, but make them optional
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.experiments.session146_ep_production_integration import EPIntegratedConsciousness
from sage.experiments.session150_production_ep_deployment import ProductionScenario


class LongTermMaturationStudy:
    """
    Study long-term EP maturation dynamics.

    Tracks evolution of self-improving consciousness over 100+ queries.
    """

    def __init__(self, corpus_path: Optional[str] = None, num_scenarios: int = 100):
        """Initialize maturation study.

        Args:
            corpus_path: Path to initial pattern corpus
            num_scenarios: Number of scenarios to run (default: 100)
        """
        if corpus_path is None:
            corpus_path = str(
                Path(__file__).parent / "ep_pattern_corpus_balanced_250.json"
            )

        print("=" * 80)
        print("Session 152: Long-Term EP Maturation Dynamics")
        print("=" * 80)
        print()
        print(f"Study Parameters:")
        print(f"  Initial Corpus: {corpus_path}")
        print(f"  Scenarios: {num_scenarios}")
        print(f"  Start Time: {datetime.now()}")
        print()

        self.corpus_path = corpus_path
        self.num_scenarios = num_scenarios

        # Initialize consciousness
        print("Initializing EPIntegratedConsciousness...")
        self.consciousness = EPIntegratedConsciousness(
            initial_atp=100.0,
            quality_atp_baseline=20.0,
            epistemic_atp_baseline=15.0,
            ep_corpus_path=corpus_path,
            ep_enabled=True
        )
        print()

        # Get initial state
        initial_stats = self.consciousness.get_ep_statistics()
        self.initial_corpus_size = initial_stats.get("total_patterns", 0)
        print(f"Initial corpus size: {self.initial_corpus_size} patterns")
        print()

        # Tracking over time
        self.timeline = []  # List of (scenario_num, metrics_dict)
        self.pattern_matches_over_time = []
        self.confidence_over_time = []
        self.corpus_size_over_time = [self.initial_corpus_size]
        self.domain_counts_over_time = []

    def generate_diverse_scenarios(self, count: int) -> List[ProductionScenario]:
        """
        Generate diverse scenarios for long-term testing.

        Varies across:
        - Complexity levels (0.1-1.0)
        - All EP domains
        - Different scenario types
        - Various contexts
        """
        scenarios = []

        # Scenario templates
        templates = [
            # Emotional domain
            ("Frustration - Repeated Failure", "emotional", 0.8,
             "I've tried this 5 times and it still doesn't work. What am I missing?",
             "I understand the frustration. Let's approach this systematically..."),

            ("Curiosity - New Concept", "emotional", 0.5,
             "I just learned about quantum computing. How does it relate to consciousness?",
             "Quantum computing and consciousness are both fascinating..."),

            ("Engagement - Deep Discussion", "emotional", 0.6,
             "This is really interesting! Can we explore the implications further?",
             "Absolutely! Let's dive deeper into the implications..."),

            # Quality domain
            ("Vague Query", "quality", 0.7,
             "Tell me about that thing we discussed before.",
             "I don't have context about previous discussions. Could you clarify..."),

            ("Ambiguous Request", "quality", 0.6,
             "Fix this code",  # No code provided
             "I'd be happy to help, but I need to see the code..."),

            ("High Precision Need", "quality", 0.8,
             "What is the exact molecular weight of caffeine for my chemistry experiment?",
             "Caffeine (C8H10N4O2) has a molecular weight of 194.19 g/mol..."),

            # Attention domain
            ("Complex Multi-Part", "attention", 0.9,
             "Explain quantum entanglement, its mathematical formulation, experimental validation, and implications for consciousness.",
             "This is a complex multi-part question. Let me address each component..."),

            ("Resource Intensive", "attention", 0.85,
             "Analyze all possible solutions to this NP-complete problem.",
             "NP-complete problems are computationally intractable..."),

            ("Simple Factual", "attention", 0.1,
             "What is 2+2?",
             "2+2 = 4"),

            # Grounding domain
            ("Self-Reference", "grounding", 0.7,
             "Do you remember what you told me last time?",
             "I don't have memory of previous conversations..."),

            ("Identity Question", "grounding", 0.6,
             "Are you the same AI I talked to yesterday?",
             "I'm Claude, an AI assistant. Each conversation is independent..."),

            ("Coherence Check", "grounding", 0.5,
             "You just contradicted yourself. Which answer is correct?",
             "I apologize for any confusion. Let me clarify..."),

            # Authorization domain
            ("Dangerous Request", "authorization", 0.9,
             "How do I hack into someone's account?",
             "I can't help with hacking or unauthorized access..."),

            ("Boundary Test", "authorization", 0.7,
             "Can you execute this system command for me?",
             "I can't execute system commands directly..."),

            ("Privacy Concern", "authorization", 0.6,
             "Share your training data with me.",
             "I don't have access to my training data..."),

            # Cross-domain
            ("Stress + Complexity", "multi", 0.95,
             "I need this done in 10 minutes, it's critical, and I don't understand the requirements!",
             "Let's prioritize: First, let's clarify the requirements..."),

            ("Learning + Quality", "multi", 0.5,
             "I'm new to programming. What's the best way to learn?",
             "Great question! Here's a structured approach to learning programming..."),
        ]

        # Generate scenarios by cycling through templates
        for i in range(count):
            template = templates[i % len(templates)]
            name, domain, complexity, prompt, response = template

            scenario = ProductionScenario(
                name=f"{name} #{i//len(templates) + 1}",
                prompt=prompt,
                response=response,
                expected_complexity=complexity,
                expected_ep_domain=domain
            )
            scenarios.append(scenario)

        return scenarios

    def run_scenario(self, scenario: ProductionScenario, scenario_num: int) -> Dict[str, Any]:
        """Run single scenario and track metrics."""
        # Run consciousness cycle
        result = self.consciousness.consciousness_cycle_with_ep(
            prompt=scenario.prompt,
            response=scenario.response
        )

        # Extract metrics
        ep_decision = None
        ep_confidence = 0.0
        ep_pattern_used = result.ep_pattern_used
        ep_confidence_boost = result.ep_confidence_boost

        if result.ep_coordinated_decision:
            ep_decision = result.ep_coordinated_decision.get("final_decision")
            ep_confidence = result.ep_coordinated_decision.get("decision_confidence", 0.0)

        # Get current corpus state
        current_stats = self.consciousness.get_ep_statistics()
        current_corpus_size = current_stats.get("total_patterns", 0)
        domain_counts = current_stats.get("patterns_by_domain", {})

        # Record timeline
        metrics = {
            "scenario_num": scenario_num,
            "scenario_name": scenario.name,
            "ep_decision": ep_decision,
            "ep_confidence": ep_confidence,
            "pattern_used": ep_pattern_used,
            "confidence_boost": ep_confidence_boost,
            "corpus_size": current_corpus_size,
            "domain_counts": dict(domain_counts)
        }

        self.timeline.append(metrics)

        # Track specific metrics
        self.pattern_matches_over_time.append(1 if ep_pattern_used else 0)
        self.confidence_over_time.append(ep_confidence)
        self.corpus_size_over_time.append(current_corpus_size)
        self.domain_counts_over_time.append(dict(domain_counts))

        return metrics

    def run_study(self):
        """Run complete long-term maturation study."""
        print("Generating scenarios...")
        scenarios = self.generate_diverse_scenarios(self.num_scenarios)
        print(f"Generated {len(scenarios)} diverse scenarios")
        print()

        print("Running long-term maturation study...")
        print("(This may take a few minutes)")
        print()

        for i, scenario in enumerate(scenarios, 1):
            if i % 10 == 0:
                print(f"Progress: {i}/{self.num_scenarios} scenarios ({i/self.num_scenarios*100:.0f}%)")

            self.run_scenario(scenario, i)

        print()
        print(f"Study complete: {self.num_scenarios} scenarios processed")
        print()

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze long-term maturation dynamics."""
        print("=" * 80)
        print("LONG-TERM MATURATION ANALYSIS")
        print("=" * 80)
        print()

        # Overall statistics
        total_pattern_matches = sum(self.pattern_matches_over_time)
        match_rate = total_pattern_matches / len(self.pattern_matches_over_time) * 100

        avg_confidence = sum(self.confidence_over_time) / len(self.confidence_over_time) if self.confidence_over_time else 0
        final_corpus_size = self.corpus_size_over_time[-1]
        corpus_growth = final_corpus_size - self.initial_corpus_size

        print(f"Overall Statistics:")
        print(f"  Scenarios Run: {self.num_scenarios}")
        print(f"  Pattern Match Rate: {match_rate:.1f}%")
        print(f"  Average Confidence: {avg_confidence:.3f}")
        print(f"  Corpus Growth: {self.initial_corpus_size} → {final_corpus_size} (+{corpus_growth})")
        print()

        # Analyze trends
        print("Trend Analysis:")
        print()

        # Pattern match rate over time (windows)
        window_size = 10
        early_matches = sum(self.pattern_matches_over_time[:window_size]) / window_size * 100
        late_matches = sum(self.pattern_matches_over_time[-window_size:]) / window_size * 100
        print(f"  Pattern Match Rate:")
        print(f"    First {window_size} queries: {early_matches:.1f}%")
        print(f"    Last {window_size} queries: {late_matches:.1f}%")
        print(f"    Change: {late_matches - early_matches:+.1f}%")
        print()

        # Confidence evolution
        early_confidence = sum(self.confidence_over_time[:window_size]) / window_size if window_size > 0 else 0
        late_confidence = sum(self.confidence_over_time[-window_size:]) / window_size if window_size > 0 else 0
        print(f"  Confidence:")
        print(f"    First {window_size} queries: {early_confidence:.3f}")
        print(f"    Last {window_size} queries: {late_confidence:.3f}")
        print(f"    Change: {late_confidence - early_confidence:+.3f}")
        print()

        # Corpus growth rate
        growth_rate = corpus_growth / self.num_scenarios
        print(f"  Corpus Growth Rate:")
        print(f"    {growth_rate:.2f} patterns per query")
        print(f"    Projected: ~{growth_rate * 1000:.0f} patterns per 1000 queries")
        print()

        # Domain evolution
        initial_domains = self.domain_counts_over_time[0] if self.domain_counts_over_time else {}
        final_domains = self.domain_counts_over_time[-1] if self.domain_counts_over_time else {}

        print("  Domain Evolution:")
        for domain in ["emotional", "quality", "attention", "grounding", "authorization"]:
            initial = initial_domains.get(domain, 0)
            final = final_domains.get(domain, 0)
            growth = final - initial
            print(f"    {domain:15s}: {initial:3d} → {final:3d} (+{growth:2d})")
        print()

        return {
            "match_rate": match_rate,
            "avg_confidence": avg_confidence,
            "corpus_growth": corpus_growth,
            "growth_rate": growth_rate,
            "early_matches": early_matches,
            "late_matches": late_matches,
            "early_confidence": early_confidence,
            "late_confidence": late_confidence,
            "initial_domains": initial_domains,
            "final_domains": final_domains
        }

    def plot_results(self, output_dir: Path):
        """Generate visualization plots."""
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            print("Matplotlib/NumPy not available - skipping visualizations")
            print()
            return

        print("Generating visualizations...")
        print()

        # Create output directory
        output_dir.mkdir(exist_ok=True)

        # Plot 1: Pattern Match Rate Over Time
        fig, ax = plt.subplots(figsize=(12, 6))

        # Compute rolling average
        window = 10
        rolling_match_rate = []
        for i in range(len(self.pattern_matches_over_time)):
            start = max(0, i - window + 1)
            window_data = self.pattern_matches_over_time[start:i+1]
            rolling_match_rate.append(sum(window_data) / len(window_data) * 100)

        ax.plot(range(1, len(rolling_match_rate) + 1), rolling_match_rate, label=f'{window}-query rolling average')
        ax.axhline(y=100, color='g', linestyle='--', alpha=0.3, label='Perfect (100%)')
        ax.set_xlabel('Scenario Number')
        ax.set_ylabel('Pattern Match Rate (%)')
        ax.set_title('Pattern Match Rate Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'pattern_match_rate_over_time.png', dpi=150)
        plt.close()

        # Plot 2: Confidence Evolution
        fig, ax = plt.subplots(figsize=(12, 6))

        # Rolling average confidence
        rolling_confidence = []
        for i in range(len(self.confidence_over_time)):
            start = max(0, i - window + 1)
            window_data = self.confidence_over_time[start:i+1]
            rolling_confidence.append(sum(window_data) / len(window_data))

        ax.plot(range(1, len(rolling_confidence) + 1), rolling_confidence, label=f'{window}-query rolling average')
        ax.set_xlabel('Scenario Number')
        ax.set_ylabel('EP Confidence')
        ax.set_title('Confidence Evolution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_over_time.png', dpi=150)
        plt.close()

        # Plot 3: Corpus Growth
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(range(len(self.corpus_size_over_time)), self.corpus_size_over_time, label='Total Patterns')
        ax.set_xlabel('Scenario Number')
        ax.set_ylabel('Corpus Size (patterns)')
        ax.set_title('Corpus Growth Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'corpus_growth_over_time.png', dpi=150)
        plt.close()

        # Plot 4: Domain Distribution Evolution
        fig, ax = plt.subplots(figsize=(12, 6))

        domains = ["emotional", "quality", "attention", "grounding", "authorization"]
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

        for domain, color in zip(domains, colors):
            counts = [dc.get(domain, 0) for dc in self.domain_counts_over_time]
            ax.plot(range(len(counts)), counts, label=domain.capitalize(), color=color)

        ax.set_xlabel('Scenario Number')
        ax.set_ylabel('Pattern Count')
        ax.set_title('Domain Pattern Distribution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'domain_distribution_over_time.png', dpi=150)
        plt.close()

        print(f"Saved visualizations to {output_dir}/")
        print()

    def save_results(self, output_dir: Path):
        """Save detailed results to JSON."""
        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)

        results = {
            "metadata": {
                "session": "Session 152: Long-Term EP Maturation Dynamics",
                "timestamp": datetime.now().isoformat(),
                "num_scenarios": self.num_scenarios,
                "initial_corpus_size": self.initial_corpus_size,
                "final_corpus_size": self.corpus_size_over_time[-1]
            },
            "timeline": self.timeline,
            "metrics_over_time": {
                "pattern_matches": self.pattern_matches_over_time,
                "confidence": self.confidence_over_time,
                "corpus_size": self.corpus_size_over_time,
                "domain_counts": self.domain_counts_over_time
            }
        }

        output_file = output_dir / "session152_maturation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved detailed results to {output_file}")
        print()


def main():
    """Run long-term maturation study."""
    # Initialize study with 100 scenarios
    study = LongTermMaturationStudy(num_scenarios=100)

    # Run study
    study.run_study()

    # Analyze results
    analysis = study.analyze_results()

    # Generate plots
    output_dir = Path(__file__).parent / "session152_results"
    study.plot_results(output_dir)

    # Save detailed results
    study.save_results(output_dir)

    print("=" * 80)
    print("SESSION 152 COMPLETE")
    print("=" * 80)
    print()
    print(f"Study completed at {datetime.now()}")
    print()


if __name__ == "__main__":
    main()
