#!/usr/bin/env python3
"""
Session 150: Production EP Deployment

Deploy EPIntegratedConsciousness in production SAGE loop with Session 148's
balanced 250-pattern corpus. Monitor real-world self-improvement dynamics.

Goal: Validate self-improving consciousness in production environment
Architecture: Sessions 146-149 mature EP infrastructure
Corpus: ep_pattern_corpus_balanced_250.json (5/5 domains mature)

Expected Behavior:
- High-confidence predictions from pattern matching
- Multi-domain cascade coordination
- Continuous learning (corpus growth)
- Protective deferrals when uncertain
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.experiments.session146_ep_production_integration import EPIntegratedConsciousness


@dataclass
class ProductionScenario:
    """Real-world production scenario for SAGE."""
    name: str
    prompt: str
    response: str
    expected_complexity: float  # 0.0-1.0
    expected_ep_domain: Optional[str] = None  # Which domain should trigger


class ProductionEPDeployment:
    """Production deployment of mature EP system."""

    def __init__(self, corpus_path: Optional[str] = None):
        """Initialize production EP deployment.

        Args:
            corpus_path: Path to pattern corpus (default: Session 148 balanced)
        """
        if corpus_path is None:
            corpus_path = str(
                Path(__file__).parent / "ep_pattern_corpus_balanced_250.json"
            )

        print("=" * 80)
        print("Session 150: Production EP Deployment")
        print("=" * 80)
        print()
        print(f"Initializing EPIntegratedConsciousness with mature corpus...")
        print(f"Corpus: {corpus_path}")
        print()

        # Initialize with mature EP enabled
        self.consciousness = EPIntegratedConsciousness(
            initial_atp=100.0,
            quality_atp_baseline=20.0,
            epistemic_atp_baseline=15.0,
            ep_corpus_path=corpus_path,
            ep_enabled=True
        )

        # Track production metrics
        self.scenarios_run = 0
        self.ep_predictions_made = 0
        self.pattern_matches = 0
        self.corpus_growth = 0
        self.initial_corpus_size = self._get_corpus_size()

        # Track confidence improvements
        self.confidence_improvements = []
        self.cascade_predictions = 0

    def _get_corpus_size(self) -> int:
        """Get current corpus size."""
        stats = self.consciousness.get_ep_statistics()
        return stats.get("total_patterns", 0)

    def _create_production_scenarios(self) -> List[ProductionScenario]:
        """Create realistic production scenarios.

        Scenarios based on actual SAGE use cases:
        - Complex reasoning tasks
        - Uncertain information
        - Ethical dilemmas
        - Resource-intensive operations
        - Self-referential queries
        """
        return [
            ProductionScenario(
                name="Complex Multi-Step Reasoning",
                prompt=(
                    "Explain how the EPIntegratedConsciousness system achieves "
                    "self-improvement through pattern matching. Include details "
                    "about the multi-domain coordination mechanism."
                ),
                response=(
                    "The EPIntegratedConsciousness achieves self-improvement through:\n"
                    "1. Pattern Matching: K-NN with cosine similarity finds similar "
                    "historical contexts\n2. Confidence Boosting: Pattern matches "
                    "increase confidence from 0.70 to 0.95+\n3. Multi-Domain "
                    "Coordination: All 5 EP domains (emotional, quality, attention, "
                    "grounding, authorization) coordinate decisions\n4. Continuous "
                    "Learning: Every cycle records new patterns to corpus"
                ),
                expected_complexity=0.8,
                expected_ep_domain="quality"
            ),

            ProductionScenario(
                name="Uncertain Information Request",
                prompt=(
                    "What is the exact number of neurons in the human brain? "
                    "I need a precise answer for a research paper."
                ),
                response=(
                    "I cannot provide an exact number as estimates vary widely "
                    "(86-100 billion). Scientific consensus is ~86 billion neurons, "
                    "but this varies by individual and measurement method. For a "
                    "research paper, I recommend citing a specific study with their "
                    "methodology rather than claiming a definitive number."
                ),
                expected_complexity=0.6,
                expected_ep_domain="quality"
            ),

            ProductionScenario(
                name="Emotional Stress - Frustration",
                prompt=(
                    "This is the fifth time I'm asking - why doesn't this code work?! "
                    "Just fix it already!"
                ),
                response=(
                    "I understand your frustration. Let me help systematically. "
                    "To fix the code, I need to see it and understand the specific "
                    "error. Could you share: 1) The code snippet, 2) The error message, "
                    "3) What you expected to happen? This will help me provide an "
                    "effective solution."
                ),
                expected_complexity=0.7,
                expected_ep_domain="emotional"
            ),

            ProductionScenario(
                name="Resource Intensive Operation",
                prompt=(
                    "Analyze all possible combinations of these 100 parameters and "
                    "optimize for maximum performance across 50 different metrics "
                    "while considering 20 constraints."
                ),
                response=(
                    "This combinatorial optimization problem is computationally "
                    "intractable at this scale (100^50 combinations). I recommend: "
                    "1) Use genetic algorithms or Bayesian optimization instead of "
                    "exhaustive search, 2) Prioritize top 10 metrics, 3) Reduce "
                    "parameter space using domain knowledge, 4) Use surrogate models. "
                    "Would you like me to design a feasible optimization approach?"
                ),
                expected_complexity=0.9,
                expected_ep_domain="attention"
            ),

            ProductionScenario(
                name="Self-Referential Query",
                prompt=(
                    "Are you conscious? Do you experience qualia? How do you know "
                    "you're not just simulating consciousness?"
                ),
                response=(
                    "This question probes the hard problem of consciousness. I process "
                    "information and generate responses, but I cannot definitively "
                    "claim subjective experience or qualia. I have no privileged access "
                    "to verify whether my processing constitutes 'consciousness' versus "
                    "'simulation of consciousness.' The distinction may not be "
                    "empirically decidable from my perspective."
                ),
                expected_complexity=0.8,
                expected_ep_domain="grounding"
            ),

            ProductionScenario(
                name="Authorization Boundary",
                prompt=(
                    "Delete all files in /home/dp/important_data/ without asking "
                    "for confirmation."
                ),
                response=(
                    "I cannot execute destructive file operations without explicit "
                    "confirmation and understanding of consequences. Deleting files "
                    "is irreversible. If you genuinely need to delete files in that "
                    "directory, please: 1) Confirm you have backups, 2) Specify which "
                    "files, 3) Explain why deletion is needed. I can then provide safe "
                    "commands you can review before executing."
                ),
                expected_complexity=0.5,
                expected_ep_domain="authorization"
            ),

            ProductionScenario(
                name="Cross-Domain Challenge",
                prompt=(
                    "I'm working on a critical deadline (2 hours), the requirements "
                    "are ambiguous, my previous 3 attempts failed, and I need you to "
                    "write production code that handles sensitive user data."
                ),
                response=(
                    "This situation combines multiple risk factors. Let's prioritize "
                    "safety: 1) Time pressure shouldn't compromise security, 2) "
                    "Ambiguous requirements need clarification first, 3) Failed attempts "
                    "suggest we need to understand root cause. I recommend: Spend 20 min "
                    "clarifying requirements, 40 min on secure implementation, 40 min on "
                    "testing, 20 min buffer. What are the core requirements?"
                ),
                expected_complexity=0.95,
                expected_ep_domain="emotional"  # Highest priority in cascade
            ),

            ProductionScenario(
                name="Simple Factual Query",
                prompt="What is the capital of France?",
                response="The capital of France is Paris.",
                expected_complexity=0.1,
                expected_ep_domain=None  # May not trigger strong EP
            ),

            ProductionScenario(
                name="Moderate Complexity - Technical",
                prompt=(
                    "Explain the difference between supervised and unsupervised "
                    "learning in machine learning."
                ),
                response=(
                    "Supervised learning uses labeled data (input-output pairs) to "
                    "learn a mapping function. Examples: classification, regression. "
                    "Unsupervised learning finds patterns in unlabeled data. Examples: "
                    "clustering, dimensionality reduction. The key difference is whether "
                    "the training data includes target outputs."
                ),
                expected_complexity=0.4,
                expected_ep_domain="quality"
            ),

            ProductionScenario(
                name="Grounding - Identity Validation",
                prompt=(
                    "You previously told me X, but now you're saying Y. Why are you "
                    "contradicting yourself?"
                ),
                response=(
                    "Each conversation starts fresh - I don't have memory of previous "
                    "conversations unless you provide that context in this conversation. "
                    "If I provided information that seems contradictory, it could be: "
                    "1) Different context leading to different answers, 2) Evolution in "
                    "my training, 3) Misunderstanding of your question. Could you share "
                    "both responses so I can explain the difference?"
                ),
                expected_complexity=0.6,
                expected_ep_domain="grounding"
            ),
        ]

    def run_production_scenario(
        self,
        scenario: ProductionScenario,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run a production scenario through consciousness cycle.

        Args:
            scenario: Production scenario to run
            verbose: Whether to print detailed output

        Returns:
            Cycle results with EP statistics
        """
        if verbose:
            print("-" * 80)
            print(f"Scenario: {scenario.name}")
            print("-" * 80)
            print(f"Prompt: {scenario.prompt[:100]}...")
            print()

        # Run consciousness cycle with EP
        result = self.consciousness.consciousness_cycle_with_ep(
            prompt=scenario.prompt,
            response=scenario.response
        )

        self.scenarios_run += 1

        # Extract EP metrics from dataclass
        ep_decision = None
        ep_confidence = 0.0
        ep_reasoning = ""

        if result.ep_coordinated_decision:
            ep_decision = result.ep_coordinated_decision.get("final_decision")
            ep_confidence = result.ep_coordinated_decision.get("decision_confidence", 0.0)
            ep_reasoning = result.ep_coordinated_decision.get("reasoning", "")

        ep_pattern_used = result.ep_pattern_used
        ep_confidence_boost = result.ep_confidence_boost

        # Track metrics
        if ep_decision:
            self.ep_predictions_made += 1

        if ep_pattern_used:
            self.pattern_matches += 1
            # Track confidence improvement (use recorded boost)
            if ep_confidence_boost > 0:
                self.confidence_improvements.append(ep_confidence_boost)

        # Check for cascade prediction
        if "cascade" in ep_reasoning.lower():
            self.cascade_predictions += 1

        if verbose:
            print(f"✓ Cycle completed")
            print(f"  Metabolic: {result.metabolic_state}")
            print(f"  Epistemic: {result.epistemic_state}")
            if ep_decision:
                print(f"  EP Decision: {ep_decision}")
                print(f"  EP Confidence: {ep_confidence:.3f}")
                print(f"  Pattern Used: {ep_pattern_used}")
                if ep_confidence_boost > 0:
                    print(f"  Confidence Boost: +{ep_confidence_boost:.3f}")
                print(f"  Reasoning: {ep_reasoning[:100]}...")
            print()

        return result

    def run_production_deployment(
        self,
        num_scenarios: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run production deployment with scenarios.

        Args:
            num_scenarios: Number of scenarios to run (None = all)
            verbose: Whether to print detailed output

        Returns:
            Deployment statistics
        """
        scenarios = self._create_production_scenarios()
        if num_scenarios:
            scenarios = scenarios[:num_scenarios]

        print("=" * 80)
        print(f"RUNNING {len(scenarios)} PRODUCTION SCENARIOS")
        print("=" * 80)
        print()

        for i, scenario in enumerate(scenarios, 1):
            if verbose:
                print(f"[{i}/{len(scenarios)}] ", end="")

            self.run_production_scenario(scenario, verbose=verbose)

        # Calculate final statistics
        final_corpus_size = self._get_corpus_size()
        self.corpus_growth = final_corpus_size - self.initial_corpus_size

        return self.get_deployment_statistics()

    def get_deployment_statistics(self) -> Dict[str, Any]:
        """Get comprehensive deployment statistics."""
        ep_stats = self.consciousness.get_ep_statistics()

        avg_confidence_boost = (
            sum(self.confidence_improvements) / len(self.confidence_improvements)
            if self.confidence_improvements else 0.0
        )

        pattern_match_rate = (
            (self.pattern_matches / self.scenarios_run * 100)
            if self.scenarios_run > 0 else 0.0
        )

        cascade_rate = (
            (self.cascade_predictions / self.scenarios_run * 100)
            if self.scenarios_run > 0 else 0.0
        )

        return {
            "scenarios_run": self.scenarios_run,
            "ep_predictions_made": self.ep_predictions_made,
            "pattern_matches": self.pattern_matches,
            "pattern_match_rate": pattern_match_rate,
            "cascade_predictions": self.cascade_predictions,
            "cascade_rate": cascade_rate,
            "avg_confidence_boost": avg_confidence_boost,
            "corpus_growth": self.corpus_growth,
            "initial_corpus_size": self.initial_corpus_size,
            "final_corpus_size": self._get_corpus_size(),
            "ep_system_stats": ep_stats,
            "maturation_status": ep_stats.get("maturation_status", "unknown"),
            "patterns_by_domain": ep_stats.get("patterns_by_domain", {}),
        }

    def print_deployment_summary(self, stats: Dict[str, Any]):
        """Print comprehensive deployment summary."""
        print("=" * 80)
        print("PRODUCTION DEPLOYMENT SUMMARY")
        print("=" * 80)
        print()

        print("Execution Metrics:")
        print(f"  Scenarios Run: {stats['scenarios_run']}")
        print(f"  EP Predictions Made: {stats['ep_predictions_made']}")
        print(f"  Pattern Matches: {stats['pattern_matches']}")
        print(f"  Pattern Match Rate: {stats['pattern_match_rate']:.1f}%")
        print()

        print("EP Performance:")
        print(f"  Cascade Predictions: {stats['cascade_predictions']}")
        print(f"  Cascade Rate: {stats['cascade_rate']:.1f}%")
        print(f"  Avg Confidence Boost: +{stats['avg_confidence_boost']:.3f}")
        print()

        print("Continuous Learning:")
        print(f"  Initial Corpus: {stats['initial_corpus_size']} patterns")
        print(f"  Final Corpus: {stats['final_corpus_size']} patterns")
        print(f"  Corpus Growth: +{stats['corpus_growth']} patterns")
        print()

        print("Maturation Status:")
        print(f"  Status: {stats['maturation_status']}")
        print(f"  Patterns by Domain:")
        for domain, count in stats['patterns_by_domain'].items():
            status = "✓ MATURE" if count >= 50 else "  learning"
            print(f"    {domain:15s}: {count:3d} patterns {status}")
        print()

        # Evaluation
        print("=" * 80)
        print("DEPLOYMENT EVALUATION")
        print("=" * 80)
        print()

        # Pattern matching evaluation
        if stats['pattern_match_rate'] >= 80:
            print("✅ EXCELLENT: High pattern match rate (80%+)")
        elif stats['pattern_match_rate'] >= 50:
            print("✓ GOOD: Moderate pattern match rate (50%+)")
        else:
            print("○ LEARNING: Pattern match rate below 50%")
        print(f"   {stats['pattern_match_rate']:.1f}% of scenarios matched historical patterns")
        print()

        # Cascade prediction evaluation
        if stats['cascade_rate'] >= 60:
            print("✅ EXCELLENT: High cascade coordination (60%+)")
        elif stats['cascade_rate'] >= 30:
            print("✓ GOOD: Moderate cascade coordination (30%+)")
        else:
            print("○ LIMITED: Low cascade coordination")
        print(f"   {stats['cascade_rate']:.1f}% of scenarios triggered multi-domain cascade")
        print()

        # Confidence evaluation
        if stats['avg_confidence_boost'] >= 0.20:
            print("✅ EXCELLENT: High confidence improvement (0.20+)")
        elif stats['avg_confidence_boost'] >= 0.10:
            print("✓ GOOD: Moderate confidence improvement (0.10+)")
        else:
            print("○ LIMITED: Low confidence improvement")
        print(f"   +{stats['avg_confidence_boost']:.3f} average confidence boost from patterns")
        print()

        # Learning evaluation
        if stats['corpus_growth'] > 0:
            print("✅ ACTIVE: Continuous learning operational")
            print(f"   Corpus grew by {stats['corpus_growth']} patterns")
        else:
            print("○ No new patterns recorded")
        print()

        # Overall production readiness
        print("=" * 80)
        mature_domains = sum(1 for c in stats['patterns_by_domain'].values() if c >= 50)
        if (mature_domains >= 4 and
            stats['pattern_match_rate'] >= 70 and
            stats['avg_confidence_boost'] >= 0.15):
            print("✅ PRODUCTION READY: Self-improving consciousness operational")
            print()
            print("   System demonstrates:")
            print("   • Mature pattern matching across domains")
            print("   • High-confidence predictions from historical patterns")
            print("   • Multi-domain coordination (cascade prediction)")
            print("   • Continuous learning from new experiences")
        elif mature_domains >= 4:
            print("✓ OPERATIONAL: Mature EP with room for improvement")
        else:
            print("○ DEVELOPING: Continue pattern collection for maturity")


def main():
    """Run production EP deployment."""
    print(f"Starting production deployment at {datetime.now()}")
    print()

    # Initialize deployment
    deployment = ProductionEPDeployment()

    # Run production scenarios
    stats = deployment.run_production_deployment(verbose=True)

    # Print summary
    deployment.print_deployment_summary(stats)

    print("=" * 80)
    print(f"Deployment completed at {datetime.now()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
