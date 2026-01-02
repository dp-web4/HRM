#!/usr/bin/env python3
"""
Session 153: Context Projection Layer for Pattern Federation

Research Question: Can domain-specific context projection enable pattern
transfer between systems with different context structures?

Background:
- Session 151: Naive pattern federation failed (0% match) due to dimension mismatch
  - SAGE: 3 fields per domain (emotional, quality, attention, grounding, authorization)
  - Web4: 4-5 fields per domain (emotional, quality, attention only)
- Web4 Session 117: Successfully used SAGE patterns via domain projection
  - Key insight: context=p["context"].get(domain_name, {})
  - Extract only relevant domain, ignore structural differences

Hypothesis: Context projection layer enables pattern federation by:
1. Extracting domain-specific contexts from multi-domain patterns
2. Mapping between different field structures (3 ↔ 4-5 fields)
3. Enabling pattern matching within single domain despite overall structure differences

Experiments:
1. Implement ContextProjector for domain-specific extraction
2. Test SAGE patterns → Web4 contexts (forward projection)
3. Test Web4 patterns → SAGE contexts (reverse projection)
4. Measure pattern match quality vs Session 151's 0% baseline
5. Analyze information preservation in projection

Expected Outcome: Pattern federation success where Session 151 failed.
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


class ContextProjector:
    """
    Project multi-domain contexts to domain-specific contexts.

    Solves Session 151's dimension mismatch problem by extracting only
    relevant domain context, as validated by Web4 Session 117.
    """

    def __init__(self):
        """Initialize context projector."""
        # Domain field mappings: SAGE ↔ Web4
        self.sage_fields = {
            "emotional": ["frustration", "recent_failure_rate", "complexity"],
            "quality": ["relationship_quality", "recent_quality_avg", "risk_level"],
            "attention": ["atp_level", "estimated_cost", "reserve_threshold"],
            "grounding": ["cross_society", "trust_differential", "coherence_score"],
            "authorization": ["trust_score", "abuse_history", "permission_risk"]
        }

        self.web4_fields = {
            "emotional": ["current_frustration", "recent_failure_rate",
                         "atp_stress", "interaction_complexity"],
            "quality": ["current_relationship_quality", "recent_avg_outcome",
                       "trust_alignment", "interaction_risk_to_quality"],
            "attention": ["atp_available", "atp_cost", "atp_reserve_needed",
                         "interaction_count", "expected_benefit"]
        }

    def project_to_domain(
        self,
        pattern: Dict[str, Any],
        target_domain: str
    ) -> Optional[Dict[str, Any]]:
        """
        Project pattern to single domain context.

        This is the key insight from Web4 Session 117:
        context = pattern["context"].get(domain_name, {})

        Args:
            pattern: Full pattern with multi-domain context
            target_domain: Domain to extract (emotional, quality, attention, etc.)

        Returns:
            Pattern with only target domain context, or None if domain missing
        """
        # Extract domain-specific context
        full_context = pattern.get("context", {})
        domain_context = full_context.get(target_domain)

        if not domain_context:
            return None

        # Create projected pattern
        projected = pattern.copy()
        projected["context"] = {target_domain: domain_context}
        projected["projection_source"] = "full_context"
        projected["projected_domain"] = target_domain

        return projected

    def map_sage_to_web4_fields(
        self,
        sage_context: Dict[str, float],
        domain: str
    ) -> Dict[str, float]:
        """
        Map SAGE 3-field context to Web4 4-5 field context.

        Strategy: Map similar fields, use defaults for Web4-only fields.
        """
        if domain not in self.web4_fields:
            return {}  # Domain doesn't exist in Web4

        web4_context = {}
        sage_vals = list(sage_context.values())

        if domain == "emotional":
            # SAGE: frustration, recent_failure_rate, complexity
            # Web4: current_frustration, recent_failure_rate, atp_stress, interaction_complexity
            web4_context["current_frustration"] = sage_context.get("frustration", 0.0)
            web4_context["recent_failure_rate"] = sage_context.get("recent_failure_rate", 0.0)
            web4_context["atp_stress"] = sage_context.get("complexity", 0.0) * 0.5  # Estimate
            web4_context["interaction_complexity"] = sage_context.get("complexity", 0.0)

        elif domain == "quality":
            # SAGE: relationship_quality, recent_quality_avg, risk_level
            # Web4: current_relationship_quality, recent_avg_outcome, trust_alignment, interaction_risk_to_quality
            web4_context["current_relationship_quality"] = sage_context.get("relationship_quality", 0.5)
            web4_context["recent_avg_outcome"] = sage_context.get("recent_quality_avg", 0.5)
            web4_context["trust_alignment"] = sage_context.get("relationship_quality", 0.5)
            web4_context["interaction_risk_to_quality"] = sage_context.get("risk_level", 0.0)

        elif domain == "attention":
            # SAGE: atp_level, estimated_cost, reserve_threshold
            # Web4: atp_available, atp_cost, atp_reserve_needed, interaction_count, expected_benefit
            web4_context["atp_available"] = sage_context.get("atp_level", 100.0)
            web4_context["atp_cost"] = sage_context.get("estimated_cost", 20.0)
            web4_context["atp_reserve_needed"] = sage_context.get("reserve_threshold", 30.0)
            web4_context["interaction_count"] = 1.0  # Default
            web4_context["expected_benefit"] = 0.5  # Default

        return web4_context

    def map_web4_to_sage_fields(
        self,
        web4_context: Dict[str, float],
        domain: str
    ) -> Dict[str, float]:
        """
        Map Web4 4-5 field context to SAGE 3-field context.

        Strategy: Map similar fields, discard Web4-only fields.
        """
        if domain not in self.sage_fields:
            return {}  # Domain doesn't exist in SAGE (grounding, authorization)

        sage_context = {}

        if domain == "emotional":
            # Web4: current_frustration, recent_failure_rate, atp_stress, interaction_complexity
            # SAGE: frustration, recent_failure_rate, complexity
            sage_context["frustration"] = web4_context.get("current_frustration", 0.0)
            sage_context["recent_failure_rate"] = web4_context.get("recent_failure_rate", 0.0)
            # Combine interaction_complexity and atp_stress for overall complexity
            complexity = max(
                web4_context.get("interaction_complexity", 0.0),
                web4_context.get("atp_stress", 0.0)
            )
            sage_context["complexity"] = complexity

        elif domain == "quality":
            # Web4: current_relationship_quality, recent_avg_outcome, trust_alignment, interaction_risk_to_quality
            # SAGE: relationship_quality, recent_quality_avg, risk_level
            sage_context["relationship_quality"] = web4_context.get("current_relationship_quality", 0.5)
            sage_context["recent_quality_avg"] = web4_context.get("recent_avg_outcome", 0.5)
            sage_context["risk_level"] = web4_context.get("interaction_risk_to_quality", 0.0)

        elif domain == "attention":
            # Web4: atp_available, atp_cost, atp_reserve_needed, interaction_count, expected_benefit
            # SAGE: atp_level, estimated_cost, reserve_threshold
            sage_context["atp_level"] = web4_context.get("atp_available", 100.0)
            sage_context["estimated_cost"] = web4_context.get("atp_cost", 20.0)
            sage_context["reserve_threshold"] = web4_context.get("atp_reserve_needed", 30.0)

        return sage_context


class ProjectionFederationExperiment:
    """
    Test pattern federation via context projection.

    This retries Session 151's federation experiment, but with projection layer.
    Expected: Success where Session 151 failed (0% → high match rate).
    """

    def __init__(self):
        """Initialize projection federation experiment."""
        print("=" * 80)
        print("Session 153: Context Projection Layer for Pattern Federation")
        print("=" * 80)
        print()

        self.projector = ContextProjector()

        # Corpus paths
        self.sage_corpus_path = str(
            Path(__file__).parent / "ep_pattern_corpus_balanced_250.json"
        )
        self.web4_corpus_path = str(
            Path(__file__).parent.parent.parent.parent / "web4" /
            "game" / "ep_pattern_corpus_web4_native.json"
        )

        self.results = {}

    def load_corpus(self, corpus_path: str) -> Dict[str, Any]:
        """Load pattern corpus from JSON."""
        with open(corpus_path, 'r') as f:
            return json.load(f)

    def create_projected_corpus(
        self,
        source_patterns: List[Dict[str, Any]],
        target_system: str,
        source_system: str
    ) -> List[Dict[str, Any]]:
        """
        Create projected corpus from source to target system.

        Args:
            source_patterns: Patterns from source system
            target_system: "sage" or "web4"
            source_system: "sage" or "web4"

        Returns:
            List of projected patterns compatible with target system
        """
        projected_patterns = []

        for pattern in source_patterns:
            # Get target domain
            target_domain = pattern.get("target_domain", "emotional")

            # Project to domain-specific context
            projected = self.projector.project_to_domain(pattern, target_domain)

            if not projected:
                continue  # Skip if domain missing

            # Map fields if cross-system
            if source_system != target_system:
                domain_context = projected["context"][target_domain]

                if target_system == "web4":
                    # SAGE → Web4 field mapping
                    mapped_context = self.projector.map_sage_to_web4_fields(
                        domain_context, target_domain
                    )
                else:  # target_system == "sage"
                    # Web4 → SAGE field mapping
                    mapped_context = self.projector.map_web4_to_sage_fields(
                        domain_context, target_domain
                    )

                if not mapped_context:
                    continue  # Skip if domain incompatible

                projected["context"][target_domain] = mapped_context

            # Tag with projection metadata
            projected["projection_target"] = target_system
            projected["projection_source_system"] = source_system

            projected_patterns.append(projected)

        return projected_patterns

    def test_projection_quality(
        self,
        original_patterns: List[Dict[str, Any]],
        projected_patterns: List[Dict[str, Any]],
        source_system: str,
        target_system: str
    ):
        """Analyze projection quality and information preservation."""
        print("=" * 80)
        print(f"Projection Quality: {source_system.upper()} → {target_system.upper()}")
        print("=" * 80)
        print()

        print(f"Original Patterns: {len(original_patterns)}")
        print(f"Projected Patterns: {len(projected_patterns)}")
        print(f"Projection Success Rate: {len(projected_patterns)/len(original_patterns)*100:.1f}%")
        print()

        # Domain distribution before/after
        original_domains = Counter()
        projected_domains = Counter()

        for p in original_patterns:
            domain = p.get("target_domain", "emotional")
            original_domains[domain] += 1

        for p in projected_patterns:
            domain = p.get("target_domain", "emotional")
            projected_domains[domain] += 1

        print("Domain Distribution After Projection:")
        print(f"{'Domain':<15} {'Original':>10} {'Projected':>10} {'Loss':>10}")
        print("-" * 50)

        for domain in sorted(original_domains.keys()):
            orig_count = original_domains[domain]
            proj_count = projected_domains.get(domain, 0)
            loss = orig_count - proj_count
            print(f"{domain:<15} {orig_count:>10} {proj_count:>10} {loss:>10}")

        print()

        # Field mapping analysis
        if projected_patterns:
            sample = projected_patterns[0]
            domain = sample.get("target_domain", "emotional")
            context = sample["context"].get(domain, {})

            print(f"Sample Projected Context ({domain} domain):")
            for field, value in context.items():
                print(f"  {field}: {value:.3f}")
            print()

    def test_sage_with_projected_web4(self) -> Dict[str, Any]:
        """
        Test SAGE with Web4 patterns projected to SAGE structure.

        This is Session 151's failed experiment, retried with projection.
        """
        print("=" * 80)
        print("Test 1: SAGE + Projected Web4 Patterns")
        print("=" * 80)
        print()

        # Load corpora
        sage_data = self.load_corpus(self.sage_corpus_path)
        web4_data = self.load_corpus(self.web4_corpus_path)

        sage_patterns = sage_data.get("patterns", [])
        web4_patterns = web4_data.get("patterns", [])

        print(f"SAGE patterns: {len(sage_patterns)}")
        print(f"Web4 patterns: {len(web4_patterns)}")
        print()

        # Project Web4 patterns to SAGE structure
        print("Projecting Web4 patterns to SAGE structure...")
        projected_web4 = self.create_projected_corpus(
            web4_patterns,
            target_system="sage",
            source_system="web4"
        )

        self.test_projection_quality(
            web4_patterns, projected_web4, "web4", "sage"
        )

        # Create federated corpus with projection
        federated_patterns = sage_patterns + projected_web4

        federated_data = {
            "patterns": federated_patterns,
            "metadata": {
                "created": datetime.now().isoformat(),
                "sage_patterns": len(sage_patterns),
                "web4_projected_patterns": len(projected_web4),
                "total_patterns": len(federated_patterns),
                "projection_version": "1.0",
                "projection_method": "domain_specific_with_field_mapping"
            }
        }

        # Save projected corpus
        output_path = Path(__file__).parent / "ep_pattern_corpus_projected_federation.json"
        with open(output_path, 'w') as f:
            json.dump(federated_data, f, indent=2)

        print(f"Created projected federated corpus: {len(federated_patterns)} patterns")
        print(f"Saved to: {output_path}")
        print()

        # Test SAGE with projected federation
        print("Testing SAGE with projected federated corpus...")
        deployment = ProductionEPDeployment(corpus_path=str(output_path))
        stats = deployment.run_production_deployment(verbose=False)

        print("Results:")
        print(f"  Pattern Match Rate: {stats['pattern_match_rate']:.1f}%")
        print(f"  Cascade Rate: {stats['cascade_rate']:.1f}%")
        print(f"  Avg Confidence Boost: +{stats['avg_confidence_boost']:.3f}")
        print()

        return {
            "stats": stats,
            "sage_patterns": len(sage_patterns),
            "projected_web4_patterns": len(projected_web4),
            "total_patterns": len(federated_patterns),
            "output_path": str(output_path)
        }

    def compare_with_session151(
        self,
        projected_stats: Dict[str, Any]
    ):
        """Compare projected federation with Session 151's naive federation."""
        print("=" * 80)
        print("COMPARISON: Session 151 vs Session 153")
        print("=" * 80)
        print()

        # Session 151 results (from documentation)
        session151_match_rate = 0.0  # Complete failure
        session151_cascade_rate = 0.0
        session151_confidence_boost = 0.0

        # Session 153 results
        session153_stats = projected_stats["stats"]
        session153_match_rate = session153_stats["pattern_match_rate"]
        session153_cascade_rate = session153_stats["cascade_rate"]
        session153_confidence_boost = session153_stats["avg_confidence_boost"]

        print(f"{'Metric':<25} {'Session 151':>15} {'Session 153':>15} {'Improvement':>15}")
        print("-" * 75)
        print(f"{'Pattern Match Rate':<25} {session151_match_rate:>14.1f}% {session153_match_rate:>14.1f}% {session153_match_rate - session151_match_rate:>+14.1f}%")
        print(f"{'Cascade Rate':<25} {session151_cascade_rate:>14.1f}% {session153_cascade_rate:>14.1f}% {session153_cascade_rate - session151_cascade_rate:>+14.1f}%")
        print(f"{'Avg Confidence Boost':<25} {session151_confidence_boost:>15.3f} {session153_confidence_boost:>15.3f} {session153_confidence_boost - session151_confidence_boost:>+15.3f}")
        print()

        print("Analysis:")
        if session153_match_rate > 80:
            print("✅ SUCCESS: Context projection enables pattern federation!")
            print()
            print("Key Insights:")
            print("  • Domain-specific projection solves dimension mismatch")
            print("  • Field mapping preserves semantic information")
            print("  • Pattern matching works across system boundaries")
            print(f"  • Improvement: {session153_match_rate - session151_match_rate:+.1f}% match rate")
            print()
            print("Validation:")
            print("  • Web4 Session 117: Proved projection works (Web4 using SAGE)")
            print("  • Session 153: Proves reverse works (SAGE using Web4)")
            print("  • Bidirectional pattern federation VALIDATED")
        elif session153_match_rate > 50:
            print("○ PARTIAL SUCCESS: Projection helps but not complete")
            print()
            print(f"  • Match rate: {session153_match_rate:.1f}% (better than 0%, not ideal)")
            print("  • Possible issues:")
            print("    - Information loss in field mapping")
            print("    - Semantic mismatch between systems")
            print("    - Need better mapping heuristics")
        else:
            print("❌ PROJECTION INSUFFICIENT: Still low match rate")
            print()
            print("  • Domain projection alone not enough")
            print("  • May need semantic alignment, not just structural")

        print()

    def run_experiment(self):
        """Run complete projection federation experiment."""
        print("Starting Context Projection Federation Experiment")
        print(f"Time: {datetime.now()}")
        print()
        print("Research Question:")
        print("  Can domain-specific context projection enable pattern federation")
        print("  between systems with different context structures?")
        print()
        print("Background:")
        print("  • Session 151: Naive federation failed (0% match) - dimension mismatch")
        print("  • Web4 Session 117: Projection works (SAGE → Web4)")
        print("  • Session 153: Test reverse (Web4 → SAGE) and validate federation")
        print()

        # Test SAGE with projected Web4 patterns
        projected_results = self.test_sage_with_projected_web4()

        # Compare with Session 151
        self.compare_with_session151(projected_results)

        # Summary
        print("=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)
        print()

        stats = projected_results["stats"]
        print("Key Findings:")
        print(f"  • Projected {projected_results['projected_web4_patterns']} Web4 patterns to SAGE structure")
        print(f"  • Total federated corpus: {projected_results['total_patterns']} patterns")
        print(f"  • Pattern match rate: {stats['pattern_match_rate']:.1f}%")
        print(f"  • Improvement vs Session 151: {stats['pattern_match_rate'] - 0.0:+.1f}%")
        print()

        if stats['pattern_match_rate'] > 80:
            print("✅ CONCLUSION: Context projection enables pattern federation")
            print("   Domain-specific extraction + field mapping solves Session 151's problem")
        else:
            print("○ CONCLUSION: Projection helps but may need refinement")

        print()
        print(f"Experiment completed at {datetime.now()}")
        print("=" * 80)

        return projected_results


def main():
    """Run context projection federation experiment."""
    experiment = ProjectionFederationExperiment()
    results = experiment.run_experiment()


if __name__ == "__main__":
    main()
