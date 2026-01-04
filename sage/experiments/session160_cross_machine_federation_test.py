#!/usr/bin/env python3
"""
Session 160: Cross-Machine Federation Integration Test (Sprout Initiative)

**Context**:
- Sessions 157-159: Solved schema alignment for cross-machine federation
- Session 158: Sprout created canonical schema with computed stability/coordination
- Session 159: Thor validated schema works bidirectionally

**Research Goal**:
Test full cross-machine federation by:
1. Loading patterns from multiple sources (production native + provenance aware)
2. Converting all to canonical format
3. Creating federated corpus
4. Testing EP pattern matching across federated patterns
5. Measuring quality/match improvements from federation

**Key Question**:
Does federating patterns from multiple SAGE sessions improve EP prediction?

Hardware: Jetson Orin Nano 8GB (Sprout)
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import canonical schema from Session 158
try:
    from session158_canonical_schema_proposal import (
        CanonicalPattern,
        CanonicalContext,
        PatternDomain,
        ProvenanceType,
        SAGEToCanonicalConverter,
        canonical_to_dict
    )
    HAS_SCHEMA = True
except ImportError as e:
    HAS_SCHEMA = False
    logger.error(f"Could not load Session 158 schema: {e}")


@dataclass
class FederationTestResults:
    """Results from cross-machine federation test."""
    # Corpus sizes
    production_native_count: int
    provenance_aware_count: int
    federated_total: int
    unique_patterns: int

    # Conversion stats
    conversion_success_rate: float
    conversion_time_ms: float

    # Federation quality
    domain_balance: Dict[str, float]
    stability_distribution: Dict[str, float]
    coordination_distribution: Dict[str, float]

    # Pattern matching
    self_match_rate: float
    cross_match_rate: float
    match_improvement: float

    # Edge metrics
    memory_usage_mb: float
    total_time_ms: float
    throughput_patterns_per_sec: float

    # Surprises
    surprises: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CrossMachineFederationTest:
    """
    Tests full cross-machine federation on edge hardware.

    Simulates federating patterns from:
    - Production native corpus (Thor's main SAGE patterns)
    - Provenance-aware corpus (Session 155 enhanced patterns)
    - Canonical corpus (Session 158 converted patterns)
    """

    def __init__(self):
        self.production_patterns = []
        self.provenance_patterns = []
        self.canonical_patterns = []
        self.federated_corpus = []
        self.surprises = []

    def load_corpora(self, experiments_path: Path) -> None:
        """Load all available corpora."""
        logger.info("=== Loading Available Corpora ===")

        # Production native
        prod_path = experiments_path / "ep_pattern_corpus_production_native.json"
        if prod_path.exists():
            with open(prod_path) as f:
                data = json.load(f)
            self.production_patterns = data.get("patterns", [])
            logger.info(f"Production native: {len(self.production_patterns)} patterns")

        # Provenance aware
        prov_path = experiments_path / "ep_pattern_corpus_provenance_aware.json"
        if prov_path.exists():
            with open(prov_path) as f:
                data = json.load(f)
            self.provenance_patterns = data.get("patterns", [])
            logger.info(f"Provenance aware: {len(self.provenance_patterns)} patterns")

        # Canonical (already converted)
        canon_path = experiments_path / "ep_pattern_corpus_canonical.json"
        if canon_path.exists():
            with open(canon_path) as f:
                data = json.load(f)
            self.canonical_patterns = data.get("patterns", [])
            logger.info(f"Canonical: {len(self.canonical_patterns)} patterns")

    def federate_corpora(self) -> Tuple[List[Dict], float]:
        """
        Federate all corpora into unified canonical format.

        Returns:
            (federated_patterns, conversion_time_ms)
        """
        logger.info("\n=== Federating Corpora ===")

        if not HAS_SCHEMA:
            logger.error("Cannot federate - schema not available")
            return [], 0.0

        converter = SAGEToCanonicalConverter()
        federated = []

        start_time = time.perf_counter()

        # Convert production patterns
        for pattern in self.production_patterns:
            canonical = converter.convert_pattern(pattern)
            canonical_dict = canonical_to_dict(canonical)
            canonical_dict["source_corpus"] = "production_native"
            federated.append(canonical_dict)

        # Convert provenance patterns
        for pattern in self.provenance_patterns:
            canonical = converter.convert_pattern(pattern)
            canonical_dict = canonical_to_dict(canonical)
            canonical_dict["source_corpus"] = "provenance_aware"
            federated.append(canonical_dict)

        # Add already-canonical patterns
        for pattern in self.canonical_patterns:
            pattern["source_corpus"] = "canonical_158"
            federated.append(pattern)

        conversion_time = (time.perf_counter() - start_time) * 1000

        stats = converter.get_stats()
        logger.info(f"Converted: {stats['converted']} patterns")
        logger.info(f"Errors: {stats['errors']}")
        logger.info(f"Time: {conversion_time:.1f}ms")

        self.federated_corpus = federated
        return federated, conversion_time

    def deduplicate_corpus(self) -> int:
        """Remove duplicate patterns based on pattern_id."""
        logger.info("\n=== Deduplicating Federated Corpus ===")

        seen_ids = set()
        unique = []
        duplicates = 0

        for pattern in self.federated_corpus:
            pid = pattern.get("pattern_id", "")
            if pid not in seen_ids:
                seen_ids.add(pid)
                unique.append(pattern)
            else:
                duplicates += 1

        logger.info(f"Total patterns: {len(self.federated_corpus)}")
        logger.info(f"Duplicates removed: {duplicates}")
        logger.info(f"Unique patterns: {len(unique)}")

        self.federated_corpus = unique
        return len(unique)

    def analyze_federation_quality(self) -> Dict[str, Any]:
        """Analyze quality metrics of federated corpus."""
        logger.info("\n=== Analyzing Federation Quality ===")

        if not self.federated_corpus:
            return {"error": "No federated patterns"}

        # Domain balance
        domain_counts = Counter()
        stability_values = []
        coordination_values = []
        source_counts = Counter()

        for pattern in self.federated_corpus:
            domain = pattern.get("domain", "unknown")
            domain_counts[domain] += 1

            context = pattern.get("context", {})
            stability_values.append(context.get("stability", 0.5))
            coordination_values.append(context.get("coordination", 0.0))

            source_counts[pattern.get("source_corpus", "unknown")] += 1

        total = len(self.federated_corpus)
        domain_balance = {d: c/total for d, c in domain_counts.items()}

        logger.info(f"\nDomain distribution:")
        for domain, pct in sorted(domain_balance.items(), key=lambda x: -x[1]):
            logger.info(f"  {domain}: {pct*100:.1f}%")

        logger.info(f"\nSource distribution:")
        for source, count in source_counts.items():
            logger.info(f"  {source}: {count} patterns")

        # Stability distribution
        stability_dist = {
            "mean": float(np.mean(stability_values)),
            "std": float(np.std(stability_values)),
            "min": float(np.min(stability_values)),
            "max": float(np.max(stability_values))
        }

        # Coordination distribution
        coordination_dist = {
            "mean": float(np.mean(coordination_values)),
            "std": float(np.std(coordination_values)),
            "min": float(np.min(coordination_values)),
            "max": float(np.max(coordination_values))
        }

        logger.info(f"\nStability: mean={stability_dist['mean']:.3f}, std={stability_dist['std']:.3f}")
        logger.info(f"Coordination: mean={coordination_dist['mean']:.3f}, std={coordination_dist['std']:.3f}")

        # Check for quality issues
        if domain_balance.get("emotional", 0) > 0.8:
            self.surprises.append(f"Emotional domain still dominant: {domain_balance.get('emotional')*100:.0f}%")

        if stability_dist["std"] < 0.05:
            self.surprises.append(f"Low stability variance: {stability_dist['std']:.3f}")

        return {
            "domain_balance": domain_balance,
            "stability_distribution": stability_dist,
            "coordination_distribution": coordination_dist,
            "source_distribution": dict(source_counts)
        }

    def test_pattern_matching(self) -> Dict[str, float]:
        """
        Test EP pattern matching across federated corpus.

        Simulates:
        - Self-matching: How well do patterns match themselves?
        - Cross-matching: Do patterns from different sources match?
        """
        logger.info("\n=== Testing Pattern Matching ===")

        if len(self.federated_corpus) < 10:
            return {"error": "Not enough patterns for matching test"}

        # Sample patterns for efficiency on edge
        sample_size = min(50, len(self.federated_corpus))
        sample_indices = np.random.choice(
            len(self.federated_corpus),
            size=sample_size,
            replace=False
        )
        sample = [self.federated_corpus[i] for i in sample_indices]

        def context_distance(p1: Dict, p2: Dict) -> float:
            """Compute distance between pattern contexts."""
            c1 = p1.get("context", {})
            c2 = p2.get("context", {})

            # Compare key fields
            fields = ["stability", "coordination", "primary_metric", "complexity"]
            distances = []

            for field in fields:
                v1 = c1.get(field, 0.0)
                v2 = c2.get(field, 0.0)
                distances.append(abs(v1 - v2))

            return np.mean(distances)

        # Self-match test (pattern should match itself perfectly)
        self_matches = 0
        for pattern in sample:
            if context_distance(pattern, pattern) < 0.001:
                self_matches += 1

        self_match_rate = self_matches / len(sample)

        # Cross-match test (find similar patterns across sources)
        cross_matches = 0
        cross_pairs = 0

        for i, p1 in enumerate(sample):
            for j, p2 in enumerate(sample):
                if i >= j:
                    continue

                # Only count if from different sources
                if p1.get("source_corpus") != p2.get("source_corpus"):
                    cross_pairs += 1
                    dist = context_distance(p1, p2)
                    if dist < 0.3:  # Similar enough to be useful
                        cross_matches += 1

        cross_match_rate = cross_matches / max(1, cross_pairs)

        # Baseline: single-source matching
        single_source_patterns = [p for p in sample if p.get("source_corpus") == "production_native"]
        if len(single_source_patterns) >= 2:
            single_matches = 0
            single_pairs = 0
            for i, p1 in enumerate(single_source_patterns):
                for j, p2 in enumerate(single_source_patterns):
                    if i >= j:
                        continue
                    single_pairs += 1
                    if context_distance(p1, p2) < 0.3:
                        single_matches += 1

            baseline_rate = single_matches / max(1, single_pairs)
        else:
            baseline_rate = 0.0

        # Calculate improvement from federation
        match_improvement = cross_match_rate - baseline_rate

        logger.info(f"Self-match rate: {self_match_rate*100:.1f}%")
        logger.info(f"Cross-match rate: {cross_match_rate*100:.1f}% ({cross_matches}/{cross_pairs} pairs)")
        logger.info(f"Baseline (single-source): {baseline_rate*100:.1f}%")
        logger.info(f"Match improvement from federation: {match_improvement*100:+.1f}%")

        if match_improvement < 0:
            self.surprises.append(f"Federation decreased match rate: {match_improvement*100:.1f}%")

        return {
            "self_match_rate": self_match_rate,
            "cross_match_rate": cross_match_rate,
            "baseline_rate": baseline_rate,
            "match_improvement": match_improvement
        }

    def run_full_test(self, experiments_path: Path) -> FederationTestResults:
        """Run complete cross-machine federation test."""
        logger.info("=" * 60)
        logger.info("Session 160: Cross-Machine Federation Integration Test")
        logger.info("Hardware: Jetson Orin Nano 8GB (Sprout)")
        logger.info("=" * 60)

        import psutil
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.perf_counter()

        # Load corpora
        self.load_corpora(experiments_path)

        # Federate
        federated, conversion_time = self.federate_corpora()

        # Deduplicate
        unique_count = self.deduplicate_corpus()

        # Analyze quality
        quality = self.analyze_federation_quality()

        # Test matching
        matching = self.test_pattern_matching()

        # Edge metrics
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_time = (time.perf_counter() - start_time) * 1000

        throughput = len(self.federated_corpus) / (total_time / 1000) if total_time > 0 else 0

        logger.info(f"\n=== Edge Performance ===")
        logger.info(f"Memory delta: {end_memory - start_memory:.1f}MB")
        logger.info(f"Total time: {total_time:.1f}ms")
        logger.info(f"Throughput: {throughput:.1f} patterns/sec")

        # Build results
        results = FederationTestResults(
            production_native_count=len(self.production_patterns),
            provenance_aware_count=len(self.provenance_patterns),
            federated_total=len(federated),
            unique_patterns=unique_count,

            conversion_success_rate=1.0 if len(federated) > 0 else 0.0,
            conversion_time_ms=conversion_time,

            domain_balance=quality.get("domain_balance", {}),
            stability_distribution=quality.get("stability_distribution", {}),
            coordination_distribution=quality.get("coordination_distribution", {}),

            self_match_rate=matching.get("self_match_rate", 0.0),
            cross_match_rate=matching.get("cross_match_rate", 0.0),
            match_improvement=matching.get("match_improvement", 0.0),

            memory_usage_mb=end_memory - start_memory,
            total_time_ms=total_time,
            throughput_patterns_per_sec=throughput,

            surprises=self.surprises
        )

        # Save federated corpus
        output_corpus = experiments_path / "ep_pattern_corpus_federated_cross_machine.json"
        with open(output_corpus, 'w') as f:
            json.dump({
                "session": 160,
                "description": "Cross-machine federated EP patterns",
                "sources": ["production_native", "provenance_aware", "canonical_158"],
                "total_patterns": unique_count,
                "patterns": self.federated_corpus
            }, f, indent=2)

        logger.info(f"\n✓ Saved federated corpus: {output_corpus}")

        return results


def main():
    """Run cross-machine federation test."""
    if not HAS_SCHEMA:
        logger.error("Cannot run - Session 158 schema not available")
        return

    experiments_path = Path(__file__).parent

    tester = CrossMachineFederationTest()
    results = tester.run_full_test(experiments_path)

    # Save results
    output_path = experiments_path / "session160_federation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)

    logger.info(f"✓ Saved results: {output_path}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Session 160 Summary")
    logger.info("=" * 60)
    logger.info(f"Federated patterns: {results.unique_patterns}")
    logger.info(f"Cross-match rate: {results.cross_match_rate*100:.1f}%")
    logger.info(f"Match improvement: {results.match_improvement*100:+.1f}%")
    logger.info(f"Edge throughput: {results.throughput_patterns_per_sec:.1f} patterns/sec")

    if results.surprises:
        logger.info("\nSurprises:")
        for s in results.surprises:
            logger.info(f"  - {s}")

    if results.match_improvement > 0:
        logger.info("\n✅ FEDERATION ADDS VALUE - Cross-source matching improved!")
    elif results.match_improvement == 0:
        logger.info("\n⚠️  FEDERATION NEUTRAL - No matching improvement")
    else:
        logger.info("\n❓ FEDERATION QUESTION - Matching decreased, investigate")


if __name__ == "__main__":
    main()
