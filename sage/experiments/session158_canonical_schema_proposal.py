#!/usr/bin/env python3
"""
Session 158: Canonical EP Pattern Schema for Cross-Machine Federation

PROBLEM IDENTIFIED (Session 157):
- Thor's integrated federation checks for flat fields: stability, coordination, domain
- SAGE patterns use nested structure: context.emotional.frustration, etc.
- Web4 patterns may have different field names entirely
- Schema mismatch prevents cross-machine pattern federation

SOLUTION:
Define a canonical schema that:
1. Maps SAGE nested fields to federation fields
2. Computes derived fields (stability, coordination) from existing data
3. Provides defaults for truly missing fields
4. Works on edge hardware (Sprout)

Hardware: Jetson Orin Nano 8GB (Sprout)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class PatternDomain(Enum):
    """Canonical domain types for EP patterns."""
    EMOTIONAL = "emotional"
    QUALITY = "quality"
    ATTENTION = "attention"
    GROUNDING = "grounding"
    AUTHORIZATION = "authorization"
    UNKNOWN = "unknown"


class ProvenanceType(Enum):
    """Pattern provenance types (from Session 155)."""
    DECISION = "decision"
    OBSERVATION = "observation"
    UNKNOWN = "unknown"


@dataclass
class CanonicalContext:
    """
    Canonical context fields for cross-machine federation.

    These are the fields that Thor's Session 157 checks for.
    SAGE patterns can be converted to this format.
    """
    # Core domain metrics (always present)
    primary_metric: float = 0.0      # Domain's primary signal (frustration, quality, ATP, etc.)
    recent_trend: float = 0.0        # Recent change direction (-1 to 1)
    complexity: float = 0.5          # Task/scenario complexity (0 to 1)

    # Derived stability field (computed from variance/consistency)
    stability: float = 0.5           # How stable is the domain's state (0=chaotic, 1=stable)

    # Multi-domain coordination field
    coordination: float = 0.0        # Coordination with other domains (-1 to 1)

    # Domain-specific fields (may be null for non-matching domains)
    domain_specific: Dict[str, float] = field(default_factory=dict)


@dataclass
class CanonicalPattern:
    """
    Canonical EP pattern for cross-machine federation.

    This schema works across Thor, Legion, and Sprout.
    """
    # Identity
    pattern_id: str
    domain: PatternDomain

    # Canonical context (unified fields)
    context: CanonicalContext

    # Prediction/outcome
    prediction: float
    outcome: float
    was_correct: bool

    # Provenance (from Session 155)
    provenance: ProvenanceType = ProvenanceType.UNKNOWN
    quality_weight: float = 0.5

    # Source tracking
    source_machine: str = "unknown"
    source_session: str = "unknown"
    original_id: Optional[str] = None

    # Timestamp
    timestamp: str = ""


class SAGEToCanonicalConverter:
    """
    Converts SAGE EP patterns to canonical schema.

    SAGE Pattern Structure:
    - context: {emotional: {...}, quality: {...}, attention: {...}, ...}
    - ep_predictions: {emotional: {prediction: float}, ...}
    - outcome: {success: bool, ...}
    - coordinated_decision: {domain: str, ...}

    Canonical Structure:
    - domain: single domain enum
    - context: flat fields with derived stability/coordination
    - prediction/outcome: floats
    - provenance: type and quality weight
    """

    # Field mappings from SAGE nested structure to canonical flat fields
    DOMAIN_PRIMARY_METRICS = {
        "emotional": "frustration",
        "quality": "relationship_quality",
        "attention": "atp_level",
        "grounding": "coherence_score",
        "authorization": "trust_score"
    }

    DOMAIN_TREND_FIELDS = {
        "emotional": "recent_failure_rate",
        "quality": "recent_quality_avg",
        "attention": "estimated_cost",
        "grounding": "trust_differential",
        "authorization": "abuse_history"
    }

    def __init__(self):
        self.conversion_count = 0
        self.error_count = 0
        self.field_stats: Dict[str, int] = {}

    def convert_pattern(self, sage_pattern: Dict[str, Any]) -> CanonicalPattern:
        """Convert a single SAGE pattern to canonical format."""
        try:
            # Determine target domain
            domain_str = self._extract_domain(sage_pattern)
            domain = PatternDomain(domain_str) if domain_str in [d.value for d in PatternDomain] else PatternDomain.UNKNOWN

            # Extract context
            context = self._convert_context(sage_pattern.get("context", {}), domain_str)

            # Extract prediction and outcome
            prediction = self._extract_prediction(sage_pattern, domain_str)
            outcome = self._extract_outcome(sage_pattern)
            was_correct = sage_pattern.get("outcome", {}).get("success", False)

            # Extract provenance (if present from Session 155)
            provenance, quality_weight = self._extract_provenance(sage_pattern)

            canonical = CanonicalPattern(
                pattern_id=sage_pattern.get("pattern_id", f"sage_{self.conversion_count}"),
                domain=domain,
                context=context,
                prediction=prediction,
                outcome=outcome,
                was_correct=was_correct,
                provenance=provenance,
                quality_weight=quality_weight,
                source_machine="sprout",  # We're running on Sprout
                source_session="sage_production",
                original_id=sage_pattern.get("pattern_id"),
                timestamp=sage_pattern.get("timestamp", "")
            )

            self.conversion_count += 1
            return canonical

        except Exception as e:
            self.error_count += 1
            logger.debug(f"Conversion error: {e}")
            # Return a default pattern on error
            return CanonicalPattern(
                pattern_id=f"error_{self.error_count}",
                domain=PatternDomain.UNKNOWN,
                context=CanonicalContext(),
                prediction=0.5,
                outcome=0.5,
                was_correct=False
            )

    def _extract_domain(self, pattern: Dict) -> str:
        """Extract the target domain from a SAGE pattern."""
        # Check target_domain first (from provenance-aware patterns)
        if "target_domain" in pattern:
            return pattern["target_domain"].lower()

        # Check coordinated_decision
        decision = pattern.get("coordinated_decision", {})
        if "domain" in decision:
            return decision["domain"].lower()

        # Check for highest prediction
        predictions = pattern.get("ep_predictions", {})
        if predictions:
            max_domain = max(predictions.keys(),
                           key=lambda d: predictions[d].get("prediction", 0))
            return max_domain.lower()

        return "unknown"

    def _convert_context(self, sage_context: Dict, domain: str) -> CanonicalContext:
        """Convert SAGE nested context to canonical flat context."""
        domain_context = sage_context.get(domain, {})

        # Get primary metric for this domain
        primary_field = self.DOMAIN_PRIMARY_METRICS.get(domain, "")
        primary_metric = domain_context.get(primary_field, 0.0)
        self._track_field(f"{domain}.{primary_field}")

        # Get trend field
        trend_field = self.DOMAIN_TREND_FIELDS.get(domain, "")
        recent_trend = domain_context.get(trend_field, 0.0)
        # Normalize trend to -1 to 1 range
        if domain == "attention":
            # For ATP, higher estimated cost = negative trend
            recent_trend = -min(1.0, recent_trend / 100.0)
        elif domain == "authorization":
            # For authorization, higher abuse = negative trend
            recent_trend = -recent_trend

        # Get complexity (from emotional domain or default)
        complexity = sage_context.get("emotional", {}).get("complexity", 0.5)

        # Compute derived stability
        stability = self._compute_stability(sage_context, domain)

        # Compute coordination score
        coordination = self._compute_coordination(sage_context, domain)

        # Preserve domain-specific fields
        domain_specific = {}
        for key, value in domain_context.items():
            if isinstance(value, (int, float)):
                domain_specific[key] = float(value)

        return CanonicalContext(
            primary_metric=float(primary_metric),
            recent_trend=float(recent_trend),
            complexity=float(complexity),
            stability=float(stability),
            coordination=float(coordination),
            domain_specific=domain_specific
        )

    def _compute_stability(self, context: Dict, domain: str) -> float:
        """
        Compute stability score from context.

        Stability is high when:
        - Frustration is low (emotional)
        - Quality metrics are consistent
        - ATP reserves are adequate
        - Trust levels are stable

        Returns 0 to 1 (1 = very stable)
        """
        scores = []

        # Emotional stability: inverse of frustration
        emotional = context.get("emotional", {})
        frustration = emotional.get("frustration", 0.0)
        scores.append(1.0 - min(1.0, frustration))

        # Attention stability: ATP above threshold
        attention = context.get("attention", {})
        atp = attention.get("atp_level", 100.0)
        threshold = attention.get("reserve_threshold", 30.0)
        atp_margin = (atp - threshold) / max(1.0, atp)
        scores.append(min(1.0, max(0.0, atp_margin)))

        # Authorization stability: trust score with low abuse
        auth = context.get("authorization", {})
        trust = auth.get("trust_score", 0.5)
        abuse = auth.get("abuse_history", 0.0)
        scores.append(trust * (1.0 - abuse))

        # Grounding stability: coherence score
        grounding = context.get("grounding", {})
        coherence = grounding.get("coherence_score", 0.5)
        scores.append(coherence)

        return float(np.mean(scores)) if scores else 0.5

    def _compute_coordination(self, context: Dict, primary_domain: str) -> float:
        """
        Compute multi-domain coordination score.

        High coordination when domains are aligned:
        - Emotional frustration low AND quality high
        - ATP adequate AND authorization trust high

        Returns -1 to 1 (1 = perfectly coordinated, -1 = conflicting)
        """
        # Get key metrics from each domain
        emotional = context.get("emotional", {})
        quality = context.get("quality", {})
        attention = context.get("attention", {})
        auth = context.get("authorization", {})

        frustration = emotional.get("frustration", 0.0)
        quality_score = quality.get("relationship_quality", 0.5)
        atp = attention.get("atp_level", 100.0)
        trust = auth.get("trust_score", 0.5)

        # Coordination signals
        signals = []

        # Low frustration + high quality = positive coordination
        emotional_quality = (1.0 - frustration) * quality_score
        signals.append(emotional_quality * 2 - 1)  # Map to -1 to 1

        # Adequate ATP + high trust = positive coordination
        atp_norm = min(1.0, atp / 150.0)  # Normalize ATP
        attention_auth = atp_norm * trust
        signals.append(attention_auth * 2 - 1)

        return float(np.mean(signals)) if signals else 0.0

    def _extract_prediction(self, pattern: Dict, domain: str) -> float:
        """Extract prediction value from SAGE pattern."""
        predictions = pattern.get("ep_predictions", {})
        domain_pred = predictions.get(domain, {})
        return float(domain_pred.get("prediction", 0.5))

    def _extract_outcome(self, pattern: Dict) -> float:
        """Extract outcome value from SAGE pattern."""
        outcome = pattern.get("outcome", {})
        # Convert success to float
        if outcome.get("success", False):
            return 1.0
        # Check if there's a numeric outcome
        return float(outcome.get("value", 0.0))

    def _extract_provenance(self, pattern: Dict) -> tuple:
        """Extract provenance type and quality weight."""
        prov = pattern.get("provenance", {})
        if not prov:
            # No provenance, default to DECISION (SAGE credit assignment)
            return ProvenanceType.DECISION, 0.85

        prov_type_str = prov.get("type", "unknown").upper()
        try:
            prov_type = ProvenanceType[prov_type_str]
        except KeyError:
            prov_type = ProvenanceType.UNKNOWN

        quality_weight = prov.get("quality_weight", 0.5)
        return prov_type, float(quality_weight)

    def _track_field(self, field: str):
        """Track which fields are being used."""
        self.field_stats[field] = self.field_stats.get(field, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        """Get conversion statistics."""
        return {
            "converted": self.conversion_count,
            "errors": self.error_count,
            "error_rate": self.error_count / max(1, self.conversion_count + self.error_count),
            "field_usage": self.field_stats
        }


def canonical_to_dict(pattern: CanonicalPattern) -> Dict[str, Any]:
    """Convert canonical pattern to dictionary for JSON serialization."""
    d = asdict(pattern)
    d["domain"] = pattern.domain.value
    d["provenance"] = pattern.provenance.value
    return d


def test_conversion():
    """Test canonical schema conversion on edge."""
    logger.info("=" * 60)
    logger.info("Session 158: Canonical Schema Proposal Testing")
    logger.info("Hardware: Jetson Orin Nano 8GB (Sprout)")
    logger.info("=" * 60)

    # Load SAGE production patterns
    corpus_path = Path(__file__).parent / "ep_pattern_corpus_production_native.json"

    if not corpus_path.exists():
        logger.error(f"Corpus not found: {corpus_path}")
        return None

    with open(corpus_path) as f:
        sage_data = json.load(f)

    patterns = sage_data.get("patterns", [])
    logger.info(f"\nLoaded {len(patterns)} SAGE patterns")

    # Convert to canonical format
    converter = SAGEToCanonicalConverter()
    canonical_patterns = []

    for pattern in patterns:
        canonical = converter.convert_pattern(pattern)
        canonical_patterns.append(canonical)

    stats = converter.get_stats()
    logger.info(f"\n=== Conversion Results ===")
    logger.info(f"Converted: {stats['converted']}")
    logger.info(f"Errors: {stats['errors']} ({stats['error_rate']:.1%})")

    # Analyze canonical patterns
    logger.info(f"\n=== Canonical Field Analysis ===")

    # Check stability distribution
    stability_values = [p.context.stability for p in canonical_patterns]
    logger.info(f"Stability - mean: {np.mean(stability_values):.3f}, std: {np.std(stability_values):.3f}")

    # Check coordination distribution
    coordination_values = [p.context.coordination for p in canonical_patterns]
    logger.info(f"Coordination - mean: {np.mean(coordination_values):.3f}, std: {np.std(coordination_values):.3f}")

    # Check domain distribution
    domain_counts = {}
    for p in canonical_patterns:
        d = p.domain.value
        domain_counts[d] = domain_counts.get(d, 0) + 1

    logger.info(f"\n=== Domain Distribution ===")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = count / len(canonical_patterns) * 100
        logger.info(f"  {domain}: {count} ({pct:.1f}%)")

    # Sample canonical pattern
    logger.info(f"\n=== Sample Canonical Pattern ===")
    sample = canonical_patterns[0] if canonical_patterns else None
    if sample:
        logger.info(f"  pattern_id: {sample.pattern_id}")
        logger.info(f"  domain: {sample.domain.value}")
        logger.info(f"  context.stability: {sample.context.stability:.3f}")
        logger.info(f"  context.coordination: {sample.context.coordination:.3f}")
        logger.info(f"  context.primary_metric: {sample.context.primary_metric:.3f}")
        logger.info(f"  provenance: {sample.provenance.value}")
        logger.info(f"  quality_weight: {sample.quality_weight:.3f}")

    # Save canonical corpus
    output_path = Path(__file__).parent / "ep_pattern_corpus_canonical.json"

    canonical_corpus = {
        "session": 158,
        "description": "Canonical EP patterns for cross-machine federation",
        "schema_version": "1.0",
        "source": "sage_production_native",
        "conversion_stats": stats,
        "patterns": [canonical_to_dict(p) for p in canonical_patterns]
    }

    with open(output_path, 'w') as f:
        json.dump(canonical_corpus, f, indent=2)

    logger.info(f"\n✓ Saved canonical corpus: {output_path}")

    # Validation results
    results = {
        "session": 158,
        "schema_version": "1.0",
        "conversion_stats": stats,
        "field_analysis": {
            "stability": {
                "mean": float(np.mean(stability_values)),
                "std": float(np.std(stability_values)),
                "min": float(np.min(stability_values)),
                "max": float(np.max(stability_values))
            },
            "coordination": {
                "mean": float(np.mean(coordination_values)),
                "std": float(np.std(coordination_values)),
                "min": float(np.min(coordination_values)),
                "max": float(np.max(coordination_values))
            }
        },
        "domain_distribution": domain_counts,
        "schema_proposal": {
            "canonical_fields": [
                "primary_metric (float)",
                "recent_trend (float)",
                "complexity (float)",
                "stability (float, computed)",
                "coordination (float, computed)",
                "domain_specific (dict)"
            ],
            "computed_fields_rationale": {
                "stability": "Derived from frustration, ATP margin, trust, coherence",
                "coordination": "Derived from cross-domain alignment signals"
            },
            "backward_compatible": True,
            "new_fields_computed": True
        },
        "recommendation": "Adopt canonical schema for Session 159+ cross-machine federation"
    }

    # Save results
    results_path = Path(__file__).parent / "session158_canonical_schema_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Saved results: {results_path}")

    return results


if __name__ == "__main__":
    test_conversion()
