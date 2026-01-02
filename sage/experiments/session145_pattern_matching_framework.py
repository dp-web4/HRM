#!/usr/bin/env python3
"""
Session 145: EP Pattern Matching Framework
==========================================

Implements pattern matching framework for mature EP predictions, enabling
high-confidence predictions based on historical pattern similarity.

Context:
- Session 144: Collected 100 patterns (summary valid, corpus file incomplete)
- Current: EP uses heuristics for predictions (confidence 0.6-0.9)
- Goal: Enable pattern-based predictions (confidence 0.95+ when match found)

Approach:
Implement pattern matching infrastructure that can work with any pattern corpus:
1. Context similarity calculation (cosine similarity)
2. K-nearest neighbors pattern retrieval
3. Confidence boosting for close matches
4. Pattern-specific adjustment strategies

This framework will be ready when pattern corpus is regenerated with proper
JSON serialization.

Hardware: Thor (Jetson AGX Thor Developer Kit)
Date: 2025-12-31
"""

import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Import SAGE EP framework
sys.path.insert(0, str(Path(__file__).parent))
from multi_ep_coordinator import EPDomain, EPPrediction


# ============================================================================
# Pattern Matching Core
# ============================================================================

@dataclass
class EPPattern:
    """Single EP pattern for learning."""
    pattern_id: str
    domain: EPDomain
    context: Dict[str, Any]  # Domain-specific context
    prediction: Dict[str, Any]  # EPPrediction data
    outcome: Dict[str, Any]  # Actual outcome
    timestamp: str

    @property
    def context_vector(self) -> np.ndarray:
        """Convert context to vector for similarity calculation."""
        # Extract numeric values from context dict
        values = []
        for key in sorted(self.context.keys()):
            value = self.context[key]
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, bool):
                values.append(1.0 if value else 0.0)
        return np.array(values)


@dataclass
class PatternMatch:
    """Result of pattern matching."""
    pattern: EPPattern
    similarity: float  # 0.0-1.0
    distance: float  # Euclidean distance


class EPPatternMatcher:
    """
    Matches current context to historical patterns for high-confidence predictions.

    Uses k-nearest neighbors with cosine similarity to find relevant patterns.
    """

    def __init__(self, domain: EPDomain):
        self.domain = domain
        self.patterns: List[EPPattern] = []

    def add_pattern(self, pattern: EPPattern):
        """Add pattern to matcher's corpus."""
        if pattern.domain != self.domain:
            raise ValueError(f"Pattern domain {pattern.domain} doesn't match matcher {self.domain}")
        self.patterns.append(pattern)

    def find_similar_patterns(
        self,
        current_context: Dict[str, Any],
        k: int = 5,
        min_similarity: float = 0.7
    ) -> List[PatternMatch]:
        """
        Find k most similar patterns to current context.

        Args:
            current_context: Current context dict
            k: Number of nearest neighbors
            min_similarity: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of PatternMatch objects, sorted by similarity (highest first)
        """
        if not self.patterns:
            return []

        # Convert current context to vector
        current_vector = self._context_to_vector(current_context)

        # Calculate similarity to all patterns
        matches = []
        for pattern in self.patterns:
            pattern_vector = pattern.context_vector
            similarity = self._cosine_similarity(current_vector, pattern_vector)
            distance = np.linalg.norm(current_vector - pattern_vector)

            if similarity >= min_similarity:
                matches.append(PatternMatch(
                    pattern=pattern,
                    similarity=similarity,
                    distance=distance
                ))

        # Sort by similarity (highest first)
        matches.sort(key=lambda m: m.similarity, reverse=True)

        # Return top k
        return matches[:k]

    def predict_with_patterns(
        self,
        current_context: Dict[str, Any],
        fallback_prediction: EPPrediction
    ) -> Tuple[EPPrediction, Optional[List[PatternMatch]]]:
        """
        Predict using pattern matching, falling back to heuristic if no match.

        Args:
            current_context: Current context for prediction
            fallback_prediction: Heuristic-based prediction to use if no patterns match

        Returns:
            (prediction, matched_patterns)
            - If patterns found: High-confidence prediction based on patterns
            - If no patterns: Original fallback prediction
        """
        matches = self.find_similar_patterns(current_context, k=5, min_similarity=0.7)

        if not matches:
            # No pattern match - use fallback heuristic
            return fallback_prediction, None

        # Found pattern matches - boost confidence and use pattern-based prediction
        best_match = matches[0]
        avg_similarity = sum(m.similarity for m in matches) / len(matches)

        # Pattern-based prediction confidence
        # High similarity (0.9+) → 0.95 confidence
        # Medium similarity (0.7-0.9) → 0.80 confidence
        if avg_similarity >= 0.9:
            pattern_confidence = 0.95
        elif avg_similarity >= 0.8:
            pattern_confidence = 0.90
        else:
            pattern_confidence = 0.80

        # Use best matching pattern's prediction
        pattern_pred_data = best_match.pattern.prediction

        # Create high-confidence prediction based on pattern
        pattern_prediction = EPPrediction(
            domain=self.domain,
            outcome_probability=pattern_pred_data.get("outcome_probability", 0.5),
            confidence=pattern_confidence,  # Boosted from pattern match
            severity=pattern_pred_data.get("severity", 0.5),
            recommendation=pattern_pred_data.get("recommendation", "proceed"),
            reasoning=f"Pattern match (similarity={best_match.similarity:.2f}): {pattern_pred_data.get('reasoning', 'N/A')}",
            adjustment_strategy=pattern_pred_data.get("adjustment_strategy")
        )

        return pattern_prediction, matches

    def _context_to_vector(self, context: Dict[str, Any]) -> np.ndarray:
        """Convert context dict to numpy vector."""
        values = []
        for key in sorted(context.keys()):
            value = context[key]
            if isinstance(value, (int, float)):
                values.append(float(value))
            elif isinstance(value, bool):
                values.append(1.0 if value else 0.0)
        return np.array(values)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            # Pad shorter vector with zeros
            max_len = max(len(a), len(b))
            a = np.pad(a, (0, max_len - len(a)))
            b = np.pad(b, (0, max_len - len(b)))

        # Cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about pattern corpus."""
        if not self.patterns:
            return {"count": 0}

        # Decision distribution
        decisions = [p.prediction.get("recommendation", "unknown") for p in self.patterns]
        decision_counts = {}
        for d in decisions:
            decision_counts[d] = decision_counts.get(d, 0) + 1

        # Success rate
        successes = sum(1 for p in self.patterns if p.outcome.get("success", False))
        success_rate = successes / len(self.patterns)

        # Average confidence
        confidences = [p.prediction.get("confidence", 0.0) for p in self.patterns]
        avg_confidence = sum(confidences) / len(confidences)

        return {
            "count": len(self.patterns),
            "decision_distribution": decision_counts,
            "success_rate": success_rate,
            "avg_heuristic_confidence": avg_confidence,
            "pattern_matching_enabled": len(self.patterns) >= 5
        }


# ============================================================================
# Multi-Domain Pattern Matching System
# ============================================================================

class MatureEPSystem:
    """
    Mature EP system with pattern matching across all five domains.

    Each domain has its own pattern matcher. When predicting:
    1. Check for pattern match in relevant domain
    2. If match found: Use high-confidence pattern-based prediction
    3. If no match: Fall back to heuristic prediction
    4. After outcome: Add new pattern to corpus for future learning
    """

    def __init__(self):
        self.matchers = {
            EPDomain.EMOTIONAL: EPPatternMatcher(EPDomain.EMOTIONAL),
            EPDomain.QUALITY: EPPatternMatcher(EPDomain.QUALITY),
            EPDomain.ATTENTION: EPPatternMatcher(EPDomain.ATTENTION),
            EPDomain.GROUNDING: EPPatternMatcher(EPDomain.GROUNDING),
            EPDomain.AUTHORIZATION: EPPatternMatcher(EPDomain.AUTHORIZATION)
        }
        self.prediction_count = 0
        self.pattern_match_count = 0

    def load_patterns_from_corpus(self, corpus_data: Dict[str, Any]):
        """
        Load patterns from corpus JSON data.

        Expected format:
        {
            "domain": "emotional",
            "patterns": [
                {
                    "scenario_id": 1,
                    "context": {"emotional": {...}},
                    "ep_predictions": {"emotional": {...}},
                    "outcome": {...}
                },
                ...
            ]
        }
        """
        domain_str = corpus_data.get("domain", "")
        patterns_data = corpus_data.get("patterns", [])

        # Map domain string to EPDomain enum
        domain_map = {
            "emotional": EPDomain.EMOTIONAL,
            "quality": EPDomain.QUALITY,
            "attention": EPDomain.ATTENTION,
            "grounding": EPDomain.GROUNDING,
            "authorization": EPDomain.AUTHORIZATION
        }

        domain = domain_map.get(domain_str)
        if not domain:
            raise ValueError(f"Unknown domain: {domain_str}")

        matcher = self.matchers[domain]

        for i, pattern_data in enumerate(patterns_data):
            # Extract domain-specific context
            domain_context = pattern_data.get("context", {}).get(domain_str, {})

            # Skip patterns without context for this domain (e.g., projected patterns)
            if not domain_context:
                continue

            pattern = EPPattern(
                pattern_id=f"{domain_str}_{i}",
                domain=domain,
                context=domain_context,
                prediction=pattern_data.get("ep_predictions", {}).get(domain_str, {}),
                outcome=pattern_data.get("outcome", {}),
                timestamp=pattern_data.get("timestamp", "")
            )
            matcher.add_pattern(pattern)

    def predict_with_pattern_matching(
        self,
        domain: EPDomain,
        current_context: Dict[str, Any],
        fallback_prediction: EPPrediction
    ) -> Tuple[EPPrediction, bool, Optional[List[PatternMatch]]]:
        """
        Predict using pattern matching if available, fallback to heuristic otherwise.

        Returns:
            (prediction, pattern_used, matches)
        """
        self.prediction_count += 1

        matcher = self.matchers[domain]
        prediction, matches = matcher.predict_with_patterns(current_context, fallback_prediction)

        pattern_used = matches is not None
        if pattern_used:
            self.pattern_match_count += 1

        return prediction, pattern_used, matches

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get statistics for entire mature EP system."""
        domain_stats = {}
        for domain, matcher in self.matchers.items():
            domain_stats[domain.value] = matcher.get_pattern_statistics()

        total_patterns = sum(stats["count"] for stats in domain_stats.values())
        pattern_match_rate = self.pattern_match_count / self.prediction_count if self.prediction_count > 0 else 0.0

        return {
            "total_patterns": total_patterns,
            "patterns_by_domain": {d: stats["count"] for d, stats in domain_stats.items()},
            "domain_statistics": domain_stats,
            "predictions_made": self.prediction_count,
            "pattern_matches": self.pattern_match_count,
            "pattern_match_rate": pattern_match_rate,
            "maturation_status": self._assess_maturation_status(domain_stats)
        }

    def _assess_maturation_status(self, domain_stats: Dict[str, Any]) -> str:
        """Assess overall EP maturation level."""
        mature_domains = sum(1 for stats in domain_stats.values() if stats.get("pattern_matching_enabled", False))

        if mature_domains >= 4:
            return "Mature (4+ domains with pattern matching)"
        elif mature_domains >= 2:
            return "Learning+ (2-3 domains with pattern matching)"
        elif mature_domains >= 1:
            return "Learning (1 domain with pattern matching)"
        else:
            return "Immature (no pattern matching enabled)"


# ============================================================================
# Demo: Pattern Matching Framework
# ============================================================================

def demo_pattern_matching():
    """Demonstrate pattern matching framework."""
    print("=" * 80)
    print("Session 145: EP Pattern Matching Framework")
    print("=" * 80)
    print()
    print("Demonstrates pattern matching infrastructure for mature EP predictions")
    print()

    # Create mature EP system
    system = MatureEPSystem()

    # Example: Create a few synthetic patterns for emotional domain
    print("Creating example patterns for Emotional EP domain...")
    emotional_matcher = system.matchers[EPDomain.EMOTIONAL]

    # Pattern 1: High frustration scenario
    pattern1 = EPPattern(
        pattern_id="emotional_1",
        domain=EPDomain.EMOTIONAL,
        context={
            "current_frustration": 0.8,
            "recent_failure_rate": 0.7,
            "atp_stress": 0.6,
            "interaction_complexity": 0.5
        },
        prediction={
            "outcome_probability": 0.3,
            "confidence": 0.8,
            "severity": 0.7,
            "recommendation": "defer",
            "reasoning": "High cascade risk due to frustration and failures"
        },
        outcome={"success": False, "outcome_type": "deferred"},
        timestamp="2025-12-31T10:00:00"
    )
    emotional_matcher.add_pattern(pattern1)

    # Pattern 2: Moderate stress scenario
    pattern2 = EPPattern(
        pattern_id="emotional_2",
        domain=EPDomain.EMOTIONAL,
        context={
            "current_frustration": 0.5,
            "recent_failure_rate": 0.3,
            "atp_stress": 0.4,
            "interaction_complexity": 0.4
        },
        prediction={
            "outcome_probability": 0.6,
            "confidence": 0.8,
            "severity": 0.4,
            "recommendation": "adjust",
            "reasoning": "Moderate stress - reduce complexity"
        },
        outcome={"success": True, "outcome_type": "adjusted_and_proceeded"},
        timestamp="2025-12-31T11:00:00"
    )
    emotional_matcher.add_pattern(pattern2)

    # Pattern 3: Low stress scenario
    pattern3 = EPPattern(
        pattern_id="emotional_3",
        domain=EPDomain.EMOTIONAL,
        context={
            "current_frustration": 0.2,
            "recent_failure_rate": 0.1,
            "atp_stress": 0.2,
            "interaction_complexity": 0.3
        },
        prediction={
            "outcome_probability": 0.9,
            "confidence": 0.8,
            "severity": 0.1,
            "recommendation": "proceed",
            "reasoning": "Low emotional risk"
        },
        outcome={"success": True, "outcome_type": "proceeded"},
        timestamp="2025-12-31T12:00:00"
    )
    emotional_matcher.add_pattern(pattern3)

    print(f"Added {len(emotional_matcher.patterns)} patterns to Emotional EP matcher")
    print()

    # Test pattern matching with similar context
    print("Testing pattern matching with new context similar to Pattern 2...")
    test_context = {
        "current_frustration": 0.52,  # Close to 0.5
        "recent_failure_rate": 0.28,  # Close to 0.3
        "atp_stress": 0.42,  # Close to 0.4
        "interaction_complexity": 0.38  # Close to 0.4
    }

    # Fallback heuristic prediction
    fallback = EPPrediction(
        domain=EPDomain.EMOTIONAL,
        outcome_probability=0.5,
        confidence=0.7,
        severity=0.5,
        recommendation="unknown",
        reasoning="Heuristic fallback"
    )

    prediction, pattern_used, matches = system.predict_with_pattern_matching(
        EPDomain.EMOTIONAL,
        test_context,
        fallback
    )

    print(f"Pattern matching result:")
    print(f"  Pattern used: {pattern_used}")
    if pattern_used and matches:
        print(f"  Best match: {matches[0].pattern.pattern_id} (similarity={matches[0].similarity:.3f})")
        print(f"  Confidence: {prediction.confidence:.2f} (boosted from heuristic 0.70)")
        print(f"  Recommendation: {prediction.recommendation}")
        print(f"  Reasoning: {prediction.reasoning}")
    else:
        print(f"  No pattern match - using fallback heuristic")

    print()

    # System statistics
    stats = system.get_system_statistics()
    print("System Statistics:")
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Patterns by domain: {stats['patterns_by_domain']}")
    print(f"  Predictions made: {stats['predictions_made']}")
    print(f"  Pattern matches: {stats['pattern_matches']}")
    print(f"  Pattern match rate: {stats['pattern_match_rate']:.1%}")
    print(f"  Maturation status: {stats['maturation_status']}")
    print()

    print("=" * 80)
    print("Pattern Matching Framework Complete")
    print("=" * 80)
    print()
    print("Framework ready to use with full pattern corpus when available.")
    print("Next step: Regenerate Session 144 corpus with proper JSON serialization")

    return system, stats


if __name__ == "__main__":
    system, stats = demo_pattern_matching()
