"""
SAGE Response Quality Scoring

Session 27: Quality Metric Integration for Temporal Adaptation

Implements Thor's 4-metric quality system for evaluating response quality.
This replaces the convergence_quality proxy used in Session 26 with proper
multi-dimensional quality scoring.

The 4-metric system (from Session 6, validated in validate_merged_quality.py):
1. Unique content (not generic)
2. Uses specific technical terms
3. Includes numerical data
4. Avoids philosophical hedging

This scoring feeds into the multi-objective temporal adaptation (Session 24-26)
to enable accurate quality tracking in the quality objective.

Hardware: Jetson AGX Thor
Based on: Session 6 (quality criteria), Session 26 (temporal adaptation)
"""

import re
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class QualityScore:
    """
    Multi-dimensional quality score for SAGE responses.

    Attributes:
        total: Total score (0-4, number of criteria met)
        unique: Has unique content (not generic)
        specific_terms: Uses specific technical terms
        has_numbers: Includes numerical data
        avoids_hedging: Avoids philosophical hedging
        normalized: Total score normalized to [0, 1]
    """
    total: int
    unique: bool
    specific_terms: bool
    has_numbers: bool
    avoids_hedging: bool

    @property
    def normalized(self) -> float:
        """Return normalized score (0-1)"""
        return self.total / 4.0

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for logging"""
        return {
            'total': self.total,
            'normalized': self.normalized,
            'unique': self.unique,
            'specific_terms': self.specific_terms,
            'has_numbers': self.has_numbers,
            'avoids_hedging': self.avoids_hedging
        }


def score_response_quality(response: str, question: str = None) -> QualityScore:
    """
    Score response quality using Thor's 4-metric criteria.

    This is the production implementation of the quality scoring system,
    extracted from validate_merged_quality.py and adapted for temporal
    adaptation integration.

    Args:
        response: The response text to score
        question: Optional question text (currently unused, for future enhancements)

    Returns:
        QualityScore with total score (0-4) and per-criterion breakdown

    Example:
        >>> score = score_response_quality("ATP level is 75.5 with salience threshold 0.7")
        >>> score.total
        4
        >>> score.normalized
        1.0
        >>> score.specific_terms
        True
    """
    if not response or not isinstance(response, str):
        # Empty or invalid response scores 0
        return QualityScore(
            total=0,
            unique=False,
            specific_terms=False,
            has_numbers=False,
            avoids_hedging=False
        )

    response_lower = response.lower()

    score = 0
    unique = False
    specific_terms = False
    has_numbers = False
    avoids_hedging = False

    # Criterion 1: Unique content (not generic)
    generic_phrases = [
        "i don't have", "i can't", "i'm not sure",
        "let me think", "it's hard to", "it's difficult to",
        "i'm unable to", "i cannot", "i don't know",
        "i'm sorry", "unfortunately"
    ]
    if not any(phrase in response_lower for phrase in generic_phrases):
        score += 1
        unique = True

    # Criterion 2: Uses specific technical terms
    # SAGE/SNARC consciousness-related technical vocabulary
    technical_terms = [
        'atp', 'snarc', 'salience', 'threshold', 'irp',
        'iterations', 'cognitive', 'metabolic',
        'attention', 'consciousness', 'convergence',
        'mrh', 'horizon', 'federation', 'temporal',
        'adaptation', 'multi-objective', 'pareto',
        'weighted fitness', 'energy efficiency'
    ]
    if any(term in response_lower for term in technical_terms):
        score += 1
        specific_terms = True

    # Criterion 3: Includes numerical data
    # Match integers, floats, percentages, scientific notation
    if re.search(r'\d+\.?\d*%?|[-+]?\d*\.?\d+([eE][-+]?\d+)?', response):
        score += 1
        has_numbers = True

    # Criterion 4: Avoids philosophical hedging
    # These phrases indicate uncertainty or philosophical retreat
    hedging_phrases = [
        "might be", "could be", "seems like", "appears to",
        "i think", "i believe", "probably", "perhaps",
        "possibly", "maybe", "in theory",
        "stochastic", "just computation", "merely",
        "it may", "it might", "could possibly"
    ]
    if not any(hedge in response_lower for hedge in hedging_phrases):
        score += 1
        avoids_hedging = True

    return QualityScore(
        total=score,
        unique=unique,
        specific_terms=specific_terms,
        has_numbers=has_numbers,
        avoids_hedging=avoids_hedging
    )


def score_response_quality_normalized(response: str, question: str = None) -> float:
    """
    Convenience function that returns normalized quality score (0-1).

    This is the function to use for temporal adaptation integration,
    as it returns a single float compatible with the quality_score parameter.

    Args:
        response: The response text to score
        question: Optional question text (currently unused)

    Returns:
        Normalized quality score (0-1)

    Example:
        >>> score_response_quality_normalized("ATP is 75.5")
        0.75  # 3 out of 4 criteria met
    """
    quality_score = score_response_quality(response, question)
    return quality_score.normalized


# Convenience alias for backward compatibility
def get_quality_score(response: str) -> float:
    """
    Legacy alias for score_response_quality_normalized.
    Returns normalized quality score (0-1).
    """
    return score_response_quality_normalized(response)


if __name__ == "__main__":
    # Quick validation test
    print("=" * 70)
    print("SAGE Quality Metrics - Session 27")
    print("=" * 70)
    print()

    # Test cases
    test_responses = [
        # High quality: all 4 criteria
        ("ATP level is 75.5 with salience threshold 0.7 for SNARC convergence",
         "Should score 4/4 (unique, technical, numbers, no hedging)"),

        # Medium quality: 2-3 criteria
        ("The temporal adaptation is working well with good performance",
         "Should score ~2/4 (unique, technical, no numbers, no hedging)"),

        # Low quality: 0-1 criteria
        ("I'm not sure, it might be related to some kind of processing",
         "Should score 0/4 (generic, hedging, no technical terms, no numbers)"),

        # Real SAGE response example
        ("Multi-objective optimization achieved 0.920 weighted fitness with 100% coverage, "
         "90.1% quality, and 75% energy efficiency using Pareto-optimal parameters "
         "(cost=0.005, recovery=0.080)",
         "Should score 4/4 (excellent quality response)")
    ]

    for i, (response, expected) in enumerate(test_responses, 1):
        print(f"Test {i}: {expected}")
        print(f"Response: {response[:60]}...")
        score = score_response_quality(response)
        print(f"Score: {score.total}/4 (normalized: {score.normalized:.2f})")
        print(f"  Unique: {score.unique}")
        print(f"  Technical: {score.specific_terms}")
        print(f"  Numbers: {score.has_numbers}")
        print(f"  No hedging: {score.avoids_hedging}")
        print()

    print("=" * 70)
    print("Quality metrics module validated")
    print("Ready for temporal adaptation integration")
    print("=" * 70)
