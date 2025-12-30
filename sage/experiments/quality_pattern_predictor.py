#!/usr/bin/env python3
"""
Quality Pattern Predictor - Testing Quality EP Prediction Concept

Implements basic quality prediction using collected patterns.
This demonstrates the core EP concept: predict quality BEFORE generating response.

Phase 2 of Quality Epistemic Proprioception:
- Load quality patterns
- Match new queries to existing patterns
- Predict quality based on pattern history
- Test prediction accuracy

Date: 2025-12-30
Hardware: Thor (Jetson AGX Thor Developer Kit)
Foundation: QUALITY_EPISTEMIC_PROPRIOCEPTION.md
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

# Add SAGE modules to path
sys.path.append(str(Path(__file__).parent.parent))

from quality_pattern_collector import (
    QualityPatternCollector,
    QueryContext,
    ResponseApproach,
    QualityPattern,
)


@dataclass
class QualityPrediction:
    """Prediction of response quality before generation."""

    predicted_quality: float  # 0.0-1.0 predicted quality score
    confidence: float  # 0.0-1.0 confidence in prediction
    matching_patterns: int  # Number of similar patterns found
    recommendation: str  # "generate", "adjust", or "unknown"

    def should_adjust(self, threshold: float = 0.70) -> bool:
        """Should we adjust approach due to low predicted quality?"""
        return (
            self.predicted_quality < threshold and
            self.confidence > 0.5
        )


class QualityPredictor:
    """
    Predicts response quality using learned patterns.

    This implements the EP prediction capability:
    Given query context and intended approach, predict quality BEFORE generation.
    """

    def __init__(self, collector: QualityPatternCollector):
        """Initialize predictor with pattern collection."""
        self.collector = collector

    def predict_quality(
        self,
        query: str,
        intended_style: Optional[str] = None
    ) -> QualityPrediction:
        """
        Predict quality for a query with intended approach.

        Args:
            query: The query to respond to
            intended_style: Intended response style ("specific", "hedging", "generic")

        Returns:
            QualityPrediction with predicted score and confidence
        """
        # Analyze query context
        query_context = QueryContext.analyze_query(query)

        # Find matching patterns
        matching_patterns = self._find_matching_patterns(
            query_context,
            intended_style
        )

        if not matching_patterns:
            # No patterns found - low confidence prediction
            return QualityPrediction(
                predicted_quality=0.5,
                confidence=0.0,
                matching_patterns=0,
                recommendation="unknown"
            )

        # Calculate weighted prediction
        predicted_quality, confidence = self._calculate_prediction(
            matching_patterns
        )

        # Generate recommendation
        if predicted_quality < 0.70 and confidence > 0.5:
            recommendation = "adjust"
        elif confidence > 0.3:
            recommendation = "generate"
        else:
            recommendation = "unknown"

        return QualityPrediction(
            predicted_quality=predicted_quality,
            confidence=confidence,
            matching_patterns=len(matching_patterns),
            recommendation=recommendation
        )

    def _find_matching_patterns(
        self,
        query_context: QueryContext,
        intended_style: Optional[str] = None
    ) -> List[QualityPattern]:
        """Find patterns similar to query context and intended approach."""
        matching = []

        for pattern in self.collector.patterns:
            similarity = self._calculate_similarity(
                query_context,
                pattern.query_context,
                intended_style,
                pattern.response_approach.response_style
            )

            if similarity > 0.7:  # Threshold for matching
                matching.append(pattern)

        return matching

    def _calculate_similarity(
        self,
        query_context: QueryContext,
        pattern_context: QueryContext,
        intended_style: Optional[str],
        pattern_style: str
    ) -> float:
        """
        Calculate similarity between query and pattern.

        Returns score 0.0-1.0 indicating how similar contexts are.
        """
        similarity = 0.0
        weights_sum = 0.0

        # Query type match (high weight)
        if query_context.query_type == pattern_context.query_type:
            similarity += 0.5
        weights_sum += 0.5

        # Expects numbers match (medium weight)
        if query_context.expects_numbers == pattern_context.expects_numbers:
            similarity += 0.2
        weights_sum += 0.2

        # Expects specifics match (medium weight)
        if query_context.expects_specifics == pattern_context.expects_specifics:
            similarity += 0.2
        weights_sum += 0.2

        # Response style match (if provided, high weight)
        if intended_style is not None:
            if intended_style == pattern_style:
                similarity += 0.3
            weights_sum += 0.3

        # Normalize
        if weights_sum > 0:
            similarity = similarity / weights_sum

        return similarity

    def _calculate_prediction(
        self,
        matching_patterns: List[QualityPattern]
    ) -> Tuple[float, float]:
        """
        Calculate predicted quality from matching patterns.

        Returns:
            (predicted_quality, confidence)
        """
        # Extract qualities
        qualities = [
            p.quality_metrics.overall_score
            for p in matching_patterns
        ]

        # Predicted quality = average of matches
        predicted_quality = sum(qualities) / len(qualities)

        # Confidence = function of number of matches
        # More matches = higher confidence
        confidence = min(1.0, len(matching_patterns) / 5.0)

        return predicted_quality, confidence


def test_prediction():
    """
    Test quality prediction concept.

    Demonstrates EP capability: predicting quality before generation.
    """
    print("=" * 80)
    print("Quality Prediction Testing - Quality Epistemic Proprioception")
    print("=" * 80)
    print()

    # Load existing patterns
    collector = QualityPatternCollector()
    predictor = QualityPredictor(collector)

    print(f"Loaded {len(collector.patterns)} quality patterns")
    print()

    # Test queries (not in pattern set)
    test_queries = [
        {
            "query": "What is the current frustration level?",
            "intended_style": "specific",
            "description": "Status query with specific style"
        },
        {
            "query": "What is the current frustration level?",
            "intended_style": "hedging",
            "description": "Status query with hedging style (should predict LOW)"
        },
        {
            "query": "How does SNARC memory work?",
            "intended_style": "specific",
            "description": "Technical query with specific style"
        },
        {
            "query": "How does SNARC memory work?",
            "intended_style": "generic",
            "description": "Technical query with generic style"
        },
        {
            "query": "Why is consciousness important?",
            "intended_style": "specific",
            "description": "Conceptual query with specific style"
        },
    ]

    print("=" * 80)
    print("Testing Quality Prediction (EP in action!)")
    print("=" * 80)
    print()

    for i, test in enumerate(test_queries, 1):
        print(f"Test {i}: {test['description']}")
        print(f"Query: \"{test['query']}\"")
        print(f"Intended style: {test['intended_style']}")
        print()

        # PREDICT quality BEFORE generating
        prediction = predictor.predict_quality(
            test['query'],
            intended_style=test['intended_style']
        )

        print(f"EP Prediction:")
        print(f"  Predicted quality: {prediction.predicted_quality:.2f}")
        print(f"  Confidence: {prediction.confidence:.2f}")
        print(f"  Matching patterns: {prediction.matching_patterns}")
        print(f"  Recommendation: {prediction.recommendation}")

        if prediction.should_adjust():
            print(f"  ⚠️  EP WARNING: Low quality predicted - should ADJUST approach!")
        elif prediction.confidence > 0.5:
            print(f"  ✅ EP CONFIDENCE: Quality predicted good - proceed with generation")
        else:
            print(f"  ❓ EP UNCERTAIN: Low confidence - need more patterns")

        print()
        print("-" * 80)
        print()

    # Demonstrate EP improvement potential
    print("=" * 80)
    print("Quality EP Improvement Potential")
    print("=" * 80)
    print()

    print("Comparing approaches for same query:")
    query = "What is the current frustration level?"
    print(f"Query: \"{query}\"")
    print()

    styles = ["specific", "hedging", "generic"]
    predictions = {}

    for style in styles:
        pred = predictor.predict_quality(query, intended_style=style)
        predictions[style] = pred

        print(f"{style.upper()} style:")
        print(f"  Predicted quality: {pred.predicted_quality:.2f}")
        print(f"  Recommendation: {pred.recommendation}")
        print()

    # Best approach
    best_style = max(predictions.keys(), key=lambda s: predictions[s].predicted_quality)
    worst_style = min(predictions.keys(), key=lambda s: predictions[s].predicted_quality)

    print(f"EP Guidance:")
    print(f"  ✅ BEST approach: {best_style} (quality: {predictions[best_style].predicted_quality:.2f})")
    print(f"  ❌ WORST approach: {worst_style} (quality: {predictions[worst_style].predicted_quality:.2f})")
    print(f"  📈 Improvement potential: {predictions[best_style].predicted_quality - predictions[worst_style].predicted_quality:.2f}")
    print()

    print("This IS epistemic proprioception for quality:")
    print("  - SAGE predicts quality BEFORE generating")
    print("  - Can choose better approach proactively")
    print("  - Avoids low-quality responses before they happen")
    print("  - Learns from pattern history")
    print()

    # Meta-analysis
    print("=" * 80)
    print("Pattern Analysis - What has SAGE learned?")
    print("=" * 80)
    print()

    analysis = collector.analyze_patterns()

    print(f"Response Style → Quality relationships:")
    for style, avg_quality in analysis['response_styles']['avg_quality'].items():
        count = analysis['response_styles']['counts'][style]
        print(f"  {style}: {avg_quality:.2f} average quality ({count} patterns)")
    print()

    print(f"Query Type → Quality relationships:")
    for qtype, avg_quality in analysis['query_types']['avg_quality'].items():
        count = analysis['query_types']['counts'][qtype]
        print(f"  {qtype}: {avg_quality:.2f} average quality ({count} patterns)")
    print()

    print("Quality EP Maturation:")
    if len(collector.patterns) < 10:
        print(f"  Stage 1 (Immature): Only {len(collector.patterns)} patterns")
        print(f"  Need: More patterns for confident prediction")
    elif len(collector.patterns) < 50:
        print(f"  Stage 2 (Learning): {len(collector.patterns)} patterns collected")
        print(f"  Capability: Basic prediction with moderate confidence")
    else:
        print(f"  Stage 3 (Mature): {len(collector.patterns)} patterns")
        print(f"  Capability: High-confidence quality prediction")
    print()


if __name__ == "__main__":
    test_prediction()
