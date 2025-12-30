#!/usr/bin/env python3
"""
Quality EP Adjuster - Phase 3 of Quality Epistemic Proprioception

Implements approach adjustment based on quality predictions.
This is the core EP capability: ADJUST approach BEFORE generating when
low quality is predicted.

Phase 3: Approach Adjustment
- Receive quality prediction
- Determine if adjustment needed
- Apply adjustment strategy
- Return modified approach

Date: 2025-12-30
Hardware: Thor (Jetson AGX Thor Developer Kit)
Foundation: QUALITY_EPISTEMIC_PROPRIOCEPTION.md Phase 3
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum

# Add SAGE modules to path
sys.path.append(str(Path(__file__).parent.parent))

from quality_pattern_collector import QueryContext
from quality_pattern_predictor import QualityPrediction


class AdjustmentStrategy(Enum):
    """Available adjustment strategies for improving quality."""

    NONE = "none"  # No adjustment needed
    STYLE_SHIFT = "style_shift"  # Change response style (hedging → specific)
    CONTENT_ENRICH = "content_enrich"  # Add SAGE terms, numbers, details
    KNOWLEDGE_EXPAND = "knowledge_expand"  # Query expansion for more context
    SPECIFICITY_INCREASE = "specificity_increase"  # Add concrete examples
    HEDGING_REMOVE = "hedging_remove"  # Remove uncertainty language


@dataclass
class AdjustedApproach:
    """
    Result of approach adjustment.

    Contains both the adjustment decision and guidance for generation.
    """

    original_prediction: QualityPrediction
    adjustment_applied: bool
    strategy_used: AdjustmentStrategy

    # Guidance for response generation
    target_style: str  # "specific", "generic", "hedging"
    include_numbers: bool  # Should include numerical data
    include_sage_terms: bool  # Should use SAGE-specific terminology
    avoid_hedging: bool  # Should avoid uncertainty language
    add_examples: bool  # Should include concrete examples

    # Meta information
    adjustment_reason: str
    expected_quality_improvement: float  # How much quality should improve

    def get_generation_guidance(self) -> Dict[str, Any]:
        """
        Get guidance dictionary for response generation.

        This provides structured guidance that can be used by
        a response generator to produce higher-quality output.
        """
        return {
            "style": self.target_style,
            "requirements": {
                "numbers": self.include_numbers,
                "sage_terms": self.include_sage_terms,
                "avoid_hedging": self.avoid_hedging,
                "examples": self.add_examples,
            },
            "strategy": self.strategy_used.value,
            "reason": self.adjustment_reason,
            "expected_quality": (
                self.original_prediction.predicted_quality +
                self.expected_quality_improvement
            )
        }


class QualityEPAdjuster:
    """
    Adjusts response approach based on quality predictions.

    This is the EP adjustment component - predicts quality, then
    modifies approach BEFORE generation if low quality predicted.

    Core EP loop:
    1. Predict quality (from QualityPredictor)
    2. If quality < threshold AND confidence > threshold
    3. Apply adjustment strategy
    4. Generate with adjusted approach
    5. Measure actual quality (learn from adjustment)
    """

    def __init__(
        self,
        quality_threshold: float = 0.70,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize adjuster with thresholds.

        Args:
            quality_threshold: Adjust if predicted quality below this
            confidence_threshold: Only adjust if confidence above this
        """
        self.quality_threshold = quality_threshold
        self.confidence_threshold = confidence_threshold

        # Track adjustment effectiveness
        self.adjustments_made = 0
        self.adjustments_successful = 0  # Improved quality
        self.adjustment_history: List[Dict[str, Any]] = []

    def adjust_approach(
        self,
        query_context: QueryContext,
        prediction: QualityPrediction,
        current_approach: Optional[str] = None
    ) -> AdjustedApproach:
        """
        Adjust approach based on quality prediction.

        This is the main EP adjustment decision point.

        Args:
            query_context: Context of the query
            prediction: Quality prediction from QualityPredictor
            current_approach: Current intended approach (if any)

        Returns:
            AdjustedApproach with guidance for generation
        """
        # Decision: Should we adjust?
        should_adjust = self._should_adjust(prediction)

        if not should_adjust:
            # Quality predicted good or low confidence - use original approach
            return self._no_adjustment(query_context, prediction, current_approach)

        # Quality predicted low with confidence - ADJUST
        strategy = self._select_strategy(query_context, prediction)
        return self._apply_strategy(
            query_context,
            prediction,
            strategy,
            current_approach
        )

    def _should_adjust(self, prediction: QualityPrediction) -> bool:
        """
        Decide if adjustment is needed.

        Adjust if:
        - Predicted quality < threshold AND
        - Confidence > threshold (trust the prediction)
        """
        return (
            prediction.predicted_quality < self.quality_threshold and
            prediction.confidence > self.confidence_threshold
        )

    def _no_adjustment(
        self,
        query_context: QueryContext,
        prediction: QualityPrediction,
        current_approach: Optional[str]
    ) -> AdjustedApproach:
        """Create AdjustedApproach with no adjustment."""

        # Use current approach or default to specific style
        target_style = current_approach if current_approach else "specific"

        reason = (
            f"Quality {prediction.predicted_quality:.2f} ≥ threshold "
            f"{self.quality_threshold:.2f}"
            if prediction.predicted_quality >= self.quality_threshold
            else f"Confidence {prediction.confidence:.2f} < threshold "
                 f"{self.confidence_threshold:.2f}"
        )

        return AdjustedApproach(
            original_prediction=prediction,
            adjustment_applied=False,
            strategy_used=AdjustmentStrategy.NONE,
            target_style=target_style,
            include_numbers=query_context.expects_numbers,
            include_sage_terms=True,  # Always prefer SAGE terms
            avoid_hedging=True,  # Always avoid hedging
            add_examples=query_context.expects_specifics,
            adjustment_reason=reason,
            expected_quality_improvement=0.0
        )

    def _select_strategy(
        self,
        query_context: QueryContext,
        prediction: QualityPrediction
    ) -> AdjustmentStrategy:
        """
        Select best adjustment strategy based on context.

        Strategy selection logic:
        1. If status query missing numbers → add numbers
        2. If technical query → increase specificity
        3. If low confidence prediction → expand knowledge
        4. Default → style shift to specific
        """
        # Status queries need numbers
        if query_context.query_type == "status":
            if query_context.expects_numbers:
                return AdjustmentStrategy.CONTENT_ENRICH

        # Technical queries need specificity
        if query_context.query_type == "technical":
            if query_context.expects_specifics:
                return AdjustmentStrategy.SPECIFICITY_INCREASE

        # Low confidence → need more knowledge
        if prediction.confidence < 0.7:
            return AdjustmentStrategy.KNOWLEDGE_EXPAND

        # Default: shift to specific style
        return AdjustmentStrategy.STYLE_SHIFT

    def _apply_strategy(
        self,
        query_context: QueryContext,
        prediction: QualityPrediction,
        strategy: AdjustmentStrategy,
        current_approach: Optional[str]
    ) -> AdjustedApproach:
        """
        Apply selected adjustment strategy.

        Each strategy modifies generation guidance differently.
        """
        # Base settings
        target_style = "specific"  # Always aim for specific
        include_numbers = query_context.expects_numbers
        include_sage_terms = True
        avoid_hedging = True
        add_examples = query_context.expects_specifics
        expected_improvement = 0.15  # Base improvement estimate

        # Strategy-specific modifications
        if strategy == AdjustmentStrategy.STYLE_SHIFT:
            reason = "Shifting from hedging/generic to specific style"
            expected_improvement = 0.25  # Style shift has high impact

        elif strategy == AdjustmentStrategy.CONTENT_ENRICH:
            reason = "Enriching with SAGE terms and numerical data"
            include_numbers = True  # Force numbers
            include_sage_terms = True
            expected_improvement = 0.20

        elif strategy == AdjustmentStrategy.SPECIFICITY_INCREASE:
            reason = "Increasing specificity with examples and details"
            add_examples = True
            include_sage_terms = True
            expected_improvement = 0.20

        elif strategy == AdjustmentStrategy.KNOWLEDGE_EXPAND:
            reason = "Expanding knowledge context before generation"
            add_examples = True
            include_sage_terms = True
            expected_improvement = 0.15

        elif strategy == AdjustmentStrategy.HEDGING_REMOVE:
            reason = "Removing hedging language, increasing confidence"
            avoid_hedging = True
            target_style = "specific"
            expected_improvement = 0.30  # Removing hedging has huge impact

        else:
            reason = "Unknown strategy"
            expected_improvement = 0.10

        # Track adjustment
        self.adjustments_made += 1

        return AdjustedApproach(
            original_prediction=prediction,
            adjustment_applied=True,
            strategy_used=strategy,
            target_style=target_style,
            include_numbers=include_numbers,
            include_sage_terms=include_sage_terms,
            avoid_hedging=avoid_hedging,
            add_examples=add_examples,
            adjustment_reason=reason,
            expected_quality_improvement=expected_improvement
        )

    def record_result(
        self,
        adjusted_approach: AdjustedApproach,
        actual_quality: float
    ):
        """
        Record adjustment result for learning.

        Tracks if adjustment actually improved quality.
        This enables EP maturation - learning which adjustments work.
        """
        if not adjusted_approach.adjustment_applied:
            return  # No adjustment to learn from

        original_quality = adjusted_approach.original_prediction.predicted_quality
        quality_improved = actual_quality > original_quality

        if quality_improved:
            self.adjustments_successful += 1

        # Record history
        self.adjustment_history.append({
            "strategy": adjusted_approach.strategy_used.value,
            "predicted_quality": original_quality,
            "actual_quality": actual_quality,
            "expected_improvement": adjusted_approach.expected_quality_improvement,
            "actual_improvement": actual_quality - original_quality,
            "successful": quality_improved,
        })

    def get_effectiveness(self) -> Dict[str, Any]:
        """
        Get adjustment effectiveness statistics.

        Returns learning insights about which strategies work.
        """
        if self.adjustments_made == 0:
            return {
                "total_adjustments": 0,
                "success_rate": 0.0,
                "message": "No adjustments made yet"
            }

        success_rate = self.adjustments_successful / self.adjustments_made

        # Strategy effectiveness
        strategy_stats = {}
        for record in self.adjustment_history:
            strategy = record["strategy"]
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "count": 0,
                    "successes": 0,
                    "avg_improvement": 0.0,
                    "improvements": []
                }

            strategy_stats[strategy]["count"] += 1
            if record["successful"]:
                strategy_stats[strategy]["successes"] += 1
            strategy_stats[strategy]["improvements"].append(
                record["actual_improvement"]
            )

        # Calculate averages
        for strategy, stats in strategy_stats.items():
            stats["avg_improvement"] = (
                sum(stats["improvements"]) / len(stats["improvements"])
            )
            stats["success_rate"] = stats["successes"] / stats["count"]
            del stats["improvements"]  # Remove raw data

        return {
            "total_adjustments": self.adjustments_made,
            "successful_adjustments": self.adjustments_successful,
            "success_rate": success_rate,
            "strategy_effectiveness": strategy_stats,
        }


def demo_adjustment():
    """
    Demonstrate Quality EP adjustment in action.

    Shows the complete EP loop:
    1. Predict quality
    2. Adjust if needed
    3. Generate with guidance
    4. Measure and learn
    """
    print("=" * 80)
    print("Quality EP Adjuster Demo - Phase 3")
    print("=" * 80)
    print()

    adjuster = QualityEPAdjuster(
        quality_threshold=0.70,
        confidence_threshold=0.5
    )

    # Test cases: various predictions
    test_cases = [
        {
            "query": "What is the current ATP balance?",
            "predicted_quality": 0.40,
            "confidence": 0.8,
            "description": "Status query, low quality predicted, high confidence"
        },
        {
            "query": "How does SNARC work?",
            "predicted_quality": 0.85,
            "confidence": 0.6,
            "description": "Technical query, high quality predicted"
        },
        {
            "query": "Why is consciousness important?",
            "predicted_quality": 0.50,
            "confidence": 0.3,
            "description": "Conceptual query, low quality but low confidence"
        },
        {
            "query": "How does emotional regulation work?",
            "predicted_quality": 0.45,
            "confidence": 0.9,
            "description": "Technical query, low quality, very high confidence"
        },
    ]

    print("Testing EP Adjustment Decision Making")
    print("=" * 80)
    print()

    for i, case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {case['description']}")
        print(f"Query: \"{case['query']}\"")
        print(f"Predicted quality: {case['predicted_quality']:.2f}")
        print(f"Confidence: {case['confidence']:.2f}")
        print()

        # Create query context
        query_context = QueryContext.analyze_query(case['query'])

        # Create prediction
        prediction = QualityPrediction(
            predicted_quality=case['predicted_quality'],
            confidence=case['confidence'],
            matching_patterns=5,
            recommendation="adjust" if case['predicted_quality'] < 0.70 else "generate"
        )

        # EP ADJUSTMENT DECISION
        adjusted = adjuster.adjust_approach(query_context, prediction)

        print(f"EP Decision:")
        if adjusted.adjustment_applied:
            print(f"  ⚠️  ADJUST: {adjusted.adjustment_reason}")
            print(f"  Strategy: {adjusted.strategy_used.value}")
            print(f"  Expected improvement: +{adjusted.expected_quality_improvement:.2f}")
            print(f"  Expected final quality: {case['predicted_quality'] + adjusted.expected_quality_improvement:.2f}")
            print()
            print(f"  Generation Guidance:")
            guidance = adjusted.get_generation_guidance()
            print(f"    Style: {guidance['style']}")
            print(f"    Include numbers: {guidance['requirements']['numbers']}")
            print(f"    Include SAGE terms: {guidance['requirements']['sage_terms']}")
            print(f"    Avoid hedging: {guidance['requirements']['avoid_hedging']}")
            print(f"    Add examples: {guidance['requirements']['examples']}")
        else:
            print(f"  ✅ NO ADJUSTMENT: {adjusted.adjustment_reason}")
            print(f"  Proceed with original approach")

        print()
        print("-" * 80)
        print()

    # Simulate learning from adjustments
    print("=" * 80)
    print("EP Learning: Recording Adjustment Results")
    print("=" * 80)
    print()

    # Simulate some adjustment results
    simulated_results = [
        (0.40, 0.70, "style_shift"),  # Predicted 0.40, actual 0.70
        (0.45, 0.75, "content_enrich"),  # Predicted 0.45, actual 0.75
        (0.50, 0.65, "specificity_increase"),  # Predicted 0.50, actual 0.65
    ]

    for pred_q, actual_q, strategy in simulated_results:
        # Create mock adjusted approach
        mock_prediction = QualityPrediction(
            predicted_quality=pred_q,
            confidence=0.8,
            matching_patterns=5,
            recommendation="adjust"
        )

        mock_adjusted = AdjustedApproach(
            original_prediction=mock_prediction,
            adjustment_applied=True,
            strategy_used=AdjustmentStrategy[strategy.upper()],
            target_style="specific",
            include_numbers=True,
            include_sage_terms=True,
            avoid_hedging=True,
            add_examples=True,
            adjustment_reason=f"Test {strategy}",
            expected_quality_improvement=0.20
        )

        adjuster.record_result(mock_adjusted, actual_q)

        print(f"Adjustment: {strategy}")
        print(f"  Predicted: {pred_q:.2f} → Actual: {actual_q:.2f}")
        print(f"  Improvement: +{actual_q - pred_q:.2f}")
        print()

    # Show effectiveness
    print("=" * 80)
    print("EP Effectiveness Analysis")
    print("=" * 80)
    print()

    effectiveness = adjuster.get_effectiveness()
    print(f"Total adjustments: {effectiveness['total_adjustments']}")
    print(f"Successful: {effectiveness['successful_adjustments']}")
    print(f"Success rate: {effectiveness['success_rate']:.1%}")
    print()

    print("Strategy Effectiveness:")
    for strategy, stats in effectiveness['strategy_effectiveness'].items():
        print(f"  {strategy}:")
        print(f"    Count: {stats['count']}")
        print(f"    Success rate: {stats['success_rate']:.1%}")
        print(f"    Avg improvement: {stats['avg_improvement']:+.2f}")
    print()

    print("This demonstrates Quality EP maturation:")
    print("  - Learning which strategies work best")
    print("  - Measuring actual vs expected improvement")
    print("  - Building strategy effectiveness knowledge")
    print()


if __name__ == "__main__":
    demo_adjustment()
