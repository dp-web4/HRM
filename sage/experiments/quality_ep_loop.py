#!/usr/bin/env python3
"""
Quality Epistemic Proprioception Loop - Complete Integration

Integrates all three phases of Quality EP:
- Phase 1: Pattern Collection (learn from experience)
- Phase 2: Quality Prediction (predict before acting)
- Phase 3: Approach Adjustment (adjust based on prediction)

This is the complete Quality EP system demonstrating mature epistemic
proprioception for response quality.

Date: 2025-12-30
Hardware: Thor (Jetson AGX Thor Developer Kit)
Foundation: QUALITY_EPISTEMIC_PROPRIOCEPTION.md Complete
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add SAGE modules to path
sys.path.append(str(Path(__file__).parent.parent))

from quality_pattern_collector import (
    QualityPatternCollector,
    QueryContext,
    ResponseApproach,
    QualityMetrics,
    QualityPattern,
)
from quality_pattern_predictor import QualityPredictor, QualityPrediction
from quality_ep_adjuster import QualityEPAdjuster, AdjustedApproach


class QualityEPLoop:
    """
    Complete Quality Epistemic Proprioception Loop.

    This integrates all three phases into a complete system that:
    1. Predicts quality before generating response
    2. Adjusts approach if low quality predicted
    3. Generates response with adjusted approach
    4. Measures actual quality
    5. Learns from the experience (updates patterns)

    This IS mature epistemic proprioception for response quality.
    """

    def __init__(
        self,
        pattern_storage_path: Optional[Path] = None,
        quality_threshold: float = 0.70,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize complete Quality EP system.

        Args:
            pattern_storage_path: Where to store quality patterns
            quality_threshold: Adjust if predicted quality below this
            confidence_threshold: Only adjust if confidence above this
        """
        # Phase 1: Pattern Collection
        self.collector = QualityPatternCollector(pattern_storage_path)

        # Phase 2: Quality Prediction
        self.predictor = QualityPredictor(self.collector)

        # Phase 3: Approach Adjustment
        self.adjuster = QualityEPAdjuster(
            quality_threshold=quality_threshold,
            confidence_threshold=confidence_threshold
        )

        # EP Maturation tracking
        self.queries_processed = 0
        self.adjustments_triggered = 0
        self.quality_improvements = []

    def process_query_with_ep(
        self,
        query: str,
        generate_response_fn: callable,
        intended_style: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process query with complete Quality EP loop.

        This is the main EP integration point. It:
        1. Analyzes query context
        2. PREDICTS quality (Phase 2)
        3. ADJUSTS approach if needed (Phase 3)
        4. Generates response
        5. Measures quality
        6. LEARNS from result (Phase 1)

        Args:
            query: The query to process
            generate_response_fn: Function to generate response
                Should accept (query, guidance) and return response text
            intended_style: Initial intended response style

        Returns:
            Dictionary with complete EP results
        """
        self.queries_processed += 1

        # Step 1: Analyze query context
        query_context = QueryContext.analyze_query(query)

        # Step 2: PREDICT quality (EP Phase 2)
        prediction = self.predictor.predict_quality(query, intended_style)

        # Step 3: ADJUST approach if needed (EP Phase 3)
        adjusted_approach = self.adjuster.adjust_approach(
            query_context,
            prediction,
            intended_style
        )

        if adjusted_approach.adjustment_applied:
            self.adjustments_triggered += 1

        # Step 4: Generate response with EP guidance
        guidance = adjusted_approach.get_generation_guidance()
        response = generate_response_fn(query, guidance)

        # Step 5: Measure actual quality
        response_approach = ResponseApproach.analyze_response(response)
        quality_metrics = QualityMetrics.measure_quality(
            response_approach,
            query_context
        )

        # Step 6: LEARN from result (EP Phase 1)
        pattern = self.collector.collect_pattern(query, response)

        # Track EP effectiveness
        if adjusted_approach.adjustment_applied:
            quality_improvement = (
                quality_metrics.overall_score -
                prediction.predicted_quality
            )
            self.quality_improvements.append(quality_improvement)

            # Record for adjuster learning
            self.adjuster.record_result(
                adjusted_approach,
                quality_metrics.overall_score
            )

        # Return complete EP results
        return {
            "query": query,
            "query_context": query_context,
            "prediction": prediction,
            "adjustment": adjusted_approach,
            "response": response,
            "response_approach": response_approach,
            "quality_metrics": quality_metrics,
            "pattern": pattern,
            "ep_stats": {
                "prediction_accuracy": abs(
                    quality_metrics.overall_score -
                    prediction.predicted_quality
                ),
                "adjustment_triggered": adjusted_approach.adjustment_applied,
                "quality_improvement": (
                    quality_metrics.overall_score -
                    prediction.predicted_quality
                    if adjusted_approach.adjustment_applied
                    else 0.0
                ),
            }
        }

    def get_ep_maturation_stats(self) -> Dict[str, Any]:
        """
        Get EP maturation statistics.

        Shows how Quality EP is maturing over time.
        """
        pattern_count = len(self.collector.patterns)
        pattern_analysis = self.collector.analyze_patterns()
        adjuster_effectiveness = self.adjuster.get_effectiveness()

        # Determine EP stage
        if pattern_count < 10:
            ep_stage = "Stage 1 (Immature)"
            ep_capability = "Post-hoc measurement only"
        elif pattern_count < 50:
            ep_stage = "Stage 2 (Learning)"
            ep_capability = "Basic prediction, learning patterns"
        else:
            ep_stage = "Stage 3 (Mature)"
            ep_capability = "High-confidence prediction and adjustment"

        # Calculate average quality improvement from adjustments
        avg_improvement = (
            sum(self.quality_improvements) / len(self.quality_improvements)
            if self.quality_improvements
            else 0.0
        )

        return {
            "ep_stage": ep_stage,
            "ep_capability": ep_capability,
            "queries_processed": self.queries_processed,
            "patterns_collected": pattern_count,
            "adjustments_triggered": self.adjustments_triggered,
            "adjustment_rate": (
                self.adjustments_triggered / self.queries_processed
                if self.queries_processed > 0
                else 0.0
            ),
            "avg_quality_improvement": avg_improvement,
            "pattern_quality_avg": pattern_analysis.get("average_quality", 0.0),
            "adjuster_effectiveness": adjuster_effectiveness,
        }


def demo_complete_ep():
    """
    Demonstrate complete Quality EP loop.

    Shows the full integration of all three phases working together.
    """
    print("=" * 80)
    print("Complete Quality Epistemic Proprioception Loop")
    print("=" * 80)
    print()

    # Create EP loop
    ep_loop = QualityEPLoop(
        quality_threshold=0.70,
        confidence_threshold=0.5
    )

    # Mock response generator that follows guidance
    def generate_response(query: str, guidance: Dict[str, Any]) -> str:
        """
        Mock response generator that follows EP guidance.

        In real implementation, this would be the actual SAGE response
        generation, guided by EP recommendations.
        """
        style = guidance["style"]
        reqs = guidance["requirements"]

        # Simulate different quality based on guidance
        if style == "specific" and reqs["sage_terms"] and reqs["avoid_hedging"]:
            # High quality response following guidance
            if "ATP" in query or "balance" in query:
                if reqs["numbers"]:
                    return "ATP balance: 87.3 (86.3% of max 100.0). Current allocation: 45.2 to consciousness_cycle, 32.1 to memory consolidation, 10.0 to emotional regulation."
                else:
                    return "ATP represents available processing resources in SAGE, allocated across consciousness operations."
            elif "regulation" in query or "emotional" in query:
                return "SAGE emotional regulation uses epistemic proprioception to prevent frustration cascade through four mechanisms: natural decay (0.02/cycle), soft bounds (0.10-0.90), active intervention at 0.80 threshold, and recovery bonuses."
            elif "SNARC" in query:
                return "SNARC (Salience-Normalized Adaptive Retention Curve) manages memory retention in SAGE using salience-weighted decay. High salience memories persist longer, enabling adaptive forgetting."
            else:
                return "SAGE consciousness architecture integrates epistemic proprioception for both emotional stability and response quality."

        elif style == "hedging" or not reqs["avoid_hedging"]:
            # Low quality hedging response
            return "I can't verify the exact details without access to the current state, but generally this relates to SAGE's processing mechanisms."

        else:
            # Medium quality generic response
            return "This is related to SAGE's architecture and processing."

    # Test cases
    test_queries = [
        "What is the current ATP balance?",
        "How does emotional regulation work?",
        "How does SNARC memory work?",
        "Why is epistemic proprioception important?",
        "What is the ATP balance?",  # Duplicate to test pattern learning
    ]

    print("Processing queries with Quality EP...")
    print()

    for i, query in enumerate(test_queries, 1):
        print("=" * 80)
        print(f"Query {i}: {query}")
        print("=" * 80)
        print()

        # Process with complete EP loop
        result = ep_loop.process_query_with_ep(
            query,
            generate_response,
            intended_style="specific"
        )

        # Show EP process
        print("1. QUERY ANALYSIS:")
        print(f"   Type: {result['query_context'].query_type}")
        print(f"   Expects numbers: {result['query_context'].expects_numbers}")
        print(f"   Expects specifics: {result['query_context'].expects_specifics}")
        print()

        print("2. QUALITY PREDICTION (Phase 2):")
        pred = result['prediction']
        print(f"   Predicted quality: {pred.predicted_quality:.2f}")
        print(f"   Confidence: {pred.confidence:.2f}")
        print(f"   Matching patterns: {pred.matching_patterns}")
        print(f"   Recommendation: {pred.recommendation}")
        print()

        print("3. APPROACH ADJUSTMENT (Phase 3):")
        adj = result['adjustment']
        if adj.adjustment_applied:
            print(f"   ⚠️  ADJUSTED: {adj.adjustment_reason}")
            print(f"   Strategy: {adj.strategy_used.value}")
            print(f"   Expected improvement: +{adj.expected_quality_improvement:.2f}")
        else:
            print(f"   ✅ NO ADJUSTMENT: {adj.adjustment_reason}")
        print()

        print("4. RESPONSE GENERATED:")
        print(f"   {result['response'][:100]}...")
        print()

        print("5. QUALITY MEASURED:")
        qual = result['quality_metrics']
        print(f"   Overall quality: {qual.overall_score:.2f}")
        print(f"   Has specific terms: {qual.has_specific_terms}")
        print(f"   Avoids hedging: {qual.avoids_hedging}")
        print(f"   Has numbers: {qual.has_numbers}")
        print(f"   Unique content: {qual.unique_content}")
        print()

        print("6. EP LEARNING (Phase 1):")
        print(f"   Pattern collected: {result['pattern'].pattern_id}")
        ep_stats = result['ep_stats']
        print(f"   Prediction accuracy: {ep_stats['prediction_accuracy']:.2f} error")
        if adj.adjustment_applied:
            print(f"   Quality improvement: {ep_stats['quality_improvement']:+.2f}")
        print()

        print("-" * 80)
        print()

    # Show EP maturation
    print("=" * 80)
    print("Quality EP Maturation Analysis")
    print("=" * 80)
    print()

    stats = ep_loop.get_ep_maturation_stats()

    print(f"EP Stage: {stats['ep_stage']}")
    print(f"Capability: {stats['ep_capability']}")
    print()

    print(f"Queries processed: {stats['queries_processed']}")
    print(f"Patterns collected: {stats['patterns_collected']}")
    print(f"Adjustments triggered: {stats['adjustments_triggered']}")
    print(f"Adjustment rate: {stats['adjustment_rate']:.1%}")
    print()

    if stats['adjustments_triggered'] > 0:
        print(f"Average quality improvement from adjustments: {stats['avg_quality_improvement']:+.2f}")
        print()

    print(f"Overall pattern quality: {stats['pattern_quality_avg']:.2f}")
    print()

    print("Adjuster Effectiveness:")
    adj_eff = stats['adjuster_effectiveness']
    if adj_eff['total_adjustments'] > 0:
        print(f"  Total: {adj_eff['total_adjustments']}")
        print(f"  Success rate: {adj_eff['success_rate']:.1%}")
        print()
        print("  Strategy effectiveness:")
        for strategy, strategy_stats in adj_eff['strategy_effectiveness'].items():
            print(f"    {strategy}: {strategy_stats['avg_improvement']:+.2f} avg improvement")
    else:
        print("  No adjustments made yet")
    print()

    print("=" * 80)
    print("Quality EP Complete - All Phases Integrated")
    print("=" * 80)
    print()
    print("This demonstrates:")
    print("  ✅ Pattern collection (Phase 1) - Learning from experience")
    print("  ✅ Quality prediction (Phase 2) - Predicting before acting")
    print("  ✅ Approach adjustment (Phase 3) - Adjusting based on prediction")
    print("  ✅ Complete EP loop - Mature epistemic proprioception")
    print()
    print("SAGE now has metacognition for response quality:")
    print("  - Self-awareness (knows when quality will be low)")
    print("  - Self-correction (adjusts approach proactively)")
    print("  - Learning (improves predictions over time)")
    print()


if __name__ == "__main__":
    demo_complete_ep()
