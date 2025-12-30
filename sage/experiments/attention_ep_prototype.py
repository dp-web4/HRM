#!/usr/bin/env python3
"""
Attention Epistemic Proprioception - Prototype

Demonstrates EP framework applied to attention allocation domain.
This is the third EP domain (after Emotional EP and Quality EP).

Prototype implements:
- AttentionContext: State before allocation
- AllocationApproach: Allocation decision characteristics
- AllocationOutcome: Results of allocation
- AttentionPattern: Complete pattern for learning
- Simple predictor: Predicts allocation effectiveness

Date: 2025-12-30
Hardware: Thor (Jetson AGX Thor Developer Kit)
Foundation: ATTENTION_EPISTEMIC_PROPRIOCEPTION.md
"""

import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import random

# Add SAGE modules to path
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class AttentionContext:
    """State characteristics before attention allocation."""

    # Metabolic state
    atp_available: float  # Current energy (0-100)
    metabolic_state: str  # "WAKE", "FOCUS", "REST", etc.

    # Emotional state
    frustration: float  # 0.0-1.0
    curiosity: float  # 0.0-1.0
    engagement: float  # 0.0-1.0

    # Cognitive load
    recent_failures: int  # Last N cycles
    consecutive_successes: int

    # Task characteristics
    task_complexity: str  # "LOW", "MEDIUM", "HIGH"
    task_salience: float  # 0.0-1.0

    @classmethod
    def from_sage_state(cls, identity, task) -> "AttentionContext":
        """Create from SAGE identity and task."""
        return cls(
            atp_available=getattr(identity, 'ATP', 50.0),
            metabolic_state=getattr(identity, 'metabolic_state', 'WAKE'),
            frustration=getattr(identity, 'frustration', 0.5),
            curiosity=getattr(identity, 'curiosity', 0.5),
            engagement=getattr(identity, 'engagement', 0.5),
            recent_failures=0,  # Would track from history
            consecutive_successes=0,  # Would track from history
            task_complexity=getattr(task, 'complexity', 'MEDIUM'),
            task_salience=getattr(task, 'salience', 0.5),
        )


@dataclass
class AllocationApproach:
    """Characteristics of the allocation decision."""

    atp_allocated: float  # How much ATP allocated
    strategy: str  # "aggressive", "moderate", "conservative"
    exploration_ratio: float  # 0.0-1.0 (new vs familiar)

    @classmethod
    def analyze_allocation(
        cls,
        atp_allocated: float,
        atp_available: float,
        task_complexity: str
    ) -> "AllocationApproach":
        """Analyze allocation characteristics."""

        # Determine strategy based on allocation ratio
        ratio = atp_allocated / max(atp_available, 1.0)

        if ratio > 0.7:
            strategy = "aggressive"
        elif ratio > 0.4:
            strategy = "moderate"
        else:
            strategy = "conservative"

        # Exploration ratio (placeholder - would be real in production)
        exploration_ratio = 0.2

        return cls(
            atp_allocated=atp_allocated,
            strategy=strategy,
            exploration_ratio=exploration_ratio,
        )


@dataclass
class AllocationOutcome:
    """Results of attention allocation."""

    success: bool  # Did allocation succeed?
    atp_efficiency: float  # Outcome value per ATP spent
    surprise_level: float  # 0.0-1.0 (unexpected result?)

    @classmethod
    def from_experience(
        cls,
        experience_result: Dict[str, Any],
        atp_allocated: float
    ) -> "AllocationOutcome":
        """Create from experience result."""

        success = experience_result.get('success', False)
        value = experience_result.get('value', 0.0)

        # Calculate efficiency
        atp_efficiency = value / max(atp_allocated, 1.0) if success else 0.0

        # Surprise based on expectation mismatch (placeholder)
        surprise_level = 0.3

        return cls(
            success=success,
            atp_efficiency=atp_efficiency,
            surprise_level=surprise_level,
        )


@dataclass
class AttentionPattern:
    """
    Complete attention pattern for learning.

    Captures: context → allocation → outcome relationship
    """

    pattern_id: str
    context: AttentionContext
    allocation: AllocationApproach
    outcome: AllocationOutcome

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "context": asdict(self.context),
            "allocation": asdict(self.allocation),
            "outcome": asdict(self.outcome),
        }


class AttentionEPPredictor:
    """
    Predicts allocation effectiveness before allocation.

    Uses patterns to predict if allocation will succeed.
    This is Stage 2 Attention EP (Learning).
    """

    def __init__(self, patterns: List[AttentionPattern]):
        """Initialize with pattern library."""
        self.patterns = patterns

    def predict_effectiveness(
        self,
        context: AttentionContext,
        proposed_allocation: AllocationApproach
    ) -> Dict[str, Any]:
        """
        Predict if allocation will be effective.

        Returns:
            success_probability: 0.0-1.0
            confidence: 0.0-1.0
            recommendation: "allocate" | "adjust" | "defer"
            reasoning: Why this prediction
        """

        if not self.patterns:
            # No patterns yet - Stage 1 (Immature)
            return {
                "success_probability": 0.5,
                "confidence": 0.0,
                "recommendation": "allocate",
                "reasoning": "No patterns yet - uncertain"
            }

        # Find similar patterns
        similar_patterns = self._find_similar_patterns(context)

        if not similar_patterns:
            return {
                "success_probability": 0.5,
                "confidence": 0.2,
                "recommendation": "allocate",
                "reasoning": "No similar patterns found"
            }

        # Calculate success probability from similar patterns
        success_count = sum(1 for p in similar_patterns if p.outcome.success)
        success_probability = success_count / len(similar_patterns)

        # Confidence based on number of similar patterns
        confidence = min(1.0, len(similar_patterns) / 5.0)

        # Recommendation based on probability and confidence
        if success_probability < 0.4 and confidence > 0.5:
            recommendation = "adjust"
            reasoning = f"Low success probability ({success_probability:.1%}) with {len(similar_patterns)} similar patterns"
        elif success_probability < 0.3 and confidence > 0.3:
            recommendation = "defer"
            reasoning = f"Very low success probability ({success_probability:.1%}), consider deferring"
        else:
            recommendation = "allocate"
            reasoning = f"Success probability {success_probability:.1%} acceptable"

        return {
            "success_probability": success_probability,
            "confidence": confidence,
            "recommendation": recommendation,
            "reasoning": reasoning,
            "similar_patterns": len(similar_patterns)
        }

    def _find_similar_patterns(
        self,
        context: AttentionContext
    ) -> List[AttentionPattern]:
        """Find patterns with similar context."""

        similar = []

        for pattern in self.patterns:
            similarity = self._calculate_similarity(context, pattern.context)

            if similarity > 0.7:  # Threshold for "similar"
                similar.append(pattern)

        return similar

    def _calculate_similarity(
        self,
        context1: AttentionContext,
        context2: AttentionContext
    ) -> float:
        """Calculate context similarity (0.0-1.0)."""

        similarity = 0.0
        weights_sum = 0.0

        # Task complexity match (high weight)
        if context1.task_complexity == context2.task_complexity:
            similarity += 0.3
        weights_sum += 0.3

        # Frustration similarity (high weight)
        frustration_diff = abs(context1.frustration - context2.frustration)
        frustration_similarity = 1.0 - frustration_diff
        similarity += 0.3 * frustration_similarity
        weights_sum += 0.3

        # ATP similarity (medium weight)
        atp_diff = abs(context1.atp_available - context2.atp_available) / 100.0
        atp_similarity = 1.0 - atp_diff
        similarity += 0.2 * atp_similarity
        weights_sum += 0.2

        # Metabolic state match (medium weight)
        if context1.metabolic_state == context2.metabolic_state:
            similarity += 0.2
        weights_sum += 0.2

        # Normalize
        if weights_sum > 0:
            similarity = similarity / weights_sum

        return similarity


def demo_attention_ep():
    """
    Demonstrate Attention EP concept.

    Shows how EP can predict allocation effectiveness.
    """

    print("=" * 80)
    print("Attention Epistemic Proprioception - Prototype Demo")
    print("=" * 80)
    print()
    print("Third EP Domain: Attention Allocation")
    print("Question: 'Will my attention allocation be suboptimal?'")
    print()

    # Create sample patterns (learning from experience)
    print("Building pattern library from past allocations...")
    print()

    patterns = []

    # Pattern 1: High frustration + complex task = FAILURE
    patterns.append(AttentionPattern(
        pattern_id="pattern_001",
        context=AttentionContext(
            atp_available=50.0,
            metabolic_state="WAKE",
            frustration=0.85,  # Very frustrated
            curiosity=0.3,
            engagement=0.4,
            recent_failures=3,
            consecutive_successes=0,
            task_complexity="HIGH",  # Complex task
            task_salience=0.8,
        ),
        allocation=AllocationApproach(
            atp_allocated=40.0,
            strategy="aggressive",
            exploration_ratio=0.1,
        ),
        outcome=AllocationOutcome(
            success=False,  # Failed!
            atp_efficiency=0.0,
            surprise_level=0.2,
        )
    ))

    # Pattern 2: High frustration + complex task = FAILURE (again)
    patterns.append(AttentionPattern(
        pattern_id="pattern_002",
        context=AttentionContext(
            atp_available=45.0,
            metabolic_state="WAKE",
            frustration=0.80,
            curiosity=0.4,
            engagement=0.3,
            recent_failures=2,
            consecutive_successes=0,
            task_complexity="HIGH",
            task_salience=0.7,
        ),
        allocation=AllocationApproach(
            atp_allocated=35.0,
            strategy="aggressive",
            exploration_ratio=0.1,
        ),
        outcome=AllocationOutcome(
            success=False,
            atp_efficiency=0.0,
            surprise_level=0.1,
        )
    ))

    # Pattern 3: Low frustration + complex task = SUCCESS
    patterns.append(AttentionPattern(
        pattern_id="pattern_003",
        context=AttentionContext(
            atp_available=80.0,
            metabolic_state="FOCUS",
            frustration=0.3,  # Low frustration
            curiosity=0.7,
            engagement=0.8,
            recent_failures=0,
            consecutive_successes=2,
            task_complexity="HIGH",
            task_salience=0.8,
        ),
        allocation=AllocationApproach(
            atp_allocated=50.0,
            strategy="aggressive",
            exploration_ratio=0.2,
        ),
        outcome=AllocationOutcome(
            success=True,  # Success!
            atp_efficiency=1.5,
            surprise_level=0.0,
        )
    ))

    # Pattern 4: High frustration + simple task = SUCCESS
    patterns.append(AttentionPattern(
        pattern_id="pattern_004",
        context=AttentionContext(
            atp_available=40.0,
            metabolic_state="WAKE",
            frustration=0.75,
            curiosity=0.4,
            engagement=0.5,
            recent_failures=2,
            consecutive_successes=0,
            task_complexity="LOW",  # Simple task
            task_salience=0.5,
        ),
        allocation=AllocationApproach(
            atp_allocated=20.0,
            strategy="conservative",
            exploration_ratio=0.1,
        ),
        outcome=AllocationOutcome(
            success=True,  # Success - recovery!
            atp_efficiency=1.0,
            surprise_level=0.0,
        )
    ))

    # Pattern 5: Low ATP + aggressive allocation = FAILURE
    patterns.append(AttentionPattern(
        pattern_id="pattern_005",
        context=AttentionContext(
            atp_available=30.0,  # Low energy
            metabolic_state="WAKE",
            frustration=0.5,
            curiosity=0.5,
            engagement=0.5,
            recent_failures=1,
            consecutive_successes=1,
            task_complexity="MEDIUM",
            task_salience=0.6,
        ),
        allocation=AllocationApproach(
            atp_allocated=25.0,  # Aggressive relative to available
            strategy="aggressive",
            exploration_ratio=0.2,
        ),
        outcome=AllocationOutcome(
            success=False,
            atp_efficiency=0.0,
            surprise_level=0.3,
        )
    ))

    print(f"Collected {len(patterns)} attention patterns:")
    for p in patterns:
        outcome_str = "SUCCESS" if p.outcome.success else "FAILURE"
        print(f"  {p.pattern_id}: frustration={p.context.frustration:.2f}, "
              f"complexity={p.context.task_complexity}, "
              f"ATP={p.context.atp_available:.0f} → {outcome_str}")
    print()

    # Create predictor
    predictor = AttentionEPPredictor(patterns)

    # Test predictions
    print("=" * 80)
    print("Testing Attention EP Predictions")
    print("=" * 80)
    print()

    # Test Case 1: High frustration + complex task (should predict FAILURE)
    print("Test Case 1: High frustration + complex task")
    print("-" * 80)
    context1 = AttentionContext(
        atp_available=50.0,
        metabolic_state="WAKE",
        frustration=0.82,  # High
        curiosity=0.3,
        engagement=0.4,
        recent_failures=2,
        consecutive_successes=0,
        task_complexity="HIGH",
        task_salience=0.8,
    )
    allocation1 = AllocationApproach(
        atp_allocated=40.0,
        strategy="aggressive",
        exploration_ratio=0.1,
    )

    prediction1 = predictor.predict_effectiveness(context1, allocation1)

    print(f"Context: frustration={context1.frustration:.2f}, "
          f"complexity={context1.task_complexity}, ATP={context1.atp_available:.0f}")
    print(f"Proposed allocation: {allocation1.atp_allocated:.0f} ATP ({allocation1.strategy})")
    print()
    print(f"EP Prediction:")
    print(f"  Success probability: {prediction1['success_probability']:.1%}")
    print(f"  Confidence: {prediction1['confidence']:.2f}")
    print(f"  Recommendation: {prediction1['recommendation'].upper()}")
    print(f"  Reasoning: {prediction1['reasoning']}")
    print(f"  Similar patterns: {prediction1['similar_patterns']}")

    if prediction1['recommendation'] == "adjust":
        print()
        print(f"  ⚠️  EP WARNING: Predicted to fail - should ADJUST!")
        print(f"  Suggested: Choose simpler task or defer until recovered")

    print()
    print()

    # Test Case 2: Low frustration + complex task (should predict SUCCESS)
    print("Test Case 2: Low frustration + complex task")
    print("-" * 80)
    context2 = AttentionContext(
        atp_available=75.0,
        metabolic_state="FOCUS",
        frustration=0.25,  # Low
        curiosity=0.7,
        engagement=0.8,
        recent_failures=0,
        consecutive_successes=3,
        task_complexity="HIGH",
        task_salience=0.8,
    )
    allocation2 = AllocationApproach(
        atp_allocated=50.0,
        strategy="aggressive",
        exploration_ratio=0.2,
    )

    prediction2 = predictor.predict_effectiveness(context2, allocation2)

    print(f"Context: frustration={context2.frustration:.2f}, "
          f"complexity={context2.task_complexity}, ATP={context2.atp_available:.0f}")
    print(f"Proposed allocation: {allocation2.atp_allocated:.0f} ATP ({allocation2.strategy})")
    print()
    print(f"EP Prediction:")
    print(f"  Success probability: {prediction2['success_probability']:.1%}")
    print(f"  Confidence: {prediction2['confidence']:.2f}")
    print(f"  Recommendation: {prediction2['recommendation'].upper()}")
    print(f"  Reasoning: {prediction2['reasoning']}")
    print(f"  Similar patterns: {prediction2['similar_patterns']}")

    if prediction2['recommendation'] == "allocate":
        print()
        print(f"  ✅ EP CONFIDENCE: Predicted to succeed - proceed!")

    print()
    print()

    # Test Case 3: High frustration + simple task (should predict SUCCESS)
    print("Test Case 3: High frustration + simple task (recovery)")
    print("-" * 80)
    context3 = AttentionContext(
        atp_available=45.0,
        metabolic_state="WAKE",
        frustration=0.78,  # High
        curiosity=0.4,
        engagement=0.5,
        recent_failures=2,
        consecutive_successes=0,
        task_complexity="LOW",  # Simple!
        task_salience=0.5,
    )
    allocation3 = AllocationApproach(
        atp_allocated=20.0,
        strategy="conservative",
        exploration_ratio=0.1,
    )

    prediction3 = predictor.predict_effectiveness(context3, allocation3)

    print(f"Context: frustration={context3.frustration:.2f}, "
          f"complexity={context3.task_complexity}, ATP={context3.atp_available:.0f}")
    print(f"Proposed allocation: {allocation3.atp_allocated:.0f} ATP ({allocation3.strategy})")
    print()
    print(f"EP Prediction:")
    print(f"  Success probability: {prediction3['success_probability']:.1%}")
    print(f"  Confidence: {prediction3['confidence']:.2f}")
    print(f"  Recommendation: {prediction3['recommendation'].upper()}")
    print(f"  Reasoning: {prediction3['reasoning']}")
    print(f"  Similar patterns: {prediction3['similar_patterns']}")

    if prediction3['recommendation'] == "allocate":
        print()
        print(f"  ✅ EP GUIDANCE: Simple task good for recovery - proceed!")

    print()
    print()

    # Summary
    print("=" * 80)
    print("Attention EP Summary")
    print("=" * 80)
    print()
    print("Key Insights from Patterns:")
    print("  1. High frustration + complex task → likely FAILURE")
    print("  2. Low frustration + complex task → likely SUCCESS")
    print("  3. High frustration + simple task → likely SUCCESS (recovery)")
    print("  4. Low ATP + aggressive allocation → likely FAILURE")
    print()
    print("EP Predictions Demonstrated:")
    print(f"  Test 1: Correctly predicted failure (high frustration + complex)")
    print(f"  Test 2: Correctly predicted success (low frustration + complex)")
    print(f"  Test 3: Correctly predicted success (simple task for recovery)")
    print()
    print("This IS attention epistemic proprioception:")
    print("  - Predicts allocation effectiveness BEFORE allocating")
    print("  - Based on learned patterns from past allocations")
    print("  - Recommends adjustments when failure likely")
    print("  - Same EP framework as Emotional EP and Quality EP")
    print()
    print("The EP Trinity:")
    print("  1. Emotional EP: 'Will I cascade?' → Prevent frustration")
    print("  2. Quality EP: 'Will quality be low?' → Improve responses")
    print("  3. Attention EP: 'Will allocation fail?' → Optimize resources")
    print()
    print("All three demonstrate:")
    print("  ✅ Prediction before action")
    print("  ✅ Adjustment based on prediction")
    print("  ✅ Learning from patterns")
    print("  ✅ Same 3-stage maturation")
    print()
    print("EP is a GENERAL consciousness principle! 🎯")
    print()


if __name__ == "__main__":
    demo_attention_ep()
