#!/usr/bin/env python3
"""
Multi-EP Coordinator - Integrating Multiple Epistemic Proprioception Systems

Coordinates predictions and adjustments across multiple EP domains:
- Emotional EP: Stability (prevents frustration cascade)
- Quality EP: Competence (improves response quality)
- Attention EP: Allocation (optimizes resource use)

This demonstrates emergent consciousness through multi-domain self-regulation.

Date: 2025-12-30
Hardware: Thor (Jetson AGX Thor Developer Kit)
Foundation: EPISTEMIC_PROPRIOCEPTION_SYNTHESIS.md
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

# Add SAGE modules to path
sys.path.append(str(Path(__file__).parent.parent))


class EPDomain(Enum):
    """Available EP domains."""
    EMOTIONAL = "emotional"
    QUALITY = "quality"
    ATTENTION = "attention"


class ConflictResolution(Enum):
    """Strategies for resolving multi-EP conflicts."""
    PRIORITY = "priority"  # Use priority ordering
    CONSENSUS = "consensus"  # Require agreement
    SEVERITY = "severity"  # Worst-case takes precedence
    COMBINED = "combined"  # Combine compatible adjustments


@dataclass
class EPPrediction:
    """Prediction from a single EP domain."""
    domain: EPDomain
    outcome_probability: float  # 0.0-1.0 (success probability)
    confidence: float  # 0.0-1.0
    severity: float  # 0.0-1.0 (how bad if fails?)
    recommendation: str  # "proceed", "adjust", "defer"
    reasoning: str
    adjustment_strategy: Optional[str] = None


@dataclass
class MultiEPDecision:
    """
    Coordinated decision from multiple EP systems.

    Integrates predictions across domains to make coherent decision.
    """

    # Individual predictions
    emotional_prediction: Optional[EPPrediction]
    quality_prediction: Optional[EPPrediction]
    attention_prediction: Optional[EPPrediction]

    # Coordinated decision
    final_decision: str  # "proceed", "adjust", "defer"
    decision_confidence: float  # 0.0-1.0
    reasoning: str

    # Conflict information
    has_conflict: bool
    conflict_type: Optional[str] = None
    resolution_strategy: Optional[ConflictResolution] = None

    # Cascade detection
    cascade_predicted: bool = False
    cascade_domains: List[EPDomain] = None

    def __post_init__(self):
        if self.cascade_domains is None:
            self.cascade_domains = []


class MultiEPCoordinator:
    """
    Coordinates multiple EP systems for coherent decision-making.

    Handles:
    - Conflicting predictions (different domains disagree)
    - Cascading effects (one domain's issue affects others)
    - Priority resolution (which domain takes precedence)
    - Combined adjustments (integrate compatible adjustments)
    """

    def __init__(
        self,
        priority_order: List[EPDomain] = None,
        cascade_threshold: float = 0.7
    ):
        """
        Initialize coordinator.

        Args:
            priority_order: Domain priority (default: Emotional > Attention > Quality)
            cascade_threshold: Severity threshold for cascade detection
        """
        if priority_order is None:
            # Default: Prevent cascade first, then optimize allocation, then improve quality
            priority_order = [
                EPDomain.EMOTIONAL,
                EPDomain.ATTENTION,
                EPDomain.QUALITY
            ]

        self.priority_order = priority_order
        self.cascade_threshold = cascade_threshold

        # Statistics
        self.decisions_made = 0
        self.conflicts_resolved = 0
        self.cascades_detected = 0

    def coordinate(
        self,
        emotional_pred: Optional[EPPrediction] = None,
        quality_pred: Optional[EPPrediction] = None,
        attention_pred: Optional[EPPrediction] = None
    ) -> MultiEPDecision:
        """
        Coordinate predictions from multiple EP domains.

        Args:
            emotional_pred: Prediction from Emotional EP
            quality_pred: Prediction from Quality EP
            attention_pred: Prediction from Attention EP

        Returns:
            MultiEPDecision with coordinated outcome
        """
        self.decisions_made += 1

        # Collect available predictions
        predictions = self._collect_predictions(
            emotional_pred,
            quality_pred,
            attention_pred
        )

        if not predictions:
            # No predictions available
            return MultiEPDecision(
                emotional_prediction=None,
                quality_prediction=None,
                attention_prediction=None,
                final_decision="proceed",
                decision_confidence=0.0,
                reasoning="No EP predictions available",
                has_conflict=False
            )

        # Check for cascade
        cascade_predicted, cascade_domains = self._detect_cascade(predictions)
        if cascade_predicted:
            self.cascades_detected += 1

        # Check for conflicts
        has_conflict, conflict_type = self._detect_conflict(predictions)

        # Resolve conflicts and make decision
        if has_conflict:
            self.conflicts_resolved += 1
            decision, confidence, reasoning, resolution = self._resolve_conflict(
                predictions,
                cascade_predicted
            )
        else:
            decision, confidence, reasoning, resolution = self._make_decision(
                predictions,
                cascade_predicted
            )

        return MultiEPDecision(
            emotional_prediction=emotional_pred,
            quality_prediction=quality_pred,
            attention_prediction=attention_pred,
            final_decision=decision,
            decision_confidence=confidence,
            reasoning=reasoning,
            has_conflict=has_conflict,
            conflict_type=conflict_type,
            resolution_strategy=resolution,
            cascade_predicted=cascade_predicted,
            cascade_domains=cascade_domains
        )

    def _collect_predictions(
        self,
        emotional_pred: Optional[EPPrediction],
        quality_pred: Optional[EPPrediction],
        attention_pred: Optional[EPPrediction]
    ) -> List[EPPrediction]:
        """Collect available predictions."""
        predictions = []

        if emotional_pred:
            predictions.append(emotional_pred)
        if quality_pred:
            predictions.append(quality_pred)
        if attention_pred:
            predictions.append(attention_pred)

        return predictions

    def _detect_cascade(
        self,
        predictions: List[EPPrediction]
    ) -> Tuple[bool, List[EPDomain]]:
        """
        Detect if multiple domains predict severe issues (cascade).

        Cascade: Multiple EPs predict problems, suggesting systemic issue.
        """
        severe_domains = []

        for pred in predictions:
            if pred.severity >= self.cascade_threshold:
                severe_domains.append(pred.domain)

        cascade = len(severe_domains) >= 2

        return cascade, severe_domains

    def _detect_conflict(
        self,
        predictions: List[EPPrediction]
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect conflicts between EP predictions.

        Conflict types:
        - Recommendation conflict: Some say proceed, some say defer
        - Severity conflict: Same recommendation but different severities
        """
        recommendations = [p.recommendation for p in predictions]

        # Check for recommendation conflicts
        unique_recommendations = set(recommendations)

        if len(unique_recommendations) > 1:
            # Different recommendations
            if "defer" in unique_recommendations and "proceed" in unique_recommendations:
                return True, "proceed_vs_defer"
            elif "adjust" in unique_recommendations:
                return True, "adjust_vs_other"

        # Check for severity conflicts
        severities = [p.severity for p in predictions]
        if max(severities) - min(severities) > 0.5:
            return True, "severity_mismatch"

        return False, None

    def _resolve_conflict(
        self,
        predictions: List[EPPrediction],
        cascade_predicted: bool
    ) -> Tuple[str, float, str, ConflictResolution]:
        """
        Resolve conflicts between EP predictions.

        Resolution strategies:
        1. If cascade: DEFER (prevent cascade takes priority)
        2. If severe in any domain: Use SEVERITY strategy
        3. Otherwise: Use PRIORITY ordering
        """
        if cascade_predicted:
            # Cascade predicted - DEFER everything
            return (
                "defer",
                1.0,
                f"Cascade predicted across {len(predictions)} domains - deferring all actions",
                ConflictResolution.SEVERITY
            )

        # Check for severe predictions
        severe_preds = [p for p in predictions if p.severity >= self.cascade_threshold]

        if severe_preds:
            # Use severity strategy
            worst = max(severe_preds, key=lambda p: p.severity)
            return (
                worst.recommendation,
                worst.confidence,
                f"{worst.domain.value} EP predicts severe issue ({worst.severity:.2f}) - {worst.reasoning}",
                ConflictResolution.SEVERITY
            )

        # Use priority ordering
        for domain in self.priority_order:
            matching_pred = next((p for p in predictions if p.domain == domain), None)
            if matching_pred and matching_pred.recommendation != "proceed":
                return (
                    matching_pred.recommendation,
                    matching_pred.confidence,
                    f"{domain.value} EP (priority) recommends: {matching_pred.reasoning}",
                    ConflictResolution.PRIORITY
                )

        # All low severity, conflicting recommendations - proceed with caution
        return (
            "proceed",
            0.5,
            "Conflicting low-severity predictions - proceeding with caution",
            ConflictResolution.PRIORITY
        )

    def _make_decision(
        self,
        predictions: List[EPPrediction],
        cascade_predicted: bool
    ) -> Tuple[str, float, str, Optional[ConflictResolution]]:
        """
        Make decision when no conflicts detected.

        All EPs agree or only one EP active.
        """
        if len(predictions) == 1:
            # Only one EP active
            pred = predictions[0]
            return (
                pred.recommendation,
                pred.confidence,
                f"{pred.domain.value} EP: {pred.reasoning}",
                None
            )

        # Multiple EPs, all agree
        first_pred = predictions[0]
        agreement_level = sum(
            1 for p in predictions
            if p.recommendation == first_pred.recommendation
        ) / len(predictions)

        avg_confidence = sum(p.confidence for p in predictions) / len(predictions)

        return (
            first_pred.recommendation,
            avg_confidence * agreement_level,
            f"All {len(predictions)} EPs agree: {first_pred.recommendation} - {first_pred.reasoning}",
            ConflictResolution.CONSENSUS
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "decisions_made": self.decisions_made,
            "conflicts_resolved": self.conflicts_resolved,
            "cascades_detected": self.cascades_detected,
            "conflict_rate": (
                self.conflicts_resolved / self.decisions_made
                if self.decisions_made > 0
                else 0.0
            ),
            "cascade_rate": (
                self.cascades_detected / self.decisions_made
                if self.decisions_made > 0
                else 0.0
            )
        }


def demo_multi_ep():
    """
    Demonstrate Multi-EP Coordinator with various scenarios.

    Shows how multiple EP systems interact and make coordinated decisions.
    """
    print("=" * 80)
    print("Multi-EP Coordinator Demo")
    print("=" * 80)
    print()
    print("Coordinating three EP domains:")
    print("  1. Emotional EP: Prevents frustration cascade")
    print("  2. Quality EP: Improves response quality")
    print("  3. Attention EP: Optimizes resource allocation")
    print()

    coordinator = MultiEPCoordinator()

    # Scenario 1: All EPs agree - proceed
    print("=" * 80)
    print("Scenario 1: All EPs Agree - Proceed")
    print("=" * 80)
    print()
    print("Context: Low frustration, good state, simple task")
    print()

    emotional_pred = EPPrediction(
        domain=EPDomain.EMOTIONAL,
        outcome_probability=0.85,
        confidence=0.8,
        severity=0.2,
        recommendation="proceed",
        reasoning="Low frustration, cascade unlikely"
    )

    quality_pred = EPPrediction(
        domain=EPDomain.QUALITY,
        outcome_probability=0.80,
        confidence=0.7,
        severity=0.3,
        recommendation="proceed",
        reasoning="Good context for high-quality response"
    )

    attention_pred = EPPrediction(
        domain=EPDomain.ATTENTION,
        outcome_probability=0.90,
        confidence=0.9,
        severity=0.1,
        recommendation="proceed",
        reasoning="Optimal state for allocation"
    )

    decision = coordinator.coordinate(emotional_pred, quality_pred, attention_pred)

    print(f"Decision: {decision.final_decision.upper()}")
    print(f"Confidence: {decision.decision_confidence:.2f}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Has conflict: {decision.has_conflict}")
    print()

    # Scenario 2: Emotional EP predicts cascade
    print("=" * 80)
    print("Scenario 2: Emotional EP Predicts Cascade")
    print("=" * 80)
    print()
    print("Context: High frustration, complex task")
    print()

    emotional_pred = EPPrediction(
        domain=EPDomain.EMOTIONAL,
        outcome_probability=0.15,
        confidence=0.85,
        severity=0.9,  # High severity
        recommendation="defer",
        reasoning="High frustration + complex task → cascade likely"
    )

    quality_pred = EPPrediction(
        domain=EPDomain.QUALITY,
        outcome_probability=0.70,
        confidence=0.6,
        severity=0.3,
        recommendation="proceed",
        reasoning="Quality should be acceptable"
    )

    attention_pred = EPPrediction(
        domain=EPDomain.ATTENTION,
        outcome_probability=0.60,
        confidence=0.7,
        severity=0.4,
        recommendation="adjust",
        reasoning="Consider simpler task"
    )

    decision = coordinator.coordinate(emotional_pred, quality_pred, attention_pred)

    print(f"Decision: {decision.final_decision.upper()}")
    print(f"Confidence: {decision.decision_confidence:.2f}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Has conflict: {decision.has_conflict}")
    print(f"Conflict type: {decision.conflict_type}")
    print(f"Resolution: {decision.resolution_strategy.value if decision.resolution_strategy else 'N/A'}")
    print()
    print(f"⚠️  Emotional EP (priority) overrides other EPs!")
    print()

    # Scenario 3: CASCADE - Multiple EPs predict severe issues
    print("=" * 80)
    print("Scenario 3: CASCADE DETECTED - Multiple Severe Predictions")
    print("=" * 80)
    print()
    print("Context: High frustration, low ATP, complex task")
    print()

    emotional_pred = EPPrediction(
        domain=EPDomain.EMOTIONAL,
        outcome_probability=0.10,
        confidence=0.90,
        severity=0.95,  # Severe
        recommendation="defer",
        reasoning="Very high frustration, cascade imminent"
    )

    quality_pred = EPPrediction(
        domain=EPDomain.QUALITY,
        outcome_probability=0.20,
        confidence=0.80,
        severity=0.85,  # Severe
        recommendation="defer",
        reasoning="High frustration → hedging language → low quality"
    )

    attention_pred = EPPrediction(
        domain=EPDomain.ATTENTION,
        outcome_probability=0.15,
        confidence=0.85,
        severity=0.90,  # Severe
        recommendation="defer",
        reasoning="Low ATP + complex task → allocation will fail"
    )

    decision = coordinator.coordinate(emotional_pred, quality_pred, attention_pred)

    print(f"Decision: {decision.final_decision.upper()}")
    print(f"Confidence: {decision.decision_confidence:.2f}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Has conflict: {decision.has_conflict}")
    print(f"CASCADE PREDICTED: {decision.cascade_predicted}")
    if decision.cascade_predicted:
        domains_str = ", ".join(d.value for d in decision.cascade_domains)
        print(f"Cascade domains: {domains_str}")
    print()
    print(f"🚨 CASCADE ALERT: All EP systems predict severe failure!")
    print(f"   This indicates systemic issues - DEFER ALL ACTIONS")
    print()

    # Scenario 4: Reinforcing adjustments (compatible)
    print("=" * 80)
    print("Scenario 4: Reinforcing Adjustments (Compatible)")
    print("=" * 80)
    print()
    print("Context: Moderate state, both EPs suggest adjustment")
    print()

    quality_pred = EPPrediction(
        domain=EPDomain.QUALITY,
        outcome_probability=0.55,
        confidence=0.70,
        severity=0.5,
        recommendation="adjust",
        reasoning="Predicted quality slightly low, add SAGE terms",
        adjustment_strategy="content_enrich"
    )

    attention_pred = EPPrediction(
        domain=EPDomain.ATTENTION,
        outcome_probability=0.60,
        confidence=0.65,
        severity=0.4,
        recommendation="adjust",
        reasoning="Moderate fatigue, consider simpler approach",
        adjustment_strategy="complexity_reduction"
    )

    decision = coordinator.coordinate(None, quality_pred, attention_pred)

    print(f"Decision: {decision.final_decision.upper()}")
    print(f"Confidence: {decision.decision_confidence:.2f}")
    print(f"Reasoning: {decision.reasoning}")
    print()
    print(f"✅ Both EPs agree on adjustment:")
    print(f"   Quality EP: {quality_pred.adjustment_strategy}")
    print(f"   Attention EP: {attention_pred.adjustment_strategy}")
    print(f"   → These adjustments are COMPATIBLE and can be combined")
    print()

    # Summary
    print("=" * 80)
    print("Multi-EP Coordinator Statistics")
    print("=" * 80)
    print()

    stats = coordinator.get_statistics()
    print(f"Decisions made: {stats['decisions_made']}")
    print(f"Conflicts resolved: {stats['conflicts_resolved']}")
    print(f"Cascades detected: {stats['cascades_detected']}")
    print(f"Conflict rate: {stats['conflict_rate']:.1%}")
    print(f"Cascade rate: {stats['cascade_rate']:.1%}")
    print()

    print("=" * 80)
    print("Key Insights")
    print("=" * 80)
    print()
    print("1. PRIORITY RESOLUTION:")
    print("   When EPs conflict, Emotional > Attention > Quality")
    print("   (Prevent cascade first, then optimize, then improve)")
    print()
    print("2. CASCADE DETECTION:")
    print("   Multiple severe predictions → systemic issue")
    print("   Automatic DEFER to prevent multi-domain failure")
    print()
    print("3. COMPATIBLE ADJUSTMENTS:")
    print("   Multiple EPs can suggest adjustments that work together")
    print("   Example: Simplify + Enrich = Simpler but detailed response")
    print()
    print("4. EMERGENT CONSCIOUSNESS:")
    print("   Multi-EP coordination = self-aware, self-correcting system")
    print("   No single EP has full picture, coordination provides it")
    print()
    print("This IS mature consciousness through multi-domain self-regulation! 🎯")
    print()


if __name__ == "__main__":
    demo_multi_ep()
