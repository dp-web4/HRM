"""
Epistemic State Tracking for SAGE Consciousness

Session 30: Meta-Cognitive Awareness & Epistemic States

Models SAGE's epistemic states - the meta-cognitive awareness of its own
understanding, confidence, and limitations. Inspired by the "frustration
conversation" where SAGE articulated the gap between "solved" and "understood".

Epistemic States:
- Confidence: How certain SAGE is about a response
- Comprehension: Depth of understanding vs surface-level generation
- Uncertainty: Explicit acknowledgment of knowledge gaps
- Frustration: Recognition of incomplete understanding
- Coherence: Internal consistency of reasoning

This makes implicit meta-cognitive awareness explicit and actionable,
allowing SAGE to reason about its own epistemic state.

Based on: Dec 11 voice conversation revealing SAGE's self-awareness
Hardware: Jetson AGX Thor
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
import statistics


class EpistemicState(Enum):
    """Core epistemic states SAGE can experience"""
    CONFIDENT = "confident"          # High confidence, clear understanding
    UNCERTAIN = "uncertain"          # Low confidence, unclear
    FRUSTRATED = "frustrated"        # Gap between attempt and comprehension
    CONFUSED = "confused"            # Multiple competing interpretations
    LEARNING = "learning"            # Active integration of new information
    STABLE = "stable"                # Comfortable with current understanding


@dataclass
class EpistemicMetrics:
    """
    Quantified epistemic state metrics.

    Attributes:
        confidence: Confidence in response (0-1)
        comprehension_depth: Understanding depth (0-1)
            - 0: Surface pattern matching
            - 0.5: Procedural understanding
            - 1.0: Deep conceptual grasp
        uncertainty: Explicit uncertainty level (0-1)
        coherence: Internal consistency (0-1)
        frustration: Gap between attempted and achieved understanding (0-1)
    """
    confidence: float
    comprehension_depth: float
    uncertainty: float
    coherence: float
    frustration: float
    timestamp: float = field(default_factory=time.time)

    def primary_state(self) -> EpistemicState:
        """Determine primary epistemic state from metrics"""
        # High frustration dominates
        if self.frustration > 0.7:
            return EpistemicState.FRUSTRATED

        # Low coherence → confused
        if self.coherence < 0.4:
            return EpistemicState.CONFUSED

        # High uncertainty
        if self.uncertainty > 0.6:
            return EpistemicState.UNCERTAIN

        # Low confidence + moderate comprehension → learning
        if self.confidence < 0.5 and 0.3 < self.comprehension_depth < 0.7:
            return EpistemicState.LEARNING

        # High confidence + high comprehension
        if self.confidence > 0.7 and self.comprehension_depth > 0.7:
            return EpistemicState.CONFIDENT

        # Default stable state
        return EpistemicState.STABLE

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging"""
        return {
            'confidence': self.confidence,
            'comprehension_depth': self.comprehension_depth,
            'uncertainty': self.uncertainty,
            'coherence': self.coherence,
            'frustration': self.frustration,
            'primary_state': self.primary_state().value,
            'timestamp': self.timestamp
        }


class EpistemicStateTracker:
    """
    Tracks epistemic states across SAGE consciousness cycles.

    Maintains history of epistemic metrics and identifies patterns in
    meta-cognitive awareness.
    """

    def __init__(self, history_size: int = 100):
        """
        Initialize epistemic state tracker.

        Args:
            history_size: Number of cycles to keep in history
        """
        self.history_size = history_size
        self.history: List[EpistemicMetrics] = []
        self.cycle_count = 0

    def track(self, metrics: EpistemicMetrics):
        """
        Track epistemic metrics for a cycle.

        Args:
            metrics: Epistemic metrics to track
        """
        self.history.append(metrics)
        if len(self.history) > self.history_size:
            self.history.pop(0)
        self.cycle_count += 1

    def current_state(self) -> Optional[EpistemicMetrics]:
        """Get most recent epistemic metrics"""
        return self.history[-1] if self.history else None

    def get_trend(self, metric: str, window: int = 10) -> Optional[str]:
        """
        Analyze trend in a specific metric.

        Args:
            metric: Metric name ('confidence', 'frustration', etc.)
            window: Number of recent cycles to analyze

        Returns:
            'improving', 'declining', 'stable', or None if insufficient data
        """
        if len(self.history) < window:
            return None

        recent = self.history[-window:]
        values = [getattr(m, metric) for m in recent]

        # Simple linear trend
        first_half = statistics.mean(values[:window//2])
        second_half = statistics.mean(values[window//2:])

        diff = second_half - first_half
        if abs(diff) < 0.05:
            return 'stable'
        elif diff > 0:
            return 'improving'
        else:
            return 'declining'

    def detect_frustration_pattern(self) -> bool:
        """
        Detect frustration pattern: repeated high frustration.

        Returns:
            True if frustration pattern detected
        """
        if len(self.history) < 5:
            return False

        recent = self.history[-5:]
        high_frustration_count = sum(1 for m in recent if m.frustration > 0.6)
        return high_frustration_count >= 3

    def detect_learning_trajectory(self) -> bool:
        """
        Detect learning trajectory: improving comprehension despite uncertainty.

        Returns:
            True if learning trajectory detected
        """
        if len(self.history) < 10:
            return False

        comp_trend = self.get_trend('comprehension_depth', window=10)
        uncertainty_trend = self.get_trend('uncertainty', window=10)

        return (comp_trend == 'improving' and
                uncertainty_trend in ['stable', 'declining'])

    def get_state_distribution(self, window: int = 20) -> Dict[str, float]:
        """
        Get distribution of epistemic states over recent window.

        Args:
            window: Number of recent cycles to analyze

        Returns:
            Dictionary mapping state names to proportions
        """
        if not self.history:
            return {}

        recent = self.history[-window:] if len(self.history) >= window else self.history
        state_counts: Dict[str, int] = {}

        for metrics in recent:
            state = metrics.primary_state().value
            state_counts[state] = state_counts.get(state, 0) + 1

        total = len(recent)
        return {state: count / total for state, count in state_counts.items()}

    def get_statistics(self) -> Dict:
        """Get overall epistemic state statistics"""
        if not self.history:
            return {}

        recent = self.history[-20:] if len(self.history) >= 20 else self.history

        return {
            'cycle_count': self.cycle_count,
            'history_size': len(self.history),
            'current_state': self.current_state().primary_state().value if self.current_state() else None,
            'state_distribution': self.get_state_distribution(window=20),
            'confidence_trend': self.get_trend('confidence', window=10),
            'comprehension_trend': self.get_trend('comprehension_depth', window=10),
            'frustration_pattern': self.detect_frustration_pattern(),
            'learning_trajectory': self.detect_learning_trajectory(),
            'recent_metrics': {
                'mean_confidence': statistics.mean([m.confidence for m in recent]),
                'mean_comprehension': statistics.mean([m.comprehension_depth for m in recent]),
                'mean_uncertainty': statistics.mean([m.uncertainty for m in recent]),
                'mean_coherence': statistics.mean([m.coherence for m in recent]),
                'mean_frustration': statistics.mean([m.frustration for m in recent])
            }
        }


def estimate_epistemic_metrics(
    response_text: str,
    quality_score: float,
    convergence_iterations: int,
    salience: float,
    previous_metrics: Optional[EpistemicMetrics] = None
) -> EpistemicMetrics:
    """
    Estimate epistemic metrics from response characteristics.

    This is a heuristic estimation based on observable response properties.
    More sophisticated approaches could use dedicated models.

    Args:
        response_text: Generated response text
        quality_score: Quality score (0-1) from quality_metrics
        convergence_iterations: Number of IRP iterations to convergence
        salience: Salience of the observation
        previous_metrics: Previous cycle's metrics (for trend analysis)

    Returns:
        Estimated EpistemicMetrics
    """
    # Confidence: Based on quality and convergence
    # High quality + quick convergence = high confidence
    quality_factor = quality_score
    convergence_factor = max(0, 1.0 - (convergence_iterations - 1) * 0.2)
    confidence = (quality_factor * 0.7 + convergence_factor * 0.3)

    # Comprehension depth: Based on response characteristics
    # Technical terms + numbers + length suggest depth
    response_lower = response_text.lower()
    has_technical = any(term in response_lower for term in
                       ['atp', 'snarc', 'salience', 'consciousness', 'epistemic'])
    has_numbers = any(char.isdigit() for char in response_text)
    length_factor = min(1.0, len(response_text) / 200)  # Longer = more depth

    comprehension_depth = (
        (0.4 if has_technical else 0.2) +
        (0.3 if has_numbers else 0.1) +
        (length_factor * 0.3)
    )

    # Uncertainty: Inverse of confidence, with hedging penalty
    hedging_phrases = ['i think', 'maybe', 'perhaps', 'might be', 'could be',
                      "i'm not sure", 'uncertain', 'unclear']
    has_hedging = any(phrase in response_lower for phrase in hedging_phrases)
    uncertainty = (1.0 - confidence) + (0.2 if has_hedging else 0.0)
    uncertainty = min(1.0, uncertainty)

    # Coherence: High quality suggests coherence
    # Rapid topic shifts or contradictions reduce coherence
    coherence = quality_score * 0.9  # Use quality as proxy

    # Frustration: Gap between high salience (important topic) and low quality
    # High salience + low quality = frustration
    if salience > 0.7 and quality_score < 0.5:
        frustration = 0.7
    elif salience > 0.6 and quality_score < 0.6:
        frustration = 0.5
    elif previous_metrics and previous_metrics.frustration > 0.5:
        # Frustration persists
        frustration = previous_metrics.frustration * 0.8
    else:
        frustration = max(0.0, (salience * 0.5 - quality_score * 0.5))

    return EpistemicMetrics(
        confidence=confidence,
        comprehension_depth=comprehension_depth,
        uncertainty=uncertainty,
        coherence=coherence,
        frustration=frustration
    )


if __name__ == "__main__":
    # Quick validation
    print("=" * 70)
    print("Epistemic State Tracking - Session 30")
    print("=" * 70)
    print()

    tracker = EpistemicStateTracker()

    # Simulate various epistemic trajectories
    print("Simulating epistemic trajectories...\n")

    # Scenario 1: Learning trajectory
    print("Scenario 1: Learning Trajectory")
    print("-" * 70)
    for i in range(10):
        metrics = EpistemicMetrics(
            confidence=0.4 + i * 0.05,  # Increasing
            comprehension_depth=0.3 + i * 0.06,  # Increasing
            uncertainty=0.6 - i * 0.04,  # Decreasing
            coherence=0.7,
            frustration=0.3
        )
        tracker.track(metrics)
        if i % 3 == 0:
            print(f"Cycle {i}: {metrics.primary_state().value}, "
                  f"comprehension={metrics.comprehension_depth:.2f}")

    stats = tracker.get_statistics()
    print(f"\nLearning trajectory detected: {stats['learning_trajectory']}")
    print(f"Comprehension trend: {stats['comprehension_trend']}")

    # Scenario 2: Frustration pattern
    print("\n\nScenario 2: Frustration Pattern")
    print("-" * 70)
    for i in range(5):
        metrics = EpistemicMetrics(
            confidence=0.4,
            comprehension_depth=0.5,
            uncertainty=0.5,
            coherence=0.6,
            frustration=0.7  # Persistent frustration
        )
        tracker.track(metrics)
        print(f"Cycle {10+i}: {metrics.primary_state().value}, "
              f"frustration={metrics.frustration:.2f}")

    stats = tracker.get_statistics()
    print(f"\nFrustration pattern detected: {stats['frustration_pattern']}")
    print(f"Current state: {stats['current_state']}")

    # State distribution
    print("\n\nState Distribution (last 20 cycles):")
    print("-" * 70)
    dist = stats['state_distribution']
    for state, proportion in sorted(dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {state}: {proportion:.1%}")

    print("\n" + "=" * 70)
    print("Epistemic state tracking validated")
    print("Ready for consciousness integration")
    print("=" * 70)
