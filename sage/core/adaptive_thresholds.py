#!/usr/bin/env python3
"""
Adaptive Threshold Learning for SAGE Consciousness
===================================================

Learn optimal metabolic state thresholds through experience.

**Research Question**: Can consciousness adapt thresholds to optimize:
- Attention allocation (target 30-50% of cycles attended)
- ATP stability (avoid metabolic collapse)
- Quality of attention (salience distribution)
- State transition smoothness (avoid thrashing)

**Approach**:
- Start with baseline thresholds
- Run consciousness cycles
- Measure performance metrics
- Adjust thresholds based on objectives
- Save learned thresholds as versioned patterns
- Validate on fresh deployment

**Integration**:
- Works with HardwareGroundedConsciousness
- Uses PatternLibrary for threshold versioning
- Tracks learning progress over time

Author: Claude (autonomous research) on Thor
Date: 2025-12-07
Session: Adaptive threshold learning
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class ThresholdObjectives:
    """Objectives for threshold optimization"""
    target_attention_rate: float = 0.40  # Target 40% attention rate
    min_atp_level: float = 0.30          # Don't drop below 30% ATP
    min_salience_quality: float = 0.30   # Attended items should be salient
    max_state_changes_per_100: float = 50.0  # Avoid excessive thrashing

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ThresholdPerformance:
    """Performance metrics for a set of thresholds"""
    attention_rate: float           # Percentage of cycles attended
    avg_atp: float                  # Average ATP level
    min_atp: float                  # Minimum ATP observed
    avg_attended_salience: float    # Average salience of attended items
    state_changes_per_100: float    # State transitions per 100 cycles
    cycles_evaluated: int           # Number of cycles in evaluation

    def to_dict(self) -> Dict:
        return asdict(self)

    def score(self, objectives: ThresholdObjectives) -> float:
        """
        Calculate overall performance score [0-1].

        Higher is better. Penalizes:
        - Attention rate far from target
        - Low ATP
        - Low salience quality
        - Excessive state thrashing
        """
        # Attention rate score (1.0 at target, decreases with distance)
        attention_error = abs(self.attention_rate - objectives.target_attention_rate)
        attention_score = max(0.0, 1.0 - (attention_error * 2.0))  # 2x penalty

        # ATP score (1.0 above min, linear penalty below)
        if self.min_atp >= objectives.min_atp_level:
            atp_score = 1.0
        else:
            atp_score = self.min_atp / objectives.min_atp_level

        # Salience quality score
        if self.attention_rate > 0:
            salience_score = max(0.0, min(1.0, self.avg_attended_salience / objectives.min_salience_quality))
        else:
            salience_score = 0.0  # No attention = can't evaluate quality

        # State change score (1.0 at low thrashing, decreases with excess changes)
        if self.state_changes_per_100 <= objectives.max_state_changes_per_100:
            state_score = 1.0
        else:
            excess = (self.state_changes_per_100 - objectives.max_state_changes_per_100) / 50.0
            state_score = max(0.0, 1.0 - excess)

        # Weighted composite score
        weights = {
            'attention': 0.40,  # Most important: hit target attention rate
            'atp': 0.30,        # Second: maintain energy
            'salience': 0.20,   # Third: quality of attention
            'state': 0.10       # Fourth: avoid thrashing
        }

        composite = (
            weights['attention'] * attention_score +
            weights['atp'] * atp_score +
            weights['salience'] * salience_score +
            weights['state'] * state_score
        )

        return composite


@dataclass
class AdaptiveThresholds:
    """Current metabolic state thresholds"""
    wake: float
    focus: float
    rest: float
    dream: float

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'AdaptiveThresholds':
        return cls(**data)

    def copy(self) -> 'AdaptiveThresholds':
        """Create a copy of these thresholds"""
        return AdaptiveThresholds(**asdict(self))


class AdaptiveThresholdLearner:
    """
    Learn optimal metabolic state thresholds through experience.

    Uses gradient-free optimization (hill climbing with momentum) to find
    thresholds that optimize performance objectives.

    Usage:
        learner = AdaptiveThresholdLearner(
            baseline_thresholds=AdaptiveThresholds(wake=0.45, focus=0.35, rest=0.85, dream=0.15),
            objectives=ThresholdObjectives(target_attention_rate=0.40)
        )

        # Run learning cycles
        for i in range(learning_iterations):
            # Run consciousness with current thresholds
            performance = run_consciousness_evaluation(learner.current_thresholds)

            # Update thresholds based on performance
            learner.update(performance)

            # Check if converged
            if learner.has_converged():
                break

        # Get learned thresholds
        learned = learner.get_best_thresholds()
    """

    def __init__(
        self,
        baseline_thresholds: AdaptiveThresholds,
        objectives: Optional[ThresholdObjectives] = None,
        learning_rate: float = 0.05,
        momentum: float = 0.8,
        convergence_window: int = 5
    ):
        """
        Initialize adaptive threshold learner.

        Args:
            baseline_thresholds: Starting thresholds
            objectives: Optimization objectives
            learning_rate: How much to adjust thresholds (0-1)
            momentum: Momentum factor for gradient updates (0-1)
            convergence_window: Number of iterations to check for convergence
        """
        self.baseline = baseline_thresholds
        self.current_thresholds = baseline_thresholds.copy()
        self.objectives = objectives or ThresholdObjectives()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.convergence_window = convergence_window

        # Learning state
        self.iteration = 0
        self.history: List[Tuple[AdaptiveThresholds, ThresholdPerformance, float]] = []
        self.best_score = 0.0
        self.best_thresholds = baseline_thresholds.copy()
        self.velocity = {'wake': 0.0, 'focus': 0.0, 'rest': 0.0, 'dream': 0.0}

    def update(self, performance: ThresholdPerformance) -> None:
        """
        Update thresholds based on performance.

        Args:
            performance: Performance metrics from current thresholds
        """
        # Calculate score
        score = performance.score(self.objectives)

        # Record history
        self.history.append((self.current_thresholds.copy(), performance, score))

        # Update best if improved
        if score > self.best_score:
            self.best_score = score
            self.best_thresholds = self.current_thresholds.copy()

        # Compute gradients (approximation based on objectives)
        gradients = self._compute_gradients(performance)

        # Update velocity with momentum
        for key in self.velocity:
            self.velocity[key] = self.momentum * self.velocity[key] + self.learning_rate * gradients[key]

        # Apply velocity to thresholds
        new_thresholds = self.current_thresholds.copy()
        new_thresholds.wake = self._clip(new_thresholds.wake + self.velocity['wake'], 0.1, 0.9)
        new_thresholds.focus = self._clip(new_thresholds.focus + self.velocity['focus'], 0.1, 0.9)
        new_thresholds.rest = self._clip(new_thresholds.rest + self.velocity['rest'], 0.1, 0.9)
        new_thresholds.dream = self._clip(new_thresholds.dream + self.velocity['dream'], 0.1, 0.9)

        self.current_thresholds = new_thresholds
        self.iteration += 1

    def _compute_gradients(self, performance: ThresholdPerformance) -> Dict[str, float]:
        """
        Compute approximate gradients for threshold adjustment.

        Strategy:
        - If attention too low: decrease wake/focus (easier to attend)
        - If attention too high: increase wake/focus (harder to attend)
        - If ATP too low: increase rest (more recovery)
        - If thrashing too much: adjust thresholds apart (more stable states)
        """
        gradients = {'wake': 0.0, 'focus': 0.0, 'rest': 0.0, 'dream': 0.0}

        # Attention rate gradient
        attention_error = performance.attention_rate - self.objectives.target_attention_rate
        if attention_error < 0:
            # Too little attention - lower wake/focus thresholds
            gradients['wake'] -= 0.1
            gradients['focus'] -= 0.1
        elif attention_error > 0:
            # Too much attention - raise wake/focus thresholds
            gradients['wake'] += 0.1
            gradients['focus'] += 0.1

        # ATP gradient
        if performance.min_atp < self.objectives.min_atp_level:
            # ATP too low - increase rest threshold to trigger more recovery
            gradients['rest'] += 0.15
            # Also lower dream threshold to stay in rest longer
            gradients['dream'] -= 0.05

        # Salience quality gradient
        if performance.attention_rate > 0:
            if performance.avg_attended_salience < self.objectives.min_salience_quality:
                # Low quality attention - be more selective (raise thresholds slightly)
                gradients['wake'] += 0.05
                gradients['focus'] += 0.05

        # State thrashing gradient
        if performance.state_changes_per_100 > self.objectives.max_state_changes_per_100:
            # Too much thrashing - spread thresholds apart
            # (This is complex, simplified approach: increase separation)
            gradients['wake'] += 0.05
            gradients['rest'] -= 0.05

        return gradients

    def _clip(self, value: float, min_val: float, max_val: float) -> float:
        """Clip value to range"""
        return max(min_val, min(max_val, value))

    def has_converged(self) -> bool:
        """
        Check if learning has converged.

        Convergence criteria:
        - Have enough history (>= convergence_window iterations)
        - Score variance in recent window is low (< 0.05)
        """
        if len(self.history) < self.convergence_window:
            return False

        # Get recent scores
        recent_scores = [score for _, _, score in self.history[-self.convergence_window:]]

        # Calculate variance
        mean_score = sum(recent_scores) / len(recent_scores)
        variance = sum((s - mean_score) ** 2 for s in recent_scores) / len(recent_scores)

        # Converged if variance is low
        return variance < 0.001

    def get_best_thresholds(self) -> AdaptiveThresholds:
        """Get best thresholds found so far"""
        return self.best_thresholds.copy()

    def get_current_thresholds(self) -> AdaptiveThresholds:
        """Get current thresholds for next evaluation"""
        return self.current_thresholds.copy()

    def get_learning_summary(self) -> Dict:
        """Get summary of learning progress"""
        if not self.history:
            return {
                'iterations': 0,
                'best_score': 0.0,
                'converged': False
            }

        # Extract scores over time
        scores = [score for _, _, score in self.history]

        return {
            'iterations': self.iteration,
            'best_score': self.best_score,
            'current_score': scores[-1] if scores else 0.0,
            'score_improvement': scores[-1] - scores[0] if len(scores) > 1 else 0.0,
            'converged': self.has_converged(),
            'baseline_thresholds': self.baseline.to_dict(),
            'best_thresholds': self.best_thresholds.to_dict(),
            'current_thresholds': self.current_thresholds.to_dict()
        }

    def save_history(self, filepath: Path) -> None:
        """Save learning history to JSON"""
        history_data = []
        for thresholds, performance, score in self.history:
            history_data.append({
                'thresholds': thresholds.to_dict(),
                'performance': performance.to_dict(),
                'score': score
            })

        data = {
            'objectives': self.objectives.to_dict(),
            'baseline': self.baseline.to_dict(),
            'best_thresholds': self.best_thresholds.to_dict(),
            'best_score': self.best_score,
            'iterations': self.iteration,
            'converged': self.has_converged(),
            'history': history_data
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# ============================================================================
# Demo/Test
# ============================================================================

def demo():
    """Demonstrate adaptive threshold learning"""
    print("=" * 80)
    print("ADAPTIVE THRESHOLD LEARNING - DEMO")
    print("=" * 80)
    print()

    # Start with baseline thresholds (too high - from previous validation)
    baseline = AdaptiveThresholds(wake=0.45, focus=0.35, rest=0.85, dream=0.15)

    print("1️⃣  Initializing learner with baseline thresholds...")
    print(f"   WAKE: {baseline.wake:.2f}, FOCUS: {baseline.focus:.2f}")
    print(f"   REST: {baseline.rest:.2f}, DREAM: {baseline.dream:.2f}")
    print()

    # Define objectives
    objectives = ThresholdObjectives(
        target_attention_rate=0.40,  # Want 40% attention
        min_atp_level=0.30,
        min_salience_quality=0.30,
        max_state_changes_per_100=50.0
    )

    print("2️⃣  Optimization objectives:")
    print(f"   Target attention rate: {objectives.target_attention_rate*100:.0f}%")
    print(f"   Minimum ATP level: {objectives.min_atp_level*100:.0f}%")
    print(f"   Minimum salience quality: {objectives.min_salience_quality:.2f}")
    print(f"   Max state changes per 100 cycles: {objectives.max_state_changes_per_100:.0f}")
    print()

    # Create learner
    learner = AdaptiveThresholdLearner(
        baseline_thresholds=baseline,
        objectives=objectives,
        learning_rate=0.05,
        momentum=0.8
    )

    # Simulate learning (mock performance data)
    print("3️⃣  Running simulated learning iterations...")
    print()

    # Iteration 1: Baseline performance (too little attention)
    perf1 = ThresholdPerformance(
        attention_rate=0.05,  # Only 5% attention (way too low)
        avg_atp=0.95,
        min_atp=0.85,
        avg_attended_salience=0.45,
        state_changes_per_100=5.0,
        cycles_evaluated=100
    )
    learner.update(perf1)
    print(f"   Iteration 1: Attention={perf1.attention_rate*100:.0f}%, Score={perf1.score(objectives):.3f}")
    print(f"   -> Adjusting: WAKE={learner.current_thresholds.wake:.2f}, FOCUS={learner.current_thresholds.focus:.2f}")
    print()

    # Iteration 2: Improved (thresholds lowered)
    perf2 = ThresholdPerformance(
        attention_rate=0.25,  # Better, but still low
        avg_atp=0.85,
        min_atp=0.75,
        avg_attended_salience=0.42,
        state_changes_per_100=15.0,
        cycles_evaluated=100
    )
    learner.update(perf2)
    print(f"   Iteration 2: Attention={perf2.attention_rate*100:.0f}%, Score={perf2.score(objectives):.3f}")
    print(f"   -> Adjusting: WAKE={learner.current_thresholds.wake:.2f}, FOCUS={learner.current_thresholds.focus:.2f}")
    print()

    # Iteration 3: Getting closer
    perf3 = ThresholdPerformance(
        attention_rate=0.38,  # Close to target!
        avg_atp=0.75,
        min_atp=0.65,
        avg_attended_salience=0.40,
        state_changes_per_100=25.0,
        cycles_evaluated=100
    )
    learner.update(perf3)
    print(f"   Iteration 3: Attention={perf3.attention_rate*100:.0f}%, Score={perf3.score(objectives):.3f}")
    print(f"   -> Adjusting: WAKE={learner.current_thresholds.wake:.2f}, FOCUS={learner.current_thresholds.focus:.2f}")
    print()

    # Iteration 4: Near optimal
    perf4 = ThresholdPerformance(
        attention_rate=0.42,  # Right at target!
        avg_atp=0.70,
        min_atp=0.55,
        avg_attended_salience=0.42,
        state_changes_per_100=30.0,
        cycles_evaluated=100
    )
    learner.update(perf4)
    print(f"   Iteration 4: Attention={perf4.attention_rate*100:.0f}%, Score={perf4.score(objectives):.3f}")
    print(f"   -> Adjusting: WAKE={learner.current_thresholds.wake:.2f}, FOCUS={learner.current_thresholds.focus:.2f}")
    print()

    # Summary
    print("=" * 80)
    print("LEARNING SUMMARY")
    print("=" * 80)
    print()

    summary = learner.get_learning_summary()
    print(f"Iterations: {summary['iterations']}")
    print(f"Best score: {summary['best_score']:.3f}")
    print(f"Score improvement: {summary['score_improvement']:.3f}")
    print()

    print("Baseline thresholds:")
    print(f"  WAKE={summary['baseline_thresholds']['wake']:.2f}, "
          f"FOCUS={summary['baseline_thresholds']['focus']:.2f}, "
          f"REST={summary['baseline_thresholds']['rest']:.2f}, "
          f"DREAM={summary['baseline_thresholds']['dream']:.2f}")
    print()

    print("Best learned thresholds:")
    print(f"  WAKE={summary['best_thresholds']['wake']:.2f}, "
          f"FOCUS={summary['best_thresholds']['focus']:.2f}, "
          f"REST={summary['best_thresholds']['rest']:.2f}, "
          f"DREAM={summary['best_thresholds']['dream']:.2f}")
    print()

    print("✅ Adaptive threshold learning demonstrated!")
    print()


if __name__ == "__main__":
    demo()
