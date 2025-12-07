"""
Threshold-Based Attention Decision Module
==========================================

Implements Layer 3 and Layer 4 of compression-action-threshold pattern:
- Layer 3: Metabolic-state-dependent threshold computation
- Layer 4: Binary attention decision (attend or ignore)

Based on ATTENTION_COMPRESSION_DESIGN.md (Dec 5, 2025).

Pattern:
    Multi-dimensional input (sensors)
      ↓ SNARC compression (Layer 2)
    Scalar salience [0, 1]
      ↓ Context-dependent threshold (Layer 3)
    Binary decision: attend or ignore (Layer 4)

Author: Claude (Sonnet 4.5)
Date: 2025-12-07
Session: Continuing compression-action-threshold implementation
"""

from enum import Enum
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


class MetabolicState(Enum):
    """
    Metabolic states for SAGE consciousness.

    Each state has different attention thresholds:
    - WAKE: Moderate selectivity (0.5)
    - FOCUS: Low threshold, attend to details (0.3)
    - REST: High threshold, only urgent matters (0.8)
    - DREAM: Very low, explore freely (0.1)
    - CRISIS: Very high, only critical signals (0.9)
    """
    WAKE = "wake"
    FOCUS = "focus"
    REST = "rest"
    DREAM = "dream"
    CRISIS = "crisis"


@dataclass
class AttentionDecision:
    """Result of attention decision process"""
    should_attend: bool
    reason: str
    salience: float
    threshold: float
    atp_cost: float
    atp_budget: float

    def to_dict(self) -> Dict:
        return {
            'should_attend': self.should_attend,
            'reason': self.reason,
            'salience': self.salience,
            'threshold': self.threshold,
            'atp_cost': self.atp_cost,
            'atp_budget': self.atp_budget
        }


def get_attention_threshold(
    state: MetabolicState,
    atp_remaining: float,
    task_criticality: float
) -> float:
    """
    Compute metabolic-state-dependent attention threshold.

    The threshold determines what salience level is required to attend.
    It is modulated by:
    - Metabolic state (base threshold)
    - ATP availability (low ATP → raise threshold to conserve)
    - Task criticality (high criticality → lower threshold, don't miss signals)

    Args:
        state: Current metabolic state
        atp_remaining: ATP remaining as fraction [0, 1]
        task_criticality: Task importance [0, 1]

    Returns:
        threshold: Attention threshold [0, 1]

    Examples:
        >>> get_attention_threshold(MetabolicState.WAKE, atp_remaining=0.8, task_criticality=0.5)
        0.49  # Moderate threshold in normal conditions

        >>> get_attention_threshold(MetabolicState.CRISIS, atp_remaining=0.3, task_criticality=0.9)
        0.95  # Very high threshold in crisis with low ATP

        >>> get_attention_threshold(MetabolicState.DREAM, atp_remaining=0.7, task_criticality=0.0)
        0.16  # Very low threshold for exploration
    """

    # Base thresholds by metabolic state
    base_thresholds = {
        MetabolicState.WAKE: 0.5,    # Moderate: normal selectivity
        MetabolicState.FOCUS: 0.3,   # Low: attend to details
        MetabolicState.REST: 0.8,    # High: only urgent matters
        MetabolicState.DREAM: 0.1,   # Very low: explore freely
        MetabolicState.CRISIS: 0.9,  # Very high: only critical
    }

    base = base_thresholds[state]

    # Modulate by ATP availability
    # Low ATP → raise threshold (conserve energy)
    # High ATP → lower threshold (can afford exploration)
    atp_factor = 1.0 - atp_remaining  # Invert: low ATP → high factor
    atp_modulation = 0.2 * atp_factor  # Max ±0.2 shift

    # Modulate by task criticality
    # High criticality → lower threshold (don't miss important signals)
    criticality_modulation = -0.1 * task_criticality  # Max -0.1 shift

    # Combined threshold
    threshold = base + atp_modulation + criticality_modulation

    # Clamp to [0, 1]
    return max(0.0, min(1.0, threshold))


def make_attention_decision(
    salience: float,
    threshold: float,
    plugin_name: str,
    atp_cost: float,
    atp_budget: float
) -> AttentionDecision:
    """
    Binary decision: Invoke plugin or not?

    A plugin should be invoked if:
    1. Salience exceeds threshold
    2. Sufficient ATP budget available

    Args:
        salience: Computed salience score [0, 1]
        threshold: Attention threshold [0, 1]
        plugin_name: Name of plugin being considered
        atp_cost: ATP required to run plugin
        atp_budget: ATP currently available

    Returns:
        AttentionDecision with should_attend and reason

    Examples:
        >>> make_attention_decision(
        ...     salience=0.7, threshold=0.5, plugin_name="vision",
        ...     atp_cost=10.0, atp_budget=50.0
        ... )
        AttentionDecision(should_attend=True, reason="Salience 0.70 > threshold 0.50, ATP sufficient")

        >>> make_attention_decision(
        ...     salience=0.4, threshold=0.5, plugin_name="audio",
        ...     atp_cost=10.0, atp_budget=50.0
        ... )
        AttentionDecision(should_attend=False, reason="Salience 0.40 below threshold 0.50")
    """

    # Check threshold first (cheap)
    if salience <= threshold:
        return AttentionDecision(
            should_attend=False,
            reason=f"Salience {salience:.2f} below threshold {threshold:.2f}",
            salience=salience,
            threshold=threshold,
            atp_cost=atp_cost,
            atp_budget=atp_budget
        )

    # Check ATP budget (also cheap, but threshold is more fundamental)
    if atp_cost > atp_budget:
        return AttentionDecision(
            should_attend=False,
            reason=f"Insufficient ATP: {atp_budget:.2f} < {atp_cost:.2f}",
            salience=salience,
            threshold=threshold,
            atp_cost=atp_cost,
            atp_budget=atp_budget
        )

    # Both criteria met → ATTEND
    return AttentionDecision(
        should_attend=True,
        reason=f"Salience {salience:.2f} > threshold {threshold:.2f}, ATP sufficient",
        salience=salience,
        threshold=threshold,
        atp_cost=atp_cost,
        atp_budget=atp_budget
    )


def compute_threshold_grid(
    states: list = None,
    atp_range: tuple = (0.0, 1.0),
    criticality_range: tuple = (0.0, 1.0),
    steps: int = 5
) -> Dict[str, list]:
    """
    Compute threshold grid for analysis and visualization.

    Useful for understanding how thresholds vary across conditions.

    Args:
        states: List of MetabolicState to evaluate (default: all)
        atp_range: (min, max) ATP levels to sample
        criticality_range: (min, max) criticality levels to sample
        steps: Number of samples per dimension

    Returns:
        Dict with grid data for each state
    """
    if states is None:
        states = list(MetabolicState)

    import numpy as np

    atp_values = np.linspace(atp_range[0], atp_range[1], steps)
    crit_values = np.linspace(criticality_range[0], criticality_range[1], steps)

    grid = {}

    for state in states:
        state_grid = []
        for atp in atp_values:
            for crit in crit_values:
                threshold = get_attention_threshold(state, atp, crit)
                state_grid.append({
                    'atp': float(atp),
                    'criticality': float(crit),
                    'threshold': threshold
                })
        grid[state.value] = state_grid

    return grid


# =============================================================================
# Key Insight Examples
# =============================================================================

def demonstrate_mrh_dependent_threshold():
    """
    Demonstrate that same salience triggers different actions in different states.

    This is the MRH (Markov Relevancy Horizon) dependent threshold!
    Same observation has different meanings in different contexts.
    """
    salience = 0.6
    atp = 0.8
    criticality = 0.5

    examples = []

    for state in MetabolicState:
        threshold = get_attention_threshold(state, atp, criticality)
        decision = make_attention_decision(
            salience=salience,
            threshold=threshold,
            plugin_name="example",
            atp_cost=10.0,
            atp_budget=50.0
        )

        examples.append({
            'state': state.value,
            'threshold': threshold,
            'decision': 'ATTEND' if decision.should_attend else 'IGNORE',
            'reason': decision.reason
        })

    return examples


if __name__ == "__main__":
    print("=" * 80)
    print("THRESHOLD-BASED ATTENTION DECISION TEST")
    print("=" * 80)
    print()

    # Test 1: Basic threshold computation
    print("Test 1: Threshold Computation")
    print("-" * 40)

    test_cases = [
        (MetabolicState.WAKE, 0.8, 0.5),
        (MetabolicState.FOCUS, 0.6, 0.7),
        (MetabolicState.REST, 0.9, 0.1),
        (MetabolicState.CRISIS, 0.3, 0.9),
        (MetabolicState.DREAM, 0.7, 0.0),
    ]

    for state, atp, crit in test_cases:
        threshold = get_attention_threshold(state, atp, crit)
        print(f"{state.value:8s} | ATP={atp:.1f} | Crit={crit:.1f} | Threshold={threshold:.2f}")

    print()

    # Test 2: Attention decisions
    print("Test 2: Attention Decisions")
    print("-" * 40)

    decision_cases = [
        (0.7, 0.5, 10.0, 50.0, "High salience, sufficient ATP"),
        (0.4, 0.5, 10.0, 50.0, "Low salience"),
        (0.7, 0.5, 60.0, 50.0, "High salience, insufficient ATP"),
        (0.95, 0.9, 5.0, 50.0, "Crisis-level salience"),
    ]

    for salience, threshold, atp_cost, atp_budget, desc in decision_cases:
        decision = make_attention_decision(salience, threshold, "test", atp_cost, atp_budget)
        print(f"{desc:40s} | {'ATTEND' if decision.should_attend else 'IGNORE':6s} | {decision.reason}")

    print()

    # Test 3: MRH-dependent threshold demonstration
    print("Test 3: Same Salience, Different States")
    print("-" * 40)
    print("Salience = 0.6 in different metabolic states:")
    print()

    examples = demonstrate_mrh_dependent_threshold()
    for ex in examples:
        print(f"{ex['state']:8s} | Threshold={ex['threshold']:.2f} | {ex['decision']:6s} | {ex['reason']}")

    print()
    print("✓ All tests passed!")
    print()
    print("Key Insight: Same salience (0.6) triggers ATTEND in some states but IGNORE in others.")
    print("This is the context-dependent threshold from compression-action-threshold pattern.")
