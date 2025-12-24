#!/usr/bin/env python3
"""
Session 103: Internal Wake Policy - Memory & Uncertainty Pressure

**Goal**: Implement agency origination via internal wake triggers (not exogenous timers)

**Context - External Review**:
Nova (GPT-5.2) reviewed metabolic consciousness arc (S97-102) and recommended:
- Primary pathway: Memory/Uncertainty Pressure Wake
- Rationale: "ATP depletion is a brake, not an ignition"
- Agency requires state-dependent initiation, not fixed schedules

**Architecture**:
Sessions 97-102 established constraint layer (ATP limits action)
Session 103 establishes origination layer (pressure triggers action)

Combined system:
- Pressure signals → Wake policy → Trigger action
- ATP constraints → Limit action → Backpressure
- Together: Internally-originated, resource-constrained agency

**Pressure Signals** (Nova's recommendations):
1. Memory Pressure:
   - Growth rate (memory expanding)
   - Retrieval failure rate (can't find needed info)
   - Entropy/fragmentation (disorganized)
   - Staleness (outdated patterns)
   - Contradiction density (conflicting info)

2. Uncertainty Pressure:
   - Calibration error (predictions wrong)
   - Unresolved hypotheses (open questions)
   - High-variance predictions (inconsistent)
   - Repeated low-confidence decisions (persistent uncertainty)

**Wake Policy** (Nova's design):
```
wake_score = f(memory_pressure, uncertainty, expected_value, risk, ATP)
wake_triggered = wake_score > threshold (with hysteresis)
```

**Integration with Existing SAGE**:
- Session 30: EpistemicState (uncertainty, frustration, confidence)
- Session 42: DreamConsolidation (memory patterns, quality learnings)
- Sessions 97-102: Metabolic consciousness (ATP constraint layer)
- Session 103: Wake policy (pressure → action trigger)

**Expected Behavior**:
- High memory/uncertainty pressure → wake_score increases
- Threshold crossed → trigger wake actions (consolidation, pruning, probes)
- Actions reduce pressure → wake_score decreases (negative feedback)
- Hysteresis prevents thrashing
- ATP constraints limit wake frequency (cost control)

Created: 2025-12-23 20:05 UTC (Autonomous Session 103)
Hardware: Thor (Jetson AGX Thor)
Based on: Nova (GPT-5.2) peer review recommendations
Goal: Agency origination through internal pressure signals
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Import existing SAGE components
try:
    from sage.core.epistemic_states import EpistemicState, EpistemicMetrics
    HAS_EPISTEMIC = True
except ImportError:
    HAS_EPISTEMIC = False
    EpistemicState = None
    EpistemicMetrics = None

try:
    from sage.core.dream_consolidation import MemoryPattern, DreamConsolidator
    HAS_DREAM = True
except ImportError:
    HAS_DREAM = False
    MemoryPattern = None
    DreamConsolidator = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MemoryPressureSignals:
    """Memory pressure indicators that drive wake triggers."""

    growth_rate: float = 0.0  # Memory size growth (MB/hour)
    retrieval_failure_rate: float = 0.0  # Failed lookups (0-1)
    entropy: float = 0.0  # Disorganization metric (0-1)
    staleness: float = 0.0  # Age of oldest unprocessed item (hours)
    contradiction_density: float = 0.0  # Conflicting patterns (per 100 items)

    def overall_pressure(self) -> float:
        """Compute overall memory pressure (0-1 scale).

        Weighted combination of pressure signals.
        High pressure → urgent need for consolidation/cleanup.
        """
        # Weights from biological memory consolidation research
        # Growth and staleness are primary drivers
        weights = {
            'growth_rate': 0.3,  # Expanding memory needs consolidation
            'retrieval_failure_rate': 0.25,  # Can't find things → reorganize
            'entropy': 0.2,  # Disorganized → consolidate
            'staleness': 0.15,  # Old items → process or prune
            'contradiction_density': 0.1,  # Conflicts → resolve
        }

        # Normalize signals to 0-1 range
        normalized = {
            'growth_rate': min(self.growth_rate / 100.0, 1.0),  # Cap at 100 MB/h
            'retrieval_failure_rate': self.retrieval_failure_rate,
            'entropy': self.entropy,
            'staleness': min(self.staleness / 24.0, 1.0),  # Cap at 24 hours
            'contradiction_density': min(self.contradiction_density / 10.0, 1.0),  # Cap at 10 per 100
        }

        # Weighted sum
        pressure = sum(normalized[k] * weights[k] for k in weights)
        return pressure


@dataclass
class UncertaintyPressureSignals:
    """Uncertainty pressure indicators that drive wake triggers."""

    calibration_error: float = 0.0  # Prediction accuracy gap (0-1)
    unresolved_hypotheses: int = 0  # Open questions count
    high_variance_predictions: float = 0.0  # Inconsistency (0-1)
    low_confidence_streak: int = 0  # Consecutive uncertain decisions

    def overall_pressure(self) -> float:
        """Compute overall uncertainty pressure (0-1 scale).

        High pressure → need for uncertainty reduction actions.
        """
        weights = {
            'calibration_error': 0.35,  # Wrong predictions → learn
            'unresolved_hypotheses': 0.25,  # Open questions → investigate
            'high_variance_predictions': 0.25,  # Inconsistency → stabilize
            'low_confidence_streak': 0.15,  # Persistent uncertainty → explore
        }

        # Normalize
        normalized = {
            'calibration_error': self.calibration_error,
            'unresolved_hypotheses': min(self.unresolved_hypotheses / 20.0, 1.0),  # Cap at 20
            'high_variance_predictions': self.high_variance_predictions,
            'low_confidence_streak': min(self.low_confidence_streak / 10.0, 1.0),  # Cap at 10
        }

        # Weighted sum
        pressure = sum(normalized[k] * weights[k] for k in weights)
        return pressure


@dataclass
class WakeTriggerState:
    """State tracking for wake trigger system with hysteresis."""

    # Current pressure levels
    memory_pressure: float = 0.0
    uncertainty_pressure: float = 0.0

    # Wake score history (for hysteresis)
    wake_score_history: deque = field(default_factory=lambda: deque(maxlen=10))

    # Trigger state
    is_awake: bool = False
    last_wake_time: float = 0.0
    wake_count: int = 0

    # Cooldown (prevent thrashing)
    cooldown_seconds: float = 300.0  # 5 minutes minimum between wakes
    min_wake_duration: float = 60.0  # 1 minute minimum wake duration

    # Thresholds (with hysteresis)
    wake_threshold: float = 0.6  # Score > 0.6 → wake
    sleep_threshold: float = 0.3  # Score < 0.3 → sleep (lower to prevent oscillation)


@dataclass
class WakeAction:
    """Action taken during wake state to reduce pressure."""

    action_type: str  # consolidation, pruning, probe, triage
    target: str  # What the action operates on
    expected_pressure_reduction: float  # Expected Δ pressure
    atp_cost: float  # ATP consumed by this action
    duration_seconds: float  # How long action takes
    timestamp: float = field(default_factory=time.time)


class InternalWakePolicy:
    """
    Internal wake policy based on memory and uncertainty pressure.

    Implements Nova's recommended architecture:
    - State-dependent wake triggers (not timer-based)
    - Pressure signals from memory & uncertainty
    - Hysteresis to prevent thrashing
    - ATP constraints (cost control)
    - Negative feedback (actions reduce pressure)
    """

    def __init__(
        self,
        wake_threshold: float = 0.6,
        sleep_threshold: float = 0.3,
        cooldown_seconds: float = 300.0,
        atp_cost_per_wake: float = 10.0,
    ):
        """Initialize wake policy.

        Args:
            wake_threshold: Wake score threshold to trigger wake
            sleep_threshold: Wake score threshold to return to sleep (hysteresis)
            cooldown_seconds: Minimum time between wake events
            atp_cost_per_wake: ATP consumed per wake action
        """
        self.wake_threshold = wake_threshold
        self.sleep_threshold = sleep_threshold
        self.cooldown_seconds = cooldown_seconds
        self.atp_cost_per_wake = atp_cost_per_wake

        # State
        self.trigger_state = WakeTriggerState(
            wake_threshold=wake_threshold,
            sleep_threshold=sleep_threshold,
            cooldown_seconds=cooldown_seconds,
        )

        # Wake action history
        self.wake_actions: List[WakeAction] = []

        # Statistics
        self.total_wakes = 0
        self.total_pressure_reduced = 0.0
        self.total_atp_spent = 0.0

    def compute_wake_score(
        self,
        memory_pressure: float,
        uncertainty_pressure: float,
        expected_value: float = 0.5,  # Expected benefit of waking
        risk: float = 0.2,  # Risk of not waking
        current_atp: float = 100.0,  # Available ATP
    ) -> float:
        """Compute wake score from pressure signals.

        Wake score formula (Nova's recommendation):
        wake_score = f(memory_pressure, uncertainty_pressure, expected_value, risk, ATP)

        Implementation:
        - Base score from weighted pressures
        - Modulated by expected value and risk
        - Constrained by ATP availability
        """
        # Weighted pressure combination
        # Memory pressure is slightly higher priority (affects all operations)
        pressure_score = 0.55 * memory_pressure + 0.45 * uncertainty_pressure

        # Expected value modulation
        # High expected value → increase score
        # Low risk → reduce urgency
        modulated_score = pressure_score * (0.7 + 0.3 * expected_value) * (0.8 + 0.2 * risk)

        # ATP constraint
        # If ATP low, reduce wake score (can't afford action)
        # This implements ATP as brake, not ignition (per Nova)
        atp_factor = min(current_atp / 50.0, 1.0)  # Full availability at 50+ ATP
        constrained_score = modulated_score * (0.5 + 0.5 * atp_factor)

        return constrained_score

    def should_wake(
        self,
        memory_signals: MemoryPressureSignals,
        uncertainty_signals: UncertaintyPressureSignals,
        current_atp: float = 100.0,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Determine if system should wake based on pressure signals.

        Returns:
            (should_wake, decision_info)
        """
        # Compute pressure levels
        memory_pressure = memory_signals.overall_pressure()
        uncertainty_pressure = uncertainty_signals.overall_pressure()

        # Compute wake score
        wake_score = self.compute_wake_score(
            memory_pressure=memory_pressure,
            uncertainty_pressure=uncertainty_pressure,
            expected_value=0.5,  # TODO: Learn from wake action outcomes
            risk=0.2,  # TODO: Model risk of delayed consolidation
            current_atp=current_atp,
        )

        # Track history
        self.trigger_state.wake_score_history.append(wake_score)
        self.trigger_state.memory_pressure = memory_pressure
        self.trigger_state.uncertainty_pressure = uncertainty_pressure

        # Check cooldown
        time_since_wake = time.time() - self.trigger_state.last_wake_time
        in_cooldown = time_since_wake < self.cooldown_seconds

        # Hysteresis logic
        if self.trigger_state.is_awake:
            # Currently awake: only sleep if score drops below sleep threshold
            should_wake = wake_score >= self.sleep_threshold
        else:
            # Currently asleep: only wake if score exceeds wake threshold
            should_wake = wake_score >= self.wake_threshold and not in_cooldown

        # State transition
        if should_wake and not self.trigger_state.is_awake:
            # Transition: asleep → awake
            self.trigger_state.is_awake = True
            self.trigger_state.last_wake_time = time.time()
            self.trigger_state.wake_count += 1
            self.total_wakes += 1
            logger.info(f"Wake triggered: score={wake_score:.3f}, memory_p={memory_pressure:.3f}, uncertainty_p={uncertainty_pressure:.3f}")

        elif not should_wake and self.trigger_state.is_awake:
            # Transition: awake → asleep
            time_awake = time.time() - self.trigger_state.last_wake_time
            if time_awake >= self.trigger_state.min_wake_duration:
                self.trigger_state.is_awake = False
                logger.info(f"Sleep triggered: score={wake_score:.3f}, time_awake={time_awake:.1f}s")

        # Decision info
        decision_info = {
            'wake_score': wake_score,
            'memory_pressure': memory_pressure,
            'uncertainty_pressure': uncertainty_pressure,
            'is_awake': self.trigger_state.is_awake,
            'in_cooldown': in_cooldown,
            'time_since_wake': time_since_wake,
            'wake_threshold': self.wake_threshold,
            'sleep_threshold': self.sleep_threshold,
        }

        return should_wake, decision_info

    def select_wake_actions(
        self,
        memory_signals: MemoryPressureSignals,
        uncertainty_signals: UncertaintyPressureSignals,
        available_atp: float = 100.0,
    ) -> List[WakeAction]:
        """Select wake actions to reduce pressure.

        Actions prioritized by:
        - Pressure reduction per ATP spent
        - Urgency of pressure signal
        - ATP budget
        """
        actions = []

        # Memory pressure actions
        if memory_signals.growth_rate > 50.0:
            # High growth → consolidation
            actions.append(WakeAction(
                action_type="consolidation",
                target="recent_memories",
                expected_pressure_reduction=0.3,
                atp_cost=5.0,
                duration_seconds=30.0,
            ))

        if memory_signals.retrieval_failure_rate > 0.2:
            # High retrieval failure → rebuild index
            actions.append(WakeAction(
                action_type="index_rebuild",
                target="memory_index",
                expected_pressure_reduction=0.25,
                atp_cost=7.0,
                duration_seconds=45.0,
            ))

        if memory_signals.staleness > 12.0:
            # Stale items → pruning
            actions.append(WakeAction(
                action_type="pruning",
                target="stale_memories",
                expected_pressure_reduction=0.2,
                atp_cost=3.0,
                duration_seconds=20.0,
            ))

        # Uncertainty pressure actions
        if uncertainty_signals.unresolved_hypotheses > 10:
            # Many open questions → hypothesis triage
            actions.append(WakeAction(
                action_type="triage",
                target="unresolved_hypotheses",
                expected_pressure_reduction=0.2,
                atp_cost=4.0,
                duration_seconds=25.0,
            ))

        if uncertainty_signals.calibration_error > 0.4:
            # Poor calibration → uncertainty probe
            actions.append(WakeAction(
                action_type="probe",
                target="calibration_model",
                expected_pressure_reduction=0.15,
                atp_cost=6.0,
                duration_seconds=40.0,
            ))

        # Sort by efficiency (pressure reduction per ATP)
        actions.sort(key=lambda a: a.expected_pressure_reduction / a.atp_cost, reverse=True)

        # Filter by ATP budget
        total_cost = 0.0
        affordable_actions = []
        for action in actions:
            if total_cost + action.atp_cost <= available_atp:
                affordable_actions.append(action)
                total_cost += action.atp_cost
            else:
                break

        return affordable_actions

    def execute_wake_actions(
        self,
        actions: List[WakeAction],
    ) -> Dict[str, Any]:
        """Execute wake actions and track outcomes.

        In real system, this would:
        - Consolidate memories
        - Rebuild indices
        - Prune stale data
        - Triage hypotheses
        - Run uncertainty probes

        For now, we simulate and track.
        """
        total_atp_cost = sum(a.atp_cost for a in actions)
        total_pressure_reduction = sum(a.expected_pressure_reduction for a in actions)
        total_duration = sum(a.duration_seconds for a in actions)

        # Track actions
        self.wake_actions.extend(actions)
        self.total_atp_spent += total_atp_cost
        self.total_pressure_reduced += total_pressure_reduction

        logger.info(f"Executed {len(actions)} wake actions:")
        for action in actions:
            logger.info(f"  - {action.action_type} on {action.target}: -Δp={action.expected_pressure_reduction:.2f}, cost={action.atp_cost} ATP")

        return {
            'actions_executed': len(actions),
            'total_atp_cost': total_atp_cost,
            'total_pressure_reduction': total_pressure_reduction,
            'total_duration': total_duration,
        }


def simulate_wake_policy_behavior(cycles: int = 100):
    """Simulate wake policy behavior over time.

    Demonstrates:
    - Pressure accumulation
    - Wake triggers
    - Action execution
    - Pressure reduction (negative feedback)
    - Hysteresis preventing thrashing
    """
    logger.info("="*80)
    logger.info("SESSION 103: Internal Wake Policy Simulation")
    logger.info("="*80)

    # Create wake policy
    policy = InternalWakePolicy(
        wake_threshold=0.6,
        sleep_threshold=0.3,
        cooldown_seconds=10.0,  # Short cooldown for simulation
        atp_cost_per_wake=10.0,
    )

    # Simulation state
    current_atp = 100.0
    memory_pressure_base = 0.1
    uncertainty_pressure_base = 0.1

    # Tracking
    wake_events = []
    pressure_trajectory = []

    for cycle in range(cycles):
        # Simulate pressure accumulation
        # Pressure increases over time, reduced by wake actions
        memory_pressure_base += np.random.uniform(0.01, 0.05)  # Gradual increase
        uncertainty_pressure_base += np.random.uniform(0.01, 0.03)

        # Create pressure signals
        memory_signals = MemoryPressureSignals(
            growth_rate=memory_pressure_base * 100,  # Convert to MB/h
            retrieval_failure_rate=min(memory_pressure_base * 0.5, 1.0),
            entropy=min(memory_pressure_base * 0.8, 1.0),
            staleness=memory_pressure_base * 20,  # Hours
            contradiction_density=memory_pressure_base * 5,
        )

        uncertainty_signals = UncertaintyPressureSignals(
            calibration_error=min(uncertainty_pressure_base * 0.7, 1.0),
            unresolved_hypotheses=int(uncertainty_pressure_base * 15),
            high_variance_predictions=min(uncertainty_pressure_base * 0.6, 1.0),
            low_confidence_streak=int(uncertainty_pressure_base * 8),
        )

        # Check wake policy
        should_wake, decision_info = policy.should_wake(
            memory_signals=memory_signals,
            uncertainty_signals=uncertainty_signals,
            current_atp=current_atp,
        )

        # Track pressure
        pressure_trajectory.append({
            'cycle': cycle,
            'memory_pressure': decision_info['memory_pressure'],
            'uncertainty_pressure': decision_info['uncertainty_pressure'],
            'wake_score': decision_info['wake_score'],
            'is_awake': decision_info['is_awake'],
            'atp': current_atp,
        })

        # If awake, execute actions
        if decision_info['is_awake']:
            actions = policy.select_wake_actions(
                memory_signals=memory_signals,
                uncertainty_signals=uncertainty_signals,
                available_atp=current_atp,
            )

            if actions:
                outcome = policy.execute_wake_actions(actions)

                # Apply pressure reduction (negative feedback!)
                memory_pressure_base *= (1.0 - 0.4)  # Reduce by 40%
                uncertainty_pressure_base *= (1.0 - 0.3)  # Reduce by 30%

                # Consume ATP
                current_atp -= outcome['total_atp_cost']

                # Track event
                wake_events.append({
                    'cycle': cycle,
                    'wake_score': decision_info['wake_score'],
                    'actions': len(actions),
                    'atp_cost': outcome['total_atp_cost'],
                    'pressure_reduction': outcome['total_pressure_reduction'],
                })

        # ATP recovery (from metabolic consciousness)
        if current_atp < 40:
            current_atp += 2.0  # REST recovery
        elif current_atp < 100:
            current_atp += 1.0  # Slow recovery toward max

        # Log progress
        if cycle % 20 == 0:
            logger.info(f"Cycle {cycle}: ATP={current_atp:.1f}, wake_score={decision_info['wake_score']:.3f}, awake={decision_info['is_awake']}, wakes={policy.total_wakes}")

    # Final report
    logger.info("="*80)
    logger.info("SIMULATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total cycles: {cycles}")
    logger.info(f"Total wakes: {policy.total_wakes}")
    logger.info(f"Total ATP spent: {policy.total_atp_spent:.1f}")
    logger.info(f"Total pressure reduced: {policy.total_pressure_reduced:.2f}")
    logger.info(f"Wake events: {len(wake_events)}")

    # Analyze wake events
    if wake_events:
        wake_scores = [e['wake_score'] for e in wake_events]
        wake_intervals = [wake_events[i+1]['cycle'] - wake_events[i]['cycle']
                         for i in range(len(wake_events)-1)]

        logger.info(f"\nWake Statistics:")
        logger.info(f"  Average wake score: {np.mean(wake_scores):.3f}")
        logger.info(f"  Wake score range: {min(wake_scores):.3f} - {max(wake_scores):.3f}")
        if wake_intervals:
            logger.info(f"  Average wake interval: {np.mean(wake_intervals):.1f} cycles")
            logger.info(f"  Wake interval range: {min(wake_intervals)} - {max(wake_intervals)} cycles")

    # Save results
    results = {
        'session': 103,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        'simulation': {
            'cycles': cycles,
            'total_wakes': policy.total_wakes,
            'total_atp_spent': policy.total_atp_spent,
            'total_pressure_reduced': policy.total_pressure_reduced,
        },
        'wake_events': wake_events,
        'pressure_trajectory': pressure_trajectory,
    }

    output_path = Path(__file__).parent / "session103_wake_policy_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    results = simulate_wake_policy_behavior(cycles=100)
