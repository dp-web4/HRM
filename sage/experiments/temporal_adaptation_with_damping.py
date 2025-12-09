#!/usr/bin/env python3
"""
Session 17: Temporal Adaptation with Damping Mechanism

Improvement over Session 16 to prevent over-adaptation.

Research Context:
- Session 16: Continuous temporal adaptation validated
- Problem: 93 consecutive micro-tunings during low-salience period
- Root cause: No mechanism to stop when performance is excellent
- Solution: Implement damping, satisfaction thresholds, adaptive windows

Research Questions:
1. Can damping prevent over-adaptation while maintaining responsiveness?
2. What's the right balance between stability and responsiveness?
3. How do satisfaction thresholds affect adaptation behavior?

Approach:
- Exponential backoff: Reduce adaptation rate after consecutive similar triggers
- Satisfaction threshold: Stop adapting when coverage >95% and stable
- Adaptive stabilization: Increase window after successful adaptations
- Reset damping on new problem types

Hardware: Jetson AGX Thor
"""

import time
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from validate_atp_on_real_consciousness import ATPTunedConsciousness, SensorObservation


@dataclass
class TemporalWindow:
    """Sliding window of consciousness performance over time (unchanged from Session 16)"""
    window_minutes: int = 15
    attention_rates: deque = field(default_factory=lambda: deque(maxlen=1000))
    salience_values: deque = field(default_factory=lambda: deque(maxlen=1000))
    atp_levels: deque = field(default_factory=lambda: deque(maxlen=1000))
    coverage_scores: deque = field(default_factory=lambda: deque(maxlen=100))
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    cycle_count: int = 0

    def add_cycle(
        self,
        attended: bool,
        salience: float,
        atp_level: float,
        high_salience_count: int = 0,
        attended_high_salience: int = 0
    ):
        """Add metrics from a single consciousness cycle"""
        self.attention_rates.append(1.0 if attended else 0.0)
        if attended:
            self.salience_values.append(salience)
        self.atp_levels.append(atp_level)

        if self.cycle_count % 100 == 0 and high_salience_count > 0:
            coverage = attended_high_salience / high_salience_count
            self.coverage_scores.append(coverage)

        self.cycle_count += 1
        self.last_update = time.time()

    def get_metrics(self) -> Dict[str, float]:
        """Calculate current window metrics"""
        if not self.attention_rates:
            return {}

        return {
            'attention_rate': statistics.mean(self.attention_rates),
            'mean_salience': statistics.mean(self.salience_values) if self.salience_values else 0.0,
            'mean_atp': statistics.mean(self.atp_levels),
            'atp_std': statistics.stdev(self.atp_levels) if len(self.atp_levels) > 1 else 0.0,
            'coverage': statistics.mean(self.coverage_scores) if self.coverage_scores else 0.0,
            'cycles': self.cycle_count,
            'duration_minutes': (self.last_update - self.start_time) / 60.0
        }


@dataclass
class AdaptationEvent:
    """Record of a single adaptation decision (unchanged from Session 16)"""
    timestamp: float
    trigger: str
    old_params: Tuple[float, float]
    new_params: Tuple[float, float]
    metrics_before: Dict[str, float]
    metrics_after: Optional[Dict[str, float]] = None
    success: Optional[bool] = None
    damping_factor: float = 1.0  # NEW: Track damping applied


@dataclass
class TemporalPattern:
    """Detected temporal pattern (placeholder)"""
    pattern_type: str
    time_range: Tuple[int, int]
    optimal_params: Tuple[float, float]
    confidence: float
    observations: int = 0


class DampedTemporalAdapter:
    """
    Temporal consciousness adapter with damping mechanism.

    NEW features over Session 16:
    1. Exponential backoff: Consecutive similar triggers reduce adaptation rate
    2. Satisfaction threshold: Stop adapting when performance is excellent (>95% coverage)
    3. Adaptive stabilization: Increase cycles between adaptations after successes
    4. Trigger tracking: Reset damping when problem type changes

    This prevents over-adaptation while maintaining responsiveness to genuine problems.
    """

    def __init__(
        self,
        initial_cost: float = 0.01,
        initial_recovery: float = 0.05,
        base_adaptation_rate: float = 0.1,  # NEW: Base rate before damping
        window_minutes: int = 15,
        adaptation_threshold: float = 0.05,
        satisfaction_threshold: float = 0.95,  # NEW: Coverage threshold for satisfaction
        enable_damping: bool = True,  # NEW: Toggle damping on/off
        damping_decay: float = 0.5,  # NEW: Decay factor for consecutive adaptations
        min_damping: float = 0.1  # NEW: Minimum damping factor
    ):
        """
        Initialize damped temporal adaptation system.

        Args:
            initial_cost: Starting ATP attention cost
            initial_recovery: Starting ATP recovery rate
            base_adaptation_rate: Base adaptation rate before damping
            window_minutes: Time window for performance monitoring
            adaptation_threshold: Performance change that triggers adaptation
            satisfaction_threshold: Coverage level considered "excellent"
            enable_damping: Whether to apply damping mechanism
            damping_decay: Exponential decay factor for consecutive adaptations
            min_damping: Minimum allowed damping factor
        """
        self.current_cost = initial_cost
        self.current_recovery = initial_recovery
        self.base_adaptation_rate = base_adaptation_rate
        self.adaptation_threshold = adaptation_threshold
        self.satisfaction_threshold = satisfaction_threshold
        self.enable_damping = enable_damping
        self.damping_decay = damping_decay
        self.min_damping = min_damping

        # Performance monitoring
        self.current_window = TemporalWindow(window_minutes=window_minutes)
        self.baseline_metrics: Optional[Dict[str, float]] = None

        # Adaptation history
        self.adaptations: List[AdaptationEvent] = []
        self.temporal_patterns: List[TemporalPattern] = []

        # State
        self.total_cycles = 0
        self.last_adaptation_cycle = 0
        self.base_stabilization_cycles = 500  # Base minimum between adaptations
        self.current_stabilization_cycles = 500  # Adaptive value

        # NEW: Damping state
        self.consecutive_similar_triggers = 0
        self.last_trigger_type = None
        self.current_damping_factor = 1.0
        self.satisfaction_stable_windows = 0  # Count of consecutive satisfied windows

    def should_adapt(self) -> Tuple[bool, str]:
        """
        Determine if adaptation is needed based on current metrics.

        NEW: Includes satisfaction check to prevent unnecessary adaptations.

        Returns:
            (should_adapt, reason)
        """
        # Need baseline first
        if self.baseline_metrics is None:
            if self.current_window.cycle_count >= 500:
                self.baseline_metrics = self.current_window.get_metrics()
                return False, "Baseline established"
            return False, "Collecting baseline"

        # Enforce stabilization period
        cycles_since_last = self.total_cycles - self.last_adaptation_cycle
        if cycles_since_last < self.current_stabilization_cycles:
            return False, f"Stabilizing ({cycles_since_last}/{self.current_stabilization_cycles})"

        # Check current performance
        current = self.current_window.get_metrics()
        if not current:
            return False, "Insufficient data"

        # NEW: Satisfaction check
        if 'coverage' in current:
            coverage = current['coverage']
            if coverage >= self.satisfaction_threshold:
                # Track consecutive satisfied windows
                self.satisfaction_stable_windows += 1

                # If satisfied for multiple windows, stop adapting
                if self.satisfaction_stable_windows >= 3:
                    return False, f"Satisfied (coverage {coverage:.1%} for {self.satisfaction_stable_windows} windows)"
            else:
                # Reset satisfaction counter if drops below threshold
                self.satisfaction_stable_windows = 0

        # Detect degradation (high priority - always trigger)
        if 'coverage' in current and 'coverage' in self.baseline_metrics:
            coverage_delta = current['coverage'] - self.baseline_metrics['coverage']
            if coverage_delta < -self.adaptation_threshold:
                return True, f"Coverage degraded ({coverage_delta:+.1%})"

        # Detect opportunity for improvement (lower priority - with damping)
        if 'attention_rate' in current:
            attn_rate = current['attention_rate']

            # Too low → might be missing events
            if attn_rate < 0.15:
                return True, f"Attention very low ({attn_rate:.1%})"

            # NEW: Modified ATP surplus check
            # Only trigger if ATP high AND attention not already maxed
            if 'mean_atp' in current and current['mean_atp'] > 0.85:
                if attn_rate < 0.80:  # Only if not already giving high attention
                    return True, f"ATP surplus ({current['mean_atp']:.1%})"

        return False, "Performance stable"

    def update_damping(self, trigger_type: str):
        """
        Update damping factor based on trigger history.

        NEW: Core damping mechanism.
        - Consecutive similar triggers → increase damping (reduce adaptation rate)
        - Different trigger type → reset damping
        - Exponential decay: damping_factor *= damping_decay
        """
        if not self.enable_damping:
            self.current_damping_factor = 1.0
            return

        # Categorize trigger types
        if "degraded" in trigger_type.lower():
            category = "degradation"
        elif "very low" in trigger_type.lower():
            category = "low_attention"
        elif "surplus" in trigger_type.lower():
            category = "atp_surplus"
        else:
            category = "other"

        # Check if same category as last trigger
        if category == self.last_trigger_type:
            self.consecutive_similar_triggers += 1
            # Apply exponential damping
            self.current_damping_factor *= self.damping_decay
            # Enforce minimum
            self.current_damping_factor = max(self.min_damping, self.current_damping_factor)
        else:
            # New trigger type - reset damping
            self.consecutive_similar_triggers = 0
            self.current_damping_factor = 1.0
            self.last_trigger_type = category

    def compute_adaptation(self, reason: str) -> Tuple[float, float]:
        """
        Compute new ATP parameters based on adaptation reason.

        NEW: Applies damping factor to reduce adaptation magnitude.

        Returns:
            (new_cost, new_recovery)
        """
        # Update damping based on trigger history
        self.update_damping(reason)

        # Apply damping to adaptation rate
        effective_rate = self.base_adaptation_rate * self.current_damping_factor

        current_metrics = self.current_window.get_metrics()

        new_cost = self.current_cost
        new_recovery = self.current_recovery

        if "degraded" in reason.lower():
            # Performance drop → try increasing attention
            new_cost = max(0.005, self.current_cost * (1 - effective_rate))
            new_recovery = min(0.15, self.current_recovery * (1 + effective_rate))

        elif "very low" in reason.lower():
            # Attention too low → encourage more attention
            new_cost = max(0.005, self.current_cost * (1 - effective_rate))
            new_recovery = min(0.15, self.current_recovery * (1 + effective_rate * 0.5))

        elif "surplus" in reason.lower():
            # ATP wasted → can afford slightly more attention
            # NEW: Smaller adjustment for surplus (less aggressive)
            new_cost = max(0.005, self.current_cost * (1 - effective_rate * 0.5))

        return new_cost, new_recovery

    def apply_adaptation(self, reason: str) -> AdaptationEvent:
        """
        Execute adaptation and record event.

        NEW: Tracks damping factor, adjusts stabilization period.

        Returns:
            AdaptationEvent record
        """
        old_params = (self.current_cost, self.current_recovery)
        new_params = self.compute_adaptation(reason)
        metrics_before = self.current_window.get_metrics()

        # Update parameters
        self.current_cost, self.current_recovery = new_params

        # Record event with damping factor
        event = AdaptationEvent(
            timestamp=time.time(),
            trigger=reason,
            old_params=old_params,
            new_params=new_params,
            metrics_before=metrics_before,
            damping_factor=self.current_damping_factor
        )
        self.adaptations.append(event)
        self.last_adaptation_cycle = self.total_cycles

        # NEW: Adaptive stabilization
        # Increase stabilization period after consecutive similar adaptations
        if self.consecutive_similar_triggers > 2:
            # Double stabilization period (up to 2000 cycles max)
            self.current_stabilization_cycles = min(2000, self.base_stabilization_cycles * 2)
        else:
            self.current_stabilization_cycles = self.base_stabilization_cycles

        # Reset window for fresh measurement
        self.current_window = TemporalWindow()
        # Reset satisfaction counter after adapting
        self.satisfaction_stable_windows = 0

        return event

    def evaluate_last_adaptation(self) -> bool:
        """
        Evaluate success of most recent adaptation (unchanged from Session 16).

        Returns:
            True if adaptation improved performance
        """
        if not self.adaptations:
            return False

        event = self.adaptations[-1]
        if event.success is not None:
            return event.success

        if self.current_window.cycle_count < 500:
            return False

        metrics_after = self.current_window.get_metrics()
        event.metrics_after = metrics_after

        if 'coverage' in metrics_after and 'coverage' in event.metrics_before:
            improvement = metrics_after['coverage'] - event.metrics_before['coverage']
            event.success = improvement > 0.01
            return event.success

        if 'attention_rate' in metrics_after and 'attention_rate' in event.metrics_before:
            improvement = metrics_after['attention_rate'] - event.metrics_before['attention_rate']
            event.success = improvement > 0.02
            return event.success

        return False


def run_damped_adaptation_experiment(
    duration_minutes: float = 3.0,
    workload_shifts: List[Tuple[float, float, float]] = None,
    enable_damping: bool = True
):
    """
    Test damped temporal adaptation.

    Args:
        duration_minutes: How long to run simulation
        workload_shifts: List of (time_minutes, alpha, beta) for Beta distribution changes
        enable_damping: Whether to enable damping mechanism
    """
    print("=" * 80)
    print("Session 17: Temporal Adaptation with Damping")
    print("=" * 80)
    print()
    print(f"Damping: {'ENABLED' if enable_damping else 'DISABLED'} (comparison mode)")
    print(f"Duration: {duration_minutes} minutes simulated")
    print()

    if workload_shifts is None:
        workload_shifts = [
            (0, 5, 2),
            (1, 8, 2),
            (2, 2, 8),
        ]

    print("Workload Timeline:")
    for time_min, alpha, beta in workload_shifts:
        print(f"  {time_min:4.0f} min: Beta({alpha},{beta})")
    print()

    # Initialize damped adapter
    adapter = DampedTemporalAdapter(
        initial_cost=0.03,
        initial_recovery=0.04,
        base_adaptation_rate=0.1,
        window_minutes=5,
        adaptation_threshold=0.05,
        satisfaction_threshold=0.95,
        enable_damping=enable_damping,
        damping_decay=0.5,
        min_damping=0.1
    )

    # Create consciousness instance
    consciousness = ATPTunedConsciousness(
        identity_name="damped-adapter",
        attention_cost=adapter.current_cost,
        rest_recovery=adapter.current_recovery
    )

    print(f"Starting experiment (damping {'ON' if enable_damping else 'OFF'})...")
    print()

    start_time = time.time()
    current_workload_idx = 0
    cycle = 0

    high_salience_window = deque(maxlen=100)
    attended_high_salience_window = deque(maxlen=100)

    while True:
        elapsed_minutes = (time.time() - start_time) / 60.0
        if elapsed_minutes >= duration_minutes:
            break

        # Check for workload shift
        if (current_workload_idx + 1 < len(workload_shifts) and
            elapsed_minutes >= workload_shifts[current_workload_idx + 1][0]):
            current_workload_idx += 1
            _, alpha, beta = workload_shifts[current_workload_idx]
            print(f"\n[{elapsed_minutes:5.1f} min] Workload shift: Beta({alpha},{beta})")

        # Generate observation
        _, alpha, beta = workload_shifts[current_workload_idx]
        salience = random.betavariate(alpha, beta)

        obs = SensorObservation(
            sensor_name="sensor_0",
            salience=salience,
            data=f"cycle_{cycle}"
        )

        consciousness.process_cycle([obs])

        attended = consciousness.observations_attended > (cycle if cycle == 0 else (cycle % 1000))
        is_high_salience = salience > 0.7

        if is_high_salience:
            high_salience_window.append(1)
            attended_high_salience_window.append(1 if attended else 0)

        high_salience_count = len(high_salience_window)
        attended_count = sum(attended_high_salience_window)

        adapter.current_window.add_cycle(
            attended=attended,
            salience=salience,
            atp_level=consciousness.atp_level,
            high_salience_count=high_salience_count,
            attended_high_salience=attended_count
        )
        adapter.total_cycles += 1

        # Check if adaptation needed
        should_adapt, reason = adapter.should_adapt()

        if should_adapt:
            event = adapter.apply_adaptation(reason)
            print(f"[{elapsed_minutes:5.1f} min] ADAPTATION #{len(adapter.adaptations)}")
            print(f"  Trigger: {event.trigger}")
            print(f"  Damping: {event.damping_factor:.2f}x (consecutive: {adapter.consecutive_similar_triggers})")
            print(f"  Params: cost={event.old_params[0]:.4f}→{event.new_params[0]:.4f}, "
                  f"recovery={event.old_params[1]:.4f}→{event.new_params[1]:.4f}")
            print(f"  Stabilization window: {adapter.current_stabilization_cycles} cycles")

            # Apply new params
            consciousness.attention_cost = adapter.current_cost
            consciousness.rest_recovery = adapter.current_recovery

        # Evaluate previous adaptation
        if len(adapter.adaptations) > 0:
            last_event = adapter.adaptations[-1]
            if last_event.success is None and adapter.current_window.cycle_count >= 500:
                success = adapter.evaluate_last_adaptation()
                if last_event.metrics_after:
                    status = "SUCCESS" if success else "FAILED"
                    print(f"[{elapsed_minutes:5.1f} min] Adaptation #{len(adapter.adaptations)} {status}")

        cycle += 1

        # Status every 5 minutes
        if cycle % 500 == 0:
            metrics = adapter.current_window.get_metrics()
            print(f"[{elapsed_minutes:5.1f} min] Status: cycles={cycle}, "
                  f"attention={metrics.get('attention_rate', 0):.1%}, "
                  f"coverage={metrics.get('coverage', 0):.1%}, "
                  f"ATP={metrics.get('mean_atp', 0):.2f}, "
                  f"damping={adapter.current_damping_factor:.2f}x")

        time.sleep(0.001)

    # Final analysis
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    print(f"\nDuration: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Total cycles: {cycle}")
    print(f"Total adaptations: {len(adapter.adaptations)}")

    if adapter.adaptations:
        successful = sum(1 for e in adapter.adaptations if e.success)
        print(f"Successful adaptations: {successful}/{len(adapter.adaptations)}")

        # Count adaptations by trigger type
        degradation = sum(1 for e in adapter.adaptations if "degraded" in e.trigger.lower())
        low_attention = sum(1 for e in adapter.adaptations if "very low" in e.trigger.lower())
        atp_surplus = sum(1 for e in adapter.adaptations if "surplus" in e.trigger.lower())

        print(f"\nAdaptations by trigger:")
        print(f"  Degradation: {degradation}")
        print(f"  Low attention: {low_attention}")
        print(f"  ATP surplus: {atp_surplus}")

    final_metrics = adapter.current_window.get_metrics()
    print(f"\nFinal Performance:")
    print(f"  Attention rate: {final_metrics.get('attention_rate', 0):.1%}")
    print(f"  Coverage: {final_metrics.get('coverage', 0):.1%}")
    print(f"  Mean ATP: {final_metrics.get('mean_atp', 0):.2f}")
    print(f"  Current params: cost={adapter.current_cost:.4f}, recovery={adapter.current_recovery:.4f}")
    print(f"  Final damping factor: {adapter.current_damping_factor:.2f}x")

    return adapter


if __name__ == '__main__':
    print("Running comparison: Damping ON vs OFF")
    print()

    # Run with damping
    print("=" * 80)
    print("TEST 1: WITH DAMPING")
    print("=" * 80)
    adapter_damped = run_damped_adaptation_experiment(
        duration_minutes=3.0,
        enable_damping=True
    )

    print("\n\n")

    # Run without damping (Session 16 behavior)
    print("=" * 80)
    print("TEST 2: WITHOUT DAMPING (Session 16 baseline)")
    print("=" * 80)
    adapter_undamped = run_damped_adaptation_experiment(
        duration_minutes=3.0,
        enable_damping=False
    )

    # Compare results
    print("\n\n")
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"\nWith damping:")
    print(f"  Total adaptations: {len(adapter_damped.adaptations)}")
    print(f"  Final coverage: {adapter_damped.current_window.get_metrics().get('coverage', 0):.1%}")
    print(f"  Final damping: {adapter_damped.current_damping_factor:.2f}x")

    print(f"\nWithout damping:")
    print(f"  Total adaptations: {len(adapter_undamped.adaptations)}")
    print(f"  Final coverage: {adapter_undamped.current_window.get_metrics().get('coverage', 0):.1%}")
    print(f"  Final damping: {adapter_undamped.current_damping_factor:.2f}x")

    reduction = (1 - len(adapter_damped.adaptations) / len(adapter_undamped.adaptations)) * 100
    print(f"\nAdaptation reduction: {reduction:.1f}%")
    print("\nSession 17 complete!")
