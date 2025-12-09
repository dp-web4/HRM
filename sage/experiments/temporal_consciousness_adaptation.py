#!/usr/bin/env python3
"""
Session 16: Temporal Consciousness Adaptation

Building on Sessions 14-15, implement continuous adaptation over real-world time.

Research Context:
- Session 14: Self-tuning works for static workloads (+3.5% improvement)
- Session 15: Energy is negligible (no power constraints)
- Gap: How does consciousness adapt over time in real deployments?

Research Questions:
1. Do optimal ATP parameters drift over hours/days?
2. Can we detect and respond to time-of-day patterns?
3. How quickly should adaptation respond to changes?
4. What metrics indicate adaptation is needed?

Approach:
- Track consciousness quality metrics over time windows
- Detect degradation or opportunity for improvement
- Trigger micro-adaptations (small ATP adjustments)
- Maintain long-term performance history
- Learn temporal patterns (time-of-day, day-of-week)

This implements "online continuous adaptation" - the production deployment
mode where consciousness automatically tunes itself based on real experience.

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
    """
    Sliding window of consciousness performance over time.

    Tracks metrics to detect when adaptation is beneficial.
    """
    window_minutes: int = 15  # Time window for metric aggregation

    # Performance metrics (recent history)
    attention_rates: deque = field(default_factory=lambda: deque(maxlen=1000))
    salience_values: deque = field(default_factory=lambda: deque(maxlen=1000))
    atp_levels: deque = field(default_factory=lambda: deque(maxlen=1000))
    coverage_scores: deque = field(default_factory=lambda: deque(maxlen=100))

    # Timestamp tracking
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

        # Coverage (every 100 cycles)
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
    """Record of a single adaptation decision"""
    timestamp: float
    trigger: str  # Why adaptation occurred
    old_params: Tuple[float, float]  # (cost, recovery)
    new_params: Tuple[float, float]
    metrics_before: Dict[str, float]
    metrics_after: Optional[Dict[str, float]] = None
    success: Optional[bool] = None  # Evaluated after stabilization


@dataclass
class TemporalPattern:
    """
    Detected temporal pattern in consciousness behavior.

    Examples:
    - "High salience mornings" → increase attention 8am-11am
    - "Low activity nights" → conserve ATP 10pm-6am
    - "Variable weekdays" → adaptive mode Mon-Fri
    """
    pattern_type: str  # "time_of_day", "day_of_week", "hourly"
    time_range: Tuple[int, int]  # e.g., (8, 11) for 8am-11am
    optimal_params: Tuple[float, float]  # (cost, recovery)
    confidence: float  # 0-1, based on observation count
    observations: int = 0


class TemporalConsciousnessAdapter:
    """
    Continuous online adaptation of ATP parameters over time.

    Learns from experience and adjusts consciousness parameters to maintain
    optimal performance as workload patterns change over hours/days.

    Adaptation Strategy:
    1. Monitor performance in sliding windows (15min)
    2. Detect degradation or opportunity signals
    3. Apply small ATP adjustments (±10% micro-tuning)
    4. Evaluate adjustment success
    5. Learn temporal patterns over days
    6. Proactively adjust based on time-of-day
    """

    def __init__(
        self,
        initial_cost: float = 0.01,
        initial_recovery: float = 0.05,
        adaptation_rate: float = 0.1,  # Max adjustment per step (10%)
        window_minutes: int = 15,
        adaptation_threshold: float = 0.05  # 5% performance change triggers adaptation
    ):
        """
        Initialize temporal adaptation system.

        Args:
            initial_cost: Starting ATP attention cost
            initial_recovery: Starting ATP recovery rate
            adaptation_rate: Maximum parameter adjustment per adaptation (fraction)
            window_minutes: Time window for performance monitoring
            adaptation_threshold: Performance change that triggers adaptation
        """
        self.current_cost = initial_cost
        self.current_recovery = initial_recovery
        self.adaptation_rate = adaptation_rate
        self.adaptation_threshold = adaptation_threshold

        # Performance monitoring
        self.current_window = TemporalWindow(window_minutes=window_minutes)
        self.baseline_metrics: Optional[Dict[str, float]] = None

        # Adaptation history
        self.adaptations: List[AdaptationEvent] = []
        self.temporal_patterns: List[TemporalPattern] = []

        # State
        self.total_cycles = 0
        self.last_adaptation_cycle = 0
        self.min_cycles_between_adaptations = 500  # Stabilization period

    def should_adapt(self) -> Tuple[bool, str]:
        """
        Determine if adaptation is needed based on current metrics.

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
        if cycles_since_last < self.min_cycles_between_adaptations:
            return False, f"Stabilizing ({cycles_since_last}/{self.min_cycles_between_adaptations})"

        # Check current performance
        current = self.current_window.get_metrics()
        if not current:
            return False, "Insufficient data"

        # Detect degradation
        if 'coverage' in current and 'coverage' in self.baseline_metrics:
            coverage_delta = current['coverage'] - self.baseline_metrics['coverage']
            if coverage_delta < -self.adaptation_threshold:
                return True, f"Coverage degraded ({coverage_delta:+.1%})"

        # Detect opportunity for improvement
        if 'attention_rate' in current:
            attn_rate = current['attention_rate']
            # Too low → might be missing events
            if attn_rate < 0.15:
                return True, f"Attention very low ({attn_rate:.1%})"
            # ATP waste → could increase attention
            if 'mean_atp' in current and current['mean_atp'] > 0.85:
                return True, f"ATP surplus ({current['mean_atp']:.1%})"

        return False, "Performance stable"

    def compute_adaptation(self, reason: str) -> Tuple[float, float]:
        """
        Compute new ATP parameters based on adaptation reason.

        Returns:
            (new_cost, new_recovery)
        """
        current_metrics = self.current_window.get_metrics()

        new_cost = self.current_cost
        new_recovery = self.current_recovery

        if "degraded" in reason.lower():
            # Performance drop → try increasing attention
            # Lower cost = more attention
            new_cost = max(0.005, self.current_cost * (1 - self.adaptation_rate))
            # Higher recovery = faster ATP restoration
            new_recovery = min(0.15, self.current_recovery * (1 + self.adaptation_rate))

        elif "very low" in reason.lower():
            # Attention too low → encourage more attention
            new_cost = max(0.005, self.current_cost * 0.9)
            new_recovery = min(0.15, self.current_recovery * 1.1)

        elif "surplus" in reason.lower():
            # ATP wasted → can afford more attention
            new_cost = max(0.005, self.current_cost * 0.95)

        return new_cost, new_recovery

    def apply_adaptation(self, reason: str) -> AdaptationEvent:
        """
        Execute adaptation and record event.

        Returns:
            AdaptationEvent record
        """
        old_params = (self.current_cost, self.current_recovery)
        new_params = self.compute_adaptation(reason)
        metrics_before = self.current_window.get_metrics()

        # Update parameters
        self.current_cost, self.current_recovery = new_params

        # Record event
        event = AdaptationEvent(
            timestamp=time.time(),
            trigger=reason,
            old_params=old_params,
            new_params=new_params,
            metrics_before=metrics_before
        )
        self.adaptations.append(event)
        self.last_adaptation_cycle = self.total_cycles

        # Reset window for fresh measurement
        self.current_window = TemporalWindow()

        return event

    def evaluate_last_adaptation(self) -> bool:
        """
        Evaluate success of most recent adaptation.

        Returns:
            True if adaptation improved performance
        """
        if not self.adaptations:
            return False

        event = self.adaptations[-1]
        if event.success is not None:
            return event.success

        # Need enough cycles for evaluation
        if self.current_window.cycle_count < 500:
            return False

        metrics_after = self.current_window.get_metrics()
        event.metrics_after = metrics_after

        # Compare coverage (primary metric)
        if 'coverage' in metrics_after and 'coverage' in event.metrics_before:
            improvement = metrics_after['coverage'] - event.metrics_before['coverage']
            event.success = improvement > 0.01  # 1% improvement threshold
            return event.success

        # Fallback: attention rate improvement
        if 'attention_rate' in metrics_after and 'attention_rate' in event.metrics_before:
            improvement = metrics_after['attention_rate'] - event.metrics_before['attention_rate']
            event.success = improvement > 0.02  # 2% improvement
            return event.success

        return False

    def detect_temporal_patterns(self) -> List[TemporalPattern]:
        """
        Analyze adaptation history to detect time-based patterns.

        Not implemented in initial version - placeholder for future enhancement.
        """
        # Future: Analyze adaptations by hour-of-day, day-of-week
        # Cluster successful adaptations by time
        # Learn "morning config", "night config", etc.
        return []


def run_temporal_adaptation_experiment(
    duration_minutes: float = 30.0,
    workload_shifts: List[Tuple[float, float, float]] = None
):
    """
    Test temporal adaptation over simulated time with changing workload.

    Args:
        duration_minutes: How long to run simulation
        workload_shifts: List of (time_minutes, alpha, beta) for Beta distribution changes
    """
    print("=" * 80)
    print("Session 16: Temporal Consciousness Adaptation")
    print("=" * 80)
    print()
    print("Research Goal: Continuous online adaptation over real-world time")
    print(f"Duration: {duration_minutes} minutes simulated")
    print()

    # Default workload shifts (simulate changing environment)
    if workload_shifts is None:
        workload_shifts = [
            (0, 5, 2),      # Start: Balanced workload
            (10, 8, 2),     # Shift to high-salience (busy period)
            (20, 2, 8),     # Shift to low-salience (quiet period)
        ]

    print("Workload Timeline:")
    for time_min, alpha, beta in workload_shifts:
        print(f"  {time_min:4.0f} min: Beta({alpha},{beta})")
    print()

    # Initialize adaptation system
    adapter = TemporalConsciousnessAdapter(
        initial_cost=0.03,        # Start with Balanced config
        initial_recovery=0.04,
        adaptation_rate=0.1,      # 10% max adjustment
        window_minutes=5,         # 5min windows for faster response
        adaptation_threshold=0.05 # 5% performance change
    )

    # Create consciousness instance
    consciousness = ATPTunedConsciousness(
        identity_name="temporal-adapter",
        attention_cost=adapter.current_cost,
        rest_recovery=adapter.current_recovery
    )

    print("Starting temporal adaptation experiment...")
    print()

    start_time = time.time()
    current_workload_idx = 0
    cycle = 0

    # Track high-salience observations for coverage metric
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

        # Generate observation with current workload
        _, alpha, beta = workload_shifts[current_workload_idx]
        salience = random.betavariate(alpha, beta)

        obs = SensorObservation(
            sensor_name="sensor_0",
            salience=salience,
            data=f"cycle_{cycle}"
        )

        # Process cycle
        consciousness.process_cycle([obs])

        # Track metrics
        attended = consciousness.observations_attended > (cycle if cycle == 0 else (cycle % 1000))
        is_high_salience = salience > 0.7

        if is_high_salience:
            high_salience_window.append(1)
            attended_high_salience_window.append(1 if attended else 0)

        # Update adapter window
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
            print(f"  Params: cost={event.old_params[0]:.4f}→{event.new_params[0]:.4f}, "
                  f"recovery={event.old_params[1]:.4f}→{event.new_params[1]:.4f}")
            print(f"  Metrics before: coverage={event.metrics_before.get('coverage', 0):.1%}, "
                  f"attention={event.metrics_before.get('attention_rate', 0):.1%}")

            # Apply new params to consciousness
            consciousness.attention_cost = adapter.current_cost
            consciousness.rest_recovery = adapter.current_recovery

        # Evaluate previous adaptation if ready
        if len(adapter.adaptations) > 0:
            last_event = adapter.adaptations[-1]
            if last_event.success is None:
                if adapter.current_window.cycle_count >= 500:
                    success = adapter.evaluate_last_adaptation()
                    if last_event.metrics_after:
                        status = "SUCCESS" if success else "FAILED"
                        print(f"[{elapsed_minutes:5.1f} min] Adaptation #{len(adapter.adaptations)} {status}")
                        print(f"  Coverage: {last_event.metrics_before.get('coverage', 0):.1%} → "
                              f"{last_event.metrics_after.get('coverage', 0):.1%}")

        cycle += 1

        # Status every 5 minutes
        if cycle % 500 == 0:
            metrics = adapter.current_window.get_metrics()
            print(f"[{elapsed_minutes:5.1f} min] Status: cycles={cycle}, "
                  f"attention={metrics.get('attention_rate', 0):.1%}, "
                  f"coverage={metrics.get('coverage', 0):.1%}, "
                  f"ATP={metrics.get('mean_atp', 0):.2f}")

        # Small delay
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

        print("\nAdaptation History:")
        for i, event in enumerate(adapter.adaptations, 1):
            elapsed = (event.timestamp - start_time) / 60.0
            status = "✓" if event.success else "✗" if event.success is False else "?"
            print(f"  {i}. [{elapsed:5.1f} min] {status} {event.trigger}")
            print(f"      cost: {event.old_params[0]:.4f} → {event.new_params[0]:.4f}")
            print(f"      recovery: {event.old_params[1]:.4f} → {event.new_params[1]:.4f}")
            if event.metrics_after:
                cov_before = event.metrics_before.get('coverage', 0)
                cov_after = event.metrics_after.get('coverage', 0)
                print(f"      coverage: {cov_before:.1%} → {cov_after:.1%} ({(cov_after-cov_before)*100:+.1f}%)")

    final_metrics = adapter.current_window.get_metrics()
    print(f"\nFinal Performance:")
    print(f"  Attention rate: {final_metrics.get('attention_rate', 0):.1%}")
    print(f"  Coverage: {final_metrics.get('coverage', 0):.1%}")
    print(f"  Mean ATP: {final_metrics.get('mean_atp', 0):.2f}")
    print(f"  Current params: cost={adapter.current_cost:.4f}, recovery={adapter.current_recovery:.4f}")

    return adapter


if __name__ == '__main__':
    # Run 3-minute simulation with workload changes (faster for validation)
    adapter = run_temporal_adaptation_experiment(
        duration_minutes=3.0,
        workload_shifts=[
            (0, 5, 2),      # Balanced workload (start)
            (1, 8, 2),      # High-salience period (busy) - shift at 1 min
            (2, 2, 8),      # Low-salience period (quiet) - shift at 2 min
        ]
    )

    print("\nSession 16 complete!")
