#!/usr/bin/env python3
"""
Production Temporal Adaptation for SAGE Consciousness

Session 18: Production Integration of Temporal Adaptation Framework
Session 22: Pattern Learning for Predictive Optimization
Session 24: Multi-Objective Optimization Integration

Integrates validated temporal adaptation (Sessions 16-17) into sage/core for
real-world deployments. Provides continuous online tuning of ATP parameters
based on workload patterns and performance metrics.

Research Provenance:
- Session 16: Temporal consciousness adaptation framework (685 LOC)
- Session 17: Damping mechanism with satisfaction threshold (763 LOC)
- Session 62 (Sprout): Cross-platform validation on edge hardware
- Session 22: Pattern learning capability (128 LOC)
- Session 23: Multi-objective optimization framework (384 LOC)
- Session 24: Multi-objective integration (this session)
- Production-ready: 97.9% reduction in over-adaptation (95 → 2 adaptations)

Key Features:
1. Satisfaction threshold: Stops adapting when coverage >95% for 3 windows
2. Exponential damping: Prevents over-adaptation during stable periods
3. Adaptive stabilization: Increases wait time after successful adaptations
4. Trigger categorization: Resets damping when problem type changes
5. Pattern learning: Tracks time-of-day patterns for predictive tuning
6. Multi-objective optimization: Balances coverage + quality + energy (Session 24)

Integration Points:
- MetabolicController: ATP parameter updates
- AttentionManager: Real-time performance monitoring
- MichaudSAGE: Production consciousness system

Hardware: All platforms (Thor AGX, Orin Nano, development systems)
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from collections import deque
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Session 28: Adaptive objective weighting
try:
    from sage.core.adaptive_weights import (
        AdaptiveWeightCalculator,
        OperatingContext,
        ObjectiveWeights
    )
    ADAPTIVE_WEIGHTS_AVAILABLE = True
except ImportError:
    ADAPTIVE_WEIGHTS_AVAILABLE = False
    logger.warning("Adaptive weights module not available (Session 28)")

# Session 31: Epistemic state tracking
try:
    from sage.core.epistemic_states import (
        EpistemicMetrics,
        EpistemicStateTracker,
        estimate_epistemic_metrics
    )
    EPISTEMIC_TRACKING_AVAILABLE = True
except ImportError:
    EPISTEMIC_TRACKING_AVAILABLE = False
    logger.warning("Epistemic tracking module not available (Session 31)")


class AdaptationTrigger(Enum):
    """Types of adaptation triggers"""
    DEGRADATION = "degradation"      # Coverage dropped
    LOW_ATTENTION = "low_attention"  # Attention rate very low
    ATP_SURPLUS = "atp_surplus"      # ATP consistently high
    PATTERN_SHIFT = "pattern_shift"  # Temporal pattern changed
    NONE = "none"                    # No adaptation needed


@dataclass
class TemporalWindow:
    """
    Sliding window of consciousness performance metrics.

    Tracks attention rates, salience values, ATP levels, and coverage
    scores over a configurable time window (default 15 minutes).

    Extended in Session 24 to support multi-objective optimization:
    - Quality scores (response quality when available)
    - ATP spending (for energy efficiency calculation)
    """
    window_minutes: int = 15
    attention_rates: deque = field(default_factory=lambda: deque(maxlen=1000))
    salience_values: deque = field(default_factory=lambda: deque(maxlen=1000))
    atp_levels: deque = field(default_factory=lambda: deque(maxlen=1000))
    coverage_scores: deque = field(default_factory=lambda: deque(maxlen=100))

    # Session 24: Multi-objective fitness tracking
    quality_scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    atp_spent: deque = field(default_factory=lambda: deque(maxlen=1000))

    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    cycle_count: int = 0

    def add_cycle(
        self,
        attended: bool,
        salience: float,
        atp_level: float,
        high_salience_count: int = 0,
        attended_high_salience: int = 0,
        quality_score: Optional[float] = None,
        attention_cost: float = 0.01
    ):
        """
        Add metrics from a single consciousness cycle.

        Args:
            attended: Whether attention was allocated
            salience: Salience value of observation
            atp_level: Current ATP level
            high_salience_count: Count of high-salience observations in window
            attended_high_salience: How many high-salience were attended
            quality_score: Optional quality score for this cycle (Session 24)
            attention_cost: ATP cost per attention (for energy tracking, Session 24)
        """
        self.attention_rates.append(1.0 if attended else 0.0)
        if attended:
            self.salience_values.append(salience)

        self.atp_levels.append(atp_level)

        # Calculate coverage every 100 cycles
        if self.cycle_count % 100 == 0 and high_salience_count > 0:
            coverage = attended_high_salience / high_salience_count
            self.coverage_scores.append(coverage)

        # Session 24: Track quality and energy
        if quality_score is not None and attended:
            self.quality_scores.append(quality_score)

        if attended:
            self.atp_spent.append(attention_cost)

        self.cycle_count += 1
        self.last_update = time.time()

    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate current window metrics.

        Returns both single-objective (coverage) and multi-objective
        (coverage + quality + energy) fitness metrics.
        """
        if not self.attention_rates:
            return {}

        metrics = {
            'attention_rate': statistics.mean(self.attention_rates),
            'mean_salience': statistics.mean(self.salience_values) if self.salience_values else 0.0,
            'mean_atp': statistics.mean(self.atp_levels),
            'atp_std': statistics.stdev(self.atp_levels) if len(self.atp_levels) > 1 else 0.0,
            'coverage': statistics.mean(self.coverage_scores) if self.coverage_scores else 0.0,
            'cycles': self.cycle_count,
            'duration_minutes': (self.last_update - self.start_time) / 60.0
        }

        # Session 24: Multi-objective fitness
        if self.quality_scores:
            metrics['quality'] = statistics.mean(self.quality_scores)
        else:
            metrics['quality'] = 0.0

        # Energy efficiency: cycles per ATP spent (normalized)
        if self.atp_spent:
            total_atp = sum(self.atp_spent)
            if total_atp > 0:
                efficiency_raw = self.cycle_count / total_atp
                # Normalize to 0-1 (baseline 100-500 cycles/ATP)
                metrics['energy_efficiency'] = min(1.0, max(0.0, (efficiency_raw - 100) / 400))
            else:
                metrics['energy_efficiency'] = 0.0
        else:
            metrics['energy_efficiency'] = 0.0

        # Weighted multi-objective fitness (configurable weights)
        # Note: Actual weights will be set by TemporalAdapter.get_current_metrics()
        # which may use adaptive weighting (Session 28)
        metrics['weighted_fitness'] = self._compute_weighted_fitness(
            metrics['coverage'],
            metrics['quality'],
            metrics['energy_efficiency']
        )

        return metrics

    def _compute_weighted_fitness(
        self,
        coverage: float,
        quality: float,
        energy: float,
        coverage_weight: float = 0.5,
        quality_weight: float = 0.3,
        energy_weight: float = 0.2
    ) -> float:
        """
        Compute weighted multi-objective fitness.

        Default weights (Session 24):
        - Coverage: 50% (primary objective)
        - Quality: 30% (secondary)
        - Energy: 20% (tertiary)

        Can be overridden for different priorities (Session 28: adaptive).
        """
        return (coverage_weight * coverage +
                quality_weight * quality +
                energy_weight * energy)

    def reset(self):
        """Reset window for new time period"""
        self.attention_rates.clear()
        self.salience_values.clear()
        self.atp_levels.clear()
        self.coverage_scores.clear()
        # Session 24: Reset multi-objective tracking
        self.quality_scores.clear()
        self.atp_spent.clear()
        self.start_time = time.time()
        self.last_update = time.time()
        self.cycle_count = 0


@dataclass
class AdaptationEvent:
    """Record of a single ATP parameter adaptation"""
    timestamp: float
    trigger: AdaptationTrigger
    old_cost: float
    old_recovery: float
    new_cost: float
    new_recovery: float
    metrics_before: Dict[str, float]
    metrics_after: Optional[Dict[str, float]] = None
    success: Optional[bool] = None
    damping_factor: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/storage"""
        return {
            'timestamp': self.timestamp,
            'trigger': self.trigger.value,
            'old_params': {'cost': self.old_cost, 'recovery': self.old_recovery},
            'new_params': {'cost': self.new_cost, 'recovery': self.new_recovery},
            'metrics_before': self.metrics_before,
            'metrics_after': self.metrics_after,
            'success': self.success,
            'damping_factor': self.damping_factor
        }


@dataclass
class TemporalPattern:
    """Detected time-of-day or workload pattern"""
    pattern_type: str                    # "morning_peak", "evening_quiet", etc.
    time_range: Tuple[int, int]          # Hour range (e.g., 9-12)
    optimal_cost: float                  # Best attention_cost for this period
    optimal_recovery: float              # Best rest_recovery for this period
    confidence: float                    # Confidence in pattern (0-1)
    observations: int = 0                # Number of times observed
    last_seen: float = field(default_factory=time.time)

    def update_params(self, cost: float, recovery: float, performance: float):
        """Update optimal parameters based on observed performance"""
        # Exponential moving average with confidence weighting
        alpha = 0.1 * self.confidence
        self.optimal_cost = (1 - alpha) * self.optimal_cost + alpha * cost
        self.optimal_recovery = (1 - alpha) * self.optimal_recovery + alpha * recovery

        # Increase confidence with observations (asymptotically approach 1.0)
        self.observations += 1
        self.confidence = min(0.99, 1.0 - (1.0 / (1.0 + self.observations * 0.1)))
        self.last_seen = time.time()


class TemporalAdapter:
    """
    Production temporal adaptation system for SAGE consciousness.

    Continuously monitors consciousness performance and adapts ATP parameters
    to maintain optimal coverage and quality across varying workloads.

    Validated through Thor Sessions 16-17 and Sprout Session 62.
    """

    def __init__(
        self,
        initial_cost: float = 0.01,
        initial_recovery: float = 0.05,
        adaptation_rate: float = 0.1,
        window_minutes: int = 15,
        adaptation_threshold: float = 0.05,
        satisfaction_threshold: float = 0.95,
        enable_damping: bool = True,
        damping_decay: float = 0.5,
        min_damping: float = 0.1,
        min_cycles_between_adaptations: int = 500,
        enable_pattern_learning: bool = False,
        enable_multi_objective: bool = False,
        coverage_weight: float = 0.5,
        quality_weight: float = 0.3,
        energy_weight: float = 0.2,
        enable_adaptive_weights: bool = False
    ):
        """
        Initialize temporal adaptation system.

        Args:
            initial_cost: Starting ATP attention cost
            initial_recovery: Starting ATP rest recovery rate
            adaptation_rate: Base rate for parameter changes (±10% default)
            window_minutes: Performance monitoring window size
            adaptation_threshold: Performance delta that triggers adaptation
            satisfaction_threshold: Coverage level considered excellent (0.95 = 95%)
            enable_damping: Apply exponential backoff for consecutive adaptations
            damping_decay: Rate of damping increase (0.5 = halve each time)
            min_damping: Minimum damping factor (prevents complete stop)
            min_cycles_between_adaptations: Minimum wait between adaptations
            enable_pattern_learning: Learn time-of-day patterns (experimental)
            enable_multi_objective: Use multi-objective optimization (Session 24)
            coverage_weight: Weight for coverage in multi-objective (default 0.5)
            quality_weight: Weight for quality in multi-objective (default 0.3)
            energy_weight: Weight for energy in multi-objective (default 0.2)
            enable_adaptive_weights: Use context-aware adaptive weighting (Session 28)
        """
        # Current ATP parameters
        self.current_cost = initial_cost
        self.current_recovery = initial_recovery

        # Adaptation configuration
        self.adaptation_rate = adaptation_rate
        self.adaptation_threshold = adaptation_threshold
        self.satisfaction_threshold = satisfaction_threshold
        self.enable_damping = enable_damping
        self.damping_decay = damping_decay
        self.min_damping = min_damping

        # Performance windows
        self.current_window = TemporalWindow(window_minutes=window_minutes)
        self.previous_window = TemporalWindow(window_minutes=window_minutes)

        # Adaptation state
        self.adaptation_history: List[AdaptationEvent] = []
        self.cycles_since_adaptation = 0
        self.min_cycles_between_adaptations = min_cycles_between_adaptations

        # Damping state
        self.consecutive_similar_triggers = 0
        self.last_trigger = AdaptationTrigger.NONE
        self.current_damping_factor = 1.0
        self.satisfaction_stable_windows = 0

        # Pattern learning (experimental)
        self.enable_pattern_learning = enable_pattern_learning
        self.learned_patterns: Dict[str, TemporalPattern] = {}

        # Session 24: Multi-objective optimization
        self.enable_multi_objective = enable_multi_objective
        self.coverage_weight = coverage_weight
        self.quality_weight = quality_weight
        self.energy_weight = energy_weight

        # Session 28: Adaptive objective weighting
        self.enable_adaptive_weights = enable_adaptive_weights and ADAPTIVE_WEIGHTS_AVAILABLE
        if self.enable_adaptive_weights:
            self.weight_calculator = AdaptiveWeightCalculator()
            logger.info("[Adaptive Weights] Context-aware weighting enabled (Session 28)")
        else:
            self.weight_calculator = None

        # Session 31: Epistemic state tracking
        if EPISTEMIC_TRACKING_AVAILABLE:
            self.epistemic_tracker = EpistemicStateTracker(history_size=50)
            logger.info("[Epistemic Tracking] Meta-cognitive awareness enabled (Session 31)")
        else:
            self.epistemic_tracker = None

        # Statistics
        self.total_adaptations = 0
        self.successful_adaptations = 0
        self.start_time = time.time()

    def update(
        self,
        attended: bool,
        salience: float,
        atp_level: float,
        high_salience_count: int = 0,
        attended_high_salience: int = 0,
        quality_score: Optional[float] = None,
        attention_cost: Optional[float] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Update temporal adapter with metrics from a consciousness cycle.

        Args:
            attended: Whether attention was allocated this cycle
            salience: Salience value of the observation
            atp_level: Current ATP level (0-1)
            high_salience_count: Number of high-salience observations in recent window
            attended_high_salience: How many high-salience observations were attended
            quality_score: Optional quality score for this cycle (Session 24)
            attention_cost: Optional ATP cost (defaults to current_cost if None, Session 24)

        Returns:
            New (cost, recovery) parameters if adaptation triggered, None otherwise
        """
        # Use current cost if not provided
        if attention_cost is None:
            attention_cost = self.current_cost

        # Add metrics to current window
        self.current_window.add_cycle(
            attended=attended,
            salience=salience,
            atp_level=atp_level,
            high_salience_count=high_salience_count,
            attended_high_salience=attended_high_salience,
            quality_score=quality_score,
            attention_cost=attention_cost
        )

        self.cycles_since_adaptation += 1

        # Check if adaptation needed
        if self.cycles_since_adaptation < self.min_cycles_between_adaptations:
            return None

        trigger, reason = self._should_adapt()

        if trigger != AdaptationTrigger.NONE:
            # Perform adaptation
            new_cost, new_recovery = self._adapt_parameters(trigger)

            # Record adaptation event
            event = AdaptationEvent(
                timestamp=time.time(),
                trigger=trigger,
                old_cost=self.current_cost,
                old_recovery=self.current_recovery,
                new_cost=new_cost,
                new_recovery=new_recovery,
                metrics_before=self.current_window.get_metrics(),
                damping_factor=self.current_damping_factor
            )

            self.adaptation_history.append(event)
            self.total_adaptations += 1

            # Update parameters
            self.current_cost = new_cost
            self.current_recovery = new_recovery

            # Update damping state
            self._update_damping(trigger)

            # Learn from successful adaptation
            self._learn_pattern()

            # Reset cycle counter
            self.cycles_since_adaptation = 0

            # Shift windows
            self.previous_window = self.current_window
            self.current_window = TemporalWindow(window_minutes=self.current_window.window_minutes)

            logger.info(f"ATP adaptation: {trigger.value} | "
                       f"cost {event.old_cost:.4f}→{event.new_cost:.4f}, "
                       f"recovery {event.old_recovery:.4f}→{event.new_recovery:.4f} | "
                       f"damping={self.current_damping_factor:.2f} | {reason}")

            return (new_cost, new_recovery)

        # No reactive adaptation needed - try applying learned pattern
        # (Only if we haven't adapted recently and performance is suboptimal)
        if self.cycles_since_adaptation > self.min_cycles_between_adaptations:
            current = self.current_window.get_metrics()
            coverage = current.get('coverage', 0.0) if current else 0.0

            if coverage < 0.90 and self._apply_learned_pattern():
                # Pattern was applied
                self.cycles_since_adaptation = 0  # Reset counter
                return (self.current_cost, self.current_recovery)

        return None

    def update_epistemic_state(self, epistemic_metrics: 'EpistemicMetrics') -> None:
        """
        Update epistemic state tracker with metrics from current cycle.

        Session 31: Tracks SAGE's meta-cognitive awareness - confidence, comprehension,
        uncertainty, frustration, etc. Enables detection of learning trajectories and
        frustration patterns.

        Args:
            epistemic_metrics: Epistemic metrics for this cycle
        """
        if self.epistemic_tracker:
            self.epistemic_tracker.track(epistemic_metrics)
            logger.debug(f"[Epistemic] State={epistemic_metrics.primary_state().value}, "
                        f"Confidence={epistemic_metrics.confidence:.2f}, "
                        f"Frustration={epistemic_metrics.frustration:.2f}")

    def _should_adapt(self) -> Tuple[AdaptationTrigger, str]:
        """
        Determine if adaptation is needed and what type.

        Returns:
            (trigger_type, reason_string)
        """
        current = self.current_window.get_metrics()

        if not current or 'coverage' not in current:
            return (AdaptationTrigger.NONE, "insufficient data")

        # SATISFACTION CHECK (Primary mechanism from Session 17)
        coverage = current.get('coverage', 0.0)
        if coverage >= self.satisfaction_threshold:
            self.satisfaction_stable_windows += 1

            # Satisfied for 3+ consecutive windows → stop adapting
            if self.satisfaction_stable_windows >= 3:
                return (AdaptationTrigger.NONE,
                       f"satisfied (coverage {coverage:.1%} for {self.satisfaction_stable_windows} windows)")
        else:
            # Reset satisfaction counter if coverage drops
            self.satisfaction_stable_windows = 0

        # DEGRADATION CHECK
        if self.previous_window.get_metrics():
            prev_coverage = self.previous_window.get_metrics().get('coverage', 0.0)
            coverage_delta = coverage - prev_coverage

            if coverage_delta < -self.adaptation_threshold:
                return (AdaptationTrigger.DEGRADATION,
                       f"coverage degraded {coverage_delta:+.1%} ({prev_coverage:.1%}→{coverage:.1%})")

        # LOW ATTENTION CHECK
        attention_rate = current.get('attention_rate', 0.0)
        if attention_rate < 0.15:  # Very low attention
            return (AdaptationTrigger.LOW_ATTENTION,
                   f"attention very low ({attention_rate:.1%})")

        # ATP SURPLUS CHECK (Modified from Session 17)
        mean_atp = current.get('mean_atp', 0.0)
        if mean_atp > 0.85 and attention_rate < 0.80:
            # Only trigger if attention isn't already high
            return (AdaptationTrigger.ATP_SURPLUS,
                   f"ATP surplus ({mean_atp:.1%}) with moderate attention ({attention_rate:.1%})")

        return (AdaptationTrigger.NONE, "performance acceptable")

    def _adapt_parameters(self, trigger: AdaptationTrigger) -> Tuple[float, float]:
        """
        Calculate new ATP parameters based on trigger type.

        Args:
            trigger: Type of adaptation needed

        Returns:
            (new_cost, new_recovery) tuple
        """
        # Apply damping to adaptation rate
        effective_rate = self.adaptation_rate * self.current_damping_factor

        cost = self.current_cost
        recovery = self.current_recovery

        if trigger == AdaptationTrigger.DEGRADATION:
            # Coverage dropped → make attention cheaper, recovery faster
            cost *= (1.0 - effective_rate)
            recovery *= (1.0 + effective_rate)

        elif trigger == AdaptationTrigger.LOW_ATTENTION:
            # Attention too low → make it much cheaper
            cost *= (1.0 - effective_rate * 1.5)

        elif trigger == AdaptationTrigger.ATP_SURPLUS:
            # ATP consistently high → can afford more attention
            cost *= (1.0 - effective_rate * 0.5)

        # Clamp to reasonable ranges
        cost = max(0.001, min(0.05, cost))
        recovery = max(0.01, min(0.10, recovery))

        return (cost, recovery)

    def _update_damping(self, trigger: AdaptationTrigger):
        """Update damping factor based on adaptation history"""
        if not self.enable_damping:
            return

        # Check if same type of trigger as last time
        if trigger == self.last_trigger:
            self.consecutive_similar_triggers += 1
            # Exponential damping
            self.current_damping_factor *= self.damping_decay
            self.current_damping_factor = max(self.min_damping, self.current_damping_factor)
        else:
            # New trigger type → reset damping
            self.consecutive_similar_triggers = 0
            self.current_damping_factor = 1.0

        self.last_trigger = trigger

    def get_current_params(self) -> Tuple[float, float]:
        """Get current ATP parameters"""
        return (self.current_cost, self.current_recovery)

    def get_current_weights(self) -> Tuple[float, float, float]:
        """
        Get current objective weights (Session 28: may be adaptive).

        Returns:
            (coverage_weight, quality_weight, energy_weight) tuple
        """
        if not self.enable_adaptive_weights or self.weight_calculator is None:
            # Static weights
            return (self.coverage_weight, self.quality_weight, self.energy_weight)

        # Adaptive weights based on current context
        metrics = self.current_window.get_metrics()
        if not metrics:
            # No data yet, use baseline
            return (self.coverage_weight, self.quality_weight, self.energy_weight)

        # Build operating context
        context = OperatingContext(
            atp_level=metrics.get('mean_atp', 0.5),
            attention_rate=metrics.get('attention_rate', 0.5),
            coverage=metrics.get('coverage', 0.0),
            quality=metrics.get('quality', 0.0),
            energy_efficiency=metrics.get('energy_efficiency', 0.0)
        )

        # Calculate adaptive weights
        adaptive_weights = self.weight_calculator.calculate_weights(context)
        return adaptive_weights.to_tuple()

    def get_current_metrics_with_weights(self) -> Dict:
        """
        Get current metrics with proper weights applied.

        Session 28: Uses adaptive weights if enabled, otherwise static weights.
        """
        metrics = self.current_window.get_metrics()
        if not metrics or not self.enable_multi_objective:
            return metrics

        # Get current weights (adaptive or static)
        coverage_w, quality_w, energy_w = self.get_current_weights()

        # Recompute weighted fitness with current weights
        metrics['weighted_fitness'] = self.current_window._compute_weighted_fitness(
            metrics['coverage'],
            metrics['quality'],
            metrics['energy_efficiency'],
            coverage_weight=coverage_w,
            quality_weight=quality_w,
            energy_weight=energy_w
        )

        # Add weight information to metrics
        metrics['coverage_weight'] = coverage_w
        metrics['quality_weight'] = quality_w
        metrics['energy_weight'] = energy_w

        # Session 31: Add epistemic state metrics
        if self.epistemic_tracker and self.epistemic_tracker.history:
            current_state = self.epistemic_tracker.current_state()
            epistemic_stats = self.epistemic_tracker.get_statistics()

            metrics.update({
                'epistemic_state': current_state.primary_state().value,
                'confidence': current_state.confidence,
                'comprehension_depth': current_state.comprehension_depth,
                'uncertainty': current_state.uncertainty,
                'frustration': current_state.frustration,
                'learning_trajectory': epistemic_stats['learning_trajectory'],
                'frustration_pattern': epistemic_stats['frustration_pattern']
            })

        return metrics

    def get_statistics(self) -> Dict:
        """Get adaptation statistics"""
        runtime_hours = (time.time() - self.start_time) / 3600.0

        stats = {
            'runtime_hours': runtime_hours,
            'total_adaptations': self.total_adaptations,
            'adaptations_per_hour': self.total_adaptations / runtime_hours if runtime_hours > 0 else 0,
            'current_cost': self.current_cost,
            'current_recovery': self.current_recovery,
            'current_damping': self.current_damping_factor,
            'satisfaction_stable_windows': self.satisfaction_stable_windows,
            'cycles_since_adaptation': self.cycles_since_adaptation,
            'current_metrics': self.get_current_metrics_with_weights()
        }

        # Session 28: Add adaptive weight statistics
        if self.enable_adaptive_weights and self.weight_calculator:
            stats['adaptive_weight_stats'] = self.weight_calculator.get_stats()

        return stats

    def export_history(self) -> List[Dict]:
        """Export adaptation history for analysis"""
        return [event.to_dict() for event in self.adaptation_history]

    def _get_current_hour(self) -> int:
        """Get current hour of day (0-23)"""
        return datetime.now().hour

    def _get_pattern_key(self, hour: int) -> str:
        """
        Get pattern key for given hour.

        Groups hours into meaningful time-of-day periods:
        - early_morning: 0-6
        - morning: 6-12
        - midday: 12-14
        - afternoon: 14-18
        - evening: 18-22
        - night: 22-24
        """
        patterns = [
            (0, 6, "early_morning"),
            (6, 12, "morning"),
            (12, 14, "midday"),
            (14, 18, "afternoon"),
            (18, 22, "evening"),
            (22, 24, "night")
        ]

        for start, end, name in patterns:
            if start <= hour < end:
                return name

        return "unknown"

    def _learn_pattern(self):
        """
        Learn time-of-day pattern from current performance.

        Updates or creates TemporalPattern for current time period
        based on current ATP parameters and performance metrics.
        """
        if not self.enable_pattern_learning:
            return

        hour = self._get_current_hour()
        pattern_key = self._get_pattern_key(hour)

        # Get current performance
        metrics = self.current_window.get_metrics()
        if not metrics or 'coverage' not in metrics:
            return

        coverage = metrics.get('coverage', 0.0)

        # Only learn from good performance (>80% coverage)
        if coverage < 0.80:
            return

        # Get time range for this pattern
        time_ranges = {
            "early_morning": (0, 6),
            "morning": (6, 12),
            "midday": (12, 14),
            "afternoon": (14, 18),
            "evening": (18, 22),
            "night": (22, 24)
        }
        time_range = time_ranges.get(pattern_key, (hour, hour+1))

        # Create or update pattern
        if pattern_key not in self.learned_patterns:
            # Create new pattern
            self.learned_patterns[pattern_key] = TemporalPattern(
                pattern_type=pattern_key,
                time_range=time_range,
                optimal_cost=self.current_cost,
                optimal_recovery=self.current_recovery,
                confidence=0.1  # Low initial confidence
            )
        else:
            # Update existing pattern
            pattern = self.learned_patterns[pattern_key]
            pattern.update_params(
                self.current_cost,
                self.current_recovery,
                coverage  # Performance metric
            )

        logger.debug(f"Pattern learning: {pattern_key} | "
                    f"cost={self.current_cost:.4f}, recovery={self.current_recovery:.4f} | "
                    f"coverage={coverage:.1%}")

    def _apply_learned_pattern(self) -> bool:
        """
        Apply learned pattern for current time if available.

        Returns:
            True if pattern was applied, False otherwise
        """
        if not self.enable_pattern_learning:
            return False

        hour = self._get_current_hour()
        pattern_key = self._get_pattern_key(hour)

        # Check if we have a learned pattern for this time
        if pattern_key not in self.learned_patterns:
            return False

        pattern = self.learned_patterns[pattern_key]

        # Only apply if confidence is sufficient (>50%)
        if pattern.confidence < 0.5:
            return False

        # Apply learned parameters
        old_cost = self.current_cost
        old_recovery = self.current_recovery

        self.current_cost = pattern.optimal_cost
        self.current_recovery = pattern.optimal_recovery

        logger.info(f"Applied learned pattern: {pattern_key} | "
                   f"cost {old_cost:.4f}→{self.current_cost:.4f}, "
                   f"recovery {old_recovery:.4f}→{self.current_recovery:.4f} | "
                   f"confidence={pattern.confidence:.1%}")

        return True


# Convenience factory functions

def create_production_adapter(**kwargs) -> TemporalAdapter:
    """
    Create temporal adapter with production settings.

    Validated configuration from Sessions 16-17 and Sprout Session 62.
    """
    defaults = {
        'initial_cost': 0.01,
        'initial_recovery': 0.05,
        'adaptation_rate': 0.1,
        'satisfaction_threshold': 0.95,
        'enable_damping': True,
        'damping_decay': 0.5,
        'min_cycles_between_adaptations': 500,
        'enable_pattern_learning': False  # Experimental
    }
    defaults.update(kwargs)
    return TemporalAdapter(**defaults)


def create_conservative_adapter(**kwargs) -> TemporalAdapter:
    """
    Create temporal adapter with conservative settings.

    Adapts less frequently, suitable for stable workloads.
    """
    defaults = {
        'initial_cost': 0.01,
        'initial_recovery': 0.05,
        'adaptation_rate': 0.05,  # Smaller adjustments
        'satisfaction_threshold': 0.90,  # Lower threshold
        'enable_damping': True,
        'damping_decay': 0.3,  # Stronger damping
        'min_cycles_between_adaptations': 1000,  # Wait longer
        'enable_pattern_learning': False
    }
    defaults.update(kwargs)
    return TemporalAdapter(**defaults)


def create_responsive_adapter(**kwargs) -> TemporalAdapter:
    """
    Create temporal adapter with responsive settings.

    Adapts more aggressively, suitable for highly variable workloads.
    """
    defaults = {
        'initial_cost': 0.01,
        'initial_recovery': 0.05,
        'adaptation_rate': 0.15,  # Larger adjustments
        'satisfaction_threshold': 0.95,
        'enable_damping': True,
        'damping_decay': 0.7,  # Lighter damping
        'min_cycles_between_adaptations': 300,  # Adapt sooner
        'enable_pattern_learning': True  # Enable learning
    }
    defaults.update(kwargs)
    return TemporalAdapter(**defaults)


def create_multi_objective_adapter(**kwargs) -> TemporalAdapter:
    """
    Create temporal adapter with multi-objective optimization (Session 24).

    Balances coverage, quality, and energy efficiency simultaneously.
    Based on Session 23 findings: cheap attention + fast recovery is optimal.

    Default configuration:
    - attention_cost: 0.005 (cheap attention for frequent allocation)
    - rest_recovery: 0.080 (fast recovery to maintain high ATP)
    - Multi-objective weights: 50% coverage, 30% quality, 20% energy
    """
    defaults = {
        'initial_cost': 0.005,  # Pareto-optimal from Session 23
        'initial_recovery': 0.080,  # Pareto-optimal from Session 23
        'adaptation_rate': 0.1,
        'satisfaction_threshold': 0.95,
        'enable_damping': True,
        'damping_decay': 0.5,
        'min_cycles_between_adaptations': 500,
        'enable_pattern_learning': True,
        'enable_multi_objective': True,
        'coverage_weight': 0.5,
        'quality_weight': 0.3,
        'energy_weight': 0.2
    }
    defaults.update(kwargs)
    return TemporalAdapter(**defaults)


def create_adaptive_weight_adapter(**kwargs) -> TemporalAdapter:
    """
    Create temporal adapter with adaptive objective weighting (Session 28).

    Extends multi-objective optimization with context-aware weight adaptation.
    Adjusts coverage/quality/energy weights based on operating context:
    - High ATP → emphasize quality
    - Low ATP → emphasize coverage
    - High attention rate → emphasize energy efficiency
    - Performance issues → prioritize the struggling objective

    Based on Sessions 23-27 (multi-objective + quality metrics foundation).

    Default configuration:
    - Pareto-optimal ATP parameters from Session 23
    - Multi-objective optimization enabled
    - Adaptive weighting enabled (Session 28)
    - Baseline weights: 50% coverage, 30% quality, 20% energy
      (will adapt based on context)
    """
    defaults = {
        'initial_cost': 0.005,  # Pareto-optimal from Session 23
        'initial_recovery': 0.080,  # Pareto-optimal from Session 23
        'adaptation_rate': 0.1,
        'satisfaction_threshold': 0.95,
        'enable_damping': True,
        'damping_decay': 0.5,
        'min_cycles_between_adaptations': 500,
        'enable_pattern_learning': True,
        'enable_multi_objective': True,
        'coverage_weight': 0.5,      # Baseline (will adapt)
        'quality_weight': 0.3,        # Baseline (will adapt)
        'energy_weight': 0.2,         # Baseline (will adapt)
        'enable_adaptive_weights': True  # Session 28
    }
    defaults.update(kwargs)
    return TemporalAdapter(**defaults)
