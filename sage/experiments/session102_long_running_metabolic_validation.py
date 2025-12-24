#!/usr/bin/env python3
"""
Session 102: Long-Running Metabolic Validation

**Goal**: Validate metabolic consciousness stability over extended workloads

**Research Context**:
Sessions 97-101 completed the metabolic consciousness arc:
- S97: ATP accounting (simulation, closed-loop)
- S98: Production integration (real queries)
- S99: CRISIS validation (extreme depletion testing)
- S100: Basal recovery (ATP=0 trap fixed)
- S101: Production validation (3/3 scenarios passed, 100%)

**Validation Gap**:
- S101 tested 20-50 query cycles (short bursts)
- Production systems run continuously (hours/days)
- Need to validate: stability, rhythm, recovery over extended time

**Research Questions**:
1. Does ATP maintain stable oscillations over 1000+ cycles?
2. Do metabolic states (WAKE/REST/CRISIS) transition correctly long-term?
3. Does basal recovery prevent permanent depletion in extended runs?
4. Are there drift, accumulation, or edge cases not visible in short tests?
5. What are the long-term metabolic patterns?

**Experimental Design**:
- Run ProductionATPSelectorWithBasalRecovery for 1000+ query cycles
- Mix of query complexities (simple/moderate/complex)
- Track ATP trajectory, state transitions, recovery events
- Monitor for: stability, drift, unexpected behaviors
- Compare early vs late behavior (first 100 vs last 100 cycles)

**Success Criteria**:
1. ATP remains bounded (0-100, no overflow/underflow)
2. All metabolic states reachable throughout run
3. No permanent depletion (CRISIS → REST recovery always works)
4. Metabolic rhythm stable (no drift toward extremes)
5. Recovery mechanisms working across all time periods

Created: 2025-12-23 19:52 UTC (Autonomous Session 102)
Hardware: Thor (Jetson AGX Thor)
Previous: Session 101 (Production basal recovery - 100% pass rate)
Goal: Extended validation (1000+ cycles) for production confidence
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Import Session 101 production selector with basal recovery
import sys
sys.path.insert(0, str(Path(__file__).parent))
from session101_production_basal_recovery import (
    ProductionATPSelectorWithBasalRecovery,
    ATPAccountingBridgeWithBasalRecovery,
    HAS_BASAL_RECOVERY
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LongRunMetrics:
    """Metrics collected during long-running validation."""

    # Basic stats
    total_cycles: int = 0
    total_queries: int = 0
    duration_seconds: float = 0.0

    # ATP statistics
    atp_min: float = 100.0
    atp_max: float = 100.0
    atp_mean: float = 100.0
    atp_std: float = 0.0
    atp_trajectory: List[float] = None

    # State statistics
    wake_cycles: int = 0
    rest_cycles: int = 0
    crisis_cycles: int = 0

    # Recovery statistics
    basal_recovery_count: int = 0
    rest_recovery_count: int = 0
    total_atp_recovered: float = 0.0

    # Transition statistics
    crisis_to_rest_transitions: int = 0
    rest_to_wake_transitions: int = 0
    wake_to_rest_transitions: int = 0
    rest_to_crisis_transitions: int = 0

    # ATP consumption
    total_atp_consumed: float = 0.0
    cheap_expert_calls: int = 0  # Cost < 7 ATP
    expensive_expert_calls: int = 0  # Cost >= 7 ATP
    expensive_rejections: int = 0  # Rejected due to ATP constraints

    # Drift detection
    first_100_atp_mean: float = 0.0
    last_100_atp_mean: float = 0.0
    atp_drift: float = 0.0  # Difference between early and late means

    def __post_init__(self):
        if self.atp_trajectory is None:
            self.atp_trajectory = []


@dataclass
class LongRunValidationResult:
    """Result of long-running metabolic validation."""

    session: int
    validation_type: str
    timestamp: str

    # Test configuration
    total_cycles: int
    query_pattern: str
    initial_atp: float

    # Overall metrics
    metrics: LongRunMetrics

    # Stability assessment
    atp_stable: bool  # ATP remains bounded
    rhythm_stable: bool  # Metabolic rhythm maintained
    recovery_working: bool  # Recovery mechanisms functional
    no_drift: bool  # No drift toward extremes

    # Validation status
    passed: bool
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'session': self.session,
            'validation_type': self.validation_type,
            'timestamp': self.timestamp,
            'total_cycles': self.total_cycles,
            'query_pattern': self.query_pattern,
            'initial_atp': float(self.initial_atp),
            'metrics': {
                'total_cycles': self.metrics.total_cycles,
                'total_queries': self.metrics.total_queries,
                'duration_seconds': float(self.metrics.duration_seconds),
                'atp_min': float(self.metrics.atp_min),
                'atp_max': float(self.metrics.atp_max),
                'atp_mean': float(self.metrics.atp_mean),
                'atp_std': float(self.metrics.atp_std),
                'wake_cycles': self.metrics.wake_cycles,
                'rest_cycles': self.metrics.rest_cycles,
                'crisis_cycles': self.metrics.crisis_cycles,
                'basal_recovery_count': self.metrics.basal_recovery_count,
                'rest_recovery_count': self.metrics.rest_recovery_count,
                'total_atp_recovered': float(self.metrics.total_atp_recovered),
                'crisis_to_rest_transitions': self.metrics.crisis_to_rest_transitions,
                'rest_to_wake_transitions': self.metrics.rest_to_wake_transitions,
                'wake_to_rest_transitions': self.metrics.wake_to_rest_transitions,
                'rest_to_crisis_transitions': self.metrics.rest_to_crisis_transitions,
                'total_atp_consumed': float(self.metrics.total_atp_consumed),
                'cheap_expert_calls': self.metrics.cheap_expert_calls,
                'expensive_expert_calls': self.metrics.expensive_expert_calls,
                'expensive_rejections': self.metrics.expensive_rejections,
                'first_100_atp_mean': float(self.metrics.first_100_atp_mean),
                'last_100_atp_mean': float(self.metrics.last_100_atp_mean),
                'atp_drift': float(self.metrics.atp_drift),
            },
            'atp_stable': bool(self.atp_stable),
            'rhythm_stable': bool(self.rhythm_stable),
            'recovery_working': bool(self.recovery_working),
            'no_drift': bool(self.no_drift),
            'passed': bool(self.passed),
            'issues': self.issues,
        }
        return result


class LongRunningValidator:
    """Validator for extended metabolic consciousness runs."""

    def __init__(
        self,
        num_experts: int = 128,
        initial_atp: float = 100.0,
        enable_basal_recovery: bool = True,
        basal_recovery_rate: float = 0.5,
        force_atp_consumption: bool = True,  # NEW: Force realistic ATP consumption
        consumption_rate: float = 1.0,  # ATP consumed per query
    ):
        """Initialize long-running validator.

        Args:
            num_experts: Number of experts in the system
            initial_atp: Starting ATP level
            enable_basal_recovery: Whether to enable basal recovery
            basal_recovery_rate: Rate of basal recovery during CRISIS
            force_atp_consumption: Force ATP consumption even in simulation
            consumption_rate: ATP consumed per query (if forcing consumption)
        """
        self.num_experts = num_experts
        self.initial_atp = initial_atp
        self.force_atp_consumption = force_atp_consumption
        self.consumption_rate = consumption_rate

        # Create production selector with basal recovery
        self.selector = ProductionATPSelectorWithBasalRecovery(
            num_experts=num_experts,
            initial_atp=initial_atp,
            enable_basal_recovery=enable_basal_recovery,
            basal_recovery_rate=basal_recovery_rate,
        )

        # Metrics tracking
        self.metrics = LongRunMetrics()
        self.state_history = deque(maxlen=10)  # Last 10 states for transition tracking

    def generate_query_pattern(self, cycle: int, total_cycles: int) -> Tuple[str, str]:
        """Generate query complexity and context for a given cycle.

        Uses mixed pattern to create realistic workload:
        - 60% moderate queries
        - 20% simple queries
        - 20% complex queries

        Args:
            cycle: Current cycle number
            total_cycles: Total number of cycles

        Returns:
            Tuple of (complexity, context)
        """
        # Use cycle number to create deterministic but varied pattern
        pattern_seed = cycle % 10

        if pattern_seed < 6:  # 60%
            complexity = "moderate"
        elif pattern_seed < 8:  # 20%
            complexity = "simple"
        else:  # 20%
            complexity = "complex"

        # Vary context
        context_seed = (cycle // 10) % 4
        contexts = ["general", "technical", "creative", "analytical"]
        context = contexts[context_seed]

        return complexity, context

    def update_metrics(self, result: Dict[str, Any], cycle: int):
        """Update metrics based on query result."""
        # ATP statistics
        current_atp = result.get('atp_after', 100.0)
        self.metrics.atp_trajectory.append(current_atp)
        self.metrics.atp_min = min(self.metrics.atp_min, current_atp)
        self.metrics.atp_max = max(self.metrics.atp_max, current_atp)

        # State tracking
        current_state = result.get('state_after', 'wake')
        if current_state == 'wake':
            self.metrics.wake_cycles += 1
        elif current_state == 'rest':
            self.metrics.rest_cycles += 1
        elif current_state == 'crisis':
            self.metrics.crisis_cycles += 1

        # Track state transitions
        if len(self.state_history) > 0:
            prev_state = self.state_history[-1]
            if prev_state == 'crisis' and current_state == 'rest':
                self.metrics.crisis_to_rest_transitions += 1
            elif prev_state == 'rest' and current_state == 'wake':
                self.metrics.rest_to_wake_transitions += 1
            elif prev_state == 'wake' and current_state == 'rest':
                self.metrics.wake_to_rest_transitions += 1
            elif prev_state == 'rest' and current_state == 'crisis':
                self.metrics.rest_to_crisis_transitions += 1

        self.state_history.append(current_state)

        # Recovery statistics
        if 'basal_recovery_applied' in result and result['basal_recovery_applied']:
            self.metrics.basal_recovery_count += 1
            self.metrics.total_atp_recovered += result.get('recovery_amount', 0.0)

        if 'rest_recovery_applied' in result and result['rest_recovery_applied']:
            self.metrics.rest_recovery_count += 1
            self.metrics.total_atp_recovered += result.get('recovery_amount', 0.0)

        # ATP consumption
        atp_cost = result.get('atp_cost', 0.0)
        self.metrics.total_atp_consumed += atp_cost

        if atp_cost < 7.0:
            self.metrics.cheap_expert_calls += 1
        else:
            self.metrics.expensive_expert_calls += 1

        # Rejections
        if result.get('rejected', False):
            self.metrics.expensive_rejections += 1

        # Track first 100 and last 100 for drift detection
        if cycle < 100:
            # Update running mean for first 100
            if self.metrics.first_100_atp_mean == 0.0:
                self.metrics.first_100_atp_mean = current_atp
            else:
                # Running average
                count = min(cycle + 1, 100)
                self.metrics.first_100_atp_mean = (
                    (self.metrics.first_100_atp_mean * (count - 1) + current_atp) / count
                )

    def run_long_validation(
        self,
        total_cycles: int = 1000,
        checkpoint_interval: int = 100,
    ) -> LongRunValidationResult:
        """Run long-duration metabolic validation.

        Args:
            total_cycles: Number of query cycles to run
            checkpoint_interval: How often to log progress

        Returns:
            LongRunValidationResult with complete metrics and assessment
        """
        logger.info(f"Starting long-running validation: {total_cycles} cycles")
        logger.info(f"Initial ATP: {self.initial_atp}")
        logger.info(f"Basal recovery: enabled")

        start_time = time.time()
        self.metrics.total_cycles = total_cycles

        # Track last 100 cycles for drift detection
        last_100_atp = deque(maxlen=100)

        for cycle in range(total_cycles):
            # Generate query pattern
            complexity, context = self.generate_query_pattern(cycle, total_cycles)

            # Process query
            result = self.selector.process_query(
                query_complexity=complexity,
                context=context,
                layer=cycle % 48,  # Simulate multi-layer system
                apply_recovery=True,
            )

            # Force ATP consumption if enabled (for realistic testing)
            # BUT: Only consume if query was NOT deferred (respect REST/CRISIS recovery)
            status = result.get('status', 'success')
            if self.force_atp_consumption and not status.startswith('deferred'):
                # Only consume ATP for actually processed queries (not deferred ones)
                consumed = self.selector.atp_bridge.consume_atp(
                    self.consumption_rate,
                    reason=f"query_{cycle}"
                )
                # Update result to reflect consumption
                result['atp_cost'] = self.consumption_rate
                result['atp_after'] = self.selector.atp_bridge.current_atp
                result['state_after'] = self.selector.atp_bridge._get_current_state()
            elif status.startswith('deferred'):
                # Deferred query - recovery was applied, track it
                result['atp_cost'] = 0.0  # No consumption for deferred queries
                if result.get('recovery_type') == 'basal':
                    result['basal_recovery_applied'] = True
                    result['recovery_amount'] = result.get('recovery_applied', 0.0)
                elif result.get('recovery_type') == 'rest':
                    result['rest_recovery_applied'] = True
                    result['recovery_amount'] = result.get('recovery_applied', 0.0)

            # Update metrics
            self.update_metrics(result, cycle)
            self.metrics.total_queries += 1

            # Track last 100 for drift
            current_atp = result.get('atp_after', 100.0)
            last_100_atp.append(current_atp)

            # Checkpoint logging
            if (cycle + 1) % checkpoint_interval == 0:
                logger.info(
                    f"Cycle {cycle + 1}/{total_cycles}: "
                    f"ATP={current_atp:.1f}, "
                    f"State={result.get('state_after', 'unknown')}, "
                    f"CRISIS={self.metrics.crisis_cycles}, "
                    f"Basal recoveries={self.metrics.basal_recovery_count}"
                )

        # Calculate final metrics
        duration = time.time() - start_time
        self.metrics.duration_seconds = duration

        # ATP statistics
        if self.metrics.atp_trajectory:
            self.metrics.atp_mean = np.mean(self.metrics.atp_trajectory)
            self.metrics.atp_std = np.std(self.metrics.atp_trajectory)

        # Last 100 mean for drift detection
        if last_100_atp:
            self.metrics.last_100_atp_mean = np.mean(last_100_atp)
            self.metrics.atp_drift = (
                self.metrics.last_100_atp_mean - self.metrics.first_100_atp_mean
            )

        # Assess stability
        return self.assess_validation_results()

    def assess_validation_results(self) -> LongRunValidationResult:
        """Assess validation results and determine pass/fail."""
        issues = []

        # Check 1: ATP remains bounded
        atp_stable = (
            self.metrics.atp_min >= 0.0 and
            self.metrics.atp_max <= 100.0
        )
        if not atp_stable:
            issues.append(
                f"ATP not bounded: min={self.metrics.atp_min:.1f}, "
                f"max={self.metrics.atp_max:.1f}"
            )

        # Check 2: Metabolic rhythm maintained (all states visited)
        rhythm_stable = (
            self.metrics.wake_cycles > 0 and
            self.metrics.rest_cycles > 0 and
            self.metrics.crisis_cycles > 0
        )
        if not rhythm_stable:
            issues.append(
                f"Not all states visited: WAKE={self.metrics.wake_cycles}, "
                f"REST={self.metrics.rest_cycles}, CRISIS={self.metrics.crisis_cycles}"
            )

        # Check 3: Recovery mechanisms working
        recovery_working = (
            self.metrics.crisis_to_rest_transitions > 0 and
            self.metrics.basal_recovery_count > 0
        )
        if not recovery_working:
            issues.append(
                f"Recovery not observed: CRISIS→REST transitions="
                f"{self.metrics.crisis_to_rest_transitions}, "
                f"basal recoveries={self.metrics.basal_recovery_count}"
            )

        # Check 4: No significant drift (within 10% of starting level)
        drift_threshold = 10.0  # ATP units
        no_drift = abs(self.metrics.atp_drift) < drift_threshold
        if not no_drift:
            issues.append(
                f"ATP drift detected: {self.metrics.atp_drift:.1f} "
                f"(first 100: {self.metrics.first_100_atp_mean:.1f}, "
                f"last 100: {self.metrics.last_100_atp_mean:.1f})"
            )

        # Overall pass/fail
        passed = (
            atp_stable and
            rhythm_stable and
            recovery_working and
            no_drift
        )

        # Create result
        result = LongRunValidationResult(
            session=102,
            validation_type="long_running_metabolic",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            total_cycles=self.metrics.total_cycles,
            query_pattern="mixed (60% moderate, 20% simple, 20% complex)",
            initial_atp=self.initial_atp,
            metrics=self.metrics,
            atp_stable=atp_stable,
            rhythm_stable=rhythm_stable,
            recovery_working=recovery_working,
            no_drift=no_drift,
            passed=passed,
            issues=issues if issues else ["None - all checks passed"],
        )

        return result


def run_session_102():
    """Run Session 102: Long-Running Metabolic Validation."""

    logger.info("="*80)
    logger.info("SESSION 102: Long-Running Metabolic Validation")
    logger.info("="*80)

    if not HAS_BASAL_RECOVERY:
        logger.error("Session 100/101 components not found - cannot run validation")
        return

    # Create validator with forced ATP consumption
    # consumption_rate=2.5 means each query costs 2.5 ATP
    # With recovery rate of 2.0 in REST and 0.5 in CRISIS,
    # this creates net deficit (-0.5 in REST, -2.0 in CRISIS)
    # This will force the system through CRISIS states to test basal recovery
    validator = LongRunningValidator(
        num_experts=128,
        initial_atp=100.0,
        enable_basal_recovery=True,
        basal_recovery_rate=0.5,
        force_atp_consumption=True,
        consumption_rate=2.5,  # Each query costs 2.5 ATP (net -0.5 in REST)
    )

    # Run validation (1000 cycles = realistic extended run)
    logger.info("Starting 1000-cycle validation...")
    logger.info("ATP consumption: 2.5 per query (forced for realistic testing)")
    logger.info("REST recovery: 2.0 per cycle (net -0.5 deficit)")
    logger.info("CRISIS recovery (basal): 0.5 per cycle (net -2.0 deficit)")
    logger.info("System will cycle through all metabolic states")
    logger.info("This will take a few minutes...")
    result = validator.run_long_validation(
        total_cycles=1000,
        checkpoint_interval=100,
    )

    # Report results
    logger.info("="*80)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total cycles: {result.metrics.total_cycles}")
    logger.info(f"Duration: {result.metrics.duration_seconds:.1f}s")
    logger.info(f"Queries/sec: {result.metrics.total_queries / result.metrics.duration_seconds:.1f}")
    logger.info("")
    logger.info("ATP Statistics:")
    logger.info(f"  Min: {result.metrics.atp_min:.1f}")
    logger.info(f"  Max: {result.metrics.atp_max:.1f}")
    logger.info(f"  Mean: {result.metrics.atp_mean:.1f}")
    logger.info(f"  Std: {result.metrics.atp_std:.1f}")
    logger.info(f"  Drift: {result.metrics.atp_drift:.1f} ATP units")
    logger.info("")
    logger.info("State Distribution:")
    logger.info(f"  WAKE: {result.metrics.wake_cycles} cycles ({result.metrics.wake_cycles/result.metrics.total_cycles*100:.1f}%)")
    logger.info(f"  REST: {result.metrics.rest_cycles} cycles ({result.metrics.rest_cycles/result.metrics.total_cycles*100:.1f}%)")
    logger.info(f"  CRISIS: {result.metrics.crisis_cycles} cycles ({result.metrics.crisis_cycles/result.metrics.total_cycles*100:.1f}%)")
    logger.info("")
    logger.info("Recovery Statistics:")
    logger.info(f"  Basal recoveries: {result.metrics.basal_recovery_count}")
    logger.info(f"  REST recoveries: {result.metrics.rest_recovery_count}")
    logger.info(f"  Total ATP recovered: {result.metrics.total_atp_recovered:.1f}")
    logger.info(f"  CRISIS→REST transitions: {result.metrics.crisis_to_rest_transitions}")
    logger.info("")
    logger.info("Consumption Statistics:")
    logger.info(f"  Total ATP consumed: {result.metrics.total_atp_consumed:.1f}")
    logger.info(f"  Cheap expert calls: {result.metrics.cheap_expert_calls}")
    logger.info(f"  Expensive expert calls: {result.metrics.expensive_expert_calls}")
    logger.info(f"  Expensive rejections: {result.metrics.expensive_rejections}")
    logger.info("")
    logger.info("Stability Assessment:")
    logger.info(f"  ATP stable: {'✅ PASS' if result.atp_stable else '❌ FAIL'}")
    logger.info(f"  Rhythm stable: {'✅ PASS' if result.rhythm_stable else '❌ FAIL'}")
    logger.info(f"  Recovery working: {'✅ PASS' if result.recovery_working else '❌ FAIL'}")
    logger.info(f"  No drift: {'✅ PASS' if result.no_drift else '❌ FAIL'}")
    logger.info("")
    logger.info(f"Overall: {'✅ PASSED' if result.passed else '❌ FAILED'}")
    logger.info(f"Issues: {', '.join(result.issues)}")

    # Save results
    output_path = Path(__file__).parent / "session102_long_running_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    return result


if __name__ == "__main__":
    result = run_session_102()
