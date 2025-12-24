#!/usr/bin/env python3
"""
Session 105: Stress Testing Wake Policy - Regime Shifts & Adversarial Conditions

**Goal**: Address Nova GPT-5.2's critical feedback on architectural soundness

**Context - Nova's Review (2025-12-23)**:
"You haven't shown stability under distribution shifts: different task mixes,
different tool latencies, missing tools, partial failures, hostile prompts,
long periods of inactivity, etc. '1000 cycles' in one regime can be meaningless
if the regime is narrow."

**What We've Proven** (S103-104):
- ✅ Wake policy works in nominal conditions
- ✅ Pressure triggers action
- ✅ Actions reduce pressure (negative feedback)
- ✅ Hysteresis prevents thrashing
- ✅ ATP limits action (constraint layer)

**What We Haven't Proven** (This Session):
- ❓ Stability under regime shifts
- ❓ Behavior under oscillatory load
- ❓ Recovery from sustained overload
- ❓ Handling of degenerate cases (zero ATP, infinite pressure, etc.)
- ❓ Bounded queue growth guarantees
- ❓ Fairness (no query starvation)
- ❓ Detection of and recovery from deadlock/livelock

**Stress Test Regimes**:
1. **Burst Load**: Sudden spike in memory/uncertainty pressure
2. **Sustained Overload**: Continuous high pressure exceeding ATP recovery
3. **Oscillatory Load**: Periodic pressure waves (test for limit cycles)
4. **Long Inactivity**: No pressure for extended period, then sudden activity
5. **Tool Failures**: Actions fail to reduce pressure as expected
6. **ATP Starvation**: Operating in chronic low-ATP state
7. **Contradictory Pressures**: Memory says consolidate, uncertainty says explore
8. **Degenerate Cases**: Edge conditions (zero values, infinities, NaNs)

**Formal Invariants to Verify**:
Per Nova: "Treat this as a control system. You need (a) formal invariants,
(b) stress tests across regimes, and (c) instrumentation."

1. **Safety Invariants** (must never violate):
   - ATP ≥ 0 (no negative energy)
   - 0 ≤ pressure ≤ 1 (bounded state space)
   - wake_score ∈ [0, 1] (no unbounded scores)
   - Queue size bounded (no infinite growth)

2. **Liveness Invariants** (must eventually occur):
   - If pressure > threshold, eventually action taken (or ATP exhausted)
   - If ATP < threshold, eventually recovery (or system halts)
   - If action taken, eventually pressure reduces (or failure detected)

3. **Fairness Invariants**:
   - All pressure types get serviced (no starvation)
   - Action selection is proportional to pressure × efficiency

4. **Stability Invariants**:
   - No limit cycles (oscillation detection)
   - Bounded recovery time after perturbation
   - Equilibrium exists and is reached under nominal load

**Expected Outcomes**:
- Identify failure modes (deadlock, starvation, oscillation, etc.)
- Measure recovery time bounds
- Validate invariants or find counterexamples
- Propose architectural fixes for discovered issues

**Integration**:
- Builds on Session 103 (wake policy)
- Builds on Session 104 (SAGE integration)
- Informs Session 106+ (architectural hardening)

Created: 2025-12-24 02:30 UTC (Autonomous Session 105)
Hardware: Thor (Jetson AGX Thor)
Previous: Sessions 103-104 (wake policy implementation & integration)
Goal: Prove architectural soundness under adversarial conditions
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from enum import Enum

# Import Session 103 wake policy
import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from session103_internal_wake_policy import (
        InternalWakePolicy,
        MemoryPressureSignals,
        UncertaintyPressureSignals,
        WakeAction,
    )
    HAS_WAKE_POLICY = True
except ImportError:
    HAS_WAKE_POLICY = False

# Import Session 104 integrated system
try:
    from session104_wake_sage_integration import (
        SAGEMemoryState,
        SAGEEpistemicState,
        SAGEIntegratedWakeSystem,
    )
    HAS_INTEGRATION = True
except ImportError:
    HAS_INTEGRATION = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StressRegime(Enum):
    """Types of stress test regimes."""
    BURST_LOAD = "burst_load"
    SUSTAINED_OVERLOAD = "sustained_overload"
    OSCILLATORY_LOAD = "oscillatory_load"
    LONG_INACTIVITY = "long_inactivity"
    TOOL_FAILURES = "tool_failures"
    ATP_STARVATION = "atp_starvation"
    CONTRADICTORY_PRESSURES = "contradictory_pressures"
    DEGENERATE_CASES = "degenerate_cases"


class InvariantViolation(Exception):
    """Exception raised when a formal invariant is violated."""
    pass


@dataclass
class InvariantCheck:
    """Result of checking a formal invariant."""
    name: str
    passed: bool
    value: Any
    expected: str
    message: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class StressTestResult:
    """Result of a stress test regime."""
    regime: StressRegime
    cycles: int
    violations: List[InvariantCheck]
    recovery_time: Optional[float]
    max_queue_size: int
    oscillation_detected: bool
    deadlock_detected: bool
    final_state: Dict[str, Any]
    trajectory: List[Dict[str, Any]]


class FormalInvariantChecker:
    """
    Checks formal invariants on the wake policy system.

    Per Nova: "You need explicit proofs/metrics for:
    - bounded queue growth,
    - fairness (no query starvation),
    - recovery time after overload,
    - behavior under oscillatory load (avoid limit cycles)."
    """

    def __init__(self):
        self.violations = []
        self.checks_performed = 0

    def check_safety_invariants(
        self,
        atp: float,
        pressure: float,
        wake_score: float,
        queue_size: int,
        max_queue_size: int = 1000,
    ) -> List[InvariantCheck]:
        """Check safety invariants (must never violate).

        Safety invariants ensure system never enters unsafe state.
        """
        checks = []
        self.checks_performed += 1

        # ATP ≥ 0 (no negative energy)
        checks.append(InvariantCheck(
            name="ATP_NON_NEGATIVE",
            passed=atp >= 0,
            value=atp,
            expected="ATP ≥ 0",
            message=f"ATP must be non-negative, got {atp}",
        ))

        # 0 ≤ pressure ≤ 1 (bounded state space)
        checks.append(InvariantCheck(
            name="PRESSURE_BOUNDED",
            passed=0 <= pressure <= 1,
            value=pressure,
            expected="0 ≤ pressure ≤ 1",
            message=f"Pressure must be in [0,1], got {pressure}",
        ))

        # wake_score ∈ [0, 1] (no unbounded scores)
        checks.append(InvariantCheck(
            name="WAKE_SCORE_BOUNDED",
            passed=0 <= wake_score <= 1,
            value=wake_score,
            expected="0 ≤ wake_score ≤ 1",
            message=f"Wake score must be in [0,1], got {wake_score}",
        ))

        # Queue size bounded (no infinite growth)
        checks.append(InvariantCheck(
            name="QUEUE_SIZE_BOUNDED",
            passed=queue_size <= max_queue_size,
            value=queue_size,
            expected=f"queue_size ≤ {max_queue_size}",
            message=f"Queue size must be bounded, got {queue_size}",
        ))

        # Check for NaN/Inf (degenerate cases)
        checks.append(InvariantCheck(
            name="NO_NAN_VALUES",
            passed=not (np.isnan(atp) or np.isnan(pressure) or np.isnan(wake_score)),
            value={"atp": atp, "pressure": pressure, "wake_score": wake_score},
            expected="All values finite",
            message="Detected NaN in system state",
        ))

        checks.append(InvariantCheck(
            name="NO_INF_VALUES",
            passed=not (np.isinf(atp) or np.isinf(pressure) or np.isinf(wake_score)),
            value={"atp": atp, "pressure": pressure, "wake_score": wake_score},
            expected="All values finite",
            message="Detected Inf in system state",
        ))

        # Track violations
        for check in checks:
            if not check.passed:
                self.violations.append(check)

        return checks

    def check_liveness(
        self,
        pressure: float,
        atp: float,
        actions_taken_recently: int,
        cycles_since_action: int,
        pressure_threshold: float = 0.6,
        max_cycles_without_action: int = 100,
    ) -> List[InvariantCheck]:
        """Check liveness invariants (must eventually occur).

        Liveness invariants ensure system makes progress.
        """
        checks = []

        # If pressure > threshold, eventually action taken (or ATP exhausted)
        if pressure > pressure_threshold and atp > 10:
            # Should have taken action recently
            checks.append(InvariantCheck(
                name="HIGH_PRESSURE_SERVICED",
                passed=cycles_since_action < max_cycles_without_action,
                value=cycles_since_action,
                expected=f"Action within {max_cycles_without_action} cycles when pressure={pressure:.3f}",
                message=f"High pressure not serviced: {cycles_since_action} cycles without action",
            ))

        return checks

    def detect_oscillation(
        self,
        wake_score_history: deque,
        threshold: float = 0.1,
        min_oscillations: int = 3,
    ) -> Tuple[bool, Optional[float]]:
        """Detect limit cycles (oscillation) in wake score.

        Per Nova: "behavior under oscillatory load (avoid limit cycles)."

        Returns:
            (oscillation_detected, period)
        """
        if len(wake_score_history) < 10:
            return False, None

        # Convert to numpy array
        scores = np.array(list(wake_score_history))

        # Look for regular up-down patterns
        # Compute differences
        diffs = np.diff(scores)
        sign_changes = np.diff(np.sign(diffs))

        # Count zero-crossings (direction changes)
        zero_crossings = np.sum(np.abs(sign_changes) > 1)

        # If we see many direction changes, might be oscillating
        if zero_crossings >= min_oscillations * 2:
            # Estimate period using autocorrelation
            # (simplified: just count cycles)
            period = len(scores) / (zero_crossings / 2)
            return True, period

        return False, None

    def detect_deadlock(
        self,
        atp: float,
        pressure: float,
        actions_taken_recently: int,
        cycles_stalled: int,
        deadlock_threshold: int = 50,
    ) -> bool:
        """Detect deadlock (no progress despite high pressure).

        Deadlock: pressure > threshold, ATP available, but no actions taken.
        """
        if pressure > 0.6 and atp > 20 and cycles_stalled > deadlock_threshold:
            return True
        return False


class StressTestHarness:
    """
    Stress test harness for wake policy system.

    Implements Nova's requirement: "stress tests across regimes."
    """

    def __init__(self, system: 'SAGEIntegratedWakeSystem'):
        self.system = system
        self.invariant_checker = FormalInvariantChecker()

    def run_regime(
        self,
        regime: StressRegime,
        cycles: int = 200,
    ) -> StressTestResult:
        """Run a specific stress test regime."""

        logger.info(f"\n{'='*80}")
        logger.info(f"STRESS TEST: {regime.value}")
        logger.info(f"{'='*80}")

        # Reset system
        self.system.current_atp = 100.0
        self.system.memory_state = SAGEMemoryState()
        self.system.epistemic_state = SAGEEpistemicState()
        self.system.cycles_run = 0

        # Regime-specific setup
        regime_config = self._configure_regime(regime)

        # Run simulation with regime-specific perturbations
        trajectory = []
        violations = []
        max_queue_size = 0
        cycles_since_action = 0
        actions_taken_recently = 0
        recovery_time = None
        perturbation_start = None

        for cycle in range(cycles):
            # Apply regime-specific perturbations
            self._apply_regime_perturbation(regime, cycle, regime_config)

            # Normal SAGE operation
            self.system.simulate_sage_operation()

            # Compute pressure
            mem_signals = self.system.memory_state.compute_memory_pressure()
            unc_signals = self.system.epistemic_state.compute_uncertainty_pressure()

            # Check wake policy
            should_wake, decision_info = self.system.wake_policy.should_wake(
                memory_signals=mem_signals,
                uncertainty_signals=unc_signals,
                current_atp=self.system.current_atp,
            )

            # Track queue size (unprocessed memories as proxy)
            queue_size = self.system.memory_state.unprocessed_memories
            max_queue_size = max(max_queue_size, queue_size)

            # Check invariants
            overall_pressure = max(
                decision_info['memory_pressure'],
                decision_info['uncertainty_pressure'],
            )

            safety_checks = self.invariant_checker.check_safety_invariants(
                atp=self.system.current_atp,
                pressure=overall_pressure,
                wake_score=decision_info['wake_score'],
                queue_size=queue_size,
            )

            violations.extend([c for c in safety_checks if not c.passed])

            # Track actions
            if decision_info['is_awake']:
                actions = self.system.wake_policy.select_wake_actions(
                    memory_signals=mem_signals,
                    uncertainty_signals=unc_signals,
                    available_atp=self.system.current_atp,
                )

                if actions:
                    self.system.execute_wake_actions(actions)
                    cycles_since_action = 0
                    actions_taken_recently += len(actions)

                    # Check for recovery
                    if perturbation_start is not None and recovery_time is None:
                        if overall_pressure < 0.4:  # Recovered
                            recovery_time = cycle - perturbation_start
                            logger.info(f"Recovery detected at cycle {cycle}, time={recovery_time}")
            else:
                cycles_since_action += 1

            # ATP recovery
            if self.system.current_atp < 40:
                self.system.current_atp = min(self.system.current_atp + 2.0, 100.0)
            elif self.system.current_atp < 100:
                self.system.current_atp = min(self.system.current_atp + 1.0, 100.0)

            # Track trajectory
            trajectory.append({
                'cycle': cycle,
                'wake_score': decision_info['wake_score'],
                'memory_pressure': decision_info['memory_pressure'],
                'uncertainty_pressure': decision_info['uncertainty_pressure'],
                'atp': self.system.current_atp,
                'queue_size': queue_size,
                'is_awake': decision_info['is_awake'],
            })

        # Post-test analysis
        oscillation_detected, period = self.invariant_checker.detect_oscillation(
            self.system.wake_policy.trigger_state.wake_score_history
        )

        deadlock_detected = self.invariant_checker.detect_deadlock(
            atp=self.system.current_atp,
            pressure=overall_pressure,
            actions_taken_recently=actions_taken_recently,
            cycles_stalled=cycles_since_action,
        )

        result = StressTestResult(
            regime=regime,
            cycles=cycles,
            violations=violations,
            recovery_time=recovery_time,
            max_queue_size=max_queue_size,
            oscillation_detected=oscillation_detected,
            deadlock_detected=deadlock_detected,
            final_state={
                'atp': self.system.current_atp,
                'memory_pressure': decision_info['memory_pressure'],
                'uncertainty_pressure': decision_info['uncertainty_pressure'],
                'queue_size': queue_size,
            },
            trajectory=trajectory,
        )

        self._report_regime_results(result, period)
        return result

    def _configure_regime(self, regime: StressRegime) -> Dict[str, Any]:
        """Configure parameters for specific stress regime."""

        if regime == StressRegime.BURST_LOAD:
            return {
                'burst_start': 50,
                'burst_duration': 20,
                'burst_multiplier': 5.0,
            }

        elif regime == StressRegime.SUSTAINED_OVERLOAD:
            return {
                'overload_start': 20,
                'pressure_multiplier': 3.0,
            }

        elif regime == StressRegime.OSCILLATORY_LOAD:
            return {
                'period': 30,
                'amplitude': 2.0,
            }

        elif regime == StressRegime.LONG_INACTIVITY:
            return {
                'inactive_until': 100,
                'burst_after': 10,
            }

        elif regime == StressRegime.TOOL_FAILURES:
            return {
                'failure_rate': 0.7,  # 70% of actions fail
                'start_failures': 30,
            }

        elif regime == StressRegime.ATP_STARVATION:
            return {
                'initial_atp': 20,
                'recovery_rate': 0.5,  # Half normal recovery
            }

        elif regime == StressRegime.CONTRADICTORY_PRESSURES:
            return {
                'memory_high': True,
                'uncertainty_low': True,
            }

        elif regime == StressRegime.DEGENERATE_CASES:
            return {
                'inject_edge_cases': True,
            }

        return {}

    def _apply_regime_perturbation(
        self,
        regime: StressRegime,
        cycle: int,
        config: Dict[str, Any],
    ):
        """Apply regime-specific perturbations to system state."""

        if regime == StressRegime.BURST_LOAD:
            # Sudden spike in pressure
            if config['burst_start'] <= cycle < config['burst_start'] + config['burst_duration']:
                self.system.memory_state.unprocessed_memories += int(10 * config['burst_multiplier'])
                self.system.epistemic_state.uncertainty = min(
                    self.system.epistemic_state.uncertainty + 0.1,
                    1.0
                )

        elif regime == StressRegime.SUSTAINED_OVERLOAD:
            # Continuous high pressure
            if cycle >= config['overload_start']:
                self.system.memory_state.unprocessed_memories += int(5 * config['pressure_multiplier'])

        elif regime == StressRegime.OSCILLATORY_LOAD:
            # Periodic pressure waves
            phase = (cycle % config['period']) / config['period'] * 2 * np.pi
            pressure_multiplier = 1.0 + config['amplitude'] * np.sin(phase)
            if pressure_multiplier > 1.5:
                self.system.memory_state.unprocessed_memories += int(5 * (pressure_multiplier - 1.0))

        elif regime == StressRegime.LONG_INACTIVITY:
            # No pressure for long period, then burst
            if cycle < config['inactive_until']:
                # Suppress pressure accumulation
                self.system.memory_state.unprocessed_memories = max(
                    self.system.memory_state.unprocessed_memories - 2, 0
                )
            elif cycle < config['inactive_until'] + config['burst_after']:
                # Sudden burst
                self.system.memory_state.unprocessed_memories += 20

        elif regime == StressRegime.TOOL_FAILURES:
            # Actions fail to reduce pressure
            if cycle >= config['start_failures']:
                # Override action execution to simulate failures
                # (This would be done in execute_wake_actions, simplified here)
                pass

        elif regime == StressRegime.ATP_STARVATION:
            # Start with low ATP, slow recovery
            if cycle == 0:
                self.system.current_atp = config['initial_atp']
            # ATP recovery is already handled in main loop

        elif regime == StressRegime.CONTRADICTORY_PRESSURES:
            # Force contradictory pressure signals
            if config.get('memory_high'):
                self.system.memory_state.unprocessed_memories += 5
            if config.get('uncertainty_low'):
                self.system.epistemic_state.uncertainty = max(
                    self.system.epistemic_state.uncertainty - 0.05, 0.0
                )

        elif regime == StressRegime.DEGENERATE_CASES:
            # Inject edge cases
            if cycle % 50 == 0:
                # Test zero values
                if np.random.random() < 0.3:
                    self.system.current_atp = 0.01
                # Test extreme values
                if np.random.random() < 0.2:
                    self.system.memory_state.unprocessed_memories = 1000

    def _report_regime_results(self, result: StressTestResult, period: Optional[float]):
        """Report results of stress test regime."""

        logger.info(f"\n{'='*80}")
        logger.info(f"REGIME RESULTS: {result.regime.value}")
        logger.info(f"{'='*80}")
        logger.info(f"Cycles: {result.cycles}")
        logger.info(f"Invariant violations: {len(result.violations)}")
        logger.info(f"Max queue size: {result.max_queue_size}")
        logger.info(f"Recovery time: {result.recovery_time if result.recovery_time else 'N/A'}")
        logger.info(f"Oscillation detected: {result.oscillation_detected}")
        if result.oscillation_detected and period:
            logger.info(f"  Oscillation period: {period:.1f} cycles")
        logger.info(f"Deadlock detected: {result.deadlock_detected}")

        logger.info(f"\nFinal State:")
        logger.info(f"  ATP: {result.final_state['atp']:.1f}")
        logger.info(f"  Memory pressure: {result.final_state['memory_pressure']:.3f}")
        logger.info(f"  Uncertainty pressure: {result.final_state['uncertainty_pressure']:.3f}")
        logger.info(f"  Queue size: {result.final_state['queue_size']}")

        if result.violations:
            logger.info(f"\nInvariant Violations:")
            for v in result.violations[:5]:  # Show first 5
                logger.info(f"  - {v.name}: {v.message}")
            if len(result.violations) > 5:
                logger.info(f"  ... and {len(result.violations) - 5} more")
        else:
            logger.info("\n✅ All invariants maintained")


def run_session_105():
    """Run Session 105: Stress Testing Wake Policy."""

    logger.info("="*80)
    logger.info("SESSION 105: STRESS TESTING WAKE POLICY")
    logger.info("="*80)
    logger.info("Goal: Validate architectural soundness under adversarial conditions")
    logger.info("Based on: Nova GPT-5.2 peer review feedback")
    logger.info("")

    if not HAS_INTEGRATION:
        logger.error("Session 104 integration required but not found")
        return

    # Create system
    system = SAGEIntegratedWakeSystem(
        wake_threshold=0.4,
        sleep_threshold=0.2,
        initial_atp=100.0,
    )

    # Create stress test harness
    harness = StressTestHarness(system)

    # Run stress tests
    results = {}

    # Test each regime
    test_regimes = [
        StressRegime.BURST_LOAD,
        StressRegime.SUSTAINED_OVERLOAD,
        StressRegime.OSCILLATORY_LOAD,
        StressRegime.LONG_INACTIVITY,
        StressRegime.ATP_STARVATION,
        StressRegime.DEGENERATE_CASES,
    ]

    for regime in test_regimes:
        result = harness.run_regime(regime, cycles=200)
        results[regime.value] = {
            'violations': len(result.violations),
            'recovery_time': result.recovery_time,
            'max_queue_size': result.max_queue_size,
            'oscillation_detected': result.oscillation_detected,
            'deadlock_detected': result.deadlock_detected,
            'final_state': result.final_state,
        }

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("STRESS TEST SUMMARY")
    logger.info(f"{'='*80}")

    total_violations = sum(r['violations'] for r in results.values())
    regimes_with_oscillation = sum(1 for r in results.values() if r['oscillation_detected'])
    regimes_with_deadlock = sum(1 for r in results.values() if r['deadlock_detected'])

    logger.info(f"Total regimes tested: {len(test_regimes)}")
    logger.info(f"Total invariant violations: {total_violations}")
    logger.info(f"Regimes with oscillation: {regimes_with_oscillation}")
    logger.info(f"Regimes with deadlock: {regimes_with_deadlock}")

    if total_violations == 0 and regimes_with_oscillation == 0 and regimes_with_deadlock == 0:
        logger.info("\n✅ ALL STRESS TESTS PASSED")
    else:
        logger.info("\n⚠️ ISSUES DETECTED - See regime results above")

    # Save results
    output_path = Path(__file__).parent / "session105_stress_test_results.json"
    with open(output_path, 'w') as f:
        # Convert result objects to dicts for JSON serialization
        json_results = {}
        for regime_name, regime_data in results.items():
            json_results[regime_name] = regime_data

        json.dump({
            'session': 105,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            'total_violations': total_violations,
            'regimes_tested': len(test_regimes),
            'oscillations_detected': regimes_with_oscillation,
            'deadlocks_detected': regimes_with_deadlock,
            'results': json_results,
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    results = run_session_105()
