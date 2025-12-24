#!/usr/bin/env python3
"""
Session 106: Stress Test Validation - Verify Architectural Fixes

**Goal**: Re-run Session 105 stress tests with hardened system to validate fixes

**Context**:
Session 105 identified critical failures:
- Sustained overload: Queue → 1962 (85 violations)
- All regimes: Oscillation detected

Session 106 implemented fixes:
- Queue crisis mode (admission control + load shedding)
- Anti-oscillation controller (cooldown + EMA smoothing)

**This Script**: Re-run stress tests to validate fixes work

Expected Outcomes:
- Sustained overload: Queue stays < 1000 (no violations)
- All regimes: Reduced or eliminated oscillation
- All invariants maintained

Created: 2025-12-24 08:00 UTC (Autonomous Session 106)
Hardware: Thor (Jetson AGX Thor)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

# Import Session 105 stress testing framework
import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from session105_stress_testing_wake_policy import (
        StressRegime,
        FormalInvariantChecker,
        StressTestHarness,
        StressTestResult,
    )
    HAS_STRESS_TESTING = True
except ImportError:
    HAS_STRESS_TESTING = False

# Import Session 106 hardened system
try:
    from session106_architectural_hardening import HardenedWakeSystem
    HAS_HARDENED = True
except ImportError:
    HAS_HARDENED = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HardenedStressTestHarness(StressTestHarness):
    """
    Stress test harness adapted for hardened wake system.

    Uses Session 106 hardened system instead of Session 104 base system.
    """

    def __init__(self, system: HardenedWakeSystem):
        """Initialize with hardened system."""
        # Don't call parent __init__ as it expects different system type
        self.system = system
        self.invariant_checker = FormalInvariantChecker()

    def run_regime(
        self,
        regime: StressRegime,
        cycles: int = 200,
    ) -> StressTestResult:
        """Run stress test regime with hardened system."""

        logger.info(f"\n{'='*80}")
        logger.info(f"STRESS TEST (HARDENED): {regime.value}")
        logger.info(f"{'='*80}")

        # Reset system
        from session104_wake_sage_integration import SAGEMemoryState, SAGEEpistemicState
        self.system.current_atp = 100.0
        self.system.memory_state = SAGEMemoryState()
        self.system.epistemic_state = SAGEEpistemicState()
        self.system.cycles_run = 0

        # Reset crisis controllers
        from session106_architectural_hardening import QueueCrisisState, AntiOscillationState
        self.system.queue_crisis.state = QueueCrisisState(
            SOFT_LIMIT=500,
            HARD_LIMIT=1000,
            EMERGENCY_LIMIT=1500,
        )
        self.system.anti_oscillation.state = AntiOscillationState(
            MIN_WAKE_DURATION=10,
            MIN_SLEEP_DURATION=5,
            PRESSURE_ALPHA=0.3,
        )
        self.system.total_crisis_events = 0
        self.system.total_oscillations_prevented = 0

        # Regime-specific configuration
        regime_config = self._configure_regime(regime)

        # Run simulation
        trajectory = []
        violations = []
        max_queue_size = 0
        cycles_since_action = 0
        actions_taken_recently = 0
        recovery_time = None
        perturbation_start = None
        arrival_multiplier = 1.0

        for cycle in range(cycles):
            # Apply regime perturbations
            self._apply_regime_perturbation(regime, cycle, regime_config)

            # SAGE operation with crisis control
            self.system.simulate_sage_operation_with_crisis_control(arrival_multiplier)

            # Apply queue crisis response
            arrival_multiplier, crisis_info = self.system.apply_queue_crisis_response()

            # Compute pressure
            mem_signals = self.system.memory_state.compute_memory_pressure()
            unc_signals = self.system.epistemic_state.compute_uncertainty_pressure()

            # Smoothed pressure
            raw_pressure = max(
                mem_signals.overall_pressure(),
                unc_signals.overall_pressure(),
            )
            smoothed_pressure = self.system.anti_oscillation.smooth_pressure(raw_pressure)

            # Wake policy
            should_wake, decision_info = self.system.wake_policy.should_wake(
                memory_signals=mem_signals,
                uncertainty_signals=unc_signals,
                current_atp=self.system.current_atp,
            )

            # Anti-oscillation cooldown
            allowed, oscillation_prevented, cooldown_info = self.system.anti_oscillation.check_cooldown(
                proposed_wake_state=should_wake,
                current_cycle=cycle,
            )

            if not allowed:
                should_wake = self.system.anti_oscillation.state.current_state_is_awake

            # Track queue
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

            # Execute actions
            if should_wake:
                actions = self.system.wake_policy.select_wake_actions(
                    memory_signals=mem_signals,
                    uncertainty_signals=unc_signals,
                    available_atp=self.system.current_atp,
                )

                if actions:
                    self.system.execute_wake_actions(actions)
                    cycles_since_action = 0
                    actions_taken_recently += len(actions)

                    if perturbation_start is not None and recovery_time is None:
                        if overall_pressure < 0.4:
                            recovery_time = cycle - perturbation_start
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
                'is_awake': should_wake,
                'crisis_level': crisis_info['level'],
                'cooldown_active': cooldown_info.get('cooldown_active', False),
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
                'crisis_events': self.system.total_crisis_events,
                'oscillations_prevented': self.system.total_oscillations_prevented,
            },
            trajectory=trajectory,
        )

        self._report_regime_results(result, period)
        return result


def run_session_106_validation():
    """Run Session 106 stress test validation."""

    logger.info("="*80)
    logger.info("SESSION 106: STRESS TEST VALIDATION")
    logger.info("="*80)
    logger.info("Goal: Validate architectural fixes under stress")
    logger.info("Comparing Session 105 (unfixed) vs Session 106 (hardened)")
    logger.info("")

    if not HAS_HARDENED:
        logger.error("Session 106 hardened system required")
        return

    if not HAS_STRESS_TESTING:
        logger.error("Session 105 stress testing framework required")
        return

    # Create hardened system
    system = HardenedWakeSystem(
        wake_threshold=0.4,
        sleep_threshold=0.2,
        initial_atp=100.0,
        queue_soft_limit=500,
        queue_hard_limit=1000,
        queue_emergency_limit=1500,
        min_wake_duration=10,
        min_sleep_duration=5,
        pressure_alpha=0.3,
    )

    # Create harness
    harness = HardenedStressTestHarness(system)

    # Run stress tests (focus on previously failing regimes)
    results = {}

    test_regimes = [
        StressRegime.SUSTAINED_OVERLOAD,  # Previously: 85 violations
        StressRegime.BURST_LOAD,
        StressRegime.OSCILLATORY_LOAD,
    ]

    for regime in test_regimes:
        result = harness.run_regime(regime, cycles=200)
        results[regime.value] = {
            'violations': len(result.violations),
            'recovery_time': result.recovery_time,
            'max_queue_size': result.max_queue_size,
            'oscillation_detected': result.oscillation_detected,
            'deadlock_detected': result.deadlock_detected,
            'crisis_events': result.final_state['crisis_events'],
            'oscillations_prevented': result.final_state['oscillations_prevented'],
            'final_state': {
                'atp': result.final_state['atp'],
                'queue_size': result.final_state['queue_size'],
                'memory_pressure': result.final_state['memory_pressure'],
                'uncertainty_pressure': result.final_state['uncertainty_pressure'],
            },
        }

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*80}")

    total_violations = sum(r['violations'] for r in results.values())
    regimes_with_oscillation = sum(1 for r in results.values() if r['oscillation_detected'])

    logger.info(f"Total regimes tested: {len(test_regimes)}")
    logger.info(f"Total violations: {total_violations} (Session 105: 85)")
    logger.info(f"Regimes with oscillation: {regimes_with_oscillation} (Session 105: 6/6)")

    # Compare to Session 105
    logger.info(f"\nComparison to Session 105:")
    logger.info(f"  Queue violations: {total_violations} vs 85 (Session 105)")
    logger.info(f"  Oscillation rate: {regimes_with_oscillation}/{len(test_regimes)} vs 6/6 (Session 105)")

    if total_violations == 0:
        logger.info(f"\n✅ QUEUE GROWTH FIXED - No violations!")
    else:
        logger.info(f"\n⚠️ Queue violations still present: {total_violations}")

    if regimes_with_oscillation == 0:
        logger.info(f"✅ OSCILLATION FIXED - No limit cycling!")
    elif regimes_with_oscillation < len(test_regimes):
        logger.info(f"⚠️ Oscillation reduced but not eliminated: {regimes_with_oscillation}/{len(test_regimes)}")
    else:
        logger.info(f"❌ Oscillation not fixed: {regimes_with_oscillation}/{len(test_regimes)}")

    # Save results
    output_path = Path(__file__).parent / "session106_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            'session': 106,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            'total_violations': total_violations,
            'regimes_tested': len(test_regimes),
            'oscillations_detected': regimes_with_oscillation,
            'results': results,
            'comparison_to_s105': {
                'violations': {'s105': 85, 's106': total_violations},
                'oscillation_rate': {'s105': '6/6', 's106': f'{regimes_with_oscillation}/{len(test_regimes)}'},
            },
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    results = run_session_106_validation()
