#!/usr/bin/env python3
"""
Session 108: Stress Testing Multi-Resource System

**Goal**: Validate multi-resource budget system under adversarial conditions

**Context**:
Session 107 discovered emergent prioritization in nominal conditions:
- Compute bottleneck: 79% of time
- System favored low-compute actions (pruning: 70%)
- Emergent optimization without explicit programming

**Research Questions**:
1. Does bottleneck shift under sustained load?
2. What happens when multiple resources depleted simultaneously?
3. Can system recover from multi-resource exhaustion?
4. Do new failure modes emerge with multi-dimensional constraints?

**Stress Regimes to Test**:
1. **Sustained Compute Starvation**: Force compute to stay depleted
2. **Multi-Resource Depletion**: Drain all resources simultaneously
3. **Bottleneck Oscillation**: Alternate which resource is limiting
4. **Tool Rate Limiting**: Saturate tool budget to force internal ops
5. **Risk Accumulation**: Force high-risk operations until budget exhausted

**Expected Insights**:
- Bottleneck shifts reveal system adaptability
- Multi-resource exhaustion creates new failure modes
- Recovery patterns differ per resource
- Action selection adapts to changing constraints

**Hypothesis**:
Multi-resource system will exhibit **different bottleneck dynamics** under stress,
revealing Pareto trade-offs invisible in nominal conditions.

Created: 2025-12-24 14:00 UTC (Autonomous Session 108)
Hardware: Thor (Jetson AGX Thor)
Previous: Session 107 (multi-resource budgets)
Goal: Stress test multi-dimensional constraint system
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from enum import Enum

# Import Session 107 multi-resource system
import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from session107_multi_resource_budgets import (
        MultiResourceBudget,
        MultiResourceAction,
        MultiResourceWakeSystem,
    )
    HAS_MULTI_RESOURCE = True
except ImportError:
    HAS_MULTI_RESOURCE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiResourceStressRegime(Enum):
    """Stress test regimes for multi-resource system."""
    COMPUTE_STARVATION = "compute_starvation"
    MULTI_RESOURCE_DEPLETION = "multi_resource_depletion"
    BOTTLENECK_OSCILLATION = "bottleneck_oscillation"
    TOOL_RATE_LIMITING = "tool_rate_limiting"
    RISK_ACCUMULATION = "risk_accumulation"


@dataclass
class StressTestResult:
    """Result of multi-resource stress test."""
    regime: MultiResourceStressRegime
    cycles: int

    # Bottleneck dynamics
    bottleneck_distribution: Dict[str, int]
    bottleneck_transitions: int  # How many times bottleneck changed

    # Resource exhaustion events
    compute_exhausted_cycles: int
    memory_exhausted_cycles: int
    tool_exhausted_cycles: int
    latency_exhausted_cycles: int
    risk_exhausted_cycles: int

    # Action execution
    actions_executed: int
    actions_blocked: int  # Couldn't afford due to resource constraints
    action_distribution: Dict[str, int]

    # Final state
    final_budgets: Dict[str, float]
    recovery_achieved: bool

    # Trajectory
    trajectory: List[Dict[str, Any]]


class MultiResourceStressTestHarness:
    """
    Stress test harness for multi-resource budget system.

    Tests system behavior under adversarial resource conditions.
    """

    def __init__(self):
        if not HAS_MULTI_RESOURCE:
            raise ImportError("Session 107 multi-resource system required")

    def run_regime(
        self,
        regime: MultiResourceStressRegime,
        cycles: int = 200,
    ) -> StressTestResult:
        """Run a specific stress test regime."""

        logger.info(f"\n{'='*80}")
        logger.info(f"MULTI-RESOURCE STRESS TEST: {regime.value}")
        logger.info(f"{'='*80}")

        # Create fresh system
        system = MultiResourceWakeSystem(
            wake_threshold=0.4,
            sleep_threshold=0.2,
            initial_compute=100.0,
            initial_memory=100.0,
            initial_tool=100.0,
            initial_latency=1000.0,
            initial_risk=1.0,
        )

        # Regime-specific configuration
        regime_config = self._configure_regime(regime)

        # Tracking
        trajectory = []
        bottleneck_history = []
        previous_bottleneck = None
        bottleneck_transitions = 0

        resource_exhausted_cycles = {
            'compute': 0,
            'memory': 0,
            'tool': 0,
            'latency': 0,
            'risk': 0,
        }

        actions_executed = 0
        actions_blocked = 0
        action_distribution = defaultdict(int)

        for cycle in range(cycles):
            system.cycles_run = cycle + 1

            # Apply regime-specific stress
            self._apply_stress(regime, cycle, system, regime_config)

            # Simulate pressure accumulation
            memory_pressure = min(0.1 + cycle * 0.005, 1.0)
            uncertainty_pressure = min(0.1 + cycle * 0.003, 1.0)

            # Check wake policy
            overall_pressure = max(memory_pressure, uncertainty_pressure)
            should_wake = overall_pressure > system.wake_threshold

            # Identify limiting resource
            limiting_resource, depletion_ratio = system.budget.get_limiting_resource()

            # Track bottleneck
            bottleneck_history.append(limiting_resource)
            if previous_bottleneck is not None and limiting_resource != previous_bottleneck:
                bottleneck_transitions += 1
                logger.info(f"Cycle {cycle}: Bottleneck shift {previous_bottleneck} → {limiting_resource}")

            previous_bottleneck = limiting_resource

            # Track resource exhaustion (< 10% remaining)
            if system.budget.compute_atp < 10.0:
                resource_exhausted_cycles['compute'] += 1
            if system.budget.memory_atp < 10.0:
                resource_exhausted_cycles['memory'] += 1
            if system.budget.tool_atp < 10.0:
                resource_exhausted_cycles['tool'] += 1
            if system.budget.latency_budget < 100.0:
                resource_exhausted_cycles['latency'] += 1
            if system.budget.risk_budget < 0.1:
                resource_exhausted_cycles['risk'] += 1

            # Execute actions if awake
            if should_wake:
                if not system.is_awake:
                    system.is_awake = True
                    system.total_wakes += 1

                # Select actions
                actions = system.select_actions_multi_resource(
                    memory_pressure=memory_pressure,
                    uncertainty_pressure=uncertainty_pressure,
                )

                if actions:
                    # Execute first action
                    system.execute_action(actions[0])
                    actions_executed += 1
                    action_distribution[actions[0].action_type] += 1
                else:
                    # No affordable actions
                    actions_blocked += 1
                    logger.debug(f"Cycle {cycle}: All actions blocked by resource constraints")

            else:
                if system.is_awake:
                    system.is_awake = False

            # Resource recovery (adjusted by regime)
            recovery_state = 'WAKE' if system.is_awake else 'REST'

            # Apply regime-specific recovery modulation
            if regime == MultiResourceStressRegime.COMPUTE_STARVATION:
                # Suppress compute recovery
                original_recovery = system.budget.recover(recovery_state)
                system.budget.compute_atp = max(system.budget.compute_atp - 1.0, 0.0)

            elif regime == MultiResourceStressRegime.MULTI_RESOURCE_DEPLETION:
                # Suppress all recovery
                original_recovery = system.budget.recover('CRISIS')

            else:
                # Normal recovery
                recovery = system.budget.recover(recovery_state)

            # Track trajectory
            trajectory.append({
                'cycle': cycle,
                'memory_pressure': memory_pressure,
                'uncertainty_pressure': uncertainty_pressure,
                'is_awake': system.is_awake,
                'limiting_resource': limiting_resource,
                'budgets': system.budget.to_dict(),
            })

            # Periodic logging
            if cycle % 50 == 0:
                logger.info(
                    f"Cycle {cycle}: "
                    f"bottleneck={limiting_resource}, "
                    f"compute={system.budget.compute_atp:.1f}, "
                    f"memory={system.budget.memory_atp:.1f}, "
                    f"tool={system.budget.tool_atp:.1f}, "
                    f"actions_exec={actions_executed}, "
                    f"actions_blocked={actions_blocked}"
                )

        # Analyze bottleneck distribution
        bottleneck_counts = defaultdict(int)
        for resource in bottleneck_history:
            bottleneck_counts[resource] += 1

        # Check recovery
        final_budgets = system.budget.to_dict()
        recovery_achieved = all(v > 20.0 for v in final_budgets.values() if v != final_budgets.get('risk_budget', 0))

        # Create result
        result = StressTestResult(
            regime=regime,
            cycles=cycles,
            bottleneck_distribution=dict(bottleneck_counts),
            bottleneck_transitions=bottleneck_transitions,
            compute_exhausted_cycles=resource_exhausted_cycles['compute'],
            memory_exhausted_cycles=resource_exhausted_cycles['memory'],
            tool_exhausted_cycles=resource_exhausted_cycles['tool'],
            latency_exhausted_cycles=resource_exhausted_cycles['latency'],
            risk_exhausted_cycles=resource_exhausted_cycles['risk'],
            actions_executed=actions_executed,
            actions_blocked=actions_blocked,
            action_distribution=dict(action_distribution),
            final_budgets=final_budgets,
            recovery_achieved=recovery_achieved,
            trajectory=trajectory,
        )

        self._report_results(result)
        return result

    def _configure_regime(self, regime: MultiResourceStressRegime) -> Dict[str, Any]:
        """Configure regime-specific parameters."""

        if regime == MultiResourceStressRegime.COMPUTE_STARVATION:
            return {
                'compute_drain_rate': 2.0,  # Drain compute faster than it recovers
            }

        elif regime == MultiResourceStressRegime.MULTI_RESOURCE_DEPLETION:
            return {
                'depletion_start': 20,
                'drain_all': True,
            }

        elif regime == MultiResourceStressRegime.BOTTLENECK_OSCILLATION:
            return {
                'oscillation_period': 40,  # Switch bottleneck every 40 cycles
            }

        elif regime == MultiResourceStressRegime.TOOL_RATE_LIMITING:
            return {
                'tool_drain_rate': 5.0,  # Saturate tool budget
            }

        elif regime == MultiResourceStressRegime.RISK_ACCUMULATION:
            return {
                'force_risky_ops': True,
                'risk_drain_rate': 0.05,
            }

        return {}

    def _apply_stress(
        self,
        regime: MultiResourceStressRegime,
        cycle: int,
        system: MultiResourceWakeSystem,
        config: Dict[str, Any],
    ):
        """Apply regime-specific stress to system."""

        if regime == MultiResourceStressRegime.COMPUTE_STARVATION:
            # Continuously drain compute
            drain = config['compute_drain_rate']
            system.budget.compute_atp = max(system.budget.compute_atp - drain, 0.0)

        elif regime == MultiResourceStressRegime.MULTI_RESOURCE_DEPLETION:
            # Drain all resources after cycle 20
            if cycle >= config['depletion_start']:
                system.budget.compute_atp = max(system.budget.compute_atp - 3.0, 0.0)
                system.budget.memory_atp = max(system.budget.memory_atp - 3.0, 0.0)
                system.budget.tool_atp = max(system.budget.tool_atp - 1.0, 0.0)
                system.budget.latency_budget = max(system.budget.latency_budget - 50.0, 0.0)
                system.budget.risk_budget = max(system.budget.risk_budget - 0.05, 0.0)

        elif regime == MultiResourceStressRegime.BOTTLENECK_OSCILLATION:
            # Periodically drain different resources
            period = config['oscillation_period']
            phase = (cycle // period) % 5

            if phase == 0:
                # Drain compute
                system.budget.compute_atp = max(system.budget.compute_atp - 2.0, 5.0)
            elif phase == 1:
                # Drain memory
                system.budget.memory_atp = max(system.budget.memory_atp - 2.0, 5.0)
            elif phase == 2:
                # Drain tool
                system.budget.tool_atp = max(system.budget.tool_atp - 1.0, 5.0)
            elif phase == 3:
                # Drain latency
                system.budget.latency_budget = max(system.budget.latency_budget - 100.0, 100.0)
            elif phase == 4:
                # Drain risk
                system.budget.risk_budget = max(system.budget.risk_budget - 0.1, 0.1)

        elif regime == MultiResourceStressRegime.TOOL_RATE_LIMITING:
            # Saturate tool budget
            drain = config['tool_drain_rate']
            system.budget.tool_atp = max(system.budget.tool_atp - drain, 0.0)

        elif regime == MultiResourceStressRegime.RISK_ACCUMULATION:
            # Drain risk budget (force low-risk operations)
            drain = config['risk_drain_rate']
            system.budget.risk_budget = max(system.budget.risk_budget - drain, 0.0)

    def _report_results(self, result: StressTestResult):
        """Report stress test results."""

        logger.info(f"\n{'='*80}")
        logger.info(f"STRESS TEST RESULTS: {result.regime.value}")
        logger.info(f"{'='*80}")
        logger.info(f"Cycles: {result.cycles}")
        logger.info(f"Actions executed: {result.actions_executed}")
        logger.info(f"Actions blocked: {result.actions_blocked}")

        logger.info(f"\nBottleneck Distribution:")
        for resource, count in sorted(result.bottleneck_distribution.items(), key=lambda x: x[1], reverse=True):
            pct = (count / result.cycles) * 100
            logger.info(f"  {resource}: {count} cycles ({pct:.1f}%)")

        logger.info(f"\nBottleneck Transitions: {result.bottleneck_transitions}")

        logger.info(f"\nResource Exhaustion (<10% remaining):")
        logger.info(f"  Compute: {result.compute_exhausted_cycles} cycles")
        logger.info(f"  Memory: {result.memory_exhausted_cycles} cycles")
        logger.info(f"  Tool: {result.tool_exhausted_cycles} cycles")
        logger.info(f"  Latency: {result.latency_exhausted_cycles} cycles")
        logger.info(f"  Risk: {result.risk_exhausted_cycles} cycles")

        logger.info(f"\nAction Distribution:")
        for action, count in sorted(result.action_distribution.items(), key=lambda x: x[1], reverse=True):
            pct = (count / max(result.actions_executed, 1)) * 100
            logger.info(f"  {action}: {count} ({pct:.1f}%)")

        logger.info(f"\nFinal Budgets:")
        for resource, value in result.final_budgets.items():
            logger.info(f"  {resource}: {value:.1f}")

        logger.info(f"\nRecovery: {'✅ ACHIEVED' if result.recovery_achieved else '❌ FAILED'}")


def run_session_108():
    """Run Session 108: Multi-Resource Stress Testing."""

    logger.info("="*80)
    logger.info("SESSION 108: MULTI-RESOURCE STRESS TESTING")
    logger.info("="*80)
    logger.info("Goal: Validate multi-resource system under adversarial conditions")
    logger.info("")

    if not HAS_MULTI_RESOURCE:
        logger.error("Session 107 multi-resource system required")
        return

    # Create stress test harness
    harness = MultiResourceStressTestHarness()

    # Run stress tests
    results = {}

    test_regimes = [
        MultiResourceStressRegime.COMPUTE_STARVATION,
        MultiResourceStressRegime.MULTI_RESOURCE_DEPLETION,
        MultiResourceStressRegime.BOTTLENECK_OSCILLATION,
        MultiResourceStressRegime.TOOL_RATE_LIMITING,
    ]

    for regime in test_regimes:
        result = harness.run_regime(regime, cycles=200)

        results[regime.value] = {
            'bottleneck_distribution': result.bottleneck_distribution,
            'bottleneck_transitions': result.bottleneck_transitions,
            'actions_executed': result.actions_executed,
            'actions_blocked': result.actions_blocked,
            'action_distribution': result.action_distribution,
            'resource_exhaustion': {
                'compute': result.compute_exhausted_cycles,
                'memory': result.memory_exhausted_cycles,
                'tool': result.tool_exhausted_cycles,
                'latency': result.latency_exhausted_cycles,
                'risk': result.risk_exhausted_cycles,
            },
            'final_budgets': result.final_budgets,
            'recovery_achieved': result.recovery_achieved,
        }

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("STRESS TEST SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Regimes tested: {len(test_regimes)}")

    total_transitions = sum(r['bottleneck_transitions'] for r in results.values())
    logger.info(f"Total bottleneck transitions: {total_transitions}")

    regimes_recovered = sum(1 for r in results.values() if r['recovery_achieved'])
    logger.info(f"Regimes with recovery: {regimes_recovered}/{len(test_regimes)}")

    # Save results
    output_path = Path(__file__).parent / "session108_multi_resource_stress_results.json"
    with open(output_path, 'w') as f:
        json.dump({
            'session': 108,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            'regimes_tested': len(test_regimes),
            'total_bottleneck_transitions': total_transitions,
            'regimes_recovered': regimes_recovered,
            'results': results,
        }, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    results = run_session_108()
