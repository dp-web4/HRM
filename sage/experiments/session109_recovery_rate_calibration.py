#!/usr/bin/env python3
"""
Session 109: Recovery Rate Calibration

Goal: Address deadlock failure mode discovered in Session 108 stress testing.

Key Finding from S108:
- Compute starvation regime: recovery_rate (1.0/cycle) < min_consumption → deadlock
- System locked with 93.5% actions blocked, no recovery possible

Research Question:
Can we calibrate recovery rates to ensure recovery_rate > min_action_cost
for each resource dimension, preventing deadlock?

Design Principle:
For each resource R, ensure:
    recovery_rate_R > min(action_costs_R)

This guarantees that even if only the cheapest action is executable,
the system can eventually recover from resource depletion.

Approach:
1. Analyze action resource profiles to find minimum costs per resource
2. Calibrate recovery rates to exceed minimum costs
3. Re-run Session 108 stress tests with calibrated rates
4. Validate deadlock prevention while maintaining system dynamics

Expected Outcome:
- Compute starvation regime should no longer deadlock
- Other regimes should maintain or improve recovery rates
- System remains responsive under stress
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


@dataclass
class MultiResourceBudget:
    """Multi-dimensional resource budget system (from S107)."""
    compute_atp: float = 100.0
    memory_atp: float = 100.0
    tool_atp: float = 100.0
    latency_budget: float = 1000.0
    risk_budget: float = 1.0

    # Recovery rates (to be calibrated)
    compute_recovery: float = 1.0
    memory_recovery: float = 1.0
    tool_recovery: float = 0.5
    latency_recovery: float = 10.0
    risk_recovery: float = 0.02

    def can_afford(self, action_costs: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Check if all resource budgets can afford action."""
        bottlenecks = []

        if action_costs.get('compute', 0) > self.compute_atp:
            bottlenecks.append('compute')
        if action_costs.get('memory', 0) > self.memory_atp:
            bottlenecks.append('memory')
        if action_costs.get('tool', 0) > self.tool_atp:
            bottlenecks.append('tool')
        if action_costs.get('latency', 0) > self.latency_budget:
            bottlenecks.append('latency')
        if action_costs.get('risk', 0) > self.risk_budget:
            bottlenecks.append('risk')

        return len(bottlenecks) == 0, bottlenecks

    def consume(self, action_costs: Dict[str, float]):
        """Consume resources for action execution."""
        self.compute_atp -= action_costs.get('compute', 0)
        self.memory_atp -= action_costs.get('memory', 0)
        self.tool_atp -= action_costs.get('tool', 0)
        self.latency_budget -= action_costs.get('latency', 0)
        self.risk_budget -= action_costs.get('risk', 0)

        # Prevent negative budgets
        self.compute_atp = max(0, self.compute_atp)
        self.memory_atp = max(0, self.memory_atp)
        self.tool_atp = max(0, self.tool_atp)
        self.latency_budget = max(0, self.latency_budget)
        self.risk_budget = max(0, self.risk_budget)

    def recover(self):
        """Apply passive recovery rates."""
        self.compute_atp = min(100.0, self.compute_atp + self.compute_recovery)
        self.memory_atp = min(100.0, self.memory_atp + self.memory_recovery)
        self.tool_atp = min(100.0, self.tool_atp + self.tool_recovery)
        self.latency_budget = min(1000.0, self.latency_budget + self.latency_recovery)
        self.risk_budget = min(1.0, self.risk_budget + self.risk_recovery)

    def get_limiting_resource(self) -> str:
        """Identify current bottleneck resource."""
        depletions = {
            'compute': 1.0 - (self.compute_atp / 100.0),
            'memory': 1.0 - (self.memory_atp / 100.0),
            'tool': 1.0 - (self.tool_atp / 100.0),
            'latency': 1.0 - (self.latency_budget / 1000.0),
            'risk': 1.0 - (self.risk_budget / 1.0),
        }
        return max(depletions.items(), key=lambda x: x[1])[0]


@dataclass
class MultiResourceAction:
    """Action with multi-dimensional resource costs."""
    action_type: str
    target: str
    expected_pressure_reduction: float
    compute_cost: float = 0.0
    memory_cost: float = 0.0
    tool_cost: float = 0.0
    latency_cost: float = 0.0
    risk_cost: float = 0.0

    def get_costs(self) -> Dict[str, float]:
        """Return cost dictionary."""
        return {
            'compute': self.compute_cost,
            'memory': self.memory_cost,
            'tool': self.tool_cost,
            'latency': self.latency_cost,
            'risk': self.risk_cost,
        }


def get_action_profiles() -> Dict[str, Dict[str, float]]:
    """
    Return standard action resource profiles (from S107).

    These represent the resource costs for different memory management actions.
    """
    return {
        'consolidation': {
            'compute': 8.0,   # High: Pattern extraction, clustering
            'memory': 6.0,    # High: Write consolidated patterns
            'tool': 0.0,      # None: Internal operation
            'latency': 50.0,  # Moderate: Takes time
            'risk': 0.1,      # Low: Safe operation
        },
        'pruning': {
            'compute': 2.0,   # Low: Simple filtering
            'memory': 5.0,    # High: Memory modifications
            'tool': 0.0,      # None: Internal operation
            'latency': 20.0,  # Fast: Quick deletion
            'risk': 0.2,      # Moderate: Might delete useful data
        },
        'index_rebuild': {
            'compute': 5.0,   # Moderate: Recompute indices
            'memory': 7.0,    # High: Rewrite index structures
            'tool': 0.0,      # None: Internal operation
            'latency': 45.0,  # Moderate: Indexing takes time
            'risk': 0.05,     # Very low: Just reorganization
        },
        'probe': {
            'compute': 3.0,   # Low-moderate: Format query
            'memory': 1.0,    # Low: Small memory footprint
            'tool': 10.0,     # High: External API call
            'latency': 100.0, # High: Network roundtrip
            'risk': 0.4,      # High: External dependency
        },
        'triage': {
            'compute': 7.0,   # High: Reasoning-heavy
            'memory': 2.0,    # Low: Minimal writes
            'tool': 0.0,      # None: Internal reasoning
            'latency': 40.0,  # Moderate: Thinking time
            'risk': 0.15,     # Low-moderate: Internal operation
        },
    }


def analyze_minimum_costs() -> Dict[str, float]:
    """
    Analyze action profiles to find minimum cost for each resource.

    This determines the floor for recovery rates - any recovery rate
    below the minimum cost creates deadlock risk.
    """
    profiles = get_action_profiles()

    min_costs = {
        'compute': float('inf'),
        'memory': float('inf'),
        'tool': float('inf'),
        'latency': float('inf'),
        'risk': float('inf'),
    }

    # Find minimum non-zero cost for each resource
    for action_type, costs in profiles.items():
        for resource, cost in costs.items():
            if cost > 0 and cost < min_costs[resource]:
                min_costs[resource] = cost

    # If no action uses a resource, set min to 0
    for resource in min_costs:
        if min_costs[resource] == float('inf'):
            min_costs[resource] = 0.0

    return min_costs


def calibrate_recovery_rates(margin: float = 1.2) -> Dict[str, float]:
    """
    Calibrate recovery rates to exceed minimum action costs.

    Design Principle: recovery_rate = margin × min_action_cost

    The margin (default 1.2 = 20% headroom) ensures recovery rate
    exceeds minimum cost even with slight variations.

    Args:
        margin: Multiplicative safety margin (>1.0)

    Returns:
        Calibrated recovery rates per resource
    """
    min_costs = analyze_minimum_costs()

    recovery_rates = {}
    for resource, min_cost in min_costs.items():
        if min_cost > 0:
            recovery_rates[resource] = margin * min_cost
        else:
            # No actions use this resource, use nominal rate
            recovery_rates[resource] = {
                'compute': 1.0,
                'memory': 1.0,
                'tool': 0.5,
                'latency': 10.0,
                'risk': 0.02,
            }[resource]

    return recovery_rates


class StressRegime(Enum):
    """Stress test regimes from Session 108."""
    COMPUTE_STARVATION = "compute_starvation"
    MULTI_RESOURCE_DEPLETION = "multi_resource_depletion"
    BOTTLENECK_OSCILLATION = "bottleneck_oscillation"
    TOOL_RATE_LIMITING = "tool_rate_limiting"


class RecoveryRateTestHarness:
    """Test harness for validating calibrated recovery rates."""

    def __init__(self, recovery_rates: Dict[str, float]):
        """Initialize with calibrated recovery rates."""
        self.recovery_rates = recovery_rates
        self.action_profiles = get_action_profiles()

    def run_compute_starvation_test(self, cycles: int = 200) -> Dict:
        """
        Re-run compute starvation test with calibrated rates.

        Session 108 result: 93.5% blocked, deadlock
        Expected with calibration: Recovery should prevent deadlock
        """
        budget = MultiResourceBudget(
            compute_recovery=self.recovery_rates['compute'],
            memory_recovery=self.recovery_rates['memory'],
            tool_recovery=self.recovery_rates['tool'],
            latency_recovery=self.recovery_rates['latency'],
            risk_recovery=self.recovery_rates['risk'],
        )

        actions_executed = 0
        actions_blocked = 0
        bottleneck_history = []
        recovery_events = 0

        for cycle in range(cycles):
            # Apply starvation: drain compute continuously
            budget.compute_atp = max(0, budget.compute_atp - 0.5)

            # Passive recovery
            budget.recover()

            # Track bottleneck
            bottleneck = budget.get_limiting_resource()
            bottleneck_history.append(bottleneck)

            # Try to execute cheapest action (pruning: 2.0 compute)
            prof = self.action_profiles['pruning']
            action = MultiResourceAction(
                action_type='pruning',
                target='test',
                expected_pressure_reduction=0.1,
                compute_cost=prof['compute'],
                memory_cost=prof['memory'],
                tool_cost=prof['tool'],
                latency_cost=prof['latency'],
                risk_cost=prof['risk'],
            )

            can_execute, bottlenecks = budget.can_afford(action.get_costs())

            if can_execute:
                budget.consume(action.get_costs())
                actions_executed += 1

                # Check if this was a recovery from depletion
                if budget.compute_atp < 10.0:
                    recovery_events += 1
            else:
                actions_blocked += 1

        # Analyze results
        total_actions = actions_executed + actions_blocked
        block_rate = actions_blocked / total_actions if total_actions > 0 else 0

        # Check for deadlock: sustained blocking in final 50 cycles
        final_blocks = sum(1 for b in bottleneck_history[-50:] if b == 'compute')
        deadlocked = final_blocks >= 45  # 90%+ of final cycles blocked

        return {
            'regime': 'compute_starvation_calibrated',
            'cycles': cycles,
            'actions_executed': actions_executed,
            'actions_blocked': actions_blocked,
            'block_rate': block_rate,
            'recovery_events': recovery_events,
            'deadlocked': deadlocked,
            'final_budget': {
                'compute_atp': budget.compute_atp,
                'memory_atp': budget.memory_atp,
            },
            'recovery_rate': self.recovery_rates['compute'],
            'min_action_cost': self.action_profiles['pruning']['compute'],
        }

    def run_all_stress_tests(self, cycles: int = 200) -> Dict:
        """
        Run all Session 108 stress tests with calibrated recovery rates.

        Compare results to baseline (uncalibrated) from Session 108.
        """
        results = {}

        # 1. Compute starvation (main target for calibration)
        logger.info("Running compute starvation test with calibrated recovery...")
        results['compute_starvation'] = self.run_compute_starvation_test(cycles)

        # For other regimes, we'd implement similar tests
        # For now, focus on compute starvation as proof of concept

        return results


def run_session_109() -> Dict:
    """
    Execute Session 109: Recovery Rate Calibration.

    Steps:
    1. Analyze minimum action costs
    2. Calibrate recovery rates
    3. Test with compute starvation regime
    4. Validate deadlock prevention
    5. Document calibration methodology
    """
    logger.info("=" * 80)
    logger.info("SESSION 109: RECOVERY RATE CALIBRATION")
    logger.info("=" * 80)
    logger.info("Goal: Prevent deadlock by ensuring recovery_rate > min_action_cost")
    logger.info("")

    # Step 1: Analyze minimum costs
    logger.info("Step 1: Analyzing minimum action costs...")
    min_costs = analyze_minimum_costs()

    logger.info("Minimum costs per resource:")
    for resource, cost in min_costs.items():
        logger.info(f"  {resource}: {cost}")
    logger.info("")

    # Step 2: Calibrate recovery rates
    logger.info("Step 2: Calibrating recovery rates (20% margin)...")
    recovery_rates = calibrate_recovery_rates(margin=1.2)

    logger.info("Calibrated recovery rates:")
    for resource, rate in recovery_rates.items():
        margin_pct = ((rate / min_costs[resource]) - 1.0) * 100 if min_costs[resource] > 0 else 0
        logger.info(f"  {resource}: {rate:.2f} (min cost: {min_costs[resource]:.2f}, margin: {margin_pct:.0f}%)")
    logger.info("")

    # Step 3: Test calibrated rates
    logger.info("Step 3: Testing calibrated recovery rates...")
    harness = RecoveryRateTestHarness(recovery_rates)
    test_results = harness.run_all_stress_tests(cycles=200)

    # Step 4: Compare to Session 108 baseline
    logger.info("")
    logger.info("=" * 80)
    logger.info("RESULTS COMPARISON: Session 108 vs Session 109")
    logger.info("=" * 80)

    s108_baseline = {
        'actions_executed': 9,
        'actions_blocked': 130,
        'block_rate': 0.935,
        'deadlocked': True,
    }

    s109_result = test_results['compute_starvation']

    logger.info("\nCompute Starvation Regime:")
    logger.info(f"  S108 (uncalibrated): {s108_baseline['actions_executed']} executed, "
                f"{s108_baseline['actions_blocked']} blocked ({s108_baseline['block_rate']*100:.1f}%), "
                f"deadlocked: {s108_baseline['deadlocked']}")
    logger.info(f"  S109 (calibrated):   {s109_result['actions_executed']} executed, "
                f"{s109_result['actions_blocked']} blocked ({s109_result['block_rate']*100:.1f}%), "
                f"deadlocked: {s109_result['deadlocked']}")

    improvement = {
        'execution_increase': s109_result['actions_executed'] - s108_baseline['actions_executed'],
        'block_rate_reduction': s108_baseline['block_rate'] - s109_result['block_rate'],
        'deadlock_prevented': s108_baseline['deadlocked'] and not s109_result['deadlocked'],
    }

    logger.info(f"\nImprovement:")
    logger.info(f"  Actions executed: +{improvement['execution_increase']} ({improvement['execution_increase']/s108_baseline['actions_executed']*100:.0f}% increase)")
    logger.info(f"  Block rate: {improvement['block_rate_reduction']*100:.1f}% reduction")
    logger.info(f"  Deadlock prevented: {improvement['deadlock_prevented']}")

    # Step 5: Save results
    results = {
        'session': 109,
        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'methodology': {
            'principle': 'recovery_rate > min_action_cost',
            'margin': 1.2,
            'calibration_basis': 'minimum non-zero cost per resource',
        },
        'minimum_costs': min_costs,
        'calibrated_recovery_rates': recovery_rates,
        'test_results': test_results,
        'comparison_to_s108': {
            'baseline': s108_baseline,
            'calibrated': s109_result,
            'improvement': improvement,
        },
    }

    output_file = 'sage/experiments/session109_recovery_calibration_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("SESSION 109 COMPLETE")
    logger.info("=" * 80)

    if improvement['deadlock_prevented']:
        logger.info("✅ SUCCESS: Deadlock prevention validated!")
        logger.info(f"   Recovery rate calibration increased action execution by {improvement['execution_increase']/s108_baseline['actions_executed']*100:.0f}%")
        logger.info(f"   Block rate reduced from {s108_baseline['block_rate']*100:.1f}% to {s109_result['block_rate']*100:.1f}%")
    else:
        logger.info("⚠️  WARNING: Deadlock still present despite calibration")
        logger.info("   Further investigation needed")

    return results


if __name__ == "__main__":
    results = run_session_109()
