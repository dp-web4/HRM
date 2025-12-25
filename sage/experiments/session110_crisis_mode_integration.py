#!/usr/bin/env python3
"""
Session 110: Crisis Mode Integration

Goal: Formalize "passive recovery through inactivity" (Session 108 discovery)
as operational "sleep mode" triggered by multi-resource exhaustion.

Key Findings from Session 108:
- Multi-resource depletion regime: Complete exhaustion → recovery via "sleep"
- Mechanism: When no actions consume resources, natural recovery rates restore budgets
- Biological parallel: Sleep allows resource restoration

Session 109 Enhancement:
- Calibrated recovery rates ensure recovery_rate > min_action_cost
- Deadlock prevented: system can recover even under stress

Session 110 Goal:
Implement crisis mode controller that:
1. Detects multi-resource exhaustion (multiple resources <10%)
2. Triggers "sleep mode" (halt all non-emergency actions)
3. Maximizes recovery (reduce consumption to near-zero)
4. Resumes normal operation when resources restored (>50%)

Design Principle:
Sleep mode is not a failure state - it's a valid operational mode
for resource restoration when normal operation unsustainable.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


class OperationalMode(Enum):
    """Operational modes for multi-resource consciousness system."""
    NORMAL = "normal"           # Normal operation with full action selection
    STRESSED = "stressed"       # Multiple resources low (<25%), conservative action selection
    CRISIS = "crisis"           # Multiple resources critical (<10%), emergency-only actions
    SLEEP = "sleep"             # Near-complete exhaustion, halt all actions for recovery


@dataclass
class MultiResourceBudget:
    """Multi-dimensional resource budget system with crisis detection."""
    compute_atp: float = 100.0
    memory_atp: float = 100.0
    tool_atp: float = 100.0
    latency_budget: float = 1000.0
    risk_budget: float = 1.0

    # Calibrated recovery rates (from Session 109)
    compute_recovery: float = 2.4
    memory_recovery: float = 1.2
    tool_recovery: float = 12.0
    latency_recovery: float = 24.0
    risk_recovery: float = 0.06

    # Operational mode tracking
    mode: OperationalMode = OperationalMode.NORMAL
    sleep_cycles: int = 0
    crisis_entries: int = 0

    def get_resource_levels(self) -> Dict[str, float]:
        """Get current resource levels as fractions (0.0-1.0)."""
        return {
            'compute': self.compute_atp / 100.0,
            'memory': self.memory_atp / 100.0,
            'tool': self.tool_atp / 100.0,
            'latency': self.latency_budget / 1000.0,
            'risk': self.risk_budget / 1.0,
        }

    def count_depleted_resources(self, threshold: float = 0.1) -> Tuple[int, List[str]]:
        """
        Count resources below threshold.

        Args:
            threshold: Resource level threshold (default 0.1 = 10%)

        Returns:
            (count, list of depleted resource names)
        """
        levels = self.get_resource_levels()
        depleted = [name for name, level in levels.items() if level < threshold]
        return len(depleted), depleted

    def assess_operational_mode(self) -> OperationalMode:
        """
        Assess appropriate operational mode based on resource levels.

        Thresholds:
        - NORMAL: All resources >25%
        - STRESSED: 1-2 resources <25%
        - CRISIS: 3+ resources <10%
        - SLEEP: 4+ resources <5% (near-complete exhaustion)
        """
        levels = self.get_resource_levels()

        # Count resources at different thresholds
        critical = sum(1 for level in levels.values() if level < 0.05)  # <5%
        depleted = sum(1 for level in levels.values() if level < 0.10)  # <10%
        low = sum(1 for level in levels.values() if level < 0.25)       # <25%

        # Determine mode
        if critical >= 4:
            return OperationalMode.SLEEP
        elif depleted >= 3:
            return OperationalMode.CRISIS
        elif low >= 1:
            return OperationalMode.STRESSED
        else:
            return OperationalMode.NORMAL

    def update_mode(self) -> Tuple[OperationalMode, OperationalMode]:
        """
        Update operational mode based on current resource levels.

        Returns:
            (previous_mode, new_mode)
        """
        prev_mode = self.mode
        new_mode = self.assess_operational_mode()

        # Track mode transitions
        if new_mode == OperationalMode.SLEEP and prev_mode != OperationalMode.SLEEP:
            self.sleep_cycles = 0  # Reset sleep counter
        if new_mode == OperationalMode.CRISIS and prev_mode not in [OperationalMode.CRISIS, OperationalMode.SLEEP]:
            self.crisis_entries += 1

        self.mode = new_mode
        return prev_mode, new_mode

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

    def recover(self, mode_override: Optional[OperationalMode] = None):
        """
        Apply passive recovery rates.

        In SLEEP mode, apply enhanced recovery (2x normal rates)
        to accelerate restoration.
        """
        active_mode = mode_override if mode_override else self.mode

        if active_mode == OperationalMode.SLEEP:
            # Enhanced recovery in sleep mode
            recovery_multiplier = 2.0
            self.sleep_cycles += 1
        else:
            recovery_multiplier = 1.0

        self.compute_atp = min(100.0, self.compute_atp + self.compute_recovery * recovery_multiplier)
        self.memory_atp = min(100.0, self.memory_atp + self.memory_recovery * recovery_multiplier)
        self.tool_atp = min(100.0, self.tool_atp + self.tool_recovery * recovery_multiplier)
        self.latency_budget = min(1000.0, self.latency_budget + self.latency_recovery * recovery_multiplier)
        self.risk_budget = min(1.0, self.risk_budget + self.risk_recovery * recovery_multiplier)


@dataclass
class MultiResourceAction:
    """Action with multi-dimensional resource costs and priority."""
    action_type: str
    target: str
    expected_pressure_reduction: float
    priority: str = "normal"  # "emergency", "high", "normal", "low"
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

    def is_allowed_in_mode(self, mode: OperationalMode) -> bool:
        """Check if action is allowed in current operational mode."""
        if mode == OperationalMode.NORMAL:
            return True  # All actions allowed
        elif mode == OperationalMode.STRESSED:
            return self.priority in ["emergency", "high", "normal"]  # Block low-priority
        elif mode == OperationalMode.CRISIS:
            return self.priority in ["emergency", "high"]  # Emergency and high only
        elif mode == OperationalMode.SLEEP:
            return self.priority == "emergency"  # Emergency only
        return False


def get_action_profiles() -> Dict[str, Dict]:
    """
    Return action resource profiles with priorities.

    New in Session 110: Added priority levels for crisis mode filtering.
    """
    return {
        'consolidation': {
            'compute': 8.0,
            'memory': 6.0,
            'tool': 0.0,
            'latency': 50.0,
            'risk': 0.1,
            'priority': 'normal',
        },
        'pruning': {
            'compute': 2.0,
            'memory': 5.0,
            'tool': 0.0,
            'latency': 20.0,
            'risk': 0.2,
            'priority': 'high',  # High priority: Frees memory under stress
        },
        'index_rebuild': {
            'compute': 5.0,
            'memory': 7.0,
            'tool': 0.0,
            'latency': 45.0,
            'risk': 0.05,
            'priority': 'low',  # Low priority: Can defer
        },
        'probe': {
            'compute': 3.0,
            'memory': 1.0,
            'tool': 10.0,
            'latency': 100.0,
            'risk': 0.4,
            'priority': 'normal',
        },
        'triage': {
            'compute': 7.0,
            'memory': 2.0,
            'tool': 0.0,
            'latency': 40.0,
            'risk': 0.15,
            'priority': 'high',  # High priority: Reduces risk
        },
        'log_state': {
            'compute': 0.5,
            'memory': 0.1,
            'tool': 0.0,
            'latency': 5.0,
            'risk': 0.0,
            'priority': 'emergency',  # Emergency: Always executable
        },
    }


class CrisisModeController:
    """Controller for crisis mode detection and sleep mode triggering."""

    def __init__(self, budget: MultiResourceBudget):
        """Initialize with budget instance."""
        self.budget = budget
        self.mode_history: List[Tuple[int, OperationalMode]] = []
        self.mode_transitions: List[Tuple[int, OperationalMode, OperationalMode]] = []

    def update_and_log_mode(self, cycle: int) -> Tuple[OperationalMode, OperationalMode]:
        """
        Update operational mode and log transition.

        Returns:
            (previous_mode, new_mode)
        """
        prev_mode, new_mode = self.budget.update_mode()

        self.mode_history.append((cycle, new_mode))

        if prev_mode != new_mode:
            self.mode_transitions.append((cycle, prev_mode, new_mode))
            logger.info(f"Cycle {cycle}: Mode transition {prev_mode.value} → {new_mode.value}")

        return prev_mode, new_mode

    def get_mode_distribution(self) -> Dict[str, int]:
        """Get cycle count per operational mode."""
        distribution = {mode.value: 0 for mode in OperationalMode}
        for _, mode in self.mode_history:
            distribution[mode.value] += 1
        return distribution


def run_crisis_mode_test(regime: str, cycles: int = 200) -> Dict:
    """
    Test crisis mode integration with different stress regimes.

    Args:
        regime: Stress regime name
        cycles: Number of simulation cycles

    Returns:
        Test results dictionary
    """
    logger.info(f"Running crisis mode test: {regime}")

    budget = MultiResourceBudget()
    controller = CrisisModeController(budget)
    profiles = get_action_profiles()

    actions_executed = 0
    actions_blocked_resources = 0
    actions_blocked_mode = 0
    sleep_mode_entered = False
    max_sleep_duration = 0
    current_sleep_duration = 0

    for cycle in range(cycles):
        # Update operational mode
        prev_mode, new_mode = controller.update_and_log_mode(cycle)

        # Track sleep mode entry
        if new_mode == OperationalMode.SLEEP and prev_mode != OperationalMode.SLEEP:
            sleep_mode_entered = True
            current_sleep_duration = 0

        if new_mode == OperationalMode.SLEEP:
            current_sleep_duration += 1
            max_sleep_duration = max(max_sleep_duration, current_sleep_duration)
        else:
            current_sleep_duration = 0

        # Apply regime-specific stress
        if regime == "multi_resource_depletion":
            # Drain all resources simultaneously
            budget.compute_atp = max(0, budget.compute_atp - 0.5)
            budget.memory_atp = max(0, budget.memory_atp - 0.5)
            budget.tool_atp = max(0, budget.tool_atp - 0.5)
            budget.risk_budget = max(0, budget.risk_budget - 0.005)

        elif regime == "compute_starvation":
            # Focus stress on compute
            budget.compute_atp = max(0, budget.compute_atp - 0.5)

        # Passive recovery
        budget.recover()

        # Try to execute action based on mode
        if new_mode == OperationalMode.SLEEP:
            # In sleep mode, only emergency actions allowed
            action_type = 'log_state'
        elif new_mode == OperationalMode.CRISIS:
            # In crisis, prefer high-priority actions
            action_type = 'pruning'  # High priority
        else:
            # Normal/stressed: prefer consolidation
            action_type = 'consolidation'

        prof = profiles[action_type]
        action = MultiResourceAction(
            action_type=action_type,
            target='test',
            expected_pressure_reduction=0.1,
            priority=prof['priority'],
            compute_cost=prof['compute'],
            memory_cost=prof['memory'],
            tool_cost=prof['tool'],
            latency_cost=prof['latency'],
            risk_cost=prof['risk'],
        )

        # Check mode permission
        if not action.is_allowed_in_mode(new_mode):
            actions_blocked_mode += 1
            continue

        # Check resource availability
        can_execute, bottlenecks = budget.can_afford(action.get_costs())

        if can_execute:
            budget.consume(action.get_costs())
            actions_executed += 1
        else:
            actions_blocked_resources += 1

    # Collect results
    mode_distribution = controller.get_mode_distribution()

    return {
        'regime': regime,
        'cycles': cycles,
        'actions_executed': actions_executed,
        'actions_blocked_resources': actions_blocked_resources,
        'actions_blocked_mode': actions_blocked_mode,
        'total_blocked': actions_blocked_resources + actions_blocked_mode,
        'sleep_mode_entered': sleep_mode_entered,
        'max_sleep_duration': max_sleep_duration,
        'crisis_entries': budget.crisis_entries,
        'mode_distribution': mode_distribution,
        'mode_transitions': len(controller.mode_transitions),
        'final_mode': budget.mode.value,
        'final_budgets': {
            'compute': budget.compute_atp,
            'memory': budget.memory_atp,
            'tool': budget.tool_atp,
        },
    }


def run_session_110() -> Dict:
    """
    Execute Session 110: Crisis Mode Integration.

    Tests crisis mode controller with different stress regimes
    to validate sleep mode triggers and recovery.
    """
    logger.info("=" * 80)
    logger.info("SESSION 110: CRISIS MODE INTEGRATION")
    logger.info("=" * 80)
    logger.info("Goal: Formalize passive recovery through 'sleep mode'")
    logger.info("")

    results = {}

    # Test 1: Multi-resource depletion (should trigger sleep mode)
    logger.info("\nTest 1: Multi-Resource Depletion (sleep mode trigger test)")
    logger.info("-" * 80)
    results['multi_resource_depletion'] = run_crisis_mode_test('multi_resource_depletion', cycles=200)

    # Test 2: Compute starvation (should trigger crisis but maybe not sleep)
    logger.info("\nTest 2: Compute Starvation (crisis mode test)")
    logger.info("-" * 80)
    results['compute_starvation'] = run_crisis_mode_test('compute_starvation', cycles=200)

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SESSION 110 RESULTS SUMMARY")
    logger.info("=" * 80)

    for regime_name, regime_results in results.items():
        logger.info(f"\n{regime_name.upper()}:")
        logger.info(f"  Sleep mode entered: {regime_results['sleep_mode_entered']}")
        logger.info(f"  Max sleep duration: {regime_results['max_sleep_duration']} cycles")
        logger.info(f"  Crisis entries: {regime_results['crisis_entries']}")
        logger.info(f"  Mode transitions: {regime_results['mode_transitions']}")
        logger.info(f"  Actions executed: {regime_results['actions_executed']}")
        logger.info(f"  Actions blocked (mode): {regime_results['actions_blocked_mode']}")
        logger.info(f"  Actions blocked (resources): {regime_results['actions_blocked_resources']}")
        logger.info(f"  Final mode: {regime_results['final_mode']}")

        logger.info(f"  Mode distribution:")
        for mode, count in regime_results['mode_distribution'].items():
            pct = (count / regime_results['cycles']) * 100
            logger.info(f"    {mode}: {count} cycles ({pct:.1f}%)")

    # Save results
    output = {
        'session': 110,
        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'results': results,
    }

    output_file = 'sage/experiments/session110_crisis_mode_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("SESSION 110 COMPLETE")
    logger.info("=" * 80)

    return output


if __name__ == "__main__":
    results = run_session_110()
