#!/usr/bin/env python3
"""
Session 107: Multi-Resource Budgets - Beyond Scalar ATP

**Goal**: Address Nova's "semantic placeholders" critique with multi-dimensional resource budgets

**Context - Nova's Critique** (2025-12-23):
> "Your variables risk being 'semantic placeholders': ATP in particular easily
> becomes a renamed budget counter unless it is tightly grounded in measurable
> costs (latency, $ cost, error rates, memory growth, rate limits) and has
> conserved dynamics."

**Recommendation**:
> "If 'biological realism' is a goal, move from one global ATP to:
> - multi-budget (compute, tool calls, memory writes, risk exposure, latency),
> - multi-timescale controllers (fast gating vs slow learning),
> - explicit prediction-error channels and credit assignment."

**Research Question**:
What happens when we move from scalar ATP to multi-dimensional resource budgets?

**Hypothesis**:
Multi-dimensional constraints will reveal **emergent prioritization patterns** that
don't exist with scalar budgets. Different resource types will create different
bottlenecks, leading to more nuanced action selection.

**Multi-Resource Dimensions**:
1. **Compute ATP** - LLM inference cost (tokens, FLOPs)
2. **Memory ATP** - Memory writes, storage growth (bytes)
3. **Tool ATP** - External tool calls (API calls, latency)
4. **Latency Budget** - Time constraints (milliseconds)
5. **Risk Budget** - Uncertainty tolerance, error exposure

**Expected Insights**:
- Actions have different resource profiles (memory-heavy vs compute-heavy)
- Bottleneck shifts reveal system constraints
- Multi-dimensional optimization creates trade-offs (Pareto fronts)
- Recovery times differ per resource type
- Failure modes change (one resource exhausted while others available)

**Architecture**:
- Extends Session 106 hardened wake system
- Replaces scalar ATP with MultiResourceBudget
- Action costs become multi-dimensional vectors
- Wake policy checks all resource constraints
- Separate recovery rates per resource

Created: 2025-12-24 08:15 UTC (Autonomous Session 107)
Hardware: Thor (Jetson AGX Thor)
Previous: Session 106 (architectural hardening)
Goal: Ground ATP in measurable costs via multi-dimensional budgets
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from enum import Enum

# Import Session 106 hardened system
import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from session106_architectural_hardening import (
        HardenedWakeSystem,
        QueueCrisisController,
        AntiOscillationController,
    )
    HAS_HARDENED = True
except ImportError:
    HAS_HARDENED = False

# Import base components
try:
    from session103_internal_wake_policy import (
        MemoryPressureSignals,
        UncertaintyPressureSignals,
        WakeAction,
    )
    HAS_WAKE_POLICY = True
except ImportError:
    HAS_WAKE_POLICY = False

try:
    from session104_wake_sage_integration import (
        SAGEMemoryState,
        SAGEEpistemicState,
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


@dataclass
class MultiResourceBudget:
    """
    Multi-dimensional resource budget system.

    Addresses Nova's critique: "ATP risks being a semantic placeholder."

    Each dimension maps to measurable costs:
    - compute_atp: LLM inference (tokens × cost_per_token)
    - memory_atp: Memory writes (bytes × cost_per_byte)
    - tool_atp: API calls (calls × cost_per_call)
    - latency_budget: Time constraints (milliseconds available)
    - risk_budget: Uncertainty tolerance (0-1 scale)

    Unlike scalar ATP, multi-dimensional budgets create trade-offs:
    - Actions have different resource profiles
    - Bottlenecks shift dynamically
    - Recovery rates differ per resource
    """

    # Current budgets
    compute_atp: float = 100.0  # LLM inference budget
    memory_atp: float = 100.0   # Memory write budget
    tool_atp: float = 100.0     # Tool call budget
    latency_budget: float = 1000.0  # Time budget (ms)
    risk_budget: float = 1.0    # Uncertainty tolerance

    # Maximum budgets
    compute_atp_max: float = 100.0
    memory_atp_max: float = 100.0
    tool_atp_max: float = 100.0
    latency_budget_max: float = 1000.0
    risk_budget_max: float = 1.0

    # Recovery rates (per cycle)
    compute_recovery_rate: float = 1.0
    memory_recovery_rate: float = 2.0  # Memory recovers faster
    tool_recovery_rate: float = 0.5    # Tool calls rate-limited
    latency_recovery_rate: float = 100.0  # Time replenishes quickly
    risk_recovery_rate: float = 0.1    # Risk tolerance recovers slowly

    def can_afford(self, action_costs: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Check if all resource budgets can afford action.

        Returns:
            (can_afford, bottleneck_resources)
        """

        bottlenecks = []

        if action_costs.get('compute', 0) > self.compute_atp:
            bottlenecks.append('compute_atp')

        if action_costs.get('memory', 0) > self.memory_atp:
            bottlenecks.append('memory_atp')

        if action_costs.get('tool', 0) > self.tool_atp:
            bottlenecks.append('tool_atp')

        if action_costs.get('latency', 0) > self.latency_budget:
            bottlenecks.append('latency_budget')

        if action_costs.get('risk', 0) > self.risk_budget:
            bottlenecks.append('risk_budget')

        can_afford = len(bottlenecks) == 0

        return can_afford, bottlenecks

    def consume(self, action_costs: Dict[str, float]) -> Dict[str, float]:
        """Consume resources for action.

        Returns:
            actual_costs (may be clamped to available budgets)
        """

        actual_costs = {}

        # Consume from each budget
        compute_cost = min(action_costs.get('compute', 0), self.compute_atp)
        self.compute_atp -= compute_cost
        actual_costs['compute'] = compute_cost

        memory_cost = min(action_costs.get('memory', 0), self.memory_atp)
        self.memory_atp -= memory_cost
        actual_costs['memory'] = memory_cost

        tool_cost = min(action_costs.get('tool', 0), self.tool_atp)
        self.tool_atp -= tool_cost
        actual_costs['tool'] = tool_cost

        latency_cost = min(action_costs.get('latency', 0), self.latency_budget)
        self.latency_budget -= latency_cost
        actual_costs['latency'] = latency_cost

        risk_cost = min(action_costs.get('risk', 0), self.risk_budget)
        self.risk_budget -= risk_cost
        actual_costs['risk'] = risk_cost

        return actual_costs

    def recover(self, state: str = 'WAKE') -> Dict[str, float]:
        """Recover resources based on metabolic state.

        Recovery rates depend on state:
        - WAKE: Normal recovery
        - REST: 2x recovery (ATP, latency, risk faster)
        - CRISIS: Minimal recovery (only latency)

        Returns:
            recovery_amounts per resource
        """

        recovery = {}

        if state == 'CRISIS':
            # CRISIS: Only latency recovers
            latency_recovery = min(
                self.latency_recovery_rate * 0.5,
                self.latency_budget_max - self.latency_budget
            )
            self.latency_budget += latency_recovery
            recovery['latency'] = latency_recovery

            return recovery

        elif state == 'REST':
            # REST: 2x recovery on most resources
            multiplier = 2.0
        else:
            # WAKE: Normal recovery
            multiplier = 1.0

        # Recover compute ATP
        compute_recovery = min(
            self.compute_recovery_rate * multiplier,
            self.compute_atp_max - self.compute_atp
        )
        self.compute_atp += compute_recovery
        recovery['compute'] = compute_recovery

        # Recover memory ATP
        memory_recovery = min(
            self.memory_recovery_rate * multiplier,
            self.memory_atp_max - self.memory_atp
        )
        self.memory_atp += memory_recovery
        recovery['memory'] = memory_recovery

        # Recover tool ATP
        tool_recovery = min(
            self.tool_recovery_rate * multiplier,
            self.tool_atp_max - self.tool_atp
        )
        self.tool_atp += tool_recovery
        recovery['tool'] = tool_recovery

        # Recover latency budget
        latency_recovery = min(
            self.latency_recovery_rate * multiplier,
            self.latency_budget_max - self.latency_budget
        )
        self.latency_budget += latency_recovery
        recovery['latency'] = latency_recovery

        # Recover risk budget
        risk_recovery = min(
            self.risk_recovery_rate * multiplier,
            self.risk_budget_max - self.risk_budget
        )
        self.risk_budget += risk_recovery
        recovery['risk'] = risk_recovery

        return recovery

    def get_limiting_resource(self) -> Tuple[str, float]:
        """Identify which resource is most depleted (bottleneck).

        Returns:
            (resource_name, depletion_ratio)
        """

        ratios = {
            'compute': 1.0 - (self.compute_atp / self.compute_atp_max),
            'memory': 1.0 - (self.memory_atp / self.memory_atp_max),
            'tool': 1.0 - (self.tool_atp / self.tool_atp_max),
            'latency': 1.0 - (self.latency_budget / self.latency_budget_max),
            'risk': 1.0 - (self.risk_budget / self.risk_budget_max),
        }

        limiting_resource = max(ratios.items(), key=lambda x: x[1])
        return limiting_resource

    def to_dict(self) -> Dict[str, float]:
        """Export current state."""
        return {
            'compute_atp': self.compute_atp,
            'memory_atp': self.memory_atp,
            'tool_atp': self.tool_atp,
            'latency_budget': self.latency_budget,
            'risk_budget': self.risk_budget,
        }


@dataclass
class MultiResourceAction:
    """
    Wake action with multi-dimensional resource costs.

    Extends WakeAction with resource profile.
    Different actions have different resource signatures:
    - Consolidation: High compute, high memory, low tool
    - Pruning: Low compute, high memory, low tool
    - Index rebuild: Medium compute, high memory, low tool
    - Hypothesis triage: High compute, low memory, low tool
    - Uncertainty probe: Low compute, low memory, high tool, high risk
    """

    action_type: str
    target: str

    # Resource costs
    compute_cost: float = 0.0
    memory_cost: float = 0.0
    tool_cost: float = 0.0
    latency_cost: float = 0.0
    risk_cost: float = 0.0

    # Expected outcomes
    expected_pressure_reduction: float = 0.0
    duration_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def get_cost_dict(self) -> Dict[str, float]:
        """Get costs as dictionary for budget checking."""
        return {
            'compute': self.compute_cost,
            'memory': self.memory_cost,
            'tool': self.tool_cost,
            'latency': self.latency_cost,
            'risk': self.risk_cost,
        }

    def get_efficiency(self, resource_type: str) -> float:
        """Get pressure reduction per unit of specific resource.

        Allows prioritizing actions based on which resource is limiting.
        """

        cost = self.get_cost_dict().get(resource_type, 0.0)

        if cost == 0.0:
            return float('inf')  # Free actions are infinitely efficient

        return self.expected_pressure_reduction / cost


class MultiResourceWakeSystem:
    """
    Wake system with multi-dimensional resource budgets.

    Replaces scalar ATP (Session 106) with MultiResourceBudget.

    Key differences:
    - Actions have resource profiles (not scalar costs)
    - Can afford checks all dimensions
    - Recovery rates differ per resource
    - Bottleneck shifts dynamically
    - Action selection adapts to limiting resource
    """

    def __init__(
        self,
        # Wake policy parameters
        wake_threshold: float = 0.4,
        sleep_threshold: float = 0.2,
        # Initial resource budgets
        initial_compute: float = 100.0,
        initial_memory: float = 100.0,
        initial_tool: float = 100.0,
        initial_latency: float = 1000.0,
        initial_risk: float = 1.0,
        # Crisis and anti-oscillation (from S106)
        queue_soft_limit: int = 500,
        queue_hard_limit: int = 1000,
        queue_emergency_limit: int = 1500,
        min_wake_duration: int = 10,
        min_sleep_duration: int = 5,
        pressure_alpha: float = 0.3,
    ):
        """Initialize multi-resource wake system."""

        # Multi-resource budget
        self.budget = MultiResourceBudget(
            compute_atp=initial_compute,
            memory_atp=initial_memory,
            tool_atp=initial_tool,
            latency_budget=initial_latency,
            risk_budget=initial_risk,
        )

        # Wake policy (simplified - will integrate with Session 103 later)
        self.wake_threshold = wake_threshold
        self.sleep_threshold = sleep_threshold
        self.is_awake = False

        # Crisis controllers (from Session 106)
        if HAS_HARDENED:
            self.queue_crisis = QueueCrisisController(
                soft_limit=queue_soft_limit,
                hard_limit=queue_hard_limit,
                emergency_limit=queue_emergency_limit,
            )

            self.anti_oscillation = AntiOscillationController(
                min_wake_duration=min_wake_duration,
                min_sleep_duration=min_sleep_duration,
                pressure_alpha=pressure_alpha,
            )
        else:
            self.queue_crisis = None
            self.anti_oscillation = None

        # SAGE state (simplified)
        if HAS_INTEGRATION:
            self.memory_state = SAGEMemoryState()
            self.epistemic_state = SAGEEpistemicState()
        else:
            # Minimal state for testing
            self.memory_state = type('MemoryState', (), {
                'unprocessed_memories': 0,
                'total_memories': 0,
            })()
            self.epistemic_state = type('EpistemicState', (), {
                'uncertainty': 0.0,
                'confidence': 0.8,
            })()

        # Statistics
        self.cycles_run = 0
        self.total_wakes = 0
        self.actions_executed = []
        self.bottleneck_history = []

    def define_action_resource_profiles(self) -> Dict[str, Dict[str, float]]:
        """Define resource cost profiles for each action type.

        Different actions have different resource signatures:
        - Consolidation: High compute + high memory (intensive processing)
        - Pruning: Low compute + high memory (selective deletion)
        - Index rebuild: Medium compute + high memory (reorganization)
        - Hypothesis triage: High compute + low memory (reasoning-heavy)
        - Uncertainty probe: Low compute + high tool + high risk (external query)
        """

        profiles = {
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
                'compute': 5.0,   # Medium: Rebuild indices
                'memory': 7.0,    # High: Rewrite index structures
                'tool': 0.0,      # None: Internal operation
                'latency': 45.0,  # Moderate: Reorganization takes time
                'risk': 0.05,     # Very low: Just reorganizing
            },
            'triage': {
                'compute': 7.0,   # High: Hypothesis evaluation
                'memory': 2.0,    # Low: Just marking/discarding
                'tool': 0.0,      # None: Internal reasoning
                'latency': 30.0,  # Moderate: Evaluation takes time
                'risk': 0.15,     # Low-moderate: Might discard useful hypotheses
            },
            'probe': {
                'compute': 3.0,   # Low-medium: Query formulation
                'memory': 1.0,    # Low: Minimal writes
                'tool': 10.0,     # HIGH: External API call
                'latency': 200.0, # High: Network latency
                'risk': 0.4,      # High: External query, uncertain results
            },
        }

        return profiles

    def select_actions_multi_resource(
        self,
        memory_pressure: float,
        uncertainty_pressure: float,
    ) -> List[MultiResourceAction]:
        """Select actions based on multi-resource constraints.

        Strategy:
        1. Identify limiting resource (biggest bottleneck)
        2. Select actions efficient for that resource
        3. Filter by affordability (all dimensions)
        4. Sort by efficiency for limiting resource
        """

        actions = []
        profiles = self.define_action_resource_profiles()

        # Identify limiting resource
        limiting_resource, depletion_ratio = self.budget.get_limiting_resource()

        logger.info(f"Limiting resource: {limiting_resource} (depletion={depletion_ratio:.2f})")

        # Generate candidate actions based on pressure
        if memory_pressure > 0.4:
            # Need memory management
            prof = profiles['consolidation']
            actions.append(MultiResourceAction(
                action_type='consolidation',
                target='recent_memories',
                expected_pressure_reduction=0.3,
                compute_cost=prof['compute'],
                memory_cost=prof['memory'],
                tool_cost=prof['tool'],
                latency_cost=prof['latency'],
                risk_cost=prof['risk'],
            ))

            prof = profiles['pruning']
            actions.append(MultiResourceAction(
                action_type='pruning',
                target='stale_memories',
                expected_pressure_reduction=0.2,
                compute_cost=prof['compute'],
                memory_cost=prof['memory'],
                tool_cost=prof['tool'],
                latency_cost=prof['latency'],
                risk_cost=prof['risk'],
            ))

            prof = profiles['index_rebuild']
            actions.append(MultiResourceAction(
                action_type='index_rebuild',
                target='memory_index',
                expected_pressure_reduction=0.25,
                compute_cost=prof['compute'],
                memory_cost=prof['memory'],
                tool_cost=prof['tool'],
                latency_cost=prof['latency'],
                risk_cost=prof['risk'],
            ))

        if uncertainty_pressure > 0.4:
            # Need uncertainty reduction
            prof = profiles['triage']
            actions.append(MultiResourceAction(
                action_type='triage',
                target='unresolved_hypotheses',
                expected_pressure_reduction=0.2,
                compute_cost=prof['compute'],
                memory_cost=prof['memory'],
                tool_cost=prof['tool'],
                latency_cost=prof['latency'],
                risk_cost=prof['risk'],
            ))

            prof = profiles['probe']
            actions.append(MultiResourceAction(
                action_type='probe',
                target='calibration_model',
                expected_pressure_reduction=0.15,
                compute_cost=prof['compute'],
                memory_cost=prof['memory'],
                tool_cost=prof['tool'],
                latency_cost=prof['latency'],
                risk_cost=prof['risk'],
            ))

        # Filter by affordability (all resources)
        affordable_actions = []
        for action in actions:
            can_afford, bottlenecks = self.budget.can_afford(action.get_cost_dict())

            if can_afford:
                affordable_actions.append(action)
            else:
                logger.debug(f"Cannot afford {action.action_type}: bottlenecks={bottlenecks}")

        # Sort by efficiency for limiting resource
        affordable_actions.sort(
            key=lambda a: a.get_efficiency(limiting_resource),
            reverse=True
        )

        return affordable_actions

    def execute_action(self, action: MultiResourceAction):
        """Execute action and consume resources."""

        actual_costs = self.budget.consume(action.get_cost_dict())

        logger.info(
            f"Executed {action.action_type}: "
            f"compute={actual_costs['compute']:.1f}, "
            f"memory={actual_costs['memory']:.1f}, "
            f"tool={actual_costs['tool']:.1f}, "
            f"latency={actual_costs['latency']:.1f}ms, "
            f"risk={actual_costs['risk']:.2f}"
        )

        self.actions_executed.append({
            'cycle': self.cycles_run,
            'action': action.action_type,
            'costs': actual_costs,
        })

    def run_multi_resource_simulation(self, cycles: int = 200):
        """Run simulation demonstrating multi-resource dynamics.

        Expected insights:
        - Bottleneck shifts between resources
        - Different actions selected based on limiting resource
        - Recovery rates create different dynamics
        - Multi-dimensional constraints reveal trade-offs
        """

        logger.info("="*80)
        logger.info("SESSION 107: MULTI-RESOURCE BUDGET SIMULATION")
        logger.info("="*80)
        logger.info(f"Cycles: {cycles}")
        logger.info(f"Resource budgets: {self.budget.to_dict()}")
        logger.info("")

        trajectory = []
        wake_events = []

        for cycle in range(cycles):
            self.cycles_run = cycle + 1

            # Simulate pressure accumulation (simplified)
            memory_pressure = min(0.1 + cycle * 0.005, 1.0)
            uncertainty_pressure = min(0.1 + cycle * 0.003, 1.0)

            # Check wake policy (simplified - just use pressure)
            overall_pressure = max(memory_pressure, uncertainty_pressure)

            should_wake = overall_pressure > self.wake_threshold

            # Identify limiting resource
            limiting_resource, depletion_ratio = self.budget.get_limiting_resource()

            self.bottleneck_history.append({
                'cycle': cycle,
                'resource': limiting_resource,
                'depletion': depletion_ratio,
            })

            # Execute actions if awake
            if should_wake:
                if not self.is_awake:
                    self.is_awake = True
                    self.total_wakes += 1
                    logger.info(f"Wake triggered at cycle {cycle}")

                # Select actions based on multi-resource constraints
                actions = self.select_actions_multi_resource(
                    memory_pressure=memory_pressure,
                    uncertainty_pressure=uncertainty_pressure,
                )

                if actions:
                    # Execute top action (most efficient for limiting resource)
                    self.execute_action(actions[0])

                    wake_events.append({
                        'cycle': cycle,
                        'action': actions[0].action_type,
                        'limiting_resource': limiting_resource,
                    })

            else:
                if self.is_awake:
                    self.is_awake = False
                    logger.info(f"Sleep triggered at cycle {cycle}")

            # Resource recovery
            recovery = self.budget.recover(state='WAKE' if self.is_awake else 'REST')

            # Track trajectory
            trajectory.append({
                'cycle': cycle,
                'memory_pressure': memory_pressure,
                'uncertainty_pressure': uncertainty_pressure,
                'is_awake': self.is_awake,
                'limiting_resource': limiting_resource,
                'budgets': self.budget.to_dict(),
                'recovery': recovery,
            })

            # Periodic logging
            if cycle % 50 == 0:
                logger.info(
                    f"Cycle {cycle}: "
                    f"pressure={overall_pressure:.3f}, "
                    f"awake={self.is_awake}, "
                    f"limiting={limiting_resource}, "
                    f"compute={self.budget.compute_atp:.1f}, "
                    f"memory={self.budget.memory_atp:.1f}, "
                    f"tool={self.budget.tool_atp:.1f}"
                )

        # Final report
        logger.info("="*80)
        logger.info("SIMULATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total cycles: {cycles}")
        logger.info(f"Total wakes: {self.total_wakes}")
        logger.info(f"Actions executed: {len(self.actions_executed)}")

        # Analyze bottleneck dynamics
        bottleneck_counts = defaultdict(int)
        for entry in self.bottleneck_history:
            bottleneck_counts[entry['resource']] += 1

        logger.info(f"\nBottleneck distribution:")
        for resource, count in sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(self.bottleneck_history)) * 100
            logger.info(f"  {resource}: {count} cycles ({pct:.1f}%)")

        # Save results
        results = {
            'session': 107,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            'cycles': cycles,
            'total_wakes': self.total_wakes,
            'actions_executed': len(self.actions_executed),
            'bottleneck_distribution': dict(bottleneck_counts),
            'action_details': self.actions_executed,
            'trajectory': trajectory,
            'final_budgets': self.budget.to_dict(),
        }

        output_path = Path(__file__).parent / "session107_multi_resource_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")

        return results


def run_session_107():
    """Run Session 107: Multi-Resource Budgets."""

    logger.info("="*80)
    logger.info("SESSION 107: MULTI-RESOURCE BUDGETS")
    logger.info("="*80)
    logger.info("Goal: Move from scalar ATP to multi-dimensional resource budgets")
    logger.info("Context: Address Nova's 'semantic placeholders' critique")
    logger.info("")

    # Create multi-resource system
    system = MultiResourceWakeSystem(
        wake_threshold=0.4,
        sleep_threshold=0.2,
        initial_compute=100.0,
        initial_memory=100.0,
        initial_tool=100.0,
        initial_latency=1000.0,
        initial_risk=1.0,
    )

    # Run simulation
    results = system.run_multi_resource_simulation(cycles=200)

    return results


if __name__ == "__main__":
    results = run_session_107()
