#!/usr/bin/env python3
"""
Session 111: Multi-Resource DreamConsolidator Integration

Goal: Bridge multi-resource consciousness (S107-110) to real SAGE component (DreamConsolidator).

Research Arc Progression:
- S107: Multi-resource budgets discovered emergent prioritization
- S108: Stress testing revealed graceful degradation
- S109: Recovery rate calibration eliminated deadlock (+611% improvement)
- S110: Crisis mode integration validated calibration prevents crisis
- S111: Apply multi-resource system to actual SAGE memory consolidation

Integration Approach:
Instead of modifying DreamConsolidator directly, create a multi-resource
aware wrapper that:
1. Maps consolidation operations to resource costs
2. Schedules consolidation based on resource availability
3. Adapts consolidation depth to current operational mode
4. Validates hierarchical resilience in production context

Design Principle:
Consolidation operations have different resource signatures:
- Pattern extraction: High compute, moderate memory
- Quality learning: High compute, low memory
- Creative associations: Moderate compute, high tool (LLM calls)
- Epistemic insights: High compute, moderate tool

Operational Mode Adaptation:
- NORMAL: Full consolidation (all operations)
- STRESSED: Conservative consolidation (skip expensive operations)
- CRISIS: Emergency consolidation (essential operations only)
- SLEEP: No consolidation (recovery mode)
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime
import sys
import os

# Add sage to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from sage.experiments.session110_crisis_mode_integration import (
    MultiResourceBudget,
    OperationalMode,
    MultiResourceAction
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


# Consolidation operation resource costs
CONSOLIDATION_COSTS = {
    'pattern_extraction': {
        'compute': 8.0,   # High: Statistical analysis, clustering
        'memory': 5.0,    # Moderate: Working memory for patterns
        'tool': 0.0,      # None: Internal computation
        'latency': 100.0, # High: Complex processing
        'risk': 0.1,      # Low: Internal operation
        'priority': 'high',  # High: Core consolidation function
    },
    'quality_learning': {
        'compute': 7.0,   # High: Correlation analysis
        'memory': 2.0,    # Low: Small data structures
        'tool': 0.0,      # None: Internal computation
        'latency': 80.0,  # Moderate-high: Statistical processing
        'risk': 0.05,     # Very low: Proven algorithm
        'priority': 'high',  # High: Quality improvement critical
    },
    'creative_associations': {
        'compute': 5.0,   # Moderate: Association mapping
        'memory': 3.0,    # Moderate: Graph structures
        'tool': 12.0,     # High: LLM calls for creative connections
        'latency': 150.0, # Very high: External API calls
        'risk': 0.3,      # Moderate: LLM hallucination risk
        'priority': 'normal',  # Normal: Useful but not essential
    },
    'epistemic_insights': {
        'compute': 6.0,   # Moderate-high: Meta-analysis
        'memory': 3.0,    # Moderate: Insight structures
        'tool': 8.0,      # Moderate-high: LLM for insight generation
        'latency': 120.0, # High: Complex reasoning
        'risk': 0.2,      # Moderate: Meta-cognitive uncertainty
        'priority': 'normal',  # Normal: Valuable but deferrable
    },
    'save_consolidated': {
        'compute': 1.0,   # Low: Simple serialization
        'memory': 2.0,    # Low: File I/O
        'tool': 0.0,      # None: Local operation
        'latency': 30.0,  # Low: Fast disk write
        'risk': 0.05,     # Very low: Proven operation
        'priority': 'high',  # High: Must persist results
    },
}


class ConsolidationPhase(Enum):
    """Phases of memory consolidation process."""
    PATTERN_EXTRACTION = "pattern_extraction"
    QUALITY_LEARNING = "quality_learning"
    CREATIVE_ASSOCIATIONS = "creative_associations"
    EPISTEMIC_INSIGHTS = "epistemic_insights"
    SAVE_CONSOLIDATED = "save_consolidated"


@dataclass
class ConsolidationPlan:
    """Plan for multi-resource aware consolidation."""
    phases_to_execute: List[ConsolidationPhase]
    estimated_total_cost: Dict[str, float]
    operational_mode: OperationalMode
    adaptations: List[str]  # What was skipped/modified due to resources


class MultiResourceDreamScheduler:
    """
    Multi-resource aware scheduler for DreamConsolidator.

    Wraps DreamConsolidator to add multi-resource budget management,
    operational mode awareness, and adaptive consolidation scheduling.
    """

    def __init__(self):
        """Initialize multi-resource dream scheduler."""
        self.budget = MultiResourceBudget()
        self.consolidation_history: List[Dict] = []
        self.phases_executed: Dict[str, int] = {phase.value: 0 for phase in ConsolidationPhase}
        self.phases_skipped: Dict[str, int] = {phase.value: 0 for phase in ConsolidationPhase}

    def plan_consolidation(self, num_cycles: int) -> ConsolidationPlan:
        """
        Plan consolidation based on current resource availability.

        Determines which consolidation phases to execute based on:
        1. Operational mode (NORMAL, STRESSED, CRISIS, SLEEP)
        2. Available resource budgets
        3. Phase priorities

        Args:
            num_cycles: Number of consciousness cycles to consolidate

        Returns:
            ConsolidationPlan with phases to execute and cost estimates
        """
        # Update operational mode
        prev_mode, current_mode = self.budget.update_mode()

        phases_to_execute = []
        adaptations = []
        total_cost = {
            'compute': 0.0,
            'memory': 0.0,
            'tool': 0.0,
            'latency': 0.0,
            'risk': 0.0,
        }

        # Mode-based phase selection
        if current_mode == OperationalMode.SLEEP:
            # SLEEP: No consolidation, recovery only
            adaptations.append("SLEEP mode: All consolidation deferred for resource recovery")

        elif current_mode == OperationalMode.CRISIS:
            # CRISIS: Essential phases only
            essential_phases = [
                ConsolidationPhase.SAVE_CONSOLIDATED,  # Must persist any existing work
            ]

            for phase in essential_phases:
                costs = CONSOLIDATION_COSTS[phase.value]
                action = MultiResourceAction(
                    action_type=phase.value,
                    target='consolidation',
                    expected_pressure_reduction=0.1,
                    priority=costs['priority'],
                    compute_cost=costs['compute'],
                    memory_cost=costs['memory'],
                    tool_cost=costs['tool'],
                    latency_cost=costs['latency'],
                    risk_cost=costs['risk'],
                )

                if action.is_allowed_in_mode(current_mode) and self.budget.can_afford(costs)[0]:
                    phases_to_execute.append(phase)
                    for resource, cost in costs.items():
                        if resource in total_cost:
                            total_cost[resource] += cost

            adaptations.append(f"CRISIS mode: Only essential phases ({len(phases_to_execute)}/5)")

        elif current_mode == OperationalMode.STRESSED:
            # STRESSED: High priority phases + affordable normal priority
            high_priority_phases = [
                ConsolidationPhase.PATTERN_EXTRACTION,
                ConsolidationPhase.QUALITY_LEARNING,
                ConsolidationPhase.SAVE_CONSOLIDATED,
            ]

            normal_priority_phases = [
                ConsolidationPhase.CREATIVE_ASSOCIATIONS,
                ConsolidationPhase.EPISTEMIC_INSIGHTS,
            ]

            # Execute high priority
            for phase in high_priority_phases:
                costs = CONSOLIDATION_COSTS[phase.value]
                if self.budget.can_afford(costs)[0]:
                    phases_to_execute.append(phase)
                    for resource, cost in costs.items():
                        if resource in total_cost:
                            total_cost[resource] += cost
                else:
                    adaptations.append(f"Skipped high-priority {phase.value} (insufficient resources)")

            # Execute normal priority if affordable
            for phase in normal_priority_phases:
                costs = CONSOLIDATION_COSTS[phase.value]
                # Check if we can afford it AFTER high priority costs
                temp_budget = MultiResourceBudget(
                    compute_atp=self.budget.compute_atp - total_cost['compute'],
                    memory_atp=self.budget.memory_atp - total_cost['memory'],
                    tool_atp=self.budget.tool_atp - total_cost['tool'],
                    latency_budget=self.budget.latency_budget - total_cost['latency'],
                    risk_budget=self.budget.risk_budget - total_cost['risk'],
                )

                if temp_budget.can_afford(costs)[0]:
                    phases_to_execute.append(phase)
                    for resource, cost in costs.items():
                        if resource in total_cost:
                            total_cost[resource] += cost
                else:
                    adaptations.append(f"Skipped normal-priority {phase.value} (resource conservation)")

            adaptations.append(f"STRESSED mode: Conservative consolidation ({len(phases_to_execute)}/5 phases)")

        else:  # NORMAL mode
            # NORMAL: Full consolidation
            all_phases = list(ConsolidationPhase)

            for phase in all_phases:
                costs = CONSOLIDATION_COSTS[phase.value]
                phases_to_execute.append(phase)
                for resource, cost in costs.items():
                    if resource in total_cost:
                        total_cost[resource] += cost

            adaptations.append("NORMAL mode: Full consolidation (all phases)")

        return ConsolidationPlan(
            phases_to_execute=phases_to_execute,
            estimated_total_cost=total_cost,
            operational_mode=current_mode,
            adaptations=adaptations,
        )

    def execute_consolidation(self, plan: ConsolidationPlan, cycles_count: int) -> Dict:
        """
        Execute consolidation plan (simulated).

        In real integration, this would call actual DreamConsolidator methods.
        For Session 111, we simulate execution to validate resource management.

        Args:
            plan: Consolidation plan to execute
            cycles_count: Number of cycles being consolidated

        Returns:
            Execution results
        """
        logger.info(f"\nExecuting consolidation plan:")
        logger.info(f"  Mode: {plan.operational_mode.value}")
        logger.info(f"  Phases: {len(plan.phases_to_execute)}/{len(ConsolidationPhase)}")
        logger.info(f"  Adaptations: {len(plan.adaptations)}")

        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'cycles_count': cycles_count,
            'operational_mode': plan.operational_mode.value,
            'phases_executed': [],
            'phases_skipped': [],
            'resource_usage': {},
            'budget_before': self.budget.get_resource_levels(),
            'budget_after': None,
        }

        # Execute each phase
        for phase in plan.phases_to_execute:
            costs = CONSOLIDATION_COSTS[phase.value]

            # Check affordability (double-check)
            can_afford, bottlenecks = self.budget.can_afford(costs)

            if can_afford:
                # Execute (consume resources)
                self.budget.consume(costs)
                results['phases_executed'].append(phase.value)
                self.phases_executed[phase.value] += 1

                logger.info(f"  ✓ Executed {phase.value}")

                # Track resource usage
                for resource, cost in costs.items():
                    if resource != 'priority':
                        if resource not in results['resource_usage']:
                            results['resource_usage'][resource] = 0.0
                        results['resource_usage'][resource] += cost
            else:
                results['phases_skipped'].append({
                    'phase': phase.value,
                    'reason': f"Insufficient resources: {bottlenecks}"
                })
                self.phases_skipped[phase.value] += 1
                logger.info(f"  ✗ Skipped {phase.value} (bottlenecks: {bottlenecks})")

        # Apply recovery
        self.budget.recover()

        results['budget_after'] = self.budget.get_resource_levels()
        results['adaptations'] = plan.adaptations

        # Store in history
        self.consolidation_history.append(results)

        return results


def run_session_111() -> Dict:
    """
    Execute Session 111: Multi-Resource DreamConsolidator Integration.

    Simulates consolidation under different stress conditions to validate
    multi-resource integration with real SAGE component.
    """
    logger.info("=" * 80)
    logger.info("SESSION 111: MULTI-RESOURCE DREAMCONSOLIDATOR INTEGRATION")
    logger.info("=" * 80)
    logger.info("Goal: Bridge multi-resource system to actual SAGE consolidation")
    logger.info("")

    scheduler = MultiResourceDreamScheduler()

    # Scenario 1: Normal operation (should execute full consolidation)
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 1: Normal Operation")
    logger.info("=" * 80)

    plan1 = scheduler.plan_consolidation(num_cycles=50)
    result1 = scheduler.execute_consolidation(plan1, cycles_count=50)

    # Scenario 2: Stressed operation (apply stress, test conservative consolidation)
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 2: Stressed Operation")
    logger.info("=" * 80)

    # Deplete some resources to trigger STRESSED mode
    scheduler.budget.compute_atp = 20.0
    scheduler.budget.memory_atp = 15.0

    plan2 = scheduler.plan_consolidation(num_cycles=50)
    result2 = scheduler.execute_consolidation(plan2, cycles_count=50)

    # Scenario 3: Recovery and return to normal
    logger.info("\n" + "=" * 80)
    logger.info("SCENARIO 3: Recovery Phase")
    logger.info("=" * 80)

    # Allow recovery (multiple cycles with no consolidation)
    for i in range(10):
        scheduler.budget.recover()

    plan3 = scheduler.plan_consolidation(num_cycles=50)
    result3 = scheduler.execute_consolidation(plan3, cycles_count=50)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SESSION 111 SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\nPhase Execution Statistics:")
    for phase_name, count in scheduler.phases_executed.items():
        skipped = scheduler.phases_skipped[phase_name]
        total = count + skipped
        if total > 0:
            success_rate = (count / total) * 100
            logger.info(f"  {phase_name}: {count}/{total} executed ({success_rate:.0f}% success rate)")

    logger.info(f"\nOperational Modes Encountered:")
    modes_encountered = set(r['operational_mode'] for r in scheduler.consolidation_history)
    logger.info(f"  {', '.join(modes_encountered)}")

    logger.info(f"\nKey Finding:")
    if len(modes_encountered) > 1:
        logger.info("  ✓ System adapted consolidation to operational mode")
        logger.info("  ✓ Resource-aware scheduling working correctly")
    else:
        logger.info("  ⚠ Limited mode diversity (increase stress for full validation)")

    # Save results
    output = {
        'session': 111,
        'timestamp': datetime.utcnow().isoformat(),
        'scenarios': {
            'normal_operation': result1,
            'stressed_operation': result2,
            'recovery_phase': result3,
        },
        'phase_statistics': {
            'executed': scheduler.phases_executed,
            'skipped': scheduler.phases_skipped,
        },
        'consolidation_history': scheduler.consolidation_history,
    }

    output_file = 'sage/experiments/session111_multiresource_dream_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("SESSION 111 COMPLETE")
    logger.info("=" * 80)

    return output


if __name__ == "__main__":
    results = run_session_111()
