"""
Session 117: Recalibrated Stress Testing

Goal: Discover emergent behaviors with properly calibrated resource costs

Session 116 revealed that costs (2-12 ATP attention, 1-5 ATP memory) were too low
relative to budgets (100 ATP) and recovery (2.4 ATP/cycle). System never left
NORMAL mode - no stress, no conflicts, no emergent behaviors discovered.

Session 117: Increase costs 10x to actually force STRESSED/CRISIS modes

Recalibrated Costs:
- Attention: 20-120 ATP (was 2-12 ATP) - 10x increase
- Memory: 10-50 ATP (was 1-5 ATP) - 10x increase
- Budgets: Still 100 ATP (compute), 100 ATP (memory)
- Recovery: Still 2.4 ATP/cycle (compute), 1.2 ATP/cycle (memory)

Expected: With 10x costs, system should:
- Enter STRESSED mode (1-2 resources <25%)
- Enter CRISIS mode (3+ resources <10%)
- Experience resource conflicts and deferrals
- Show degraded strategies (simplified_encoding, degraded_attention)
- Demonstrate priority-based allocation

Research Questions:
1. What emergent behaviors appear in STRESSED mode?
2. How does priority system resolve resource conflicts?
3. What's the degradation path NORMAL → STRESSED → CRISIS?
4. How do components compete for scarce resources?
5. What recovery dynamics appear after stress?

Test Scenarios (same as S116, recalibrated costs):
A. Heavy Attention Load (many high-salience targets)
B. Heavy Memory Load (many turns, large buffers)
C. Simultaneous Stress (attention + memory both demanding)
D. Resource Starvation (continuous depletion without recovery)
"""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime, timezone
import sys
import os

# Add sage to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from sage.experiments.session110_crisis_mode_integration import (
    MultiResourceBudget,
    OperationalMode,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ResourceCost:
    """Cost of an operation in terms of resources."""
    compute_atp: float = 0.0
    memory_atp: float = 0.0
    tool_atp: float = 0.0
    latency_budget: float = 0.0
    risk_budget: float = 0.0


class ComponentPriority(Enum):
    """Priority levels for resource allocation when scarce."""
    CRITICAL = 4  # Survival-critical (crisis attention, emergency ops)
    HIGH = 3      # Important (normal attention, consensus)
    NORMAL = 2    # Standard (memory encoding, routine processing)
    LOW = 1       # Deferrable (consolidation, creative processing)


@dataclass
class StressTestScenario:
    """A stress test scenario definition."""
    name: str
    description: str
    turns: List[Dict]  # List of turn specifications
    initial_budget: Optional[Dict] = None  # Override starting budget
    disable_recovery: bool = False  # Disable ATP recovery for starvation test


class MultiResourceSAGE:
    """
    Full SAGE system with coordinated multi-resource management.

    Simplified version for stress testing (Attention + Memory only).
    """

    def __init__(self, initial_budget: Optional[Dict] = None):
        """Initialize SAGE with optional custom starting budget."""
        # Shared resource budget
        self.budget = MultiResourceBudget()

        # Override initial budget if provided
        if initial_budget:
            for resource, value in initial_budget.items():
                setattr(self.budget, resource, value)

        # Component state
        self.attention_state = "wake"
        self.memory_buffer = []
        self.operation_history = []
        self.resource_conflicts = 0
        self.deferrals_by_component = {
            'attention': 0,
            'memory': 0,
        }

    def process_turn(
        self,
        speaker: str,
        text: str,
        salience_map: Optional[Dict[str, float]] = None,
        priority: ComponentPriority = ComponentPriority.NORMAL,
        disable_recovery: bool = False,
    ) -> Dict:
        """
        Process a single conversation turn.

        Args:
            speaker: Who is speaking
            text: What they said
            salience_map: Attention targets {target: salience}
            priority: Priority level for this turn
            disable_recovery: Disable ATP recovery (for starvation test)

        Returns:
            Processing results
        """
        turn_start_time = datetime.now(timezone.utc)
        operational_mode = self.budget.assess_operational_mode()

        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING TURN: {speaker} ({len(text)} chars)")
        logger.info(f"Operational Mode: {operational_mode.value}")
        logger.info(f"Priority: {priority.name}")
        logger.info(f"Resources: compute={self.budget.compute_atp:.1f}, "
                   f"memory={self.budget.memory_atp:.1f}, "
                   f"tool={self.budget.tool_atp:.1f}")
        logger.info(f"{'='*80}")

        # Track budget before operations
        budget_start = {
            'compute': self.budget.compute_atp,
            'memory': self.budget.memory_atp,
            'tool': self.budget.tool_atp,
            'latency': self.budget.latency_budget,
            'risk': self.budget.risk_budget,
        }

        # Results tracking
        attention_result = None
        memory_result = None
        deferrals = []

        # 1. ATTENTION ALLOCATION (if salience_map provided)
        if salience_map:
            logger.info(f"\n[1] ATTENTION ALLOCATION")
            attention_result = self._allocate_attention(
                salience_map, operational_mode, priority
            )
            if attention_result['deferred']:
                deferrals.append('attention')
                self.deferrals_by_component['attention'] += 1
                self.resource_conflicts += 1

        # 2. MEMORY ENCODING
        logger.info(f"\n[2] MEMORY ENCODING")
        memory_result = self._encode_memory(
            speaker, text, operational_mode, priority
        )
        if memory_result['deferred']:
            deferrals.append('memory')
            self.deferrals_by_component['memory'] += 1
            self.resource_conflicts += 1

        # 3. RECOVERY (if enabled)
        if not disable_recovery:
            self.budget.recover()

        # Track budget after operations
        budget_end = {
            'compute': self.budget.compute_atp,
            'memory': self.budget.memory_atp,
            'tool': self.budget.tool_atp,
            'latency': self.budget.latency_budget,
            'risk': self.budget.risk_budget,
        }

        # Record operation
        operation = {
            'timestamp': turn_start_time.isoformat(),
            'speaker': speaker,
            'operational_mode': operational_mode.value,
            'priority': priority.name,
            'budget_start': budget_start,
            'attention_result': attention_result,
            'memory_result': memory_result,
            'deferrals': deferrals,
            'budget_end': budget_end,
        }
        self.operation_history.append(operation)

        logger.info(f"\n{'='*80}")
        logger.info(f"TURN COMPLETE")
        logger.info(f"Resources after: compute={self.budget.compute_atp:.1f}, "
                   f"memory={self.budget.memory_atp:.1f}, "
                   f"tool={self.budget.tool_atp:.1f}")
        logger.info(f"Deferrals: {deferrals if deferrals else 'None'}")
        logger.info(f"{'='*80}\n")

        return operation

    def _allocate_attention(
        self,
        salience_map: Dict[str, float],
        mode: OperationalMode,
        priority: ComponentPriority,
    ) -> Dict:
        """Allocate attention across targets."""
        num_targets = len(salience_map)
        max_salience = max(salience_map.values()) if salience_map else 0

        logger.info(f"\nAllocating attention for {num_targets} targets (max salience={max_salience:.2f})")

        # Define resource cost based on mode (RECALIBRATED 10x from Session 116)
        if mode == OperationalMode.NORMAL:
            cost = {
                'compute': 20.0 * num_targets,  # 10x increase: was 2.0
                'memory': 10.0 * num_targets,   # 10x increase: was 1.0
                'latency': 50.0 * num_targets,  # 10x increase: was 5.0
            }
            strategy = "full_attention"
        elif mode == OperationalMode.STRESSED:
            # Reduce quality in stressed mode
            cost = {
                'compute': 10.0 * num_targets,  # 10x increase: was 1.0
                'memory': 5.0 * num_targets,    # 10x increase: was 0.5
                'latency': 30.0 * num_targets,  # 10x increase: was 3.0
            }
            strategy = "degraded_attention"
        else:  # CRISIS or SLEEP
            if priority.value >= ComponentPriority.HIGH.value:
                # Only high-priority attention in crisis
                cost = {
                    'compute': 5.0 * num_targets,   # 10x increase: was 0.5
                    'memory': 2.0 * num_targets,    # 10x increase: was 0.2
                    'latency': 20.0 * num_targets,  # 10x increase: was 2.0
                }
                strategy = "minimal_attention"
            else:
                # Defer low-priority attention
                logger.info(f"  ✗ Attention deferred (CRISIS mode, priority={priority.name})")
                return {'allocated': False, 'deferred': True, 'strategy': 'deferred'}

        # Check if we can afford it
        affordable, _ = self.budget.can_afford(cost)
        if affordable:
            self.budget.consume(cost)
            logger.info(f"  Strategy: {strategy}")
            logger.info(f"  Cost: compute={cost['compute']:.1f}, memory={cost['memory']:.1f}")
            logger.info(f"  ✓ Attention allocated to {num_targets} targets")
            return {
                'allocated': True,
                'deferred': False,
                'strategy': strategy,
                'targets': num_targets,
                'cost': {'compute': cost['compute'], 'memory': cost['memory']}
            }
        else:
            # Cannot afford - defer
            logger.info(f"  ✗ Attention deferred (insufficient resources)")
            logger.info(f"  Required: compute={cost['compute']:.1f}, memory={cost['memory']:.1f}")
            logger.info(f"  Available: compute={self.budget.compute_atp:.1f}, memory={self.budget.memory_atp:.1f}")
            return {'allocated': False, 'deferred': True, 'strategy': 'deferred'}

    def _encode_memory(
        self,
        speaker: str,
        text: str,
        mode: OperationalMode,
        priority: ComponentPriority,
    ) -> Dict:
        """Encode memory for this turn."""
        text_length = len(text)

        logger.info(f"\nEncoding memory: {speaker}... ({text_length} chars)")

        # Define resource cost based on mode (RECALIBRATED 10x from Session 116)
        if mode == OperationalMode.NORMAL:
            cost = {
                'compute': 10.0 + (text_length / 10.0),  # 10x increase: was 1.0 + /100
                'memory': 20.0 + (text_length / 5.0),    # 10x increase: was 2.0 + /50
            }
            strategy = "full_encoding"
            quality = 1.0
        elif mode == OperationalMode.STRESSED:
            # Simplified encoding in stressed mode
            cost = {
                'compute': 5.0 + (text_length / 20.0),  # 10x increase: was 0.5 + /200
                'memory': 10.0 + (text_length / 10.0),  # 10x increase: was 1.0 + /100
            }
            strategy = "simplified_encoding"
            quality = 0.6
        else:  # CRISIS or SLEEP
            if priority.value >= ComponentPriority.NORMAL.value:
                # Minimal encoding for normal+ priority
                cost = {
                    'compute': 2.0,  # 10x increase: was 0.2
                    'memory': 5.0,   # 10x increase: was 0.5
                }
                strategy = "minimal_encoding"
                quality = 0.3
            else:
                # Defer low-priority memory
                logger.info(f"  ✗ Memory deferred (CRISIS mode, priority={priority.name})")
                return {'encoded': False, 'deferred': True, 'strategy': 'deferred'}

        # Check if we can afford it
        affordable, _ = self.budget.can_afford(cost)
        if affordable:
            self.budget.consume(cost)
            self.memory_buffer.append({
                'speaker': speaker,
                'text': text,
                'quality': quality,
                'strategy': strategy,
            })
            logger.info(f"  Strategy: {strategy}")
            logger.info(f"  Quality: {quality:.1f}")
            logger.info(f"  Cost: compute={cost['compute']:.1f}, memory={cost['memory']:.1f}")
            logger.info(f"  ✓ Memory encoded (buffer size={len(self.memory_buffer)})")
            return {
                'encoded': True,
                'deferred': False,
                'strategy': strategy,
                'quality': quality,
                'cost': {'compute': cost['compute'], 'memory': cost['memory']}
            }
        else:
            # Cannot afford - defer
            logger.info(f"  ✗ Memory deferred (insufficient resources)")
            logger.info(f"  Required: compute={cost['compute']:.1f}, memory={cost['memory']:.1f}")
            logger.info(f"  Available: compute={self.budget.compute_atp:.1f}, memory={self.budget.memory_atp:.1f}")
            return {'encoded': False, 'deferred': True, 'strategy': 'deferred'}

    def get_stats(self) -> Dict:
        """Get comprehensive statistics."""
        mode_distribution = {}
        for op in self.operation_history:
            mode = op['operational_mode']
            mode_distribution[mode] = mode_distribution.get(mode, 0) + 1

        strategy_distribution = {
            'attention': {},
            'memory': {},
        }

        for op in self.operation_history:
            if op['attention_result']:
                strategy = op['attention_result']['strategy']
                strategy_distribution['attention'][strategy] = \
                    strategy_distribution['attention'].get(strategy, 0) + 1
            if op['memory_result']:
                strategy = op['memory_result']['strategy']
                strategy_distribution['memory'][strategy] = \
                    strategy_distribution['memory'].get(strategy, 0) + 1

        return {
            'total_turns': len(self.operation_history),
            'resource_conflicts': self.resource_conflicts,
            'deferrals_by_component': self.deferrals_by_component,
            'operational_mode_distribution': mode_distribution,
            'strategy_distribution': strategy_distribution,
            'final_budget': {
                'compute': self.budget.compute_atp,
                'memory': self.budget.memory_atp,
                'tool': self.budget.tool_atp,
                'latency': self.budget.latency_budget,
                'risk': self.budget.risk_budget,
            },
            'memory_buffer_size': len(self.memory_buffer),
        }


def run_session_117():
    """Run Session 117 recalibrated stress testing experiments."""

    logger.info("="*80)
    logger.info("SESSION 117: RECALIBRATED STRESS TESTING")
    logger.info("="*80)
    logger.info("Goal: Discover emergent behaviors with 10x recalibrated costs")
    logger.info("Costs: Attention 20-120 ATP, Memory 10-50 ATP (was 2-12, 1-5 ATP)")
    logger.info("\n")

    # Define stress test scenarios
    scenarios = [
        StressTestScenario(
            name="A. Heavy Attention Load",
            description="Many high-salience targets drain compute rapidly",
            turns=[
                {
                    'speaker': 'User',
                    'text': 'Multiple urgent issues need attention!',
                    'salience_map': {
                        'issue1': 0.95, 'issue2': 0.92, 'issue3': 0.90,
                        'issue4': 0.88, 'issue5': 0.85, 'issue6': 0.82,
                    },
                    'priority': ComponentPriority.HIGH,
                },
                {
                    'speaker': 'Assistant',
                    'text': 'Analyzing all issues...',
                    'salience_map': {
                        'analysis1': 0.80, 'analysis2': 0.78, 'analysis3': 0.75,
                        'analysis4': 0.72, 'analysis5': 0.70,
                    },
                    'priority': ComponentPriority.HIGH,
                },
                {
                    'speaker': 'User',
                    'text': 'More critical items!',
                    'salience_map': {
                        'critical1': 0.98, 'critical2': 0.96, 'critical3': 0.94,
                        'critical4': 0.92,
                    },
                    'priority': ComponentPriority.CRITICAL,
                },
            ],
        ),
        StressTestScenario(
            name="B. Heavy Memory Load",
            description="Many turns with large texts drain memory ATP",
            turns=[
                {'speaker': 'User', 'text': 'A' * 200, 'priority': ComponentPriority.NORMAL},
                {'speaker': 'Assistant', 'text': 'B' * 200, 'priority': ComponentPriority.NORMAL},
                {'speaker': 'User', 'text': 'C' * 200, 'priority': ComponentPriority.NORMAL},
                {'speaker': 'Assistant', 'text': 'D' * 200, 'priority': ComponentPriority.NORMAL},
                {'speaker': 'User', 'text': 'E' * 200, 'priority': ComponentPriority.NORMAL},
                {'speaker': 'Assistant', 'text': 'F' * 200, 'priority': ComponentPriority.NORMAL},
            ],
        ),
        StressTestScenario(
            name="C. Simultaneous Stress",
            description="Both attention and memory demanding simultaneously",
            turns=[
                {
                    'speaker': 'User',
                    'text': 'X' * 150,
                    'salience_map': {'t1': 0.9, 't2': 0.85, 't3': 0.8, 't4': 0.75},
                    'priority': ComponentPriority.HIGH,
                },
                {
                    'speaker': 'Assistant',
                    'text': 'Y' * 150,
                    'salience_map': {'t5': 0.88, 't6': 0.83, 't7': 0.78},
                    'priority': ComponentPriority.HIGH,
                },
                {
                    'speaker': 'User',
                    'text': 'Z' * 150,
                    'salience_map': {'t8': 0.92, 't9': 0.87, 't10': 0.82},
                    'priority': ComponentPriority.HIGH,
                },
                {
                    'speaker': 'Assistant',
                    'text': 'W' * 150,
                    'salience_map': {'t11': 0.85, 't12': 0.80},
                    'priority': ComponentPriority.NORMAL,
                },
            ],
        ),
        StressTestScenario(
            name="D. Resource Starvation",
            description="Continuous depletion without recovery (force CRISIS)",
            disable_recovery=True,
            turns=[
                {
                    'speaker': 'User',
                    'text': 'Turn 1',
                    'salience_map': {'t1': 0.8, 't2': 0.7},
                    'priority': ComponentPriority.CRITICAL,
                },
                {
                    'speaker': 'Assistant',
                    'text': 'Turn 2',
                    'salience_map': {'t3': 0.8, 't4': 0.7},
                    'priority': ComponentPriority.HIGH,
                },
                {
                    'speaker': 'User',
                    'text': 'Turn 3',
                    'salience_map': {'t5': 0.8, 't6': 0.7},
                    'priority': ComponentPriority.HIGH,
                },
                {
                    'speaker': 'Assistant',
                    'text': 'Turn 4',
                    'salience_map': {'t7': 0.8, 't8': 0.7},
                    'priority': ComponentPriority.NORMAL,
                },
                {
                    'speaker': 'User',
                    'text': 'Turn 5',
                    'salience_map': {'t9': 0.8, 't10': 0.7},
                    'priority': ComponentPriority.NORMAL,
                },
                {
                    'speaker': 'Assistant',
                    'text': 'Turn 6',
                    'salience_map': {'t11': 0.8, 't12': 0.7},
                    'priority': ComponentPriority.LOW,
                },
            ],
        ),
    ]

    # Run all scenarios
    all_results = {}

    for scenario in scenarios:
        logger.info("\n\n")
        logger.info("="*80)
        logger.info(f"SCENARIO: {scenario.name}")
        logger.info("="*80)
        logger.info(f"Description: {scenario.description}")
        logger.info(f"Turns: {len(scenario.turns)}")
        logger.info(f"Disable recovery: {scenario.disable_recovery}")
        logger.info("\n")

        # Initialize SAGE for this scenario
        sage = MultiResourceSAGE(initial_budget=scenario.initial_budget)

        # Process all turns
        for i, turn_spec in enumerate(scenario.turns, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"TURN {i}/{len(scenario.turns)}")
            logger.info(f"{'='*80}")

            sage.process_turn(
                speaker=turn_spec['speaker'],
                text=turn_spec['text'],
                salience_map=turn_spec.get('salience_map'),
                priority=turn_spec.get('priority', ComponentPriority.NORMAL),
                disable_recovery=scenario.disable_recovery,
            )

        # Get statistics
        stats = sage.get_stats()

        logger.info(f"\n\n{'='*80}")
        logger.info(f"SCENARIO COMPLETE: {scenario.name}")
        logger.info(f"{'='*80}")
        logger.info(f"\nStatistics:")
        logger.info(f"  Total turns: {stats['total_turns']}")
        logger.info(f"  Resource conflicts: {stats['resource_conflicts']}")
        logger.info(f"  Deferrals: attention={stats['deferrals_by_component']['attention']}, memory={stats['deferrals_by_component']['memory']}")
        logger.info(f"\nOperational Mode Distribution:")
        for mode, count in stats['operational_mode_distribution'].items():
            logger.info(f"  {mode}: {count} turns ({count/stats['total_turns']*100:.1f}%)")
        logger.info(f"\nFinal Resource Budget:")
        for resource, value in stats['final_budget'].items():
            logger.info(f"  {resource}: {value*100:.1f}%")
        logger.info(f"\nMemory Buffer: {stats['memory_buffer_size']} items")

        all_results[scenario.name] = {
            'description': scenario.description,
            'statistics': stats,
            'operation_history': sage.operation_history,
        }

    # Save all results
    output = {
        'session': 117,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'recalibration': '10x costs from Session 116',
        'scenarios': all_results,
    }

    output_file = 'sage/experiments/session117_recalibrated_stress_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n\nResults saved to: {output_file}")

    logger.info(f"\n\n{'='*80}")
    logger.info("SESSION 117 COMPLETE - RECALIBRATED STRESS TESTING DISCOVERIES!")
    logger.info(f"{'='*80}\n")

    return all_results


if __name__ == '__main__':
    results = run_session_117()
