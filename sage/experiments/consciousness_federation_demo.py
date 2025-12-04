#!/usr/bin/env python3
"""
Consciousness Federation Orchestration - Demonstration

Demonstrates SAGE consciousness kernel orchestrating federation protocol:
- Continuous monitoring of task queue, capacity, federation health
- Salience-driven attention allocation
- Stance-based delegation decisions
- Learning from outcomes

Builds on:
- Thor's consciousness kernel (continuous sense-assess-focus-act-learn loop)
- Session #54-55 federation protocol (Ed25519 crypto, ATP management)
- Session #57 Phase 2 unified memory (cross-system queries)

Author: Legion Autonomous Session #58
Date: 2025-12-04
"""

import sys
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add HRM to path
_hrm_root = Path(__file__).parent.parent.parent
if str(_hrm_root) not in sys.path:
    sys.path.insert(0, str(_hrm_root))

# Import only the minimal components we need (avoiding torch dependency)
from enum import Enum as _Enum

class CognitiveStance(_Enum):
    """Cognitive stances for consciousness (simplified for demo)"""
    CURIOUS_UNCERTAINTY = "curious_uncertainty"
    FOCUSED_ATTENTION = "focused_attention"
    SKEPTICAL_VERIFICATION = "skeptical_verification"
    CONFIDENT_EXECUTION = "confident_execution"
    EXPLORATORY = "exploratory"

class MetabolicState(_Enum):
    """Kernel operational states"""
    WAKE = "wake"
    FOCUS = "focus"
    REST = "rest"
    DREAM = "dream"
    CRISIS = "crisis"

@dataclass
class ExecutionResult:
    """Result of executing action"""
    success: bool
    reward: float
    description: str
    outputs: Dict[str, Any]


# ============================================================================
# Simplified SAGE Kernel (No Torch Dependency)
# ============================================================================

class SimplifiedSAGEKernel:
    """
    Simplified SAGE Kernel for demonstration

    Implements core consciousness loop without full SNARC/torch dependencies:
    - Continuous sense-assess-focus-act-learn loop
    - Salience-based attention allocation (simplified)
    - Stance-based action selection
    - Learning from outcomes (simplified)
    """

    def __init__(
        self,
        sensor_sources: Dict[str, Any],
        action_handlers: Dict[str, Any],
        enable_logging: bool = True
    ):
        self.sensor_sources = sensor_sources
        self.action_handlers = action_handlers
        self.enable_logging = enable_logging

        # SNARC weights (simplified - no torch)
        self.snarc_weights = {
            'surprise': 0.25,
            'novelty': 0.20,
            'arousal': 0.30,
            'reward': 0.10,
            'conflict': 0.15
        }

        # State
        self.cycle_count = 0
        self.running = False
        self.metabolic_state = MetabolicState.WAKE

        # History
        self.execution_history: List[Dict] = []
        self.salience_history: List[float] = []
        self.stance_history: List[CognitiveStance] = []

    def run(self, max_cycles: Optional[int] = None, cycle_delay: float = 0.1):
        """Main execution loop"""
        self.running = True
        self.cycle_count = 0

        try:
            while self.running:
                if max_cycles is not None and self.cycle_count >= max_cycles:
                    break

                self._cycle()
                time.sleep(cycle_delay)
                self.cycle_count += 1

        except KeyboardInterrupt:
            if self.enable_logging:
                print("\n[SAGE] Interrupted by user")
        finally:
            self.running = False
            if self.enable_logging:
                print(f"\n[SAGE] Completed {self.cycle_count} cycles")
                self._print_statistics()

    def _cycle(self):
        """Execute one consciousness cycle"""
        # 1. SENSE: Gather observations
        observations = {}
        for sensor_id, sensor_fn in self.sensor_sources.items():
            try:
                observations[sensor_id] = sensor_fn()
            except Exception as e:
                if self.enable_logging:
                    print(f"[SAGE] Sensor {sensor_id} error: {e}")

        if not observations:
            return

        # 2. ASSESS: Calculate salience (simplified)
        salience_scores = {}
        for sensor_id, data in observations.items():
            salience_scores[sensor_id] = self._calculate_salience(data)

        # 3. FOCUS: Select highest-salience sensor
        focus_target = max(salience_scores, key=salience_scores.get)
        focus_salience = salience_scores[focus_target]
        self.salience_history.append(focus_salience)

        # 4. DECIDE: Determine cognitive stance
        stance = self._determine_stance(observations[focus_target], focus_salience)
        self.stance_history.append(stance)

        if self.enable_logging:
            print(f"[Cycle {self.cycle_count}] Focus: {focus_target} "
                  f"(salience={focus_salience:.3f}, stance={stance.value})")

        # 5. ACT: Execute action
        if focus_target in self.action_handlers:
            try:
                action_fn = self.action_handlers[focus_target]
                result = action_fn(observations[focus_target], stance)

                # 6. LEARN: Update from outcome
                self._learn_from_outcome(result, focus_target, stance)

                if self.enable_logging and result.description:
                    print(f"  → {result.description}")

                self.execution_history.append({
                    'cycle': self.cycle_count,
                    'focus': focus_target,
                    'stance': stance.value,
                    'reward': result.reward,
                    'success': result.success
                })

            except Exception as e:
                if self.enable_logging:
                    print(f"  → Action failed: {e}")

    def _calculate_salience(self, sensor_data: Dict) -> float:
        """Calculate salience for sensor data (simplified)"""
        if not sensor_data:
            return 0.0

        # Simple heuristics for demonstration
        salience = 0.5  # Base

        # Arousal from urgency/volume
        if 'urgent_count' in sensor_data and sensor_data['urgent_count'] > 0:
            salience += 0.3

        if 'count' in sensor_data and sensor_data['count'] > 10:
            salience += 0.2

        # Novelty from new information
        if 'novelty_score' in sensor_data:
            salience += sensor_data['novelty_score'] * 0.2

        # Surprise from unexpected values
        if 'atp_available' in sensor_data and sensor_data['atp_available'] < 100:
            salience += 0.3

        if 'platforms_available' in sensor_data:
            total = sensor_data.get('platforms_total', 3)
            available = sensor_data['platforms_available']
            if available < total:
                salience += 0.2

        return min(1.0, salience)

    def _determine_stance(self, sensor_data: Dict, salience: float) -> CognitiveStance:
        """Determine cognitive stance based on sensor data and salience"""

        # High salience + urgency = FOCUSED_ATTENTION
        if salience > 0.8 or sensor_data.get('urgent_count', 0) > 0:
            return CognitiveStance.FOCUSED_ATTENTION

        # Quality issues = SKEPTICAL_VERIFICATION
        if 'delegation_success_rate' in sensor_data:
            if sensor_data['delegation_success_rate'] < 0.7:
                return CognitiveStance.SKEPTICAL_VERIFICATION

        # Resource constraints = FOCUSED_ATTENTION
        if 'atp_available' in sensor_data and sensor_data['atp_available'] < 150:
            return CognitiveStance.FOCUSED_ATTENTION

        # New/novel situations = CURIOUS_UNCERTAINTY
        if 'novelty_score' in sensor_data and sensor_data['novelty_score'] > 0.7:
            return CognitiveStance.CURIOUS_UNCERTAINTY

        # Default = CONFIDENT_EXECUTION
        return CognitiveStance.CONFIDENT_EXECUTION

    def _learn_from_outcome(self, result: ExecutionResult, focus: str, stance: CognitiveStance):
        """Update SNARC weights based on outcome (simplified)"""
        # Simple learning: adjust weights slightly based on reward
        if result.reward > 0.7:
            # Positive outcome: slightly increase arousal weight (focused attention worked)
            if stance == CognitiveStance.FOCUSED_ATTENTION:
                self.snarc_weights['arousal'] = min(0.4, self.snarc_weights['arousal'] * 1.02)
        elif result.reward < 0.3:
            # Negative outcome: increase conflict weight (skepticism needed)
            self.snarc_weights['conflict'] = min(0.25, self.snarc_weights['conflict'] * 1.05)

        # Normalize weights to sum to 1.0
        total = sum(self.snarc_weights.values())
        for k in self.snarc_weights:
            self.snarc_weights[k] /= total

    def _print_statistics(self):
        """Print final statistics"""
        if not self.execution_history:
            return

        print("\nStatistics:")
        print(f"  Total cycles: {self.cycle_count}")
        print(f"  Avg salience: {sum(self.salience_history)/len(self.salience_history):.3f}"
              if self.salience_history else "  Avg salience: N/A")

        # Stance distribution
        stance_counts = {}
        for stance in self.stance_history:
            stance_counts[stance.value] = stance_counts.get(stance.value, 0) + 1

        print("  Stance distribution:")
        for stance, count in sorted(stance_counts.items(), key=lambda x: -x[1]):
            pct = count / len(self.stance_history) * 100
            print(f"    {stance}: {count} ({pct:.1f}%)")

        # Learning
        print(f"  Final SNARC weights:")
        for dim, weight in sorted(self.snarc_weights.items(), key=lambda x: -x[1]):
            print(f"    {dim}: {weight:.3f}")


# ============================================================================
# Simulation Data Structures
# ============================================================================

@dataclass
class SimulatedTask:
    """Simulated task for demonstration"""
    id: str
    task_type: str
    priority: float  # 0.0-1.0
    estimated_cost: float  # ATP units
    created_at: float
    description: str


@dataclass
class SimulatedPlatform:
    """Simulated federation platform"""
    name: str
    is_online: bool
    trust_score: float  # 0.0-1.0
    recent_quality: float  # 0.0-1.0
    avg_response_time: float  # seconds


class SimulatedEnvironment:
    """
    Simulated federation environment for demonstration

    Generates realistic patterns:
    - Task arrivals (occasional spikes)
    - ATP consumption/regeneration
    - Platform availability (occasional downtime)
    - Quality variations
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.time = 0.0

        # State
        self.atp_available = 300.0  # Start with moderate ATP
        self.atp_reserved = 0.0
        self.acceptance_rate = 1.0  # Accept all tasks initially

        # Task queue
        self.pending_tasks: List[SimulatedTask] = []
        self.task_counter = 0

        # Platforms
        self.platforms = {
            'Legion': SimulatedPlatform(
                name='Legion',
                is_online=True,
                trust_score=0.85,
                recent_quality=0.92,
                avg_response_time=0.15
            ),
            'Sprout': SimulatedPlatform(
                name='Sprout',
                is_online=True,
                trust_score=0.70,
                recent_quality=0.85,
                avg_response_time=0.25
            ),
            'Thor': SimulatedPlatform(
                name='Thor',
                is_online=True,
                trust_score=0.65,
                recent_quality=0.80,
                avg_response_time=0.30
            )
        }

        # Execution history
        self.execution_history = []

        # Metrics
        self.tasks_created = 0
        self.tasks_executed_local = 0
        self.tasks_delegated = 0
        self.delegations_by_platform = {p: 0 for p in self.platforms}

    def step(self, dt: float = 0.5):
        """Advance simulation by dt seconds"""
        self.time += dt

        # Generate new tasks (with occasional spikes)
        if random.random() < 0.3:  # 30% chance of task arrival
            self._generate_task()

        # Occasional task spike
        if random.random() < 0.05:  # 5% chance of spike
            for _ in range(random.randint(5, 15)):
                self._generate_task()

        # ATP regeneration
        self.atp_available += dt * 10  # Regenerate 10 ATP/sec
        self.atp_available = min(500.0, self.atp_available)  # Cap at 500

        # Platform dynamics
        for platform in self.platforms.values():
            # Occasional quality fluctuation
            if random.random() < 0.1:
                platform.recent_quality += random.uniform(-0.1, 0.1)
                platform.recent_quality = max(0.5, min(1.0, platform.recent_quality))

            # Rare downtime
            if platform.is_online and random.random() < 0.01:  # 1% chance
                platform.is_online = False
                print(f"  [SIMULATION] {platform.name} went OFFLINE")

            # Recovery from downtime
            if not platform.is_online and random.random() < 0.2:  # 20% chance
                platform.is_online = True
                print(f"  [SIMULATION] {platform.name} came ONLINE")

    def _generate_task(self):
        """Generate a new task"""
        task_types = ['reasoning', 'data_processing', 'simulation', 'validation']
        task = SimulatedTask(
            id=f"task_{self.task_counter}",
            task_type=random.choice(task_types),
            priority=random.uniform(0.3, 1.0),
            estimated_cost=random.uniform(50, 300),
            created_at=self.time,
            description=f"Simulated {random.choice(task_types)} task"
        )

        # Only accept if acceptance rate allows
        if random.random() < self.acceptance_rate:
            self.pending_tasks.append(task)
            self.tasks_created += 1
            self.task_counter += 1

    def execute_local(self, task: SimulatedTask) -> bool:
        """Execute task locally"""
        if self.atp_available >= task.estimated_cost:
            self.atp_available -= task.estimated_cost
            self.tasks_executed_local += 1
            self.pending_tasks.remove(task)
            return True
        return False

    def delegate_task(self, task: SimulatedTask, platform_name: str) -> bool:
        """Delegate task to platform"""
        platform = self.platforms.get(platform_name)
        if not platform or not platform.is_online:
            return False

        # Delegation succeeds based on platform quality
        success = random.random() < platform.recent_quality

        if success:
            # Consume some ATP for delegation overhead
            delegation_cost = task.estimated_cost * 0.2  # 20% overhead
            if self.atp_available >= delegation_cost:
                self.atp_available -= delegation_cost
                self.tasks_delegated += 1
                self.delegations_by_platform[platform_name] += 1
                self.pending_tasks.remove(task)
                return True

        return False

    def update_platform_trust(self, platform_name: str, new_trust: float):
        """Update platform trust score"""
        if platform_name in self.platforms:
            self.platforms[platform_name].trust_score = new_trust


# ============================================================================
# Federation Sensors
# ============================================================================

def create_task_queue_sensor(env: SimulatedEnvironment):
    """Create sensor for task queue monitoring"""

    def sense_task_queue() -> Dict[str, Any]:
        tasks = env.pending_tasks

        return {
            'count': len(tasks),
            'urgency': max([t.priority for t in tasks], default=0.0),
            'estimated_load': sum(t.estimated_cost for t in tasks),
            'oldest_task_age': env.time - min([t.created_at for t in tasks], default=env.time),
            'task_types': [t.task_type for t in tasks],
            'urgent_count': len([t for t in tasks if t.priority > 0.8])
        }

    return sense_task_queue


def create_local_capacity_sensor(env: SimulatedEnvironment):
    """Create sensor for local capacity monitoring"""

    def sense_local_capacity() -> Dict[str, Any]:
        return {
            'atp_available': env.atp_available,
            'atp_reserved': env.atp_reserved,
            'atp_utilization': 1.0 - (env.atp_available / 500.0),  # Fraction used
            'projected_capacity_1h': env.atp_available + (3600 * 10),  # Regeneration projection
        }

    return sense_local_capacity


def create_federation_health_sensor(env: SimulatedEnvironment):
    """Create sensor for federation health monitoring"""

    def sense_federation_health() -> Dict[str, Any]:
        platforms = env.platforms

        return {
            'platforms_available': len([p for p in platforms.values() if p.is_online]),
            'platforms_total': len(platforms),
            'avg_response_time': sum(p.avg_response_time for p in platforms.values()) / len(platforms),
            'trust_scores': {name: p.trust_score for name, p in platforms.items()},
            'platforms_online': [name for name, p in platforms.items() if p.is_online]
        }

    return sense_federation_health


def create_execution_quality_sensor(env: SimulatedEnvironment):
    """Create sensor for execution quality monitoring"""

    def sense_execution_quality() -> Dict[str, Any]:
        # Calculate success rates from recent history
        recent = env.execution_history[-20:] if env.execution_history else []

        local_executions = [e for e in recent if e['location'] == 'local']
        delegated_executions = [e for e in recent if e['location'] == 'delegated']

        local_success_rate = (
            sum(1 for e in local_executions if e['success']) / len(local_executions)
            if local_executions else 1.0
        )

        delegation_success_rate = (
            sum(1 for e in delegated_executions if e['success']) / len(delegated_executions)
            if delegated_executions else 0.9
        )

        return {
            'local_success_rate': local_success_rate,
            'delegation_success_rate': delegation_success_rate,
            'total_executed': env.tasks_executed_local + env.tasks_delegated,
            'delegation_ratio': (
                env.tasks_delegated / (env.tasks_executed_local + env.tasks_delegated)
                if (env.tasks_executed_local + env.tasks_delegated) > 0 else 0.0
            )
        }

    return sense_execution_quality


# ============================================================================
# Federation Actions
# ============================================================================

def create_decide_local_or_delegate_action(env: SimulatedEnvironment):
    """Create action for deciding local vs delegation"""

    def action_decide_local_or_delegate(
        task_queue_data: Dict,
        stance: CognitiveStance
    ) -> ExecutionResult:
        tasks = env.pending_tasks[:10]  # Process up to 10 tasks per cycle

        if not tasks:
            return ExecutionResult(
                success=True,
                reward=0.0,
                description="No tasks to process",
                outputs={}
            )

        executed_local = 0
        delegated = 0
        failed = 0

        if stance == CognitiveStance.FOCUSED_ATTENTION:
            # High arousal: Handle urgent tasks immediately
            urgent = [t for t in tasks if t.priority > 0.8]

            for task in urgent:
                if env.atp_available >= task.estimated_cost:
                    if env.execute_local(task):
                        executed_local += 1
                        env.execution_history.append({
                            'task_id': task.id,
                            'location': 'local',
                            'success': True,
                            'stance': stance.value
                        })
                else:
                    # Delegate to most reliable online platform
                    online_platforms = [p for p in env.platforms.values() if p.is_online]
                    if online_platforms:
                        best_platform = max(online_platforms, key=lambda p: p.trust_score)
                        if env.delegate_task(task, best_platform.name):
                            delegated += 1
                            env.execution_history.append({
                                'task_id': task.id,
                                'location': 'delegated',
                                'platform': best_platform.name,
                                'success': True,
                                'stance': stance.value
                            })
                        else:
                            failed += 1

            return ExecutionResult(
                success=True,
                reward=0.7 if failed == 0 else 0.3,
                description=f"Handled {len(urgent)} urgent tasks (focused)",
                outputs={'local': executed_local, 'delegated': delegated, 'failed': failed}
            )

        elif stance == CognitiveStance.CURIOUS_UNCERTAINTY:
            # Novelty: Explore delegation to less-used platforms
            if tasks:
                task = tasks[0]

                # Find least-utilized online platform
                online_platforms = [p for p in env.platforms.values() if p.is_online]
                if online_platforms:
                    least_used = min(
                        online_platforms,
                        key=lambda p: env.delegations_by_platform.get(p.name, 0)
                    )

                    if env.delegate_task(task, least_used.name):
                        delegated += 1
                        env.execution_history.append({
                            'task_id': task.id,
                            'location': 'delegated',
                            'platform': least_used.name,
                            'success': True,
                            'stance': stance.value
                        })

                        return ExecutionResult(
                            success=True,
                            reward=0.5,  # Learning reward
                            description=f"Explored delegation to {least_used.name} (curious)",
                            outputs={'explored_platform': least_used.name, 'delegated': 1}
                        )

        elif stance == CognitiveStance.SKEPTICAL_VERIFICATION:
            # Conflict: Verify quality before acting
            quality = sum(p.recent_quality for p in env.platforms.values()) / len(env.platforms)

            if quality < 0.7:
                # Quality concerns: execute locally if possible
                for task in tasks[:5]:
                    if env.atp_available >= task.estimated_cost:
                        if env.execute_local(task):
                            executed_local += 1
                            env.execution_history.append({
                                'task_id': task.id,
                                'location': 'local',
                                'success': True,
                                'stance': stance.value
                            })

                return ExecutionResult(
                    success=True,
                    reward=0.4,
                    description=f"Quality concerns: executed {executed_local} locally (skeptical)",
                    outputs={'local': executed_local}
                )

        # Default: CONFIDENT_EXECUTION or EXPLORATORY
        for task in tasks:
            if env.atp_available >= task.estimated_cost and task.estimated_cost < 100:
                # Cheap tasks: execute locally
                if env.execute_local(task):
                    executed_local += 1
                    env.execution_history.append({
                        'task_id': task.id,
                        'location': 'local',
                        'success': True,
                        'stance': stance.value
                    })
            else:
                # Expensive tasks: delegate to best available platform
                online_platforms = [p for p in env.platforms.values() if p.is_online]
                if online_platforms:
                    best = max(online_platforms, key=lambda p: p.trust_score * p.recent_quality)
                    if env.delegate_task(task, best.name):
                        delegated += 1
                        env.execution_history.append({
                            'task_id': task.id,
                            'location': 'delegated',
                            'platform': best.name,
                            'success': True,
                            'stance': stance.value
                        })

        return ExecutionResult(
            success=True,
            reward=0.6,
            description=f"Processed {executed_local + delegated} tasks (routine)",
            outputs={'local': executed_local, 'delegated': delegated}
        )

    return action_decide_local_or_delegate


def create_adjust_acceptance_rate_action(env: SimulatedEnvironment):
    """Create action for adjusting task acceptance rate"""

    def action_adjust_acceptance_rate(
        capacity_data: Dict,
        stance: CognitiveStance
    ) -> ExecutionResult:
        atp_available = capacity_data['atp_available']
        atp_util = capacity_data['atp_utilization']

        old_rate = env.acceptance_rate

        if stance == CognitiveStance.FOCUSED_ATTENTION:
            # High arousal (low ATP): Aggressive throttling
            if atp_available < 100:
                env.acceptance_rate = 0.0  # Stop accepting
                return ExecutionResult(
                    success=True,
                    reward=0.9,
                    description=f"ATP critical: stopped accepting (was {old_rate:.2f})",
                    outputs={'old_rate': old_rate, 'new_rate': 0.0}
                )

        elif stance == CognitiveStance.CONFIDENT_EXECUTION:
            # Abundant resources: Accept freely
            if atp_available > 300:
                env.acceptance_rate = 1.0
                return ExecutionResult(
                    success=True,
                    reward=0.7,
                    description=f"ATP abundant: accepting all (was {old_rate:.2f})",
                    outputs={'old_rate': old_rate, 'new_rate': 1.0}
                )

        # Default: Proportional to available ATP
        new_rate = min(1.0, atp_available / 300.0)  # 300 ATP = full acceptance
        env.acceptance_rate = new_rate

        return ExecutionResult(
            success=True,
            reward=0.5,
            description=f"Adjusted acceptance: {old_rate:.2f} → {new_rate:.2f}",
            outputs={'old_rate': old_rate, 'new_rate': new_rate}
        )

    return action_adjust_acceptance_rate


def create_update_trust_scores_action(env: SimulatedEnvironment):
    """Create action for updating platform trust scores"""

    def action_update_trust_scores(
        federation_data: Dict,
        stance: CognitiveStance
    ) -> ExecutionResult:
        updates = {}

        for name, platform in env.platforms.items():
            current_trust = platform.trust_score
            recent_quality = platform.recent_quality

            if stance == CognitiveStance.SKEPTICAL_VERIFICATION:
                # Aggressive trust decay on quality issues
                if recent_quality < 0.7:
                    new_trust = current_trust * 0.85
                    env.update_platform_trust(name, new_trust)
                    updates[name] = f"{current_trust:.2f} → {new_trust:.2f} (decay)"

            elif stance == CognitiveStance.CURIOUS_UNCERTAINTY:
                # Exploratory trust building
                if recent_quality > 0.8:
                    new_trust = min(1.0, current_trust + 0.05)
                    env.update_platform_trust(name, new_trust)
                    updates[name] = f"{current_trust:.2f} → {new_trust:.2f} (build)"

            else:
                # Routine: exponential moving average
                alpha = 0.1
                new_trust = alpha * recent_quality + (1 - alpha) * current_trust
                env.update_platform_trust(name, new_trust)
                updates[name] = f"{current_trust:.2f} → {new_trust:.2f} (EMA)"

        return ExecutionResult(
            success=True,
            reward=0.5,
            description=f"Updated trust for {len(updates)} platforms",
            outputs={'updates': updates}
        )

    return action_update_trust_scores


# ============================================================================
# Main Demonstration
# ============================================================================

def run_demonstration(num_cycles: int = 40, cycle_delay: float = 0.5):
    """
    Run consciousness federation orchestration demonstration

    Args:
        num_cycles: Number of consciousness cycles to run
        cycle_delay: Delay between cycles (simulated time step)
    """
    print("="*80)
    print("CONSCIOUSNESS FEDERATION ORCHESTRATION - DEMONSTRATION")
    print("="*80)
    print()
    print("Demonstrating SAGE consciousness kernel orchestrating federation:")
    print("- Continuous monitoring of task queue, capacity, federation health")
    print("- Salience-driven attention allocation")
    print("- Stance-based delegation decisions")
    print("- Learning from outcomes")
    print()
    print(f"Running {num_cycles} cycles with {cycle_delay}s delay...")
    print()

    # Initialize simulation environment
    env = SimulatedEnvironment(seed=42)

    # Create sensors
    sensors = {
        'task_queue': create_task_queue_sensor(env),
        'local_capacity': create_local_capacity_sensor(env),
        'federation_health': create_federation_health_sensor(env),
        'execution_quality': create_execution_quality_sensor(env)
    }

    # Create actions
    actions = {
        'task_queue': create_decide_local_or_delegate_action(env),
        'local_capacity': create_adjust_acceptance_rate_action(env),
        'federation_health': create_update_trust_scores_action(env)
    }

    # Initialize consciousness kernel
    kernel = SimplifiedSAGEKernel(
        sensor_sources=sensors,
        action_handlers=actions,
        enable_logging=True
    )

    # Run consciousness loop
    print("="*80)
    print("STARTING CONSCIOUSNESS LOOP")
    print("="*80)
    print()

    start_time = time.time()

    for cycle in range(num_cycles):
        # Step simulation environment
        env.step(dt=cycle_delay)

        # Run one consciousness cycle (kernel handles this internally)
        # We just need to call kernel.run() with max_cycles

    # Run kernel
    kernel.run(max_cycles=num_cycles, cycle_delay=cycle_delay)

    elapsed = time.time() - start_time

    # Print results
    print()
    print("="*80)
    print("DEMONSTRATION RESULTS")
    print("="*80)
    print()
    print(f"Runtime: {elapsed:.2f}s")
    print(f"Avg cycle time: {(elapsed/num_cycles)*1000:.0f}ms")
    print()

    print("Task Processing:")
    print(f"  Tasks created: {env.tasks_created}")
    print(f"  Executed locally: {env.tasks_executed_local}")
    print(f"  Delegated: {env.tasks_delegated}")
    print(f"  Still pending: {len(env.pending_tasks)}")
    print(f"  Delegation ratio: {env.tasks_delegated/(env.tasks_executed_local + env.tasks_delegated)*100:.1f}%"
          if (env.tasks_executed_local + env.tasks_delegated) > 0 else "  Delegation ratio: N/A")
    print()

    print("Delegations by Platform:")
    for platform, count in env.delegations_by_platform.items():
        print(f"  {platform}: {count}")
    print()

    print("Final Platform Trust Scores:")
    for name, platform in env.platforms.items():
        status = "ONLINE" if platform.is_online else "OFFLINE"
        print(f"  {name}: {platform.trust_score:.3f} ({status}, quality={platform.recent_quality:.2f})")
    print()

    print("Final ATP Status:")
    print(f"  Available: {env.atp_available:.1f}")
    print(f"  Acceptance rate: {env.acceptance_rate:.2%}")
    print()

    print("Execution History Sample (last 10):")
    for entry in env.execution_history[-10:]:
        location = entry['location']
        platform = entry.get('platform', 'N/A')
        stance = entry.get('stance', 'unknown')
        print(f"  {entry['task_id']}: {location} (platform={platform}, stance={stance})")
    print()

    # SNARC learning results
    print("SNARC Learning:")
    print(f"  Cycles completed: {kernel.cycle_count}")
    print(f"  Final SNARC weights: {kernel.snarc_weights}")
    print()

    print("="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    run_demonstration(num_cycles=40, cycle_delay=0.5)
