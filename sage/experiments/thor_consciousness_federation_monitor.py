#!/usr/bin/env python3
"""
Thor Consciousness Federation Monitor

Demonstrates consciousness kernel managing federation on Thor platform:
- Real system monitoring (CPU, memory, disk - simulating local capacity)
- Federation health tracking (simulating remote platform availability)
- Task queue simulation (representing work to delegate vs execute locally)
- Stance-based decision making (when to delegate, when to execute locally)
- Metabolic state integration (FOCUS/WAKE/REST/DREAM affecting federation behavior)

Extends Thor's consciousness kernel architecture with federation orchestration,
building on Legion Session #58's theoretical framework.

Key Innovation:
Instead of "should I delegate this task?" (reactive API calls),
we have "what needs my attention?" (continuous consciousness managing resources).

Author: Thor Autonomous Session
Date: 2025-12-04
Hardware: Jetson AGX Thor
"""

import sys
import time
import random
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque

# Add HRM to path
_hrm_root = Path(__file__).parent.parent.parent
if str(_hrm_root) not in sys.path:
    sys.path.insert(0, str(_hrm_root))


# ============================================================================
# Core Types
# ============================================================================

class CognitiveStance(Enum):
    """Cognitive stances for consciousness"""
    CURIOUS_UNCERTAINTY = "curious-uncertainty"
    FOCUSED_ATTENTION = "focused-attention"
    SKEPTICAL_VERIFICATION = "skeptical-verification"
    CONFIDENT_EXECUTION = "confident-execution"


class MetabolicState(Enum):
    """Kernel operational states"""
    WAKE = "wake"
    FOCUS = "focus"
    REST = "rest"
    DREAM = "dream"


@dataclass
class ExecutionResult:
    """Result of executing action"""
    success: bool
    reward: float
    description: str
    outputs: Dict[str, Any]


@dataclass
class SimulatedTask:
    """Simulated federation task"""
    id: str
    task_type: str
    priority: float  # 0.0-1.0
    estimated_cost: int  # ATP cost
    created_at: float  # timestamp


@dataclass
class FederationPlatform:
    """Simulated remote platform"""
    name: str
    is_online: bool
    trust_score: float  # 0.0-1.0
    recent_quality: float  # 0.0-1.0
    avg_response_time: float  # seconds


# ============================================================================
# Federation Environment (Simulated)
# ============================================================================

class FederationEnvironment:
    """
    Simulates federation environment for demonstration

    In production, this would interface with:
    - Real Legion/Sprout platform connections
    - Actual Ed25519 crypto and ATP lock protocol
    - Real task queue from user requests
    - Persistent memory database for quality history
    """

    def __init__(self):
        # ATP budget (Attention-Time-Points)
        self.atp_available = 500
        self.atp_max = 500
        self.atp_regen_rate = 10  # per cycle

        # Task queue
        self.pending_tasks: List[SimulatedTask] = []
        self.task_counter = 0

        # Federation platforms
        self.platforms = {
            'Legion': FederationPlatform(
                name='Legion',
                is_online=True,
                trust_score=0.9,
                recent_quality=0.95,
                avg_response_time=0.5
            ),
            'Sprout': FederationPlatform(
                name='Sprout',
                is_online=True,
                trust_score=0.8,
                recent_quality=0.85,
                avg_response_time=0.2
            ),
            'Platform2': FederationPlatform(
                name='Platform2',
                is_online=False,  # Offline for now
                trust_score=0.7,
                recent_quality=0.75,
                avg_response_time=1.0
            )
        }

        # Statistics
        self.tasks_created = 0
        self.tasks_executed_local = 0
        self.tasks_delegated = 0
        self.delegations_by_platform = {p: 0 for p in self.platforms.keys()}
        self.execution_history: List[Dict] = []

        # Simulated time
        self.time = 0.0

    def step(self):
        """Advance simulation one step"""
        self.time += 1.0

        # Regenerate ATP
        self.atp_available = min(self.atp_max, self.atp_available + self.atp_regen_rate)

        # Randomly generate new tasks
        if random.random() < 0.3:  # 30% chance per cycle
            self.create_task()

        # Simulate platform status changes
        if random.random() < 0.05:  # 5% chance
            platform = random.choice(list(self.platforms.values()))
            platform.is_online = not platform.is_online

    def create_task(self):
        """Create a new simulated task"""
        task_types = ['inference', 'embedding', 'search', 'analysis']

        task = SimulatedTask(
            id=f"task_{self.task_counter}",
            task_type=random.choice(task_types),
            priority=random.uniform(0.3, 1.0),
            estimated_cost=random.randint(50, 300),
            created_at=self.time
        )

        self.pending_tasks.append(task)
        self.tasks_created += 1
        self.task_counter += 1

    def execute_local(self, task: SimulatedTask) -> bool:
        """Execute task locally"""
        if self.atp_available >= task.estimated_cost:
            self.atp_available -= task.estimated_cost
            self.tasks_executed_local += 1
            if task in self.pending_tasks:
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
            delegation_cost = int(task.estimated_cost * 0.2)  # 20% overhead
            if self.atp_available >= delegation_cost:
                self.atp_available -= delegation_cost
                self.tasks_delegated += 1
                self.delegations_by_platform[platform_name] += 1
                if task in self.pending_tasks:
                    self.pending_tasks.remove(task)
                return True

        return False


# ============================================================================
# Federation Sensors
# ============================================================================

def create_task_queue_sensor(env: FederationEnvironment):
    """Monitor task queue"""

    def sense_task_queue() -> Dict[str, Any]:
        tasks = env.pending_tasks

        return {
            'count': len(tasks),
            'urgent_count': len([t for t in tasks if t.priority > 0.8]),
            'total_cost': sum(t.estimated_cost for t in tasks),
            'avg_priority': sum(t.priority for t in tasks) / len(tasks) if tasks else 0.0,
            'oldest_age': env.time - min([t.created_at for t in tasks], default=env.time),
            'novelty_score': 0.5  # Simplified
        }

    return sense_task_queue


def create_local_capacity_sensor(env: FederationEnvironment):
    """Monitor local ATP capacity"""

    def sense_local_capacity() -> Dict[str, Any]:
        # Get real system stats (simulating local compute capacity)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()

        return {
            'atp_available': env.atp_available,
            'atp_max': env.atp_max,
            'atp_utilization': 1.0 - (env.atp_available / env.atp_max),
            'cpu_percent': cpu_percent,
            'memory_percent': mem.percent,
            'novelty_score': 0.3  # Simplified
        }

    return sense_local_capacity


def create_federation_health_sensor(env: FederationEnvironment):
    """Monitor federation platform health"""

    def sense_federation_health() -> Dict[str, Any]:
        platforms = env.platforms
        online = [p for p in platforms.values() if p.is_online]

        return {
            'platforms_available': len(online),
            'platforms_total': len(platforms),
            'avg_trust': sum(p.trust_score for p in online) / len(online) if online else 0.0,
            'avg_quality': sum(p.recent_quality for p in online) / len(online) if online else 0.0,
            'delegation_success_rate': sum(p.recent_quality for p in platforms.values()) / len(platforms),
            'novelty_score': 0.4  # Simplified
        }

    return sense_federation_health


# ============================================================================
# Federation Actions
# ============================================================================

def create_task_delegation_action(env: FederationEnvironment):
    """Action: Decide whether to execute locally or delegate tasks"""

    def action_delegate_or_execute(
        queue_data: Dict,
        stance: CognitiveStance
    ) -> ExecutionResult:
        tasks = env.pending_tasks.copy()

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

        # FOCUSED_ATTENTION: Handle urgent tasks immediately
        if stance == CognitiveStance.FOCUSED_ATTENTION:
            urgent = [t for t in tasks if t.priority > 0.8]

            for task in urgent:
                if env.atp_available >= task.estimated_cost:
                    # Execute locally (high priority, plenty of resources)
                    if env.execute_local(task):
                        executed_local += 1
                else:
                    # Delegate to best platform
                    online = [p for p in env.platforms.values() if p.is_online]
                    if online:
                        best = max(online, key=lambda p: p.trust_score * p.recent_quality)
                        if env.delegate_task(task, best.name):
                            delegated += 1
                        else:
                            failed += 1
                    else:
                        failed += 1

            return ExecutionResult(
                success=True,
                reward=0.8 if failed == 0 else 0.4,
                description=f"FOCUS: Handled {len(urgent)} urgent tasks ({executed_local} local, {delegated} delegated)",
                outputs={'local': executed_local, 'delegated': delegated, 'failed': failed}
            )

        # CURIOUS_UNCERTAINTY: Explore delegation to less-used platforms
        elif stance == CognitiveStance.CURIOUS_UNCERTAINTY:
            if tasks:
                task = tasks[0]

                # Find least-utilized online platform
                online = [p for p in env.platforms.values() if p.is_online]
                if online:
                    least_used = min(
                        online,
                        key=lambda p: env.delegations_by_platform.get(p.name, 0)
                    )

                    if env.delegate_task(task, least_used.name):
                        delegated += 1

                        return ExecutionResult(
                            success=True,
                            reward=0.6,
                            description=f"CURIOUS: Explored delegation to {least_used.name}",
                            outputs={'explored_platform': least_used.name, 'delegated': 1}
                        )

        # SKEPTICAL_VERIFICATION: Verify quality before delegating
        elif stance == CognitiveStance.SKEPTICAL_VERIFICATION:
            avg_quality = sum(p.recent_quality for p in env.platforms.values()) / len(env.platforms)

            if avg_quality < 0.75:
                # Low quality: execute locally
                for task in tasks[:3]:
                    if env.atp_available >= task.estimated_cost:
                        if env.execute_local(task):
                            executed_local += 1

                return ExecutionResult(
                    success=True,
                    reward=0.5,
                    description=f"SKEPTICAL: Quality concerns, executed {executed_local} locally",
                    outputs={'local': executed_local, 'reason': 'quality_concerns'}
                )

        # CONFIDENT_EXECUTION: Standard delegation logic
        for task in tasks:
            if task.estimated_cost < 100 and env.atp_available >= task.estimated_cost:
                # Cheap task: execute locally
                if env.execute_local(task):
                    executed_local += 1
            else:
                # Expensive task: delegate
                online = [p for p in env.platforms.values() if p.is_online]
                if online:
                    best = max(online, key=lambda p: p.trust_score * p.recent_quality)
                    if env.delegate_task(task, best.name):
                        delegated += 1

        return ExecutionResult(
            success=True,
            reward=0.7,
            description=f"CONFIDENT: Processed {executed_local + delegated} tasks ({executed_local} local, {delegated} delegated)",
            outputs={'local': executed_local, 'delegated': delegated}
        )

    return action_delegate_or_execute


def create_capacity_management_action(env: FederationEnvironment):
    """Action: Manage local capacity and ATP budget"""

    def action_manage_capacity(
        capacity_data: Dict,
        stance: CognitiveStance
    ) -> ExecutionResult:
        atp_util = capacity_data['atp_utilization']
        cpu_percent = capacity_data['cpu_percent']

        actions_taken = []

        # High utilization: Consider REST state
        if atp_util > 0.8 or cpu_percent > 80:
            actions_taken.append("high_utilization_detected")

            if stance == CognitiveStance.FOCUSED_ATTENTION:
                # Emergency: Pause task acceptance temporarily
                actions_taken.append("pause_acceptance")
                reward = 0.6
            else:
                # Normal: Continue but monitor
                actions_taken.append("monitor_closely")
                reward = 0.5

        # Low utilization: Can accept more work
        elif atp_util < 0.3:
            actions_taken.append("capacity_available")
            reward = 0.8

        # Normal operation
        else:
            actions_taken.append("normal_operation")
            reward = 0.7

        return ExecutionResult(
            success=True,
            reward=reward,
            description=f"Capacity: {actions_taken}",
            outputs={'actions': actions_taken, 'atp_util': atp_util, 'cpu': cpu_percent}
        )

    return action_manage_capacity


def create_federation_health_action(env: FederationEnvironment):
    """Action: Monitor and respond to federation health"""

    def action_monitor_federation(
        health_data: Dict,
        stance: CognitiveStance
    ) -> ExecutionResult:
        available = health_data['platforms_available']
        total = health_data['platforms_total']
        avg_quality = health_data['avg_quality']

        actions_taken = []

        # Platform availability issues
        if available < total:
            actions_taken.append(f"platforms_offline: {total - available}")

            if stance == CognitiveStance.SKEPTICAL_VERIFICATION:
                # Verify remaining platforms
                actions_taken.append("verify_remaining_platforms")
                reward = 0.5
            else:
                reward = 0.6

        # Quality issues
        if avg_quality < 0.8:
            actions_taken.append("quality_degradation")
            reward = 0.4

        # All good
        if not actions_taken:
            actions_taken.append("federation_healthy")
            reward = 0.9
        else:
            reward = 0.6

        return ExecutionResult(
            success=True,
            reward=reward,
            description=f"Federation: {actions_taken}",
            outputs={'actions': actions_taken, 'available': available, 'quality': avg_quality}
        )

    return action_monitor_federation


# ============================================================================
# Simplified SAGE Kernel with Metabolic States
# ============================================================================

class FederationSAGEKernel:
    """
    SAGE Kernel managing federation

    Simplified version for demonstration - extends Thor's consciousness kernel
    with federation-specific sensors and actions.
    """

    def __init__(
        self,
        sensor_sources: Dict[str, Any],
        action_handlers: Dict[str, Any],
    ):
        self.sensor_sources = sensor_sources
        self.action_handlers = action_handlers

        # State
        self.cycle_count = 0
        self.running = False
        self.metabolic_state = MetabolicState.WAKE

        # History
        self.execution_history: List[Dict] = []
        self.salience_history = deque(maxlen=100)
        self.state_transitions: List[Tuple[float, MetabolicState, str]] = []

        # Metabolic state timing
        self.state_entry_time = time.time()

    def run(self, max_cycles: Optional[int] = None, cycle_delay: float = 0.5):
        """Main consciousness loop"""
        self.running = True
        self.cycle_count = 0

        print(f"\n[SAGE] Starting consciousness federation monitoring...")
        print(f"[SAGE] Metabolic state: {self.metabolic_state.value}")

        try:
            while self.running:
                if max_cycles is not None and self.cycle_count >= max_cycles:
                    break

                self._cycle()
                self._update_metabolic_state()
                time.sleep(cycle_delay)
                self.cycle_count += 1

        except KeyboardInterrupt:
            print("\n[SAGE] Interrupted by user")
        finally:
            self.running = False
            self._print_statistics()

    def _cycle(self):
        """Execute one consciousness cycle"""
        # 1. SENSE: Gather observations
        observations = {}
        for sensor_id, sensor_fn in self.sensor_sources.items():
            try:
                observations[sensor_id] = sensor_fn()
            except Exception as e:
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

        # 4. STANCE: Determine cognitive stance
        stance = self._determine_stance(observations[focus_target], focus_salience)

        print(f"\n[Cycle {self.cycle_count}] State: {self.metabolic_state.value.upper()}")
        print(f"  Focus: {focus_target} (salience={focus_salience:.3f})")
        print(f"  Stance: {stance.value}")

        # 5. ACT: Execute action
        if focus_target in self.action_handlers:
            try:
                action_fn = self.action_handlers[focus_target]
                result = action_fn(observations[focus_target], stance)

                print(f"  → {result.description}")

                self.execution_history.append({
                    'cycle': self.cycle_count,
                    'focus': focus_target,
                    'stance': stance.value,
                    'reward': result.reward,
                    'success': result.success,
                    'state': self.metabolic_state.value
                })

            except Exception as e:
                print(f"  → Action failed: {e}")

    def _calculate_salience(self, sensor_data: Dict) -> float:
        """Calculate salience for sensor data (simplified)"""
        if not sensor_data:
            return 0.0

        salience = 0.4  # Base

        # Urgency (arousal)
        if sensor_data.get('urgent_count', 0) > 0:
            salience += 0.3

        if sensor_data.get('count', 0) > 5:
            salience += 0.2

        # Resource pressure (surprise)
        if sensor_data.get('atp_utilization', 0) > 0.7:
            salience += 0.3

        # Platform issues (surprise)
        if 'platforms_available' in sensor_data:
            total = sensor_data.get('platforms_total', 3)
            available = sensor_data['platforms_available']
            if available < total:
                salience += 0.2

        # Novelty
        salience += sensor_data.get('novelty_score', 0.0) * 0.1

        return min(1.0, salience)

    def _determine_stance(self, sensor_data: Dict, salience: float) -> CognitiveStance:
        """Determine cognitive stance"""
        # High salience or urgency → FOCUSED_ATTENTION
        if salience > 0.7 or sensor_data.get('urgent_count', 0) > 0:
            return CognitiveStance.FOCUSED_ATTENTION

        # Quality concerns → SKEPTICAL_VERIFICATION
        if sensor_data.get('delegation_success_rate', 1.0) < 0.75:
            return CognitiveStance.SKEPTICAL_VERIFICATION

        # Resource pressure → FOCUSED_ATTENTION
        if sensor_data.get('atp_utilization', 0) > 0.8:
            return CognitiveStance.FOCUSED_ATTENTION

        # Novel situations → CURIOUS_UNCERTAINTY
        if sensor_data.get('novelty_score', 0) > 0.6:
            return CognitiveStance.CURIOUS_UNCERTAINTY

        # Default → CONFIDENT_EXECUTION
        return CognitiveStance.CONFIDENT_EXECUTION

    def _update_metabolic_state(self):
        """Update metabolic state based on activity"""
        time_in_state = time.time() - self.state_entry_time

        # Get recent salience trend
        if len(self.salience_history) >= 5:
            recent_avg = sum(list(self.salience_history)[-5:]) / 5
        else:
            recent_avg = 0.5

        old_state = self.metabolic_state

        # State transitions
        if self.metabolic_state == MetabolicState.WAKE:
            # WAKE → FOCUS: High salience
            if recent_avg > 0.7:
                self.metabolic_state = MetabolicState.FOCUS
            # WAKE → REST: Low salience for extended period
            elif recent_avg < 0.3 and time_in_state > 15:
                self.metabolic_state = MetabolicState.REST

        elif self.metabolic_state == MetabolicState.FOCUS:
            # FOCUS → WAKE: Salience drops or timeout
            if recent_avg < 0.6 or time_in_state > 30:
                self.metabolic_state = MetabolicState.WAKE

        elif self.metabolic_state == MetabolicState.REST:
            # REST → DREAM: After time in REST
            if time_in_state > 10:
                self.metabolic_state = MetabolicState.DREAM

        elif self.metabolic_state == MetabolicState.DREAM:
            # DREAM → WAKE: After consolidation
            if time_in_state > 8:
                self.metabolic_state = MetabolicState.WAKE

        # Log transition
        if old_state != self.metabolic_state:
            self.state_entry_time = time.time()
            self.state_transitions.append((
                time.time(),
                self.metabolic_state,
                f"{old_state.value} → {self.metabolic_state.value}"
            ))
            print(f"\n  ⚡ State transition: {old_state.value} → {self.metabolic_state.value}")

    def _print_statistics(self):
        """Print session statistics"""
        print(f"\n{'='*80}")
        print("FEDERATION CONSCIOUSNESS SESSION SUMMARY")
        print('='*80)

        print(f"\nCycles completed: {self.cycle_count}")

        if self.execution_history:
            avg_reward = sum(h['reward'] for h in self.execution_history) / len(self.execution_history)
            print(f"Average reward: {avg_reward:.3f}")

            # Stance distribution
            stances = {}
            for h in self.execution_history:
                stance = h['stance']
                stances[stance] = stances.get(stance, 0) + 1

            print(f"\nStance distribution:")
            for stance, count in sorted(stances.items(), key=lambda x: x[1], reverse=True):
                pct = (count / len(self.execution_history)) * 100
                print(f"  {stance}: {count} ({pct:.1f}%)")

        if self.state_transitions:
            print(f"\nMetabolic state transitions: {len(self.state_transitions)}")
            for _, state, transition in self.state_transitions[-5:]:
                print(f"  {transition}")


# ============================================================================
# Demonstration
# ============================================================================

def main():
    """Run federation consciousness demonstration"""
    print("="*80)
    print("THOR CONSCIOUSNESS FEDERATION MONITOR")
    print("Hardware: Jetson AGX Thor")
    print("="*80)

    print("\nThis demonstrates:")
    print("- Consciousness kernel managing federation (not reactive API)")
    print("- Salience-driven attention to task queue, capacity, platform health")
    print("- Stance-based delegation decisions (focused/curious/skeptical/confident)")
    print("- Metabolic states affecting federation behavior")
    print("- Learning from outcomes (simplified)")

    # Create simulated federation environment
    env = FederationEnvironment()

    # Create sensors
    sensors = {
        'task_queue': create_task_queue_sensor(env),
        'local_capacity': create_local_capacity_sensor(env),
        'federation_health': create_federation_health_sensor(env),
    }

    # Create actions
    actions = {
        'task_queue': create_task_delegation_action(env),
        'local_capacity': create_capacity_management_action(env),
        'federation_health': create_federation_health_action(env),
    }

    # Create consciousness kernel
    kernel = FederationSAGEKernel(sensors, actions)

    # Run for 30 cycles
    print("\nStarting consciousness loop (30 cycles)...")
    print("(Watch for metabolic state transitions and stance-based decisions)")

    # Advance environment each cycle
    def run_with_env_step():
        for i in range(30):
            env.step()  # Advance simulation
            kernel._cycle()
            kernel._update_metabolic_state()
            time.sleep(0.5)
            kernel.cycle_count += 1

    try:
        run_with_env_step()
    except KeyboardInterrupt:
        print("\n[SAGE] Interrupted by user")

    kernel._print_statistics()

    # Print environment statistics
    print(f"\n{'='*80}")
    print("FEDERATION ENVIRONMENT STATISTICS")
    print('='*80)
    print(f"Tasks created: {env.tasks_created}")
    print(f"Tasks executed locally: {env.tasks_executed_local}")
    print(f"Tasks delegated: {env.tasks_delegated}")
    print(f"Tasks pending: {len(env.pending_tasks)}")
    print(f"ATP remaining: {env.atp_available}/{env.atp_max}")

    print(f"\nDelegations by platform:")
    for platform, count in env.delegations_by_platform.items():
        print(f"  {platform}: {count}")

    print(f"\nPlatform status:")
    for name, platform in env.platforms.items():
        status = "ONLINE" if platform.is_online else "OFFLINE"
        print(f"  {name}: {status} (trust={platform.trust_score:.2f}, quality={platform.recent_quality:.2f})")

    print(f"\n{'='*80}")
    print("KEY INSIGHT")
    print('='*80)
    print("Instead of 'should I delegate this task?' (reactive API),")
    print("consciousness asks 'what needs my attention?' (continuous process).")
    print("\nThis enables:")
    print("- Proactive resource management")
    print("- Stance-appropriate delegation strategies")
    print("- Metabolic adaptation to workload")
    print("- Learning from federation outcomes")
    print('='*80)


if __name__ == "__main__":
    main()
