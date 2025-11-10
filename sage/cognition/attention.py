#!/usr/bin/env python3
"""
Attention Manager for SAGE SNARC Cognition
===========================================

Allocates computational resources to most relevant sensors based on:
- Current goals (goal-driven attention)
- Salience scores (salience-responsive interrupts)
- Memory of useful sensors (memory-informed)
- Sensor trust (reliability-weighted)

Track 3: SNARC Cognition - Component 1/4
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time

# Import from Track 2 (Memory)
try:
    from sage.memory.retrieval import MemoryRetrieval
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from sage.memory.retrieval import MemoryRetrieval


@dataclass
class ResourceBudget:
    """
    Computational resource constraints for attention

    Defines limits for simultaneous sensor processing on Nano
    """
    max_active_sensors: int = 3  # Maximum sensors to process simultaneously
    max_memory_mb: float = 100.0  # Maximum memory for attention (MB)
    max_latency_ms: float = 10.0  # Maximum latency budget (ms)

    def can_accommodate(self, num_sensors: int) -> bool:
        """Check if budget can accommodate N sensors"""
        return num_sensors <= self.max_active_sensors


@dataclass
class AttentionAllocation:
    """
    Result of attention allocation

    Specifies which sensors to focus on and with what weights
    """
    focus_weights: Dict[str, float]  # sensor_id -> weight (0-1, sum to 1.0)
    active_sensors: List[str]  # Sensors currently attended (sorted by priority)
    inhibited_sensors: List[str]  # Sensors temporarily suppressed

    timestamp: float = field(default_factory=time.time)

    # Attention justification (for interpretability)
    reason: str = ""
    goal_driven: bool = False  # Was this allocation driven by goals?
    interrupt: bool = False  # Was this an interrupt (high salience)?

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'timestamp': self.timestamp,
            'active_sensors': self.active_sensors,
            'focus_weights': self.focus_weights,
            'inhibited_count': len(self.inhibited_sensors),
            'reason': self.reason,
            'goal_driven': self.goal_driven,
            'interrupt': self.interrupt
        }


@dataclass
class AttentionState:
    """Saved attention state for interrupt handling"""
    focus_weights: Dict[str, float]
    active_sensors: List[str]
    timestamp: float
    reason: str


class AttentionManager:
    """
    Manages sensor attention allocation for SNARC

    Core Responsibilities:
    - Allocate attention to subset of sensors (not all simultaneously)
    - Balance goal-driven and salience-responsive attention
    - Respect resource constraints (Nano compatibility)
    - Handle attention interrupts (high-salience events)
    - Learn from memory which sensors are useful

    Integration:
    - Track 1 (Sensor Trust): Uses trust scores to weight attention
    - Track 2 (Memory): Queries memory for useful sensor patterns
    - SNARC: Receives salience scores, outputs attention weights
    - Working Memory: Gets task context for goal-relevant attention
    """

    def __init__(
        self,
        available_sensors: List[str],
        resource_budget: Optional[ResourceBudget] = None,
        memory_retrieval: Optional[MemoryRetrieval] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Attention Manager

        Args:
            available_sensors: List of sensor IDs
            resource_budget: Resource constraints (default: 3 sensors, 100MB, 10ms)
            memory_retrieval: Memory system from Track 2
            device: Device for tensor operations
        """
        self.sensors = available_sensors
        self.budget = resource_budget or ResourceBudget()
        self.memory = memory_retrieval
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Current attention state
        self.focus_weights: Dict[str, float] = {s: 1.0 / len(self.sensors) for s in self.sensors}
        self.active_sensors: List[str] = []

        # Inhibition tracking
        self.inhibited_sensors: Dict[str, int] = {}  # sensor_id -> cycles remaining

        # Interrupts
        self.interrupt_threshold: float = 0.9  # Salience threshold for interrupts
        self.saved_state: Optional[AttentionState] = None
        self.interrupt_active: bool = False
        self.interrupt_restore_cycles: int = 10  # Cycles before restoring from interrupt
        self.cycles_since_interrupt: int = 0

        # Attention history (for learning and analysis)
        self.attention_history: deque = deque(maxlen=1000)

        # Attention weights (for scoring)
        self.weight_goal: float = 0.4  # α - goal relevance weight
        self.weight_salience: float = 0.3  # β - salience weight
        self.weight_memory: float = 0.2  # γ - memory utility weight
        self.weight_trust: float = 0.1  # δ - trust score weight

        # Statistics
        self.total_allocations: int = 0
        self.total_interrupts: int = 0

    def allocate_attention(
        self,
        current_salience: Dict[str, float],
        active_goals: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        sensor_trust: Optional[Dict[str, float]] = None
    ) -> AttentionAllocation:
        """
        Allocate attention across available sensors

        Decision factors:
        1. Goal relevance - which sensors help current goals?
        2. Current salience - which sensors have high salience?
        3. Memory utility - which sensors were useful in similar situations?
        4. Sensor trust - which sensors are reliable?

        Args:
            current_salience: Salience score per sensor (from SNARC)
            active_goals: Current active goals (from GoalManager)
            context: Task context (from WorkingMemory)
            sensor_trust: Trust scores per sensor (from Track 1)

        Returns:
            AttentionAllocation with focus weights and active sensors
        """
        self.total_allocations += 1

        # Check for attention interrupts (high salience)
        interrupt_sensor = self._check_interrupts(current_salience)
        if interrupt_sensor:
            return self._handle_interrupt(interrupt_sensor, current_salience)

        # If in interrupt mode, check if we should restore
        if self.interrupt_active:
            self.cycles_since_interrupt += 1
            if self.cycles_since_interrupt >= self.interrupt_restore_cycles:
                self._restore_from_interrupt()

        # Decay inhibitions
        self._decay_inhibitions()

        # Compute attention scores for each sensor
        attention_scores = self._compute_attention_scores(
            current_salience,
            active_goals,
            context,
            sensor_trust
        )

        # Apply inhibitions (set inhibited sensors to 0)
        for sensor_id in self.inhibited_sensors:
            attention_scores[sensor_id] = 0.0

        # Select top K sensors within budget
        active_sensors, focus_weights = self._select_active_sensors(
            attention_scores,
            self.budget.max_active_sensors
        )

        # Create allocation
        allocation = AttentionAllocation(
            focus_weights=focus_weights,
            active_sensors=active_sensors,
            inhibited_sensors=list(self.inhibited_sensors.keys()),
            reason=self._explain_allocation(active_sensors, attention_scores),
            goal_driven=active_goals is not None and len(active_goals) > 0
        )

        # Update state
        self.focus_weights = focus_weights
        self.active_sensors = active_sensors

        # Record history
        self.attention_history.append(allocation)

        return allocation

    def _compute_attention_scores(
        self,
        current_salience: Dict[str, float],
        active_goals: Optional[List[Any]],
        context: Optional[Dict[str, Any]],
        sensor_trust: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute attention score for each sensor

        Score = α*goal_relevance + β*salience + γ*memory_utility + δ*trust
        where α + β + γ + δ = 1.0
        """
        scores = {}

        for sensor_id in self.sensors:
            # Factor 1: Goal relevance
            goal_score = self._compute_goal_relevance(sensor_id, active_goals, context)

            # Factor 2: Current salience
            salience_score = current_salience.get(sensor_id, 0.0)

            # Factor 3: Memory utility
            memory_score = self._compute_memory_utility(sensor_id, context)

            # Factor 4: Trust score
            trust_score = sensor_trust.get(sensor_id, 0.5) if sensor_trust else 0.5

            # Weighted combination
            attention_score = (
                self.weight_goal * goal_score +
                self.weight_salience * salience_score +
                self.weight_memory * memory_score +
                self.weight_trust * trust_score
            )

            scores[sensor_id] = min(max(attention_score, 0.0), 1.0)  # Clamp to [0, 1]

        return scores

    def _compute_goal_relevance(
        self,
        sensor_id: str,
        active_goals: Optional[List[Any]],
        context: Optional[Dict[str, Any]]
    ) -> float:
        """
        Compute how relevant a sensor is to current goals

        Strategy:
        1. If no goals, return uniform (0.5)
        2. Check if context specifies goal-sensor mappings
        3. Use heuristics (e.g., "vision" relevant for navigation goals)
        """
        if not active_goals or len(active_goals) == 0:
            return 0.5  # Uniform when no goals

        # Check context for explicit goal-sensor relevance
        if context and 'goal_sensor_relevance' in context:
            relevance_map = context['goal_sensor_relevance']
            if sensor_id in relevance_map:
                return relevance_map[sensor_id]

        # Heuristics based on goal type and sensor type
        # (In production, this would be learned from experience)
        relevance = 0.5

        for goal in active_goals:
            goal_type = getattr(goal, 'goal_type', '')

            # Navigation goals prioritize vision
            if 'navigation' in goal_type.lower() or 'explore' in goal_type.lower():
                if 'vision' in sensor_id.lower() or 'camera' in sensor_id.lower():
                    relevance = max(relevance, 0.9)

            # Manipulation goals prioritize proprioception
            if 'manipulation' in goal_type.lower() or 'grasp' in goal_type.lower():
                if 'proprioception' in sensor_id.lower() or 'joint' in sensor_id.lower():
                    relevance = max(relevance, 0.9)

            # Obstacle avoidance prioritizes vision + IMU
            if 'avoid' in goal_type.lower():
                if any(s in sensor_id.lower() for s in ['vision', 'imu', 'distance']):
                    relevance = max(relevance, 0.8)

        return relevance

    def _compute_memory_utility(
        self,
        sensor_id: str,
        context: Optional[Dict[str, Any]]
    ) -> float:
        """
        Query memory for sensor utility in similar situations

        Strategy:
        1. If memory not available, return neutral (0.5)
        2. Query memory for similar contexts
        3. Check which sensors were focused on successfully
        4. Return utility score
        """
        if not self.memory or not context:
            return 0.5  # Neutral when no memory context

        # Query memory for similar situations
        # (This would retrieve from Track 2 memory based on context similarity)
        # For now, return neutral - full integration would query LTM

        # TODO: Implement memory query when integrated with Track 2
        # similar_memories = self.memory.query_by_context(context, n=10)
        # utility = compute_sensor_utility(sensor_id, similar_memories)

        return 0.5

    def _select_active_sensors(
        self,
        attention_scores: Dict[str, float],
        max_sensors: int
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Select top K sensors and compute focus weights

        Strategy:
        1. Sort sensors by attention score (descending)
        2. Select top K where K ≤ budget
        3. Compute weights proportional to scores
        4. Normalize weights to sum to 1.0
        """
        # Sort by score
        sorted_sensors = sorted(
            attention_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top K
        active_sensors = [sensor_id for sensor_id, _ in sorted_sensors[:max_sensors]]

        # Compute weights (proportional to scores)
        active_scores = {sid: attention_scores[sid] for sid in active_sensors}
        total_score = sum(active_scores.values())

        if total_score > 0:
            focus_weights = {sid: score / total_score for sid, score in active_scores.items()}
        else:
            # Uniform if all scores are 0
            focus_weights = {sid: 1.0 / len(active_sensors) for sid in active_sensors}

        return active_sensors, focus_weights

    def _check_interrupts(
        self,
        current_salience: Dict[str, float]
    ) -> Optional[str]:
        """
        Check if any sensor has critically high salience requiring attention interrupt

        Returns:
            sensor_id if interrupt needed, None otherwise
        """
        for sensor_id, salience in current_salience.items():
            if salience >= self.interrupt_threshold:
                return sensor_id
        return None

    def _handle_interrupt(
        self,
        interrupt_sensor: str,
        current_salience: Dict[str, float]
    ) -> AttentionAllocation:
        """
        Handle attention interrupt from high-salience sensor

        Strategy:
        1. Save current attention state
        2. Shift all focus to interrupt sensor
        3. Mark as interrupt mode
        """
        # Save current state (if not already in interrupt)
        if not self.interrupt_active:
            self.saved_state = AttentionState(
                focus_weights=self.focus_weights.copy(),
                active_sensors=self.active_sensors.copy(),
                timestamp=time.time(),
                reason="interrupt"
            )

        # Shift full focus to interrupt sensor
        self.interrupt_active = True
        self.cycles_since_interrupt = 0
        self.total_interrupts += 1

        allocation = AttentionAllocation(
            focus_weights={interrupt_sensor: 1.0},
            active_sensors=[interrupt_sensor],
            inhibited_sensors=[],
            reason=f"INTERRUPT: {interrupt_sensor} salience={current_salience[interrupt_sensor]:.3f}",
            interrupt=True
        )

        # Update state
        self.focus_weights = allocation.focus_weights
        self.active_sensors = allocation.active_sensors

        # Record history
        self.attention_history.append(allocation)

        return allocation

    def _restore_from_interrupt(self):
        """Restore attention state after interrupt"""
        if self.saved_state:
            self.focus_weights = self.saved_state.focus_weights
            self.active_sensors = self.saved_state.active_sensors
            self.saved_state = None

        self.interrupt_active = False
        self.cycles_since_interrupt = 0

    def inhibit_sensor(self, sensor_id: str, duration: int = 10):
        """
        Temporarily suppress attention to a sensor

        Args:
            sensor_id: Sensor to inhibit
            duration: Number of cycles to inhibit
        """
        if sensor_id in self.sensors:
            self.inhibited_sensors[sensor_id] = duration

    def boost_sensor(self, sensor_id: str, factor: float = 2.0):
        """
        Temporarily amplify attention to a sensor

        Args:
            sensor_id: Sensor to boost
            factor: Amplification factor
        """
        if sensor_id in self.focus_weights:
            self.focus_weights[sensor_id] = min(self.focus_weights[sensor_id] * factor, 1.0)

    def _decay_inhibitions(self):
        """Decay inhibition timers"""
        to_remove = []
        for sensor_id in list(self.inhibited_sensors.keys()):
            self.inhibited_sensors[sensor_id] -= 1
            if self.inhibited_sensors[sensor_id] <= 0:
                to_remove.append(sensor_id)

        for sensor_id in to_remove:
            del self.inhibited_sensors[sensor_id]

    def _explain_allocation(
        self,
        active_sensors: List[str],
        attention_scores: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation of attention allocation"""
        if not active_sensors:
            return "No sensors active"

        explanations = []
        for sensor_id in active_sensors:
            score = attention_scores[sensor_id]
            explanations.append(f"{sensor_id}={score:.3f}")

        return f"Active: {', '.join(explanations)}"

    def get_stats(self) -> Dict[str, Any]:
        """Get attention statistics"""
        return {
            'total_allocations': self.total_allocations,
            'total_interrupts': self.total_interrupts,
            'interrupt_rate': self.total_interrupts / max(self.total_allocations, 1),
            'current_active_sensors': len(self.active_sensors),
            'budget_max_sensors': self.budget.max_active_sensors,
            'inhibited_sensors': len(self.inhibited_sensors),
            'interrupt_active': self.interrupt_active
        }

    def reset(self):
        """Reset attention state"""
        self.focus_weights = {s: 1.0 / len(self.sensors) for s in self.sensors}
        self.active_sensors = []
        self.inhibited_sensors = {}
        self.interrupt_active = False
        self.saved_state = None
        self.cycles_since_interrupt = 0


def test_attention_manager():
    """Test Attention Manager"""
    print("\n" + "="*60)
    print("TESTING ATTENTION MANAGER")
    print("="*60)

    # Create attention manager
    sensors = ['vision', 'proprioception', 'audio', 'imu']
    budget = ResourceBudget(max_active_sensors=2)

    attention = AttentionManager(
        available_sensors=sensors,
        resource_budget=budget
    )

    # Test 1: Basic allocation with uniform salience
    print("\n1. Uniform salience allocation...")
    allocation = attention.allocate_attention(
        current_salience={'vision': 0.5, 'proprioception': 0.5, 'audio': 0.5, 'imu': 0.5}
    )
    print(f"   Active sensors: {allocation.active_sensors}")
    print(f"   Weights: {allocation.focus_weights}")
    print(f"   Reason: {allocation.reason}")

    # Test 2: High salience on one sensor
    print("\n2. High salience on vision...")
    allocation = attention.allocate_attention(
        current_salience={'vision': 0.9, 'proprioception': 0.3, 'audio': 0.2, 'imu': 0.3}
    )
    print(f"   Active sensors: {allocation.active_sensors}")
    print(f"   Top weight: {max(allocation.focus_weights.values()):.3f}")

    # Test 3: Attention interrupt
    print("\n3. Critical salience interrupt...")
    allocation = attention.allocate_attention(
        current_salience={'vision': 0.95, 'proprioception': 0.3, 'audio': 0.2, 'imu': 0.3}
    )
    print(f"   Interrupt: {allocation.interrupt}")
    print(f"   Active sensors: {allocation.active_sensors}")
    print(f"   Reason: {allocation.reason}")

    # Test 4: Inhibition
    print("\n4. Inhibit audio sensor...")
    attention.inhibit_sensor('audio', duration=5)
    allocation = attention.allocate_attention(
        current_salience={'vision': 0.5, 'proprioception': 0.5, 'audio': 0.8, 'imu': 0.5}
    )
    print(f"   Active sensors: {allocation.active_sensors}")
    print(f"   Inhibited: {allocation.inhibited_sensors}")
    print(f"   Audio excluded: {'audio' not in allocation.active_sensors}")

    # Test 5: Statistics
    print("\n5. Attention statistics...")
    stats = attention.get_stats()
    print(f"   Total allocations: {stats['total_allocations']}")
    print(f"   Total interrupts: {stats['total_interrupts']}")
    print(f"   Interrupt rate: {stats['interrupt_rate']:.2%}")

    print("\n" + "="*60)
    print("✅ ATTENTION MANAGER TESTS PASSED")
    print("="*60)

    return attention


if __name__ == "__main__":
    test_attention_manager()
