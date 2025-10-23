#!/usr/bin/env python3
"""
Memory-Aware Attention Kernel
Extends attention switching with episodic and working memory.

Memory Systems:
1. Working Memory (circular buffer) - Recent context across modalities
2. Episodic Memory - Significant events with salience scores
3. Conversation Memory - Dialogue history for context-aware responses
4. Attention History - Past focus patterns inform future decisions

Optimized for Jetson:
- Circular buffers (fixed size, no growth)
- Efficient deque structures
- Configurable memory limits
- Minimal overhead per cycle
"""

import sys
import os
from pathlib import Path
import random
import time
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime

hrm_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(hrm_root))
os.chdir(hrm_root)

from sage.services.snarc.data_structures import CognitiveStance

@dataclass
class ExecutionResult:
    """Result from action execution"""
    success: bool
    reward: float
    description: str
    outputs: Dict[str, Any]

@dataclass
class MemoryEvent:
    """Single memory event"""
    cycle: int
    timestamp: float
    modality: str
    observation: Dict[str, Any]
    result: ExecutionResult
    salience: float
    importance: float

@dataclass
class ConversationTurn:
    """Single conversation turn"""
    cycle: int
    speaker: str  # 'user' or 'sage'
    text: str
    stance: Optional[CognitiveStance] = None
    importance: float = 0.5

class MemoryAwareKernel:
    """
    Attention switching kernel with integrated memory systems.

    Optimized for Jetson deployment:
    - Fixed-size circular buffers (no unbounded growth)
    - Efficient event storage with automatic pruning
    - Configurable memory limits
    - Low per-cycle overhead
    """

    def __init__(
        self,
        sensor_sources: Dict[str, Callable],
        action_handlers: Dict[str, Callable],
        epsilon: float = 0.15,
        decay_rate: float = 0.97,
        exploration_weight: float = 0.05,
        urgency_threshold: float = 0.90,
        working_memory_size: int = 20,  # Recent context per modality
        episodic_memory_size: int = 100,  # Significant events
        conversation_memory_size: int = 10,  # Recent conversation turns
    ):
        """
        Initialize memory-aware kernel

        Args:
            sensor_sources: Dict of sensor_id -> callable
            action_handlers: Dict of sensor_id -> handler function
            epsilon: Probability of random exploration
            decay_rate: Salience decay for focused sensor
            exploration_weight: Bonus for less-visited sensors
            urgency_threshold: Importance for immediate interrupt
            working_memory_size: Size of recent context buffer per modality
            episodic_memory_size: Max significant events to remember
            conversation_memory_size: Max conversation turns to track
        """
        self.sensor_sources = sensor_sources
        self.action_handlers = action_handlers

        # Attention parameters
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.exploration_weight = exploration_weight
        self.urgency_threshold = urgency_threshold

        # Memory limits (Jetson optimization)
        self.working_memory_size = working_memory_size
        self.episodic_memory_size = episodic_memory_size
        self.conversation_memory_size = conversation_memory_size

        # Attention tracking
        self.current_focus = None
        self.visit_counts = {sensor: 0 for sensor in sensor_sources}
        self.salience_estimates = {sensor: 0.5 for sensor in sensor_sources}
        self.recent_rewards = {sensor: [] for sensor in sensor_sources}

        # Memory systems (circular buffers for efficiency)
        self.working_memory = {
            sensor: deque(maxlen=working_memory_size)
            for sensor in sensor_sources
        }
        self.episodic_memory = deque(maxlen=episodic_memory_size)
        self.conversation_memory = deque(maxlen=conversation_memory_size)

        # Attention history (for pattern learning)
        self.attention_history = deque(maxlen=50)  # Recent focus patterns

        # Statistics
        self.history = []
        self.cycle_count = 0
        self.running = False
        self.urgency_overrides = 0

    def _compute_salience(self, sensor_id: str, observation: Any) -> float:
        """
        Compute salience with memory influence.

        Memory enhancement: Recent experiences inform salience estimation.
        """
        # Base novelty (inverse of visit frequency)
        total_visits = sum(self.visit_counts.values()) + 1
        novelty = 1.0 - (self.visit_counts[sensor_id] / total_visits)

        # Reward expectation (from recent rewards)
        if self.recent_rewards[sensor_id]:
            reward_estimate = sum(self.recent_rewards[sensor_id]) / len(self.recent_rewards[sensor_id])
        else:
            reward_estimate = 0.5

        # Exploration bonus
        exploration_bonus = self.exploration_weight / (self.visit_counts[sensor_id] + 1)

        # MEMORY INFLUENCE: Recent working memory events
        memory_boost = 0.0
        if sensor_id in self.working_memory and len(self.working_memory[sensor_id]) > 0:
            # Recent events with high importance boost salience
            recent_events = list(self.working_memory[sensor_id])[-3:]  # Last 3
            recent_importance = [e.importance for e in recent_events]
            if recent_importance:
                memory_boost = 0.1 * (sum(recent_importance) / len(recent_importance))

        # Combined salience with memory
        salience = 0.3 * novelty + 0.4 * reward_estimate + 0.2 * exploration_bonus + 0.1 * memory_boost

        return salience

    def _check_urgency(self, observations: Dict[str, Any]) -> tuple:
        """Check for urgent observations"""
        urgent_sensor = None
        max_importance = 0.0

        for sensor_id, obs in observations.items():
            if obs is not None and 'importance' in obs:
                importance = obs['importance']
                if importance > self.urgency_threshold:
                    if importance > max_importance:
                        max_importance = importance
                        urgent_sensor = sensor_id

        return urgent_sensor, max_importance if urgent_sensor else None

    def _select_focus(self, observations: Dict[str, Any]) -> tuple:
        """
        Select focus with urgency override and memory-enhanced salience.

        Returns: (focus_sensor_id, selection_type)
        """
        # Urgency check first
        urgent_sensor, urgency_level = self._check_urgency(observations)
        if urgent_sensor:
            self.urgency_overrides += 1
            return urgent_sensor, "urgency"

        # Compute salience for all sensors (with memory influence)
        fresh_salience = {}
        for sensor_id, obs in observations.items():
            if obs is not None:
                salience = self._compute_salience(sensor_id, obs)

                # Decay current focus (boredom)
                if sensor_id == self.current_focus:
                    salience *= self.decay_rate

                fresh_salience[sensor_id] = salience

        if not fresh_salience:
            return None, None

        # ε-greedy selection
        if random.random() < self.epsilon:
            focus = random.choice(list(fresh_salience.keys()))
            selection_type = "random"
        else:
            focus = max(fresh_salience, key=fresh_salience.get)
            selection_type = "greedy"

        self.salience_estimates.update(fresh_salience)

        return focus, selection_type

    def add_to_working_memory(self, sensor_id: str, event: MemoryEvent):
        """Add event to working memory (circular buffer, auto-pruning)"""
        self.working_memory[sensor_id].append(event)

    def add_to_episodic_memory(self, event: MemoryEvent):
        """Add significant event to episodic memory"""
        # Only store events with salience above threshold
        if event.salience > 0.6 or event.importance > 0.7:
            self.episodic_memory.append(event)

    def add_conversation_turn(self, turn: ConversationTurn):
        """Add conversation turn to memory"""
        self.conversation_memory.append(turn)

    def get_recent_conversation(self, n: int = 5) -> List[ConversationTurn]:
        """Get recent conversation turns"""
        return list(self.conversation_memory)[-n:]

    def get_working_memory_summary(self, sensor_id: str) -> str:
        """Get text summary of recent events for a sensor"""
        if sensor_id not in self.working_memory:
            return "No recent memory"

        events = list(self.working_memory[sensor_id])
        if not events:
            return "No recent events"

        summary_parts = []
        for event in events[-5:]:  # Last 5 events
            summary_parts.append(f"Cycle {event.cycle}: {event.result.description}")

        return " | ".join(summary_parts)

    def _cycle(self):
        """Execute one attention cycle with memory updates"""
        # Gather observations
        observations = {}
        for sensor_id, sensor_fn in self.sensor_sources.items():
            obs = sensor_fn()
            observations[sensor_id] = obs

        # Filter to active observations
        active_observations = {
            sid: obs for sid, obs in observations.items()
            if obs is not None
        }

        if not active_observations:
            return

        # Select focus (with urgency override and memory-enhanced salience)
        focus, selection_type = self._select_focus(active_observations)

        if focus is None:
            return

        # Execute action
        handler = self.action_handlers[focus]
        observation = active_observations[focus]
        stance = CognitiveStance.CURIOUS_UNCERTAINTY

        result = handler(observation, stance)

        # Update attention tracking
        self.visit_counts[focus] += 1
        self.current_focus = focus

        # Update reward history
        self.recent_rewards[focus].append(result.reward)
        if len(self.recent_rewards[focus]) > 5:
            self.recent_rewards[focus].pop(0)

        # Create memory event
        importance = observation.get('importance', result.reward)
        salience = self.salience_estimates.get(focus, 0.5)

        memory_event = MemoryEvent(
            cycle=self.cycle_count,
            timestamp=time.time(),
            modality=focus,
            observation=observation,
            result=result,
            salience=salience,
            importance=importance
        )

        # Store in memory systems
        self.add_to_working_memory(focus, memory_event)
        self.add_to_episodic_memory(memory_event)

        # Track attention pattern
        self.attention_history.append({
            'cycle': self.cycle_count,
            'focus': focus,
            'salience': salience
        })

        # Record in history
        self.history.append({
            'cycle': self.cycle_count,
            'focus': focus,
            'selection_type': selection_type,
            'salience': salience,
            'result': result,
            'memory_event': memory_event
        })

    def run(self, max_cycles: int = 100, cycle_delay: float = 0.1):
        """Run the attention loop with memory"""
        self.running = True
        self.cycle_count = 0

        print(f"Starting memory-aware kernel...")
        print(f"  Memory limits: Working={self.working_memory_size}, "
              f"Episodic={self.episodic_memory_size}, "
              f"Conversation={self.conversation_memory_size}")
        print(f"  Attention: ε={self.epsilon}, decay={self.decay_rate}, "
              f"urgency_threshold={self.urgency_threshold}")
        print()

        try:
            while self.running and self.cycle_count < max_cycles:
                self._cycle()
                time.sleep(cycle_delay)
                self.cycle_count += 1

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.running = False

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        total_working_memory = sum(len(wm) for wm in self.working_memory.values())

        return {
            'working_memory_total': total_working_memory,
            'working_memory_per_modality': {
                sensor: len(wm) for sensor, wm in self.working_memory.items()
            },
            'episodic_memory_count': len(self.episodic_memory),
            'conversation_memory_count': len(self.conversation_memory),
            'attention_history_count': len(self.attention_history),
            'high_salience_events': sum(1 for e in self.episodic_memory if e.salience > 0.8),
            'high_importance_events': sum(1 for e in self.episodic_memory if e.importance > 0.8),
        }

    def get_history(self):
        """Get execution history"""
        return self.history

    def get_statistics(self):
        """Get attention and memory statistics"""
        if not self.history:
            return {}

        # Focus distribution
        focus_counts = {}
        for h in self.history:
            focus = h['focus']
            focus_counts[focus] = focus_counts.get(focus, 0) + 1

        # Selection types
        urgency_count = sum(1 for h in self.history if h['selection_type'] == 'urgency')
        random_count = sum(1 for h in self.history if h['selection_type'] == 'random')
        greedy_count = sum(1 for h in self.history if h['selection_type'] == 'greedy')

        # Switches
        switches = 0
        for i in range(1, len(self.history)):
            if self.history[i]['focus'] != self.history[i-1]['focus']:
                switches += 1

        return {
            'total_cycles': len(self.history),
            'focus_distribution': focus_counts,
            'urgency_overrides': urgency_count,
            'random_exploration': random_count,
            'greedy_exploitation': greedy_count,
            'attention_switches': switches,
            'memory': self.get_memory_statistics()
        }
