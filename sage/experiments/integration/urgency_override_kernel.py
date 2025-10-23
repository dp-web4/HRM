#!/usr/bin/env python3
"""
Urgency Override Kernel
Extends attention switching with emergency interrupt capability.

New mechanism: If any observation has importance above urgency threshold,
immediately switch focus to that sensor, bypassing ε-greedy and salience.

This implements biological "salience interrupt" - critical events override
current attention regardless of boredom, exploration, or competition.
"""

import sys
import os
from pathlib import Path
import random
import time
from typing import Dict, Any, Callable, List
from dataclasses import dataclass

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

class UrgencyOverrideKernel:
    """
    Attention switching kernel with urgency override.

    Adds critical event interrupt to attention switching:
    - Normal cycles use ε-greedy + salience decay + exploration
    - But if any observation importance > urgency_threshold:
      → Immediately switch to that sensor (bypass all mechanisms)

    This implements biological salience interrupts.
    """

    def __init__(
        self,
        sensor_sources: Dict[str, Callable],
        action_handlers: Dict[str, Callable],
        epsilon: float = 0.15,
        decay_rate: float = 0.97,
        exploration_weight: float = 0.05,
        urgency_threshold: float = 0.90  # New parameter
    ):
        """
        Initialize urgency override kernel

        Args:
            sensor_sources: Dict of sensor_id -> callable
            action_handlers: Dict of sensor_id -> handler function
            epsilon: Probability of random exploration (0-1)
            decay_rate: Salience decay for focused sensor (0-1)
            exploration_weight: Bonus for less-visited sensors
            urgency_threshold: Importance above this triggers immediate switch
        """
        self.sensor_sources = sensor_sources
        self.action_handlers = action_handlers

        # Attention parameters
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.exploration_weight = exploration_weight
        self.urgency_threshold = urgency_threshold

        # Tracking
        self.current_focus = None
        self.visit_counts = {sensor: 0 for sensor in sensor_sources}
        self.salience_estimates = {sensor: 0.5 for sensor in sensor_sources}
        self.recent_rewards = {sensor: [] for sensor in sensor_sources}

        # Urgency tracking
        self.urgency_overrides = 0
        self.critical_events = []

        # History
        self.history = []
        self.cycle_count = 0
        self.running = False

    def _compute_salience(self, sensor_id: str, observation: Any) -> float:
        """Compute salience for a sensor"""
        # Novelty (inverse of visit frequency)
        total_visits = sum(self.visit_counts.values()) + 1
        novelty = 1.0 - (self.visit_counts[sensor_id] / total_visits)

        # Reward expectation (from history)
        if self.recent_rewards[sensor_id]:
            reward_estimate = sum(self.recent_rewards[sensor_id]) / len(self.recent_rewards[sensor_id])
        else:
            reward_estimate = 0.5  # Default

        # Exploration bonus (curiosity about less-visited)
        exploration_bonus = self.exploration_weight / (self.visit_counts[sensor_id] + 1)

        # Combined salience
        salience = 0.3 * novelty + 0.5 * reward_estimate + 0.2 * exploration_bonus

        return salience

    def _check_urgency(self, observations: Dict[str, Any]) -> tuple:
        """
        Check for urgent observations that should override normal attention.

        Returns:
            (urgent_sensor_id, importance) if urgent event found, else (None, None)
        """
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
        Select which sensor to focus on.

        NEW: First checks for urgent events that override normal selection.
        Then uses ε-greedy with salience decay and exploration bonus.

        Returns:
            (focus_sensor_id, selection_type)
            selection_type: "urgency", "random", or "greedy"
        """
        # URGENCY CHECK (NEW)
        urgent_sensor, urgency_level = self._check_urgency(observations)
        if urgent_sensor:
            self.urgency_overrides += 1
            self.critical_events.append({
                'cycle': self.cycle_count,
                'sensor': urgent_sensor,
                'importance': urgency_level
            })
            return urgent_sensor, "urgency"

        # Recompute salience for ALL sensors (normal path)
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
            exploration_type = "random"
        else:
            focus = max(fresh_salience, key=fresh_salience.get)
            exploration_type = "greedy"

        # Store salience estimates
        self.salience_estimates.update(fresh_salience)

        return focus, exploration_type

    def _cycle(self):
        """Execute one attention cycle"""
        # Gather all observations
        observations = {}
        for sensor_id, sensor_fn in self.sensor_sources.items():
            obs = sensor_fn()
            observations[sensor_id] = obs

        # Filter to only active observations
        active_observations = {
            sid: obs for sid, obs in observations.items()
            if obs is not None
        }

        if not active_observations:
            return

        # Select focus (with urgency override and exploration)
        focus, selection_type = self._select_focus(active_observations)

        if focus is None:
            return

        # Execute action
        handler = self.action_handlers[focus]
        observation = active_observations[focus]

        stance = CognitiveStance.CURIOUS_UNCERTAINTY

        result = handler(observation, stance)

        # Update tracking
        self.visit_counts[focus] += 1
        self.current_focus = focus

        # Store reward
        self.recent_rewards[focus].append(result.reward)
        if len(self.recent_rewards[focus]) > 5:
            self.recent_rewards[focus].pop(0)

        # Record history
        self.history.append({
            'cycle': self.cycle_count,
            'focus': focus,
            'selection_type': selection_type,  # "urgency", "random", or "greedy"
            'salience': self.salience_estimates.get(focus, 0.0),
            'result': result,
            'all_salience': self.salience_estimates.copy()
        })

    def run(self, max_cycles: int = 100, cycle_delay: float = 0.1):
        """Run the attention loop"""
        self.running = True
        self.cycle_count = 0

        print(f"Starting urgency-override kernel...")
        print(f"  ε-greedy: {self.epsilon} exploration")
        print(f"  Decay: {self.decay_rate} focus decay")
        print(f"  Exploration weight: {self.exploration_weight}")
        print(f"  Urgency threshold: {self.urgency_threshold} (NEW)")
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

    def get_history(self):
        """Get execution history"""
        return self.history

    def get_statistics(self):
        """Get attention statistics"""
        if not self.history:
            return {}

        # Focus distribution
        focus_counts = {}
        for h in self.history:
            focus = h['focus']
            focus_counts[focus] = focus_counts.get(focus, 0) + 1

        # Selection type distribution
        urgency_count = sum(1 for h in self.history if h['selection_type'] == 'urgency')
        random_count = sum(1 for h in self.history if h['selection_type'] == 'random')
        greedy_count = sum(1 for h in self.history if h['selection_type'] == 'greedy')

        # Attention switches
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
            'visit_counts': self.visit_counts.copy(),
            'critical_events': self.critical_events.copy()
        }
