#!/usr/bin/env python3
"""
Attention Switching Kernel
Modified SAGE kernel with exploration mechanisms to prevent
attentional monopolization.

Implements:
1. Salience decay (boredom with current focus)
2. Exploration bonus (curiosity about unvisited)
3. ε-greedy selection (guaranteed sampling)
4. Fresh assessment every cycle (dynamic response)

Tests whether these mechanisms enable multi-modal awareness.
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

class AttentionSwitchingKernel:
    """
    SAGE kernel with attention switching mechanisms.

    Key differences from base SAGE:
    - Recomputes salience for ALL sensors each cycle
    - Decays focus sensor salience (boredom)
    - Adds exploration bonus (curiosity)
    - Uses ε-greedy selection
    """

    def __init__(
        self,
        sensor_sources: Dict[str, Callable],
        action_handlers: Dict[str, Callable],
        epsilon: float = 0.15,  # 15% exploration
        decay_rate: float = 0.97,  # 3% boredom per cycle
        exploration_weight: float = 0.05
    ):
        """
        Initialize switching kernel

        Args:
            sensor_sources: Dict of sensor_id -> callable
            action_handlers: Dict of sensor_id -> handler function
            epsilon: Probability of random exploration (0-1)
            decay_rate: Salience decay for focused sensor (0-1)
            exploration_weight: Bonus for less-visited sensors
        """
        self.sensor_sources = sensor_sources
        self.action_handlers = action_handlers

        # Attention parameters
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.exploration_weight = exploration_weight

        # Tracking
        self.current_focus = None
        self.visit_counts = {sensor: 0 for sensor in sensor_sources}
        self.salience_estimates = {sensor: 0.5 for sensor in sensor_sources}
        self.recent_rewards = {sensor: [] for sensor in sensor_sources}

        # History
        self.history = []
        self.cycle_count = 0
        self.running = False

    def _compute_salience(self, sensor_id: str, observation: Any) -> float:
        """
        Compute salience for a sensor.

        Simplified SNARC-like assessment:
        - Novelty: Less-visited = more novel
        - Reward: Expected value from history
        - Exploration bonus: Curiosity
        """
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

    def _select_focus(self, observations: Dict[str, Any]) -> str:
        """
        Select which sensor to focus on.

        Uses ε-greedy with salience decay and exploration bonus.
        """
        # Recompute salience for ALL sensors
        fresh_salience = {}
        for sensor_id, obs in observations.items():
            if obs is not None:  # Only consider sensors with observations
                salience = self._compute_salience(sensor_id, obs)

                # Decay current focus (boredom)
                if sensor_id == self.current_focus:
                    salience *= self.decay_rate

                fresh_salience[sensor_id] = salience

        if not fresh_salience:
            return None  # No observations available

        # ε-greedy selection
        if random.random() < self.epsilon:
            # EXPLORE: Random choice
            focus = random.choice(list(fresh_salience.keys()))
            exploration_type = "random"
        else:
            # EXPLOIT: Highest salience
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
            # No observations, skip cycle
            return

        # Select focus (with exploration)
        focus, exploration_type = self._select_focus(active_observations)

        if focus is None:
            return

        # Execute action
        handler = self.action_handlers[focus]
        observation = active_observations[focus]

        # Get simple stance (simplified - full SAGE would use SNARC)
        stance = CognitiveStance.CURIOUS_UNCERTAINTY

        result = handler(observation, stance)

        # Update tracking
        self.visit_counts[focus] += 1
        self.current_focus = focus

        # Store reward (keep last 5 for moving average)
        self.recent_rewards[focus].append(result.reward)
        if len(self.recent_rewards[focus]) > 5:
            self.recent_rewards[focus].pop(0)

        # Record history
        self.history.append({
            'cycle': self.cycle_count,
            'focus': focus,
            'exploration_type': exploration_type,
            'salience': self.salience_estimates[focus],
            'result': result,
            'all_salience': self.salience_estimates.copy()
        })

    def run(self, max_cycles: int = 100, cycle_delay: float = 0.1):
        """Run the attention loop"""
        self.running = True
        self.cycle_count = 0

        print(f"Starting attention-switching kernel...")
        print(f"  ε-greedy: {self.epsilon} exploration rate")
        print(f"  Decay: {self.decay_rate} focus decay")
        print(f"  Exploration weight: {self.exploration_weight}")
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

        # Exploration vs exploitation
        explore_count = sum(1 for h in self.history if h['exploration_type'] == 'random')
        exploit_count = len(self.history) - explore_count

        # Attention switches
        switches = 0
        for i in range(1, len(self.history)):
            if self.history[i]['focus'] != self.history[i-1]['focus']:
                switches += 1

        return {
            'total_cycles': len(self.history),
            'focus_distribution': focus_counts,
            'exploration_cycles': explore_count,
            'exploitation_cycles': exploit_count,
            'attention_switches': switches,
            'visit_counts': self.visit_counts.copy()
        }
