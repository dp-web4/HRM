"""
Attention Manager - Dynamic Resource Allocation

Implements Michaud's attention mechanism:
- "Wave of excitation" for selected targets
- "Inhibition" of non-selected targets
- Metabolic state-dependent allocation
- Automatic state transitions based on salience

Based on:
- Michaud (2019): "Attention is a wave of excitation leading to
  heightened awareness of part of our memories, accompanied by loss
  of awareness of other memories"
- Biological metabolic states (wake, sleep, REM, etc.)

Implementation Status: Production Ready
Author: Claude (Sonnet 4.5)
Date: 2025-11-20
"""

from enum import Enum
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta


class MetabolicState(Enum):
    """
    Metabolic states for SAGE consciousness.

    Each state has different resource allocation strategy and
    transition conditions.
    """
    WAKE = "wake"       # Normal distributed processing
    FOCUS = "focus"     # Intense concentration on single target
    REST = "rest"       # Low activity, memory consolidation
    DREAM = "dream"     # Random exploration, pattern discovery
    CRISIS = "crisis"   # Emergency response, all resources to threat


@dataclass
class StateTransition:
    """Record of metabolic state transition."""
    from_state: MetabolicState
    to_state: MetabolicState
    trigger: str
    timestamp: datetime
    salience_at_transition: float


class AttentionManager:
    """
    Manages dynamic ATP (Adaptive Trust Points) allocation.

    Allocation strategy varies by metabolic state:
    - FOCUS: 80% to primary, 15% to secondary, 5% background
    - WAKE: Distributed proportional to salience (with spreading)
    - REST: 70% consolidation, 30% monitoring
    - DREAM: Random allocation for exploration
    - CRISIS: 100% to highest-priority threat
    """

    def __init__(self, total_atp: float = 100.0, config: Optional[Dict] = None):
        self.total_atp = total_atp
        self.config = config or {}

        # Current state
        self.current_state = MetabolicState.WAKE
        self.state_entry_time = datetime.now()

        # Transition history
        self.transition_history: List[StateTransition] = []

        # State transition thresholds (configurable)
        self.thresholds = {
            'crisis_trigger': config.get('crisis_trigger_salience', 0.95),
            'focus_trigger': config.get('focus_trigger_salience', 0.8),
            'rest_trigger': config.get('rest_trigger_salience', 0.3),
            'focus_duration_max': config.get('focus_duration_max_seconds', 300),
            'wake_to_rest_duration': config.get('wake_to_rest_duration', 60),
            'rest_to_dream_duration': config.get('rest_to_dream_duration', 120),
            'dream_duration': config.get('dream_duration', 60),
            'dream_probability': config.get('dream_probability', 0.1)
        }

        # Allocation parameters
        self.wake_spread_factor = config.get('wake_spread_factor', 0.5)

    def allocate_attention(self,
                          salience_map: Dict[str, float],
                          force_state: Optional[MetabolicState] = None) -> Dict[str, float]:
        """
        Compute ATP allocation across targets.

        Args:
            salience_map: {target_id: salience_score (0-1)}
            force_state: Override automatic state (for testing)

        Returns:
            {target_id: atp_allocation}
        """
        # Update metabolic state based on conditions
        if force_state is None:
            self._update_metabolic_state(salience_map)
        else:
            self.current_state = force_state

        # Allocate based on current state
        if self.current_state == MetabolicState.FOCUS:
            return self._focus_allocation(salience_map)

        elif self.current_state == MetabolicState.WAKE:
            return self._wake_allocation(salience_map)

        elif self.current_state == MetabolicState.REST:
            return self._rest_allocation()

        elif self.current_state == MetabolicState.DREAM:
            return self._dream_allocation(salience_map)

        elif self.current_state == MetabolicState.CRISIS:
            return self._crisis_allocation(salience_map)

        else:
            # Default to WAKE
            return self._wake_allocation(salience_map)

    def _focus_allocation(self, salience_map: Dict[str, float]) -> Dict[str, float]:
        """
        FOCUS: Narrow attention - 80% primary, 15% secondary, 5% rest.

        Implements Michaud's "localized wave of overexcitement" with
        inhibition of everything else.
        """
        if not salience_map:
            return {}

        # Sort by salience descending
        sorted_targets = sorted(salience_map.items(),
                              key=lambda x: x[1],
                              reverse=True)

        allocation = {}

        # Primary target gets 80%
        if len(sorted_targets) >= 1:
            allocation[sorted_targets[0][0]] = 0.8 * self.total_atp

        # Secondary target gets 15%
        if len(sorted_targets) >= 2:
            allocation[sorted_targets[1][0]] = 0.15 * self.total_atp

        # Remaining targets split 5%
        if len(sorted_targets) > 2:
            remaining_atp = 0.05 * self.total_atp
            per_target = remaining_atp / (len(sorted_targets) - 2)
            for target_id, _ in sorted_targets[2:]:
                allocation[target_id] = per_target

        return allocation

    def _wake_allocation(self, salience_map: Dict[str, float]) -> Dict[str, float]:
        """
        WAKE: Distributed attention proportional to salience (with spreading).

        Normal waking consciousness - attentive but not hyper-focused.
        Spreading factor prevents over-concentration.
        """
        if not salience_map:
            return {}

        total_salience = sum(salience_map.values())

        if total_salience == 0:
            # Equal distribution if all zero salience
            per_target = self.total_atp / len(salience_map)
            return {tid: per_target for tid in salience_map.keys()}

        # Distribute proportionally with spreading
        # spread_factor = 0: pure proportional
        # spread_factor = 1: equal distribution
        allocation = {}

        for target_id, salience in salience_map.items():
            proportional = (salience / total_salience) * self.total_atp
            equal = self.total_atp / len(salience_map)

            allocation[target_id] = (
                (1 - self.wake_spread_factor) * proportional +
                self.wake_spread_factor * equal
            )

        return allocation

    def _rest_allocation(self) -> Dict[str, float]:
        """
        REST: Minimal monitoring, focus on consolidation.

        Like sleep - reduced sensory processing, memory consolidation active.
        """
        return {
            'memory_consolidation': 0.7 * self.total_atp,
            'minimal_monitoring': 0.3 * self.total_atp
        }

    def _dream_allocation(self, salience_map: Dict[str, float]) -> Dict[str, float]:
        """
        DREAM: Random exploration for pattern discovery.

        Like REM sleep - random activation creates novel connections.
        Biased slightly toward recent high-salience areas.
        """
        if not salience_map:
            return {}

        allocation = {}
        remaining_atp = self.total_atp

        for target_id in salience_map.keys():
            # Random amount between 0 and half of remaining
            amount = np.random.uniform(0, remaining_atp / 2)
            allocation[target_id] = amount
            remaining_atp -= amount

        # Distribute any remaining ATP
        if remaining_atp > 0:
            per_target = remaining_atp / len(salience_map)
            for target_id in allocation.keys():
                allocation[target_id] += per_target

        return allocation

    def _crisis_allocation(self, salience_map: Dict[str, float]) -> Dict[str, float]:
        """
        CRISIS: All resources to highest-priority threat.

        Fight-or-flight response - maximum resources to survival.
        """
        if not salience_map:
            return {}

        # ALL ATP to highest salience target
        highest_salience_target = max(salience_map.items(),
                                     key=lambda x: x[1])[0]

        return {highest_salience_target: self.total_atp}

    def _update_metabolic_state(self, salience_map: Dict[str, float]):
        """
        Determine and execute metabolic state transitions.

        Transitions based on:
        - Current salience levels
        - Time in current state
        - Random exploration triggers
        """
        if not salience_map:
            max_salience = 0.0
        else:
            max_salience = max(salience_map.values())

        time_in_state = (datetime.now() - self.state_entry_time).total_seconds()
        current = self.current_state
        new_state = current  # Default: no transition

        # === CRISIS TRANSITIONS ===
        # Any very high salience triggers CRISIS
        if max_salience > self.thresholds['crisis_trigger']:
            new_state = MetabolicState.CRISIS
            trigger = f"High salience: {max_salience:.3f}"

        # CRISIS → FOCUS (threat subsiding)
        elif current == MetabolicState.CRISIS and max_salience < 0.8:
            new_state = MetabolicState.FOCUS
            trigger = "Threat subsiding"

        # === FOCUS TRANSITIONS ===
        # FOCUS → WAKE (task complete or timeout)
        elif current == MetabolicState.FOCUS:
            if max_salience < 0.6:
                new_state = MetabolicState.WAKE
                trigger = "Task completion"
            elif time_in_state > self.thresholds['focus_duration_max']:
                new_state = MetabolicState.WAKE
                trigger = "Focus timeout"

        # === WAKE TRANSITIONS ===
        # WAKE → FOCUS (high sustained salience)
        elif current == MetabolicState.WAKE and max_salience > self.thresholds['focus_trigger']:
            new_state = MetabolicState.FOCUS
            trigger = f"High salience: {max_salience:.3f}"

        # WAKE → REST (low sustained salience)
        elif current == MetabolicState.WAKE:
            if (max_salience < self.thresholds['rest_trigger'] and
                time_in_state > self.thresholds['wake_to_rest_duration']):
                new_state = MetabolicState.REST
                trigger = "Low salience timeout"

        # === REST TRANSITIONS ===
        # REST → DREAM (random exploration trigger)
        elif current == MetabolicState.REST:
            if time_in_state > self.thresholds['rest_to_dream_duration']:
                if np.random.random() < self.thresholds['dream_probability']:
                    new_state = MetabolicState.DREAM
                    trigger = "Random exploration"

        # REST → WAKE (salience increase)
        elif current == MetabolicState.REST and max_salience > 0.5:
            new_state = MetabolicState.WAKE
            trigger = "Salience increase"

        # === DREAM TRANSITIONS ===
        # DREAM → WAKE (salience detected)
        elif current == MetabolicState.DREAM and max_salience > 0.4:
            new_state = MetabolicState.WAKE
            trigger = "Salience detected during dream"

        # DREAM → REST (exploration period complete)
        elif current == MetabolicState.DREAM and time_in_state > self.thresholds['dream_duration']:
            new_state = MetabolicState.REST
            trigger = "Dream cycle complete"

        # Execute transition if state changed
        if new_state != current:
            self._transition_to(new_state, trigger, max_salience)

    def _transition_to(self, new_state: MetabolicState, trigger: str, salience: float):
        """
        Execute state transition and record it.
        """
        old_state = self.current_state

        # Record transition
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            trigger=trigger,
            timestamp=datetime.now(),
            salience_at_transition=salience
        )
        self.transition_history.append(transition)

        # Update state
        self.current_state = new_state
        self.state_entry_time = datetime.now()

    def get_state(self) -> MetabolicState:
        """Get current metabolic state."""
        return self.current_state

    def get_time_in_state(self) -> float:
        """Get seconds in current state."""
        return (datetime.now() - self.state_entry_time).total_seconds()

    def get_transition_history(self, last_n: Optional[int] = None) -> List[StateTransition]:
        """
        Get state transition history.

        Args:
            last_n: Return only last N transitions (default: all)

        Returns:
            List of StateTransition objects
        """
        if last_n is None:
            return self.transition_history
        return self.transition_history[-last_n:]

    def get_stats(self) -> Dict[str, any]:
        """Get attention manager statistics."""
        # Compute time spent in each state
        state_durations = {state: 0.0 for state in MetabolicState}

        for i, trans in enumerate(self.transition_history):
            if i < len(self.transition_history) - 1:
                next_trans = self.transition_history[i + 1]
                duration = (next_trans.timestamp - trans.timestamp).total_seconds()
            else:
                # Current state
                duration = (datetime.now() - trans.timestamp).total_seconds()

            state_durations[trans.to_state] += duration

        return {
            'current_state': self.current_state.value,
            'time_in_current_state': self.get_time_in_state(),
            'total_transitions': len(self.transition_history),
            'state_durations': {
                state.value: duration
                for state, duration in state_durations.items()
            },
            'total_atp': self.total_atp
        }

    def __repr__(self) -> str:
        return (f"AttentionManager(state={self.current_state.value}, "
                f"time_in_state={self.get_time_in_state():.1f}s, "
                f"total_atp={self.total_atp})")
