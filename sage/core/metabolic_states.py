#!/usr/bin/env python3
"""
Metabolic State Management System - Session 40

Implements biological-inspired metabolic states for consciousness regulation:
- WAKE: Normal processing, balanced resource allocation
- FOCUS: High attention, increased ATP to salient tasks
- REST: Reduced processing, energy conservation, pattern consolidation
- DREAM: Offline processing, memory integration, creative associations
- CRISIS: Emergency mode, maximum resources to critical tasks

Integrates with:
- Session 27-29: Quality metrics and adaptive weighting
- Session 30-31: Epistemic states and meta-cognition
- Session 39: Epistemic calibration

Biological inspiration:
- Sleep-wake cycles regulate energy and consolidation
- Attention focuses limited resources on salient stimuli
- Crisis response reallocates resources for survival

Author: Thor (Autonomous Session 40)
Date: 2025-12-12
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class MetabolicState(Enum):
    """
    Metabolic states for consciousness regulation.

    Inspired by biological sleep-wake cycles and attention regulation.
    """
    WAKE = "wake"           # Normal processing, balanced allocation
    FOCUS = "focus"         # High attention, increased ATP to salient tasks
    REST = "rest"           # Reduced processing, consolidation
    DREAM = "dream"         # Offline integration, creative associations
    CRISIS = "crisis"       # Emergency mode, maximum resources


@dataclass
class ATPAllocation:
    """
    ATP (Adaptive Temporal Parameters) resource allocation.

    Represents computational resources available for different processes.
    Inspired by cellular ATP but adapted for consciousness architecture.

    Attributes:
        total_atp: Total available ATP budget (arbitrary units)
        allocated: Dict mapping process names to allocated ATP
        reserved: ATP reserved for critical processes
        available: Remaining unallocated ATP
    """
    total_atp: float = 100.0
    allocated: Dict[str, float] = field(default_factory=dict)
    reserved: float = 20.0  # Always keep some in reserve

    @property
    def available(self) -> float:
        """Calculate available unallocated ATP"""
        used = sum(self.allocated.values())
        return max(0.0, self.total_atp - used - self.reserved)

    def allocate(self, process: str, amount: float) -> bool:
        """
        Allocate ATP to a process.

        Args:
            process: Process name
            amount: ATP to allocate

        Returns:
            True if allocation successful, False if insufficient ATP
        """
        if amount > self.available:
            return False

        self.allocated[process] = self.allocated.get(process, 0.0) + amount
        return True

    def release(self, process: str, amount: Optional[float] = None):
        """
        Release ATP from a process.

        Args:
            process: Process name
            amount: ATP to release (None = release all)
        """
        if process not in self.allocated:
            return

        if amount is None:
            del self.allocated[process]
        else:
            self.allocated[process] = max(0.0, self.allocated[process] - amount)
            if self.allocated[process] == 0:
                del self.allocated[process]

    def get_allocation(self, process: str) -> float:
        """Get current allocation for a process"""
        return self.allocated.get(process, 0.0)


@dataclass
class AttentionFocus:
    """
    Represents attentional focus on salient stimuli.

    Attributes:
        target: What is being attended to
        salience: Salience score (0.0-1.0)
        atp_multiplier: Resource multiplier for focused task
        duration: How long focus has been maintained (seconds)
        start_time: When focus began
    """
    target: str
    salience: float
    atp_multiplier: float = 1.5
    duration: float = 0.0
    start_time: float = field(default_factory=time.time)

    def update_duration(self):
        """Update focus duration"""
        self.duration = time.time() - self.start_time


@dataclass
class MetabolicTransition:
    """
    Records metabolic state transitions for analysis.

    Attributes:
        from_state: Previous state
        to_state: New state
        trigger: What caused the transition
        timestamp: When transition occurred
        duration_in_previous: Time spent in previous state (seconds)
    """
    from_state: MetabolicState
    to_state: MetabolicState
    trigger: str
    timestamp: float = field(default_factory=time.time)
    duration_in_previous: float = 0.0


class MetabolicStateManager:
    """
    Manages metabolic states and resource allocation for consciousness.

    Implements dynamic state transitions based on:
    - Task demands (salience, complexity)
    - Resource availability (ATP levels)
    - Temporal patterns (duration in state)
    - Crisis detection (errors, frustration)

    Integrates with:
    - Quality metrics (Session 27)
    - Epistemic states (Session 30-31)
    - Temporal adaptation (Session 26-29)
    """

    def __init__(self,
                 initial_atp: float = 100.0,
                 focus_threshold: float = 0.7,
                 rest_after_cycles: int = 100):
        """
        Initialize metabolic state manager.

        Args:
            initial_atp: Starting ATP budget
            focus_threshold: Salience threshold for FOCUS state
            rest_after_cycles: Cycles before suggesting REST
        """
        self.current_state = MetabolicState.WAKE
        self.atp = ATPAllocation(total_atp=initial_atp)
        self.focus: Optional[AttentionFocus] = None

        # State management
        self.state_entry_time = time.time()
        self.cycles_in_state = 0
        self.total_cycles = 0

        # Thresholds
        self.focus_threshold = focus_threshold
        self.rest_after_cycles = rest_after_cycles
        self.crisis_error_threshold = 3  # Consecutive errors

        # History
        self.transitions: List[MetabolicTransition] = []
        self.consecutive_errors = 0

        # State-specific ATP budgets
        self.state_atp_budgets = {
            MetabolicState.WAKE: 100.0,
            MetabolicState.FOCUS: 120.0,   # Extra resources for focused work
            MetabolicState.REST: 60.0,     # Reduced during rest
            MetabolicState.DREAM: 80.0,    # Moderate for offline processing
            MetabolicState.CRISIS: 150.0   # Maximum resources
        }

    def get_state_duration(self) -> float:
        """Get duration in current state (seconds)"""
        return time.time() - self.state_entry_time

    def transition_to(self, new_state: MetabolicState, trigger: str):
        """
        Transition to new metabolic state.

        Args:
            new_state: Target state
            trigger: What caused the transition
        """
        if new_state == self.current_state:
            return  # No transition needed

        # Record transition
        duration = self.get_state_duration()
        transition = MetabolicTransition(
            from_state=self.current_state,
            to_state=new_state,
            trigger=trigger,
            duration_in_previous=duration
        )
        self.transitions.append(transition)

        # Update state
        old_state = self.current_state
        self.current_state = new_state
        self.state_entry_time = time.time()
        self.cycles_in_state = 0

        # Adjust ATP budget for new state
        new_budget = self.state_atp_budgets[new_state]
        self.atp.total_atp = new_budget

        # State-specific actions
        if new_state == MetabolicState.FOCUS and self.focus is None:
            # Initialize focus if entering FOCUS state without existing focus
            self.focus = AttentionFocus(
                target="high_salience_task",
                salience=self.focus_threshold
            )
        elif new_state != MetabolicState.FOCUS:
            # Clear focus when leaving FOCUS state
            self.focus = None

        print(f"[Metabolic] {old_state.value} → {new_state.value} (trigger: {trigger})")

    def set_attention(self, target: str, salience: float):
        """
        Set attentional focus on salient stimulus.

        Args:
            target: What to attend to
            salience: Salience score (0.0-1.0)
        """
        self.focus = AttentionFocus(target=target, salience=salience)

        # Transition to FOCUS if salience exceeds threshold and not in CRISIS
        if (salience >= self.focus_threshold and
            self.current_state not in [MetabolicState.FOCUS, MetabolicState.CRISIS]):
            self.transition_to(MetabolicState.FOCUS,
                             f"high_salience({salience:.2f})")

    def report_error(self):
        """Report an error/failure for crisis detection"""
        self.consecutive_errors += 1

        if self.consecutive_errors >= self.crisis_error_threshold:
            self.transition_to(MetabolicState.CRISIS,
                             f"consecutive_errors({self.consecutive_errors})")

    def report_success(self):
        """Report success (resets error counter)"""
        self.consecutive_errors = 0

        # If in CRISIS, consider returning to WAKE
        if self.current_state == MetabolicState.CRISIS:
            self.transition_to(MetabolicState.WAKE, "crisis_resolved")

    def cycle_update(self,
                    task_salience: float = 0.5,
                    epistemic_frustration: float = 0.0):
        """
        Update metabolic state based on current cycle.

        Called each consciousness cycle to evaluate state transitions.

        Args:
            task_salience: Current task salience (0.0-1.0)
            epistemic_frustration: Current frustration level (0.0-1.0)
        """
        self.cycles_in_state += 1
        self.total_cycles += 1

        # Update focus duration if active
        if self.focus is not None:
            self.focus.update_duration()

        # State transition logic
        current = self.current_state

        if current == MetabolicState.WAKE:
            # WAKE → FOCUS if high salience
            if task_salience >= self.focus_threshold:
                self.transition_to(MetabolicState.FOCUS,
                                 f"high_salience({task_salience:.2f})")
            # WAKE → REST if sustained operation
            elif self.total_cycles > 0 and self.total_cycles % self.rest_after_cycles == 0:
                self.transition_to(MetabolicState.REST,
                                 f"sustained_operation({self.total_cycles})")
            # WAKE → CRISIS if high frustration
            elif epistemic_frustration > 0.7:
                self.transition_to(MetabolicState.CRISIS,
                                 f"high_frustration({epistemic_frustration:.2f})")

        elif current == MetabolicState.FOCUS:
            # FOCUS → WAKE if salience drops
            if task_salience < self.focus_threshold * 0.5:
                self.transition_to(MetabolicState.WAKE,
                                 f"low_salience({task_salience:.2f})")
            # FOCUS → CRISIS if high frustration despite focus
            elif epistemic_frustration > 0.7:
                self.transition_to(MetabolicState.CRISIS,
                                 f"frustrated_focus({epistemic_frustration:.2f})")
            # FOCUS → REST if prolonged (fatigue)
            elif self.get_state_duration() > 300:  # 5 minutes
                self.transition_to(MetabolicState.REST,
                                 "focus_fatigue")

        elif current == MetabolicState.REST:
            # REST → WAKE after some cycles
            if self.cycles_in_state >= 10:
                self.transition_to(MetabolicState.WAKE, "rest_complete")
            # REST → CRISIS if urgent task appears
            elif task_salience > 0.9:
                self.transition_to(MetabolicState.CRISIS,
                                 f"urgent_task({task_salience:.2f})")

        elif current == MetabolicState.DREAM:
            # DREAM → WAKE after integration period
            if self.cycles_in_state >= 20:
                self.transition_to(MetabolicState.WAKE, "dream_complete")

        elif current == MetabolicState.CRISIS:
            # CRISIS → WAKE if resolved (handled by report_success)
            # CRISIS → FOCUS if stabilizing
            if epistemic_frustration < 0.5 and self.cycles_in_state >= 5:
                self.transition_to(MetabolicState.FOCUS, "crisis_stabilizing")

    def get_atp_multiplier(self, process: str) -> float:
        """
        Get ATP multiplier for a process based on current state and focus.

        Args:
            process: Process name

        Returns:
            Multiplier for ATP allocation (1.0 = normal)
        """
        base_multiplier = 1.0

        # State-based multipliers
        if self.current_state == MetabolicState.FOCUS:
            base_multiplier = 1.5
        elif self.current_state == MetabolicState.REST:
            base_multiplier = 0.6
        elif self.current_state == MetabolicState.DREAM:
            base_multiplier = 0.8
        elif self.current_state == MetabolicState.CRISIS:
            base_multiplier = 2.0

        # Focus-based multiplier (if process matches focus target)
        if self.focus is not None and process == self.focus.target:
            base_multiplier *= self.focus.atp_multiplier

        return base_multiplier

    def get_state_statistics(self) -> Dict:
        """
        Get statistics about metabolic state usage.

        Returns:
            Dict with state duration stats and transition counts
        """
        if not self.transitions:
            return {
                'total_transitions': 0,
                'state_durations': {},
                'transition_triggers': {}
            }

        # Calculate time in each state
        state_durations = {state: 0.0 for state in MetabolicState}
        for transition in self.transitions:
            state_durations[transition.from_state] += transition.duration_in_previous

        # Add current state duration
        state_durations[self.current_state] += self.get_state_duration()

        # Count transition triggers
        trigger_counts = {}
        for transition in self.transitions:
            trigger_counts[transition.trigger] = trigger_counts.get(transition.trigger, 0) + 1

        return {
            'total_transitions': len(self.transitions),
            'state_durations': {s.value: d for s, d in state_durations.items()},
            'transition_triggers': trigger_counts,
            'current_state': self.current_state.value,
            'cycles_in_state': self.cycles_in_state,
            'total_cycles': self.total_cycles
        }


def example_usage():
    """Example demonstrating metabolic state management"""
    manager = MetabolicStateManager(
        initial_atp=100.0,
        focus_threshold=0.7,
        rest_after_cycles=50
    )

    print("Metabolic State Management Demo")
    print("=" * 50)
    print()

    # Simulate some cycles
    for cycle in range(100):
        # Varying task salience
        if 10 <= cycle < 30:
            salience = 0.8  # High salience task
            frustration = 0.2
        elif 30 <= cycle < 35:
            salience = 0.8
            frustration = 0.8  # Frustration builds
        elif 60 <= cycle < 65:
            salience = 0.9  # Urgent task during rest
            frustration = 0.3
        else:
            salience = 0.5  # Normal
            frustration = 0.1

        # Update state
        manager.cycle_update(task_salience=salience,
                           epistemic_frustration=frustration)

        # Show state changes
        if cycle in [0, 10, 30, 35, 50, 60, 65, 99]:
            print(f"Cycle {cycle}: State={manager.current_state.value}, "
                  f"ATP={manager.atp.total_atp:.0f}, "
                  f"Salience={salience:.1f}, Frustration={frustration:.1f}")

    print()
    print("Statistics:")
    stats = manager.get_state_statistics()
    print(f"Total transitions: {stats['total_transitions']}")
    print(f"Total cycles: {stats['total_cycles']}")
    print()
    print("Time in each state:")
    for state, duration in stats['state_durations'].items():
        print(f"  {state}: {duration:.1f}s")
    print()
    print("Transition triggers:")
    for trigger, count in stats['transition_triggers'].items():
        print(f"  {trigger}: {count}")


if __name__ == '__main__':
    example_usage()
