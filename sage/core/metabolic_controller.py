#!/usr/bin/env python3
"""
Metabolic State Controller for SAGE

Manages transitions between metabolic states based on energy levels,
attention load, and environmental conditions.

States:
- WAKE: Alert, processing sensory input, moderate ATP consumption
- FOCUS: High attention on specific task, high ATP consumption
- REST: Low processing, ATP recovery, no new learning
- DREAM: Memory consolidation, pattern extraction, moderate ATP
- CRISIS: Emergency mode, minimal processing, survival only

Design Philosophy:
- State transitions driven by ATP levels and attention demands
- Each state has different resource allocation policies
- Smooth transitions preserve system stability
- Learning happens in WAKE/FOCUS, consolidation in DREAM
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Callable
import time
from pathlib import Path
import sys

# Add sage to path for circadian clock import
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from core.circadian_clock import CircadianClock, create_day_night_clock


class MetabolicState(Enum):
    """Five metabolic states for SAGE"""
    WAKE = "wake"         # Normal operation, balanced processing
    FOCUS = "focus"       # High attention, resource intensive
    REST = "rest"         # Low activity, ATP recovery
    DREAM = "dream"       # Offline consolidation, pattern extraction
    CRISIS = "crisis"     # Emergency mode, survival only


@dataclass
class StateConfig:
    """Configuration for a metabolic state"""
    name: str
    atp_consumption_rate: float  # ATP per cycle
    atp_recovery_rate: float     # ATP recovery per cycle
    max_active_plugins: int      # How many IRP plugins can run
    sensor_poll_rate: float      # Hz for sensor polling
    learning_enabled: bool       # Can update weights/trust
    consolidation_enabled: bool  # Can run memory consolidation
    min_atp_threshold: float     # Minimum ATP to stay in this state
    max_atp_threshold: float     # Maximum ATP for this state


class MetabolicController:
    """
    Controls metabolic state transitions for SAGE

    Monitors:
    - ATP levels (energy budget)
    - Attention load (how many sensors demanding attention)
    - Task complexity (from SNARC salience)
    - Time in current state

    Decides:
    - When to transition states
    - Resource allocation policies
    - Plugin loading/unloading
    """

    def __init__(
        self,
        initial_atp: float = 100.0,
        max_atp: float = 100.0,
        device: Optional[str] = None,
        circadian_period: int = 100,
        enable_circadian: bool = True
    ):
        """
        Initialize metabolic controller

        Args:
            initial_atp: Starting ATP budget
            max_atp: Maximum ATP capacity
            device: Compute device
            circadian_period: Cycles per day (default 100)
            enable_circadian: Enable circadian rhythm biasing
        """
        self.current_state = MetabolicState.WAKE
        self.previous_state = None
        self.state_entry_time = time.time()

        self.atp_current = initial_atp
        self.atp_max = max_atp

        # Circadian rhythm integration
        self.enable_circadian = enable_circadian
        if enable_circadian:
            self.circadian_clock = create_day_night_clock(period=circadian_period)
        else:
            self.circadian_clock = None

        # State configurations
        self.state_configs = {
            MetabolicState.WAKE: StateConfig(
                name="wake",
                atp_consumption_rate=0.5,
                atp_recovery_rate=0.1,
                max_active_plugins=3,
                sensor_poll_rate=30.0,
                learning_enabled=True,
                consolidation_enabled=False,
                min_atp_threshold=30.0,
                max_atp_threshold=100.0
            ),
            MetabolicState.FOCUS: StateConfig(
                name="focus",
                atp_consumption_rate=2.0,
                atp_recovery_rate=0.0,
                max_active_plugins=1,  # Focus on one thing
                sensor_poll_rate=60.0,  # Higher rate for focused task
                learning_enabled=True,
                consolidation_enabled=False,
                min_atp_threshold=20.0,
                max_atp_threshold=100.0
            ),
            MetabolicState.REST: StateConfig(
                name="rest",
                atp_consumption_rate=0.1,
                atp_recovery_rate=1.0,  # Fast recovery
                max_active_plugins=0,
                sensor_poll_rate=1.0,   # Minimal monitoring
                learning_enabled=False,
                consolidation_enabled=False,
                min_atp_threshold=0.0,
                max_atp_threshold=50.0
            ),
            MetabolicState.DREAM: StateConfig(
                name="dream",
                atp_consumption_rate=0.3,
                atp_recovery_rate=0.5,
                max_active_plugins=0,   # No real-time processing
                sensor_poll_rate=0.1,   # Nearly off
                learning_enabled=False,
                consolidation_enabled=True,  # Memory consolidation
                min_atp_threshold=40.0,
                max_atp_threshold=80.0
            ),
            MetabolicState.CRISIS: StateConfig(
                name="crisis",
                atp_consumption_rate=0.05,
                atp_recovery_rate=0.2,
                max_active_plugins=1,   # Only critical systems
                sensor_poll_rate=5.0,   # Monitor for danger
                learning_enabled=False,
                consolidation_enabled=False,
                min_atp_threshold=0.0,
                max_atp_threshold=20.0
            )
        }

        # Transition history
        self.state_history = []
        self.transition_callbacks = {}

    def get_current_config(self) -> StateConfig:
        """Get configuration for current state"""
        return self.state_configs[self.current_state]

    def update(self, cycle_data: Dict) -> MetabolicState:
        """
        Update metabolic state based on system conditions

        Args:
            cycle_data: Dict containing:
                - atp_consumed: ATP used this cycle
                - attention_load: Number of salient sensors
                - max_salience: Highest salience score
                - crisis_detected: Emergency condition

        Returns:
            New metabolic state (may be same as current)
        """
        # Extract cycle data
        atp_consumed = cycle_data.get('atp_consumed', 0.0)
        attention_load = cycle_data.get('attention_load', 0)
        max_salience = cycle_data.get('max_salience', 0.0)
        crisis_detected = cycle_data.get('crisis_detected', False)

        # Update ATP
        config = self.get_current_config()
        self.atp_current -= atp_consumed
        self.atp_current += config.atp_recovery_rate
        self.atp_current = max(0.0, min(self.atp_max, self.atp_current))

        # Determine new state
        new_state = self._determine_next_state(
            attention_load=attention_load,
            max_salience=max_salience,
            crisis_detected=crisis_detected
        )

        # Transition if needed
        if new_state != self.current_state:
            self._transition_to(new_state)

        return self.current_state

    def _determine_next_state(
        self,
        attention_load: int,
        max_salience: float,
        crisis_detected: bool
    ) -> MetabolicState:
        """Determine next state based on conditions with circadian biasing"""

        # Advance circadian clock and get biases
        if self.circadian_clock:
            circadian_ctx = self.circadian_clock.tick()
            wake_bias = self.circadian_clock.get_metabolic_bias('wake')
            focus_bias = self.circadian_clock.get_metabolic_bias('focus')
            dream_bias = self.circadian_clock.get_metabolic_bias('dream')
        else:
            wake_bias = focus_bias = dream_bias = 1.0

        # Crisis overrides everything
        if crisis_detected or self.atp_current < 10.0:
            return MetabolicState.CRISIS

        # Current state config
        config = self.get_current_config()
        time_in_state = time.time() - self.state_entry_time

        # State-specific transition logic with circadian modulation
        if self.current_state == MetabolicState.WAKE:
            # WAKE → FOCUS: High salience with sufficient ATP
            # Focus threshold lowered during day (easier to focus)
            focus_threshold = 50.0 / focus_bias
            if max_salience > 0.8 and self.atp_current > focus_threshold:
                return MetabolicState.FOCUS

            # WAKE → REST: Low ATP
            # Rest threshold raised at night (easier to rest)
            rest_threshold = 30.0 * wake_bias
            if self.atp_current < rest_threshold:
                return MetabolicState.REST

            # WAKE → DREAM: Moderate ATP, been awake long enough
            # Dream heavily biased toward night
            dream_time_threshold = max(5, 300 / dream_bias)  # Shorter wait at night
            if 40.0 < self.atp_current < 80.0 and time_in_state > dream_time_threshold:
                return MetabolicState.DREAM

            return MetabolicState.WAKE

        elif self.current_state == MetabolicState.FOCUS:
            # FOCUS → WAKE: Salience dropped or ATP low
            if max_salience < 0.5 or self.atp_current < 20.0:
                return MetabolicState.WAKE

            # FOCUS → REST: ATP critical
            if self.atp_current < 15.0:
                return MetabolicState.REST

            return MetabolicState.FOCUS

        elif self.current_state == MetabolicState.REST:
            # REST → WAKE: ATP recovered
            # Wake threshold raised at night (harder to wake up)
            wake_threshold = 50.0 * wake_bias
            if self.atp_current > wake_threshold:
                return MetabolicState.WAKE

            # REST → DREAM: ATP partially recovered, time to consolidate
            # Dream strongly preferred at night
            dream_atp_threshold = 40.0 / dream_bias
            dream_time_threshold = max(5, 60 / dream_bias)
            if self.atp_current > dream_atp_threshold and time_in_state > dream_time_threshold:
                return MetabolicState.DREAM

            return MetabolicState.REST

        elif self.current_state == MetabolicState.DREAM:
            # DREAM → WAKE: ATP recovered, consolidation complete
            # Harder to leave dream at night
            wake_threshold = 70.0 * wake_bias
            max_dream_time = 180 / dream_bias  # Longer dreams at night
            if self.atp_current > wake_threshold or time_in_state > max_dream_time:
                return MetabolicState.WAKE

            # DREAM → REST: ATP dropped during consolidation
            if self.atp_current < 40.0:
                return MetabolicState.REST

            return MetabolicState.DREAM

        elif self.current_state == MetabolicState.CRISIS:
            # CRISIS → REST: Immediate danger passed, start recovery
            if self.atp_current > 15.0 and not crisis_detected:
                return MetabolicState.REST

            return MetabolicState.CRISIS

        return self.current_state

    def _transition_to(self, new_state: MetabolicState):
        """Execute state transition"""
        old_state = self.current_state

        # Record transition
        self.state_history.append({
            'from': old_state,
            'to': new_state,
            'timestamp': time.time(),
            'atp_at_transition': self.atp_current
        })

        # Execute transition callbacks
        if new_state in self.transition_callbacks:
            for callback in self.transition_callbacks[new_state]:
                callback(old_state, new_state, self.atp_current)

        # Update state
        self.previous_state = old_state
        self.current_state = new_state
        self.state_entry_time = time.time()

    def register_transition_callback(
        self,
        target_state: MetabolicState,
        callback: Callable
    ):
        """Register callback for state transitions"""
        if target_state not in self.transition_callbacks:
            self.transition_callbacks[target_state] = []
        self.transition_callbacks[target_state].append(callback)

    def force_state(self, state: MetabolicState):
        """Force transition to specific state (for testing/debugging)"""
        if state != self.current_state:
            self._transition_to(state)

    def get_stats(self) -> Dict:
        """Get controller statistics"""
        stats = {
            'current_state': self.current_state.value,
            'atp_current': self.atp_current,
            'atp_max': self.atp_max,
            'atp_percentage': (self.atp_current / self.atp_max) * 100,
            'time_in_state': time.time() - self.state_entry_time,
            'config': self.get_current_config(),
            'transition_count': len(self.state_history)
        }

        # Add circadian info if enabled
        if self.circadian_clock:
            ctx = self.circadian_clock.get_context()
            stats['circadian'] = {
                'cycle': ctx.cycle,
                'phase': ctx.phase.value,
                'is_day': ctx.is_day,
                'day_strength': ctx.day_strength,
                'night_strength': ctx.night_strength
            }

        return stats

    def should_consolidate(self) -> bool:
        """Check if memory consolidation should run"""
        config_allows = self.get_current_config().consolidation_enabled

        # Also check circadian timing - consolidation preferred at night
        if self.circadian_clock:
            circadian_appropriate = self.circadian_clock.should_consolidate_memory()
            return config_allows and circadian_appropriate

        return config_allows

    def should_learn(self) -> bool:
        """Check if learning/trust updates should occur"""
        return self.get_current_config().learning_enabled

    def get_max_active_plugins(self) -> int:
        """Get maximum plugins allowed in current state"""
        return self.get_current_config().max_active_plugins

    def get_sensor_poll_rate(self) -> float:
        """Get sensor polling rate for current state"""
        return self.get_current_config().sensor_poll_rate
