#!/usr/bin/env python3
"""
Circadian Clock for SAGE

Provides temporal context for:
- Metabolic state biasing (sleep/wake cycles)
- Context-dependent trust modulation (day/night sensor reliability)
- Temporal expectations for SNARC (surprise/novelty relative to time)

Design Philosophy:
- Time is synthetic (not wall-clock) - runs at simulation speed
- 1 cycle = 1 "minute" (configurable)
- Period = 100 cycles = 1 "day" (tunable for testing)
- Phases: day (0-60), night (60-100)

This enables SAGE to:
- Anticipate rather than just react
- Persist in states during appropriate phases
- Modulate sensor trust by temporal context
- Schedule consolidation during "sleep" periods
"""

import math
from enum import Enum
from typing import Dict, Callable, Optional
from dataclasses import dataclass


class CircadianPhase(Enum):
    """Circadian phase of the day"""
    DAWN = "dawn"       # Transition to day (0-10)
    DAY = "day"         # Active period (10-50)
    DUSK = "dusk"       # Transition to night (50-60)
    NIGHT = "night"     # Rest period (60-90)
    DEEP_NIGHT = "deep_night"  # Deep rest (90-100)


@dataclass
class CircadianContext:
    """Complete temporal context"""
    cycle: int                    # Current cycle number
    phase: CircadianPhase         # Current phase
    time_in_phase: float          # Progress through phase [0, 1]
    phase_progression: float      # Overall day progression [0, 1]
    is_day: bool                  # Daytime flag
    is_night: bool                # Nighttime flag
    day_strength: float           # How "day-like" (0=night, 1=peak day)
    night_strength: float         # How "night-like" (0=day, 1=peak night)


class CircadianClock:
    """
    Circadian clock providing temporal context for SAGE

    Manages synthetic time with configurable period and phase structure.
    Provides smooth transitions between phases and context-dependent biases.
    """

    def __init__(
        self,
        period_cycles: int = 100,        # Cycles per "day"
        day_ratio: float = 0.6,          # Fraction of period that is "day"
        cycles_per_minute: int = 1,      # Time scaling factor
        start_cycle: int = 0             # Starting cycle (0 = dawn)
    ):
        """
        Initialize circadian clock

        Args:
            period_cycles: Full circadian period in cycles
            day_ratio: Fraction of period considered "day" (rest is night)
            cycles_per_minute: Simulation time scaling
            start_cycle: Initial cycle offset (for starting at different times)
        """
        self.period_cycles = period_cycles
        self.day_ratio = day_ratio
        self.cycles_per_minute = cycles_per_minute
        self.current_cycle = start_cycle

        # Phase boundaries (in cycles)
        self.dawn_start = 0
        self.dawn_end = int(0.1 * period_cycles)
        self.day_start = self.dawn_end
        self.day_end = int(0.5 * period_cycles)
        self.dusk_start = self.day_end
        self.dusk_end = int(0.6 * period_cycles)
        self.night_start = self.dusk_end
        self.night_end = int(0.9 * period_cycles)
        self.deep_night_start = self.night_end
        self.deep_night_end = period_cycles

    def tick(self) -> CircadianContext:
        """
        Advance clock by one cycle and return current context

        Returns:
            CircadianContext with all temporal information
        """
        self.current_cycle += 1
        return self.get_context()

    def get_context(self) -> CircadianContext:
        """Get current circadian context without advancing"""
        # Normalize to period
        phase_cycle = self.current_cycle % self.period_cycles
        phase_progression = phase_cycle / self.period_cycles

        # Determine phase
        if self.dawn_start <= phase_cycle < self.dawn_end:
            phase = CircadianPhase.DAWN
            phase_start = self.dawn_start
            phase_end = self.dawn_end
        elif self.day_start <= phase_cycle < self.day_end:
            phase = CircadianPhase.DAY
            phase_start = self.day_start
            phase_end = self.day_end
        elif self.dusk_start <= phase_cycle < self.dusk_end:
            phase = CircadianPhase.DUSK
            phase_start = self.dusk_start
            phase_end = self.dusk_end
        elif self.night_start <= phase_cycle < self.night_end:
            phase = CircadianPhase.NIGHT
            phase_start = self.night_start
            phase_end = self.night_end
        else:  # deep_night
            phase = CircadianPhase.DEEP_NIGHT
            phase_start = self.deep_night_start
            phase_end = self.deep_night_end

        # Progress through current phase
        time_in_phase = (phase_cycle - phase_start) / (phase_end - phase_start)

        # Day/night flags
        is_day = phase in [CircadianPhase.DAWN, CircadianPhase.DAY, CircadianPhase.DUSK]
        is_night = not is_day

        # Smooth strength curves (sinusoidal for natural transitions)
        # Day strength peaks at midday (phase_progression ≈ 0.3)
        # Night strength peaks at midnight (phase_progression ≈ 0.75)
        day_strength = max(0.0, math.cos(2 * math.pi * (phase_progression - 0.25)))
        night_strength = max(0.0, math.cos(2 * math.pi * (phase_progression - 0.75)))

        return CircadianContext(
            cycle=self.current_cycle,
            phase=phase,
            time_in_phase=time_in_phase,
            phase_progression=phase_progression,
            is_day=is_day,
            is_night=is_night,
            day_strength=day_strength,
            night_strength=night_strength
        )

    def get_metabolic_bias(self, state_name: str) -> float:
        """
        Get circadian bias for metabolic state

        Returns multiplier for state transition thresholds:
        - > 1.0: state is favored (easier to enter/stay in)
        - < 1.0: state is disfavored (harder to enter)
        - 1.0: neutral (no circadian influence)

        Args:
            state_name: 'wake', 'focus', 'rest', 'dream', 'crisis'

        Returns:
            Bias multiplier [0.5, 2.0]
        """
        context = self.get_context()

        if state_name == 'wake':
            # Favored during day, neutral at night
            return 1.0 + 0.5 * context.day_strength

        elif state_name == 'focus':
            # Strongly favored during peak day
            return 1.0 + 1.0 * context.day_strength

        elif state_name == 'rest':
            # Always available (emergency recovery)
            return 1.0

        elif state_name == 'dream':
            # Strongly favored during night (consolidation time)
            return 1.0 + 2.0 * context.night_strength

        elif state_name == 'crisis':
            # Circadian-independent (survival mode)
            return 1.0

        else:
            return 1.0

    def get_trust_modifier(self, sensor_type: str) -> float:
        """
        Get circadian trust modifier for sensor type

        Modulates base trust based on time-of-day appropriateness:
        - Visual sensors less reliable at night
        - Audio sensors more reliable at night (less ambient noise)
        - Proprioceptive sensors unaffected

        Args:
            sensor_type: 'camera', 'lidar', 'microphone', 'imu', etc.

        Returns:
            Trust multiplier [0.3, 1.5]
        """
        context = self.get_context()

        if sensor_type in ['camera', 'visual', 'rgb']:
            # Vision degraded at night
            return 1.0 if context.is_day else 0.3

        elif sensor_type in ['lidar', 'depth', 'radar']:
            # Active sensors less affected by lighting
            return 1.0 if context.is_day else 0.7

        elif sensor_type in ['microphone', 'audio']:
            # Audio actually better at night (less ambient noise)
            return 0.8 if context.is_day else 1.2

        elif sensor_type in ['imu', 'proprioception', 'gyro', 'accelerometer']:
            # Internal sensors unaffected by time
            return 1.0

        elif sensor_type in ['gps', 'compass']:
            # Position sensors work anytime
            return 1.0

        else:
            # Unknown sensor: assume slight night penalty
            return 1.0 if context.is_day else 0.8

    def get_temporal_expectation(self, observation_type: str) -> float:
        """
        Get expected intensity/frequency for observation type at current time

        Used by SNARC to modulate surprise:
        - High expectation → lower surprise for matching observation
        - Low expectation → higher surprise for matching observation

        Args:
            observation_type: 'brightness', 'movement', 'sound_level', etc.

        Returns:
            Expected intensity [0, 1]
        """
        context = self.get_context()

        if observation_type == 'brightness':
            # Expect high brightness during day
            return context.day_strength

        elif observation_type == 'movement':
            # Expect more movement during day
            return 0.3 + 0.7 * context.day_strength

        elif observation_type == 'sound_level':
            # Expect moderate sound during day, quiet at night
            return 0.4 + 0.6 * context.day_strength

        elif observation_type == 'visual_change':
            # Visual changes more expected during day
            return 0.2 + 0.8 * context.day_strength

        else:
            # Default: moderate expectation
            return 0.5

    def should_consolidate_memory(self) -> bool:
        """Check if current time is appropriate for memory consolidation"""
        context = self.get_context()
        # Consolidation during night and deep night phases
        return context.phase in [CircadianPhase.NIGHT, CircadianPhase.DEEP_NIGHT]

    def get_attention_capacity_modifier(self) -> float:
        """
        Get multiplier for attention capacity based on time of day

        During day: higher capacity (can handle more concurrent tasks)
        During night: lower capacity (more focused/filtered)

        Returns:
            Capacity multiplier [0.5, 1.5]
        """
        context = self.get_context()
        # Higher capacity during day
        return 0.5 + 0.5 * (1.0 + context.day_strength)

    def __repr__(self) -> str:
        ctx = self.get_context()
        return (f"CircadianClock(cycle={ctx.cycle}, phase={ctx.phase.value}, "
                f"day_strength={ctx.day_strength:.2f}, night_strength={ctx.night_strength:.2f})")


# Utility functions for common patterns

def create_day_night_clock(period: int = 100) -> CircadianClock:
    """Create standard day/night clock"""
    return CircadianClock(period_cycles=period, day_ratio=0.6)


def create_always_day_clock() -> CircadianClock:
    """Create clock locked to day (for testing)"""
    clock = CircadianClock(period_cycles=1000, day_ratio=1.0)
    return clock


def create_always_night_clock() -> CircadianClock:
    """Create clock locked to night (for testing)"""
    clock = CircadianClock(period_cycles=1000, day_ratio=0.0, start_cycle=750)
    return clock


if __name__ == "__main__":
    # Test circadian clock
    print("="*80)
    print("Circadian Clock Test")
    print("="*80)
    print()

    clock = create_day_night_clock(period=100)

    print("Simulating 1 full day (100 cycles):")
    print()

    # Sample key moments
    test_cycles = [0, 5, 25, 45, 55, 65, 75, 85, 95]

    for target_cycle in test_cycles:
        # Advance to target
        while clock.current_cycle < target_cycle:
            clock.tick()

        ctx = clock.get_context()

        print(f"Cycle {ctx.cycle:3d} | {ctx.phase.value:10s} | "
              f"Day:{ctx.day_strength:4.2f} Night:{ctx.night_strength:4.2f} | "
              f"Camera trust:{clock.get_trust_modifier('camera'):.2f} | "
              f"Dream bias:{clock.get_metabolic_bias('dream'):.2f}")

    print()
    print("Temporal Expectations:")
    clock.current_cycle = 30  # Midday
    ctx = clock.get_context()
    print(f"\nAt cycle 30 (midday):")
    print(f"  Brightness expectation: {clock.get_temporal_expectation('brightness'):.2f}")
    print(f"  Movement expectation: {clock.get_temporal_expectation('movement'):.2f}")

    clock.current_cycle = 80  # Night
    ctx = clock.get_context()
    print(f"\nAt cycle 80 (night):")
    print(f"  Brightness expectation: {clock.get_temporal_expectation('brightness'):.2f}")
    print(f"  Movement expectation: {clock.get_temporal_expectation('movement'):.2f}")
    print(f"  Should consolidate memory: {clock.should_consolidate_memory()}")

    print()
    print("="*80)
    print("Circadian Clock Working")
    print("="*80)
