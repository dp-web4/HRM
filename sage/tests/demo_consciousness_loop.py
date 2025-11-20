"""
Demo: SAGE Unified Consciousness Loop

Demonstrates the complete consciousness system running continuously with:
- Metabolic state transitions (WAKE → FOCUS → REST → DREAM → CRISIS)
- ATP budget management
- Sensor observation and salience computation
- Plugin selection and execution
- Memory system updates
- Circadian rhythm modulation

This shows all components working together as designed.
"""

import asyncio
import sys
from pathlib import Path

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from core.sage_consciousness import SAGEConsciousness


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


async def demo_basic_consciousness(cycles: int = 100):
    """
    Demo 1: Basic consciousness loop with metabolic state transitions.

    Shows:
    - Continuous operation
    - State transitions based on ATP and salience
    - Memory system updates
    - ATP recovery in REST state
    """
    print_header("Demo 1: Basic Consciousness Loop")

    print("Running SAGE for", cycles, "cycles...")
    print("Watch for metabolic state transitions:\n")

    sage = SAGEConsciousness(
        initial_atp=100.0,
        enable_circadian=True,
        simulation_mode=True
    )

    await sage.run(max_cycles=cycles)

    # Print detailed memory stats
    print_header("Memory System Analysis")

    print(f"SNARC Memory ({len(sage.snarc_memory)} entries):")
    if sage.snarc_memory:
        print("  Sample salient experiences:")
        for i, entry in enumerate(sage.snarc_memory[:5]):
            print(f"    Cycle {entry['cycle']:3d}: "
                  f"Plugin={entry['plugin']:8s} "
                  f"Salience={entry['salience']:.3f}")

    print(f"\nIRP Pattern Library ({len(sage.irp_memory)} patterns):")
    if sage.irp_memory:
        print("  Sample convergence patterns:")
        for i, entry in enumerate(sage.irp_memory[:5]):
            trust = entry['trust']
            print(f"    Cycle {entry['cycle']:3d}: "
                  f"Plugin={entry['plugin']:8s} "
                  f"Monotonicity={trust.get('monotonicity_ratio', 0):.2f}")

    print(f"\nCircular Buffer: {len(sage.circular_buffer)} recent events")

    print(f"\nDream Consolidations: {len(sage.verbatim_storage)} records")

    # Print trust weights
    print_header("Learned Plugin Trust Weights")
    for plugin, trust in sorted(sage.plugin_trust_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {plugin:15s}: {trust:.3f}")

    return sage


async def demo_state_transitions():
    """
    Demo 2: Focus on metabolic state transitions.

    Shows:
    - WAKE → FOCUS (high salience)
    - FOCUS → WAKE (salience drops)
    - WAKE → REST (low ATP)
    - REST → WAKE (ATP recovers)
    - WAKE → DREAM (moderate ATP, time passed)
    - DREAM → WAKE (consolidation complete)
    """
    print_header("Demo 2: Metabolic State Transition Patterns")

    print("Running longer session to observe all state transitions...")
    print("This may take a minute...\n")

    sage = SAGEConsciousness(
        initial_atp=100.0,
        enable_circadian=True,
        simulation_mode=True
    )

    # Track state transitions
    state_history = []

    # Run and track states
    sage.running = True
    for cycle in range(200):
        await sage.step()

        state_history.append({
            'cycle': cycle,
            'state': sage.metabolic.current_state.value,
            'atp': sage.metabolic.atp_current
        })

        # Print every 20 cycles
        if cycle % 20 == 0:
            print(f"[Cycle {cycle:3d}] "
                  f"State: {sage.metabolic.current_state.value:8s} "
                  f"ATP: {sage.metabolic.atp_current:5.1f}")

    sage.running = False
    sage._print_summary()

    # Analyze state transitions
    print_header("State Transition Analysis")

    states_visited = set(h['state'] for h in state_history)
    print(f"States visited: {', '.join(sorted(states_visited))}")

    # Count time in each state
    state_durations = {}
    for h in state_history:
        state = h['state']
        state_durations[state] = state_durations.get(state, 0) + 1

    print("\nTime in each state (cycles):")
    for state, cycles in sorted(state_durations.items(), key=lambda x: x[1], reverse=True):
        pct = (cycles / len(state_history)) * 100
        print(f"  {state:8s}: {cycles:3d} cycles ({pct:5.1f}%)")

    return sage


async def demo_circadian_modulation():
    """
    Demo 3: Circadian rhythm effects on state transitions.

    Shows:
    - Day bias toward WAKE/FOCUS
    - Night bias toward DREAM
    - DAWN and DUSK transitions
    """
    print_header("Demo 3: Circadian Rhythm Modulation")

    print("Running full circadian cycle (100 cycles = 1 day)...")
    print("Observe how state preferences change with time of day:\n")

    sage = SAGEConsciousness(
        initial_atp=100.0,
        enable_circadian=True,
        simulation_mode=True
    )

    # Track circadian phases
    phase_history = []

    sage.running = True
    for cycle in range(100):
        await sage.step()

        if sage.metabolic.circadian_clock:
            phase = sage.metabolic.circadian_clock.current_phase.value
        else:
            phase = "unknown"

        phase_history.append({
            'cycle': cycle,
            'phase': phase,
            'state': sage.metabolic.current_state.value,
            'atp': sage.metabolic.atp_current
        })

        # Print every 10 cycles
        if cycle % 10 == 0:
            print(f"[Cycle {cycle:3d}] "
                  f"Phase: {phase:11s} "
                  f"State: {sage.metabolic.current_state.value:8s} "
                  f"ATP: {sage.metabolic.atp_current:5.1f}")

    sage.running = False
    sage._print_summary()

    # Analyze circadian effects
    print_header("Circadian Analysis")

    # Group by phase
    phase_states = {}
    for h in phase_history:
        phase = h['phase']
        state = h['state']
        if phase not in phase_states:
            phase_states[phase] = {}
        phase_states[phase][state] = phase_states[phase].get(state, 0) + 1

    print("States by circadian phase:")
    for phase in ['dawn', 'day', 'dusk', 'night', 'deep_night']:
        if phase in phase_states:
            print(f"\n  {phase.upper()}:")
            total = sum(phase_states[phase].values())
            for state, count in sorted(phase_states[phase].items(), key=lambda x: x[1], reverse=True):
                pct = (count / total) * 100
                print(f"    {state:8s}: {count:2d} cycles ({pct:4.1f}%)")

    return sage


async def demo_memory_consolidation():
    """
    Demo 4: Memory consolidation during DREAM state.

    Shows:
    - Verbatim storage only happens in DREAM
    - Memory consolidation process
    - Pattern extraction from experiences
    """
    print_header("Demo 4: Memory Consolidation During Dreams")

    print("Running session with focus on dream state...")
    print("Verbatim storage only activates during DREAM:\n")

    # Start with moderate ATP to encourage dreaming
    sage = SAGEConsciousness(
        initial_atp=60.0,  # Moderate ATP
        enable_circadian=True,
        simulation_mode=True
    )

    sage.running = True
    dream_cycles = 0

    for cycle in range(150):
        await sage.step()

        # Track dream cycles
        if sage.metabolic.current_state.value == 'dream':
            dream_cycles += 1

        # Print every 15 cycles
        if cycle % 15 == 0:
            print(f"[Cycle {cycle:3d}] "
                  f"State: {sage.metabolic.current_state.value:8s} "
                  f"Dreams: {len(sage.verbatim_storage):2d}")

    sage.running = False
    sage._print_summary()

    print_header("Dream Consolidation Analysis")

    print(f"Total dream cycles: {dream_cycles}")
    print(f"Verbatim records stored: {len(sage.verbatim_storage)}")

    if sage.verbatim_storage:
        print("\nSample dream consolidations:")
        for i, record in enumerate(sage.verbatim_storage[:10]):
            print(f"  Cycle {record['cycle']:3d}: "
                  f"State={record['state']:8s} "
                  f"Plugin={record['plugin']:8s}")

    print(f"\nComparison:")
    print(f"  Total experiences: {len(sage.circular_buffer)}")
    print(f"  Salient (SNARC): {len(sage.snarc_memory)}")
    print(f"  Good patterns (IRP): {len(sage.irp_memory)}")
    print(f"  Dream consolidations: {len(sage.verbatim_storage)}")
    print(f"\nConsolidation rate: {len(sage.verbatim_storage) / max(1, dream_cycles):.2f} per dream cycle")

    return sage


async def main():
    """Run all demos"""
    print_header("SAGE Unified Consciousness Loop - Comprehensive Demo")

    print("This demo shows the complete consciousness system with:")
    print("  • Continuous operation")
    print("  • Metabolic state transitions")
    print("  • ATP budget management")
    print("  • Sensor observation and salience")
    print("  • Plugin selection and execution")
    print("  • Memory system updates")
    print("  • Circadian rhythm modulation")
    print("\nPress Ctrl+C at any time to skip to next demo\n")

    input("Press Enter to start Demo 1 (Basic Consciousness)...")
    try:
        await demo_basic_consciousness(cycles=100)
    except KeyboardInterrupt:
        print("\n[Skipped]")

    input("\nPress Enter to start Demo 2 (State Transitions)...")
    try:
        await demo_state_transitions()
    except KeyboardInterrupt:
        print("\n[Skipped]")

    input("\nPress Enter to start Demo 3 (Circadian Rhythm)...")
    try:
        await demo_circadian_modulation()
    except KeyboardInterrupt:
        print("\n[Skipped]")

    input("\nPress Enter to start Demo 4 (Memory Consolidation)...")
    try:
        await demo_memory_consolidation()
    except KeyboardInterrupt:
        print("\n[Skipped]")

    print_header("All Demos Complete!")

    print("Key Takeaways:")
    print("  ✓ SAGE consciousness loop runs continuously")
    print("  ✓ Metabolic states transition based on ATP and salience")
    print("  ✓ Memory systems capture different aspects of experience")
    print("  ✓ Circadian rhythms modulate state preferences")
    print("  ✓ Dreams enable memory consolidation")
    print("\nNext steps:")
    print("  • Integrate real sensors (vision, audio, proprioception)")
    print("  • Implement real SNARC salience computation")
    print("  • Add effector system for actions")
    print("  • Deploy on Jetson for edge validation")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
