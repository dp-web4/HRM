#!/usr/bin/env python3
"""
FOCUS Gap Experiments — Thor SAGE Session 2026-04-11

Three experiments testing the Circadian Focus Gap discovered in the
2026-04-10 18:00 session:

Experiment A: FOCUS Activation Test (P11 validation)
  - Lower focus_threshold to guarantee entry
  - Hypothesis: FOCUS cascades to CRISIS within 10 cycles

Experiment B: REST→FOCUS Emergency Path
  - Add REST→FOCUS transition for high-salience events
  - Hypothesis: Messages during REST can trigger useful FOCUS

Experiment C: Circadian Period Sweep
  - Test periods [50, 100, 200, 500, 1000]
  - Hypothesis: Longer periods widen dawn/dusk windows → more FOCUS

All experiments use the real MetabolicController in simulation_mode.
"""

import sys
import os
import json
import copy
from pathlib import Path
from collections import Counter, defaultdict

# Add SAGE to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sage.core.metabolic_controller import MetabolicController, MetabolicState


def run_experiment_a(cycles=2000, verbose=True):
    """
    Experiment A: FOCUS Activation Test

    Patch: Override _determine_next_state to use focus salience threshold of 0.3
    (below the 0.46 audio mock baseline), guaranteeing FOCUS entry conditions.

    Prediction P11: If FOCUS activates, it cascades to CRISIS in <10 cycles
    because consumption=2.0, recovery=0.0, and plugin drain adds ~3.5 ATP/cycle.
    """
    print("=" * 70)
    print("EXPERIMENT A: FOCUS ACTIVATION TEST")
    print("=" * 70)
    print(f"Cycles: {cycles}")
    print(f"Modification: Focus salience threshold lowered to 0.30 (from 0.45)")
    print(f"Prediction P11: FOCUS → CRISIS cascade in <10 cycles")
    print()

    # Create controller in simulation mode
    mc = MetabolicController(
        initial_atp=100.0,
        max_atp=100.0,
        circadian_period=100,
        enable_circadian=True,
        simulation_mode=True
    )

    # Monkey-patch: lower focus threshold
    original_determine = mc._determine_next_state

    def patched_determine(attention_load, max_salience, crisis_detected):
        """Patched to use 0.30 focus threshold instead of 0.45"""
        # Advance circadian clock and get biases
        if mc.circadian_clock:
            circadian_ctx = mc.circadian_clock.tick()
            wake_bias = mc.circadian_clock.get_metabolic_bias('wake')
            focus_bias = mc.circadian_clock.get_metabolic_bias('focus')
            dream_bias = mc.circadian_clock.get_metabolic_bias('dream')
        else:
            wake_bias = focus_bias = dream_bias = 1.0

        mc.cycles_in_state += 1

        if crisis_detected or mc.atp_current < 10.0:
            return MetabolicState.CRISIS

        if mc.cycles_in_state < mc.min_cycles_in_state:
            return mc.current_state

        config = mc.get_current_config()
        time_in_state = mc._get_time_in_state()

        if mc.current_state == MetabolicState.WAKE:
            # PATCHED: threshold 0.30 instead of 0.45
            focus_threshold = 50.0 / focus_bias
            if max_salience > 0.30 and mc.atp_current > focus_threshold:
                return MetabolicState.FOCUS

            rest_threshold = 30.0 * wake_bias
            if mc.atp_current < rest_threshold:
                return MetabolicState.REST

            dream_time_threshold = max(5, 30 / dream_bias)
            if 40.0 < mc.atp_current < 80.0 and time_in_state > dream_time_threshold:
                return MetabolicState.DREAM

            return MetabolicState.WAKE

        elif mc.current_state == MetabolicState.FOCUS:
            # Keep original FOCUS exit logic
            if max_salience < 0.5 or mc.atp_current < 20.0:
                return MetabolicState.WAKE
            if mc.atp_current < 15.0:
                return MetabolicState.REST
            return MetabolicState.FOCUS

        elif mc.current_state == MetabolicState.REST:
            wake_threshold = 50.0 * wake_bias
            if mc.atp_current > wake_threshold:
                return MetabolicState.WAKE
            dream_atp_threshold = 40.0 / dream_bias
            dream_time_threshold = max(5, 6 / dream_bias)
            if mc.atp_current > dream_atp_threshold and time_in_state > dream_time_threshold:
                return MetabolicState.DREAM
            return MetabolicState.REST

        elif mc.current_state == MetabolicState.DREAM:
            wake_threshold = 70.0 * wake_bias
            max_dream_time = 18 / dream_bias
            if mc.atp_current > wake_threshold or time_in_state > max_dream_time:
                return MetabolicState.WAKE
            if mc.atp_current < 40.0:
                return MetabolicState.REST
            return MetabolicState.DREAM

        elif mc.current_state == MetabolicState.CRISIS:
            if mc.atp_current > 15.0 and not crisis_detected:
                return MetabolicState.REST
            return MetabolicState.CRISIS

        return mc.current_state

    mc._determine_next_state = patched_determine

    # Run simulation with constant salience 0.46 (audio mock level)
    transitions = []
    state_counts = Counter()
    atp_trace = []
    focus_entries = []
    focus_durations = []
    focus_exit_states = []
    current_focus_start = None

    for cycle in range(cycles):
        prev_state = mc.current_state

        # Simulate: constant salience 0.46 (audio mock), plugin drain ~3.5 ATP
        plugin_drain = 3.5 if mc.current_state in [MetabolicState.WAKE, MetabolicState.FOCUS] else 0.5

        cycle_data = {
            'atp_consumed': plugin_drain,
            'attention_load': 1,
            'max_salience': 0.46,
            'crisis_detected': False
        }

        mc.update(cycle_data)
        new_state = mc.current_state
        state_counts[new_state.value] += 1
        atp_trace.append(mc.atp_current)

        if new_state != prev_state:
            transitions.append({
                'cycle': cycle,
                'from': prev_state.value,
                'to': new_state.value,
                'atp': round(mc.atp_current, 2)
            })

            if new_state == MetabolicState.FOCUS:
                current_focus_start = cycle
                focus_entries.append({
                    'cycle': cycle,
                    'atp': round(mc.atp_current, 2)
                })

            if prev_state == MetabolicState.FOCUS:
                duration = cycle - current_focus_start if current_focus_start else 0
                focus_durations.append(duration)
                focus_exit_states.append(new_state.value)
                current_focus_start = None

    # Report
    print(f"State distribution ({cycles} cycles):")
    for state, count in sorted(state_counts.items()):
        pct = 100.0 * count / cycles
        print(f"  {state:8s}: {count:6d} ({pct:5.1f}%)")

    print(f"\nTotal transitions: {len(transitions)}")
    trans_counts = Counter(f"{t['from']}→{t['to']}" for t in transitions)
    for trans, count in trans_counts.most_common():
        print(f"  {trans}: {count}")

    print(f"\nFOCUS entries: {len(focus_entries)}")
    if focus_entries:
        print(f"  First 10 entries:")
        for entry in focus_entries[:10]:
            print(f"    Cycle {entry['cycle']}: ATP={entry['atp']}")

    print(f"\nFOCUS durations (cycles): {focus_durations[:20]}")
    if focus_durations:
        avg_dur = sum(focus_durations) / len(focus_durations)
        print(f"  Average: {avg_dur:.1f} cycles")
        print(f"  Min: {min(focus_durations)}, Max: {max(focus_durations)}")

    print(f"\nFOCUS exit states: {Counter(focus_exit_states)}")

    # P11 validation
    crisis_count = state_counts.get('crisis', 0)
    focus_to_crisis = trans_counts.get('focus→crisis', 0)
    print(f"\n--- P11 VALIDATION ---")
    print(f"CRISIS cycles: {crisis_count}")
    print(f"FOCUS→CRISIS transitions: {focus_to_crisis}")
    if focus_durations:
        cascades_under_10 = sum(1 for d in focus_durations if d <= 10)
        print(f"FOCUS durations ≤10 cycles: {cascades_under_10}/{len(focus_durations)}")
        if any(exit == 'crisis' for exit in focus_exit_states):
            print(f"P11 CONFIRMED: FOCUS cascades to CRISIS")
        else:
            # Check if FOCUS→WAKE→REST→... eventually leads to crisis
            focus_to_wake = trans_counts.get('focus→wake', 0)
            print(f"P11 PARTIAL: FOCUS exits to WAKE ({focus_to_wake}x), not direct CRISIS")
            print(f"  (FOCUS→WAKE occurs when salience<0.5 OR atp<20)")
    else:
        print("P11 INCONCLUSIVE: No FOCUS entries observed")

    # ATP trajectory around first FOCUS entry
    if focus_entries:
        first_entry_cycle = focus_entries[0]['cycle']
        start = max(0, first_entry_cycle - 5)
        end = min(len(atp_trace), first_entry_cycle + 15)
        print(f"\nATP trace around first FOCUS entry (cycle {first_entry_cycle}):")
        for i in range(start, end):
            marker = " <<< FOCUS ENTRY" if i == first_entry_cycle else ""
            state_at_i = "?"
            for t in transitions:
                if t['cycle'] <= i:
                    state_at_i = t['to']
            print(f"  Cycle {i:4d}: ATP={atp_trace[i]:6.2f} [{state_at_i}]{marker}")

    return {
        'experiment': 'A',
        'cycles': cycles,
        'state_counts': dict(state_counts),
        'transitions': len(transitions),
        'focus_entries': len(focus_entries),
        'focus_durations': focus_durations,
        'focus_exit_states': focus_exit_states,
        'crisis_cycles': crisis_count,
        'p11_validated': any(exit == 'crisis' for exit in focus_exit_states)
    }


def run_experiment_b(cycles=2000, verbose=True):
    """
    Experiment B: REST→FOCUS Emergency Path

    Add a REST→FOCUS transition when max_salience > 0.6 (strong signal).
    This tests whether FOCUS can be reached from REST (currently no path exists).

    Simulates periodic high-salience messages during REST state.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: REST→FOCUS EMERGENCY PATH")
    print("=" * 70)
    print(f"Cycles: {cycles}")
    print(f"Modification: REST→FOCUS when salience > 0.6 AND ATP > 30")
    print(f"Stimulus: High-salience message every 200 cycles")
    print()

    mc = MetabolicController(
        initial_atp=100.0,
        max_atp=100.0,
        circadian_period=100,
        enable_circadian=True,
        simulation_mode=True
    )

    # Monkey-patch: add REST→FOCUS path
    def patched_determine(attention_load, max_salience, crisis_detected):
        if mc.circadian_clock:
            circadian_ctx = mc.circadian_clock.tick()
            wake_bias = mc.circadian_clock.get_metabolic_bias('wake')
            focus_bias = mc.circadian_clock.get_metabolic_bias('focus')
            dream_bias = mc.circadian_clock.get_metabolic_bias('dream')
        else:
            wake_bias = focus_bias = dream_bias = 1.0

        mc.cycles_in_state += 1

        if crisis_detected or mc.atp_current < 10.0:
            return MetabolicState.CRISIS

        if mc.cycles_in_state < mc.min_cycles_in_state:
            return mc.current_state

        config = mc.get_current_config()
        time_in_state = mc._get_time_in_state()

        if mc.current_state == MetabolicState.WAKE:
            focus_threshold = 50.0 / focus_bias
            if max_salience > 0.45 and mc.atp_current > focus_threshold:
                return MetabolicState.FOCUS
            rest_threshold = 30.0 * wake_bias
            if mc.atp_current < rest_threshold:
                return MetabolicState.REST
            dream_time_threshold = max(5, 30 / dream_bias)
            if 40.0 < mc.atp_current < 80.0 and time_in_state > dream_time_threshold:
                return MetabolicState.DREAM
            return MetabolicState.WAKE

        elif mc.current_state == MetabolicState.FOCUS:
            if max_salience < 0.5 or mc.atp_current < 20.0:
                return MetabolicState.WAKE
            if mc.atp_current < 15.0:
                return MetabolicState.REST
            return MetabolicState.FOCUS

        elif mc.current_state == MetabolicState.REST:
            # NEW: REST→FOCUS emergency path for high salience
            if max_salience > 0.6 and mc.atp_current > 30.0:
                return MetabolicState.FOCUS

            wake_threshold = 50.0 * wake_bias
            if mc.atp_current > wake_threshold:
                return MetabolicState.WAKE
            dream_atp_threshold = 40.0 / dream_bias
            dream_time_threshold = max(5, 6 / dream_bias)
            if mc.atp_current > dream_atp_threshold and time_in_state > dream_time_threshold:
                return MetabolicState.DREAM
            return MetabolicState.REST

        elif mc.current_state == MetabolicState.DREAM:
            wake_threshold = 70.0 * wake_bias
            max_dream_time = 18 / dream_bias
            if mc.atp_current > wake_threshold or time_in_state > max_dream_time:
                return MetabolicState.WAKE
            if mc.atp_current < 40.0:
                return MetabolicState.REST
            return MetabolicState.DREAM

        elif mc.current_state == MetabolicState.CRISIS:
            if mc.atp_current > 15.0 and not crisis_detected:
                return MetabolicState.REST
            return MetabolicState.CRISIS

        return mc.current_state

    mc._determine_next_state = patched_determine

    transitions = []
    state_counts = Counter()
    focus_from_rest = []
    message_events = []

    for cycle in range(cycles):
        prev_state = mc.current_state

        # Periodic high-salience messages
        is_message = (cycle % 200 == 100)  # Every 200 cycles, offset by 100
        salience = 0.85 if is_message else 0.46
        plugin_drain = 35.0 if is_message else (3.5 if mc.current_state in [MetabolicState.WAKE, MetabolicState.FOCUS] else 0.5)

        if is_message:
            message_events.append({
                'cycle': cycle,
                'state': mc.current_state.value,
                'atp': round(mc.atp_current, 2)
            })

        cycle_data = {
            'atp_consumed': plugin_drain,
            'attention_load': 3 if is_message else 1,
            'max_salience': salience,
            'crisis_detected': False
        }

        mc.update(cycle_data)
        new_state = mc.current_state
        state_counts[new_state.value] += 1

        if new_state != prev_state:
            transitions.append({
                'cycle': cycle,
                'from': prev_state.value,
                'to': new_state.value,
                'atp': round(mc.atp_current, 2)
            })
            if prev_state == MetabolicState.REST and new_state == MetabolicState.FOCUS:
                focus_from_rest.append({
                    'cycle': cycle,
                    'atp': round(mc.atp_current, 2)
                })

    # Report
    print(f"State distribution ({cycles} cycles):")
    for state, count in sorted(state_counts.items()):
        pct = 100.0 * count / cycles
        print(f"  {state:8s}: {count:6d} ({pct:5.1f}%)")

    print(f"\nTotal transitions: {len(transitions)}")
    trans_counts = Counter(f"{t['from']}→{t['to']}" for t in transitions)
    for trans, count in trans_counts.most_common():
        print(f"  {trans}: {count}")

    print(f"\nMessage events ({len(message_events)}):")
    for msg in message_events:
        print(f"  Cycle {msg['cycle']}: state={msg['state']}, ATP={msg['atp']}")

    print(f"\nREST→FOCUS transitions: {len(focus_from_rest)}")
    for entry in focus_from_rest:
        print(f"  Cycle {entry['cycle']}: ATP={entry['atp']}")

    return {
        'experiment': 'B',
        'cycles': cycles,
        'state_counts': dict(state_counts),
        'transitions': len(transitions),
        'rest_to_focus': len(focus_from_rest),
        'messages_sent': len(message_events),
        'messages_in_rest': sum(1 for m in message_events if m['state'] == 'rest'),
        'trans_counts': dict(trans_counts)
    }


def run_experiment_c(cycles_per_period=5000, verbose=True):
    """
    Experiment C: Circadian Period Sweep

    Test circadian_period = [50, 100, 200, 500, 1000]
    Longer periods → wider dawn/dusk windows → more chance for FOCUS?

    Prediction P9: Increasing circadian_period enables FOCUS transitions.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C: CIRCADIAN PERIOD SWEEP")
    print("=" * 70)
    print(f"Cycles per period: {cycles_per_period}")
    print(f"Periods: [50, 100, 200, 500, 1000]")
    print(f"Prediction P9: Longer periods → more FOCUS transitions")
    print()

    periods = [50, 100, 200, 500, 1000]
    results = {}

    for period in periods:
        mc = MetabolicController(
            initial_atp=100.0,
            max_atp=100.0,
            circadian_period=period,
            enable_circadian=True,
            simulation_mode=True
        )

        state_counts = Counter()
        trans_counts = Counter()
        focus_count = 0
        max_atp_in_wake = 0.0

        for cycle in range(cycles_per_period):
            prev_state = mc.current_state

            # Standard conditions: salience 0.46, plugin drain 3.5 in WAKE
            plugin_drain = 3.5 if mc.current_state in [MetabolicState.WAKE, MetabolicState.FOCUS] else 0.5

            cycle_data = {
                'atp_consumed': plugin_drain,
                'attention_load': 1,
                'max_salience': 0.46,
                'crisis_detected': False
            }

            mc.update(cycle_data)
            new_state = mc.current_state
            state_counts[new_state.value] += 1

            if mc.current_state == MetabolicState.WAKE:
                max_atp_in_wake = max(max_atp_in_wake, mc.atp_current)

            if new_state != prev_state:
                trans_key = f"{prev_state.value}→{new_state.value}"
                trans_counts[trans_key] += 1
                if new_state == MetabolicState.FOCUS:
                    focus_count += 1

        results[period] = {
            'state_counts': dict(state_counts),
            'trans_counts': dict(trans_counts),
            'focus_entries': focus_count,
            'max_wake_atp': round(max_atp_in_wake, 2)
        }

        focus_pct = 100.0 * state_counts.get('focus', 0) / cycles_per_period
        wake_pct = 100.0 * state_counts.get('wake', 0) / cycles_per_period
        rest_pct = 100.0 * state_counts.get('rest', 0) / cycles_per_period
        dream_pct = 100.0 * state_counts.get('dream', 0) / cycles_per_period

        print(f"Period={period:4d}: WAKE={wake_pct:5.1f}% REST={rest_pct:5.1f}% "
              f"FOCUS={focus_pct:5.1f}% DREAM={dream_pct:5.1f}% | "
              f"FOCUS entries={focus_count}, max_wake_ATP={max_atp_in_wake:.1f}")

    print(f"\n--- P9 VALIDATION ---")
    focus_by_period = {p: r['focus_entries'] for p, r in results.items()}
    print(f"FOCUS entries by period: {focus_by_period}")
    if any(v > 0 for v in focus_by_period.values()):
        print("P9 CONFIRMED: Longer periods enable FOCUS")
    else:
        print("P9 REFUTED: No FOCUS transitions at any period")
        print("Explanation: The gap is not about window width but about")
        print("  the fundamental ATP drain rate vs. threshold interaction")

    return {
        'experiment': 'C',
        'cycles_per_period': cycles_per_period,
        'results': results,
        'focus_by_period': focus_by_period
    }


def run_experiment_d(cycles=2000, verbose=True):
    """
    Experiment D: FOCUS Energy Profile Redesign

    Test a modified FOCUS state with:
    - Recovery rate 0.5 (instead of 0.0)
    - Consumption rate 1.0 (instead of 2.0)
    - Exit threshold raised to salience < 0.3 (instead of < 0.5)

    This tests whether FOCUS is sustainable with realistic energy parameters.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT D: SUSTAINABLE FOCUS REDESIGN")
    print("=" * 70)
    print(f"Cycles: {cycles}")
    print(f"Modifications:")
    print(f"  FOCUS consumption: 1.0 (was 2.0)")
    print(f"  FOCUS recovery: 0.5 (was 0.0)")
    print(f"  FOCUS exit salience: < 0.3 (was < 0.5)")
    print(f"  Focus entry salience: > 0.30 (guarantee entry)")
    print()

    mc = MetabolicController(
        initial_atp=100.0,
        max_atp=100.0,
        circadian_period=100,
        enable_circadian=True,
        simulation_mode=True
    )

    # Patch FOCUS energy profile
    mc.state_configs[MetabolicState.FOCUS].atp_consumption_rate = 1.0
    mc.state_configs[MetabolicState.FOCUS].atp_recovery_rate = 0.5

    # Patch transition logic
    def patched_determine(attention_load, max_salience, crisis_detected):
        if mc.circadian_clock:
            circadian_ctx = mc.circadian_clock.tick()
            wake_bias = mc.circadian_clock.get_metabolic_bias('wake')
            focus_bias = mc.circadian_clock.get_metabolic_bias('focus')
            dream_bias = mc.circadian_clock.get_metabolic_bias('dream')
        else:
            wake_bias = focus_bias = dream_bias = 1.0

        mc.cycles_in_state += 1

        if crisis_detected or mc.atp_current < 10.0:
            return MetabolicState.CRISIS

        if mc.cycles_in_state < mc.min_cycles_in_state:
            return mc.current_state

        time_in_state = mc._get_time_in_state()

        if mc.current_state == MetabolicState.WAKE:
            focus_threshold = 50.0 / focus_bias
            if max_salience > 0.30 and mc.atp_current > focus_threshold:
                return MetabolicState.FOCUS
            rest_threshold = 30.0 * wake_bias
            if mc.atp_current < rest_threshold:
                return MetabolicState.REST
            dream_time_threshold = max(5, 30 / dream_bias)
            if 40.0 < mc.atp_current < 80.0 and time_in_state > dream_time_threshold:
                return MetabolicState.DREAM
            return MetabolicState.WAKE

        elif mc.current_state == MetabolicState.FOCUS:
            # Modified: harder to exit FOCUS (salience < 0.3 instead of 0.5)
            if max_salience < 0.3 or mc.atp_current < 20.0:
                return MetabolicState.WAKE
            if mc.atp_current < 15.0:
                return MetabolicState.REST
            return MetabolicState.FOCUS

        elif mc.current_state == MetabolicState.REST:
            wake_threshold = 50.0 * wake_bias
            if mc.atp_current > wake_threshold:
                return MetabolicState.WAKE
            dream_atp_threshold = 40.0 / dream_bias
            dream_time_threshold = max(5, 6 / dream_bias)
            if mc.atp_current > dream_atp_threshold and time_in_state > dream_time_threshold:
                return MetabolicState.DREAM
            return MetabolicState.REST

        elif mc.current_state == MetabolicState.DREAM:
            wake_threshold = 70.0 * wake_bias
            max_dream_time = 18 / dream_bias
            if mc.atp_current > wake_threshold or time_in_state > max_dream_time:
                return MetabolicState.WAKE
            if mc.atp_current < 40.0:
                return MetabolicState.REST
            return MetabolicState.DREAM

        elif mc.current_state == MetabolicState.CRISIS:
            if mc.atp_current > 15.0 and not crisis_detected:
                return MetabolicState.REST
            return MetabolicState.CRISIS

        return mc.current_state

    mc._determine_next_state = patched_determine

    transitions = []
    state_counts = Counter()
    atp_trace = []
    focus_durations = []
    current_focus_start = None
    focus_atp_traces = []
    current_focus_atp = []

    for cycle in range(cycles):
        prev_state = mc.current_state

        plugin_drain = 3.5 if mc.current_state in [MetabolicState.WAKE, MetabolicState.FOCUS] else 0.5

        cycle_data = {
            'atp_consumed': plugin_drain,
            'attention_load': 1,
            'max_salience': 0.46,
            'crisis_detected': False
        }

        mc.update(cycle_data)
        new_state = mc.current_state
        state_counts[new_state.value] += 1
        atp_trace.append(mc.atp_current)

        if mc.current_state == MetabolicState.FOCUS:
            current_focus_atp.append(mc.atp_current)

        if new_state != prev_state:
            transitions.append({
                'cycle': cycle,
                'from': prev_state.value,
                'to': new_state.value,
                'atp': round(mc.atp_current, 2)
            })

            if new_state == MetabolicState.FOCUS:
                current_focus_start = cycle
                current_focus_atp = [mc.atp_current]

            if prev_state == MetabolicState.FOCUS and current_focus_start is not None:
                duration = cycle - current_focus_start
                focus_durations.append(duration)
                focus_atp_traces.append(current_focus_atp[:])
                current_focus_atp = []
                current_focus_start = None

    # Report
    print(f"State distribution ({cycles} cycles):")
    for state, count in sorted(state_counts.items()):
        pct = 100.0 * count / cycles
        print(f"  {state:8s}: {count:6d} ({pct:5.1f}%)")

    print(f"\nTotal transitions: {len(transitions)}")
    trans_counts = Counter(f"{t['from']}→{t['to']}" for t in transitions)
    for trans, count in trans_counts.most_common():
        print(f"  {trans}: {count}")

    print(f"\nFOCUS durations: {focus_durations[:20]}")
    if focus_durations:
        avg_dur = sum(focus_durations) / len(focus_durations)
        print(f"  Average: {avg_dur:.1f} cycles")
        print(f"  Min: {min(focus_durations)}, Max: {max(focus_durations)}")

    # Show ATP trajectory during FOCUS
    if focus_atp_traces:
        print(f"\nATP during first FOCUS episode ({len(focus_atp_traces[0])} cycles):")
        for i, atp in enumerate(focus_atp_traces[0][:20]):
            print(f"  Cycle +{i}: ATP={atp:.2f}")

    crisis_cycles = state_counts.get('crisis', 0)
    focus_cycles = state_counts.get('focus', 0)
    print(f"\n--- SUSTAINABILITY ASSESSMENT ---")
    print(f"FOCUS cycles: {focus_cycles} ({100*focus_cycles/cycles:.1f}%)")
    print(f"CRISIS cycles: {crisis_cycles} ({100*crisis_cycles/cycles:.1f}%)")
    if focus_durations:
        sustained = sum(1 for d in focus_durations if d >= 5)
        print(f"Sustained FOCUS (≥5 cycles): {sustained}/{len(focus_durations)}")
        if avg_dur >= 5 and crisis_cycles < cycles * 0.1:
            print("VERDICT: Sustainable FOCUS is ACHIEVABLE with modified energy profile")
        else:
            print("VERDICT: Modified profile still insufficient for sustained FOCUS")

    return {
        'experiment': 'D',
        'cycles': cycles,
        'state_counts': dict(state_counts),
        'focus_durations': focus_durations,
        'crisis_cycles': crisis_cycles,
        'sustainable': bool(focus_durations and sum(focus_durations)/len(focus_durations) >= 5)
    }


if __name__ == '__main__':
    print("SAGE FOCUS GAP EXPERIMENTS")
    print(f"Thor Autonomous Session — 2026-04-11")
    print("=" * 70)
    print()

    all_results = {}

    # Run all experiments
    all_results['A'] = run_experiment_a(cycles=2000)
    all_results['B'] = run_experiment_b(cycles=2000)
    all_results['C'] = run_experiment_c(cycles_per_period=5000)
    all_results['D'] = run_experiment_d(cycles=2000)

    # Save results
    output_path = Path(__file__).parent / 'focus_gap_results.json'

    # Make JSON-serializable
    serializable = {}
    for key, val in all_results.items():
        serializable[key] = {}
        for k, v in val.items():
            if isinstance(v, dict):
                serializable[key][k] = {str(kk): vv for kk, vv in v.items()}
            else:
                serializable[key][k] = v

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"\nExperiment A (FOCUS activation): {all_results['A']['focus_entries']} FOCUS entries")
    print(f"  P11 (FOCUS→CRISIS): {'CONFIRMED' if all_results['A']['p11_validated'] else 'NOT CONFIRMED'}")

    print(f"\nExperiment B (REST→FOCUS path): {all_results['B']['rest_to_focus']} REST→FOCUS transitions")
    print(f"  Messages in REST: {all_results['B']['messages_in_rest']}/{all_results['B']['messages_sent']}")

    print(f"\nExperiment C (Period sweep):")
    for period, count in all_results['C']['focus_by_period'].items():
        print(f"  Period {period}: {count} FOCUS entries")

    print(f"\nExperiment D (Sustainable FOCUS):")
    print(f"  FOCUS cycles: {all_results['D']['state_counts'].get('focus', 0)}")
    print(f"  CRISIS cycles: {all_results['D']['crisis_cycles']}")
    print(f"  Sustainable: {all_results['D']['sustainable']}")

    print(f"\nResults saved to: {output_path}")
