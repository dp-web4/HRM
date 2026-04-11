#!/usr/bin/env python3
"""
FOCUS Gap Fix Validation — Thor SAGE Session 2026-04-11

Validates the three changes to metabolic_controller.py:
1. FOCUS exit salience: 0.50 → 0.35
2. FOCUS recovery: 0.0 → 0.3
3. CRISIS recovery: 0.2 → 0.8
4. consumption_rate wired into update()

Uses the REAL (now-fixed) MetabolicController — no monkey-patching.
"""

import sys
from pathlib import Path
from collections import Counter
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sage.core.metabolic_controller import MetabolicController, MetabolicState


def validate_focus_sustainability(cycles=5000):
    """Test: Can FOCUS sustain with audio mock salience 0.46?"""
    print("=" * 70)
    print("VALIDATION 1: FOCUS Sustainability (salience=0.46, 5000 cycles)")
    print("=" * 70)

    mc = MetabolicController(
        initial_atp=100.0, max_atp=100.0,
        circadian_period=100, enable_circadian=True,
        simulation_mode=True
    )

    state_counts = Counter()
    trans_counts = Counter()
    focus_durations = []
    current_focus_start = None
    max_focus_atp = 0
    min_focus_atp = 100

    for cycle in range(cycles):
        prev = mc.current_state

        # Note: atp_consumed=0 because in real system, plugin costs are
        # deducted separately. With consumption_rate now wired, the
        # metabolic cost is applied in update().
        cycle_data = {
            'atp_consumed': 0.0,
            'attention_load': 1,
            'max_salience': 0.46,
            'crisis_detected': False
        }
        mc.update(cycle_data)
        state_counts[mc.current_state.value] += 1

        if mc.current_state == MetabolicState.FOCUS:
            max_focus_atp = max(max_focus_atp, mc.atp_current)
            min_focus_atp = min(min_focus_atp, mc.atp_current)

        if mc.current_state != prev:
            trans_counts[f"{prev.value}→{mc.current_state.value}"] += 1
            if mc.current_state == MetabolicState.FOCUS:
                current_focus_start = cycle
            if prev == MetabolicState.FOCUS and current_focus_start is not None:
                focus_durations.append(cycle - current_focus_start)
                current_focus_start = None

    # If still in FOCUS at end
    if mc.current_state == MetabolicState.FOCUS and current_focus_start is not None:
        focus_durations.append(cycles - current_focus_start)

    print(f"\nState distribution:")
    for state, count in sorted(state_counts.items()):
        print(f"  {state:8s}: {count:6d} ({100*count/cycles:5.1f}%)")

    print(f"\nTransitions:")
    for trans, count in trans_counts.most_common():
        print(f"  {trans}: {count}")

    print(f"\nFOCUS durations: {focus_durations[:20]}")
    if focus_durations:
        print(f"  Average: {sum(focus_durations)/len(focus_durations):.1f}")
        print(f"  Min: {min(focus_durations)}, Max: {max(focus_durations)}")
        print(f"  ATP range in FOCUS: {min_focus_atp:.1f} - {max_focus_atp:.1f}")

    return {
        'test': 'focus_sustainability_no_plugin_drain',
        'state_counts': dict(state_counts),
        'focus_entries': len(focus_durations),
        'focus_durations': focus_durations,
        'crisis_cycles': state_counts.get('crisis', 0)
    }


def validate_with_plugin_drain(cycles=5000):
    """Test: FOCUS with realistic plugin drain (3.5 ATP/cycle in active states)"""
    print("\n" + "=" * 70)
    print("VALIDATION 2: FOCUS with Plugin Drain (3.5 ATP/cycle, 5000 cycles)")
    print("=" * 70)

    mc = MetabolicController(
        initial_atp=100.0, max_atp=100.0,
        circadian_period=100, enable_circadian=True,
        simulation_mode=True
    )

    state_counts = Counter()
    trans_counts = Counter()
    focus_durations = []
    current_focus_start = None
    atp_trace = []

    for cycle in range(cycles):
        prev = mc.current_state

        # Plugin drain varies by state
        if mc.current_state in [MetabolicState.WAKE, MetabolicState.FOCUS]:
            plugin_drain = 3.5
        elif mc.current_state == MetabolicState.CRISIS:
            plugin_drain = 0.5  # Minimal plugins
        elif mc.current_state == MetabolicState.REST:
            plugin_drain = 0.5
        else:
            plugin_drain = 0.0  # DREAM: no plugins

        cycle_data = {
            'atp_consumed': plugin_drain,
            'attention_load': 1,
            'max_salience': 0.46,
            'crisis_detected': False
        }
        mc.update(cycle_data)
        state_counts[mc.current_state.value] += 1
        atp_trace.append(mc.atp_current)

        if mc.current_state != prev:
            trans_counts[f"{prev.value}→{mc.current_state.value}"] += 1
            if mc.current_state == MetabolicState.FOCUS:
                current_focus_start = cycle
            if prev == MetabolicState.FOCUS and current_focus_start is not None:
                focus_durations.append(cycle - current_focus_start)
                current_focus_start = None

    if mc.current_state == MetabolicState.FOCUS and current_focus_start is not None:
        focus_durations.append(cycles - current_focus_start)

    print(f"\nState distribution:")
    for state, count in sorted(state_counts.items()):
        print(f"  {state:8s}: {count:6d} ({100*count/cycles:5.1f}%)")

    print(f"\nTransitions:")
    for trans, count in trans_counts.most_common():
        print(f"  {trans}: {count}")

    print(f"\nFOCUS durations: {focus_durations[:20]}")
    if focus_durations:
        print(f"  Average: {sum(focus_durations)/len(focus_durations):.1f}")
        print(f"  Min: {min(focus_durations)}, Max: {max(focus_durations)}")

    # Check CRISIS recovery
    crisis_count = state_counts.get('crisis', 0)
    crisis_exits = sum(1 for t in trans_counts if t.startswith('crisis→'))
    print(f"\nCRISIS cycles: {crisis_count} ({100*crisis_count/cycles:.1f}%)")
    print(f"CRISIS exits: {sum(trans_counts[t] for t in trans_counts if t.startswith('crisis→'))}")

    return {
        'test': 'focus_with_plugin_drain',
        'state_counts': dict(state_counts),
        'focus_entries': len(focus_durations),
        'focus_durations': focus_durations,
        'crisis_cycles': crisis_count
    }


def validate_crisis_recovery(cycles=1000):
    """Test: Can CRISIS recover now? (recovery 0.8 vs plugin drain 0.5)"""
    print("\n" + "=" * 70)
    print("VALIDATION 3: CRISIS Recovery (recovery=0.8, drain=0.5)")
    print("=" * 70)

    mc = MetabolicController(
        initial_atp=5.0,  # Start near CRISIS
        max_atp=100.0,
        circadian_period=100, enable_circadian=True,
        simulation_mode=True
    )

    state_trace = []
    atp_trace = []

    for cycle in range(cycles):
        plugin_drain = 0.5 if mc.current_state != MetabolicState.DREAM else 0.0

        cycle_data = {
            'atp_consumed': plugin_drain,
            'attention_load': 0,
            'max_salience': 0.1,
            'crisis_detected': False
        }
        mc.update(cycle_data)
        state_trace.append(mc.current_state.value)
        atp_trace.append(mc.atp_current)

    # Find when CRISIS exits
    crisis_exit_cycle = None
    for i, s in enumerate(state_trace):
        if i > 0 and state_trace[i-1] == 'crisis' and s != 'crisis':
            crisis_exit_cycle = i
            break

    print(f"Initial state: {state_trace[0]}, ATP: {atp_trace[0]:.2f}")
    if crisis_exit_cycle:
        print(f"CRISIS exited at cycle {crisis_exit_cycle}, ATP: {atp_trace[crisis_exit_cycle]:.2f}")
        print(f"Recovery rate: {(atp_trace[crisis_exit_cycle] - atp_trace[0]) / crisis_exit_cycle:.3f} ATP/cycle")
    else:
        print("CRISIS never exited!")

    # Show first 50 cycles
    print(f"\nFirst 50 cycles:")
    for i in range(min(50, len(state_trace))):
        if i % 5 == 0:
            print(f"  Cycle {i:3d}: {state_trace[i]:8s} ATP={atp_trace[i]:.2f}")

    return {
        'test': 'crisis_recovery',
        'crisis_exit_cycle': crisis_exit_cycle,
        'final_state': state_trace[-1],
        'final_atp': atp_trace[-1]
    }


def validate_message_handling(cycles=2000):
    """Test: Messages no longer cause permanent CRISIS"""
    print("\n" + "=" * 70)
    print("VALIDATION 4: Message Handling (periodic 35 ATP messages)")
    print("=" * 70)

    mc = MetabolicController(
        initial_atp=100.0, max_atp=100.0,
        circadian_period=100, enable_circadian=True,
        simulation_mode=True
    )

    state_counts = Counter()
    message_states = []

    for cycle in range(cycles):
        is_message = (cycle % 200 == 100)
        salience = 0.85 if is_message else 0.46

        if mc.current_state in [MetabolicState.WAKE, MetabolicState.FOCUS]:
            plugin_drain = 35.0 if is_message else 3.5
        else:
            plugin_drain = 35.0 if is_message else 0.5

        cycle_data = {
            'atp_consumed': plugin_drain,
            'attention_load': 3 if is_message else 1,
            'max_salience': salience,
            'crisis_detected': False
        }
        mc.update(cycle_data)
        state_counts[mc.current_state.value] += 1

        if is_message:
            message_states.append({
                'cycle': cycle,
                'state': mc.current_state.value,
                'atp': round(mc.atp_current, 2)
            })

    print(f"\nState distribution:")
    for state, count in sorted(state_counts.items()):
        print(f"  {state:8s}: {count:6d} ({100*count/cycles:5.1f}%)")

    print(f"\nMessage events:")
    for msg in message_states:
        print(f"  Cycle {msg['cycle']}: {msg['state']:8s} ATP={msg['atp']}")

    crisis_pct = 100 * state_counts.get('crisis', 0) / cycles
    print(f"\nCRISIS percentage: {crisis_pct:.1f}% (was 95% before fix)")
    if crisis_pct < 50:
        print("VERDICT: Message handling IMPROVED — no longer causes CRISIS death spiral")
    else:
        print("VERDICT: CRISIS still dominates — further tuning needed")

    return {
        'test': 'message_handling',
        'state_counts': dict(state_counts),
        'crisis_pct': crisis_pct,
        'message_states': message_states
    }


if __name__ == '__main__':
    print("FOCUS GAP FIX VALIDATION")
    print(f"Thor SAGE Session — 2026-04-11")
    print("=" * 70)
    print()
    print("Changes validated:")
    print("  1. FOCUS exit salience: 0.50 → 0.35")
    print("  2. FOCUS recovery: 0.0 → 0.3")
    print("  3. CRISIS recovery: 0.2 → 0.8")
    print("  4. consumption_rate wired into update()")
    print()

    results = {}
    results['v1'] = validate_focus_sustainability()
    results['v2'] = validate_with_plugin_drain()
    results['v3'] = validate_crisis_recovery()
    results['v4'] = validate_message_handling()

    # Save results
    output_path = Path(__file__).parent / 'focus_gap_fix_validation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print("=" * 70)

    v1_focus = results['v1']['focus_entries']
    v2_focus = results['v2']['focus_entries']
    v3_recovery = results['v3']['crisis_exit_cycle']
    v4_crisis = results['v4']['crisis_pct']

    print(f"\n1. Focus sustainability (no plugins): {v1_focus} FOCUS entries")
    print(f"2. Focus with plugin drain: {v2_focus} FOCUS entries")
    print(f"3. Crisis recovery: {'Cycle ' + str(v3_recovery) if v3_recovery else 'FAILED'}")
    print(f"4. Message handling: {v4_crisis:.1f}% CRISIS (was 95%)")

    print(f"\nResults saved to: {output_path}")
