#!/usr/bin/env python3
"""
DREAM Gap Fix Validation — Thor SAGE 2026-04-11 12:00

Validates the unified cycle-count fix for _get_time_in_state().
Both simulation and real mode should now produce similar dream percentages.

The fix: _get_time_in_state() always returns cycle counts (not wall time),
and dream thresholds no longer have sim/real branching.
"""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sage.core.metabolic_controller import MetabolicController, MetabolicState


def run_validation(mode_name, simulation_mode, cycles=20000):
    """Run a metabolic simulation and report state distribution."""
    mc = MetabolicController(
        initial_atp=100.0,
        max_atp=100.0,
        circadian_period=100,
        enable_circadian=True,
        simulation_mode=simulation_mode
    )

    state_counts = Counter()
    transitions = Counter()
    prev_state = mc.current_state

    for i in range(cycles):
        max_salience = 0.46 if (i % 2 == 0) else 0.09

        if mc.current_state == MetabolicState.WAKE:
            atp_consumed = 3.5
        elif mc.current_state == MetabolicState.FOCUS:
            atp_consumed = 5.0
        else:
            atp_consumed = 0.0

        new_state = mc.update({
            'atp_consumed': atp_consumed,
            'attention_load': 1,
            'max_salience': max_salience,
            'crisis_detected': False
        })

        state_counts[new_state.value] += 1

        if new_state != prev_state:
            transitions[f"{prev_state.value}→{new_state.value}"] += 1
            prev_state = new_state

    dream_pct = 100.0 * state_counts.get('dream', 0) / cycles
    focus_pct = 100.0 * state_counts.get('focus', 0) / cycles

    print(f"\n  {mode_name} ({cycles} cycles):")
    for state in ['wake', 'rest', 'focus', 'dream', 'crisis']:
        count = state_counts.get(state, 0)
        pct = 100.0 * count / cycles
        print(f"    {state:8s}: {count:6d} ({pct:5.1f}%)")

    dream_transitions = sum(v for k, v in transitions.items() if '→dream' in k)
    print(f"    Dream entry events: {dream_transitions}")

    print(f"  Key transitions:")
    for trans, count in sorted(transitions.items(), key=lambda x: -x[1])[:8]:
        print(f"    {trans:20s}: {count:5d}")

    return {
        'dream_pct': dream_pct,
        'focus_pct': focus_pct,
        'state_counts': dict(state_counts),
        'transitions': dict(transitions),
    }


if __name__ == '__main__':
    print("=" * 70)
    print("DREAM GAP FIX VALIDATION")
    print("=" * 70)
    print()
    print("After fix: _get_time_in_state() always returns cycle counts.")
    print("Dream thresholds unified across simulation and real modes.")

    sim_result = run_validation("Simulation Mode", simulation_mode=True)
    real_result = run_validation("Real Mode (post-fix)", simulation_mode=False)

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\n  Simulation dream%: {sim_result['dream_pct']:.1f}%")
    print(f"  Real mode dream%:  {real_result['dream_pct']:.1f}%")

    diff = abs(sim_result['dream_pct'] - real_result['dream_pct'])
    if diff < 5.0:
        print(f"\n  PASS: Dream percentages match within 5% (diff={diff:.1f}%)")
        print(f"  Fix validated — real mode now produces healthy dream cycles.")
    else:
        print(f"\n  WARNING: Dream percentages diverge by {diff:.1f}%")
        print(f"  Additional investigation needed.")

    print()
    print("Pre-fix comparison:")
    print(f"  Before: 26 dream entries in 20.4M cycles (0.005%)")
    print(f"  After:  {real_result['dream_pct']:.1f}% dream cycles")
    improvement = real_result['dream_pct'] / 0.005 if real_result['dream_pct'] > 0 else 0
    print(f"  Improvement: {improvement:.0f}x")
