#!/usr/bin/env python3
"""
DREAM Gap Real-Mode Analysis — Thor SAGE 2026-04-11 12:00

Finding from D1-D5: DREAM works perfectly in simulation (47.87% of cycles).
The gap exists only in REAL mode because time thresholds use wall-clock seconds.

This experiment simulates real-mode behavior by running the controller
in real-mode but with accelerated time to map the barrier precisely.

Key question: What is the maximum REST duration in real mode, and how does
it compare to the dream time threshold?
"""

import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sage.core.metabolic_controller import MetabolicController, MetabolicState


def analyze_real_mode_timing():
    """Analyze exact timing constraints that prevent DREAM in real mode."""
    print("=" * 70)
    print("REAL-MODE DREAM GAP ANALYSIS")
    print("=" * 70)
    print()

    # Simulate what happens in real mode by computing exact cycle durations
    # based on the energy economics

    # REST state economics:
    # consumption_rate = 0.1, recovery_rate = 1.0 → net +0.9/cycle
    # REST enters when ATP < 30 * wake_bias
    # REST exits (→WAKE) when ATP > 50 * wake_bias

    # At different circadian phases:
    phases = {
        'DAY (peak)': {'wake_bias': 1.5, 'dream_bias': 1.0},
        'DUSK': {'wake_bias': 1.25, 'dream_bias': 1.0},
        'NIGHT (early)': {'wake_bias': 1.0, 'dream_bias': 2.0},
        'NIGHT (peak)': {'wake_bias': 1.0, 'dream_bias': 3.0},
        'DEEP_NIGHT': {'wake_bias': 1.0, 'dream_bias': 2.0},
        'DAWN': {'wake_bias': 1.25, 'dream_bias': 1.0},
    }

    # Daemon cycle rate from empirical data:
    # Wake: 12 cycles avg in 1.2s → 0.1s/cycle
    # Rest: 28 cycles avg in 2.8s → 0.1s/cycle
    cycle_duration_s = 0.1  # seconds per cycle

    print(f"Assumed cycle rate: {cycle_duration_s}s per cycle")
    print()
    print(f"{'Phase':<18} | {'Wake Bias':>9} | {'Dream Bias':>10} | "
          f"{'REST Enter ATP':>14} | {'REST Exit ATP':>13} | "
          f"{'Cycles in REST':>14} | {'REST Duration':>13} | "
          f"{'Dream Time Thr':>14} | {'Can Dream?':>10}")
    print("-" * 140)

    for phase_name, biases in phases.items():
        wb = biases['wake_bias']
        db = biases['dream_bias']

        # REST entry: ATP ~ 30 * wake_bias (from WAKE→REST threshold)
        rest_entry_atp = 30.0 * wb
        # But with FOCUS fix, WAKE might enter REST from lower ATP
        # Conservative: REST entry at 30 * wb - some margin
        rest_entry_atp_low = max(10, rest_entry_atp - 5)

        # REST exit: ATP > 50 * wake_bias (REST→WAKE threshold)
        rest_exit_atp = 50.0 * wb

        # NET recovery in REST: +1.0 - 0.1 = +0.9 per cycle
        net_recovery = 1.0 - 0.1

        # Cycles to recover from entry to exit
        cycles_in_rest = (rest_exit_atp - rest_entry_atp) / net_recovery
        rest_duration_s = cycles_in_rest * cycle_duration_s

        # Dream time threshold (real mode)
        dream_time_threshold = max(5, 60.0 / db)

        # Dream ATP threshold
        dream_atp_threshold = 40.0 / db

        can_dream = rest_duration_s > dream_time_threshold

        print(f"{phase_name:<18} | {wb:>9.2f} | {db:>10.2f} | "
              f"{rest_entry_atp:>14.1f} | {rest_exit_atp:>13.1f} | "
              f"{cycles_in_rest:>14.1f} | {rest_duration_s:>11.1f}s | "
              f"{dream_time_threshold:>12.1f}s | "
              f"{'YES' if can_dream else 'NO':>10}")

    print()
    print("=" * 70)
    print("ANALYSIS: Real-Mode REST Duration vs Dream Time Threshold")
    print("=" * 70)
    print()

    # The worst case for dream: how long would REST need to last?
    # At peak night (dream_bias=3.0): threshold = max(5, 60/3) = 20s
    # REST duration at night (wake_bias=1.0): (50-30)/0.9 * 0.1 = 2.2s
    # Gap: 20s needed, 2.2s available → 9x shortfall

    print("Shortfall analysis (dream_time_threshold / actual_rest_duration):")
    for phase_name, biases in phases.items():
        wb = biases['wake_bias']
        db = biases['dream_bias']
        rest_duration = ((50.0 * wb) - (30.0 * wb)) / 0.9 * cycle_duration_s
        dream_threshold = max(5, 60.0 / db)
        shortfall = dream_threshold / rest_duration if rest_duration > 0 else float('inf')
        print(f"  {phase_name:<18}: need {dream_threshold:.1f}s, have {rest_duration:.1f}s → {shortfall:.1f}x shortfall")

    print()
    print("=" * 70)
    print("WAKE→DREAM Real-Mode Analysis")
    print("=" * 70)
    print()

    # WAKE→DREAM: need time > max(5, 300/dream_bias) AND 40 < ATP < 80
    # WAKE drains at ~3.5 ATP/cycle (plugin mock) + 0 consumption (WAKE consumption=0)
    # Actually: WAKE StateConfig has no consumption_rate listed... let me check
    # From the config: WAKE isn't listed explicitly, defaults may vary
    # Plugin drain is ~3.5 ATP/cycle, recovery is 0 in WAKE
    # Actually: the WAKE config recovery_rate = 0.0 was set somewhere
    # Let me just compute from known data: WAKE lasts ~1.2s average

    print(f"WAKE→DREAM threshold (real mode): max(5, 300/dream_bias) seconds")
    for phase_name, biases in phases.items():
        db = biases['dream_bias']
        threshold = max(5, 300.0 / db)
        print(f"  {phase_name:<18}: {threshold:.1f}s needed (WAKE lasts ~1.2s)")

    print()
    print("=" * 70)
    print("PROPOSED FIX: Align Real-Mode Thresholds with Simulation")
    print("=" * 70)
    print()
    print("Current simulation thresholds work correctly (47.87% dream).")
    print("The fix: make real-mode thresholds proportional to cycle rate,")
    print("not absolute wall-clock seconds.")
    print()
    print("Proposed changes to metabolic_controller.py:")
    print()
    print("  WAKE→DREAM (line 287):")
    print("    Before: max(5, 300 / dream_bias)  # 100-300 seconds")
    print("    After:  max(5, 30 / dream_bias)   # Use sim-mode values always")
    print()
    print("  REST→DREAM (line 318):")
    print("    Before: max(5, 60 / dream_bias)   # 20-60 seconds")
    print("    After:  max(5, 6 / dream_bias)    # Use sim-mode values always")
    print()
    print("Alternative: Use cycle counts instead of wall time for all thresholds.")
    print("The _get_time_in_state() method already returns cycles in sim mode.")
    print("Making it always return cycles would unify behavior across modes.")


if __name__ == '__main__':
    analyze_real_mode_timing()
