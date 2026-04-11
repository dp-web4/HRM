#!/usr/bin/env python3
"""
DREAM Gap Experiments — Thor SAGE Session 2026-04-11 12:00

Investigation: Only 26 dream entries in 20.4M consciousness cycles (0.005%).
This mirrors the FOCUS gap (0 in 20.4M) discovered in the 18:00 session.

Hypothesis: REST→WAKE always wins the race against REST→DREAM because:
  1. ATP recovery is fast (~+0.9 net/cycle)
  2. REST→WAKE threshold (ATP > 50) reached in ~22 cycles
  3. REST→DREAM requires 20-60 seconds in REST (200-600 cycles)
  4. REST→WAKE is checked BEFORE REST→DREAM in the transition logic

Similarly, WAKE→DREAM requires 100-300 seconds in WAKE, but WAKE→REST
fires in ~6 cycles due to fast ATP drain.

Experiments:
  D1: Baseline — measure dream probability across 50,000 simulated cycles
  D2: Circadian sweep — dream_bias range vs dream entry frequency
  D3: Time threshold analysis — what time_in_state does REST/WAKE actually reach?
  D4: Transition priority analysis — does check order matter?
  D5: Fix validation — test proposed fixes

All experiments use real MetabolicController in simulation_mode.
"""

import sys
import os
import json
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sage.core.metabolic_controller import MetabolicController, MetabolicState


def simulate_cycles(mc, cycles, salience_pattern='audio_mock', verbose=False):
    """Run N cycles with standard mock sensor pattern.

    Returns dict with state counts, transitions, and timing stats.
    """
    state_counts = Counter()
    transitions = []
    time_in_state_samples = defaultdict(list)  # state -> [durations]

    prev_state = mc.current_state
    state_entry_cycle = 0

    for i in range(cycles):
        # Audio mock pattern: 50% chance of salience 0.46, otherwise 0.09
        if salience_pattern == 'audio_mock':
            max_salience = 0.46 if (i % 2 == 0) else 0.09
        elif salience_pattern == 'high':
            max_salience = 0.8
        else:
            max_salience = salience_pattern if isinstance(salience_pattern, float) else 0.09

        # Standard plugin drain in WAKE
        if mc.current_state == MetabolicState.WAKE:
            atp_consumed = 3.5  # Mock plugin heartbeat cost
        elif mc.current_state == MetabolicState.FOCUS:
            atp_consumed = 5.0  # Higher during focus
        else:
            atp_consumed = 0.0

        cycle_data = {
            'atp_consumed': atp_consumed,
            'attention_load': 1,
            'max_salience': max_salience,
            'crisis_detected': False
        }

        new_state = mc.update(cycle_data)

        state_counts[new_state.value] += 1

        if new_state != prev_state:
            # Record transition
            duration = i - state_entry_cycle
            time_in_state_samples[prev_state.value].append(duration)
            transitions.append({
                'cycle': i,
                'from': prev_state.value,
                'to': new_state.value,
                'duration_in_prev': duration,
                'atp': mc.atp_current,
            })
            state_entry_cycle = i
            prev_state = new_state

    # Collect stats
    transition_counts = Counter()
    for t in transitions:
        transition_counts[f"{t['from']}→{t['to']}"] += 1

    return {
        'state_counts': dict(state_counts),
        'transition_counts': dict(transition_counts),
        'transitions': transitions,
        'time_in_state': {
            k: {
                'count': len(v),
                'min': min(v) if v else 0,
                'max': max(v) if v else 0,
                'avg': sum(v) / len(v) if v else 0,
            }
            for k, v in time_in_state_samples.items()
        },
        'total_cycles': cycles,
    }


def experiment_d1_baseline(cycles=50000):
    """D1: Baseline dream frequency measurement."""
    print("=" * 70)
    print("EXPERIMENT D1: BASELINE DREAM FREQUENCY")
    print("=" * 70)
    print(f"Cycles: {cycles}")
    print()

    mc = MetabolicController(
        initial_atp=100.0,
        max_atp=100.0,
        circadian_period=100,
        enable_circadian=True,
        simulation_mode=True
    )

    results = simulate_cycles(mc, cycles)

    dream_count = results['state_counts'].get('dream', 0)
    dream_pct = 100.0 * dream_count / cycles

    print(f"State distribution:")
    for state, count in sorted(results['state_counts'].items()):
        pct = 100.0 * count / cycles
        print(f"  {state:8s}: {count:6d} ({pct:.2f}%)")

    print(f"\nTransition counts:")
    for trans, count in sorted(results['transition_counts'].items(), key=lambda x: -x[1]):
        print(f"  {trans:20s}: {count:5d}")

    print(f"\nTime-in-state stats (cycles):")
    for state, stats in sorted(results['time_in_state'].items()):
        print(f"  {state:8s}: min={stats['min']:4d}  max={stats['max']:4d}  avg={stats['avg']:.1f}  count={stats['count']:5d}")

    print(f"\n{'='*70}")
    print(f"DREAM entries: {dream_count} cycles ({dream_pct:.4f}%)")
    dream_transitions = sum(v for k, v in results['transition_counts'].items() if '→dream' in k)
    print(f"DREAM transition events: {dream_transitions}")
    print(f"{'='*70}")

    return results


def experiment_d2_circadian_sweep():
    """D2: How does circadian phase affect dream entry?"""
    print("=" * 70)
    print("EXPERIMENT D2: CIRCADIAN PHASE VS DREAM ENTRY")
    print("=" * 70)
    print()

    # Run with different starting phases to sample full circadian cycle
    # Period=100, so we test starting at different offsets
    cycles_per_run = 10000

    phase_results = {}

    for start_cycle in range(0, 100, 10):
        mc = MetabolicController(
            initial_atp=50.0,
            max_atp=100.0,
            circadian_period=100,
            enable_circadian=True,
            simulation_mode=True
        )
        # Advance clock to desired phase
        mc.circadian_clock.current_cycle = start_cycle

        results = simulate_cycles(mc, cycles_per_run)
        dream_transitions = sum(v for k, v in results['transition_counts'].items() if '→dream' in k)

        phase_name = ['DAWN', 'DAY', 'DAY', 'DAY', 'DAY', 'DUSK', 'NIGHT', 'NIGHT', 'NIGHT', 'DEEP_NIGHT'][start_cycle // 10]
        phase_results[start_cycle] = {
            'phase': phase_name,
            'dream_transitions': dream_transitions,
            'dream_cycles': results['state_counts'].get('dream', 0),
            'dream_pct': 100.0 * results['state_counts'].get('dream', 0) / cycles_per_run,
        }

        dream_bias = mc.circadian_clock.get_metabolic_bias('dream')
        print(f"  Start cycle {start_cycle:3d} ({phase_name:10s}): "
              f"dream_bias={dream_bias:.2f}  "
              f"dream_transitions={dream_transitions:3d}  "
              f"dream_cycles={results['state_counts'].get('dream', 0):5d} "
              f"({phase_results[start_cycle]['dream_pct']:.2f}%)")

    return phase_results


def experiment_d3_time_threshold_analysis(cycles=10000):
    """D3: What time_in_state does REST/WAKE actually reach before transitioning?"""
    print("=" * 70)
    print("EXPERIMENT D3: TIME-IN-STATE ANALYSIS")
    print("=" * 70)
    print(f"Cycles: {cycles}")
    print()

    mc = MetabolicController(
        initial_atp=100.0,
        max_atp=100.0,
        circadian_period=100,
        enable_circadian=True,
        simulation_mode=True
    )

    results = simulate_cycles(mc, cycles)

    print("Time-in-state distributions:")
    for state, stats in sorted(results['time_in_state'].items()):
        print(f"\n  {state.upper()}:")
        print(f"    Min duration:  {stats['min']} cycles")
        print(f"    Max duration:  {stats['max']} cycles")
        print(f"    Avg duration:  {stats['avg']:.1f} cycles")
        print(f"    Transitions:   {stats['count']}")

    # Key analysis: what are the dream time thresholds?
    print(f"\n{'='*70}")
    print("THRESHOLD ANALYSIS (simulation mode):")
    print()

    # In simulation mode:
    # WAKE→DREAM: max(5, 30 / dream_bias) cycles, need 40 < ATP < 80
    # REST→DREAM: max(5, 6 / dream_bias) cycles, need ATP > 40/dream_bias
    for dream_bias in [1.0, 1.5, 2.0, 3.0]:
        wake_dream_time = max(5, 30 / dream_bias)
        rest_dream_time = max(5, 6 / dream_bias)
        rest_dream_atp = 40.0 / dream_bias
        print(f"  dream_bias={dream_bias:.1f}:")
        print(f"    WAKE→DREAM needs: time>{wake_dream_time:.0f} cycles AND 40<ATP<80")
        print(f"    REST→DREAM needs: time>{rest_dream_time:.0f} cycles AND ATP>{rest_dream_atp:.1f}")

    wake_stats = results['time_in_state'].get('wake', {})
    rest_stats = results['time_in_state'].get('rest', {})
    print(f"\n  Actual WAKE max duration: {wake_stats.get('max', 'N/A')} cycles")
    print(f"  Actual REST max duration: {rest_stats.get('max', 'N/A')} cycles")

    # Check if any REST durations exceeded dream threshold
    # Re-run with detailed per-state tracking
    mc2 = MetabolicController(
        initial_atp=100.0, max_atp=100.0,
        circadian_period=100, enable_circadian=True, simulation_mode=True
    )

    rest_durations = []
    wake_durations = []
    prev_state = mc2.current_state
    entry_cycle = 0

    for i in range(cycles):
        max_salience = 0.46 if (i % 2 == 0) else 0.09
        atp_consumed = 3.5 if mc2.current_state == MetabolicState.WAKE else (5.0 if mc2.current_state == MetabolicState.FOCUS else 0.0)

        new_state = mc2.update({
            'atp_consumed': atp_consumed,
            'attention_load': 1,
            'max_salience': max_salience,
            'crisis_detected': False
        })

        if new_state != prev_state:
            dur = i - entry_cycle
            if prev_state == MetabolicState.REST:
                rest_durations.append(dur)
            elif prev_state == MetabolicState.WAKE:
                wake_durations.append(dur)
            entry_cycle = i
            prev_state = new_state

    print(f"\n  REST duration histogram (cycles):")
    if rest_durations:
        for bucket_start in range(0, max(rest_durations) + 5, 5):
            count = sum(1 for d in rest_durations if bucket_start <= d < bucket_start + 5)
            if count:
                bar = '#' * min(count, 60)
                print(f"    {bucket_start:4d}-{bucket_start+4:4d}: {count:4d} {bar}")

    print(f"\n  WAKE duration histogram (cycles):")
    if wake_durations:
        for bucket_start in range(0, max(wake_durations) + 5, 5):
            count = sum(1 for d in wake_durations if bucket_start <= d < bucket_start + 5)
            if count:
                bar = '#' * min(count, 60)
                print(f"    {bucket_start:4d}-{bucket_start+4:4d}: {count:4d} {bar}")

    return results


def experiment_d4_transition_priority(cycles=10000):
    """D4: Does REST→WAKE always beat REST→DREAM due to check order?

    Test by patching: swap the check order so DREAM is checked before WAKE in REST.
    """
    print("=" * 70)
    print("EXPERIMENT D4: TRANSITION PRIORITY (CHECK ORDER)")
    print("=" * 70)
    print()

    # Baseline: standard order
    mc_baseline = MetabolicController(
        initial_atp=100.0, max_atp=100.0,
        circadian_period=100, enable_circadian=True, simulation_mode=True
    )
    results_baseline = simulate_cycles(mc_baseline, cycles)
    dream_baseline = sum(v for k, v in results_baseline['transition_counts'].items() if '→dream' in k)

    # Patched: override _determine_next_state to check dream before wake in REST
    mc_patched = MetabolicController(
        initial_atp=100.0, max_atp=100.0,
        circadian_period=100, enable_circadian=True, simulation_mode=True
    )

    original_determine = mc_patched._determine_next_state.__func__

    def patched_determine_next_state(self, attention_load, max_salience, crisis_detected):
        """REST state: check DREAM before WAKE."""
        import time as time_mod

        if self.circadian_clock:
            circadian_ctx = self.circadian_clock.tick()
            wake_bias = self.circadian_clock.get_metabolic_bias('wake')
            focus_bias = self.circadian_clock.get_metabolic_bias('focus')
            dream_bias = self.circadian_clock.get_metabolic_bias('dream')
        else:
            wake_bias = focus_bias = dream_bias = 1.0

        self.cycles_in_state += 1

        if crisis_detected or self.atp_current < 10.0:
            return MetabolicState.CRISIS

        if self.cycles_in_state < self.min_cycles_in_state:
            return self.current_state

        time_in_state = self._get_time_in_state()

        if self.current_state == MetabolicState.REST:
            # DREAM checked FIRST (before WAKE)
            dream_atp_threshold = 40.0 / dream_bias
            dream_time_threshold = max(5, 6 / dream_bias) if self.simulation_mode else max(5, 60 / dream_bias)
            if self.atp_current > dream_atp_threshold and time_in_state > dream_time_threshold:
                return MetabolicState.DREAM

            # Then WAKE
            wake_threshold = 50.0 * wake_bias
            if self.atp_current > wake_threshold:
                return MetabolicState.WAKE

            return MetabolicState.REST
        else:
            # All other states use original logic
            return original_determine(self, attention_load, max_salience, crisis_detected)

    import types
    mc_patched._determine_next_state = types.MethodType(patched_determine_next_state, mc_patched)

    results_patched = simulate_cycles(mc_patched, cycles)
    dream_patched = sum(v for k, v in results_patched['transition_counts'].items() if '→dream' in k)

    print(f"Baseline (WAKE checked first):")
    print(f"  Dream transitions: {dream_baseline}")
    print(f"  Dream cycles: {results_baseline['state_counts'].get('dream', 0)}")
    for state, count in sorted(results_baseline['state_counts'].items()):
        print(f"    {state:8s}: {count:5d} ({100*count/cycles:.1f}%)")

    print(f"\nPatched (DREAM checked first in REST):")
    print(f"  Dream transitions: {dream_patched}")
    print(f"  Dream cycles: {results_patched['state_counts'].get('dream', 0)}")
    for state, count in sorted(results_patched['state_counts'].items()):
        print(f"    {state:8s}: {count:5d} ({100*count/cycles:.1f}%)")

    if dream_patched > dream_baseline:
        print(f"\n  CHECK ORDER MATTERS: {dream_patched - dream_baseline} more dream entries with dream-first")
    elif dream_patched == dream_baseline:
        print(f"\n  CHECK ORDER DOESN'T MATTER: same dream count either way")
        print(f"  → The real barrier is elsewhere (timing? ATP levels?)")
    else:
        print(f"\n  UNEXPECTED: fewer dreams with dream-first")

    return {
        'baseline': results_baseline,
        'patched': results_patched,
        'dream_baseline': dream_baseline,
        'dream_patched': dream_patched,
    }


def experiment_d5_proposed_fixes(cycles=10000):
    """D5: Test fixes for the dream gap.

    Fix 1: Lower REST→DREAM time threshold (60s → 6s in real mode, matching sim)
    Fix 2: Add probability-based dream entry (stochastic instead of deterministic)
    Fix 3: Reduce REST recovery rate so REST lasts longer
    """
    print("=" * 70)
    print("EXPERIMENT D5: PROPOSED DREAM GAP FIXES")
    print("=" * 70)
    print()

    # Baseline
    mc_base = MetabolicController(
        initial_atp=100.0, max_atp=100.0,
        circadian_period=100, enable_circadian=True, simulation_mode=True
    )
    results_base = simulate_cycles(mc_base, cycles)
    dream_base = results_base['state_counts'].get('dream', 0)

    # Fix 1: Lower dream time thresholds in simulation mode
    # Patch: REST→DREAM time threshold = 3 cycles (instead of max(5, 6/dream_bias))
    mc_fix1 = MetabolicController(
        initial_atp=100.0, max_atp=100.0,
        circadian_period=100, enable_circadian=True, simulation_mode=True
    )

    original_determine_fix1 = mc_fix1._determine_next_state.__func__

    def fix1_determine(self, attention_load, max_salience, crisis_detected):
        """Lower dream time thresholds."""
        if self.circadian_clock:
            circadian_ctx = self.circadian_clock.tick()
            wake_bias = self.circadian_clock.get_metabolic_bias('wake')
            focus_bias = self.circadian_clock.get_metabolic_bias('focus')
            dream_bias = self.circadian_clock.get_metabolic_bias('dream')
        else:
            wake_bias = focus_bias = dream_bias = 1.0

        self.cycles_in_state += 1

        if crisis_detected or self.atp_current < 10.0:
            return MetabolicState.CRISIS
        if self.cycles_in_state < self.min_cycles_in_state:
            return self.current_state

        time_in_state = self._get_time_in_state()

        if self.current_state == MetabolicState.WAKE:
            focus_threshold = 50.0 / focus_bias
            if max_salience > 0.45 and self.atp_current > focus_threshold:
                return MetabolicState.FOCUS
            rest_threshold = 30.0 * wake_bias
            if self.atp_current < rest_threshold:
                return MetabolicState.REST
            # Lowered dream threshold
            dream_time_threshold = max(3, 15 / dream_bias) if self.simulation_mode else max(5, 60 / dream_bias)
            if 40.0 < self.atp_current < 80.0 and time_in_state > dream_time_threshold:
                return MetabolicState.DREAM
            return MetabolicState.WAKE

        elif self.current_state == MetabolicState.REST:
            # Check DREAM first, with lower thresholds
            dream_atp_threshold = 40.0 / dream_bias
            dream_time_threshold = max(3, 3 / dream_bias) if self.simulation_mode else max(5, 15 / dream_bias)
            if self.atp_current > dream_atp_threshold and time_in_state > dream_time_threshold:
                return MetabolicState.DREAM
            wake_threshold = 50.0 * wake_bias
            if self.atp_current > wake_threshold:
                return MetabolicState.WAKE
            return MetabolicState.REST

        elif self.current_state == MetabolicState.FOCUS:
            if max_salience < 0.35 or self.atp_current < 20.0:
                return MetabolicState.WAKE
            if self.atp_current < 15.0:
                return MetabolicState.REST
            return MetabolicState.FOCUS

        elif self.current_state == MetabolicState.DREAM:
            wake_threshold = 70.0 * wake_bias
            max_dream_time = (18 / dream_bias) if self.simulation_mode else (180 / dream_bias)
            if self.atp_current > wake_threshold or time_in_state > max_dream_time:
                return MetabolicState.WAKE
            if self.atp_current < 40.0:
                return MetabolicState.REST
            return MetabolicState.DREAM

        elif self.current_state == MetabolicState.CRISIS:
            if self.atp_current > 15.0 and not crisis_detected:
                return MetabolicState.REST
            return MetabolicState.CRISIS

        return self.current_state

    import types
    mc_fix1._determine_next_state = types.MethodType(fix1_determine, mc_fix1)

    results_fix1 = simulate_cycles(mc_fix1, cycles)
    dream_fix1 = results_fix1['state_counts'].get('dream', 0)

    print(f"Baseline:")
    print(f"  Dream cycles: {dream_base} ({100*dream_base/cycles:.2f}%)")
    for state, count in sorted(results_base['state_counts'].items()):
        print(f"    {state:8s}: {count:5d} ({100*count/cycles:.1f}%)")

    print(f"\nFix 1 (lower thresholds + dream-first in REST):")
    print(f"  Dream cycles: {dream_fix1} ({100*dream_fix1/cycles:.2f}%)")
    for state, count in sorted(results_fix1['state_counts'].items()):
        print(f"    {state:8s}: {count:5d} ({100*count/cycles:.1f}%)")

    dream_fix1_transitions = sum(v for k, v in results_fix1['transition_counts'].items() if '→dream' in k)
    print(f"  Dream transitions: {dream_fix1_transitions}")
    print(f"  Transition counts:")
    for trans, count in sorted(results_fix1['transition_counts'].items(), key=lambda x: -x[1]):
        print(f"    {trans:20s}: {count:5d}")

    return {
        'baseline_dream': dream_base,
        'fix1_dream': dream_fix1,
        'baseline': results_base,
        'fix1': results_fix1,
    }


if __name__ == '__main__':
    print("DREAM GAP EXPERIMENTS — Thor SAGE 2026-04-11 12:00")
    print("=" * 70)
    print()

    results = {}

    # D1: Baseline measurement
    results['d1'] = experiment_d1_baseline(cycles=50000)
    print()

    # D2: Circadian sweep
    results['d2'] = experiment_d2_circadian_sweep()
    print()

    # D3: Time threshold analysis
    results['d3'] = experiment_d3_time_threshold_analysis(cycles=20000)
    print()

    # D4: Check order priority
    results['d4'] = experiment_d4_transition_priority(cycles=20000)
    print()

    # D5: Proposed fixes
    results['d5'] = experiment_d5_proposed_fixes(cycles=20000)
    print()

    # Summary
    print("=" * 70)
    print("DREAM GAP EXPERIMENTS: SUMMARY")
    print("=" * 70)

    d1_dream = results['d1']['state_counts'].get('dream', 0)
    d4_baseline = results['d4']['dream_baseline']
    d4_patched = results['d4']['dream_patched']
    d5_baseline = results['d5']['baseline_dream']
    d5_fix1 = results['d5']['fix1_dream']

    print(f"\n  D1 Baseline (50K cycles): {d1_dream} dream cycles")
    print(f"  D4 Check order: baseline={d4_baseline}, dream-first={d4_patched}")
    print(f"  D5 Fix: baseline={d5_baseline}, fix1={d5_fix1}")

    # Save results
    save_path = Path(__file__).parent / 'dream_gap_results.json'
    serializable = {
        'd1_dream_cycles': d1_dream,
        'd1_total': results['d1']['total_cycles'],
        'd4_baseline': d4_baseline,
        'd4_patched': d4_patched,
        'd5_baseline': d5_baseline,
        'd5_fix1': d5_fix1,
    }
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {save_path}")
