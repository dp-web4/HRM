#!/usr/bin/env python3
"""
Attention Quality vs Quantity Analysis
======================================

**Context**: Session 12 validated that ATP parameters control attention ceiling:
- Maximum (cost=0.01, recovery=0.05): 62% attention
- Balanced (cost=0.03, recovery=0.04): 42% attention
- Conservative (cost=0.05, recovery=0.02): 26% attention

**Research Question**: Does higher attention rate (62%) degrade selectivity?

**Hypothesis**: Higher attention may attend lower-salience observations, reducing
quality. Balanced config (42%) may achieve better selectivity by being more
discriminating.

**Analysis**:
1. Measure attended salience distribution for each configuration
2. Compare selectivity (how high-salience are attended items?)
3. Compare coverage (what % of high-salience items are caught?)
4. Identify optimal balance between quality and quantity

**Metrics**:
- **Selectivity**: Avg attended salience (higher = more selective)
- **Coverage**: % of high-salience (>0.7) observations attended
- **Precision**: % of attended observations that are high-salience
- **Efficiency**: Attention per unit salience (coverage/attention_rate)

**Expected Results**:
- Maximum (62%): Lower selectivity, higher coverage
- Balanced (42%): Higher selectivity, good coverage
- Conservative (26%): Highest selectivity, lower coverage

Author: Claude (autonomous research) on Thor
Date: 2025-12-08
Session: Attention quality analysis (Session 13)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'core'))

import random
from typing import Dict, List, Tuple
from collections import Counter
from dataclasses import dataclass

# Import from Session 12's validation code
from validate_atp_on_real_consciousness import (
    ATPTunedConsciousness,
    SensorObservation,
    generate_realistic_observations
)


@dataclass
class QualityMetrics:
    """Metrics for attention quality analysis"""
    attention_rate: float
    avg_attended_salience: float
    avg_unattended_salience: float
    high_salience_coverage: float  # % of >0.7 salience attended
    precision: float  # % of attended that are >0.7
    efficiency: float  # coverage / attention_rate
    salience_histogram: Dict[str, float]
    attended_salience_dist: List[float]
    unattended_salience_dist: List[float]


def analyze_quality(
    config_name: str,
    attention_cost: float,
    rest_recovery: float,
    cycles: int = 2000,
    trials: int = 3,
    high_salience_threshold: float = 0.7
) -> QualityMetrics:
    """
    Analyze attention quality for a specific ATP configuration.

    Tracks not just how much attention, but what quality of observations
    are being attended.
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {config_name}")
    print(f"  ATP params: cost={attention_cost:.3f}, recovery={rest_recovery:.3f}")
    print(f"  High-salience threshold: {high_salience_threshold}")
    print(f"{'='*70}")

    # Aggregate across trials
    all_attended_saliences = []
    all_unattended_saliences = []
    total_observations = 0
    total_attended = 0
    high_salience_count = 0
    high_salience_attended = 0
    high_salience_in_attended = 0

    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials}...", end=" ", flush=True)

        # Create consciousness
        consciousness = ATPTunedConsciousness(
            identity_name=f"quality-analysis-{config_name}-trial{trial}",
            attention_cost=attention_cost,
            rest_recovery=rest_recovery
        )

        # Track saliences
        attended_saliences = []
        unattended_saliences = []

        # Run cycles
        for _ in range(cycles):
            # Generate observation
            observations = generate_realistic_observations(1)
            obs = observations[0]

            salience = obs.salience
            total_observations += 1

            # Track high-salience observations
            if salience > high_salience_threshold:
                high_salience_count += 1

            # Process cycle and check if attended
            # We need to track before processing to know salience
            initial_attended_count = consciousness.observations_attended

            consciousness.process_cycle(observations)

            # Check if this observation was attended
            if consciousness.observations_attended > initial_attended_count:
                # Was attended
                attended_saliences.append(salience)
                total_attended += 1

                if salience > high_salience_threshold:
                    high_salience_attended += 1
                    high_salience_in_attended += 1
            else:
                # Was not attended
                unattended_saliences.append(salience)

        all_attended_saliences.extend(attended_saliences)
        all_unattended_saliences.extend(unattended_saliences)

        # Trial stats
        trial_attn_rate = len(attended_saliences) / cycles
        trial_avg_attended = sum(attended_saliences) / len(attended_saliences) if attended_saliences else 0
        print(f"Attn: {trial_attn_rate*100:.1f}%, Avg Salience: {trial_avg_attended:.3f}")

    # Calculate aggregate metrics
    attention_rate = total_attended / total_observations
    avg_attended_salience = sum(all_attended_saliences) / len(all_attended_saliences) if all_attended_saliences else 0
    avg_unattended_salience = sum(all_unattended_saliences) / len(all_unattended_saliences) if all_unattended_saliences else 0

    # Coverage: % of high-salience observations that were attended
    high_salience_coverage = high_salience_attended / high_salience_count if high_salience_count > 0 else 0

    # Precision: % of attended observations that were high-salience
    precision = high_salience_in_attended / total_attended if total_attended > 0 else 0

    # Efficiency: How much coverage per unit of attention
    efficiency = high_salience_coverage / attention_rate if attention_rate > 0 else 0

    # Histogram of attended saliences
    attended_bins = {'0.0-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-0.9': 0, '0.9-1.0': 0}
    for sal in all_attended_saliences:
        if sal < 0.3:
            attended_bins['0.0-0.3'] += 1
        elif sal < 0.5:
            attended_bins['0.3-0.5'] += 1
        elif sal < 0.7:
            attended_bins['0.5-0.7'] += 1
        elif sal < 0.9:
            attended_bins['0.7-0.9'] += 1
        else:
            attended_bins['0.9-1.0'] += 1

    # Normalize histogram
    total_attended_for_hist = sum(attended_bins.values())
    salience_histogram = {k: v/total_attended_for_hist for k, v in attended_bins.items()} if total_attended_for_hist > 0 else attended_bins

    return QualityMetrics(
        attention_rate=attention_rate,
        avg_attended_salience=avg_attended_salience,
        avg_unattended_salience=avg_unattended_salience,
        high_salience_coverage=high_salience_coverage,
        precision=precision,
        efficiency=efficiency,
        salience_histogram=salience_histogram,
        attended_salience_dist=all_attended_saliences,
        unattended_salience_dist=all_unattended_saliences
    )


def main():
    print("="*80)
    print("ATTENTION QUALITY VS QUANTITY ANALYSIS")
    print("="*80)
    print()
    print("Research Question: Does higher attention rate degrade selectivity?")
    print()
    print("Hypothesis:")
    print("  - Maximum (62%): Lower selectivity, higher coverage")
    print("  - Balanced (42%): Higher selectivity, good coverage")
    print("  - Conservative (26%): Highest selectivity, lower coverage")
    print()
    print("Metrics:")
    print("  - Selectivity: Avg attended salience")
    print("  - Coverage: % of high-salience (>0.7) attended")
    print("  - Precision: % of attended that are high-salience")
    print("  - Efficiency: Coverage per unit attention")
    print()

    # Configurations from Session 12
    configurations = [
        {
            'name': 'Maximum (62%)',
            'cost': 0.01,
            'recovery': 0.05,
            'expected': 'Lower selectivity, higher coverage'
        },
        {
            'name': 'Balanced (42%)',
            'cost': 0.03,
            'recovery': 0.04,
            'expected': 'Higher selectivity, good coverage'
        },
        {
            'name': 'Conservative (26%)',
            'cost': 0.05,
            'recovery': 0.02,
            'expected': 'Highest selectivity, lower coverage'
        }
    ]

    results = []

    for config in configurations:
        print(f"\n{'='*80}")
        print(f"Configuration: {config['name']}")
        print(f"  Parameters: cost={config['cost']:.3f}, recovery={config['recovery']:.3f}")
        print(f"  Expected: {config['expected']}")

        metrics = analyze_quality(
            config_name=config['name'],
            attention_cost=config['cost'],
            rest_recovery=config['recovery'],
            cycles=2000,
            trials=3
        )

        results.append({
            'name': config['name'],
            'config': config,
            'metrics': metrics
        })

        print(f"\n  RESULTS:")
        print(f"    Attention Rate: {metrics.attention_rate*100:.1f}%")
        print(f"    Avg Attended Salience: {metrics.avg_attended_salience:.3f}")
        print(f"    Avg Unattended Salience: {metrics.avg_unattended_salience:.3f}")
        print(f"    High-Salience Coverage: {metrics.high_salience_coverage*100:.1f}%")
        print(f"    Precision: {metrics.precision*100:.1f}%")
        print(f"    Efficiency: {metrics.efficiency:.3f}")

    # Comparative analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    print()

    # Sort by attention rate
    results_sorted = sorted(results, key=lambda r: r['metrics'].attention_rate, reverse=True)

    print("Summary Table:\n")
    print(f"{'Config':<20} {'Attention':<12} {'Selectivity':<14} {'Coverage':<12} {'Precision':<12} {'Efficiency':<12}")
    print("-"*80)

    for res in results_sorted:
        m = res['metrics']
        print(f"{res['name']:<20} {m.attention_rate*100:5.1f}%      {m.avg_attended_salience:.3f}         "
              f"{m.high_salience_coverage*100:5.1f}%      {m.precision*100:5.1f}%      {m.efficiency:.3f}")

    print()

    # Analysis: Selectivity vs Coverage trade-off
    print("="*80)
    print("SELECTIVITY VS COVERAGE TRADE-OFF")
    print("="*80)
    print()

    max_config = results_sorted[0]
    bal_config = next((r for r in results if 'Balanced' in r['name']), None)
    con_config = results_sorted[-1]

    print("1. Selectivity Analysis (Avg Attended Salience):\n")
    for res in results_sorted:
        m = res['metrics']
        delta_vs_unattended = m.avg_attended_salience - m.avg_unattended_salience
        selectivity = "HIGH" if m.avg_attended_salience > 0.75 else "MEDIUM" if m.avg_attended_salience > 0.70 else "LOW"

        print(f"   {res['name']:<20}: {m.avg_attended_salience:.3f} ({selectivity}) "
              f"[+{delta_vs_unattended:.3f} vs unattended]")

    print("\n2. Coverage Analysis (% High-Salience Captured):\n")
    for res in results_sorted:
        m = res['metrics']
        coverage_level = "EXCELLENT" if m.high_salience_coverage > 0.8 else "GOOD" if m.high_salience_coverage > 0.6 else "MODERATE"

        print(f"   {res['name']:<20}: {m.high_salience_coverage*100:5.1f}% ({coverage_level})")

    print("\n3. Precision Analysis (% Attended that are High-Quality):\n")
    for res in results_sorted:
        m = res['metrics']
        precision_level = "HIGH" if m.precision > 0.6 else "MEDIUM" if m.precision > 0.4 else "LOW"

        print(f"   {res['name']:<20}: {m.precision*100:5.1f}% ({precision_level})")

    print("\n4. Efficiency Analysis (Coverage per unit Attention):\n")
    for res in results_sorted:
        m = res['metrics']
        efficiency_level = "EXCELLENT" if m.efficiency > 1.5 else "GOOD" if m.efficiency > 1.2 else "MODERATE"

        print(f"   {res['name']:<20}: {m.efficiency:.3f} ({efficiency_level})")

    # Attended salience distributions
    print("\n" + "="*80)
    print("ATTENDED SALIENCE DISTRIBUTIONS")
    print("="*80)
    print()

    for res in results_sorted:
        print(f"{res['name']}:")
        hist = res['metrics'].salience_histogram
        for range_name in ['0.0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']:
            pct = hist[range_name] * 100
            bar = '█' * int(pct / 2)
            print(f"  {range_name}: {bar:<50} {pct:5.1f}%")
        print()

    # Key findings
    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    print()

    # Compare max vs balanced
    if bal_config:
        max_m = max_config['metrics']
        bal_m = bal_config['metrics']

        selectivity_diff = bal_m.avg_attended_salience - max_m.avg_attended_salience
        coverage_diff = max_m.high_salience_coverage - bal_m.high_salience_coverage
        efficiency_diff = bal_m.efficiency - max_m.efficiency

        print("Maximum (62%) vs Balanced (42%):")
        print(f"  Attention difference: {(max_m.attention_rate - bal_m.attention_rate)*100:+.1f}%")
        print(f"  Selectivity difference: {selectivity_diff:+.3f} salience")
        print(f"  Coverage difference: {coverage_diff*100:+.1f}%")
        print(f"  Efficiency difference: {efficiency_diff:+.3f}")
        print()

        if abs(selectivity_diff) < 0.02:
            print("✅ FINDING: Selectivity is maintained across configurations!")
            print("   Maximum attention does NOT degrade quality significantly.")
            print()
        elif selectivity_diff > 0.02:
            print("⚠️  FINDING: Balanced config is more selective")
            print(f"   {bal_m.avg_attended_salience:.3f} vs {max_m.avg_attended_salience:.3f} salience")
            print()

        if coverage_diff > 0.1:
            print("✅ FINDING: Maximum config provides superior coverage")
            print(f"   Captures {coverage_diff*100:.1f}% more high-salience observations")
            print()

    # Recommendation
    print("="*80)
    print("RECOMMENDATION")
    print("="*80)
    print()

    # Find best overall
    # Score: 0.4*selectivity + 0.4*coverage + 0.2*efficiency
    scored_results = []
    for res in results:
        m = res['metrics']
        # Normalize selectivity to [0,1] assuming 0.6-0.85 range
        norm_selectivity = (m.avg_attended_salience - 0.6) / 0.25
        norm_selectivity = max(0, min(1, norm_selectivity))

        # Coverage already [0,1]
        norm_coverage = m.high_salience_coverage

        # Normalize efficiency assuming 0.8-1.8 range
        norm_efficiency = (m.efficiency - 0.8) / 1.0
        norm_efficiency = max(0, min(1, norm_efficiency))

        score = 0.4 * norm_selectivity + 0.4 * norm_coverage + 0.2 * norm_efficiency

        scored_results.append({
            'name': res['name'],
            'score': score,
            'metrics': m
        })

    scored_results.sort(key=lambda x: x['score'], reverse=True)
    best = scored_results[0]

    print(f"Best Overall Configuration: {best['name']}")
    print(f"  Score: {best['score']:.3f}")
    print(f"  Attention: {best['metrics'].attention_rate*100:.1f}%")
    print(f"  Selectivity: {best['metrics'].avg_attended_salience:.3f}")
    print(f"  Coverage: {best['metrics'].high_salience_coverage*100:.1f}%")
    print(f"  Efficiency: {best['metrics'].efficiency:.3f}")
    print()

    print("Use Case Recommendations:")
    print()
    print("  Maximum (62%):")
    print("    - When: Need maximum environmental awareness")
    print("    - Benefit: Highest coverage of important events")
    print("    - Trade-off: Higher energy consumption")
    print()
    print("  Balanced (42%):")
    print("    - When: General-purpose consciousness")
    print("    - Benefit: Good selectivity + good coverage + sustainable")
    print("    - Trade-off: Balanced (no extremes)")
    print()
    print("  Conservative (26%):")
    print("    - When: Energy-constrained environments")
    print("    - Benefit: Maximum selectivity, lowest energy")
    print("    - Trade-off: Lower coverage of environment")
    print()


if __name__ == "__main__":
    main()
