#!/usr/bin/env python3
"""
Prediction S2 Validation: Health-Coherence Correlation

From Session #48 (SYNTHON_FORMATION_ATP_COUPLING_INTEGRATION.md):

"Prediction S2: Health-Coherence Correlation
Hypothesis: Synthon health H(p) ≈ C(p) with R² > 0.8

Test: Calculate all 7 health metrics for Session #47 data, correlate with
measured coherence.

Expected: Strong correlation between composite health and collective coherence
Falsification: R² < 0.5 or non-monotonic relationship"

This script takes Session #47 experimental results and calculates the 7 synthon
health metrics from Legion Session #23 framework:

1. Trust entropy (from C_conv)
2. Clustering coefficient (from task network)
3. MRH overlap (from shared knowledge)
4. ATP flow stability (from task completion)
5. Witness diversity (from observation sources)
6. Trust variance (from belief variance)
7. Boundary permeability (from external information rate)

Then correlates composite health H(p) with measured coherence C(p).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr


def compute_trust_entropy_from_conv(C_conv: float) -> float:
    """
    Trust entropy from convergence metric.

    C_conv measures inter-plugin belief similarity (cosine similarity).
    High C_conv → low entropy (coherent beliefs)
    Low C_conv → high entropy (divergent beliefs)

    Mapping: entropy ≈ 1 - C_conv
    """
    return 1.0 - C_conv


def compute_clustering_from_tasks(tasks_created: float, n_plugins: int = 5) -> float:
    """
    Clustering coefficient from task network density.

    Max possible tasks per round (K plugins, K-1 targets each):
    max_tasks = K × (K-1) = 5 × 4 = 20 per round

    Across 10 rounds: max = 200 tasks

    Clustering ≈ tasks_created / max_tasks
    """
    max_tasks_per_round = n_plugins * (n_plugins - 1)
    n_rounds = 10
    max_tasks = max_tasks_per_round * n_rounds

    return min(tasks_created / max_tasks, 1.0)


def compute_mrh_overlap_from_facts(facts_known: float, n_facts: int = 50) -> float:
    """
    MRH overlap from shared knowledge coverage.

    Individual observation budget: 20 obs (40% coverage)
    Full knowledge: 50 facts

    MRH overlap ≈ (facts_known - individual_baseline) / (max - baseline)
                = (facts_known - 20) / (50 - 20)
                = (facts_known - 20) / 30
    """
    individual_baseline = 20.0
    max_facts = n_facts
    return max((facts_known - individual_baseline) / (max_facts - individual_baseline), 0.0)


def compute_atp_flow_stability(tasks_created: float, tasks_completed: float) -> float:
    """
    ATP flow stability from task completion rate.

    Completion rate = tasks_completed / tasks_created (if > 0)

    For Session #47, tasks_completed not tracked, but we can estimate:
    - In well-functioning system: completion_rate ≈ 0.8-1.0
    - We'll use a proxy: stability ≈ 0.5 + 0.5 × (tasks > 0)

    This is a conservative estimate - real implementation would track actual completions.
    """
    if tasks_created == 0:
        return 0.5  # Baseline (no flow)

    # Proxy: assume ~80% completion rate when tasks exist
    # Real implementation: tasks_completed / tasks_created
    estimated_completion_rate = 0.8
    return 0.5 + 0.5 * estimated_completion_rate


def compute_witness_diversity(n_plugins: int = 5) -> float:
    """
    Witness diversity from observation source variety.

    In Session #47, each plugin observes independently (full diversity).
    Diversity = 1.0 when all K sources are active.

    Real implementation would track actual observation sources per fact.
    """
    return 1.0  # Full diversity (all plugins observe independently)


def compute_trust_variance_from_conv(C_conv: float, C_std: float = 0.03) -> float:
    """
    Trust variance from belief variance.

    High variance → low trust consistency → low health
    Low variance → high trust consistency → high health

    Metric: 1 - variance_normalized

    From Session #47, C_std ≈ 0.02-0.04 (standard deviation of C across reps)
    """
    # Normalize variance to [0, 1] range
    # Assume max variance ≈ 0.1 (10% spread)
    max_variance = 0.1
    normalized_variance = min(C_std / max_variance, 1.0)

    return 1.0 - normalized_variance


def compute_boundary_permeability(facts_known: float, n_facts: int = 50) -> float:
    """
    Boundary permeability from external information rate.

    High permeability → porous boundary → low health
    Low permeability → tight boundary → high health

    For synthons, we want SOME permeability (to acquire knowledge) but not too much
    (to maintain coherence).

    Optimal around facts_known ≈ 40-45 (80-90% of total)
    Too low (< 20) → too porous (still acquiring)
    Too high (> 48) → too rigid (not learning)

    Metric: 1 - |facts_known - optimal| / optimal
    """
    optimal_facts = 42.0  # ~84% coverage (from Session #47 p=1 data)
    deviation = abs(facts_known - optimal_facts)
    max_deviation = optimal_facts

    return max(1.0 - (deviation / max_deviation), 0.0)


def compute_synthon_health(C_conv: float, C_std: float, facts_known: float,
                           tasks_created: float, n_plugins: int = 5,
                           n_facts: int = 50) -> Dict[str, float]:
    """
    Compute all 7 synthon health metrics and composite health.

    Returns dict with individual metrics and composite H (mean of 7).
    """
    metrics = {
        'trust_entropy_health': 1.0 - compute_trust_entropy_from_conv(C_conv),
        'clustering': compute_clustering_from_tasks(tasks_created, n_plugins),
        'mrh_overlap': compute_mrh_overlap_from_facts(facts_known, n_facts),
        'atp_flow_stability': compute_atp_flow_stability(tasks_created, 0),  # No completion data
        'witness_diversity': compute_witness_diversity(n_plugins),
        'trust_consistency': compute_trust_variance_from_conv(C_conv, C_std),
        'boundary_integrity': compute_boundary_permeability(facts_known, n_facts),
    }

    # Composite health (mean of 7 metrics)
    metrics['H'] = np.mean(list(metrics.values()))

    return metrics


def validate_prediction_s2():
    """
    Validate Prediction S2: H(p) ≈ C(p) with R² > 0.8

    Load Session #47 data, calculate health metrics, correlate with coherence.
    """
    # Load Session #47 results
    results_file = Path(__file__).parent / "results" / "prediction_4a_refined_results.json"

    if not results_file.exists():
        print(f"ERROR: Session #47 results not found at {results_file}")
        print("Run prediction_4a_refined.py first to generate data.")
        return

    with open(results_file, 'r') as f:
        data = json.load(f)

    results = data['results']

    print("=" * 80)
    print("PREDICTION S2: HEALTH-COHERENCE CORRELATION VALIDATION")
    print("=" * 80)
    print()
    print("Hypothesis: Synthon health H(p) ≈ C(p) with R² > 0.8")
    print()
    print("Calculating 7 health metrics for each coupling level...")
    print()

    # Calculate health metrics for each coupling level
    health_data = []

    for result in results:
        p = result['coupling_p']
        C = result['C']
        C_conv = result['C_conv']
        C_corr = result['C_corr']
        C_std = result['C_std']
        facts_known = result['avg_facts_known']
        tasks_created = result['tasks_created']

        # Compute health metrics
        health = compute_synthon_health(
            C_conv=C_conv,
            C_std=C_std,
            facts_known=facts_known,
            tasks_created=tasks_created
        )

        health_data.append({
            'p': p,
            'C': C,
            'H': health['H'],
            **health  # Include individual metrics
        })

    # Extract arrays for correlation
    p_values = np.array([h['p'] for h in health_data])
    C_values = np.array([h['C'] for h in health_data])
    H_values = np.array([h['H'] for h in health_data])

    # Correlation analysis
    pearson_r, pearson_p = pearsonr(H_values, C_values)
    spearman_r, spearman_p = spearmanr(H_values, C_values)

    r_squared = pearson_r ** 2

    print("=" * 80)
    print("CORRELATION RESULTS")
    print("=" * 80)
    print()
    print(f"Pearson correlation:")
    print(f"  r = {pearson_r:.4f}")
    print(f"  R² = {r_squared:.4f}")
    print(f"  p-value = {pearson_p:.6f}")
    print()
    print(f"Spearman correlation:")
    print(f"  ρ = {spearman_r:.4f}")
    print(f"  p-value = {spearman_p:.6f}")
    print()

    # Validation
    if r_squared > 0.8:
        print(f"✓ VALIDATED: R² = {r_squared:.4f} > 0.8")
        print(f"  Strong positive correlation between synthon health and coherence")
    elif r_squared > 0.5:
        print(f"~ PARTIAL: R² = {r_squared:.4f} ∈ [0.5, 0.8]")
        print(f"  Moderate correlation, but weaker than predicted")
    else:
        print(f"✗ FALSIFIED: R² = {r_squared:.4f} < 0.5")
        print(f"  Weak or no correlation between health and coherence")
    print()

    # Detailed metric analysis
    print("=" * 80)
    print("INDIVIDUAL HEALTH METRIC CONTRIBUTIONS")
    print("=" * 80)
    print()

    metric_names = [
        'trust_entropy_health', 'clustering', 'mrh_overlap',
        'atp_flow_stability', 'witness_diversity',
        'trust_consistency', 'boundary_integrity'
    ]

    for metric_name in metric_names:
        metric_values = np.array([h[metric_name] for h in health_data])
        metric_r, _ = pearsonr(metric_values, C_values)
        metric_r2 = metric_r ** 2

        print(f"{metric_name:25s}: r = {metric_r:6.3f}, R² = {metric_r2:.3f}")

    print()

    # Key statistics
    print("=" * 80)
    print("KEY STATISTICS")
    print("=" * 80)
    print()

    print(f"Health Range:")
    print(f"  H(p=0) = {H_values[0]:.3f}")
    print(f"  H(p=1) = {H_values[-1]:.3f}")
    print(f"  ΔH = {H_values[-1] - H_values[0]:.3f}")
    print()

    print(f"Coherence Range (from Session #47):")
    print(f"  C(p=0) = {C_values[0]:.3f}")
    print(f"  C(p=1) = {C_values[-1]:.3f}")
    print(f"  ΔC = {C_values[-1] - C_values[0]:.3f}")
    print()

    # Find transition points
    H_transition_idx = np.argmax(np.abs(np.gradient(np.gradient(H_values, p_values), p_values)))
    C_transition_idx = np.argmax(np.abs(np.gradient(np.gradient(C_values, p_values), p_values)))

    print(f"Transition Points (max curvature):")
    print(f"  Health: p ≈ {p_values[H_transition_idx]:.4f}")
    print(f"  Coherence: p ≈ {p_values[C_transition_idx]:.4f}")
    print(f"  Difference: Δp = {abs(p_values[H_transition_idx] - p_values[C_transition_idx]):.4f}")
    print()

    # Plotting disabled due to matplotlib/numpy compatibility issues
    print("(Plotting skipped - matplotlib not available)")
    print()

    # Save results
    output_file = Path(__file__).parent / "results" / "prediction_s2_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'prediction': 's2_health_coherence_correlation',
            'hypothesis': 'H(p) ≈ C(p) with R² > 0.8',
            'validation_status': 'VALIDATED' if r_squared > 0.8 else ('PARTIAL' if r_squared > 0.5 else 'FALSIFIED'),
            'pearson_r': float(pearson_r),
            'r_squared': float(r_squared),
            'spearman_rho': float(spearman_r),
            'health_data': health_data,
            'metric_correlations': {
                metric: float(pearsonr(np.array([h[metric] for h in health_data]), C_values)[0])
                for metric in metric_names
            }
        }, f, indent=2)

    print(f"Results saved to: {output_file}")
    print("=" * 80)

    return health_data


if __name__ == '__main__':
    validate_prediction_s2()
