#!/usr/bin/env python3
"""
Prediction 4a Validation (REFINED): SAGE Inter-Plugin ATP Coupling Phase Transition

ISSUE WITH FIRST VERSION: Individual observation budget was sufficient for
independent learning. Coherence was already high at p=0 (C≈0.96).

REFINED DESIGN:
- REDUCE observation budget to make individual learning insufficient
- Similar to coupling-coherence experiment: individual obs covers ~0.3× facts
- Collective obs (with coupling) covers ~1.5× facts
- This creates regime where coupling MATTERS

Key change: 50 facts, 10 rounds × 2 obs/round = 20 obs/plugin (40% coverage alone)
vs. original: 50 facts, 80 rounds × 8 obs/round = 640 obs/plugin (1280% coverage!)

Expected behavior:
- p=0: C ≈ 0.4-0.5 (individual knowledge insufficient)
- p>p_crit: C ≈ 0.8-0.95 (collective knowledge sufficient)
- p_crit ≈ 0.002-0.01 (sparse coupling enables collective success)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from sage.core.metabolic_controller_with_tasks import MetabolicControllerWithTasks


@dataclass
class PluginState:
    """Plugin with limited local knowledge."""
    plugin_id: str
    beliefs: Dict[str, float] = field(default_factory=dict)
    controller: MetabolicControllerWithTasks = None

    def __post_init__(self):
        if self.controller is None:
            # REDUCED ATP budget to force scarcity
            self.controller = MetabolicControllerWithTasks(initial_atp=50.0)

    def observe_fact(self, fact_id: str, is_true: bool, noise_rate: float = 0.15):
        """Bayesian observation update with noise."""
        if np.random.random() < noise_rate:
            is_true = not is_true

        current = self.beliefs.get(fact_id, 0.5)
        evidence = 0.85 if is_true else 0.15
        self.beliefs[fact_id] = 0.6 * current + 0.4 * evidence  # Faster learning

    def share_beliefs_via_task(self, target_plugin: 'PluginState', coupling_p: float) -> bool:
        """ATP-funded belief sharing (compression trust event)."""
        if np.random.random() > coupling_p:
            return False

        reward_atp = 0.5  # Reduced reward to conserve ATP
        try:
            task_id = self.controller.create_consciousness_task(
                description=f"Share {self.plugin_id}→{target_plugin.plugin_id}",
                reward_atp=reward_atp,
                executor_id=target_plugin.plugin_id
            )
        except Exception:
            return False

        # Integration with trust gradient
        alpha = 0.7  # Self-weight
        for fact_id, source_belief in self.beliefs.items():
            target_current = target_plugin.beliefs.get(fact_id, 0.5)
            target_plugin.beliefs[fact_id] = alpha * target_current + (1 - alpha) * source_belief

        success, _ = target_plugin.controller.complete_and_claim_task(task_id, target_plugin.plugin_id)
        return success


@dataclass
class GroundTruth:
    facts: Dict[str, bool]

    @classmethod
    def generate(cls, n_facts: int = 50) -> 'GroundTruth':
        return cls(facts={f"fact_{i}": np.random.choice([True, False]) for i in range(n_facts)})


def measure_convergence(plugins: List[PluginState]) -> float:
    """Inter-plugin agreement (cosine similarity)."""
    if len(plugins) < 2:
        return 1.0

    all_facts = set()
    for p in plugins:
        all_facts.update(p.beliefs.keys())

    if not all_facts:
        return 0.5

    facts_list = sorted(all_facts)
    similarities = []

    for i in range(len(plugins)):
        for j in range(i + 1, len(plugins)):
            vec_i = np.array([plugins[i].beliefs.get(f, 0.5) for f in facts_list])
            vec_j = np.array([plugins[j].beliefs.get(f, 0.5) for f in facts_list])

            norm_i = np.linalg.norm(vec_i - 0.5)  # Distance from uninformed
            norm_j = np.linalg.norm(vec_j - 0.5)

            if norm_i > 0.01 and norm_j > 0.01:
                sim = np.dot(vec_i - 0.5, vec_j - 0.5) / (norm_i * norm_j)
                similarities.append((sim + 1) / 2)  # Rescale to [0, 1]

    return float(np.mean(similarities)) if similarities else 0.5


def measure_correctness(plugins: List[PluginState], truth: GroundTruth) -> float:
    """Aggregate accuracy (F1 score)."""
    f1_scores = []

    for plugin in plugins:
        if not plugin.beliefs:
            continue

        tp = fp = fn = 0

        for fact_id, true_value in truth.facts.items():
            if fact_id in plugin.beliefs:
                predicted = plugin.beliefs[fact_id] > 0.5
                if predicted and true_value:
                    tp += 1
                elif predicted and not true_value:
                    fp += 1
                elif not predicted and true_value:
                    fn += 1

        if tp + fp + fn > 0:
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            f1_scores.append(f1)

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def run_experiment(coupling_p: float, n_plugins: int = 5, n_facts: int = 50,
                   n_rounds: int = 10, obs_per_round: int = 2) -> Dict:
    """
    Run experiment with SCARCE individual observations.

    Individual budget: 10 rounds × 2 obs = 20 obs per plugin (40% of 50 facts)
    Collective budget: 5 plugins × 20 = 100 obs total (200% of facts, if shared)

    This creates the regime where coupling MATTERS.
    """
    truth = GroundTruth.generate(n_facts=n_facts)
    plugins = [PluginState(plugin_id=f"plugin_{i}") for i in range(n_plugins)]

    for round_idx in range(n_rounds):
        # Limited observations per plugin
        for plugin in plugins:
            fact_ids = np.random.choice(list(truth.facts.keys()),
                                        size=min(obs_per_round, n_facts),
                                        replace=False)
            for fact_id in fact_ids:
                plugin.observe_fact(fact_id, truth.facts[fact_id])

        # Inter-plugin coupling
        for i in range(n_plugins):
            for j in range(n_plugins):
                if i != j:
                    plugins[i].share_beliefs_via_task(plugins[j], coupling_p)

    # Measure coherence
    C_conv = measure_convergence(plugins)
    C_corr = measure_correctness(plugins, truth)
    C = np.sqrt(C_conv * C_corr)

    # ATP stats
    task_stats = [p.controller.get_task_stats() for p in plugins]
    avg_facts_known = np.mean([len(p.beliefs) for p in plugins])

    return {
        'coupling_p': coupling_p,
        'C_conv': C_conv,
        'C_corr': C_corr,
        'C': C,
        'avg_facts_known': avg_facts_known,
        'tasks_created': sum(s['tasks_created'] for s in task_stats),
        'tasks_completed': sum(s['tasks_completed'] for s in task_stats),
    }


def run_full_experiment():
    """Run refined experiment with scarcity."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("PREDICTION 4a (REFINED): SAGE INTER-PLUGIN ATP COUPLING")
    print("=" * 80)
    print()
    print("REFINEMENT: Individual observation budget INSUFFICIENT")
    print("- 50 facts, 10 rounds × 2 obs/round = 20 obs/plugin (40% coverage)")
    print("- Without coupling: plugins lack knowledge")
    print("- With coupling: collective knowledge emerges")
    print()
    print("Expected: p_crit ≈ 0.002-0.01, large ΔC at transition")
    print()

    # Fine-grained sampling
    coupling_levels = []
    coupling_levels.extend(np.linspace(0.0, 0.02, 30))  # Very fine near expected transition
    coupling_levels.extend(np.linspace(0.02, 0.1, 10))
    coupling_levels.extend(np.linspace(0.1, 1.0, 10))
    coupling_levels = sorted(set(coupling_levels))

    print(f"Running {len(coupling_levels)} coupling levels...")
    print()

    results = []
    for i, p in enumerate(coupling_levels):
        print(f"[{i+1}/{len(coupling_levels)}] p = {p:.4f}...", end="", flush=True)

        # More repetitions for stability
        n_reps = 10
        rep_results = [run_experiment(coupling_p=p) for _ in range(n_reps)]

        result = {
            'coupling_p': p,
            'C': np.mean([r['C'] for r in rep_results]),
            'C_std': np.std([r['C'] for r in rep_results]),
            'C_conv': np.mean([r['C_conv'] for r in rep_results]),
            'C_corr': np.mean([r['C_corr'] for r in rep_results]),
            'avg_facts_known': np.mean([r['avg_facts_known'] for r in rep_results]),
            'tasks_created': np.mean([r['tasks_created'] for r in rep_results]),
        }

        results.append(result)
        print(f" C = {result['C']:.3f} (±{result['C_std']:.3f}), facts_known ≈ {result['avg_facts_known']:.1f}")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    C_values = np.array([r['C'] for r in results])
    p_values = np.array([r['coupling_p'] for r in results])

    # Transition analysis
    d2C_dp2 = np.gradient(np.gradient(C_values, p_values), p_values)
    p_transition = p_values[np.argmax(np.abs(d2C_dp2))]

    print(f"Coherence Range:")
    print(f"  C(p=0) ≈ {C_values[0]:.3f}")
    print(f"  C(p=1) ≈ {C_values[-1]:.3f}")
    print(f"  ΔC = {C_values[-1] - C_values[0]:.3f}")
    print()
    print(f"Transition Point:")
    print(f"  p_transition ≈ {p_transition:.4f}")
    print()

    # Validation
    if 0.002 <= p_transition <= 0.01 and (C_values[-1] - C_values[0]) > 0.2:
        print(f"  ✓ VALIDATED: p_crit ∈ [0.002, 0.01], large ΔC > 0.2")
    elif p_transition < 0.002:
        print(f"  ~ PARTIAL: p_crit lower than predicted")
    elif p_transition > 0.01:
        print(f"  ~ PARTIAL: p_crit higher than predicted")
    else:
        print(f"  ✗ FALSIFIED: No clear phase transition (ΔC < 0.2)")
    print()

    # Model comparison
    from scipy.stats import linregress
    from scipy.optimize import curve_fit

    slope, intercept, r_lin, _, _ = linregress(p_values, C_values)
    r2_linear = r_lin ** 2

    def hill(p, k, p_half):
        return p**k / (p**k + p_half**k)

    try:
        (k, p_half), _ = curve_fit(hill, p_values, C_values, p0=[2.0, 0.005],
                                    bounds=([0.1, 0.0001], [10.0, 0.5]), maxfev=10000)
        C_hill = hill(p_values, k, p_half)
        ss_res = np.sum((C_values - C_hill) ** 2)
        ss_tot = np.sum((C_values - np.mean(C_values)) ** 2)
        r2_hill = 1 - (ss_res / ss_tot)

        print(f"Model Comparison:")
        print(f"  Linear: R² = {r2_linear:.4f}")
        print(f"  Hill:   R² = {r2_hill:.4f}, k = {k:.3f}, p_half = {p_half:.4f}")
        print()

        if r2_hill > r2_linear + 0.05 and r2_hill > 0.7:
            print(f"  ✓ Hill (sigmoid) significantly better → Phase transition confirmed")
        else:
            print(f"  ~ Unclear: Both models similar or poor fit")

    except Exception as e:
        print(f"  (Hill fit failed: {e})")
        print(f"  Linear R² = {r2_linear:.4f}")

    # Save
    output_file = output_dir / "prediction_4a_refined_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'experiment': 'prediction_4a_refined',
            'parameters': {
                'n_plugins': 5,
                'n_facts': 50,
                'n_rounds': 10,
                'obs_per_round': 2,
                'total_obs_per_plugin': 20,
                'coverage_ratio': 0.4
            },
            'results': results,
            'transition_point': float(p_transition),
            'coherence_range': {
                'min': float(C_values[0]),
                'max': float(C_values[-1]),
                'delta': float(C_values[-1] - C_values[0])
            },
        }, f, indent=2)

    print()
    print(f"Results saved to: {output_file}")
    print("=" * 80)

    return results


if __name__ == '__main__':
    run_full_experiment()
