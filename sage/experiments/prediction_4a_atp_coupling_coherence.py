#!/usr/bin/env python3
"""
Prediction 4a Validation: SAGE Inter-Plugin ATP Coupling Phase Transition

From COUPLING_COHERENCE_SAGE_SYNTHESIS.md (Session #46):

"Test prediction 4a: Vary ATP budget allocation to inter-plugin communication
(equivalent to coupling parameter p). Expected: sigmoid phase transition with
p_crit ≈ 0.002-0.01 (sparse communication suffices)."

This experiment simulates multiple SAGE plugins (analogous to agents in the
coupling-coherence experiment) sharing information via ATP-funded tasks.

Design:
- K plugins, each with local knowledge/belief state
- Coupling parameter p: fraction of ATP budget allocated to inter-plugin tasks
- Measure: Collective coherence (convergence + correctness)
- Expected: Sigmoid transition at p_crit ≈ 0.002-0.01

Connection to coupling-coherence experiment:
- p (coupling frequency) → ATP budget for communication
- Belief matrix → Plugin internal state
- Compression trust event → ATP-funded information sharing task
- C_conv (convergence) → Inter-plugin state similarity
- C_corr (correctness) → Aggregate task completion quality
- C = √(C_conv × C_corr) → Genuine collective coherence
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import sys

# Add sage module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.metabolic_controller_with_tasks import MetabolicControllerWithTasks


@dataclass
class PluginState:
    """Simple plugin state: knowledge of binary facts."""
    plugin_id: str
    beliefs: Dict[str, float] = field(default_factory=dict)  # fact_id -> confidence
    controller: MetabolicControllerWithTasks = None

    def __post_init__(self):
        if self.controller is None:
            self.controller = MetabolicControllerWithTasks(initial_atp=100.0)

    def observe_fact(self, fact_id: str, is_true: bool, noise_rate: float = 0.15):
        """Observe a fact with noise (15% chance of error)."""
        if np.random.random() < noise_rate:
            is_true = not is_true  # Flip observation

        # Bayesian update (simplified)
        current = self.beliefs.get(fact_id, 0.5)
        evidence = 0.9 if is_true else 0.1
        # Update: weighted average with new evidence
        self.beliefs[fact_id] = 0.7 * current + 0.3 * evidence

    def share_beliefs_via_task(self, target_plugin: 'PluginState', coupling_p: float,
                                task_overhead: float = 0.1) -> bool:
        """
        Share beliefs with target plugin via ATP-funded task.

        Returns True if task was created and completed (coupling occurred).
        """
        # Only attempt coupling with probability p
        if np.random.random() > coupling_p:
            return False

        # Calculate task reward (proportional to information value)
        reward_atp = 1.0  # Fixed reward for simplicity

        # Create task
        try:
            task_id = self.controller.create_consciousness_task(
                description=f"Share beliefs from {self.plugin_id} to {target_plugin.plugin_id}",
                reward_atp=reward_atp,
                executor_id=target_plugin.plugin_id
            )
        except Exception:
            # Insufficient ATP
            return False

        # Execute task: target integrates source beliefs
        # (In real SAGE, this would be message passing + integration)
        alpha = 0.7  # Self-weight (trust gradient)
        for fact_id, source_belief in self.beliefs.items():
            target_current = target_plugin.beliefs.get(fact_id, 0.5)
            # Weighted integration
            target_plugin.beliefs[fact_id] = alpha * target_current + (1 - alpha) * source_belief

        # Complete task and claim reward
        success, _ = target_plugin.controller.complete_and_claim_task(task_id, target_plugin.plugin_id)

        return success


@dataclass
class GroundTruth:
    """Ground truth: which facts are actually true."""
    facts: Dict[str, bool]  # fact_id -> is_true

    @classmethod
    def generate(cls, n_facts: int = 50) -> 'GroundTruth':
        """Generate random ground truth."""
        facts = {f"fact_{i}": np.random.choice([True, False]) for i in range(n_facts)}
        return cls(facts=facts)


def measure_convergence(plugins: List[PluginState]) -> float:
    """
    Measure inter-plugin convergence (agreement).

    Returns mean pairwise belief similarity.
    """
    if len(plugins) < 2:
        return 1.0

    # Get all facts mentioned by any plugin
    all_facts = set()
    for plugin in plugins:
        all_facts.update(plugin.beliefs.keys())

    if not all_facts:
        return 0.5  # No beliefs yet

    # Calculate pairwise similarities
    similarities = []
    for i in range(len(plugins)):
        for j in range(i + 1, len(plugins)):
            # Cosine similarity of belief vectors
            facts_list = list(all_facts)
            vec_i = np.array([plugins[i].beliefs.get(f, 0.5) for f in facts_list])
            vec_j = np.array([plugins[j].beliefs.get(f, 0.5) for f in facts_list])

            norm_i = np.linalg.norm(vec_i)
            norm_j = np.linalg.norm(vec_j)

            if norm_i > 0 and norm_j > 0:
                sim = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 0.5


def measure_correctness(plugins: List[PluginState], ground_truth: GroundTruth) -> float:
    """
    Measure aggregate correctness (agreement with ground truth).

    Returns mean accuracy across all plugins.
    """
    accuracies = []

    for plugin in plugins:
        if not plugin.beliefs:
            continue

        correct = 0
        total = 0

        for fact_id, true_value in ground_truth.facts.items():
            if fact_id in plugin.beliefs:
                belief = plugin.beliefs[fact_id]
                predicted = belief > 0.5
                if predicted == true_value:
                    correct += 1
                total += 1

        if total > 0:
            accuracies.append(correct / total)

    return float(np.mean(accuracies)) if accuracies else 0.5


def run_experiment(coupling_p: float, n_plugins: int = 5, n_facts: int = 50,
                   n_rounds: int = 80, obs_per_round: int = 8) -> Dict:
    """
    Run one experiment at a given coupling level.

    Args:
        coupling_p: Probability of inter-plugin communication (ATP budget fraction)
        n_plugins: Number of plugins (analogous to agents)
        n_facts: Number of facts in ground truth
        n_rounds: Number of observation rounds
        obs_per_round: Observations per plugin per round

    Returns:
        Dict with metrics: C_conv, C_corr, C (coherence)
    """
    # Generate ground truth
    truth = GroundTruth.generate(n_facts=n_facts)

    # Create plugins
    plugins = [PluginState(plugin_id=f"plugin_{i}") for i in range(n_plugins)]

    # Run rounds
    for round_idx in range(n_rounds):
        # Each plugin makes observations
        for plugin in plugins:
            # Randomly sample facts to observe
            fact_ids = np.random.choice(list(truth.facts.keys()), size=obs_per_round, replace=False)
            for fact_id in fact_ids:
                plugin.observe_fact(fact_id, truth.facts[fact_id])

        # Inter-plugin coupling (ATP-funded tasks)
        # Each plugin can share with others based on coupling_p
        for i in range(n_plugins):
            for j in range(n_plugins):
                if i != j:
                    plugins[i].share_beliefs_via_task(plugins[j], coupling_p)

    # Measure final coherence
    C_conv = measure_convergence(plugins)
    C_corr = measure_correctness(plugins, truth)
    C = np.sqrt(C_conv * C_corr)  # Geometric mean (catches "shared wrongness")

    # Get ATP statistics
    total_atp_remaining = sum(p.controller.atp_current for p in plugins)
    task_stats = [p.controller.get_task_stats() for p in plugins]
    total_tasks_created = sum(stats['tasks_created'] for stats in task_stats)
    total_tasks_completed = sum(stats['tasks_completed'] for stats in task_stats)

    return {
        'coupling_p': coupling_p,
        'C_conv': C_conv,
        'C_corr': C_corr,
        'C': C,
        'atp_remaining': total_atp_remaining,
        'tasks_created': total_tasks_created,
        'tasks_completed': total_tasks_completed,
    }


def run_full_experiment(output_dir: Path = None):
    """
    Run full experiment across coupling levels.

    Tests Prediction 4a: sigmoid transition with p_crit ≈ 0.002-0.01
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("PREDICTION 4a: SAGE INTER-PLUGIN ATP COUPLING PHASE TRANSITION")
    print("=" * 80)
    print()
    print("Testing hypothesis: Inter-plugin ATP budget allocation follows")
    print("sigmoid phase transition similar to coupling-coherence experiment.")
    print()
    print("Expected: p_crit ≈ 0.002-0.01 (sparse communication suffices)")
    print("Falsification: Linear relationship or p_crit >> 0.5")
    print()

    # Coupling levels (fine-grained near expected p_crit)
    coupling_levels = []
    # Very fine near 0
    coupling_levels.extend(np.linspace(0.0, 0.01, 20))
    # Medium resolution 0.01-0.1
    coupling_levels.extend(np.linspace(0.01, 0.1, 10))
    # Coarse above 0.1
    coupling_levels.extend(np.linspace(0.1, 1.0, 10))

    coupling_levels = sorted(set(coupling_levels))

    print(f"Running {len(coupling_levels)} coupling levels...")
    print(f"Parameters: K=5 plugins, 50 facts, 80 rounds, 8 obs/round")
    print()

    results = []

    for i, p in enumerate(coupling_levels):
        print(f"[{i+1}/{len(coupling_levels)}] p = {p:.4f}...", end="", flush=True)

        # Run multiple repetitions and average
        n_reps = 5
        rep_results = []
        for _ in range(n_reps):
            rep_results.append(run_experiment(coupling_p=p))

        # Average across repetitions
        result = {
            'coupling_p': p,
            'C_conv': np.mean([r['C_conv'] for r in rep_results]),
            'C_corr': np.mean([r['C_corr'] for r in rep_results]),
            'C': np.mean([r['C'] for r in rep_results]),
            'C_std': np.std([r['C'] for r in rep_results]),
            'atp_remaining': np.mean([r['atp_remaining'] for r in rep_results]),
            'tasks_created': np.mean([r['tasks_created'] for r in rep_results]),
            'tasks_completed': np.mean([r['tasks_completed'] for r in rep_results]),
        }

        results.append(result)
        print(f" C = {result['C']:.3f} (±{result['C_std']:.3f})")

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Find transition point (maximum curvature)
    C_values = np.array([r['C'] for r in results])
    p_values = np.array([r['coupling_p'] for r in results])

    # Numerical second derivative
    d2C_dp2 = np.gradient(np.gradient(C_values, p_values), p_values)
    max_curvature_idx = np.argmax(np.abs(d2C_dp2))
    p_transition = p_values[max_curvature_idx]

    print(f"Coherence Range:")
    print(f"  C(p=0) ≈ {C_values[0]:.3f}")
    print(f"  C(p=1) ≈ {C_values[-1]:.3f}")
    print(f"  ΔC = {C_values[-1] - C_values[0]:.3f}")
    print()
    print(f"Transition Point (maximum curvature):")
    print(f"  p_transition ≈ {p_transition:.4f}")
    print()
    print(f"Prediction 4a Status:")
    if 0.002 <= p_transition <= 0.01:
        print(f"  ✓ VALIDATED: p_crit ≈ {p_transition:.4f} within predicted range [0.002, 0.01]")
    elif p_transition < 0.002:
        print(f"  ~ PARTIAL: p_crit ≈ {p_transition:.4f} lower than predicted (even sparser)")
    elif 0.01 < p_transition < 0.05:
        print(f"  ~ PARTIAL: p_crit ≈ {p_transition:.4f} higher than predicted (but still sparse)")
    else:
        print(f"  ✗ FALSIFIED: p_crit ≈ {p_transition:.4f} >> 0.01 (dense coupling required)")
    print()

    # Check for sigmoid vs linear
    # Fit linear model
    from scipy.stats import linregress
    slope, intercept, r_value_linear, _, _ = linregress(p_values, C_values)
    r2_linear = r_value_linear ** 2

    # Fit Hill function (cooperative binding)
    from scipy.optimize import curve_fit
    def hill(p, k, p_half):
        return p**k / (p**k + p_half**k)

    try:
        (k_fit, p_half_fit), _ = curve_fit(hill, p_values, C_values, p0=[1.0, 0.01], bounds=([0.1, 0.0001], [10.0, 0.5]))
        C_hill = hill(p_values, k_fit, p_half_fit)
        ss_res_hill = np.sum((C_values - C_hill) ** 2)
        ss_tot = np.sum((C_values - np.mean(C_values)) ** 2)
        r2_hill = 1 - (ss_res_hill / ss_tot)

        print(f"Model Comparison:")
        print(f"  Linear: R² = {r2_linear:.4f}")
        print(f"  Hill:   R² = {r2_hill:.4f}, k = {k_fit:.3f}, p_half = {p_half_fit:.4f}")
        print()

        if r2_hill > r2_linear + 0.05:
            print(f"  ✓ Sigmoid (Hill) fits better than linear → Phase transition confirmed")
        else:
            print(f"  ✗ Linear fits as well as sigmoid → No clear phase transition")
    except Exception as e:
        print(f"  (Hill fit failed: {e})")
        print(f"  Linear: R² = {r2_linear:.4f}")

    print()

    # Save results
    output_file = output_dir / "prediction_4a_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'prediction': '4a_atp_coupling_phase_transition',
            'hypothesis': 'sigmoid transition with p_crit ≈ 0.002-0.01',
            'results': results,
            'transition_point': float(p_transition),
            'coherence_range': {
                'min': float(C_values[0]),
                'max': float(C_values[-1]),
                'delta': float(C_values[-1] - C_values[0])
            },
            'model_fits': {
                'linear_r2': float(r2_linear) if 'r2_linear' in locals() else None,
                'hill_r2': float(r2_hill) if 'r2_hill' in locals() else None,
                'hill_k': float(k_fit) if 'k_fit' in locals() else None,
                'hill_p_half': float(p_half_fit) if 'p_half_fit' in locals() else None,
            }
        }, f, indent=2)

    print(f"Results saved to: {output_file}")
    print("=" * 80)

    return results


if __name__ == '__main__':
    results = run_full_experiment()
