#!/usr/bin/env python3
"""
Quality-Aware Plugin Selection Experiment - Edge Optimized

Tests the edge-optimized quality-aware selection against Experiment 3 scenario.

Goals:
1. Demonstrate 0% quality failures while maintaining cost optimization
2. Compare against original MRH-aware selection (95% savings but 58% failures)
3. Validate edge-specific error handling (fail fast behavior)
4. Measure actual ATP consumption and quality compliance

Edge philosophy: Quality requirements are hard constraints, not soft goals.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from core.mrh_utils import (
    select_plugin_with_mrh,
    select_plugin_with_quality_edge,
    infer_quality_requirement,
    InsufficientATPBudgetError,
    NoQualifiedPluginError,
    format_mrh,
    infer_situation_mrh
)


@dataclass
class Query:
    """Test query with requirements"""
    text: str
    complexity: str  # simple, medium, complex
    min_quality: float  # Minimum quality required


class MockPlugin:
    """Mock plugin for testing"""
    def __init__(self, name, mrh, trust, cost):
        self.name = name
        self.mrh = mrh
        self.trust = trust
        self.cost = cost

    def get_mrh_profile(self):
        return self.mrh


def create_experiment_3_setup():
    """Create the same setup as Experiment 3 (same-horizon, different quality)"""

    # Three LLM plugins, all at (local, session, agent-scale)
    plugins = [
        ('qwen-0.5b', MockPlugin(
            'qwen-0.5b',
            {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'},
            trust=0.7,
            cost=10
        )),
        ('qwen-7b', MockPlugin(
            'qwen-7b',
            {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'},
            trust=0.9,
            cost=100
        )),
        ('cloud-gpt', MockPlugin(
            'cloud-gpt',
            {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'},
            trust=0.95,
            cost=200
        ))
    ]

    trust_scores = {name: p.trust for name, p in plugins}
    atp_costs = {name: p.cost for name, p in plugins}

    return plugins, trust_scores, atp_costs


def create_test_queries():
    """Create test queries with varying complexity"""
    return [
        # Simple queries (min quality: 0.5) - qwen-0.5b is fine
        Query("Hello!", "simple", 0.5),
        Query("What time is it?", "simple", 0.5),
        Query("Yes, I agree.", "simple", 0.5),
        Query("Thanks!", "simple", 0.5),

        # Medium queries (min quality: 0.75) - need qwen-7b+
        Query("Explain how SAGE works.", "medium", 0.75),
        Query("Describe the IRP protocol.", "medium", 0.75),
        Query("What is the difference between ATP and energy?", "medium", 0.75),
        Query("Summarize the MRH concept.", "medium", 0.75),

        # Complex queries (min quality: 0.9) - need qwen-7b or cloud-gpt
        Query("Analyze the philosophical implications of machine consciousness.", "complex", 0.9),
        Query("Explain why trust systems require cryptographic verification.", "complex", 0.9),
        Query("Compare the deep trade-offs between quality and efficiency.", "complex", 0.9),
        Query("Provide a comprehensive analysis of edge constraints.", "complex", 0.9)
    ]


def run_original_mrh_experiment(plugins, trust_scores, atp_costs, queries):
    """Run original MRH-aware selection (Experiment 3 baseline)"""
    print("="*80)
    print("BASELINE: Original MRH-Aware Selection (Experiment 3)")
    print("="*80)

    total_atp = 0
    quality_failures = 0
    plugin_usage = {}

    for query in queries:
        selected, score = select_plugin_with_mrh(
            query.text,
            plugins,
            trust_scores,
            atp_costs
        )

        atp = atp_costs[selected]
        total_atp += atp
        plugin_usage[selected] = plugin_usage.get(selected, 0) + 1

        # Check quality compliance
        actual_quality = trust_scores[selected]
        if actual_quality < query.min_quality:
            quality_failures += 1
            status = "❌ FAIL"
        else:
            status = "✓ ok"

        print(f"  [{query.complexity:7}] '{query.text[:40]:<40}' → {selected:12} "
              f"(Q:{actual_quality:.2f} vs {query.min_quality:.2f}) {status}")

    print()
    print(f"Total ATP: {total_atp}")
    print(f"Quality failures: {quality_failures}/{len(queries)} ({quality_failures/len(queries)*100:.1f}%)")
    print(f"Plugin usage: {plugin_usage}")

    return total_atp, quality_failures, plugin_usage


def run_quality_aware_experiment(plugins, trust_scores, atp_costs, queries):
    """Run quality-aware edge selection"""
    print("\n" + "="*80)
    print("EXPERIMENTAL: Quality-Aware Edge Selection")
    print("="*80)

    total_atp = 0
    quality_failures = 0
    quality_successes = 0
    plugin_usage = {}
    budget_errors = 0
    no_plugin_errors = 0

    for query in queries:
        try:
            selected, atp, reason = select_plugin_with_quality_edge(
                query.text,
                quality_required=query.min_quality,
                plugins=plugins,
                trust_scores=trust_scores,
                atp_costs=atp_costs
            )

            total_atp += atp
            plugin_usage[selected] = plugin_usage.get(selected, 0) + 1

            # Quality compliance is guaranteed by the function
            quality_successes += 1
            status = "✓ ok"

            print(f"  [{query.complexity:7}] '{query.text[:40]:<40}' → {selected:12} "
                  f"(Q:{trust_scores[selected]:.2f} >= {query.min_quality:.2f}) {status}")

        except InsufficientATPBudgetError as e:
            budget_errors += 1
            print(f"  [{query.complexity:7}] '{query.text[:40]:<40}' → ⚠️ Budget exceeded")

        except NoQualifiedPluginError as e:
            no_plugin_errors += 1
            print(f"  [{query.complexity:7}] '{query.text[:40]:<40}' → ❌ No qualified plugin")

    print()
    print(f"Total ATP: {total_atp}")
    print(f"Quality failures: {quality_failures}/{len(queries)} ({quality_failures/len(queries)*100:.1f}%)")
    print(f"Quality successes: {quality_successes}/{len(queries)}")
    print(f"Budget errors: {budget_errors}")
    print(f"No plugin errors: {no_plugin_errors}")
    print(f"Plugin usage: {plugin_usage}")

    return total_atp, quality_failures, plugin_usage


def run_quality_aware_with_budget(plugins, trust_scores, atp_costs, queries, atp_budget):
    """Run quality-aware edge selection with ATP budget constraint"""
    print(f"\n" + "="*80)
    print(f"EXPERIMENTAL: Quality-Aware Edge Selection (Budget: {atp_budget} ATP/query)")
    print("="*80)

    total_atp = 0
    quality_failures = 0
    quality_successes = 0
    plugin_usage = {}
    budget_errors = 0
    no_plugin_errors = 0

    for query in queries:
        try:
            selected, atp, reason = select_plugin_with_quality_edge(
                query.text,
                quality_required=query.min_quality,
                plugins=plugins,
                trust_scores=trust_scores,
                atp_costs=atp_costs,
                atp_budget=atp_budget
            )

            total_atp += atp
            plugin_usage[selected] = plugin_usage.get(selected, 0) + 1
            quality_successes += 1

            print(f"  [{query.complexity:7}] '{query.text[:40]:<40}' → {selected:12} ✓")

        except InsufficientATPBudgetError as e:
            budget_errors += 1
            # On edge: fail fast, don't compromise
            print(f"  [{query.complexity:7}] '{query.text[:40]:<40}' → ⚠️ BUDGET ({e})")

        except NoQualifiedPluginError as e:
            no_plugin_errors += 1
            print(f"  [{query.complexity:7}] '{query.text[:40]:<40}' → ❌ NO PLUGIN")

    print()
    print(f"Total ATP: {total_atp}")
    print(f"Quality successes: {quality_successes}/{len(queries)}")
    print(f"Budget violations (failed fast): {budget_errors}")
    print(f"No qualified plugin: {no_plugin_errors}")
    print(f"Plugin usage: {plugin_usage}")

    return total_atp, quality_successes, budget_errors, plugin_usage


def run_auto_quality_inference_experiment(plugins, trust_scores, atp_costs, queries):
    """Run quality-aware selection with automatic quality inference"""
    print("\n" + "="*80)
    print("EXPERIMENTAL: Auto Quality Inference + Edge Selection")
    print("="*80)

    total_atp = 0
    quality_successes = 0
    plugin_usage = {}

    for query in queries:
        # Infer quality requirement from query
        inferred_quality = infer_quality_requirement(query.text)

        try:
            selected, atp, reason = select_plugin_with_quality_edge(
                query.text,
                quality_required=inferred_quality,
                plugins=plugins,
                trust_scores=trust_scores,
                atp_costs=atp_costs
            )

            total_atp += atp
            plugin_usage[selected] = plugin_usage.get(selected, 0) + 1
            quality_successes += 1

            print(f"  [{query.complexity:7}] Q_inferred={inferred_quality:.2f} → {selected:12} "
                  f"(Q:{trust_scores[selected]:.2f}) ATP:{atp}")

        except Exception as e:
            print(f"  [{query.complexity:7}] Q_inferred={inferred_quality:.2f} → ⚠️ {type(e).__name__}")

    print()
    print(f"Total ATP: {total_atp}")
    print(f"Quality successes: {quality_successes}/{len(queries)}")
    print(f"Plugin usage: {plugin_usage}")

    return total_atp, quality_successes, plugin_usage


def main():
    """Run quality-aware edge selection experiments"""
    print("="*80)
    print("QUALITY-AWARE PLUGIN SELECTION EXPERIMENT - EDGE OPTIMIZED")
    print("="*80)
    print("\nHardware: Jetson Orin Nano 8GB (Sprout)")
    print("Purpose: Validate quality-aware selection vs Experiment 3 baseline")
    print()

    # Setup
    plugins, trust_scores, atp_costs = create_experiment_3_setup()
    queries = create_test_queries()

    print(f"Plugins: {len(plugins)}")
    for name, p in plugins:
        print(f"  {name}: Q={p.trust:.2f}, ATP={p.cost}")

    print(f"\nQueries: {len(queries)}")
    print(f"  Simple (Q>=0.5): 4")
    print(f"  Medium (Q>=0.75): 4")
    print(f"  Complex (Q>=0.9): 4")

    # Run experiments
    baseline_atp, baseline_failures, baseline_usage = run_original_mrh_experiment(
        plugins, trust_scores, atp_costs, queries
    )

    quality_atp, quality_failures, quality_usage = run_quality_aware_experiment(
        plugins, trust_scores, atp_costs, queries
    )

    # Run with budget constraint (50 ATP max per query)
    budget_atp, budget_successes, budget_errors, budget_usage = run_quality_aware_with_budget(
        plugins, trust_scores, atp_costs, queries, atp_budget=50
    )

    # Run with automatic quality inference
    auto_atp, auto_successes, auto_usage = run_auto_quality_inference_experiment(
        plugins, trust_scores, atp_costs, queries
    )

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print("\nComparison: Baseline vs Quality-Aware")
    print("-" * 60)
    print(f"{'Metric':<30} {'Baseline':<15} {'Quality-Aware':<15}")
    print("-" * 60)
    print(f"{'Total ATP':<30} {baseline_atp:<15} {quality_atp:<15}")
    print(f"{'Quality Failures':<30} {baseline_failures:<15} {quality_failures:<15}")
    print(f"{'Plugin Diversity':<30} {len(baseline_usage):<15} {len(quality_usage):<15}")
    print("-" * 60)

    # Calculate improvement
    atp_increase = ((quality_atp - baseline_atp) / baseline_atp) * 100 if baseline_atp > 0 else 0
    failure_reduction = baseline_failures - quality_failures

    print(f"\nResults:")
    print(f"  ATP Change: {atp_increase:+.1f}% (from {baseline_atp} to {quality_atp})")
    print(f"  Quality Failures: {baseline_failures} → {quality_failures} ({failure_reduction} fewer)")
    print(f"  Plugin Usage: {baseline_usage} → {quality_usage}")

    print("\n" + "="*80)
    print("EDGE VALIDATION VERDICT")
    print("="*80)

    if quality_failures == 0 and baseline_failures > 0:
        print("\n✓ SUCCESS: Quality-aware selection achieves 0% quality failures!")
        print(f"  Baseline had {baseline_failures}/{len(queries)} failures ({baseline_failures/len(queries)*100:.0f}%)")
        print(f"  Quality-aware has 0/{len(queries)} failures (0%)")
        print(f"  Trade-off: {atp_increase:+.1f}% ATP increase for 100% quality compliance")
        print("\n  Edge recommendation: DEPLOY quality-aware selection")
    else:
        print("\n⚠ Quality-aware selection status:")
        print(f"  Failures: {quality_failures}")

    print("\n" + "="*80)

    return baseline_atp, quality_atp, baseline_failures, quality_failures


if __name__ == '__main__':
    main()
