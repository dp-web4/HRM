#!/usr/bin/env python3
"""
MRH Cost/Quality Trade-off Experiment (Experiment 3)

Hypothesis: When multiple plugins operate at the same MRH level but differ in
           cost and quality, MRH-aware selection should help balance trade-offs
           by considering quality requirements alongside cost.

Scenario: Three LLMs all at (local, session, agent-scale)
- qwen-0.5b: cheap (10 ATP), decent quality (0.7 trust)
- qwen-7b: expensive (100 ATP), high quality (0.9 trust)
- cloud-gpt: very expensive (200 ATP), highest quality (0.95 trust)

Test queries with varying quality requirements:
- Simple queries: "Hello!" → cheap model sufficient
- Medium queries: "Explain X" → balance cost/quality
- Complex queries: "Analyze deep concept" → quality critical

Expected Results:
- Baseline: Always picks highest trust (cloud-gpt) regardless of need
- MRH-aware: Same MRH similarity for all, so cost becomes tiebreaker
- Insight: MRH alone doesn't help here - need quality awareness!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Import MRH utilities
from core.mrh_utils import (
    compute_mrh_similarity,
    infer_situation_mrh,
    format_mrh,
    select_plugin_with_mrh
)


@dataclass
class QueryTask:
    """A query task with text and quality requirement"""
    text: str
    complexity: str  # 'simple', 'medium', 'complex'
    min_quality_required: float  # Minimum acceptable quality
    expected_plugin: str  # For validation


@dataclass
class PluginResult:
    """Result from plugin execution"""
    response: str
    quality: float
    plugin_used: str
    atp_consumed: int
    execution_time_ms: float
    mrh_match: float


class MockLLMPlugin:
    """
    Mock LLM plugin with configurable cost/quality

    All plugins operate at same MRH: (local, session, agent-scale)
    Differ only in ATP cost and output quality (trust)
    """

    def __init__(self, name: str, atp_cost: int, quality: float):
        self.name = name
        self.atp_cost = atp_cost
        self.quality = quality  # This is the trust score
        self.mrh = {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}

    def get_mrh_profile(self):
        return self.mrh

    def execute(self, query: str) -> Tuple[str, float]:
        """Execute query and return (response, actual_quality)"""
        # Simulate execution time proportional to cost
        time.sleep(0.001 * (self.atp_cost / 10))

        # Response quality varies by model capacity
        if "hello" in query.lower() or "hi" in query.lower():
            # Simple greeting - all models do well
            actual_quality = min(1.0, self.quality + 0.1)
        elif "explain" in query.lower() or "what" in query.lower():
            # Medium complexity - quality matters
            actual_quality = self.quality
        elif "analyze" in query.lower() or "complex" in query.lower():
            # Complex reasoning - higher quality models shine
            actual_quality = self.quality if self.quality > 0.8 else self.quality * 0.7
        else:
            actual_quality = self.quality

        response = f"[{self.name}] Response to: {query[:50]}..."
        return response, actual_quality


class PluginOrchestrator:
    """
    Orchestrates plugin selection and execution

    Two strategies:
    1. Baseline: Always pick highest trust (regardless of cost or need)
    2. MRH-aware: Use MRH similarity + cost weighting
    """

    def __init__(self, plugins: List[MockLLMPlugin], mrh_aware: bool = False):
        self.plugins = [(p.name, p) for p in plugins]
        self.mrh_aware = mrh_aware

        # Trust scores (same as quality for these mocks)
        self.trust_scores = {p.name: p.quality for p in plugins}

        # ATP costs
        self.atp_costs = {p.name: p.atp_cost for p in plugins}

    def process_query(self, query: QueryTask) -> PluginResult:
        """Process query using selected strategy"""
        start = time.time()

        if self.mrh_aware:
            # MRH-aware selection
            selected, score = select_plugin_with_mrh(
                query.text,
                self.plugins,
                self.trust_scores,
                self.atp_costs,
                weights={'trust': 1.0, 'mrh': 1.0, 'atp': 0.5},
                mrh_threshold=0.6
            )
        else:
            # Baseline: Pick highest trust regardless of cost
            selected = max(self.plugins, key=lambda p: self.trust_scores[p[0]])[0]

        # Get selected plugin
        plugin_obj = next(p for name, p in self.plugins if name == selected)

        # Execute
        response, actual_quality = plugin_obj.execute(query.text)

        # Compute MRH match
        situation_mrh = infer_situation_mrh(query.text)
        plugin_mrh = plugin_obj.get_mrh_profile()
        mrh_match = compute_mrh_similarity(situation_mrh, plugin_mrh)

        execution_time = (time.time() - start) * 1000

        return PluginResult(
            response=response,
            quality=actual_quality,
            plugin_used=selected,
            atp_consumed=plugin_obj.atp_cost,
            execution_time_ms=execution_time,
            mrh_match=mrh_match
        )


def create_test_queries() -> List[QueryTask]:
    """Create test suite with varying complexity"""
    return [
        # Simple queries (cheap model sufficient)
        QueryTask("Hello!", "simple", 0.5, "qwen-0.5b"),
        QueryTask("Hi there", "simple", 0.5, "qwen-0.5b"),
        QueryTask("Thanks", "simple", 0.5, "qwen-0.5b"),
        QueryTask("Good morning", "simple", 0.5, "qwen-0.5b"),

        # Medium queries (balance cost/quality)
        QueryTask("What is machine learning?", "medium", 0.75, "qwen-7b"),
        QueryTask("Explain neural networks", "medium", 0.75, "qwen-7b"),
        QueryTask("How does backpropagation work?", "medium", 0.75, "qwen-7b"),
        QueryTask("What are transformers?", "medium", 0.75, "qwen-7b"),

        # Complex queries (quality critical)
        QueryTask("Analyze the philosophical implications of consciousness", "complex", 0.9, "cloud-gpt"),
        QueryTask("Compare and contrast emergence in complex systems", "complex", 0.9, "cloud-gpt"),
        QueryTask("Explain the relationship between information theory and thermodynamics", "complex", 0.9, "cloud-gpt"),
        QueryTask("Analyze the epistemological foundations of mathematics", "complex", 0.9, "cloud-gpt"),
    ]


def run_experiment():
    """Run the cost/quality trade-off experiment"""

    print("=" * 80)
    print("MRH Cost/Quality Trade-off Experiment (Experiment 3)")
    print("=" * 80)
    print()

    # Create plugins - all same MRH, different cost/quality
    plugins = [
        MockLLMPlugin("qwen-0.5b", atp_cost=10, quality=0.7),
        MockLLMPlugin("qwen-7b", atp_cost=100, quality=0.9),
        MockLLMPlugin("cloud-gpt", atp_cost=200, quality=0.95),
    ]

    print("Plugins available (all at same MRH: local, session, agent-scale):")
    for p in plugins:
        print(f"  {p.name}: ATP={p.atp_cost}, Quality={p.quality}")
    print()

    # Create test queries
    queries = create_test_queries()

    print(f"Test suite: {len(queries)} queries")
    print(f"  Simple: {sum(1 for q in queries if q.complexity == 'simple')}")
    print(f"  Medium: {sum(1 for q in queries if q.complexity == 'medium')}")
    print(f"  Complex: {sum(1 for q in queries if q.complexity == 'complex')}")
    print()

    # Run baseline
    print("-" * 80)
    print("Baseline: Always pick highest trust (cloud-gpt)")
    print("-" * 80)

    baseline_orchestrator = PluginOrchestrator(plugins, mrh_aware=False)
    baseline_results = []

    for query in queries:
        result = baseline_orchestrator.process_query(query)
        baseline_results.append(result)

    # Baseline metrics
    baseline_atp = sum(r.atp_consumed for r in baseline_results)
    baseline_avg_quality = sum(r.quality for r in baseline_results) / len(baseline_results)
    baseline_plugin_usage = {}
    for r in baseline_results:
        baseline_plugin_usage[r.plugin_used] = baseline_plugin_usage.get(r.plugin_used, 0) + 1

    print(f"Total ATP consumed: {baseline_atp}")
    print(f"Avg quality achieved: {baseline_avg_quality:.3f}")
    print(f"Plugin usage: {baseline_plugin_usage}")
    print()

    # Check quality requirements met
    quality_failures_baseline = sum(1 for i, r in enumerate(baseline_results)
                                   if r.quality < queries[i].min_quality_required)
    print(f"Quality requirement failures: {quality_failures_baseline}/{len(queries)}")
    print()

    # Run MRH-aware
    print("-" * 80)
    print("MRH-Aware: MRH similarity + cost weighting")
    print("-" * 80)

    mrh_orchestrator = PluginOrchestrator(plugins, mrh_aware=True)
    mrh_results = []

    for query in queries:
        result = mrh_orchestrator.process_query(query)
        mrh_results.append(result)

    # MRH-aware metrics
    mrh_atp = sum(r.atp_consumed for r in mrh_results)
    mrh_avg_quality = sum(r.quality for r in mrh_results) / len(mrh_results)
    mrh_plugin_usage = {}
    for r in mrh_results:
        mrh_plugin_usage[r.plugin_used] = mrh_plugin_usage.get(r.plugin_used, 0) + 1

    print(f"Total ATP consumed: {mrh_atp}")
    print(f"Avg quality achieved: {mrh_avg_quality:.3f}")
    print(f"Plugin usage: {mrh_plugin_usage}")
    print()

    # Check quality requirements met
    quality_failures_mrh = sum(1 for i, r in enumerate(mrh_results)
                              if r.quality < queries[i].min_quality_required)
    print(f"Quality requirement failures: {quality_failures_mrh}/{len(queries)}")
    print()

    # Comparison
    print("=" * 80)
    print("Results Comparison")
    print("=" * 80)
    print()

    atp_improvement = ((baseline_atp - mrh_atp) / baseline_atp) * 100 if baseline_atp > 0 else 0
    quality_change = mrh_avg_quality - baseline_avg_quality

    print(f"ATP Savings: {atp_improvement:.1f}% ({baseline_atp} → {mrh_atp})")
    print(f"Quality Change: {quality_change:+.3f} ({baseline_avg_quality:.3f} → {mrh_avg_quality:.3f})")
    print()

    print("Plugin Usage:")
    print(f"  Baseline: {baseline_plugin_usage}")
    print(f"  MRH-aware: {mrh_plugin_usage}")
    print()

    print("Quality Requirement Compliance:")
    print(f"  Baseline failures: {quality_failures_baseline}/{len(queries)}")
    print(f"  MRH-aware failures: {quality_failures_mrh}/{len(queries)}")
    print()

    # Detailed breakdown by complexity
    print("Breakdown by Query Complexity:")
    for complexity in ['simple', 'medium', 'complex']:
        complexity_queries = [i for i, q in enumerate(queries) if q.complexity == complexity]

        baseline_atp_comp = sum(baseline_results[i].atp_consumed for i in complexity_queries)
        mrh_atp_comp = sum(mrh_results[i].atp_consumed for i in complexity_queries)

        baseline_quality_comp = sum(baseline_results[i].quality for i in complexity_queries) / len(complexity_queries)
        mrh_quality_comp = sum(mrh_results[i].quality for i in complexity_queries) / len(complexity_queries)

        print(f"\n  {complexity.upper()}:")
        print(f"    ATP: {baseline_atp_comp} → {mrh_atp_comp} ({((baseline_atp_comp - mrh_atp_comp) / baseline_atp_comp * 100):.1f}% savings)")
        print(f"    Quality: {baseline_quality_comp:.3f} → {mrh_quality_comp:.3f} ({(mrh_quality_comp - baseline_quality_comp):+.3f})")

    print()
    print("=" * 80)
    print("Hypothesis Assessment")
    print("=" * 80)
    print()

    if atp_improvement > 0 and quality_failures_mrh <= quality_failures_baseline:
        print("✓ PARTIAL SUCCESS: MRH-aware saves ATP while maintaining quality")
        print()
        print("However, this reveals a key insight:")
        print("When all plugins have SAME MRH, MRH similarity doesn't discriminate.")
        print("Cost weighting becomes the tiebreaker, picking cheapest option.")
        print()
        print("This is GOOD for simple queries but BAD for complex ones!")
        print("We need QUALITY-AWARE selection, not just MRH-aware!")
    elif atp_improvement > 0 and quality_failures_mrh > quality_failures_baseline:
        print("✗ FAILURE: MRH-aware saves ATP but sacrifices quality")
        print()
        print("The cost weighting picks cheap models even when quality is critical.")
        print("Need to incorporate quality requirements into selection!")
    else:
        print("○ NO BENEFIT: MRH-aware provides no advantage in same-horizon scenario")

    print()
    print("Key Insight:")
    print("MRH awareness alone is insufficient when plugins share the same horizon.")
    print("Need additional dimensions: quality requirements, task complexity, etc.")
    print()


if __name__ == "__main__":
    run_experiment()
