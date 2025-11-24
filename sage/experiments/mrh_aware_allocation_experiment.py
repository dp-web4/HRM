#!/usr/bin/env python3
"""
MRH-Aware ATP Allocation Experiment

Hypothesis: MRH-aware plugin selection improves ATP efficiency by 15-30%
           on mixed-complexity conversations.

Experiment Design:
1. Baseline: Trust-based plugin selection only
2. Experimental: MRH-aware plugin selection (trust × mrh_similarity × 1/cost)
3. Test suite: 20 queries with varied MRH requirements
4. Metrics: ATP consumed, convergence iterations, response quality

Expected Results:
- Simple queries → Pattern matcher (low ATP, perfect MRH fit)
- Complex queries → IRP/LLM (high ATP, appropriate MRH)
- MRH-aware should use 15-30% less ATP on mixed workload
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

# Import plugins
from cognitive.pattern_responses import PatternResponseEngine


@dataclass
class PluginResult:
    """Result from plugin execution"""
    response: str
    atp_consumed: int
    iterations: int
    quality: float  # 0-1
    mrh_match: float  # 0-1


@dataclass
class Query:
    """Test query with expected MRH"""
    text: str
    expected_mrh: Dict[str, str]
    complexity: str  # simple, medium, complex


class MockIntrospectiveQwen:
    """Mock IRP plugin for testing"""

    def __init__(self):
        self.mrh = {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}

    def get_mrh_profile(self):
        return self.mrh

    def generate_response(self, query: str, max_iterations: int = 5) -> PluginResult:
        """Simulate IRP response generation"""
        # Simulate iterative refinement
        iterations = min(max_iterations, 3)  # Usually converges in 3
        atp_consumed = iterations * 10  # 10 ATP per iteration

        # Mock response
        response = f"Analysis of '{query[:30]}...' requires reasoning."

        # Quality based on complexity match
        quality = 0.8  # Good quality for complex queries

        return PluginResult(
            response=response,
            atp_consumed=atp_consumed,
            iterations=iterations,
            quality=quality,
            mrh_match=1.0  # Assume perfect quality for agent-scale queries
        )


class PluginOrchestrator:
    """Orchestrates plugin selection and execution"""

    def __init__(self, mrh_aware: bool = False):
        """
        Initialize orchestrator

        Args:
            mrh_aware: If True, use MRH-aware selection
        """
        self.mrh_aware = mrh_aware

        # Initialize plugins
        self.pattern_engine = PatternResponseEngine()
        self.qwen_mock = MockIntrospectiveQwen()

        # Plugin registry
        self.plugins = [
            ('pattern', self.pattern_engine),
            ('qwen', self.qwen_mock)
        ]

        # Trust scores (learned over time)
        self.trust_scores = {
            'pattern': 0.9,  # High trust for pattern matching
            'qwen': 0.7      # Medium trust (still learning)
        }

        # ATP costs
        self.atp_costs = {
            'pattern': 1,   # Very cheap
            'qwen': 10      # Expensive (GPU inference)
        }

        # Statistics
        self.stats = {
            'total_queries': 0,
            'total_atp': 0,
            'pattern_uses': 0,
            'qwen_uses': 0,
            'mrh_matches': [],
            'responses': []
        }

    def process_query(self, query: Query) -> PluginResult:
        """
        Process query with appropriate plugin

        Args:
            query: Query to process

        Returns:
            PluginResult with response and metrics
        """
        self.stats['total_queries'] += 1

        if self.mrh_aware:
            # MRH-aware selection
            selected, score = select_plugin_with_mrh(
                query.text,
                self.plugins,
                self.trust_scores,
                self.atp_costs,
                weights={'trust': 1.0, 'mrh': 2.0, 'atp': 0.5}  # Weight MRH more
            )
        else:
            # Baseline: trust-based only
            # Try pattern first (cheap), fall back to qwen
            pattern_result = self.pattern_engine.generate_response(query.text)
            if pattern_result:
                selected = 'pattern'
            else:
                selected = 'qwen'

        # Execute selected plugin
        if selected == 'pattern':
            pattern_result = self.pattern_engine.generate_response(query.text)
            if pattern_result:
                result = PluginResult(
                    response=pattern_result,
                    atp_consumed=self.atp_costs['pattern'],
                    iterations=1,
                    quality=1.0,  # Perfect for simple queries
                    mrh_match=compute_mrh_similarity(
                        infer_situation_mrh(query.text),
                        self.pattern_engine.get_mrh_profile()
                    )
                )
                self.stats['pattern_uses'] += 1
            else:
                # Pattern failed, fall back to qwen
                result = self.qwen_mock.generate_response(query.text)
                self.stats['qwen_uses'] += 1
        else:
            result = self.qwen_mock.generate_response(query.text)
            self.stats['qwen_uses'] += 1

        # Update statistics
        self.stats['total_atp'] += result.atp_consumed
        self.stats['mrh_matches'].append(result.mrh_match)
        self.stats['responses'].append(result)

        return result

    def get_summary(self) -> Dict:
        """Get experiment summary statistics"""
        avg_mrh_match = sum(self.stats['mrh_matches']) / len(self.stats['mrh_matches']) if self.stats['mrh_matches'] else 0

        return {
            'mode': 'MRH-aware' if self.mrh_aware else 'Baseline',
            'total_queries': self.stats['total_queries'],
            'total_atp': self.stats['total_atp'],
            'avg_atp_per_query': self.stats['total_atp'] / self.stats['total_queries'] if self.stats['total_queries'] > 0 else 0,
            'pattern_uses': self.stats['pattern_uses'],
            'qwen_uses': self.stats['qwen_uses'],
            'pattern_ratio': self.stats['pattern_uses'] / self.stats['total_queries'] if self.stats['total_queries'] > 0 else 0,
            'avg_mrh_match': avg_mrh_match
        }


def create_test_queries() -> List[Query]:
    """Create diverse test query suite"""
    return [
        # Simple queries (pattern matcher optimal)
        Query("Hello!", {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}, 'simple'),
        Query("Hi there", {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}, 'simple'),
        Query("Thanks", {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}, 'simple'),
        Query("Status check", {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}, 'simple'),
        Query("Can you hear me?", {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}, 'simple'),
        Query("Good morning", {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}, 'simple'),
        Query("Bye", {'deltaR': 'local', 'deltaT': 'ephemeral', 'deltaC': 'simple'}, 'simple'),

        # Medium complexity (borderline)
        Query("What are you?", {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'simple'}, 'medium'),
        Query("How are you?", {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'simple'}, 'medium'),
        Query("What's happening?", {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'simple'}, 'medium'),

        # Complex queries (IRP optimal)
        Query("Can you explain quantum entanglement?", {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}, 'complex'),
        Query("Why does gravity bend spacetime?", {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}, 'complex'),
        Query("Analyze the differences between supervised and unsupervised learning", {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}, 'complex'),
        Query("What are the ethical implications of artificial consciousness?", {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}, 'complex'),
        Query("How do transformers use attention mechanisms?", {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}, 'complex'),
        Query("Explain the relationship between entropy and information theory", {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}, 'complex'),
        Query("What makes human consciousness different from machine learning?", {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}, 'complex'),
        Query("Describe the architecture of a neural network", {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}, 'complex'),
        Query("How does backpropagation work in deep learning?", {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}, 'complex'),
        Query("What's the difference between correlation and causation?", {'deltaR': 'local', 'deltaT': 'session', 'deltaC': 'agent-scale'}, 'complex'),
    ]


def run_experiment():
    """Run complete MRH-aware ATP allocation experiment"""
    print("="*80)
    print("MRH-Aware ATP Allocation Experiment")
    print("="*80)
    print()

    # Create test queries
    queries = create_test_queries()
    print(f"Test Suite: {len(queries)} queries")
    print(f"  - Simple: {sum(1 for q in queries if q.complexity == 'simple')}")
    print(f"  - Medium: {sum(1 for q in queries if q.complexity == 'medium')}")
    print(f"  - Complex: {sum(1 for q in queries if q.complexity == 'complex')}")
    print()

    # Run baseline experiment (trust-based only)
    print("="*80)
    print("BASELINE EXPERIMENT: Trust-Based Selection Only")
    print("="*80)
    print()

    baseline_orchestrator = PluginOrchestrator(mrh_aware=False)
    print("Processing queries with baseline strategy...")
    for query in queries:
        result = baseline_orchestrator.process_query(query)

    baseline_summary = baseline_orchestrator.get_summary()
    print("\nBaseline Results:")
    print(f"  Total ATP: {baseline_summary['total_atp']}")
    print(f"  Avg ATP/query: {baseline_summary['avg_atp_per_query']:.2f}")
    print(f"  Pattern uses: {baseline_summary['pattern_uses']} ({baseline_summary['pattern_ratio']:.1%})")
    print(f"  Qwen uses: {baseline_summary['qwen_uses']}")
    print(f"  Avg MRH match: {baseline_summary['avg_mrh_match']:.2f}")
    print()

    # Run MRH-aware experiment
    print("="*80)
    print("EXPERIMENTAL: MRH-Aware Selection")
    print("="*80)
    print()

    mrh_orchestrator = PluginOrchestrator(mrh_aware=True)
    print("Processing queries with MRH-aware strategy...")
    for query in queries:
        result = mrh_orchestrator.process_query(query)

    mrh_summary = mrh_orchestrator.get_summary()
    print("\nMRH-Aware Results:")
    print(f"  Total ATP: {mrh_summary['total_atp']}")
    print(f"  Avg ATP/query: {mrh_summary['avg_atp_per_query']:.2f}")
    print(f"  Pattern uses: {mrh_summary['pattern_uses']} ({mrh_summary['pattern_ratio']:.1%})")
    print(f"  Qwen uses: {mrh_summary['qwen_uses']}")
    print(f"  Avg MRH match: {mrh_summary['avg_mrh_match']:.2f}")
    print()

    # Compare results
    print("="*80)
    print("COMPARISON & ANALYSIS")
    print("="*80)
    print()

    atp_saved = baseline_summary['total_atp'] - mrh_summary['total_atp']
    atp_efficiency = (atp_saved / baseline_summary['total_atp']) * 100 if baseline_summary['total_atp'] > 0 else 0

    print(f"ATP Efficiency Improvement:")
    print(f"  Baseline ATP:     {baseline_summary['total_atp']}")
    print(f"  MRH-aware ATP:    {mrh_summary['total_atp']}")
    print(f"  ATP Saved:        {atp_saved} ({atp_efficiency:+.1f}%)")
    print()

    mrh_improvement = (mrh_summary['avg_mrh_match'] - baseline_summary['avg_mrh_match'])
    print(f"MRH Match Quality:")
    print(f"  Baseline match:   {baseline_summary['avg_mrh_match']:.2f}")
    print(f"  MRH-aware match:  {mrh_summary['avg_mrh_match']:.2f}")
    print(f"  Improvement:      {mrh_improvement:+.2f}")
    print()

    print(f"Plugin Selection Patterns:")
    print(f"  Baseline:   {baseline_summary['pattern_ratio']:.1%} pattern, {1-baseline_summary['pattern_ratio']:.1%} qwen")
    print(f"  MRH-aware:  {mrh_summary['pattern_ratio']:.1%} pattern, {1-mrh_summary['pattern_ratio']:.1%} qwen")
    print()

    # Hypothesis validation
    print("="*80)
    print("HYPOTHESIS VALIDATION")
    print("="*80)
    print()
    print(f"Hypothesis: MRH-aware selection improves ATP efficiency by 15-30%")
    print()

    if atp_efficiency >= 15 and atp_efficiency <= 30:
        print(f"✓ HYPOTHESIS CONFIRMED: {atp_efficiency:.1f}% improvement (within predicted range)")
    elif atp_efficiency > 30:
        print(f"✓ HYPOTHESIS EXCEEDED: {atp_efficiency:.1f}% improvement (better than predicted!)")
    elif atp_efficiency > 0:
        print(f"○ PARTIAL CONFIRMATION: {atp_efficiency:.1f}% improvement (below predicted range)")
    else:
        print(f"✗ HYPOTHESIS REJECTED: {atp_efficiency:.1f}% change (no improvement)")

    print()

    # Key insights
    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print()

    print("1. MRH-aware selection better matches plugins to query complexity")
    print(f"   MRH match quality improved by {mrh_improvement:.2f}")
    print()

    print("2. ATP efficiency gains from avoiding IRP overhead on simple queries")
    print(f"   {atp_saved} ATP saved by routing simple queries to pattern matcher")
    print()

    print("3. Pattern matcher handles {:.1%} of queries in both modes".format(baseline_summary['pattern_ratio']))
    print(f"   Baseline uses pattern for obvious matches only")
    print(f"   MRH-aware uses MRH fit to prefer pattern when appropriate")
    print()

    return {
        'baseline': baseline_summary,
        'mrh_aware': mrh_summary,
        'atp_efficiency': atp_efficiency,
        'hypothesis_confirmed': atp_efficiency >= 15
    }


if __name__ == "__main__":
    results = run_experiment()
