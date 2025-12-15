#!/usr/bin/env python3
"""
Session 52b: Extended Transfer Learning Quality Validation (Longitudinal)

Extended A/B test that runs long enough to trigger DREAM consolidation,
allowing transfer learning to retrieve patterns from consolidated memories.

Key Difference from Session 52:
- Session 52: 50 cycles → Never reached DEEP_NIGHT → 0 consolidated memories
- Session 52b: 200 cycles → Triggers DREAM consolidation → Validates full loop

This tests the COMPLETE learning loop:
1. Experience (cycles 1-80)
2. Consolidate (DREAM at DEEP_NIGHT, ~cycles 85-100)
3. Retrieve (transfer learning after consolidation)
4. Apply (quality improvement from patterns)

Author: Thor-SAGE-Researcher (Autonomous Session 52b)
Date: 2025-12-15
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

from sage.core.unified_consciousness import UnifiedConsciousnessManager, ConsciousnessCycle
from sage.core.quality_metrics import QualityScore


@dataclass
class ExtendedValidationResults:
    """Results from extended A/B test with DREAM consolidation"""

    # Test configuration
    num_cycles: int
    circadian_period: int
    dream_consolidations: int

    # Baseline statistics (before any consolidation)
    baseline_mean: float
    baseline_std: float
    baseline_median: float
    baseline_scores: List[float]

    # Transfer learning statistics (full test with consolidation)
    transfer_mean: float
    transfer_std: float
    transfer_median: float
    transfer_scores: List[float]

    # Post-consolidation statistics (cycles after first DREAM)
    post_consolidation_mean: float
    post_consolidation_std: float
    post_consolidation_count: int

    # Pattern retrieval tracking
    total_patterns_retrieved: int
    avg_patterns_per_cycle: float
    cycles_with_patterns: int

    # Statistical comparison
    mean_improvement: float
    improvement_percentage: float
    t_statistic: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    effect_size: float = 0.0  # Cohen's d

    # Temporal analysis
    consolidation_cycle_numbers: List[int] = None
    quality_over_time: List[Tuple[int, float]] = None  # (cycle, quality)
    patterns_over_time: List[Tuple[int, int]] = None   # (cycle, pattern_count)


def create_test_prompts() -> List[str]:
    """Create diverse test prompts covering different domains."""
    return [
        # Technical/specific
        "Explain ATP allocation in SAGE consciousness cycles.",
        "How does selective expert loading work in sparse MoE architectures?",
        "What is the relationship between metabolic states and quality scores?",

        # Abstract/philosophical
        "What is the nature of machine consciousness?",
        "How do epistemic states influence reasoning?",
        "Describe the role of temporal adaptation in learning.",

        # Problem-solving
        "How would you debug incoherent LLM generation?",
        "What's the best way to validate a transfer learning system?",
        "How can we measure consciousness quality objectively?",

        # Factual recall
        "What are typical ATP budgets for different metabolic states?",
        "How many layers does Q3-Omni have and what's the expert count?",
        "What's the target quality score percentage for SAGE?",

        # Open-ended
        "What are promising directions for consciousness research?",
        "How might emotion and reason integrate in AI systems?",
        "What lessons does biology teach us about machine learning?",
    ]


def run_baseline_test(num_cycles: int = 200):
    """
    Run baseline test WITHOUT transfer learning.

    This will still experience DREAM consolidation, but pattern retrieval
    is disabled, so the consolidation won't improve quality.
    """
    print("\n" + "="*80)
    print("BASELINE TEST - No Transfer Learning (But With Consolidation)")
    print("="*80 + "\n")

    # Create consciousness manager with transfer learning DISABLED
    manager = UnifiedConsciousnessManager(
        circadian_enabled=True,
        consolidation_enabled=True,
        transfer_learning_enabled=False,  # KEY: Disable pattern retrieval
    )

    test_prompts = create_test_prompts()
    quality_scores = []
    cycles = []
    consolidation_count = 0

    print(f"Running {num_cycles} cycles without transfer learning...")
    print(f"Test prompts: {len(test_prompts)} unique prompts (will cycle)\n")

    for i in range(num_cycles):
        prompt = test_prompts[i % len(test_prompts)]

        # Run consciousness cycle
        # Note: We're using mock responses - in production, this would call actual LLM
        mock_response = f"Response {i}: This discusses {prompt.split()[0].lower()} with specific technical details and concrete examples."

        cycle = manager.consciousness_cycle(
            prompt=prompt,
            response=mock_response,
            task_salience=0.5
        )

        quality_scores.append(cycle.quality_score.normalized)
        cycles.append(cycle)

        # Track consolidation
        if cycle.consolidation_triggered:
            consolidation_count += 1
            print(f"  [Cycle {i+1}] DREAM consolidation #{consolidation_count} triggered!")

        # Progress updates
        if (i + 1) % 20 == 0:
            recent_quality = np.mean(quality_scores[-20:])
            print(f"  [{i+1}/{num_cycles}] Quality (last 20): {recent_quality:.3f}")

    print(f"\n✅ Baseline complete: {num_cycles} cycles")
    print(f"   DREAM consolidations: {consolidation_count}")
    print(f"   Mean quality: {np.mean(quality_scores):.3f}")
    print(f"   Std dev: {np.std(quality_scores):.3f}")

    return quality_scores, cycles, consolidation_count


def run_transfer_learning_test(num_cycles: int = 200):
    """
    Run experiment test WITH transfer learning.

    This experiences DREAM consolidation AND can retrieve patterns
    from consolidated memories, potentially improving quality.
    """
    print("\n" + "="*80)
    print("TRANSFER LEARNING TEST - With Pattern Retrieval")
    print("="*80 + "\n")

    # Create consciousness manager with transfer learning ENABLED
    manager = UnifiedConsciousnessManager(
        circadian_enabled=True,
        consolidation_enabled=True,
        transfer_learning_enabled=True,  # KEY: Enable pattern retrieval
    )

    test_prompts = create_test_prompts()
    quality_scores = []
    cycles = []
    consolidation_count = 0
    consolidation_cycles = []
    pattern_retrieval_log = []

    print(f"Running {num_cycles} cycles with transfer learning...")
    print(f"Test prompts: {len(test_prompts)} unique prompts (will cycle)\n")

    for i in range(num_cycles):
        prompt = test_prompts[i % len(test_prompts)]

        # Run consciousness cycle with pattern retrieval
        mock_response = f"Response {i}: This discusses {prompt.split()[0].lower()} with specific technical details and concrete examples."

        cycle = manager.consciousness_cycle(
            prompt=prompt,
            response=mock_response,
            task_salience=0.5
        )

        quality_scores.append(cycle.quality_score.normalized)
        cycles.append(cycle)

        # Track consolidation
        if cycle.consolidation_triggered:
            consolidation_count += 1
            consolidation_cycles.append(i + 1)
            print(f"  [Cycle {i+1}] DREAM consolidation #{consolidation_count} triggered!")

        # Track pattern retrieval
        patterns_retrieved = cycle.patterns_retrieved if hasattr(cycle, 'patterns_retrieved') else 0
        pattern_retrieval_log.append((i + 1, patterns_retrieved))

        # Progress updates
        if (i + 1) % 20 == 0:
            recent_quality = np.mean(quality_scores[-20:])
            recent_patterns = np.sum([p for _, p in pattern_retrieval_log[-20:]])
            print(f"  [{i+1}/{num_cycles}] Quality (last 20): {recent_quality:.3f}")
            print(f"              Patterns retrieved (last 20): {recent_patterns}")

    total_patterns = sum(p for _, p in pattern_retrieval_log)
    cycles_with_patterns = sum(1 for _, p in pattern_retrieval_log if p > 0)

    print(f"\n✅ Transfer learning test complete: {num_cycles} cycles")
    print(f"   DREAM consolidations: {consolidation_count}")
    print(f"   Total patterns retrieved: {total_patterns}")
    print(f"   Cycles with patterns: {cycles_with_patterns}/{num_cycles}")
    print(f"   Mean quality: {np.mean(quality_scores):.3f}")
    print(f"   Std dev: {np.std(quality_scores):.3f}")

    return quality_scores, cycles, consolidation_count, consolidation_cycles, pattern_retrieval_log


def analyze_results(baseline_scores, transfer_scores, consolidation_count,
                   consolidation_cycles, pattern_retrieval_log, test_prompts,
                   num_cycles=200):
    """Perform comprehensive statistical analysis."""

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80 + "\n")

    # Convert to numpy arrays
    baseline = np.array(baseline_scores)
    transfer = np.array(transfer_scores)

    # Descriptive statistics
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    baseline_median = np.median(baseline)

    transfer_mean = np.mean(transfer)
    transfer_std = np.std(transfer)
    transfer_median = np.median(transfer)

    mean_improvement = transfer_mean - baseline_mean
    improvement_pct = (mean_improvement / baseline_mean * 100) if baseline_mean > 0 else 0
    median_improvement = transfer_median - baseline_median

    # Post-consolidation analysis (cycles after first DREAM)
    if consolidation_cycles:
        first_consolidation = consolidation_cycles[0]
        post_consolidation_scores = transfer_scores[first_consolidation:]
        post_mean = np.mean(post_consolidation_scores) if post_consolidation_scores else 0
        post_std = np.std(post_consolidation_scores) if post_consolidation_scores else 0
        post_count = len(post_consolidation_scores)
    else:
        post_mean = 0
        post_std = 0
        post_count = 0

    # Pattern retrieval statistics
    total_patterns = sum(p for _, p in pattern_retrieval_log)
    avg_patterns = total_patterns / len(pattern_retrieval_log) if pattern_retrieval_log else 0
    cycles_with_patterns = sum(1 for _, p in pattern_retrieval_log if p > 0)

    print("Descriptive Statistics:")
    print(f"  Baseline:        μ={baseline_mean:.4f}, σ={baseline_std:.4f}, median={baseline_median:.4f}")
    print(f"  Transfer:        μ={transfer_mean:.4f}, σ={transfer_std:.4f}, median={transfer_median:.4f}")
    print(f"  Mean Δ:          {mean_improvement:+.4f} ({improvement_pct:+.2f}%)")
    print(f"  Median Δ:        {median_improvement:+.4f}")

    if post_count > 0:
        print(f"\nPost-Consolidation (after cycle {first_consolidation}):")
        print(f"  Mean quality:    {post_mean:.4f} (σ={post_std:.4f})")
        print(f"  Sample size:     {post_count} cycles")
        post_improvement = post_mean - baseline_mean
        post_pct = (post_improvement / baseline_mean * 100) if baseline_mean > 0 else 0
        print(f"  Improvement:     {post_improvement:+.4f} ({post_pct:+.2f}%)")

    print(f"\nPattern Retrieval:")
    print(f"  Total patterns:  {total_patterns}")
    print(f"  Avg per cycle:   {avg_patterns:.2f}")
    print(f"  Cycles w/patterns: {cycles_with_patterns}/{len(pattern_retrieval_log)}")

    # Statistical significance
    from scipy import stats

    if len(baseline) >= 30 and len(transfer) >= 30:
        t_stat, p_value = stats.ttest_ind(transfer, baseline)
        significant = p_value < 0.05

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((baseline_std**2 + transfer_std**2) / 2)
        cohens_d = mean_improvement / pooled_std if pooled_std > 0 else 0

        print(f"\nStatistical Significance:")
        print(f"  t-statistic:     {t_stat:.4f}")
        print(f"  p-value:         {p_value:.4f}")
        print(f"  Significant:     {'YES' if significant else 'NO'} (α=0.05)")
        print(f"  Effect size (d): {cohens_d:.4f} ({interpret_effect_size(cohens_d)})")
    else:
        t_stat = np.nan
        p_value = np.nan
        significant = False
        cohens_d = 0.0
        print(f"\nStatistical Significance:")
        print(f"  Sample too small for t-test (n<30)")

    # Create quality over time data
    quality_over_time = [(i+1, score) for i, score in enumerate(transfer_scores)]

    # Create results object
    results = ExtendedValidationResults(
        num_cycles=num_cycles,
        circadian_period=100,  # hardcoded for now
        dream_consolidations=consolidation_count,
        baseline_mean=float(baseline_mean),
        baseline_std=float(baseline_std),
        baseline_median=float(baseline_median),
        baseline_scores=baseline_scores,
        transfer_mean=float(transfer_mean),
        transfer_std=float(transfer_std),
        transfer_median=float(transfer_median),
        transfer_scores=transfer_scores,
        post_consolidation_mean=float(post_mean),
        post_consolidation_std=float(post_std),
        post_consolidation_count=post_count,
        total_patterns_retrieved=total_patterns,
        avg_patterns_per_cycle=float(avg_patterns),
        cycles_with_patterns=cycles_with_patterns,
        mean_improvement=float(mean_improvement),
        improvement_percentage=float(improvement_pct),
        t_statistic=float(t_stat) if not np.isnan(t_stat) else 0.0,
        p_value=float(p_value) if not np.isnan(p_value) else 1.0,
        significant=bool(significant),
        effect_size=float(cohens_d),
        consolidation_cycle_numbers=consolidation_cycles,
        quality_over_time=quality_over_time,
        patterns_over_time=pattern_retrieval_log
    )

    return results


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def save_results(results: ExtendedValidationResults, output_path: str = "session52b_validation_results.json"):
    """Save results to JSON for visualization and reporting"""
    results_dict = asdict(results)

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        return obj

    # Recursively convert numpy types in the dict
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        else:
            return convert_numpy(obj)

    results_dict = recursive_convert(results_dict)

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")


def print_conclusion(results: ExtendedValidationResults):
    """Print final conclusion about transfer learning effectiveness"""

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80 + "\n")

    if results.dream_consolidations == 0:
        print("⚠ No DREAM consolidations occurred")
        print("   Need longer test to reach DEEP_NIGHT phase")
        print("   Cannot validate transfer learning without consolidated memories")
        return

    if results.total_patterns_retrieved == 0:
        print("⚠ No patterns retrieved despite consolidation")
        print("   Consolidation may not have created retrievable patterns")
        print("   Or pattern matching criteria too strict")
        return

    # Interpret results
    effect = interpret_effect_size(results.effect_size)

    if results.significant and results.mean_improvement > 0:
        print(f"✅ Transfer learning shows SIGNIFICANT quality improvement")
        print(f"   Mean improvement: {results.mean_improvement:.4f} ({results.improvement_percentage:+.2f}%)")
        print(f"   Effect size: {results.effect_size:.4f} ({effect})")
        print(f"   p-value: {results.p_value:.4f} (significant at α=0.05)")
        print(f"\n   Patterns retrieved: {results.total_patterns_retrieved} across {results.cycles_with_patterns} cycles")
        print(f"   DREAM consolidations: {results.dream_consolidations}")
        print(f"\n   ✨ Session 51 transfer learning VALIDATED")
    elif results.mean_improvement > 0:
        print(f"≈ Transfer learning shows positive but NON-SIGNIFICANT improvement")
        print(f"   Mean improvement: {results.mean_improvement:.4f} ({results.improvement_percentage:+.2f}%)")
        print(f"   Effect size: {results.effect_size:.4f} ({effect})")
        print(f"   p-value: {results.p_value:.4f} (not significant at α=0.05)")
        print(f"\n   May need more cycles or different prompts to show significance")
    else:
        print(f"≈ Transfer learning shows NEGLIGIBLE effect")
        print(f"   Mean change: {results.mean_improvement:.4f} ({results.improvement_percentage:+.2f}%)")
        print(f"\n   Interpretation:")
        print(f"   Pattern retrieval working but not improving quality.")
        print(f"   May need:")
        print(f"   - More diverse experiences before consolidation")
        print(f"   - Enhanced pattern matching (semantic similarity)")
        print(f"   - Different quality metrics")


if __name__ == "__main__":
    print("="*80)
    print("SESSION 52b: EXTENDED TRANSFER LEARNING QUALITY VALIDATION")
    print("="*80)
    print("\nObjective: Validate transfer learning with DREAM consolidation")
    print("Method: Extended A/B test (200 cycles) to trigger DREAM phase")
    print("Success: Significant quality improvement with pattern retrieval\n")

    NUM_CYCLES = 200  # Long enough to trigger DREAM at ~cycle 85-100

    # Run baseline test (no transfer learning)
    baseline_scores, baseline_cycles, _ = run_baseline_test(NUM_CYCLES)

    # Run transfer learning test
    transfer_scores, transfer_cycles, consolidation_count, consolidation_cycles, pattern_log = \
        run_transfer_learning_test(NUM_CYCLES)

    # Analyze and compare
    test_prompts = create_test_prompts()
    results = analyze_results(
        baseline_scores,
        transfer_scores,
        consolidation_count,
        consolidation_cycles,
        pattern_log,
        test_prompts,
        NUM_CYCLES
    )

    # Save results
    save_results(results)

    # Print conclusion
    print_conclusion(results)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
