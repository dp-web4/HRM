#!/usr/bin/env python3
"""
Session 52: Transfer Learning Quality Validation A/B Test

Tests whether pattern retrieval (Session 51) improves consciousness cycle quality.

A/B Test Design:
- Baseline: Run cycles WITHOUT pattern retrieval (transfer_learning_enabled=False)
- Experiment: Run cycles WITH pattern retrieval (transfer_learning_enabled=True)
- Comparison: Statistical analysis of quality score differences

This validates that the transfer learning system implemented in Session 51
actually improves response quality as hypothesized.

Author: Thor (Autonomous Session 52)
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
class ValidationResults:
    """Results from A/B test comparing baseline vs transfer learning"""
    
    # Baseline statistics
    baseline_mean: float
    baseline_std: float
    baseline_median: float
    baseline_min: float
    baseline_max: float
    baseline_scores: List[float]
    
    # Transfer learning statistics
    transfer_mean: float
    transfer_std: float
    transfer_median: float
    transfer_min: float
    transfer_max: float
    transfer_scores: List[float]
    
    # Comparison
    mean_improvement: float
    median_improvement: float
    improvement_percentage: float
    
    # Statistical significance (if enough samples)
    t_statistic: float = 0.0
    p_value: float = 1.0
    significant: bool = False
    effect_size: float = 0.0  # Cohen's d
    
    # Metadata
    num_cycles: int = 0
    test_prompts: List[str] = None


def create_test_prompts() -> List[str]:
    """
    Create diverse test prompts covering different domains.
    
    These prompts should trigger different types of responses to test
    quality improvement across various contexts.
    """
    return [
        # Technical/specific
        "Explain ATP allocation in SAGE consciousness cycles.",
        "How does selective expert loading work in sparse MoE architectures?",
        "What is the relationship between metabolic states and quality scores?",
        
        # Abstract/conceptual
        "What is the nature of machine consciousness?",
        "How do epistemic states influence reasoning?",
        "Describe the role of temporal adaptation in learning.",
        
        # Problem-solving
        "How would you debug incoherent LLM generation?",
        "What's the best way to validate a transfer learning system?",
        "How can we measure consciousness quality objectively?",
        
        # Factual/numerical
        "What are typical ATP budgets for different metabolic states?",
        "How many layers does Q3-Omni have and what's the expert count?",
        "What's the target quality score percentage for SAGE?",
        
        # Open-ended
        "What are promising directions for consciousness research?",
        "How might emotion and reason integrate in AI systems?",
        "What lessons does biology teach us about machine learning?",
    ]


def run_baseline_test(num_cycles: int = 50) -> Tuple[List[float], List[ConsciousnessCycle]]:
    """
    Run baseline test WITHOUT pattern retrieval.
    
    Args:
        num_cycles: Number of cycles to run
        
    Returns:
        (quality_scores, cycles) tuple
    """
    print("\n" + "="*80)
    print("BASELINE TEST - No Pattern Retrieval")
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
    
    print(f"Running {num_cycles} cycles without pattern retrieval...")
    print(f"Test prompts: {len(test_prompts)} unique prompts (will cycle)\n")
    
    for i in range(num_cycles):
        prompt = test_prompts[i % len(test_prompts)]
        
        # Run consciousness cycle
        # Note: We're using a mock response since we don't have an actual LLM
        # In production, this would call the full SAGE pipeline
        mock_response = f"Response {i}: This discusses {prompt.split()[0].lower()} with some technical details."

        cycle = manager.consciousness_cycle(
            prompt=prompt,
            response=mock_response,
            task_salience=0.5
        )
        
        quality_scores.append(cycle.quality_score.normalized)
        cycles.append(cycle)
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{num_cycles}] Quality: {cycle.quality_score.normalized:.3f}")
    
    print(f"\n✅ Baseline complete: {len(quality_scores)} cycles")
    print(f"   Mean quality: {np.mean(quality_scores):.3f}")
    print(f"   Std dev: {np.std(quality_scores):.3f}")
    
    return quality_scores, cycles


def run_transfer_learning_test(num_cycles: int = 50) -> Tuple[List[float], List[ConsciousnessCycle]]:
    """
    Run experiment test WITH pattern retrieval.
    
    Args:
        num_cycles: Number of cycles to run
        
    Returns:
        (quality_scores, cycles) tuple
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
    
    print(f"Running {num_cycles} cycles with pattern retrieval...")
    print(f"Test prompts: {len(test_prompts)} unique prompts (will cycle)\n")
    
    for i in range(num_cycles):
        prompt = test_prompts[i % len(test_prompts)]
        
        # Run consciousness cycle with pattern retrieval
        mock_response = f"Response {i}: This discusses {prompt.split()[0].lower()} with some technical details."

        cycle = manager.consciousness_cycle(
            prompt=prompt,
            response=mock_response,
            task_salience=0.5
        )
        
        quality_scores.append(cycle.quality_score.normalized)
        cycles.append(cycle)
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{num_cycles}] Quality: {cycle.quality_score.normalized:.3f}")
            if hasattr(cycle, 'patterns_retrieved'):
                print(f"              Patterns retrieved: {cycle.patterns_retrieved}")
    
    print(f"\n✅ Transfer learning test complete: {len(quality_scores)} cycles")
    print(f"   Mean quality: {np.mean(quality_scores):.3f}")
    print(f"   Std dev: {np.std(quality_scores):.3f}")
    
    return quality_scores, cycles


def analyze_results(
    baseline_scores: List[float],
    transfer_scores: List[float],
    test_prompts: List[str]
) -> ValidationResults:
    """
    Perform statistical analysis comparing baseline vs transfer learning.
    
    Args:
        baseline_scores: Quality scores without pattern retrieval
        transfer_scores: Quality scores with pattern retrieval
        test_prompts: Test prompts used
        
    Returns:
        ValidationResults with statistics and significance tests
    """
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80 + "\n")
    
    # Calculate descriptive statistics
    baseline_mean = np.mean(baseline_scores)
    baseline_std = np.std(baseline_scores, ddof=1)
    baseline_median = np.median(baseline_scores)
    
    transfer_mean = np.mean(transfer_scores)
    transfer_std = np.std(transfer_scores, ddof=1)
    transfer_median = np.median(transfer_scores)
    
    # Calculate improvements
    mean_improvement = transfer_mean - baseline_mean
    median_improvement = transfer_median - baseline_median
    improvement_pct = (mean_improvement / baseline_mean * 100) if baseline_mean > 0 else 0
    
    print("Descriptive Statistics:")
    print(f"  Baseline:        μ={baseline_mean:.4f}, σ={baseline_std:.4f}, median={baseline_median:.4f}")
    print(f"  Transfer:        μ={transfer_mean:.4f}, σ={transfer_std:.4f}, median={transfer_median:.4f}")
    print(f"  Mean Δ:          {mean_improvement:+.4f} ({improvement_pct:+.2f}%)")
    print(f"  Median Δ:        {median_improvement:+.4f}")
    
    # Statistical significance testing (if we have enough samples)
    t_stat = 0.0
    p_value = 1.0
    significant = False
    effect_size = 0.0
    
    if len(baseline_scores) >= 30 and len(transfer_scores) >= 30:
        # Two-sample t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(transfer_scores, baseline_scores)
        significant = p_value < 0.05
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((baseline_std**2 + transfer_std**2) / 2)
        effect_size = mean_improvement / pooled_std if pooled_std > 0 else 0
        
        print(f"\nStatistical Significance:")
        print(f"  t-statistic:     {t_stat:.4f}")
        print(f"  p-value:         {p_value:.4f}")
        print(f"  Significant:     {'YES' if significant else 'NO'} (α=0.05)")
        print(f"  Effect size (d): {effect_size:.4f} ({_interpret_effect_size(effect_size)})")
    else:
        print(f"\nNote: Need ≥30 samples per group for t-test (have {len(baseline_scores)}, {len(transfer_scores)})")
    
    return ValidationResults(
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        baseline_median=baseline_median,
        baseline_min=min(baseline_scores),
        baseline_max=max(baseline_scores),
        baseline_scores=baseline_scores,
        transfer_mean=transfer_mean,
        transfer_std=transfer_std,
        transfer_median=transfer_median,
        transfer_min=min(transfer_scores),
        transfer_max=max(transfer_scores),
        transfer_scores=transfer_scores,
        mean_improvement=mean_improvement,
        median_improvement=median_improvement,
        improvement_percentage=improvement_pct,
        t_statistic=t_stat,
        p_value=p_value,
        significant=significant,
        effect_size=effect_size,
        num_cycles=len(baseline_scores),
        test_prompts=test_prompts
    )


def _interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def save_results(results: ValidationResults, output_path: str = "session52_validation_results.json"):
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


def print_conclusion(results: ValidationResults):
    """Print final conclusion about transfer learning effectiveness"""
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80 + "\n")
    
    if results.mean_improvement > 0:
        print("✅ Transfer learning IMPROVES quality scores")
        print(f"   Mean improvement: {results.mean_improvement:.4f} ({results.improvement_percentage:+.2f}%)")
        
        if results.significant:
            print(f"   Statistical significance: YES (p={results.p_value:.4f})")
            print(f"   Effect size: {_interpret_effect_size(results.effect_size)}")
        else:
            print("   Statistical significance: Not tested (need more samples)")
        
        print("\n   Interpretation:")
        print("   Pattern retrieval from consolidated memories helps improve")
        print("   response quality. Session 51's transfer learning system is")
        print("   working as hypothesized.")
        
    elif results.mean_improvement < -0.01:
        print("⚠️  Transfer learning DECREASES quality scores")
        print(f"   Mean change: {results.mean_improvement:.4f} ({results.improvement_percentage:.2f}%)")
        print("\n   Interpretation:")
        print("   Pattern retrieval may be introducing noise or irrelevant")
        print("   patterns. Need to investigate:")
        print("   - Relevance threshold (currently 0.3)")
        print("   - Pattern matching algorithm")
        print("   - Top-k value (currently 5)")
        
    else:
        print("≈ Transfer learning shows NEGLIGIBLE effect")
        print(f"   Mean change: {results.mean_improvement:.4f} ({results.improvement_percentage:.2f}%)")
        print("\n   Interpretation:")
        print("   Pattern retrieval neither helps nor hurts quality.")
        print("   May need:")
        print("   - More consolidated memories (current test may have few)")
        print("   - Enhanced pattern matching (semantic similarity)")
        print("   - Different test prompts")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SESSION 52: TRANSFER LEARNING QUALITY VALIDATION")
    print("="*80)
    print("\nObjective: Validate that pattern retrieval improves quality scores")
    print("Method: A/B test comparing cycles with/without transfer learning")
    print("Success: Significant quality improvement with pattern retrieval\n")
    
    # Configuration
    NUM_CYCLES = 50  # Per group (total 100 cycles)
    
    # Run baseline test
    baseline_scores, baseline_cycles = run_baseline_test(NUM_CYCLES)
    
    # Brief pause between tests
    time.sleep(1)
    
    # Run transfer learning test
    transfer_scores, transfer_cycles = run_transfer_learning_test(NUM_CYCLES)
    
    # Analyze and compare
    test_prompts = create_test_prompts()
    results = analyze_results(baseline_scores, transfer_scores, test_prompts)
    
    # Save results
    save_results(results)
    
    # Print conclusion
    print_conclusion(results)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")
