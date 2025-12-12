#!/usr/bin/env python3
"""
Session 38: Q1 and M3 Validation with Real SAGE Responses

Validates observational framework predictions using actual SAGE responses
collected in session38_real_conversation_collector.py:

Q1: Response Quality Threshold
    - Prediction: 4-metric quality score ≥ 0.85 for 95% of responses
    - Measurement: Proportion of high-quality responses

M3: Confidence-Quality Correlation
    - Prediction: Correlation r > 0.60 between confidence and quality
    - Measurement: Pearson correlation coefficient

This addresses Session 37 M3 gap (r=0.379 with synthetic sketches)
by using real SAGE responses with full quality scores.

Author: Thor (Autonomous Session 38)
Date: 2025-12-12
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from scipy import stats

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.sage_real_measurements import SAGERealMeasurements, ObservationResult


def load_real_conversations(data_dir: str = "/home/dp/ai-workspace/HRM/sage/data/real_conversations"):
    """
    Load real SAGE conversation data.

    Returns:
        Tuple of (quality_scores, confidence_values, responses)
    """
    data_path = Path(data_dir)

    # Find most recent conversation file
    json_files = sorted(data_path.glob("real_sage_conversation_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No conversation files found in {data_dir}")

    latest_file = json_files[-1]
    print(f"Loading: {latest_file.name}")

    with open(latest_file, 'r') as f:
        data = json.load(f)

    quality_scores = []
    confidence_values = []
    responses = []

    for r in data['responses']:
        quality_scores.append(r['quality_score']['normalized'])
        confidence_values.append(r['epistemic_metrics']['confidence'])
        responses.append(r)

    return quality_scores, confidence_values, responses


def measure_q1_quality_threshold(quality_scores: List[float]) -> ObservationResult:
    """
    Measure Q1: Response Quality Threshold

    Prediction: ≥0.85 quality for 95% of responses

    Args:
        quality_scores: List of normalized quality scores

    Returns:
        ObservationResult with proportion ≥0.85
    """
    threshold = 0.85
    exceeding_threshold = sum(1 for q in quality_scores if q >= threshold)
    proportion = exceeding_threshold / len(quality_scores)

    # Binomial standard error
    error = np.sqrt(proportion * (1 - proportion) / len(quality_scores))

    # Calculate significance
    # Null hypothesis: proportion = 0.95 (target)
    null_value = 0.95
    if error > 0:
        sigma = abs(proportion - null_value) / error
    else:
        sigma = 0.0

    validated = (proportion >= 0.95)

    return ObservationResult(
        prediction_id='Q1',
        observed_value=proportion,
        observed_error=error,
        significance=sigma,
        validated=validated,
        measurement_time=time.time(),
        sample_size=len(quality_scores),
        notes=f"Threshold: {threshold}, Count: {exceeding_threshold}/{len(quality_scores)}, Mean quality: {np.mean(quality_scores):.3f}"
    )


def measure_m3_confidence_quality_correlation(quality_scores: List[float],
                                             confidence_values: List[float]) -> ObservationResult:
    """
    Measure M3: Confidence-Quality Correlation

    Prediction: r > 0.60 correlation between confidence and quality

    Args:
        quality_scores: List of quality scores
        confidence_values: List of confidence values

    Returns:
        ObservationResult with Pearson correlation coefficient
    """
    # Calculate Pearson correlation
    r, p_value = stats.pearsonr(confidence_values, quality_scores)

    # Standard error of correlation
    n = len(quality_scores)
    se = np.sqrt((1 - r**2) / (n - 2)) if n > 2 else 0.1

    # Calculate significance
    # Null hypothesis: r = 0.60 (target threshold)
    null_value = 0.60
    if se > 0:
        sigma = abs(r - null_value) / se
    else:
        sigma = 0.0

    validated = (r > 0.60)

    return ObservationResult(
        prediction_id='M3',
        observed_value=r,
        observed_error=se,
        significance=sigma,
        validated=validated,
        measurement_time=time.time(),
        sample_size=n,
        notes=f"p-value: {p_value:.6f}, n_pairs: {n}, mean_conf: {np.mean(confidence_values):.3f}, mean_qual: {np.mean(quality_scores):.3f}"
    )


def analyze_quality_distribution(quality_scores: List[float], responses: List[dict]):
    """Analyze quality score distribution by category"""
    print("\nQuality Score Distribution:")
    print(f"  Mean: {np.mean(quality_scores):.3f}")
    print(f"  Median: {np.median(quality_scores):.3f}")
    print(f"  Std Dev: {np.std(quality_scores):.3f}")
    print(f"  Min: {np.min(quality_scores):.3f}")
    print(f"  Max: {np.max(quality_scores):.3f}")
    print()

    # Distribution by thresholds
    bins = [0.0, 0.70, 0.80, 0.85, 0.90, 0.95, 1.0]
    hist, _ = np.histogram(quality_scores, bins=bins)

    print("Quality Distribution by Threshold:")
    for i in range(len(bins) - 1):
        count = hist[i]
        pct = count / len(quality_scores) * 100
        print(f"  {bins[i]:.2f}-{bins[i+1]:.2f}: {count:2d} ({pct:5.1f}%)")
    print()

    # By category
    category_scores = {}
    for r in responses:
        # Infer category from prompt_id
        prompt_id = r['prompt_id']
        if prompt_id.startswith('tech_'):
            cat = 'technical'
        elif prompt_id.startswith('explore_'):
            cat = 'exploratory'
        elif prompt_id.startswith('problem_'):
            cat = 'problem_solving'
        elif prompt_id.startswith('ambiguous_'):
            cat = 'ambiguous'
        elif prompt_id.startswith('routine_'):
            cat = 'routine'
        elif prompt_id.startswith('synthesis_'):
            cat = 'synthesis'
        elif prompt_id.startswith('analysis_'):
            cat = 'analysis'
        else:
            cat = 'other'

        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(r['quality_score']['normalized'])

    print("Quality by Category:")
    for cat in sorted(category_scores.keys()):
        scores = category_scores[cat]
        mean_score = np.mean(scores)
        count = len(scores)
        high_quality = sum(1 for s in scores if s >= 0.85)
        print(f"  {cat:15s}: {mean_score:.3f} avg, {high_quality}/{count} ≥0.85")
    print()


def analyze_correlation_patterns(quality_scores: List[float],
                                 confidence_values: List[float],
                                 responses: List[dict]):
    """Analyze correlation patterns and outliers"""
    print("\nConfidence-Quality Correlation Analysis:")

    # Scatter plot data (text representation)
    print("\nConfidence vs Quality Pairs (sample):")
    for i in range(min(10, len(responses))):
        conf = confidence_values[i]
        qual = quality_scores[i]
        prompt_id = responses[i]['prompt_id']
        print(f"  {prompt_id:15s}: conf={conf:.3f}, qual={qual:.3f}")

    # Identify outliers
    print("\nOutliers (high confidence, low quality):")
    outliers_high_conf_low_qual = [
        (i, responses[i]) for i in range(len(responses))
        if confidence_values[i] > 0.70 and quality_scores[i] < 0.70
    ]
    if outliers_high_conf_low_qual:
        for i, r in outliers_high_conf_low_qual[:5]:
            print(f"  {r['prompt_id']}: conf={confidence_values[i]:.3f}, qual={quality_scores[i]:.3f}")
    else:
        print("  None found")

    print("\nOutliers (low confidence, high quality):")
    outliers_low_conf_high_qual = [
        (i, responses[i]) for i in range(len(responses))
        if confidence_values[i] < 0.50 and quality_scores[i] > 0.85
    ]
    if outliers_low_conf_high_qual:
        for i, r in outliers_low_conf_high_qual[:5]:
            print(f"  {r['prompt_id']}: conf={confidence_values[i]:.3f}, qual={quality_scores[i]:.3f}")
    else:
        print("  None found")
    print()


def main():
    """Run Session 38 Q1/M3 validation"""
    print()
    print("=" * 80)
    print("SESSION 38: Q1 AND M3 VALIDATION WITH REAL SAGE RESPONSES")
    print("=" * 80)
    print()

    # Load real conversation data
    print("Loading real SAGE conversation data...")
    quality_scores, confidence_values, responses = load_real_conversations()
    print(f"Loaded {len(responses)} responses")
    print()

    # Q1: Response Quality Threshold
    print("=" * 80)
    print("Q1: RESPONSE QUALITY THRESHOLD")
    print("=" * 80)
    print()
    print("Prediction: 4-metric quality score ≥ 0.85 for 95% of responses")
    print()

    result_q1 = measure_q1_quality_threshold(quality_scores)

    print(f"Sample size: {result_q1.sample_size} responses")
    print(f"Proportion ≥0.85: {result_q1.observed_value:.3f} ± {result_q1.observed_error:.3f}")
    print(f"Target: 0.95 (95% of responses)")
    print(f"Significance: {result_q1.significance:.2f}σ")
    print(f"Notes: {result_q1.notes}")
    print()

    if result_q1.validated:
        print("✅ Q1 VALIDATED")
    else:
        gap = 0.95 - result_q1.observed_value
        print(f"⚠️  Q1 gap: {result_q1.observed_value:.1%} vs 95% target (gap: {gap:.1%})")

    analyze_quality_distribution(quality_scores, responses)

    # M3: Confidence-Quality Correlation
    print("=" * 80)
    print("M3: CONFIDENCE-QUALITY CORRELATION")
    print("=" * 80)
    print()
    print("Prediction: Correlation r > 0.60 between confidence and quality")
    print()

    result_m3 = measure_m3_confidence_quality_correlation(quality_scores, confidence_values)

    print(f"Sample size: {result_m3.sample_size} pairs")
    print(f"Correlation (r): {result_m3.observed_value:.3f} ± {result_m3.observed_error:.3f}")
    print(f"Target: r > 0.60")
    print(f"Significance: {result_m3.significance:.2f}σ")
    print(f"Notes: {result_m3.notes}")
    print()

    if result_m3.validated:
        print("✅ M3 VALIDATED")
    else:
        gap = 0.60 - result_m3.observed_value
        print(f"⚠️  M3 gap: r={result_m3.observed_value:.3f} vs 0.60 target (gap: {gap:.3f})")

    analyze_correlation_patterns(quality_scores, confidence_values, responses)

    # Comparison with Session 37
    print("=" * 80)
    print("SESSION 37 vs SESSION 38 COMPARISON")
    print("=" * 80)
    print()
    print("M3 (Confidence-Quality Correlation):")
    print(f"  Session 37 (synthetic sketches): r = 0.379")
    print(f"  Session 38 (real SAGE responses): r = {result_m3.observed_value:.3f}")
    print(f"  Improvement: {result_m3.observed_value - 0.379:+.3f}")
    print()

    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()

    print("Measurements:")
    print(f"  Q1 (Quality Threshold): {result_q1.observed_value:.1%} vs 95% target")
    print(f"  M3 (Confidence-Quality): r = {result_m3.observed_value:.3f} vs 0.60 target")
    print()

    validated_count = sum([result_q1.validated, result_m3.validated])
    print(f"Validated: {validated_count}/2 predictions")
    print()

    if validated_count == 2:
        print("✅ SESSION 38 SUCCESS: Both Q1 and M3 validated with real SAGE responses")
        return 0
    elif validated_count == 1:
        print("✅ SESSION 38 PARTIAL SUCCESS: One prediction validated")
        return 0
    else:
        print("⚠️  SESSION 38: Predictions not meeting targets with real data")
        print()
        print("Analysis:")
        if not result_q1.validated:
            print(f"  Q1: Real SAGE responses achieved {result_q1.observed_value:.1%} quality")
            print(f"      Target of 95% may be too high for diverse conversation topics")
        if not result_m3.validated:
            print(f"  M3: Correlation r={result_m3.observed_value:.3f} improved from Session 37 (0.379)")
            print(f"      But still below 0.60 target - may need calibration adjustment")
        return 1


if __name__ == '__main__':
    exit(main())
