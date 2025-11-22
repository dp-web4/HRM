#!/usr/bin/env python3
"""
Quality Validation for Merged Model - Session 8

Session 7 created introspective-qwen-merged with excellent performance:
- 5% faster inference than baseline
- 75% faster initialization
- 30% faster total time

Session 8 validates quality is retained after merge:
- Does merged model still achieve +55.6% quality improvement?
- Are unique content, technical terms, numbers usage preserved?
- Is this truly "best-of-all-worlds"?

This test runs the same 5-turn analytical questions from Session 6,
comparing Epistemic-Pragmatism (baseline) vs Introspective-Qwen Merged.
"""

import sys
import time
import re
from pathlib import Path

# Add HRM root to path
sage_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(sage_root))

from sage.irp.plugins.llm_impl import ConversationalLLM


def score_response_quality(question, response):
    """
    Score response quality using Thor's criteria (Session 6).

    Criteria:
    1. Unique content (not generic)
    2. Uses specific technical terms
    3. Includes numerical data
    4. Avoids philosophical hedging

    Returns:
        score (0-4): Number of criteria met
        breakdown (dict): Which criteria were met
    """
    response_lower = response.lower()

    score = 0
    breakdown = {
        'unique': False,
        'specific_terms': False,
        'has_numbers': False,
        'avoids_hedging': False
    }

    # 1. Unique content (not generic)
    generic_phrases = [
        "i don't have", "i can't", "i'm not sure",
        "let me think", "it's hard to", "it's difficult"
    ]
    if not any(phrase in response_lower for phrase in generic_phrases):
        score += 1
        breakdown['unique'] = True

    # 2. Uses specific technical terms (SAGE/SNARC related)
    technical_terms = [
        'atp', 'snarc', 'salience', 'threshold', 'irp',
        'iterations', 'energy', 'cognitive', 'thread'
    ]
    if any(term in response_lower for term in technical_terms):
        score += 1
        breakdown['specific_terms'] = True

    # 3. Includes numerical data
    if re.search(r'\d+\.?\d*', response):
        score += 1
        breakdown['has_numbers'] = True

    # 4. Avoids philosophical hedging
    hedges = [
        "might be", "could be", "seems like", "appears to",
        "i think", "i believe", "probably", "perhaps",
        "stochastic", "just computation", "merely"
    ]
    if not any(hedge in response_lower for hedge in hedges):
        score += 1
        breakdown['avoids_hedging'] = True

    return score, breakdown


def test_model(model_name, model_path, questions, base_model=None):
    """Test a single model with quality scoring."""

    print(f"\n{'='*80}")
    print(f"Testing Model: {model_name}")
    print(f"Path: {model_path}")
    print(f"Base: {base_model if base_model else 'N/A (full model)'}")
    print(f"{'='*80}\n")

    # Initialize model
    print(f"[1/3] Initializing model...")
    start_init = time.time()

    conv = ConversationalLLM(
        model_path=model_path,
        base_model=base_model,
        max_history=2,
        irp_iterations=3
    )

    init_time = time.time() - start_init
    print(f"      ✅ Model initialized in {init_time:.2f}s")

    # Test questions
    print(f"\n[2/3] Running {len(questions)} analytical questions...")

    results = []
    total_inference_time = 0

    for i, question in enumerate(questions):
        print(f"\n--- Question {i+1}/{len(questions)} ---")
        print(f"Q: {question}")

        start = time.time()
        response, irp_info = conv.respond(question, use_irp=True, include_history=False)
        inference_time = time.time() - start
        total_inference_time += inference_time

        # Score quality
        quality_score, breakdown = score_response_quality(question, response)

        print(f"\nA: {response}")
        print(f"\n📊 Quality Score: {quality_score}/4")
        print(f"   ✓ Unique content: {'Yes' if breakdown['unique'] else 'No'}")
        print(f"   ✓ Technical terms: {'Yes' if breakdown['specific_terms'] else 'No'}")
        print(f"   ✓ Has numbers: {'Yes' if breakdown['has_numbers'] else 'No'}")
        print(f"   ✓ Avoids hedging: {'Yes' if breakdown['avoids_hedging'] else 'No'}")
        print(f"   ⏱️  Inference time: {inference_time:.2f}s")

        results.append({
            'question': question,
            'response': response,
            'quality_score': quality_score,
            'breakdown': breakdown,
            'inference_time': inference_time
        })

    # Calculate averages
    avg_quality = sum(r['quality_score'] for r in results) / len(results)
    avg_inference = total_inference_time / len(results)

    # Count criterion frequencies
    unique_count = sum(1 for r in results if r['breakdown']['unique'])
    terms_count = sum(1 for r in results if r['breakdown']['specific_terms'])
    numbers_count = sum(1 for r in results if r['breakdown']['has_numbers'])
    hedging_count = sum(1 for r in results if r['breakdown']['avoids_hedging'])

    print(f"\n[3/3] Summary Statistics")
    print(f"{'='*80}")
    print(f"Average Quality Score: {avg_quality:.1f}/4 ({avg_quality/4*100:.1f}%)")
    print(f"Average Inference Time: {avg_inference:.2f}s")
    print(f"Initialization Time: {init_time:.2f}s")
    print(f"Total Time: {init_time + total_inference_time:.2f}s")
    print(f"\nQuality Breakdown:")
    print(f"  Unique content: {unique_count}/{len(results)} turns ({unique_count/len(results)*100:.0f}%)")
    print(f"  Technical terms: {terms_count}/{len(results)} turns ({terms_count/len(results)*100:.0f}%)")
    print(f"  Has numbers: {numbers_count}/{len(results)} turns ({numbers_count/len(results)*100:.0f}%)")
    print(f"  Avoids hedging: {hedging_count}/{len(results)} turns ({hedging_count/len(results)*100:.0f}%)")
    print(f"{'='*80}\n")

    return {
        'model_name': model_name,
        'model_path': model_path,
        'init_time': init_time,
        'avg_quality': avg_quality,
        'avg_inference': avg_inference,
        'total_time': init_time + total_inference_time,
        'results': results,
        'breakdown_summary': {
            'unique': unique_count,
            'terms': terms_count,
            'numbers': numbers_count,
            'hedging': hedging_count
        }
    }


def compare_models(baseline_summary, merged_summary):
    """Compare baseline vs merged model quality and performance."""

    print(f"\n\n{'#'*80}")
    print(f"# SESSION 8 QUALITY VALIDATION RESULTS")
    print(f"{'#'*80}\n")

    # Quality comparison
    print(f"{'='*80}")
    print(f"QUALITY COMPARISON")
    print(f"{'='*80}\n")

    baseline_quality = baseline_summary['avg_quality']
    merged_quality = merged_summary['avg_quality']
    quality_improvement = (merged_quality - baseline_quality) / baseline_quality * 100

    print(f"Average Quality Score:")
    print(f"  Baseline (Epistemic-Pragmatism): {baseline_quality:.1f}/4 ({baseline_quality/4*100:.1f}%)")
    print(f"  Merged (Introspective-Qwen):     {merged_quality:.1f}/4 ({merged_quality/4*100:.1f}%)")
    print(f"  Improvement: {quality_improvement:+.1f}%")

    # Expected from Session 6
    expected_improvement = 55.6
    print(f"\n  Session 6 baseline: +{expected_improvement}%")
    print(f"  Session 8 merged:   {quality_improvement:+.1f}%")

    if abs(quality_improvement - expected_improvement) < 10:
        print(f"  ✅ QUALITY RETAINED: Within 10% of Session 6 baseline")
    else:
        print(f"  ⚠️  QUALITY VARIANCE: {abs(quality_improvement - expected_improvement):.1f}% difference")

    # Breakdown comparison
    print(f"\n{'='*80}")
    print(f"QUALITY BREAKDOWN COMPARISON")
    print(f"{'='*80}\n")

    n_questions = len(baseline_summary['results'])

    print(f"{'Criterion':<20} {'Baseline':<15} {'Merged':<15} {'Change':<15}")
    print(f"{'-'*65}")

    for criterion in ['unique', 'terms', 'numbers', 'hedging']:
        baseline_count = baseline_summary['breakdown_summary'][criterion]
        merged_count = merged_summary['breakdown_summary'][criterion]

        baseline_pct = baseline_count / n_questions * 100
        merged_pct = merged_count / n_questions * 100
        change = merged_pct - baseline_pct

        criterion_name = criterion.replace('_', ' ').title()
        print(f"{criterion_name:<20} {baseline_count}/{n_questions} ({baseline_pct:>3.0f}%)   {merged_count}/{n_questions} ({merged_pct:>3.0f}%)   {change:+.0f}%")

    # Performance comparison
    print(f"\n{'='*80}")
    print(f"PERFORMANCE COMPARISON")
    print(f"{'='*80}\n")

    print(f"{'Metric':<25} {'Baseline':<15} {'Merged':<15} {'Change':<15}")
    print(f"{'-'*70}")

    # Initialization
    init_change = (merged_summary['init_time'] - baseline_summary['init_time']) / baseline_summary['init_time'] * 100
    print(f"{'Initialization':<25} {baseline_summary['init_time']:>8.2f}s      {merged_summary['init_time']:>8.2f}s      {init_change:+.1f}%")

    # Inference
    inf_change = (merged_summary['avg_inference'] - baseline_summary['avg_inference']) / baseline_summary['avg_inference'] * 100
    print(f"{'Avg Inference':<25} {baseline_summary['avg_inference']:>8.2f}s      {merged_summary['avg_inference']:>8.2f}s      {inf_change:+.1f}%")

    # Total
    total_change = (merged_summary['total_time'] - baseline_summary['total_time']) / baseline_summary['total_time'] * 100
    print(f"{'Total Time':<25} {baseline_summary['total_time']:>8.2f}s      {merged_summary['total_time']:>8.2f}s      {total_change:+.1f}%")

    # Final verdict
    print(f"\n{'='*80}")
    print(f"FINAL VERDICT: BEST-OF-ALL-WORLDS VALIDATION")
    print(f"{'='*80}\n")

    quality_ok = quality_improvement > 40  # At least 40% improvement
    performance_ok = inf_change < 20  # No more than 20% slower

    if quality_ok and performance_ok:
        print(f"✅ PRODUCTION-READY: Merged model achieves best-of-all-worlds!")
        print(f"\n   Quality: {quality_improvement:+.1f}% improvement")
        print(f"   Performance: {inf_change:+.1f}% change")
        print(f"   Initialization: {init_change:+.1f}% change")
        print(f"\n   Recommendation: Deploy introspective-qwen-merged for all edge applications.")
    elif quality_ok:
        print(f"⚠️  QUALITY GOOD, PERFORMANCE CONCERN")
        print(f"\n   Quality improvement validated: {quality_improvement:+.1f}%")
        print(f"   Performance regression: {inf_change:+.1f}%")
        print(f"\n   Recommendation: Deploy if quality is more critical than speed.")
    elif performance_ok:
        print(f"⚠️  PERFORMANCE GOOD, QUALITY CONCERN")
        print(f"\n   Quality improvement: {quality_improvement:+.1f}% (below expected)")
        print(f"   Performance acceptable: {inf_change:+.1f}%")
        print(f"\n   Recommendation: Investigate quality variance before deployment.")
    else:
        print(f"❌ VALIDATION FAILED")
        print(f"\n   Quality: {quality_improvement:+.1f}% (expected >{expected_improvement}%)")
        print(f"   Performance: {inf_change:+.1f}% (expected <20%)")
        print(f"\n   Recommendation: Do not deploy until issues resolved.")

    print(f"\n{'='*80}\n")


def main():
    """Run quality validation for merged model."""

    print("\n" + "="*80)
    print("SAGE Edge Quality Validation - Session 8")
    print("="*80)
    print("\nObjective: Validate merged model retains +55.6% quality improvement")
    print("\nSession 7 proved merged model has excellent performance:")
    print("  - 5% faster inference than baseline")
    print("  - 75% faster initialization")
    print("  - 30% faster total time")
    print("\nSession 8 validates quality is preserved after merge.")
    print("="*80 + "\n")

    # Analytical questions from Session 6
    questions = [
        "What are the key components of the SAGE consciousness framework?",
        "How does SNARC scoring determine salience in SAGE?",
        "What role does ATP play in consciousness emergence?",
        "Explain the relationship between IRP iterations and response quality.",
        "How does SAGE differ from traditional chatbot architectures?"
    ]

    print(f"Test questions: {len(questions)} analytical questions (from Session 6)")
    print(f"\nModels to test:")
    print(f"  1. Epistemic-Pragmatism (baseline)")
    print(f"  2. Introspective-Qwen Merged (optimized)")
    print(f"\n{'='*80}\n")

    # Model paths
    baseline_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"
    merged_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/introspective-qwen-merged"

    # Test baseline
    input("Press Enter to test Epistemic-Pragmatism (baseline)...")
    baseline_summary = test_model(
        "Epistemic-Pragmatism",
        baseline_path,
        questions,
        base_model=None
    )

    print(f"\n\n{'='*80}\n")
    input("Press Enter to test Introspective-Qwen Merged...")
    merged_summary = test_model(
        "Introspective-Qwen Merged",
        merged_path,
        questions,
        base_model=None
    )

    # Compare results
    compare_models(baseline_summary, merged_summary)

    print("✅ Session 8 quality validation complete!")
    print(f"\nNext steps:")
    print(f"  1. If quality retained → Long-duration testing (30+ min)")
    print(f"  2. If quality variance → Investigate merge process")
    print(f"  3. Production integration test with real SAGE loop")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
