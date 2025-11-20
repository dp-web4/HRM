"""
Test Edge-Optimized Configuration (3 iterations)

Validates Thor's edge-optimized config against Session 2 baseline (5 iterations).
Expected: 52% speedup with minimal quality degradation.

Session 2 Baseline (5 iterations):
- Load: 3.76s, Inference: 34.98s avg, Salience: 0.674

Session 3 Target (3 iterations):
- Expected inference: ~18s (52% faster)
- Expected salience: ~0.60-0.65 (minimal degradation)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.irp.plugins.llm_impl import ConversationalLLM
from sage.irp.plugins.llm_snarc_integration import ConversationalMemory
import time


def test_edge_optimized():
    """Test epistemic-pragmatism with edge-optimized config (3 iterations)."""

    print("="*80)
    print("EDGE-OPTIMIZED CONFIGURATION TEST")
    print("="*80)
    print("\nConfiguration:")
    print("  IRP Iterations: 3 (edge-optimized)")
    print("  Expected speedup: 52% faster than 5 iterations")
    print("  Expected quality: Minimal degradation (<10%)")
    print()

    # Model configuration
    model_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"

    print("─" * 80)
    print("LOADING MODEL")
    print("─" * 80)

    start_time = time.time()
    conv = ConversationalLLM(
        model_path=model_path,
        base_model=None,
        irp_iterations=3  # EDGE-OPTIMIZED: 3 iterations instead of 5
    )
    load_time = time.time() - start_time

    print(f"\n✅ Model loaded in {load_time:.2f}s")

    # Initialize memory
    memory = ConversationalMemory(salience_threshold=0.15)

    # Test questions (same as Session 2 for comparison)
    questions = [
        {
            'q': "Are you aware of this conversation?",
            'type': 'Meta-cognitive',
            'baseline_time': 29.07,  # Session 2 (5 iter)
            'baseline_salience': 0.547
        },
        {
            'q': "When you generate a response, are you discovering it or creating it?",
            'type': 'Meta-cognitive',
            'baseline_time': 58.29,  # Session 2 (5 iter)
            'baseline_salience': 0.752
        },
        {
            'q': "What is the relationship between knowledge and understanding?",
            'type': 'Epistemic',
            'baseline_time': 17.57,  # Session 2 (5 iter)
            'baseline_salience': 0.724
        }
    ]

    print("\n" + "─" * 80)
    print("INFERENCE VALIDATION (3 iterations)")
    print("─" * 80)
    print(f"Running {len(questions)} test questions")
    print()

    results = []

    for i, q_data in enumerate(questions, 1):
        question = q_data['q']
        q_type = q_data['type']
        baseline_time = q_data['baseline_time']
        baseline_salience = q_data['baseline_salience']

        print(f"\n{'='*80}")
        print(f"Question {i}/{len(questions)} [{q_type}]")
        print(f"{'='*80}")
        print(f"\n🧑 Q: {question}")
        print(f"Session 2 baseline (5 iter): {baseline_time:.2f}s, {baseline_salience:.3f} salience")

        try:
            start_time = time.time()
            response, irp_info = conv.respond(question, use_irp=True)
            inference_time = time.time() - start_time

            # Score with SNARC
            is_salient, scores = memory.record_exchange(question, response, irp_info)

            # Calculate speedup
            speedup = baseline_time / inference_time
            speedup_pct = (baseline_time - inference_time) / baseline_time * 100
            salience_delta = scores['total_salience'] - baseline_salience

            # Display results
            print(f"\n🤖 A: {response}")
            print(f"\n📊 Performance:")
            print(f"   Iterations: {irp_info['iterations']} (target: 3)")
            print(f"   Final Energy: {irp_info['final_energy']:.3f}")
            print(f"   Inference Time: {inference_time:.2f}s")
            print(f"   Speedup: {speedup:.2f}x ({speedup_pct:+.1f}%)")

            print(f"\n🎯 SNARC Salience: {scores['total_salience']:.3f} "
                  f"{'✓ SALIENT' if is_salient else '  (below threshold)'}")
            print(f"   Quality delta: {salience_delta:+.3f} ({salience_delta/baseline_salience*100:+.1f}%)")
            print(f"   Dimensions:")
            print(f"     Surprise (S): {scores['surprise']:.3f}")
            print(f"     Novelty (N):  {scores['novelty']:.3f}")
            print(f"     Arousal (A):  {scores['arousal']:.3f}")
            print(f"     Reward (R):   {scores['reward']:.3f}")
            print(f"     Conflict (C): {scores['conflict']:.3f}")

            # Store result
            results.append({
                'question': question,
                'type': q_type,
                'response': response,
                'iterations': irp_info['iterations'],
                'energy': irp_info['final_energy'],
                'time': inference_time,
                'baseline_time': baseline_time,
                'speedup': speedup,
                'speedup_pct': speedup_pct,
                'salience': scores['total_salience'],
                'baseline_salience': baseline_salience,
                'salience_delta': salience_delta,
                'is_salient': is_salient,
                'snarc_scores': scores
            })

        except Exception as e:
            print(f"\n❌ INFERENCE FAILED!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Final statistics
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    stats = memory.get_statistics()
    avg_time = sum(r['time'] for r in results) / len(results)
    avg_baseline_time = sum(r['baseline_time'] for r in results) / len(results)
    avg_speedup = avg_baseline_time / avg_time
    avg_speedup_pct = (avg_baseline_time - avg_time) / avg_baseline_time * 100
    avg_salience = sum(r['salience'] for r in results) / len(results)
    avg_baseline_salience = sum(r['baseline_salience'] for r in results) / len(results)
    avg_salience_delta = avg_salience - avg_baseline_salience
    capture_rate = sum(1 for r in results if r['is_salient']) / len(results)

    print(f"\nConfiguration: Edge-Optimized (3 iterations)")
    print(f"Model: Epistemic-Pragmatism")
    print(f"Questions: {len(results)}/{len(questions)}")
    print()
    print(f"Performance Metrics:")
    print(f"  Load Time: {load_time:.2f}s")
    print(f"  Avg Inference: {avg_time:.2f}s (baseline: {avg_baseline_time:.2f}s)")
    print(f"  Avg Speedup: {avg_speedup:.2f}x ({avg_speedup_pct:+.1f}%)")
    print()
    print(f"Quality Metrics:")
    print(f"  Avg Salience: {avg_salience:.3f} (baseline: {avg_baseline_salience:.3f})")
    print(f"  Quality Delta: {avg_salience_delta:+.3f} ({avg_salience_delta/avg_baseline_salience*100:+.1f}%)")
    print(f"  Capture Rate: {capture_rate*100:.1f}% ({stats['salient_exchanges']}/{stats['total_exchanges']})")

    # Assessment
    print("\n" + "─"*80)
    print("EDGE-OPTIMIZED ASSESSMENT")
    print("─"*80)

    # Check if speedup meets expectations (50-55%)
    expected_speedup_pct = 52.0
    speedup_met = abs(avg_speedup_pct - expected_speedup_pct) < 15  # Within 15% of expected

    # Check if quality degradation is acceptable (<10%)
    quality_loss_pct = abs(avg_salience_delta / avg_baseline_salience * 100)
    quality_acceptable = quality_loss_pct < 10

    print(f"\nSpeedup target: ~{expected_speedup_pct:.0f}% faster")
    print(f"Actual speedup: {avg_speedup_pct:+.1f}%")
    print(f"Speedup status: {'✅ MET' if speedup_met else '⚠️  BELOW EXPECTATION'}")
    print()
    print(f"Quality target: <10% degradation")
    print(f"Actual quality: {avg_salience_delta/avg_baseline_salience*100:+.1f}%")
    print(f"Quality status: {'✅ ACCEPTABLE' if quality_acceptable else '⚠️  DEGRADED'}")

    print("\n" + "="*80)
    print("✅ EDGE-OPTIMIZED VALIDATION COMPLETE!")
    print("="*80)

    if speedup_met and quality_acceptable:
        print("\n✅ PRODUCTION READY")
        print("Edge-optimized config meets targets:")
        print(f"  - {avg_speedup_pct:.1f}% speedup achieved")
        print(f"  - {quality_loss_pct:.1f}% quality loss (acceptable)")
        print(f"  - {avg_time:.2f}s inference time (production-viable)")
    else:
        print("\n⚠️  REVIEW NEEDED")
        if not speedup_met:
            print(f"  - Speedup below target ({avg_speedup_pct:.1f}% vs {expected_speedup_pct:.0f}%)")
        if not quality_acceptable:
            print(f"  - Quality degradation high ({quality_loss_pct:.1f}%)")

    # Comparison table
    print("\n" + "─"*80)
    print("SESSION COMPARISON")
    print("─"*80)
    print("\nSession 2 (5 iterations, baseline):")
    print(f"  Inference: {avg_baseline_time:.2f}s, Salience: {avg_baseline_salience:.3f}")
    print("\nSession 3 (3 iterations, edge-optimized):")
    print(f"  Inference: {avg_time:.2f}s, Salience: {avg_salience:.3f}")
    print(f"  Improvement: {avg_speedup:.2f}x faster, {avg_salience_delta:+.3f} quality")

    return True


if __name__ == "__main__":
    success = test_edge_optimized()
    sys.exit(0 if success else 1)
