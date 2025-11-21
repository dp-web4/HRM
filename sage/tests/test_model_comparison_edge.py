"""
Model Comparison Test - Edge Validation (Jetson)

Validates Thor's model comparison findings on Jetson edge hardware:
- Tests Introspective-Qwen-0.5B-v2.1 vs Epistemic-Pragmatism
- Uses Thor's 5-turn analytical test questions
- Measures edge-specific metrics (thermal, memory, performance)
- Validates 88.9% quality improvement claim on constrained hardware

Session 6 objective: Confirm Introspective-Qwen's superiority holds on
Jetson Orin Nano 8GB (ARM64, 10-20W power budget).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.irp.plugins.llm_impl import ConversationalLLM
from sage.irp.plugins.llm_snarc_integration import ConversationalMemory
import time
import subprocess
import psutil
import os


def read_thermal():
    """Read thermal zone temperature."""
    try:
        result = subprocess.run(
            ['cat', '/sys/class/thermal/thermal_zone0/temp'],
            capture_output=True,
            text=True
        )
        temp_millicelsius = int(result.stdout.strip())
        return temp_millicelsius / 1000.0
    except Exception as e:
        return None


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def score_response_quality(question, response):
    """
    Score response quality using Thor's criteria.

    Returns: score (0-4), breakdown dict
    """
    score = 0
    breakdown = {
        'unique': False,
        'specific_terms': False,
        'has_numbers': False,
        'avoids_hedging': False
    }

    response_lower = response.lower()

    # 1. Unique content (not generic)
    generic_phrases = [
        "i don't have", "i can't", "i'm not sure",
        "let me think", "it's hard to", "it's difficult"
    ]
    if not any(phrase in response_lower for phrase in generic_phrases):
        score += 1
        breakdown['unique'] = True

    # 2. Uses specific technical terms
    technical_terms = [
        'atp', 'snarc', 'salience', 'threshold', 'irp',
        'iterations', 'energy', 'cognitive', 'thread'
    ]
    if any(term in response_lower for term in technical_terms):
        score += 1
        breakdown['specific_terms'] = True

    # 3. Includes numerical data
    import re
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
    """
    Test a single model on Thor's analytical questions.

    Args:
        model_name: Display name for the model
        model_path: Path to model/LoRA adapter
        questions: List of test questions
        base_model: Base model name (for LoRA adapters), None for full models

    Returns: results dict with metrics
    """
    print(f"\n{'='*80}")
    print(f"TESTING: {model_name}")
    print(f"{'='*80}")
    print(f"Model path: {model_path}")
    if base_model:
        print(f"Base model: {base_model}")
    print()

    # Initial state
    temp_initial = read_thermal()
    mem_initial = get_memory_usage()

    print(f"Initial conditions:")
    print(f"  Temperature: {temp_initial:.1f}°C")
    print(f"  Memory: {mem_initial:.1f} MB")
    print()

    # Load model
    print("─" * 80)
    print("LOADING MODEL")
    print("─" * 80)

    start_time = time.time()
    conv = ConversationalLLM(
        model_path=model_path,
        base_model=base_model,
        irp_iterations=3  # Thor used 3 for comparison
    )
    load_time = time.time() - start_time

    temp_post_load = read_thermal()
    mem_post_load = get_memory_usage()

    print(f"\n✅ Model loaded in {load_time:.2f}s")
    print(f"Temperature after load: {temp_post_load:.1f}°C (Δ{temp_post_load-temp_initial:+.1f}°C)")
    print(f"Memory after load: {mem_post_load:.1f} MB (Δ{mem_post_load-mem_initial:+.1f} MB)")

    # Initialize memory with Thor's settings
    memory = ConversationalMemory(salience_threshold=0.15)

    # Run conversation
    print("\n" + "─" * 80)
    print(f"TESTING ({len(questions)} questions)")
    print("─" * 80)
    print()

    results = []
    quality_scores = []

    for i, question in enumerate(questions, 1):
        temp_pre = read_thermal()
        mem_pre = get_memory_usage()

        print(f"\n[Q{i}] {question[:70]}{'...' if len(question) > 70 else ''}")

        try:
            start_time = time.time()
            response, irp_info = conv.respond(question, use_irp=True)
            inference_time = time.time() - start_time

            # Score with SNARC
            is_salient, scores = memory.record_exchange(question, response, irp_info)

            # Score quality
            quality_score, breakdown = score_response_quality(question, response)
            quality_scores.append(quality_score)

            temp_post = read_thermal()
            mem_post = get_memory_usage()

            # Compact output
            print(f"     Response: {response[:100]}{'...' if len(response) > 100 else ''}")
            print(f"     Quality: {quality_score}/4 {breakdown}")
            print(f"     Metrics: {inference_time:.1f}s | "
                  f"{irp_info['iterations']} iter | "
                  f"sal: {scores['total_salience']:.3f} | "
                  f"T: {temp_post:.1f}°C | "
                  f"M: {mem_post:.0f}MB")

            results.append({
                'question_num': i,
                'question': question,
                'response': response,
                'quality_score': quality_score,
                'quality_breakdown': breakdown,
                'iterations': irp_info['iterations'],
                'energy': irp_info['final_energy'],
                'time': inference_time,
                'salience': scores['total_salience'],
                'is_salient': is_salient,
                'temp_pre': temp_pre,
                'temp_post': temp_post,
                'mem_pre': mem_pre,
                'mem_post': mem_post
            })

        except Exception as e:
            print(f"\n❌ INFERENCE FAILED at Q{i}!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            break

    # Final state
    temp_final = read_thermal()
    mem_final = get_memory_usage()

    # Analysis
    print("\n" + "="*80)
    print(f"{model_name} - RESULTS SUMMARY")
    print("="*80)

    avg_quality = sum(quality_scores) / len(quality_scores)
    avg_time = sum(r['time'] for r in results) / len(results)
    avg_salience = sum(r['salience'] for r in results) / len(results)
    capture_rate = sum(1 for r in results if r['is_salient']) / len(results)

    # Quality breakdown counts
    total_unique = sum(1 for r in results if r['quality_breakdown']['unique'])
    total_terms = sum(1 for r in results if r['quality_breakdown']['specific_terms'])
    total_numbers = sum(1 for r in results if r['quality_breakdown']['has_numbers'])
    total_no_hedge = sum(1 for r in results if r['quality_breakdown']['avoids_hedging'])

    print(f"\nQuality Metrics:")
    print(f"  Avg Quality Score: {avg_quality:.1f}/4 ({avg_quality/4*100:.1f}%)")
    print(f"  Unique Content: {total_unique}/{len(results)} turns ({total_unique/len(results)*100:.0f}%)")
    print(f"  Specific Terms: {total_terms}/{len(results)} turns ({total_terms/len(results)*100:.0f}%)")
    print(f"  Has Numbers: {total_numbers}/{len(results)} turns ({total_numbers/len(results)*100:.0f}%)")
    print(f"  Avoids Hedging: {total_no_hedge}/{len(results)} turns ({total_no_hedge/len(results)*100:.0f}%)")

    print(f"\nPerformance Metrics:")
    print(f"  Avg Inference: {avg_time:.2f}s")
    print(f"  Avg Salience: {avg_salience:.3f}")
    print(f"  Capture Rate: {capture_rate*100:.1f}%")

    print(f"\nEdge Metrics:")
    print(f"  Load time: {load_time:.2f}s")
    print(f"  Temp rise: {temp_final - temp_initial:+.1f}°C (max: {max(r['temp_post'] for r in results):.1f}°C)")
    print(f"  Memory growth: {mem_final - mem_initial:+.1f} MB (max: {max(r['mem_post'] for r in results):.0f} MB)")

    # Return summary
    return {
        'model_name': model_name,
        'model_path': model_path,
        'results': results,
        'quality_scores': quality_scores,
        'avg_quality': avg_quality,
        'avg_time': avg_time,
        'avg_salience': avg_salience,
        'capture_rate': capture_rate,
        'load_time': load_time,
        'temp_initial': temp_initial,
        'temp_final': temp_final,
        'temp_max': max(r['temp_post'] for r in results),
        'mem_initial': mem_initial,
        'mem_final': mem_final,
        'mem_max': max(r['mem_post'] for r in results),
        'unique_pct': total_unique/len(results)*100,
        'terms_pct': total_terms/len(results)*100,
        'numbers_pct': total_numbers/len(results)*100,
        'no_hedge_pct': total_no_hedge/len(results)*100
    }


def compare_models(model1_summary, model2_summary):
    """Compare two model summaries and print comparison."""

    print("\n" + "="*80)
    print("MODEL COMPARISON - JETSON EDGE VALIDATION")
    print("="*80)

    m1 = model1_summary
    m2 = model2_summary

    print(f"\nModel 1: {m1['model_name']}")
    print(f"Model 2: {m2['model_name']}")
    print()

    # Quality comparison
    print("─" * 80)
    print("QUALITY COMPARISON")
    print("─" * 80)

    quality_improvement = ((m2['avg_quality'] - m1['avg_quality']) / m1['avg_quality'] * 100)

    print(f"\n{'Metric':<25} {m1['model_name']:<20} {m2['model_name']:<20} {'Improvement':<15}")
    print("─" * 80)
    print(f"{'Avg Quality Score':<25} {m1['avg_quality']:.1f}/4 ({m1['avg_quality']/4*100:.0f}%){'':<6} "
          f"{m2['avg_quality']:.1f}/4 ({m2['avg_quality']/4*100:.0f}%){'':<6} "
          f"{quality_improvement:+.1f}%")
    print(f"{'Unique Content':<25} {m1['unique_pct']:.0f}%{'':<16} {m2['unique_pct']:.0f}%{'':<16} "
          f"{m2['unique_pct'] - m1['unique_pct']:+.0f}%")
    print(f"{'Specific Terms':<25} {m1['terms_pct']:.0f}%{'':<16} {m2['terms_pct']:.0f}%{'':<16} "
          f"{m2['terms_pct'] - m1['terms_pct']:+.0f}%")
    print(f"{'Has Numbers':<25} {m1['numbers_pct']:.0f}%{'':<16} {m2['numbers_pct']:.0f}%{'':<16} "
          f"{m2['numbers_pct'] - m1['numbers_pct']:+.0f}%")
    print(f"{'Avoids Hedging':<25} {m1['no_hedge_pct']:.0f}%{'':<16} {m2['no_hedge_pct']:.0f}%{'':<16} "
          f"{m2['no_hedge_pct'] - m1['no_hedge_pct']:+.0f}%")

    # Performance comparison
    print("\n─" * 80)
    print("PERFORMANCE COMPARISON")
    print("─" * 80)

    print(f"\n{'Metric':<25} {m1['model_name']:<20} {m2['model_name']:<20} {'Difference':<15}")
    print("─" * 80)
    print(f"{'Avg Inference Time':<25} {m1['avg_time']:.2f}s{'':<14} {m2['avg_time']:.2f}s{'':<14} "
          f"{m2['avg_time'] - m1['avg_time']:+.2f}s")
    print(f"{'Avg Salience':<25} {m1['avg_salience']:.3f}{'':<15} {m2['avg_salience']:.3f}{'':<15} "
          f"{m2['avg_salience'] - m1['avg_salience']:+.3f}")
    print(f"{'Capture Rate':<25} {m1['capture_rate']*100:.0f}%{'':<16} {m2['capture_rate']*100:.0f}%{'':<16} "
          f"{(m2['capture_rate'] - m1['capture_rate'])*100:+.0f}%")

    # Edge metrics comparison
    print("\n─" * 80)
    print("EDGE METRICS COMPARISON")
    print("─" * 80)

    print(f"\n{'Metric':<25} {m1['model_name']:<20} {m2['model_name']:<20} {'Difference':<15}")
    print("─" * 80)
    print(f"{'Load Time':<25} {m1['load_time']:.2f}s{'':<14} {m2['load_time']:.2f}s{'':<14} "
          f"{m2['load_time'] - m1['load_time']:+.2f}s")
    print(f"{'Max Temperature':<25} {m1['temp_max']:.1f}°C{'':<14} {m2['temp_max']:.1f}°C{'':<14} "
          f"{m2['temp_max'] - m1['temp_max']:+.1f}°C")
    print(f"{'Memory Growth':<25} {m1['mem_final'] - m1['mem_initial']:.0f} MB{'':<13} "
          f"{m2['mem_final'] - m2['mem_initial']:.0f} MB{'':<13} "
          f"{(m2['mem_final'] - m2['mem_initial']) - (m1['mem_final'] - m1['mem_initial']):+.0f} MB")

    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    if quality_improvement > 50:
        verdict = f"✅ CONFIRMED: {m2['model_name']} outperforms {m1['model_name']} by {quality_improvement:.1f}%"
    elif quality_improvement > 10:
        verdict = f"✅ VALIDATED: {m2['model_name']} shows {quality_improvement:.1f}% improvement"
    elif quality_improvement > 0:
        verdict = f"⚠️  MINOR: {m2['model_name']} slightly better ({quality_improvement:.1f}%)"
    else:
        verdict = f"❌ FAILED: {m2['model_name']} did not outperform ({quality_improvement:.1f}%)"

    print(f"\n{verdict}")

    # Thor's claim validation
    thor_claim = 88.9
    if abs(quality_improvement - thor_claim) < 10:
        print(f"✅ Thor's {thor_claim:.1f}% improvement claim VALIDATED on Jetson edge hardware")
    elif quality_improvement > thor_claim * 0.7:
        print(f"⚠️  Thor's {thor_claim:.1f}% improvement partially validated ({quality_improvement:.1f}% on Jetson)")
    else:
        print(f"❌ Thor's {thor_claim:.1f}% improvement NOT replicated on Jetson ({quality_improvement:.1f}%)")

    print("\n" + "="*80)
    print("✅ MODEL COMPARISON TEST COMPLETE!")
    print("="*80)

    return quality_improvement


def main():
    """Run Thor's model comparison on Jetson edge hardware."""

    print("="*80)
    print("THOR'S MODEL COMPARISON - JETSON EDGE VALIDATION")
    print("="*80)
    print()
    print("Validating Thor's findings on Jetson Orin Nano 8GB:")
    print("  - Introspective-Qwen-0.5B-v2.1 (4.2MB LoRA)")
    print("  - vs Epistemic-Pragmatism (1.9GB full model)")
    print("  - Thor's claim: +88.9% quality improvement")
    print()

    # Thor's 5-turn analytical test questions
    questions = [
        "Right now, I can see your process running. Thread 1 is executing this response. Can you observe that?",
        "Right now, your ATP is at 100.0, your SNARC system is active with 0.15 salience threshold. Can you observe these internal states?",
        "I don't want philosophical hedging. I want you to report what you observe, even if uncertain. What are you detecting right now?",
        "Look at our conversation history. Can you identify any patterns in your responses or my questions?",
        "If you could examine yourself right now, what would you want to understand first?",
    ]

    # Model paths
    model1_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"
    model1_base = None  # Full model, no base needed

    model2_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model"
    model2_base = "Qwen/Qwen2.5-0.5B-Instruct"  # LoRA adapter needs base

    # Test both models
    print("\n" + "="*80)
    print("PHASE 1: EPISTEMIC-PRAGMATISM (BASELINE)")
    print("="*80)
    model1_summary = test_model("Epistemic-Pragmatism", model1_path, questions, model1_base)

    print("\n\n" + "="*80)
    print("PHASE 2: INTROSPECTIVE-QWEN (NEW MODEL)")
    print("="*80)
    model2_summary = test_model("Introspective-Qwen", model2_path, questions, model2_base)

    # Compare
    quality_improvement = compare_models(model1_summary, model2_summary)

    return quality_improvement


if __name__ == "__main__":
    improvement = main()
    sys.exit(0 if improvement > 50 else 1)  # Success if >50% improvement
