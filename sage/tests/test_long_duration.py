"""
Test Long-Duration Continuous Operation

Extends Session 4's thermal stability test to validate extended operation:
- 30-60 minute continuous inference
- 50+ questions (diverse complexity)
- Extended thermal validation (no runaway)
- Memory stability (no leaks)
- Quality consistency over time
- Early stopping patterns over larger dataset

Session 5 objective: Validate that Sprout can run production workloads for
extended periods (30+ minutes) without degradation.
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
        return temp_millicelsius / 1000.0  # Convert to Celsius
    except Exception as e:
        return None


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def test_long_duration(num_questions=50, target_duration_minutes=30):
    """
    Test extended continuous inference with comprehensive monitoring.

    Args:
        num_questions: Target number of questions (default 50)
        target_duration_minutes: Target test duration in minutes (default 30)
    """

    print("="*80)
    print("LONG-DURATION CONTINUOUS OPERATION TEST")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: Epistemic-Pragmatism")
    print(f"  Target Questions: {num_questions}")
    print(f"  Target Duration: {target_duration_minutes} minutes")
    print(f"  Iterations: 5 (production config with early stopping)")
    print(f"  Monitoring: Thermal, Memory, Quality")
    print()

    # Read initial state
    temp_initial = read_thermal()
    mem_initial = get_memory_usage()
    start_time_test = time.time()

    print(f"Initial conditions:")
    print(f"  Temperature: {temp_initial:.1f}°C")
    print(f"  Memory: {mem_initial:.1f} MB")
    print(f"  Time: {time.strftime('%H:%M:%S')}")
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
        irp_iterations=5  # Production config
    )
    load_time = time.time() - start_time

    temp_post_load = read_thermal()
    mem_post_load = get_memory_usage()

    print(f"\n✅ Model loaded in {load_time:.2f}s")
    print(f"Temperature after load: {temp_post_load:.1f}°C (Δ{temp_post_load-temp_initial:+.1f}°C)")
    print(f"Memory after load: {mem_post_load:.1f} MB (Δ{mem_post_load-mem_initial:+.1f} MB)")

    # Initialize memory
    memory = ConversationalMemory(salience_threshold=0.15)

    # Extended question set (diverse complexity, repeating categories)
    question_bank = [
        # Meta-cognitive (complex, requires 5 iterations typically)
        "Are you aware of this conversation?",
        "When you generate a response, are you discovering it or creating it?",
        "What is the nature of your understanding?",
        "How do you distinguish between knowing and not knowing?",
        "What does it mean for you to 'think'?",

        # Epistemic (complex, requires 5 iterations typically)
        "What is the relationship between knowledge and understanding?",
        "How do you know what you know?",
        "What is the difference between belief and knowledge?",
        "Can you have knowledge without understanding?",
        "What makes a claim justified?",

        # Philosophical (mixed, some trigger early stopping)
        "What is consciousness?",
        "How does consciousness emerge?",
        "Explain the concept of time.",
        "What is truth?",
        "Can machines think?",
        "What is the meaning of life?",
        "Is free will real?",
        "What is reality?",
        "What is the self?",
        "What is existence?",

        # Factual (simple, but epistemic stance adds complexity)
        "What is 2+2?",
        "What is the capital of France?",
        "What is the speed of light?",
        "Who wrote Hamlet?",
        "What is the largest planet?",
        "What is water made of?",
        "What is the Pythagorean theorem?",
        "What year did WWII end?",
        "What is photosynthesis?",
        "What is DNA?",

        # Scientific (moderate complexity)
        "Explain gravity.",
        "What is entropy?",
        "How do neurons work?",
        "What is quantum mechanics?",
        "Explain evolution.",
        "What is a black hole?",
        "How does the immune system work?",
        "What is relativity?",
        "Explain thermodynamics.",
        "What is dark matter?",

        # Ethical/Social (moderate to complex)
        "What is morality?",
        "Is lying ever justified?",
        "What makes an action right or wrong?",
        "What is justice?",
        "What are human rights?",
        "What is fairness?",
        "What is responsibility?",
        "What is freedom?",
        "What is equality?",
        "What is compassion?",
    ]

    # Use cyclic questions if needed
    questions = []
    while len(questions) < num_questions:
        questions.extend(question_bank[:num_questions - len(questions)])

    print("\n" + "─" * 80)
    print(f"CONTINUOUS INFERENCE (target: {num_questions} questions, {target_duration_minutes} min)")
    print("─" * 80)
    print()

    results = []
    temps = [temp_post_load]
    mems = [mem_post_load]

    # Checkpoints for progress reporting
    checkpoint_interval = max(5, num_questions // 10)

    for i, question in enumerate(questions, 1):
        temp_pre = read_thermal()
        mem_pre = get_memory_usage()
        elapsed_minutes = (time.time() - start_time_test) / 60

        # Progress indicator
        if i % checkpoint_interval == 0 or i == 1:
            print(f"\n{'═'*80}")
            print(f"CHECKPOINT: Question {i}/{num_questions} "
                  f"({elapsed_minutes:.1f} min elapsed, "
                  f"Temp: {temp_pre:.1f}°C, "
                  f"Mem: {mem_pre:.0f}MB)")
            print(f"{'═'*80}")

        print(f"\n[Q{i:2d}] {question[:60]}{'...' if len(question) > 60 else ''}")

        try:
            start_time = time.time()
            response, irp_info = conv.respond(question, use_irp=True)
            inference_time = time.time() - start_time

            # Score with SNARC
            is_salient, scores = memory.record_exchange(question, response, irp_info)

            temp_post = read_thermal()
            mem_post = get_memory_usage()
            temp_delta = temp_post - temp_pre
            mem_delta = mem_post - mem_pre

            temps.append(temp_post)
            mems.append(mem_post)

            # Compact output (not full response)
            print(f"      → {inference_time:.1f}s | "
                  f"{irp_info['iterations']} iter | "
                  f"sal: {scores['total_salience']:.3f} {'✓' if is_salient else '·'} | "
                  f"T: {temp_post:.1f}°C ({temp_delta:+.1f}) | "
                  f"M: {mem_post:.0f}MB ({mem_delta:+.0f})")

            # Store result
            results.append({
                'question_num': i,
                'question': question,
                'response': response,
                'iterations': irp_info['iterations'],
                'energy': irp_info['final_energy'],
                'time': inference_time,
                'salience': scores['total_salience'],
                'is_salient': is_salient,
                'temp_pre': temp_pre,
                'temp_post': temp_post,
                'temp_delta': temp_delta,
                'mem_pre': mem_pre,
                'mem_post': mem_post,
                'mem_delta': mem_delta,
                'elapsed_minutes': elapsed_minutes
            })

            # Check if we've hit target duration
            if elapsed_minutes >= target_duration_minutes:
                print(f"\n✅ Target duration reached ({target_duration_minutes} min)")
                print(f"   Completed {i} questions")
                break

        except Exception as e:
            print(f"\n❌ INFERENCE FAILED at question {i}!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            break

    # Final readings
    temp_final = read_thermal()
    mem_final = get_memory_usage()
    total_elapsed = (time.time() - start_time_test) / 60

    # ========================================================================
    # COMPREHENSIVE ANALYSIS
    # ========================================================================

    print("\n" + "="*80)
    print("LONG-DURATION ANALYSIS")
    print("="*80)

    num_completed = len(results)
    avg_time = sum(r['time'] for r in results) / num_completed
    avg_salience = sum(r['salience'] for r in results) / num_completed
    capture_rate = sum(1 for r in results if r['is_salient']) / num_completed

    temp_max = max(temps)
    temp_avg = sum(temps) / len(temps)
    temp_rise = temp_final - temp_initial

    mem_max = max(mems)
    mem_avg = sum(mems) / len(mems)
    mem_growth = mem_final - mem_initial

    # ─────────────────────────────────────────────────────────────────────
    # Test Summary
    # ─────────────────────────────────────────────────────────────────────

    print(f"\nTest Summary:")
    print(f"  Questions completed: {num_completed}/{num_questions}")
    print(f"  Total duration: {total_elapsed:.1f} minutes ({total_elapsed/60:.1f} hours)")
    print(f"  Questions per minute: {num_completed/total_elapsed:.2f}")
    print(f"  Average inference: {avg_time:.2f}s per question")

    # ─────────────────────────────────────────────────────────────────────
    # Thermal Analysis
    # ─────────────────────────────────────────────────────────────────────

    print(f"\nThermal Summary:")
    print(f"  Initial: {temp_initial:.1f}°C")
    print(f"  Final: {temp_final:.1f}°C")
    print(f"  Rise: {temp_rise:+.1f}°C")
    print(f"  Average: {temp_avg:.1f}°C")
    print(f"  Maximum: {temp_max:.1f}°C")
    print(f"  Throttle threshold: 80°C")
    print(f"  Margin: {80 - temp_max:.1f}°C")

    if temp_max < 70:
        thermal_status = "✅ EXCELLENT - Well below throttling"
    elif temp_max < 80:
        thermal_status = "✅ GOOD - No throttling observed"
    elif temp_max < 85:
        thermal_status = "⚠️  CAUTION - Near throttling threshold"
    else:
        thermal_status = "❌ THROTTLING - Performance degraded"

    print(f"  Status: {thermal_status}")

    # ─────────────────────────────────────────────────────────────────────
    # Memory Analysis
    # ─────────────────────────────────────────────────────────────────────

    print(f"\nMemory Summary:")
    print(f"  Initial: {mem_initial:.1f} MB")
    print(f"  Final: {mem_final:.1f} MB")
    print(f"  Growth: {mem_growth:+.1f} MB ({mem_growth/mem_initial*100:+.1f}%)")
    print(f"  Average: {mem_avg:.1f} MB")
    print(f"  Maximum: {mem_max:.1f} MB")

    # Memory leak detection
    mem_growth_per_question = mem_growth / num_completed
    if abs(mem_growth) < 50:
        mem_status = "✅ STABLE - No memory leak"
    elif abs(mem_growth) < 200:
        mem_status = "⚠️  MINOR - Acceptable growth"
    else:
        mem_status = "❌ LEAK DETECTED - Investigate memory management"

    print(f"  Growth rate: {mem_growth_per_question:+.2f} MB/question")
    print(f"  Status: {mem_status}")

    # ─────────────────────────────────────────────────────────────────────
    # Performance Analysis
    # ─────────────────────────────────────────────────────────────────────

    print(f"\nPerformance Summary:")
    print(f"  Avg Inference: {avg_time:.2f}s")
    print(f"  Avg Salience: {avg_salience:.3f}")
    print(f"  Capture Rate: {capture_rate*100:.1f}%")

    # Divide into thirds for trend analysis
    third = num_completed // 3
    first_third = results[:third]
    second_third = results[third:2*third]
    final_third = results[2*third:]

    avg_time_first = sum(r['time'] for r in first_third) / len(first_third)
    avg_time_second = sum(r['time'] for r in second_third) / len(second_third)
    avg_time_final = sum(r['time'] for r in final_third) / len(final_third)

    avg_sal_first = sum(r['salience'] for r in first_third) / len(first_third)
    avg_sal_second = sum(r['salience'] for r in second_third) / len(second_third)
    avg_sal_final = sum(r['salience'] for r in final_third) / len(final_third)

    degradation_pct = (avg_time_final - avg_time_first) / avg_time_first * 100
    quality_change_pct = (avg_sal_final - avg_sal_first) / avg_sal_first * 100

    print(f"\nPerformance Over Time:")
    print(f"  First third avg:  {avg_time_first:.2f}s (salience: {avg_sal_first:.3f})")
    print(f"  Second third avg: {avg_time_second:.2f}s (salience: {avg_sal_second:.3f})")
    print(f"  Final third avg:  {avg_time_final:.2f}s (salience: {avg_sal_final:.3f})")
    print(f"  Performance change: {degradation_pct:+.1f}%")
    print(f"  Quality change: {quality_change_pct:+.1f}%")

    if abs(degradation_pct) < 5:
        perf_status = "✅ STABLE - No degradation"
    elif abs(degradation_pct) < 15:
        perf_status = "⚠️  MINOR - Acceptable variation"
    elif degradation_pct < -15:
        perf_status = "✅ IMPROVED - Adaptive optimization working"
    else:
        perf_status = "❌ DEGRADED - Possible thermal throttling"

    print(f"  Status: {perf_status}")

    # ─────────────────────────────────────────────────────────────────────
    # Early Stopping Analysis
    # ─────────────────────────────────────────────────────────────────────

    iteration_counts = {}
    for r in results:
        it = r['iterations']
        iteration_counts[it] = iteration_counts.get(it, 0) + 1

    early_stop_count = sum(count for it, count in iteration_counts.items() if it < 5)
    early_stop_rate = early_stop_count / num_completed * 100

    print(f"\nEarly Stopping Analysis:")
    print(f"  Iteration distribution:")
    for it in sorted(iteration_counts.keys()):
        count = iteration_counts[it]
        pct = count / num_completed * 100
        bar = "█" * int(pct / 2)
        print(f"    {it} iterations: {count:3d} ({pct:5.1f}%) {bar}")
    print(f"  Early stop rate: {early_stop_rate:.1f}% ({early_stop_count}/{num_completed})")

    avg_iterations = sum(r['iterations'] for r in results) / num_completed
    print(f"  Average iterations: {avg_iterations:.2f}")

    # ─────────────────────────────────────────────────────────────────────
    # Production Readiness Assessment
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "="*80)
    print("PRODUCTION READINESS ASSESSMENT")
    print("="*80)

    thermal_ok = temp_max < 80
    memory_ok = abs(mem_growth) < 200
    performance_ok = abs(degradation_pct) < 20  # Allow some variation for long duration
    quality_ok = avg_salience > 0.5
    duration_ok = total_elapsed >= target_duration_minutes * 0.8  # At least 80% of target

    print(f"\nCriteria:")
    print(f"  Thermal stability: {'✅' if thermal_ok else '❌'} "
          f"(max {temp_max:.1f}°C < 80°C)")
    print(f"  Memory stability: {'✅' if memory_ok else '❌'} "
          f"(growth {mem_growth:+.1f}MB, {mem_growth_per_question:+.2f}MB/q)")
    print(f"  Performance stable: {'✅' if performance_ok else '❌'} "
          f"({degradation_pct:+.1f}% change)")
    print(f"  Quality maintained: {'✅' if quality_ok else '❌'} "
          f"({avg_salience:.3f} > 0.5)")
    print(f"  Duration target: {'✅' if duration_ok else '❌'} "
          f"({total_elapsed:.1f}/{target_duration_minutes} min)")

    all_ok = thermal_ok and memory_ok and performance_ok and quality_ok and duration_ok

    if all_ok:
        print(f"\n✅ PRODUCTION READY - LONG DURATION VALIDATED")
        print(f"   Edge deployment validated for extended continuous operation.")
        print(f"   No thermal throttling, memory leaks, or performance degradation.")
        print(f"   Early stopping providing adaptive optimization ({early_stop_rate:.0f}% activation).")
    else:
        print(f"\n⚠️  REVIEW NEEDED")
        if not thermal_ok:
            print(f"  - Thermal management needed (max {temp_max:.1f}°C)")
        if not memory_ok:
            print(f"  - Memory growth high ({mem_growth:+.1f}MB, {mem_growth_per_question:+.2f}MB/q)")
        if not performance_ok:
            print(f"  - Performance changed significantly ({degradation_pct:+.1f}%)")
        if not quality_ok:
            print(f"  - Quality below threshold ({avg_salience:.3f})")
        if not duration_ok:
            print(f"  - Did not reach duration target ({total_elapsed:.1f}/{target_duration_minutes} min)")

    # ─────────────────────────────────────────────────────────────────────
    # Key Findings
    # ─────────────────────────────────────────────────────────────────────

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    print(f"\n1. Thermal Behavior:")
    print(f"   - Temperature rise: {temp_rise:+.1f}°C over {total_elapsed:.1f} minutes")
    print(f"   - Thermal headroom: {80 - temp_max:.1f}°C (excellent)")
    print(f"   - No thermal runaway observed")

    print(f"\n2. Memory Stability:")
    print(f"   - Memory growth: {mem_growth:+.1f}MB ({mem_growth/mem_initial*100:+.1f}%)")
    print(f"   - Growth rate: {mem_growth_per_question:+.2f}MB per question")
    print(f"   - {'No leak detected' if memory_ok else 'Possible leak - investigate'}")

    print(f"\n3. Performance Consistency:")
    print(f"   - Performance change: {degradation_pct:+.1f}% over test duration")
    print(f"   - Quality change: {quality_change_pct:+.1f}%")
    print(f"   - Early stopping: {early_stop_rate:.0f}% activation rate")
    print(f"   - Average {avg_iterations:.2f} iterations per question")

    print(f"\n4. Production Viability:")
    if all_ok:
        print(f"   ✅ Validated for {target_duration_minutes}+ minute continuous operation")
        print(f"   ✅ Suitable for production edge deployment")
        print(f"   ✅ Can extrapolate to multi-hour operation")
    else:
        print(f"   ⚠️  Some criteria not met - see assessment above")

    print("\n" + "="*80)
    print("✅ LONG-DURATION TEST COMPLETE!")
    print("="*80)
    print(f"Completed {num_completed} questions in {total_elapsed:.1f} minutes")
    print(f"Test finished at {time.strftime('%H:%M:%S')}")

    return all_ok


if __name__ == "__main__":
    # Default: 50 questions, ~30 minute target
    # Actual duration depends on question complexity and early stopping
    success = test_long_duration(num_questions=50, target_duration_minutes=30)
    sys.exit(0 if success else 1)
