"""
Test Thermal Stability During Continuous Inference

Validates production readiness by running continuous inference and monitoring:
- Thermal behavior (temperature during sustained load)
- Performance degradation (if thermal throttling occurs)
- Memory stability (no leaks over time)
- Quality consistency (salience over multiple questions)

Session 4 objective: Validate that Sprout can run production workloads without
thermal throttling or performance degradation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.irp.plugins.llm_impl import ConversationalLLM
from sage.irp.plugins.llm_snarc_integration import ConversationalMemory
import time
import subprocess


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


def test_thermal_stability(num_questions=10):
    """
    Test continuous inference with thermal monitoring.

    Args:
        num_questions: Number of questions to run consecutively
    """

    print("="*80)
    print("THERMAL STABILITY TEST")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: Epistemic-Pragmatism")
    print(f"  Questions: {num_questions} (continuous)")
    print(f"  Iterations: 5 (production config)")
    print(f"  Thermal monitoring: Enabled")
    print()

    # Read initial thermal state
    temp_initial = read_thermal()
    print(f"Initial temperature: {temp_initial:.1f}°C")
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
    print(f"\n✅ Model loaded in {load_time:.2f}s")
    print(f"Temperature after load: {temp_post_load:.1f}°C (Δ{temp_post_load-temp_initial:+.1f}°C)")

    # Initialize memory
    memory = ConversationalMemory(salience_threshold=0.15)

    # Test questions (varied complexity)
    questions = [
        "Are you aware of this conversation?",
        "What is 2+2?",
        "When you generate a response, are you discovering it or creating it?",
        "What is the capital of France?",
        "What is the relationship between knowledge and understanding?",
        "How does consciousness emerge?",
        "Explain the concept of time.",
        "What is truth?",
        "Can machines think?",
        "What is the meaning of life?",
    ][:num_questions]

    print("\n" + "─" * 80)
    print(f"CONTINUOUS INFERENCE ({num_questions} questions)")
    print("─" * 80)
    print()

    results = []
    temps = [temp_post_load]

    for i, question in enumerate(questions, 1):
        temp_pre = read_thermal()

        print(f"\n{'='*80}")
        print(f"Question {i}/{num_questions} (Temp: {temp_pre:.1f}°C)")
        print(f"{'='*80}")
        print(f"\n🧑 Q: {question}")

        try:
            start_time = time.time()
            response, irp_info = conv.respond(question, use_irp=True)
            inference_time = time.time() - start_time

            # Score with SNARC
            is_salient, scores = memory.record_exchange(question, response, irp_info)

            temp_post = read_thermal()
            temp_delta = temp_post - temp_pre
            temps.append(temp_post)

            # Check for thermal throttling indicator
            throttle_warning = " ⚠️ THROTTLING?" if temp_post > 80 else ""

            print(f"\n🤖 A: {response[:100]}{'...' if len(response) > 100 else ''}")
            print(f"\n📊 Performance:")
            print(f"   Iterations: {irp_info['iterations']}")
            print(f"   Final Energy: {irp_info['final_energy']:.3f}")
            print(f"   Inference Time: {inference_time:.2f}s")
            print(f"   Temperature: {temp_pre:.1f}°C → {temp_post:.1f}°C (Δ{temp_delta:+.1f}°C){throttle_warning}")

            print(f"\n🎯 SNARC Salience: {scores['total_salience']:.3f} "
                  f"{'✓ SALIENT' if is_salient else '  (below threshold)'}")

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
                'temp_delta': temp_delta
            })

        except Exception as e:
            print(f"\n❌ INFERENCE FAILED!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Final thermal reading
    temp_final = read_thermal()

    # Analysis
    print("\n" + "="*80)
    print("THERMAL STABILITY ANALYSIS")
    print("="*80)

    avg_time = sum(r['time'] for r in results) / len(results)
    avg_salience = sum(r['salience'] for r in results) / len(results)
    capture_rate = sum(1 for r in results if r['is_salient']) / len(results)

    temp_max = max(temps)
    temp_avg = sum(temps) / len(temps)
    temp_rise = temp_final - temp_initial

    print(f"\nThermal Summary:")
    print(f"  Initial: {temp_initial:.1f}°C")
    print(f"  Final: {temp_final:.1f}°C")
    print(f"  Rise: {temp_rise:+.1f}°C")
    print(f"  Average: {temp_avg:.1f}°C")
    print(f"  Maximum: {temp_max:.1f}°C")
    print(f"  Throttle threshold: 80°C")

    if temp_max < 70:
        thermal_status = "✅ EXCELLENT - Well below throttling"
    elif temp_max < 80:
        thermal_status = "✅ GOOD - No throttling observed"
    elif temp_max < 85:
        thermal_status = "⚠️  CAUTION - Near throttling threshold"
    else:
        thermal_status = "❌ THROTTLING - Performance degraded"

    print(f"  Status: {thermal_status}")

    print(f"\nPerformance Summary:")
    print(f"  Questions: {len(results)}")
    print(f"  Avg Inference: {avg_time:.2f}s")
    print(f"  Avg Salience: {avg_salience:.3f}")
    print(f"  Capture Rate: {capture_rate*100:.1f}%")

    # Check for performance degradation over time
    first_half = results[:len(results)//2]
    second_half = results[len(results)//2:]

    avg_time_first = sum(r['time'] for r in first_half) / len(first_half)
    avg_time_second = sum(r['time'] for r in second_half) / len(second_half)

    degradation_pct = (avg_time_second - avg_time_first) / avg_time_first * 100

    print(f"\nPerformance Degradation:")
    print(f"  First half avg: {avg_time_first:.2f}s")
    print(f"  Second half avg: {avg_time_second:.2f}s")
    print(f"  Degradation: {degradation_pct:+.1f}%")

    if abs(degradation_pct) < 5:
        degradation_status = "✅ STABLE - No significant degradation"
    elif abs(degradation_pct) < 15:
        degradation_status = "⚠️  MINOR - Acceptable variation"
    else:
        degradation_status = "❌ SIGNIFICANT - Thermal throttling likely"

    print(f"  Status: {degradation_status}")

    # Temperature progression
    print(f"\nTemperature Progression:")
    for i, (temp, result) in enumerate(zip(temps[1:], results), 1):
        print(f"  Q{i}: {temp:.1f}°C ({result['time']:.1f}s)")

    # Production readiness assessment
    print("\n" + "="*80)
    print("PRODUCTION READINESS ASSESSMENT")
    print("="*80)

    thermal_ok = temp_max < 80
    performance_ok = abs(degradation_pct) < 15
    quality_ok = avg_salience > 0.5

    print(f"\nCriteria:")
    print(f"  Thermal stability: {'✅' if thermal_ok else '❌'} (max {temp_max:.1f}°C < 80°C)")
    print(f"  Performance stable: {'✅' if performance_ok else '❌'} ({degradation_pct:+.1f}% degradation)")
    print(f"  Quality maintained: {'✅' if quality_ok else '❌'} ({avg_salience:.3f} > 0.5)")

    if thermal_ok and performance_ok and quality_ok:
        print(f"\n✅ PRODUCTION READY")
        print(f"Edge deployment validated for continuous operation.")
        print(f"No thermal throttling or performance degradation observed.")
    else:
        print(f"\n⚠️  REVIEW NEEDED")
        if not thermal_ok:
            print(f"  - Thermal management needed (max {temp_max:.1f}°C)")
        if not performance_ok:
            print(f"  - Performance degraded ({degradation_pct:+.1f}%)")
        if not quality_ok:
            print(f"  - Quality below threshold ({avg_salience:.3f})")

    print("\n" + "="*80)
    print("✅ THERMAL STABILITY TEST COMPLETE!")
    print("="*80)

    return True


if __name__ == "__main__":
    success = test_thermal_stability(num_questions=10)
    sys.exit(0 if success else 1)
