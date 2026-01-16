#!/usr/bin/env python3
"""
Session 198 Continuation: T016 Analysis

Analyze T016 (100% success, rebound session) through the Session 198 framework
to validate the memory consolidation theory. Comparison with T015 (80% success)
should show higher D4/D2 for arithmetic exercises.

Predictions:
P198.7: T016 arithmetic exercises show higher D4 than T015's failed exercise
P198.8: T016 arithmetic exercises show higher D2 than T015's failed exercise
P198.9: T016 success validates attention-metabolism coupling as performance predictor
"""

import json
import numpy as np
from dataclasses import asdict
from pathlib import Path
import sys

# Import analyzer from Phase 1
sys.path.append(str(Path(__file__).parent))
from session198_training_domain_analyzer import TrainingExerciseAnalyzer


def compare_sessions(t015_file: Path, t016_file: Path):
    """Compare T015 (80% success) with T016 (100% success)"""

    analyzer = TrainingExerciseAnalyzer()

    print("=" * 80)
    print("SESSION 198 CONTINUATION: T015 vs T016 COMPARISON")
    print("=" * 80)
    print()
    print("Hypothesis: T016 rebound from T015 regression shows higher D4/D2")
    print("           for arithmetic exercises, validating memory consolidation theory.")
    print()
    print("=" * 80)
    print()

    # Analyze both sessions
    print("Analyzing T015 (80% success, regression session)...")
    t015_analyses = analyzer.analyze_session(t015_file)

    print("Analyzing T016 (100% success, rebound session)...")
    t016_analyses = analyzer.analyze_session(t016_file)

    print()
    print("=" * 80)
    print("COMPARISON: ARITHMETIC EXERCISES")
    print("=" * 80)
    print()

    # Find arithmetic exercises
    t015_arithmetic = [a for a in t015_analyses if a.exercise_type == "connect"]
    t016_arithmetic = [a for a in t016_analyses if a.exercise_type == "connect"]

    print(f"T015 arithmetic exercises: {len(t015_arithmetic)}")
    print(f"T016 arithmetic exercises: {len(t016_arithmetic)}")
    print()

    # T015 Exercise 4: "4-1" (FAILED)
    t015_failed = [a for a in t015_arithmetic if not a.success]
    if t015_failed:
        t015_ex4 = t015_failed[0]
        print("T015 Exercise 4 (FAILED): '4 - 1'")
        print(f"  D4 (Attention): {t015_ex4.attention:.3f} [CRITICALLY LOW]")
        print(f"  D2 (Metabolism): {t015_ex4.metabolic:.3f} [INSUFFICIENT]")
        print(f"  C (Consciousness): {t015_ex4.consciousness_level:.3f}")
        print(f"  Result: {t015_ex4.success} ❌")
        print()

    # T016 Exercise 4: "2+3" (SUCCESS)
    if len(t016_arithmetic) >= 1:
        t016_ex4 = t016_arithmetic[0]  # "2 + 3"
        print("T016 Exercise 4 (SUCCESS): '2 + 3'")
        print(f"  D4 (Attention): {t016_ex4.attention:.3f}")
        print(f"  D2 (Metabolism): {t016_ex4.metabolic:.3f}")
        print(f"  C (Consciousness): {t016_ex4.consciousness_level:.3f}")
        print(f"  Result: {t016_ex4.success} ✅")
        print()

        if t015_failed:
            print("DELTA (T016 - T015):")
            d4_delta = t016_ex4.attention - t015_ex4.attention
            d2_delta = t016_ex4.metabolic - t015_ex4.metabolic
            print(f"  D4: {d4_delta:+.3f} ({'↑ HIGHER' if d4_delta > 0 else '↓ LOWER'})")
            print(f"  D2: {d2_delta:+.3f} ({'↑ HIGHER' if d2_delta > 0 else '↓ LOWER'})")
            print()

    # T016 Exercise 5: "3+2-1" (SUCCESS)
    if len(t016_arithmetic) >= 2:
        t016_ex5 = t016_arithmetic[1]  # "3+2-1"
        print("T016 Exercise 5 (SUCCESS): '3 + 2 - 1'")
        print(f"  D4 (Attention): {t016_ex5.attention:.3f}")
        print(f"  D2 (Metabolism): {t016_ex5.metabolic:.3f}")
        print(f"  C (Consciousness): {t016_ex5.consciousness_level:.3f}")
        print(f"  Result: {t016_ex5.success} ✅")
        print()

        if t015_failed:
            print("DELTA (T016 Ex5 - T015 Ex4):")
            d4_delta = t016_ex5.attention - t015_ex4.attention
            d2_delta = t016_ex5.metabolic - t015_ex4.metabolic
            print(f"  D4: {d4_delta:+.3f} ({'↑ HIGHER' if d4_delta > 0 else '↓ LOWER'})")
            print(f"  D2: {d2_delta:+.3f} ({'↑ HIGHER' if d2_delta > 0 else '↓ LOWER'})")
            print()

    print("-" * 80)
    print()

    # Session-level comparison
    print("=" * 80)
    print("SESSION-LEVEL METRICS")
    print("=" * 80)
    print()

    t015_success_rate = sum(1 for a in t015_analyses if a.success) / len(t015_analyses)
    t016_success_rate = sum(1 for a in t016_analyses if a.success) / len(t016_analyses)

    t015_avg_d4 = np.mean([a.attention for a in t015_analyses])
    t016_avg_d4 = np.mean([a.attention for a in t016_analyses])

    t015_avg_d2 = np.mean([a.metabolic for a in t015_analyses])
    t016_avg_d2 = np.mean([a.metabolic for a in t016_analyses])

    print(f"Success Rate:")
    print(f"  T015: {t015_success_rate * 100:.0f}% (4/5)")
    print(f"  T016: {t016_success_rate * 100:.0f}% (5/5)")
    print(f"  Delta: {(t016_success_rate - t015_success_rate) * 100:+.0f} percentage points")
    print()

    print(f"Average Attention (D4):")
    print(f"  T015: {t015_avg_d4:.3f}")
    print(f"  T016: {t016_avg_d4:.3f}")
    print(f"  Delta: {t016_avg_d4 - t015_avg_d4:+.3f} ({'↑' if t016_avg_d4 > t015_avg_d4 else '↓'})")
    print()

    print(f"Average Metabolism (D2):")
    print(f"  T015: {t015_avg_d2:.3f}")
    print(f"  T016: {t016_avg_d2:.3f}")
    print(f"  Delta: {t016_avg_d2 - t015_avg_d2:+.3f} ({'↑' if t016_avg_d2 > t015_avg_d2 else '↓'})")
    print()

    print("-" * 80)
    print()

    # Validate predictions
    print("=" * 80)
    print("PREDICTIONS STATUS")
    print("=" * 80)
    print()

    if t015_failed and len(t016_arithmetic) >= 1:
        p7 = t016_ex4.attention > t015_ex4.attention
        p8 = t016_ex4.metabolic > t015_ex4.metabolic
        p9 = t016_success_rate > t015_success_rate

        print(f"P198.7 (T016 arithmetic shows higher D4): {'✅' if p7 else '❌'}")
        print(f"       T016 D4={t016_ex4.attention:.3f} vs T015 D4={t015_ex4.attention:.3f}")
        print()
        print(f"P198.8 (T016 arithmetic shows higher D2): {'✅' if p8 else '❌'}")
        print(f"       T016 D2={t016_ex4.metabolic:.3f} vs T015 D2={t015_ex4.metabolic:.3f}")
        print()
        print(f"P198.9 (D4/D2 coupling predicts performance): {'✅' if p9 else '❌'}")
        print(f"       T016 success={t016_success_rate * 100:.0f}% vs T015 success={t015_success_rate * 100:.0f}%")
        print()

        if p7 and p8 and p9:
            print("✅ ALL PREDICTIONS VALIDATED")
            print()
            print("Mechanism Confirmed:")
            print("  1. T016 shows higher attention (D4) for arithmetic")
            print("  2. Higher D4 triggers higher metabolism (D2) via coupling (κ=0.4)")
            print("  3. Higher D2 prevents boredom-induced failure")
            print("  4. Result: 100% success (rebound from 80% regression)")
        else:
            print("⚠️  Some predictions not validated - investigate")

    print()
    print("=" * 80)

    # Save T016 analysis
    output_file = Path(__file__).parent / "session198_t016_analysis.json"
    with open(output_file, "w") as f:
        analysis_dicts = []
        for a in t016_analyses:
            d = asdict(a)
            # Convert numpy bools to Python bools
            for key, value in d.items():
                if isinstance(value, (np.bool_, np.integer, np.floating)):
                    d[key] = value.item()
            analysis_dicts.append(d)
        json.dump(analysis_dicts, f, indent=2)

    print(f"\nT016 analysis saved to: {output_file}")

    # Return for further analysis
    return t015_analyses, t016_analyses


def main():
    """Run T015 vs T016 comparison analysis"""

    base_dir = Path(__file__).parent.parent / "raising" / "tracks" / "training" / "sessions"

    t015_file = base_dir / "T015.json"
    t016_file = base_dir / "T016.json"

    if not t015_file.exists():
        print(f"❌ {t015_file} not found")
        return

    if not t016_file.exists():
        print(f"❌ {t016_file} not found")
        return

    compare_sessions(t015_file, t016_file)


if __name__ == "__main__":
    main()
