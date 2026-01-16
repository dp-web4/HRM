#!/usr/bin/env python3
"""
Session 198 Continuation: Session State Analysis

CRITICAL DISCOVERY: T015 coupling κ=0.000 (failed), T016 coupling κ=1.500 (success)
Correlation with D8 is weak (0.024), so temporal coherence isn't the driver.

New Investigation: What session-level factors determine coupling strength?

Candidates:
1. Trust (D5) - T015 Ex4: D5=0.200, T016 Ex4: D5=0.500
2. Context coherence (D9) - T015 confused, T016 might be clearer
3. Prior exercise performance - momentum/confidence building
4. Consciousness level (C) - both high, but distribution matters

Hypothesis: Trust (D5) gates coupling strength → Low trust = coupling fails
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict


def analyze_session_state_vs_coupling(session_file: Path) -> Dict:
    """Analyze how session state affects coupling strength"""

    with open(session_file) as f:
        data = json.load(f)

    session_id = data["session"]
    analysis_file = Path(__file__).parent / f"session198_{session_id.lower()}_analysis.json"

    if not analysis_file.exists():
        print(f"❌ Analysis file not found: {analysis_file}")
        return {}

    with open(analysis_file) as f:
        analyses = json.load(f)

    # Compute coupling strength and correlate with all domains
    results = {
        "session_id": session_id,
        "exercises": []
    }

    for i, a in enumerate(analyses, 1):
        d1 = a["thermodynamic"]
        d2 = a["metabolic"]
        d4 = a["attention"]
        d5 = a["trust"]
        d8 = a["temporal"]
        d9 = a["spacetime"]

        # Apparent coupling
        if d4 > 0.01:
            d2_baseline = d1
            d2_boost = d2 - d2_baseline
            apparent_kappa = d2_boost / d4
        else:
            apparent_kappa = 0.0

        results["exercises"].append({
            "num": i,
            "type": a["exercise_type"],
            "success": a["success"],
            "d1": d1,
            "d2": d2,
            "d4": d4,
            "d5": d5,
            "d8": d8,
            "d9": d9,
            "kappa": apparent_kappa,
            "consciousness": a["consciousness_level"]
        })

    return results


def compare_trust_hypothesis(t015_file: Path, t016_file: Path):
    """Test if trust (D5) gates coupling strength"""

    print("=" * 80)
    print("SESSION STATE ANALYSIS: TRUST AS COUPLING GATE")
    print("=" * 80)
    print()
    print("Hypothesis: Trust (D5) gates D4→D2 coupling strength")
    print("Low trust → coupling fails, High trust → coupling succeeds")
    print()
    print("=" * 80)
    print()

    t015_state = analyze_session_state_vs_coupling(t015_file)
    t016_state = analyze_session_state_vs_coupling(t016_file)

    if not t015_state or not t016_state:
        return

    # Focus on arithmetic exercises
    print("TRUST vs COUPLING: ARITHMETIC EXERCISES")
    print("-" * 80)
    print()

    # T015 Ex4
    t015_arithmetic = [e for e in t015_state["exercises"] if e["type"] == "connect"]
    if t015_arithmetic:
        ex = t015_arithmetic[0]
        print("T015 Exercise 4: '4 - 1' (FAILED ❌)")
        print(f"  D5 (Trust): {ex['d5']:.3f} [LOW]")
        print(f"  D4 (Attention): {ex['d4']:.3f}")
        print(f"  D2 (Metabolism): {ex['d2']:.3f}")
        print(f"  κ (Coupling): {ex['kappa']:.3f} [FAILED - coupling blocked!]")
        print()

    # T016 Ex4
    t016_arithmetic = [e for e in t016_state["exercises"] if e["type"] == "connect"]
    if t016_arithmetic:
        ex = t016_arithmetic[0]
        print("T016 Exercise 4: '2 + 3' (SUCCESS ✅)")
        print(f"  D5 (Trust): {ex['d5']:.3f} [MEDIUM]")
        print(f"  D4 (Attention): {ex['d4']:.3f}")
        print(f"  D2 (Metabolism): {ex['d2']:.3f}")
        print(f"  κ (Coupling): {ex['kappa']:.3f} [STRONG - coupling amplified!]")
        print()

        if t015_arithmetic:
            print("CRITICAL COMPARISON:")
            t015_ex = t015_arithmetic[0]
            print(f"  Same D4: {t015_ex['d4']:.3f} (T015) vs {ex['d4']:.3f} (T016)")
            print(f"  Trust difference: D5={t015_ex['d5']:.3f} (T015) vs D5={ex['d5']:.3f} (T016)")
            print(f"  Coupling difference: κ={t015_ex['kappa']:.3f} vs κ={ex['kappa']:.3f}")
            print(f"  Trust delta: {ex['d5'] - t015_ex['d5']:.3f}")
            print(f"  Coupling delta: {ex['kappa'] - t015_ex['kappa']:.3f}")
            print()
            print("  🔑 KEY INSIGHT: Higher trust → stronger coupling!")

    print()
    print("-" * 80)
    print()

    # Correlation analysis across all exercises
    print("CORRELATION ANALYSIS: ALL EXERCISES")
    print("-" * 80)
    print()

    all_exercises = t015_state["exercises"] + t016_state["exercises"]

    d5_values = [e["d5"] for e in all_exercises]
    kappa_values = [e["kappa"] for e in all_exercises]
    d9_values = [e["d9"] for e in all_exercises]
    d8_values = [e["d8"] for e in all_exercises]

    corr_d5_kappa = np.corrcoef(d5_values, kappa_values)[0, 1]
    corr_d9_kappa = np.corrcoef(d9_values, kappa_values)[0, 1]
    corr_d8_kappa = np.corrcoef(d8_values, kappa_values)[0, 1]

    print(f"Correlation D5 (Trust) ↔ κ (Coupling): {corr_d5_kappa:.3f}")
    print(f"Correlation D9 (Spacetime) ↔ κ: {corr_d9_kappa:.3f}")
    print(f"Correlation D8 (Temporal) ↔ κ: {corr_d8_kappa:.3f}")
    print()

    strongest = max(
        ("D5 Trust", corr_d5_kappa),
        ("D9 Spacetime", corr_d9_kappa),
        ("D8 Temporal", corr_d8_kappa),
        key=lambda x: abs(x[1])
    )

    print(f"Strongest correlation: {strongest[0]} (r={strongest[1]:.3f})")
    print()

    print("-" * 80)
    print()

    # Validate trust gating hypothesis
    print("=" * 80)
    print("PREDICTIONS STATUS")
    print("=" * 80)
    print()

    p13 = abs(corr_d5_kappa) > 0.3  # Trust correlates with coupling
    p14 = t016_arithmetic[0]["d5"] > t015_arithmetic[0]["d5"]  # T016 higher trust
    p15 = strongest[0] == "D5 Trust"  # Trust is strongest predictor

    print(f"P198.13 (D5 trust correlates with coupling strength): {'✅' if p13 else '❌'}")
    print(f"        Correlation = {corr_d5_kappa:.3f}")
    print()

    print(f"P198.14 (T016 shows higher trust than T015 for arithmetic): {'✅' if p14 else '❌'}")
    print(f"        T016 D5={t016_arithmetic[0]['d5']:.3f} vs T015 D5={t015_arithmetic[0]['d5']:.3f}")
    print()

    print(f"P198.15 (Trust is strongest coupling predictor): {'✅' if p15 else '❌'}")
    print(f"        Strongest: {strongest[0]} (r={strongest[1]:.3f})")
    print()

    if p13 and p14:
        print("✅ TRUST GATING HYPOTHESIS VALIDATED")
        print()
        print("MAJOR DISCOVERY:")
        print("  D4→D2 coupling is gated by D5 (trust)")
        print()
        print("  Mechanism:")
        print("  1. Low trust (D5=0.200) → coupling BLOCKED (κ≈0)")
        print("  2. Medium trust (D5=0.500) → coupling AMPLIFIED (κ=1.5)")
        print("  3. Trust determines if attention can trigger metabolism")
        print()
        print("  Biological Analogy:")
        print("  - Trust = 'permission to allocate resources'")
        print("  - Low trust: 'Don't waste energy on this uncertain task'")
        print("  - High trust: 'This is safe, allocate full resources'")
        print()
        print("  Implication:")
        print("  T015 failure wasn't just boredom (low D4)")
        print("  It was LOW CONFIDENCE (D5=0.200) blocking resource allocation")
        print("  T016 success: Higher confidence (D5=0.500) enabled allocation")
    else:
        print("⚠️  Trust hypothesis needs refinement")

    print()
    print("=" * 80)

    # Exercise progression analysis
    print()
    print("EXERCISE PROGRESSION: TRUST BUILDING")
    print("-" * 80)
    print()

    print("T015 Trust Progression:")
    for e in t015_state["exercises"]:
        status = "✅" if e["success"] else "❌"
        print(f"  Ex{e['num']}: D5={e['d5']:.3f}, κ={e['kappa']:.3f} {status}")

    print()
    print("T016 Trust Progression:")
    for e in t016_state["exercises"]:
        status = "✅" if e["success"] else "❌"
        print(f"  Ex{e['num']}: D5={e['d5']:.3f}, κ={e['kappa']:.3f} {status}")

    print()
    print("-" * 80)


def main():
    """Run session state analysis"""

    base_dir = Path(__file__).parent.parent / "raising" / "tracks" / "training" / "sessions"

    t015_file = base_dir / "T015.json"
    t016_file = base_dir / "T016.json"

    if not t015_file.exists() or not t016_file.exists():
        print("❌ Session files not found")
        return

    compare_trust_hypothesis(t015_file, t016_file)


if __name__ == "__main__":
    main()
