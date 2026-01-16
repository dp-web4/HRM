#!/usr/bin/env python3
"""
Session 198 Continuation: Coupling Dynamics Investigation

DISCOVERY: T016 shows SAME low D4 (0.200) as T015 failure, but HIGHER D2 (0.734).
This suggests D4→D2 coupling strength (κ) is NOT constant across sessions!

New Hypothesis: κ_42 varies based on session state (temporal coherence, trust, etc.)

This explains:
- T015: Low D4=0.200 + weak coupling → D2=0.364 → FAILURE
- T016: Low D4=0.200 + strong coupling → D2=0.734 → SUCCESS

Predictions:
P198.10: Coupling strength κ_42 is session-dependent (not constant 0.4)
P198.11: Higher temporal coherence (D8) amplifies coupling strength
P198.12: Session state modulates attention→metabolism efficiency
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class CouplingAnalysis:
    """Analysis of D4→D2 coupling for an exercise"""
    exercise_num: int
    exercise_type: str
    success: bool
    d4: float
    d2: float
    d8: float  # Temporal coherence
    d5: float  # Trust
    apparent_kappa: float  # Inferred coupling strength
    expected_d2: float  # Expected D2 from κ=0.4


def analyze_coupling_dynamics(session_file: Path) -> List[CouplingAnalysis]:
    """Analyze D4→D2 coupling for each exercise in session"""

    with open(session_file) as f:
        data = json.load(f)

    # Load pre-computed analysis
    session_id = data["session"]
    analysis_file = Path(__file__).parent / f"session198_{session_id.lower()}_analysis.json"

    if not analysis_file.exists():
        print(f"❌ Analysis file not found: {analysis_file}")
        return []

    with open(analysis_file) as f:
        analyses = json.load(f)

    coupling_analyses = []

    for a in analyses:
        # Compute apparent coupling strength
        # If D4→D2 coupling: D2 = D1_base + κ_42 * D4
        # Rearrange: κ_apparent = (D2 - D1_base) / D4
        # Assume D1_base ≈ D1 (thermodynamic baseline)

        d1 = a["thermodynamic"]
        d2 = a["metabolic"]
        d4 = a["attention"]
        d5 = a["trust"]
        d8 = a["temporal"]

        # Apparent coupling: how much D4 actually influenced D2
        if d4 > 0.01:  # Avoid division by zero
            # Assume baseline D2 ≈ D1 (thermodynamic determines baseline metabolism)
            d2_baseline = d1
            d2_boost = d2 - d2_baseline
            apparent_kappa = d2_boost / d4 if d4 > 0 else 0.0
        else:
            apparent_kappa = 0.0

        # Expected D2 from Session 196 κ=0.4
        expected_d2 = d1 + 0.4 * d4

        coupling_analyses.append(CouplingAnalysis(
            exercise_num=a["exercise_num"],
            exercise_type=a["exercise_type"],
            success=a["success"],
            d4=d4,
            d2=d2,
            d8=d8,
            d5=d5,
            apparent_kappa=apparent_kappa,
            expected_d2=expected_d2
        ))

    return coupling_analyses


def compare_coupling_between_sessions(t015_file: Path, t016_file: Path):
    """Compare coupling dynamics between T015 (failed) and T016 (success)"""

    print("=" * 80)
    print("SESSION 198 COUPLING DYNAMICS INVESTIGATION")
    print("=" * 80)
    print()
    print("Discovery: T016 has SAME low D4 (0.200) as T015 failure, but HIGHER D2.")
    print("Hypothesis: D4→D2 coupling strength κ varies across sessions.")
    print()
    print("=" * 80)
    print()

    t015_coupling = analyze_coupling_dynamics(t015_file)
    t016_coupling = analyze_coupling_dynamics(t016_file)

    if not t015_coupling or not t016_coupling:
        print("❌ Missing analysis files")
        return

    # Focus on arithmetic exercises
    print("ARITHMETIC EXERCISE COUPLING COMPARISON")
    print("-" * 80)
    print()

    # T015 Ex4: 4-1 (FAILED)
    t015_ex4 = [c for c in t015_coupling if c.exercise_type == "connect"]
    if t015_ex4:
        ex = t015_ex4[0]
        print("T015 Exercise 4: '4 - 1' (FAILED ❌)")
        print(f"  D4 (Attention): {ex.d4:.3f}")
        print(f"  D2 (Metabolism): {ex.d2:.3f}")
        print(f"  D8 (Temporal): {ex.d8:.3f}")
        print(f"  D5 (Trust): {ex.d5:.3f}")
        print(f"  Expected D2 (κ=0.4): {ex.expected_d2:.3f}")
        print(f"  Apparent κ: {ex.apparent_kappa:.3f}")
        print()

    # T016 Ex4: 2+3 (SUCCESS)
    t016_ex4 = [c for c in t016_coupling if c.exercise_type == "connect"]
    if t016_ex4:
        ex = t016_ex4[0]
        print("T016 Exercise 4: '2 + 3' (SUCCESS ✅)")
        print(f"  D4 (Attention): {ex.d4:.3f}")
        print(f"  D2 (Metabolism): {ex.d2:.3f}")
        print(f"  D8 (Temporal): {ex.d8:.3f}")
        print(f"  D5 (Trust): {ex.d5:.3f}")
        print(f"  Expected D2 (κ=0.4): {ex.expected_d2:.3f}")
        print(f"  Apparent κ: {ex.apparent_kappa:.3f}")
        print()

        if t015_ex4:
            print("CRITICAL FINDING:")
            print(f"  Same D4: {t015_ex4[0].d4:.3f} (T015) vs {ex.d4:.3f} (T016)")
            print(f"  Different D2: {t015_ex4[0].d2:.3f} (T015) vs {ex.d2:.3f} (T016)")
            print(f"  Coupling strength: {t015_ex4[0].apparent_kappa:.3f} (T015) vs {ex.apparent_kappa:.3f} (T016)")
            kappa_ratio = ex.apparent_kappa / t015_ex4[0].apparent_kappa if t015_ex4[0].apparent_kappa > 0 else float('inf')
            print(f"  T016 coupling is {kappa_ratio:.1f}x stronger!")
            print()

    # T016 Ex5: 3+2-1 (SUCCESS)
    if len(t016_ex4) >= 2:
        ex = t016_ex4[1]
        print("T016 Exercise 5: '3 + 2 - 1' (SUCCESS ✅)")
        print(f"  D4 (Attention): {ex.d4:.3f}")
        print(f"  D2 (Metabolism): {ex.d2:.3f}")
        print(f"  D8 (Temporal): {ex.d8:.3f}")
        print(f"  D5 (Trust): {ex.d5:.3f}")
        print(f"  Expected D2 (κ=0.4): {ex.expected_d2:.3f}")
        print(f"  Apparent κ: {ex.apparent_kappa:.3f}")
        print()

    print("-" * 80)
    print()

    # Session-level coupling analysis
    print("SESSION-LEVEL COUPLING METRICS")
    print("-" * 80)
    print()

    t015_avg_kappa = np.mean([c.apparent_kappa for c in t015_coupling])
    t016_avg_kappa = np.mean([c.apparent_kappa for c in t016_coupling])

    t015_avg_d8 = np.mean([c.d8 for c in t015_coupling])
    t016_avg_d8 = np.mean([c.d8 for c in t016_coupling])

    print(f"Average Coupling Strength (κ_apparent):")
    print(f"  T015: {t015_avg_kappa:.3f}")
    print(f"  T016: {t016_avg_kappa:.3f}")
    print(f"  Delta: {t016_avg_kappa - t015_avg_kappa:+.3f} ({'↑ STRONGER' if t016_avg_kappa > t015_avg_kappa else '↓ WEAKER'})")
    print()

    print(f"Average Temporal Coherence (D8):")
    print(f"  T015: {t015_avg_d8:.3f}")
    print(f"  T016: {t016_avg_d8:.3f}")
    print(f"  Delta: {t016_avg_d8 - t015_avg_d8:+.3f} ({'↑' if t016_avg_d8 > t015_avg_d8 else '↓'})")
    print()

    # Correlation: D8 vs κ
    all_d8 = [c.d8 for c in t015_coupling] + [c.d8 for c in t016_coupling]
    all_kappa = [c.apparent_kappa for c in t015_coupling] + [c.apparent_kappa for c in t016_coupling]

    correlation = np.corrcoef(all_d8, all_kappa)[0, 1]
    print(f"Correlation (D8 ↔ κ): {correlation:.3f}")
    print()

    print("-" * 80)
    print()

    # Predictions
    print("=" * 80)
    print("PREDICTIONS STATUS")
    print("=" * 80)
    print()

    p10 = abs(t016_avg_kappa - t015_avg_kappa) > 0.1
    p11 = correlation > 0.3
    p12 = t016_avg_kappa > t015_avg_kappa

    print(f"P198.10 (κ is session-dependent, not constant): {'✅' if p10 else '❌'}")
    print(f"        T015 κ={t015_avg_kappa:.3f} vs T016 κ={t016_avg_kappa:.3f}")
    print()

    print(f"P198.11 (D8 temporal coherence correlates with κ): {'✅' if p11 else '❌'}")
    print(f"        Correlation = {correlation:.3f}")
    print()

    print(f"P198.12 (Session state modulates coupling efficiency): {'✅' if p12 else '❌'}")
    print(f"        T016 coupling stronger than T015")
    print()

    if p10 and p11 and p12:
        print("✅ ALL PREDICTIONS VALIDATED")
        print()
        print("MAJOR DISCOVERY:")
        print("  D4→D2 coupling strength is NOT constant (κ ≠ fixed 0.4)")
        print("  Instead: κ = κ(D8, session_state)")
        print()
        print("  Mechanism:")
        print("  1. Temporal coherence (D8) builds through session")
        print("  2. Higher D8 → stronger D4→D2 coupling")
        print("  3. Same low D4 can yield different D2 depending on κ")
        print("  4. T016 success: Low D4 + strong κ → sufficient D2")
        print("  5. T015 failure: Low D4 + weak κ → insufficient D2")
        print()
        print("  Implication: Session warm-up matters!")
        print("  Early exercises build D8 → strengthen coupling → prevent later failures")
    else:
        print("⚠️  Some predictions not validated")

    print()
    print("=" * 80)


def main():
    """Run coupling dynamics investigation"""

    base_dir = Path(__file__).parent.parent / "raising" / "tracks" / "training" / "sessions"

    t015_file = base_dir / "T015.json"
    t016_file = base_dir / "T016.json"

    if not t015_file.exists() or not t016_file.exists():
        print("❌ Session files not found")
        return

    compare_coupling_between_sessions(t015_file, t016_file)


if __name__ == "__main__":
    main()
