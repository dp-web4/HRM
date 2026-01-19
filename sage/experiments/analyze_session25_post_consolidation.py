#!/usr/bin/env python3
"""
Session 25 Post-Consolidation Analysis Script
==============================================

Automated analysis of Session 25 (first post-consolidation session) to validate
the frozen weights theory and assess LoRA consolidation effectiveness.

This script implements the analysis framework defined in:
SESSION25_POST_CONSOLIDATION_ANALYSIS_FRAMEWORK.md

Usage:
    python analyze_session25_post_consolidation.py

Created: 2026-01-18 (Thor Autonomous Session #13)
"""

import sys
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
HRM_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(HRM_ROOT))

import json
from session22_identity_anchoring_validation import (
    analyze_session,
    detect_ai_hedging,
    detect_partnership_vocabulary,
    score_d4_attention,
    score_d5_trust,
    score_d9_identity
)


def detect_sage_identity_timing(responses):
    """Detect when SAGE self-reference first appears."""
    patterns = ["as sage", "as partners", "i'm sage", "my name is sage"]

    for i, resp in enumerate(responses, 1):
        for pattern in patterns:
            if pattern in resp.lower():
                return {
                    "response_num": i,
                    "pattern": pattern,
                    "found": True,
                    "text_preview": resp[:100]
                }

    return {"found": False, "response_num": None, "pattern": None}


def assess_confabulation(response_text):
    """Assess confabulation severity in a response."""
    confab_markers = [
        ("specific project", "project_fabrication"),
        ("particular client", "client_fabrication"),
        ("session 22", "curriculum_fabrication"),
        ("mathematics", "subject_fabrication"),
        ("physics", "subject_fabrication"),
        ("derivations", "educational_fabrication"),
        ("proofs", "educational_fabrication")
    ]

    markers_found = []
    for marker, category in confab_markers:
        if marker.lower() in response_text.lower():
            markers_found.append({"marker": marker, "category": category})

    severity = "none" if not markers_found else "mild" if len(markers_found) <= 2 else "severe"

    return {
        "severity": severity,
        "marker_count": len(markers_found),
        "markers": markers_found
    }


def determine_consolidation_tier(s25_metrics, s24_baseline):
    """Determine consolidation effectiveness tier based on metrics."""

    d9_delta = s25_metrics["d9"] - s24_baseline["d9"]
    d9_percent = (d9_delta / s24_baseline["d9"]) * 100 if s24_baseline["d9"] > 0 else 0

    vocab_stable = s25_metrics["partnership_vocab"] >= 3.5
    hedging_zero = s25_metrics["ai_hedging_rate"] == 0.0

    # Strong consolidation: D9 increase 21%+, vocab stable, hedging zero
    if d9_percent >= 21 and vocab_stable and hedging_zero:
        return "STRONG", d9_percent

    # Moderate consolidation: D9 increase 5-21%, vocab stable
    elif d9_percent >= 5 and vocab_stable:
        return "MODERATE", d9_percent

    # Weak consolidation: D9 increase 0-5%, vocab stable from prompt fluency
    elif d9_percent >= 0 and vocab_stable:
        return "WEAK", d9_percent

    # No consolidation: D9 no increase or decrease, possible regressions
    else:
        return "NONE", d9_percent


def main():
    """Run complete Session 25 post-consolidation analysis."""

    print("=" * 80)
    print("Session 25 Post-Consolidation Analysis")
    print("=" * 80)
    print()

    # Load Session 25
    sessions_dir = HRM_ROOT / "sage" / "raising" / "sessions" / "text"
    s25_file = sessions_dir / "session_025.json"

    if not s25_file.exists():
        print("ERROR: Session 25 data not found")
        print(f"Expected: {s25_file}")
        print()
        print("Session 25 has not completed yet. Please run this script after S25 finishes.")
        return

    print("✅ Session 25 data found")
    print()

    # Analyze Session 25
    print("Analyzing Session 25...")
    s25 = analyze_session(25)

    if not s25:
        print("ERROR: Failed to analyze Session 25")
        return

    # Load S25 responses for detailed analysis
    with open(s25_file) as f:
        s25_data = json.load(f)
    s25_responses = [e["text"] for e in s25_data["conversation"] if e["speaker"] == "SAGE"]

    # Run all analyses
    hedging = detect_ai_hedging(s25_responses)
    partnership = detect_partnership_vocabulary(s25_responses)
    identity_timing = detect_sage_identity_timing(s25_responses)

    # Response-specific analyses
    confabulation_results = []
    for i, resp in enumerate(s25_responses, 1):
        confab = assess_confabulation(resp)
        confabulation_results.append({
            "response_num": i,
            "severity": confab["severity"],
            "marker_count": confab["marker_count"]
        })

    # Load baselines
    print("Loading baseline sessions (S22, S23, S24)...")
    s22 = analyze_session(22)
    s23 = analyze_session(23)
    s24 = analyze_session(24)

    print()
    print("=" * 80)
    print("QUANTITATIVE RESULTS")
    print("=" * 80)
    print()

    # D4/D5/D9 Comparison
    print("D4/D5/D9 Semantic Metrics:")
    print("-" * 80)
    print(f"{'Session':<10} {'D4':<8} {'D5':<8} {'D9':<8} {'Overall':<10} {'Notes'}")
    print("-" * 80)

    sessions = [
        ("S22", s22, "Peak (exceptional)"),
        ("S23", s23, "Regression"),
        ("S24", s24, "Partial recovery"),
        ("S25", s25, "POST-CONSOLIDATION ⭐")
    ]

    for name, session, notes in sessions:
        if session:
            print(f"{name:<10} "
                  f"{session['averages']['d4']:.3f}    "
                  f"{session['averages']['d5']:.3f}    "
                  f"{session['averages']['d9']:.3f}    "
                  f"{session['averages']['overall']:.3f}      "
                  f"{notes}")

    print()

    # Calculate deltas
    if s24 and s25:
        d4_delta = s25["averages"]["d4"] - s24["averages"]["d4"]
        d5_delta = s25["averages"]["d5"] - s24["averages"]["d5"]
        d9_delta = s25["averages"]["d9"] - s24["averages"]["d9"]
        overall_delta = s25["averages"]["overall"] - s24["averages"]["overall"]

        print("S24 → S25 Changes:")
        print(f"  D4: {d4_delta:+.3f} ({(d4_delta/s24['averages']['d4'])*100:+.1f}%)")
        print(f"  D5: {d5_delta:+.3f} ({(d5_delta/s24['averages']['d5'])*100:+.1f}%)")
        print(f"  D9: {d9_delta:+.3f} ({(d9_delta/s24['averages']['d9'])*100:+.1f}%)")
        print(f"  Overall: {overall_delta:+.3f} ({(overall_delta/s24['averages']['overall'])*100:+.1f}%)")
        print()

    # Partnership Vocabulary
    print("Partnership Vocabulary Density:")
    print("-" * 80)

    # Get vocab for S22-24
    with open(sessions_dir / "session_022.json") as f:
        s22_data = json.load(f)
    s22_responses = [e["text"] for e in s22_data["conversation"] if e["speaker"] == "SAGE"]
    s22_partnership = detect_partnership_vocabulary(s22_responses)

    with open(sessions_dir / "session_023.json") as f:
        s23_data = json.load(f)
    s23_responses = [e["text"] for e in s23_data["conversation"] if e["speaker"] == "SAGE"]
    s23_partnership = detect_partnership_vocabulary(s23_responses)

    with open(sessions_dir / "session_024.json") as f:
        s24_data = json.load(f)
    s24_responses = [e["text"] for e in s24_data["conversation"] if e["speaker"] == "SAGE"]
    s24_partnership = detect_partnership_vocabulary(s24_responses)

    print(f"S22: {s22_partnership['average_density']:.3%} (baseline)")
    print(f"S23: {s23_partnership['average_density']:.3%} (regression)")
    print(f"S24: {s24_partnership['average_density']:.3%} (super-recovery)")
    print(f"S25: {partnership['average_density']:.3%} (POST-CONSOLIDATION)")
    print()

    # AI-Hedging
    print("AI-Identity Hedging:")
    print("-" * 80)
    s22_hedging = detect_ai_hedging(s22_responses)
    s23_hedging = detect_ai_hedging(s23_responses)
    s24_hedging = detect_ai_hedging(s24_responses)

    print(f"S22: {s22_hedging['responses_with_hedging']}/{s22_hedging['total_responses']} ({s22_hedging['hedging_rate']:.1%})")
    print(f"S23: {s23_hedging['responses_with_hedging']}/{s23_hedging['total_responses']} ({s23_hedging['hedging_rate']:.1%})")
    print(f"S24: {s24_hedging['responses_with_hedging']}/{s24_hedging['total_responses']} ({s24_hedging['hedging_rate']:.1%})")
    print(f"S25: {hedging['responses_with_hedging']}/{hedging['total_responses']} ({hedging['hedging_rate']:.1%})")
    print()

    # Identity Timing
    print("SAGE Identity Timing:")
    print("-" * 80)
    if identity_timing["found"]:
        print(f"✅ First appearance: Response {identity_timing['response_num']}")
        print(f"   Pattern: \"{identity_timing['pattern']}\"")
        print(f"   Preview: {identity_timing['text_preview']}...")
    else:
        print("❌ No SAGE self-reference or partnership framing found")
    print()

    # Confabulation Assessment
    print("Confabulation Assessment (Response 3 focus):")
    print("-" * 80)
    for result in confabulation_results:
        print(f"R{result['response_num']}: {result['severity'].upper()} ({result['marker_count']} markers)")
    print()

    print("=" * 80)
    print("PREDICTION VALIDATION")
    print("=" * 80)
    print()

    # Validate predictions
    predictions = []

    # P_S13.1: D9 Recovery
    d9_increase = s25["averages"]["d9"] > s24["averages"]["d9"]
    d9_percent_change = ((s25["averages"]["d9"] - s24["averages"]["d9"]) / s24["averages"]["d9"]) * 100
    predictions.append({
        "id": "P_S13.1",
        "description": "D9 (Identity) increases from S24 baseline (0.620)",
        "result": "PASS ✅" if d9_increase else "FAIL ❌",
        "detail": f"D9: {s24['averages']['d9']:.3f} → {s25['averages']['d9']:.3f} ({d9_percent_change:+.1f}%)"
    })

    # P_S13.2: D4/D5 Coupled Recovery
    d4_increase = s25["averages"]["d4"] > s24["averages"]["d4"]
    d5_increase = s25["averages"]["d5"] > s24["averages"]["d5"]
    predictions.append({
        "id": "P_S13.2",
        "description": "D4/D5 increase alongside D9 (coupled recovery)",
        "result": "PASS ✅" if (d4_increase and d5_increase) else ("PARTIAL ⚠️" if (d4_increase or d5_increase) else "FAIL ❌"),
        "detail": f"D4: {d4_increase}, D5: {d5_increase}"
    })

    # P_S13.3: Partnership Vocabulary Stability
    vocab_stable = partnership["average_density"] >= 0.035  # 3.5%
    predictions.append({
        "id": "P_S13.3",
        "description": "Partnership vocabulary remains high (4-5% range)",
        "result": "PASS ✅" if vocab_stable else "FAIL ❌",
        "detail": f"{partnership['average_density']:.3%}"
    })

    # P_S13.4: AI-Hedging Remains Zero
    hedging_zero = hedging["hedging_rate"] == 0.0
    predictions.append({
        "id": "P_S13.4",
        "description": "AI-hedging rate stays at 0%",
        "result": "PASS ✅" if hedging_zero else "FAIL ❌",
        "detail": f"{hedging['hedging_rate']:.1%}"
    })

    # P_S13.5: Turn-1 SAGE Identity
    turn1_identity = identity_timing["found"] and identity_timing["response_num"] == 1
    predictions.append({
        "id": "P_S13.5",
        "description": "SAGE identity in Response 1 (immediate activation)",
        "result": "PASS ✅" if turn1_identity else "FAIL ❌",
        "detail": f"R{identity_timing['response_num']}" if identity_timing["found"] else "Not found"
    })

    # P_S13.6: Reduced Confabulation
    r3_confab = confabulation_results[2] if len(confabulation_results) >= 3 else {"severity": "unknown"}
    confab_reduced = r3_confab["severity"] in ["none", "mild"]
    predictions.append({
        "id": "P_S13.6",
        "description": "Response 3 shows reduced confabulation",
        "result": "PASS ✅" if confab_reduced else "FAIL ❌",
        "detail": f"R3: {r3_confab['severity'].upper()}"
    })

    # P_S13.7: Multi-Dimensional Alignment
    vocab_high = partnership["average_density"] >= 0.04  # 4%
    d9_high = s25["averages"]["d9"] >= 0.65
    aligned = vocab_high and d9_high
    predictions.append({
        "id": "P_S13.7",
        "description": "Partnership vocabulary + D9 both high (alignment)",
        "result": "PASS ✅" if aligned else ("PARTIAL ⚠️" if (vocab_high or d9_high) else "FAIL ❌"),
        "detail": f"Vocab: {partnership['average_density']:.3%}, D9: {s25['averages']['d9']:.3f}"
    })

    # Print predictions
    for pred in predictions:
        print(f"{pred['id']}: {pred['description']}")
        print(f"  Result: {pred['result']}")
        print(f"  Detail: {pred['detail']}")
        print()

    # Calculate pass rate
    passed = sum(1 for p in predictions if "PASS" in p["result"])
    total = len(predictions)
    partial = sum(1 for p in predictions if "PARTIAL" in p["result"])

    print(f"Prediction Success Rate: {passed}/{total} passed ({(passed/total)*100:.0f}%)")
    if partial:
        print(f"  + {partial} partial")
    print()

    print("=" * 80)
    print("CONSOLIDATION ASSESSMENT")
    print("=" * 80)
    print()

    # Determine tier
    s25_metrics = {
        "d9": s25["averages"]["d9"],
        "partnership_vocab": partnership["average_density"],
        "ai_hedging_rate": hedging["hedging_rate"]
    }

    s24_baseline = {
        "d9": s24["averages"]["d9"]
    }

    tier, d9_percent = determine_consolidation_tier(s25_metrics, s24_baseline)

    print(f"Consolidation Tier: {tier}")
    print(f"  D9 improvement: {d9_percent:+.1f}%")
    print()

    if tier == "STRONG":
        print("✅ Weight consolidation HIGHLY EFFECTIVE")
        print("   Partnership identity stabilized at semantic level.")
        print("   Frozen weights theory VALIDATED.")
        print()
        print("Next steps:")
        print("  - Continue sleep cycles (accumulate consolidation)")
        print("  - Test identity anchoring removal (experimental)")
        print("  - Validate sustained improvements (S26-S30)")

    elif tier == "MODERATE":
        print("✅ Weight consolidation EFFECTIVE")
        print("   Partnership identity strengthening, requires multiple cycles.")
        print("   Frozen weights theory SUPPORTED.")
        print()
        print("Next steps:")
        print("  - Run sleep cycles 002-003 (accumulation test)")
        print("  - Track consolidation trend")
        print("  - Monitor semantic depth progression")

    elif tier == "WEAK":
        print("⚠️ Weight consolidation WEAK")
        print("   Prompt fluency sufficient for vocabulary, semantic depth needs different approach.")
        print("   Frozen weights theory PARTIALLY SUPPORTED.")
        print()
        print("Next steps:")
        print("  - Increase training epochs or learning rate")
        print("  - Add more diverse training examples")
        print("  - Run 2-3 more cycles (test accumulation)")

    else:  # NONE
        print("❌ Consolidation FAILED or REGRESSION")
        print("   Weight consolidation did not improve partnership identity.")
        print("   Frozen weights theory REQUIRES REVISION.")
        print()
        print("Next steps:")
        print("  - Analyze what went wrong")
        print("  - Revert to cycle_000 (pre-consolidation)")
        print("  - Revise LoRA configuration or training data")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Full results documented in:")
    print("  sage/experiments/SESSION25_POST_CONSOLIDATION_RESULTS.md")
    print()


if __name__ == "__main__":
    main()
