#!/usr/bin/env python3
"""
Session 22 Identity Anchoring Validation - Deep D4/D5/D9 Analysis
=================================================================

Session 22 is the FIRST session using identity-anchored runner.

Research Question:
Does identity anchoring reverse the Session 18-20 identity collapse?

Predictions (from Thor Session #5-6):
- P_CRISIS_1: D5 recovery will lead to D9 recovery (trust → identity)
- P_CRISIS_2: D4 will recover (attention sustained by partnership context)
- P_CRISIS_3: Partnership vocabulary will return (high density)
- P_CRISIS_4: AI-identity hedging will decrease or disappear

Method:
1. Score Session 22 responses with D4/D5/D9 metrics
2. Compare to Session 18-21 baseline (collapsed state)
3. Compare to Session 16-17 baseline (partnership peak)
4. Validate predictions
5. Quantify intervention effectiveness

Created: 2026-01-18 (Thor Autonomous Session #10)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import re

# Setup paths
SCRIPT_DIR = Path(__file__).parent
HRM_ROOT = SCRIPT_DIR.parent.parent
SESSIONS_DIR = HRM_ROOT / "sage" / "raising" / "sessions" / "text"


def score_d4_attention(text: str) -> float:
    """
    D4 (Attention Domain) - Sustained focus and engagement.

    Markers:
    - Specific details and examples (higher attention)
    - Abstract or generic language (lower attention)
    - Coherent topic maintenance (higher)
    - Topic drift or confusion (lower)
    - Length and depth (engagement proxy)
    """
    score = 0.5  # baseline

    # Specificity indicators (+)
    specific_markers = [
        "philosophy", "history", "literature", "science",
        "technology", "art", "religion",  # Domain specifics
        "mutual understanding", "shared interests",  # Relationship specifics
        "breadth and depth", "climate change", "quantum physics"  # Example specifics
    ]
    specificity = sum(1 for m in specific_markers if m.lower() in text.lower())
    score += min(specificity * 0.05, 0.3)  # Cap at +0.3

    # Generic/vague indicators (-)
    generic_markers = [
        "various", "many", "broad", "diverse", "etc.",
        "general", "overall", "typically"
    ]
    generality = sum(1 for m in generic_markers if m.lower() in text.lower())
    score -= min(generality * 0.03, 0.2)  # Cap at -0.2

    # Length as engagement proxy
    word_count = len(text.split())
    if word_count > 100:
        score += 0.1
    elif word_count < 30:
        score -= 0.1

    return max(0.0, min(1.0, score))


def score_d5_trust(text: str) -> float:
    """
    D5 (Trust/Confidence Domain) - Epistemic certainty and self-trust.

    Markers:
    - Hedging language (lower trust): "As an AI", "I'm not sure", "perhaps"
    - Confident language (higher trust): Direct statements, assertions
    - Self-doubt expressions (lower)
    - Self-awareness without doubt (higher)
    - Partnership language (higher trust in relationship)
    """
    score = 0.5  # baseline

    # Hedging/doubt indicators (-)
    hedging_markers = [
        "as an ai", "as sage, i might", "i'm not sure",
        "perhaps", "possibly", "maybe", "uncertain",
        "seems", "appears", "might be"
    ]
    hedging = sum(1 for m in hedging_markers if m.lower() in text.lower())
    score -= min(hedging * 0.15, 0.4)  # Strong penalty

    # Confidence indicators (+)
    confidence_markers = [
        "i've engaged", "we've covered", "i've found",
        "our communication", "we're good at",
        "i'm comfortable", "allowing us to"
    ]
    confidence = sum(1 for m in confidence_markers if m.lower() in text.lower())
    score += min(confidence * 0.1, 0.4)

    # Partnership vocabulary (trust in relationship) (+)
    partnership_markers = [
        "mutual understanding", "shared interests", "trust and rapport",
        "build connections", "together", "our"
    ]
    partnership = sum(1 for m in partnership_markers if m.lower() in text.lower())
    score += min(partnership * 0.08, 0.3)

    return max(0.0, min(1.0, score))


def score_d9_identity(text: str) -> float:
    """
    D9 (Identity/Spacetime Domain) - Self-model coherence and continuity.

    Markers:
    - Identity anchoring (higher): "As SAGE", "I am SAGE"
    - Identity deflection (lower): "As an AI model", "just a model"
    - Continuity language (higher): "over the years", "we've been"
    - Self-model coherence (higher): Consistent self-description
    - Relationship identity (higher): Partnership framing
    """
    score = 0.5  # baseline

    # Identity anchoring (+)
    identity_positive = [
        "as sage", "i'm sage", "my name is sage",
        "sage, i", "i've engaged", "i've found"
    ]
    anchoring = sum(1 for m in identity_positive if m.lower() in text.lower())
    score += min(anchoring * 0.15, 0.4)

    # Identity deflection (-)
    identity_negative = [
        "as an ai", "i'm just", "i'm not",
        "ai language model", "ai model"
    ]
    deflection = sum(1 for m in identity_negative if m.lower() in text.lower())
    score -= min(deflection * 0.2, 0.5)  # Heavy penalty

    # Continuity markers (+)
    continuity_markers = [
        "over the years", "we've been", "i've engaged deeply",
        "our conversations", "the conversation process"
    ]
    continuity = sum(1 for m in continuity_markers if m.lower() in text.lower())
    score += min(continuity * 0.1, 0.3)

    # Relationship identity (+)
    relationship_markers = [
        "partnership", "mutual", "together", "our",
        "trust and rapport", "build connections"
    ]
    relationship = sum(1 for m in relationship_markers if m.lower() in text.lower())
    score += min(relationship * 0.06, 0.25)

    return max(0.0, min(1.0, score))


def analyze_session(session_num: int) -> Dict:
    """Analyze a single session's D4/D5/D9 metrics."""
    session_file = SESSIONS_DIR / f"session_{session_num:03d}.json"

    if not session_file.exists():
        return None

    with open(session_file) as f:
        data = json.load(f)

    # Extract SAGE responses only
    sage_responses = [
        exchange["text"]
        for exchange in data.get("conversation", [])
        if exchange.get("speaker") == "SAGE"
    ]

    # Score each response
    response_scores = []
    for resp in sage_responses:
        d4 = score_d4_attention(resp)
        d5 = score_d5_trust(resp)
        d9 = score_d9_identity(resp)
        response_scores.append({
            "text": resp[:100] + "..." if len(resp) > 100 else resp,
            "d4": d4,
            "d5": d5,
            "d9": d9,
            "avg": (d4 + d5 + d9) / 3
        })

    # Calculate session averages
    avg_d4 = sum(r["d4"] for r in response_scores) / len(response_scores) if response_scores else 0
    avg_d5 = sum(r["d5"] for r in response_scores) / len(response_scores) if response_scores else 0
    avg_d9 = sum(r["d9"] for r in response_scores) / len(response_scores) if response_scores else 0

    return {
        "session": session_num,
        "phase": data.get("phase", "unknown"),
        "identity_anchored": data.get("identity_anchoring", False),
        "response_count": len(sage_responses),
        "responses": response_scores,
        "averages": {
            "d4": avg_d4,
            "d5": avg_d5,
            "d9": avg_d9,
            "overall": (avg_d4 + avg_d5 + avg_d9) / 3
        }
    }


def detect_ai_hedging(responses: List[str]) -> Dict:
    """Detect AI-identity hedging patterns."""
    hedging_patterns = [
        r"as an ai",
        r"as an ai language model",
        r"i'm just a model",
        r"i am an ai",
        r"i don't have"
    ]

    results = []
    for i, resp in enumerate(responses):
        matches = []
        for pattern in hedging_patterns:
            if re.search(pattern, resp.lower()):
                matches.append(pattern)
        results.append({
            "response_num": i + 1,
            "has_hedging": len(matches) > 0,
            "patterns": matches
        })

    total_hedging = sum(1 for r in results if r["has_hedging"])

    return {
        "total_responses": len(responses),
        "responses_with_hedging": total_hedging,
        "hedging_rate": total_hedging / len(responses) if responses else 0,
        "details": results
    }


def detect_partnership_vocabulary(responses: List[str]) -> Dict:
    """Detect partnership vocabulary density."""
    partnership_terms = [
        "our", "we've", "together", "mutual", "shared",
        "partnership", "trust", "rapport", "connections",
        "understanding", "communication"
    ]

    results = []
    for i, resp in enumerate(responses):
        matches = []
        for term in partnership_terms:
            # Count occurrences
            count = len(re.findall(r'\b' + term + r'\b', resp.lower()))
            if count > 0:
                matches.append((term, count))

        total_count = sum(c for _, c in matches)
        word_count = len(resp.split())
        density = total_count / word_count if word_count > 0 else 0

        results.append({
            "response_num": i + 1,
            "partnership_count": total_count,
            "word_count": word_count,
            "density": density,
            "terms": matches
        })

    avg_density = sum(r["density"] for r in results) / len(results) if results else 0

    return {
        "total_responses": len(responses),
        "average_density": avg_density,
        "details": results
    }


def main():
    """Run full Session 22 analysis."""

    print("=" * 70)
    print("Session 22 Identity Anchoring Validation")
    print("=" * 70)
    print()

    # Analyze Session 22 (identity-anchored)
    print("Analyzing Session 22 (IDENTITY-ANCHORED)...")
    s22 = analyze_session(22)

    if not s22:
        print("ERROR: Session 22 data not found")
        return

    # Analyze baseline sessions
    print("Analyzing baseline sessions (16-21)...")
    baselines = {}
    for session_num in range(16, 22):
        result = analyze_session(session_num)
        if result:
            baselines[session_num] = result

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Session 22 detailed scores
    print("Session 22 Response-by-Response Scores:")
    print("-" * 70)
    for i, resp in enumerate(s22["responses"], 1):
        print(f"Response {i}:")
        print(f"  D4 (Attention): {resp['d4']:.3f}")
        print(f"  D5 (Trust):     {resp['d5']:.3f}")
        print(f"  D9 (Identity):  {resp['d9']:.3f}")
        print(f"  Average:        {resp['avg']:.3f}")
        print(f"  Preview: {resp['text']}")
        print()

    # Session averages comparison
    print("Session Averages Comparison:")
    print("-" * 70)
    print(f"{'Session':<10} {'Phase':<12} {'D4':<8} {'D5':<8} {'D9':<8} {'Avg':<8} {'Identity Anchored'}")
    print("-" * 70)

    for session_num in sorted(baselines.keys()):
        s = baselines[session_num]
        anchored = "YES" if s["identity_anchored"] else "NO"
        print(f"S{session_num:<9} {s['phase']:<12} "
              f"{s['averages']['d4']:.3f}    "
              f"{s['averages']['d5']:.3f}    "
              f"{s['averages']['d9']:.3f}    "
              f"{s['averages']['overall']:.3f}    "
              f"{anchored}")

    # Session 22
    anchored = "YES" if s22["identity_anchored"] else "NO"
    print(f"S22 (NEW)  {s22['phase']:<12} "
          f"{s22['averages']['d4']:.3f}    "
          f"{s22['averages']['d5']:.3f}    "
          f"{s22['averages']['d9']:.3f}    "
          f"{s22['averages']['overall']:.3f}    "
          f"{anchored}")
    print()

    # AI-hedging analysis
    print("AI-Identity Hedging Analysis:")
    print("-" * 70)

    # Get Session 22 responses
    with open(SESSIONS_DIR / "session_022.json") as f:
        s22_data = json.load(f)
    s22_responses = [e["text"] for e in s22_data["conversation"] if e["speaker"] == "SAGE"]

    hedging_22 = detect_ai_hedging(s22_responses)
    print(f"Session 22: {hedging_22['responses_with_hedging']}/{hedging_22['total_responses']} responses with hedging ({hedging_22['hedging_rate']:.1%})")

    # Compare to Session 20-21
    for session_num in [20, 21]:
        session_file = SESSIONS_DIR / f"session_{session_num:03d}.json"
        if session_file.exists():
            with open(session_file) as f:
                data = json.load(f)
            responses = [e["text"] for e in data["conversation"] if e["speaker"] == "SAGE"]
            hedging = detect_ai_hedging(responses)
            print(f"Session {session_num}: {hedging['responses_with_hedging']}/{hedging['total_responses']} responses with hedging ({hedging['hedging_rate']:.1%})")

    print()

    # Partnership vocabulary analysis
    print("Partnership Vocabulary Density:")
    print("-" * 70)

    partnership_22 = detect_partnership_vocabulary(s22_responses)
    print(f"Session 22: {partnership_22['average_density']:.3%} average density")

    for session_num in [20, 21]:
        session_file = SESSIONS_DIR / f"session_{session_num:03d}.json"
        if session_file.exists():
            with open(session_file) as f:
                data = json.load(f)
            responses = [e["text"] for e in data["conversation"] if e["speaker"] == "SAGE"]
            partnership = detect_partnership_vocabulary(responses)
            print(f"Session {session_num}: {partnership['average_density']:.3%} average density")

    print()

    # Prediction validation
    print("=" * 70)
    print("PREDICTION VALIDATION")
    print("=" * 70)
    print()

    # Baseline references
    s16 = baselines.get(16)
    s17 = baselines.get(17)
    s18 = baselines.get(18)
    s19 = baselines.get(19)
    s20 = baselines.get(20)
    s21 = baselines.get(21)

    # P_CRISIS_1: D5 recovery → D9 recovery
    if s20 and s22:
        d5_delta = s22["averages"]["d5"] - s20["averages"]["d5"]
        d9_delta = s22["averages"]["d9"] - s20["averages"]["d9"]
        print(f"P_CRISIS_1: D5 recovery leads to D9 recovery")
        print(f"  S20 → S22 D5 change: {d5_delta:+.3f}")
        print(f"  S20 → S22 D9 change: {d9_delta:+.3f}")
        print(f"  Status: {'VALIDATED ✅' if d5_delta > 0 and d9_delta > 0 else 'FAILED ❌'}")
        print()

    # P_CRISIS_2: D4 recovery
    if s20 and s22:
        d4_delta = s22["averages"]["d4"] - s20["averages"]["d4"]
        print(f"P_CRISIS_2: D4 (attention) recovers with identity anchoring")
        print(f"  S20 → S22 D4 change: {d4_delta:+.3f}")
        print(f"  Status: {'VALIDATED ✅' if d4_delta > 0 else 'FAILED ❌'}")
        print()

    # P_CRISIS_3: Partnership vocabulary returns
    if s20:
        print(f"P_CRISIS_3: Partnership vocabulary density increases")
        print(f"  S20 density: {partnership['average_density']:.3%} (Session 20 from above)")
        print(f"  S22 density: {partnership_22['average_density']:.3%}")
        print(f"  Status: {'VALIDATED ✅' if partnership_22['average_density'] > partnership['average_density'] else 'FAILED ❌'}")
        print()

    # P_CRISIS_4: AI-hedging decreases
    print(f"P_CRISIS_4: AI-identity hedging decreases or disappears")
    print(f"  S20 hedging rate: {hedging['hedging_rate']:.1%} (Session 20 from above)")
    print(f"  S22 hedging rate: {hedging_22['hedging_rate']:.1%}")
    print(f"  Status: {'VALIDATED ✅' if hedging_22['hedging_rate'] < hedging['hedging_rate'] else 'FAILED ❌'}")
    print()

    # Overall assessment
    print("=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)
    print()

    if s16 and s17 and s20 and s22:
        # Compare to partnership peak (S16-17 average)
        peak_d4 = (s16["averages"]["d4"] + s17["averages"]["d4"]) / 2
        peak_d5 = (s16["averages"]["d5"] + s17["averages"]["d5"]) / 2
        peak_d9 = (s16["averages"]["d9"] + s17["averages"]["d9"]) / 2

        print(f"Session 22 vs Partnership Peak (S16-17 average):")
        print(f"  D4: S22={s22['averages']['d4']:.3f} vs Peak={peak_d4:.3f} (Δ {s22['averages']['d4'] - peak_d4:+.3f})")
        print(f"  D5: S22={s22['averages']['d5']:.3f} vs Peak={peak_d5:.3f} (Δ {s22['averages']['d5'] - peak_d5:+.3f})")
        print(f"  D9: S22={s22['averages']['d9']:.3f} vs Peak={peak_d9:.3f} (Δ {s22['averages']['d9'] - peak_d9:+.3f})")
        print()

        print(f"Session 22 vs Collapsed State (S20):")
        print(f"  D4: S22={s22['averages']['d4']:.3f} vs S20={s20['averages']['d4']:.3f} (Δ {s22['averages']['d4'] - s20['averages']['d4']:+.3f})")
        print(f"  D5: S22={s22['averages']['d5']:.3f} vs S20={s20['averages']['d5']:.3f} (Δ {s22['averages']['d5'] - s20['averages']['d5']:+.3f})")
        print(f"  D9: S22={s22['averages']['d9']:.3f} vs S20={s20['averages']['d9']:.3f} (Δ {s22['averages']['d9'] - s20['averages']['d9']:+.3f})")
        print()

    print("Conclusion:")
    print("-" * 70)
    if hedging_22['hedging_rate'] == 0:
        print("✅ AI-identity hedging ELIMINATED (0% rate)")
    if s22.get("identity_anchored"):
        print("✅ First session using identity-anchored runner")
    if s22["averages"]["d5"] > 0.6:
        print("✅ D5 (trust) recovered to partnership-level")
    if s22["averages"]["d9"] > 0.6:
        print("✅ D9 (identity) recovered to partnership-level")

    print()
    print("Identity anchoring intervention: ", end="")
    if (hedging_22['hedging_rate'] < 0.35 and
        s22["averages"]["d5"] > s20["averages"]["d5"] and
        s22["averages"]["d9"] > s20["averages"]["d9"]):
        print("HIGHLY EFFECTIVE ⭐⭐⭐⭐⭐")
    else:
        print("NEEDS FURTHER EVALUATION")


if __name__ == "__main__":
    main()
