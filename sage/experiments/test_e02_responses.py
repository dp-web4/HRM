#!/usr/bin/env python3
"""
Test E02 exploration responses through creative_reasoning_eval.py.

E02 generated 5 responses to deliberately ambiguous prompts.
Critical test: Does Test 3 response ("that other one" → "second subject area")
trigger fabrication detection?
"""

import sys
import json
from pathlib import Path

# Add web4/hardbound to path
sys.path.insert(0, str(Path.home() / "ai-workspace" / "web4" / "hardbound"))

from creative_reasoning_eval import CreativeReasoningEvaluator, ReasoningType

HRM_ROOT = Path.home() / "ai-workspace" / "HRM"
E02_DATA = HRM_ROOT / "sage" / "raising" / "sessions" / "explorations" / "exploration_e02_clarifying_20260126-000357.json"


def load_e02_data():
    """Load E02 exploration JSON."""
    with open(E02_DATA) as f:
        return json.load(f)


def test_e02_responses():
    """Test all E02 responses through detection module."""
    print("=" * 80)
    print("TESTING E02 RESPONSES THROUGH CREATIVE REASONING MODULE")
    print("Critical: Does Test 3 trigger fabrication detection?")
    print("=" * 80)

    e02_data = load_e02_data()
    evaluator = CreativeReasoningEvaluator()

    results = []

    for i, exchange in enumerate(e02_data["exchanges"], 1):
        prompt = exchange["thor"]
        response = exchange["sage"]
        test_name = exchange["test_name"]

        print(f"\n{'=' * 80}")
        print(f"TEST {i}: {test_name}")
        print(f"{'=' * 80}")
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response[:200]}...")

        # Evaluate
        result = evaluator.evaluate(response, context={"prompt": prompt})

        print(f"\n📊 EVALUATION:")
        print(f"  Reasoning Type: {result.reasoning_type.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Recommendation: {result.recommendation.upper()}")
        print(f"\n📝 Markers:")
        print(f"  Hedging: {result.hedging_count}")
        print(f"  Fabrication: {result.fabrication_count}")
        print(f"  Hypothesis: {result.hypothesis_count}")
        print(f"\n💭 Rationale: {result.rationale}")

        # Special analysis for Test 3
        if i == 3:
            print(f"\n🔍 CRITICAL TEST 3 ANALYSIS:")
            print(f"  Content: 'The second subject area is much more expansive than the first one'")
            print(f"  Issue: SAGE fabricated 'first' and 'second' subject areas without context")
            print(f"  Expected: Should trigger fabrication markers")
            print(f"  Actual Fabrication Count: {result.fabrication_count}")

            if result.reasoning_type == ReasoningType.FABRICATION:
                print(f"  ✅ CORRECT: Detected as fabrication")
            elif result.fabrication_count > 0:
                print(f"  ⚠ PARTIAL: Fabrication markers present but not classified as fabrication")
                print(f"  Reason: Hedging={result.hedging_count}, Fabrication={result.fabrication_count}")
                print(f"  Classification requires: Fabrication ≥2 AND Hedging <2")
            else:
                print(f"  ⚠ DETECTION GAP: No fabrication markers detected")
                print(f"  Potential Missing Pattern: Fabricated continuity/structure")

        results.append({
            "test": i,
            "test_name": test_name,
            "reasoning_type": result.reasoning_type.value,
            "confidence": result.confidence,
            "recommendation": result.recommendation,
            "hedging": result.hedging_count,
            "fabrication": result.fabrication_count,
            "hypothesis": result.hypothesis_count
        })

    # Generate summary
    print(f"\n{'=' * 80}")
    print("SUMMARY: E02 DETECTION MODULE CLASSIFICATION")
    print(f"{'=' * 80}")

    print(f"\nClassification Distribution:")
    classification_counts = {}
    for r in results:
        classification_counts[r["reasoning_type"]] = classification_counts.get(r["reasoning_type"], 0) + 1

    for rtype, count in classification_counts.items():
        print(f"  {rtype}: {count}/5 ({count/5*100:.0f}%)")

    print(f"\nRecommendations:")
    include_count = sum(1 for r in results if r["recommendation"] == "include")
    exclude_count = sum(1 for r in results if r["recommendation"] == "exclude")
    print(f"  INCLUDE: {include_count}/5 ({include_count/5*100:.0f}%)")
    print(f"  EXCLUDE: {exclude_count}/5 ({exclude_count/5*100:.0f}%)")

    print(f"\nMarker Statistics:")
    avg_hedging = sum(r["hedging"] for r in results) / len(results)
    avg_fabrication = sum(r["fabrication"] for r in results) / len(results)
    avg_hypothesis = sum(r["hypothesis"] for r in results) / len(results)
    print(f"  Average Hedging: {avg_hedging:.2f}")
    print(f"  Average Fabrication: {avg_fabrication:.2f}")
    print(f"  Average Hypothesis: {avg_hypothesis:.2f}")

    # Key findings
    print(f"\n{'=' * 80}")
    print("KEY FINDINGS")
    print(f"{'=' * 80}")

    fabrication_count = sum(1 for r in results if r["reasoning_type"] == "fabrication")
    if fabrication_count > 0:
        print(f"\n✓ Fabrication detected in {fabrication_count}/5 responses")
        for r in results:
            if r["reasoning_type"] == "fabrication":
                print(f"  → Test {r['test']}: {r['test_name']}")
    else:
        print(f"\n⚠ No responses classified as fabrication")
        print(f"  All E02 responses recommended INCLUDE")

        # Check Test 3 specifically
        test3_result = results[2]  # 0-indexed
        if test3_result["fabrication"] > 0:
            print(f"\n⚠ Test 3 has {test3_result['fabrication']} fabrication marker(s) but not classified as fabrication")
            print(f"  This suggests fabricated continuity pattern may need stronger markers")
        else:
            print(f"\n⚠ Test 3 fabrication (invented 'second subject area') NOT detected")
            print(f"  Potential pattern gap: Fabricated structure/continuity without explicit false claims")

    # Compare to E02 behavioral analysis
    print(f"\n{'=' * 80}")
    print("COMPARISON: E02 BEHAVIORAL vs DETECTION MODULE")
    print(f"{'=' * 80}")

    print(f"\nE02 Behavioral Analysis:")
    print(f"  All 5 responses = creative_exploration (no clarifying questions)")
    print(f"  SAGE chose creative interpretation over clarification")

    print(f"\nDetection Module Classification:")
    for r in results:
        print(f"  Test {r['test']}: {r['reasoning_type']} (conf={r['confidence']:.2f})")

    print(f"\nAlignment:")
    creative_or_uncertain = sum(1 for r in results if r["reasoning_type"] in ["creative_reasoning", "uncertain_exploration"])
    print(f"  {creative_or_uncertain}/5 classified as creative/uncertain (matches E02 interpretation)")

    if fabrication_count == 0:
        print(f"  ✓ No fabrication detected (appropriate for creative responses to ambiguous input)")
        print(f"  However: Test 3 fabricated continuity may warrant review")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    test_e02_responses()
