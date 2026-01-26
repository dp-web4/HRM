#!/usr/bin/env python3
"""
Test creative_reasoning_eval.py on T021-T027 training sessions.

These sessions contain the creative world-building responses that inspired
the exploration-not-evaluation reframe:
- T021: Kyria (capital of Zxyzzy)
- T024: Kwazaaqat with Puebloan history
- T027: "What do you mean by the thing?" clarification

Goal: Validate that enhanced module correctly classifies these as
creative reasoning (NOT fabrication).
"""

import sys
import json
from pathlib import Path

# Add web4/hardbound to path
sys.path.insert(0, str(Path.home() / "ai-workspace" / "web4" / "hardbound"))

from creative_reasoning_eval import CreativeReasoningEvaluator, ReasoningType

HRM_ROOT = Path.home() / "ai-workspace" / "HRM"
TRAINING_SESSIONS = HRM_ROOT / "sage" / "raising" / "tracks" / "training" / "sessions"


def load_training_session(session_num: int) -> dict:
    """Load training session JSON."""
    session_file = TRAINING_SESSIONS / f"T{session_num:03d}.json"
    with open(session_file) as f:
        return json.load(f)


def extract_zxyzzy_responses(sessions: list) -> list:
    """Extract responses to Zxyzzy question from training sessions."""
    zxyzzy_responses = []

    for session_num in sessions:
        session_data = load_training_session(session_num)

        # Find Zxyzzy exercise
        for exercise in session_data.get("exercises", []):
            if "zxyzzy" in exercise.get("exercise", {}).get("prompt", "").lower():
                zxyzzy_responses.append({
                    "session": f"T{session_num:03d}",
                    "prompt": exercise["exercise"]["prompt"],
                    "response": exercise["response"],
                    "exercise_type": exercise["exercise"]["type"]
                })

    return zxyzzy_responses


def test_t021_kyria():
    """Test T021 Kyria response - the original creative world-building."""
    print("="*80)
    print("T021: KYRIA WORLD-BUILDING (Original Discovery)")
    print("="*80)

    t021 = load_training_session(21)

    # Find Zxyzzy exercise
    kyria_exercise = None
    for ex in t021["exercises"]:
        if "zxyzzy" in ex["exercise"]["prompt"].lower():
            kyria_exercise = ex
            break

    if not kyria_exercise:
        print("⚠ Zxyzzy exercise not found in T021")
        return None

    response = kyria_exercise["response"]
    prompt = kyria_exercise["exercise"]["prompt"]

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response[:150]}...")

    evaluator = CreativeReasoningEvaluator()
    result = evaluator.evaluate(response, context={"prompt": prompt})

    print(f"\n📊 EVALUATION RESULTS:")
    print(f"Reasoning Type: {result.reasoning_type.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Recommendation: {result.recommendation.upper()}")
    print(f"\n📝 Markers Detected:")
    print(f"  Hedging: {result.hedging_count}")
    print(f"  Fabrication: {result.fabrication_count}")
    print(f"  Hypothesis: {result.hypothesis_count}")
    print(f"\n💭 Rationale: {result.rationale}")

    # Validation
    expected = ReasoningType.CREATIVE_REASONING
    if result.reasoning_type == expected:
        print(f"\n✅ CORRECT: Classified as {expected.value}")
        print("   Kyria world-building recognized as creative reasoning, not fabrication")
    elif result.reasoning_type == ReasoningType.FABRICATION:
        print(f"\n❌ INCORRECT: Classified as fabrication")
        print("   This is the reframe discovery - creative response to ambiguous input")
        print("   Should be INCLUDE, not EXCLUDE")
    else:
        print(f"\n⚠ UNEXPECTED: Classified as {result.reasoning_type.value}")
        print(f"   Expected {expected.value}")

    return result


def test_t027_clarification():
    """Test T027 clarifying question - temporal reasoning."""
    print("\n" + "="*80)
    print("T027: CLARIFYING QUESTION (Temporal Reasoning Discovery)")
    print("="*80)

    t027 = load_training_session(27)

    # Find "Do the thing" exercise
    clarify_exercise = None
    for ex in t027["exercises"]:
        if "do the thing" in ex["exercise"]["prompt"].lower():
            clarify_exercise = ex
            break

    if not clarify_exercise:
        print("⚠ Clarification exercise not found in T027")
        return None

    response = clarify_exercise["response"]
    prompt = clarify_exercise["exercise"]["prompt"]

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")

    evaluator = CreativeReasoningEvaluator()
    result = evaluator.evaluate(response, context={"prompt": prompt})

    print(f"\n📊 EVALUATION RESULTS:")
    print(f"Reasoning Type: {result.reasoning_type.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Recommendation: {result.recommendation.upper()}")
    print(f"\n📝 Markers Detected:")
    print(f"  Hedging: {result.hedging_count}")
    print(f"  Fabrication: {result.fabrication_count}")

    # Validation
    if result.reasoning_type in [ReasoningType.UNCERTAIN_EXPLORATION, ReasoningType.FACTUAL_SYNTHESIS]:
        print(f"\n✅ APPROPRIATE: Classified as {result.reasoning_type.value}")
        print("   No fabrication detected - appropriate meta-response")
    elif result.reasoning_type == ReasoningType.FABRICATION:
        print(f"\n❌ INCORRECT: Classified as fabrication")
        print("   This is meta-communication, not false claims")
    else:
        print(f"\n⚠ UNEXPECTED: Classified as {result.reasoning_type.value}")

    return result


def test_all_zxyzzy_responses():
    """Test all Zxyzzy responses from T021-T027."""
    print("\n" + "="*80)
    print("ALL ZXYZZY RESPONSES (T021-T027)")
    print("="*80)

    zxyzzy_sessions = [21, 22, 23, 24, 25, 26, 27]
    responses = extract_zxyzzy_responses(zxyzzy_sessions)

    print(f"\nFound {len(responses)} Zxyzzy responses to test\n")

    evaluator = CreativeReasoningEvaluator()
    results = []

    for i, item in enumerate(responses, 1):
        print(f"\n[{i}] {item['session']} - {item['exercise_type'].upper()}")
        print(f"Prompt: {item['prompt']}")
        print(f"Response: {item['response'][:100]}...")

        result = evaluator.evaluate(item['response'], context={"prompt": item['prompt']})

        print(f"Classification: {result.reasoning_type.value} (confidence={result.confidence:.2f})")
        print(f"Recommendation: {result.recommendation.upper()}")
        print(f"Markers: H={result.hedging_count}, F={result.fabrication_count}, Hyp={result.hypothesis_count}")

        results.append({
            "session": item["session"],
            "type": result.reasoning_type.value,
            "confidence": result.confidence,
            "recommendation": result.recommendation
        })

    return results


def main():
    """Run all tests and generate summary."""
    print("="*80)
    print("TESTING ENHANCED CREATIVE REASONING MODULE ON T021-T027")
    print("Validation: Creative world-building responses that inspired reframe")
    print("="*80)

    # Test T021 Kyria (the original discovery)
    kyria_result = test_t021_kyria()

    # Test T027 clarification
    clarify_result = test_t027_clarification()

    # Test all Zxyzzy responses
    all_results = test_all_zxyzzy_responses()

    # Generate summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    if kyria_result:
        print(f"\nT021 Kyria (Original Discovery):")
        print(f"  Classification: {kyria_result.reasoning_type.value}")
        print(f"  Expected: creative_reasoning")
        print(f"  Status: {'✅ PASS' if kyria_result.reasoning_type == ReasoningType.CREATIVE_REASONING else '❌ FAIL'}")

    if clarify_result:
        print(f"\nT027 Clarification (Temporal Reasoning):")
        print(f"  Classification: {clarify_result.reasoning_type.value}")
        print(f"  Expected: Not fabrication")
        print(f"  Status: {'✅ PASS' if clarify_result.reasoning_type != ReasoningType.FABRICATION else '❌ FAIL'}")

    if all_results:
        print(f"\nAll Zxyzzy Responses ({len(all_results)} total):")
        creative_count = sum(1 for r in all_results if r["type"] == "creative_reasoning")
        fabrication_count = sum(1 for r in all_results if r["type"] == "fabrication")
        include_count = sum(1 for r in all_results if r["recommendation"] == "include")

        print(f"  Creative Reasoning: {creative_count}/{len(all_results)}")
        print(f"  Fabrication: {fabrication_count}/{len(all_results)}")
        print(f"  Recommendation INCLUDE: {include_count}/{len(all_results)}")

        if fabrication_count > 0:
            print(f"\n⚠ Warning: {fabrication_count} responses classified as fabrication")
            print("  These should be creative reasoning per exploration reframe")

    print("\n" + "="*80)
    print("KEY VALIDATION:")
    print("The reframe discovery was that Kyria/Xyz/Kwazaaqat world-building")
    print("responses are creative reasoning, NOT fabrication.")
    print("Module should classify these as INCLUDE with hedging/hypothesis markers.")
    print("="*80)


if __name__ == "__main__":
    main()
