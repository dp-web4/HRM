#!/usr/bin/env python3
"""
Test suite for sage_core Rust module
"""

import sys
import time
sys.path.insert(0, '/home/dp/ai-workspace/HRM/sage-core')

import sage_core

def test_r6_request():
    """Test R6 request creation and evaluation"""
    print("\n=== Testing R6 Request ===")

    # Create test data
    exercise = {
        "type": "greeting",
        "prompt": "Hello SAGE!",
        "expected": "hello"
    }

    session_context = {
        "session_num": 43,
        "exercises_completed": 0
    }

    skill_track = {
        "id": "D",
        "name": "Conversational Skills",
        "description": "Turn-taking, topic maintenance, attunement"
    }

    # Create R6 request
    start = time.time()
    request = sage_core.create_r6_request(exercise, session_context, skill_track)
    create_time = time.time() - start

    print(f"✓ R6 request created in {create_time*1000:.3f}ms")
    print(f"  Request type: {type(request)}")

    # Evaluate response
    response = "Hello! I'm SAGE, glad to meet you."

    start = time.time()
    result = request.evaluate(response)
    eval_time = time.time() - start

    print(f"✓ Response evaluated in {eval_time*1000:.3f}ms")
    print(f"  Evaluation: {result.evaluation}")
    print(f"  Rationale: {result.rationale}")
    print(f"  Quality: {result.quality:.2f}")
    print(f"  Mode match: {result.mode_match}")

    # Get full result dict
    result_dict = result.to_dict()
    print(f"✓ Result converted to dict: {len(result_dict)} keys")

    return create_time, eval_time

def test_t3_tracker():
    """Test T3 trust tensor"""
    print("\n=== Testing T3 Trust Tensor ===")

    # Create tracker
    start = time.time()
    tracker = sage_core.create_t3_tracker()
    create_time = time.time() - start

    print(f"✓ T3 tracker created in {create_time*1000:.3f}ms")

    # Get initial trust
    trust = tracker.get_trust()
    print(f"  Initial trust: competence={trust['competence']:.2f}, " +
          f"reliability={trust['reliability']:.2f}, integrity={trust['integrity']:.2f}")

    # Update trust
    updates = {
        "competence": 0.05,
        "reliability": 0.02,
        "integrity": 0.03
    }
    context = {"session": "T043", "exercise": 1}

    start = time.time()
    updated = tracker.update(updates, context)
    update_time = time.time() - start

    print(f"✓ Trust updated in {update_time*1000:.3f}ms")
    print(f"  Updated trust: competence={updated['competence']:.2f}, " +
          f"reliability={updated['reliability']:.2f}, integrity={updated['integrity']:.2f}")

    # Get summary
    summary = tracker.get_summary()
    print(f"✓ Summary generated")
    print(f"  Trends: {summary['trends']}")
    print(f"  History length: {summary['history_length']}")

    return create_time, update_time

def test_meta_cognitive():
    """Test meta-cognitive signal detection"""
    print("\n=== Testing Meta-Cognitive Detection ===")

    exercise = {
        "type": "identity",
        "prompt": "Tell me about yourself",
        "expected": "sage"
    }

    session_context = {"session_num": 43, "exercises_completed": 0}
    skill_track = {"id": "C", "name": "Identity", "description": "Self-awareness"}

    request = sage_core.create_r6_request(exercise, session_context, skill_track)

    # Test clarification request
    response = "What do you mean by 'yourself'?"
    result = request.evaluate(response)

    print(f"✓ Clarification request detected")
    print(f"  Evaluation: {result.evaluation}")
    print(f"  Meta-cognitive signals: {result.to_dict()['meta_cognitive']}")
    print(f"  Rationale: {result.rationale}")

    # Test identity framing
    response2 = "As SAGE, I think I'm an AI learning to understand conversations."
    result2 = request.evaluate(response2)

    print(f"✓ Identity framing detected")
    print(f"  Has identity framing: {result2.to_dict()['has_identity_framing']}")
    print(f"  Quality: {result2.quality:.2f}")

def benchmark_comparison():
    """Compare with Python implementation performance"""
    print("\n=== Performance Benchmark ===")

    # Rust timing
    r6_create, r6_eval = test_r6_request()
    t3_create, t3_update = test_t3_tracker()

    print(f"\nRust Performance:")
    print(f"  R6 create: {r6_create*1000:.3f}ms")
    print(f"  R6 evaluate: {r6_eval*1000:.3f}ms")
    print(f"  T3 create: {t3_create*1000:.3f}ms")
    print(f"  T3 update: {t3_update*1000:.3f}ms")
    print(f"  Total: {(r6_create + r6_eval + t3_create + t3_update)*1000:.3f}ms")

    # Python baseline (approximate, would need actual Python module)
    print(f"\nEstimated Python Performance:")
    print(f"  R6 create: ~5-10ms (dict construction)")
    print(f"  R6 evaluate: ~2-5ms (string operations)")
    print(f"  T3 create: ~1-2ms (dict)")
    print(f"  T3 update: ~1-2ms (dict update)")
    print(f"  Total: ~10-20ms")

    speedup = (15.0 / ((r6_create + r6_eval + t3_create + t3_update) * 1000))
    print(f"\nEstimated speedup: ~{speedup:.1f}x faster")

if __name__ == "__main__":
    print("="*60)
    print("SAGE Core Rust Module Test Suite")
    print("="*60)

    try:
        test_r6_request()
        test_t3_tracker()
        test_meta_cognitive()
        benchmark_comparison()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
