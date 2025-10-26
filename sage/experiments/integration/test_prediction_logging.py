#!/usr/bin/env python3
"""
Test Prediction Logging System

Validates that the prediction triplet capture system works correctly:
1. Hallucination detection extracts model response + predicted user response
2. Prediction logger stores pending prediction
3. Actual user response completes the triplet
4. Triplet is saved to JSONL with similarity calculation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.integration.streaming_responder import StreamingResponder
from cognitive.prediction_logger import PredictionLogger
import json
import time


def test_hallucination_extraction():
    """Test that hallucinations are correctly extracted."""
    print("\n" + "="*80)
    print("TEST 1: Hallucination Extraction")
    print("="*80)

    responder = StreamingResponder(model_name="Qwen/Qwen2.5-0.5B-Instruct")

    # Test cases with expected hallucinations
    test_cases = [
        {
            'full_response': "Yes, happiness often feels like a sense of fulfillment.",
            'chunk': "\nUser: How",
            'expected': True,
            'description': "User: marker in chunk"
        },
        {
            'full_response': "Good. We've covered emotions.",
            'chunk': "\nAssistant: Let me",
            'expected': True,
            'description': "Assistant: marker in chunk"
        },
        {
            'full_response': "That's an interesting question.",
            'chunk': " about emotions.",
            'expected': False,
            'description': "No hallucination"
        }
    ]

    all_passed = True
    for i, case in enumerate(test_cases, 1):
        result = responder._extract_hallucination(case['chunk'], case['full_response'])
        detected = result is not None

        passed = detected == case['expected']
        all_passed = all_passed and passed

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"\n{status} Test {i}: {case['description']}")
        print(f"   Expected: {'Hallucination' if case['expected'] else 'No hallucination'}")
        print(f"   Detected: {'Hallucination' if detected else 'No hallucination'}")

        if result:
            print(f"   Model response: '{result['model_response'][:60]}...'")
            print(f"   Predicted user: '{result['predicted_user_response'][:60]}...'")

    return all_passed


def test_prediction_triplet_creation():
    """Test complete prediction triplet creation."""
    print("\n" + "="*80)
    print("TEST 2: Prediction Triplet Creation")
    print("="*80)

    # Create temporary logger
    import tempfile
    temp_dir = tempfile.mkdtemp()
    logger = PredictionLogger(log_dir=temp_dir)

    print(f"\n📊 Using temporary log dir: {temp_dir}")

    # Step 1: Capture hallucination
    print("\n1️⃣ Capturing hallucination...")
    logger.capture_hallucination(
        model_response="Yes, happiness often feels like a sense of fulfillment.",
        predicted_user_response="User: How does that feel?",
        context={'test': True, 'scenario': 'emotions'}
    )

    print(f"   ✓ Pending prediction created")
    print(f"   ✓ Model response: '{logger.pending_prediction['model_response'][:60]}...'")
    print(f"   ✓ Predicted user: '{logger.pending_prediction['predicted_user_response'][:60]}...'")

    # Step 2: User actually responds
    print("\n2️⃣ Logging actual user response...")
    actual_response = "Tell me more about fulfillment."
    logger.log_actual_response(actual_response)

    print(f"   ✓ Actual response: '{actual_response}'")

    # Step 3: Verify triplet was saved
    print("\n3️⃣ Verifying saved triplet...")
    session_file = logger.session_file

    if session_file.exists():
        with open(session_file, 'r') as f:
            triplet = json.loads(f.read().strip())

        print(f"   ✓ Triplet saved to: {session_file}")
        print(f"   ✓ Model response: '{triplet['model_response'][:60]}...'")
        print(f"   ✓ Predicted user: '{triplet['predicted_user_response'][:60]}...'")
        print(f"   ✓ Actual user: '{triplet['actual_user_response'][:60]}...'")
        print(f"   ✓ Similarity: {triplet['similarity']:.2%}")
        print(f"   ✓ Status: {triplet['status']}")

        # Validate structure
        required_fields = [
            'timestamp', 'datetime', 'model_response',
            'predicted_user_response', 'actual_user_response',
            'similarity', 'status', 'context'
        ]

        missing = [f for f in required_fields if f not in triplet]
        if missing:
            print(f"   ❌ Missing fields: {missing}")
            return False
        else:
            print(f"   ✅ All required fields present")
            return True
    else:
        print(f"   ❌ Session file not created!")
        return False


def test_streaming_with_prediction_logger():
    """Test streaming generation with prediction logger integration."""
    print("\n" + "="*80)
    print("TEST 3: Streaming Generation with Prediction Logging")
    print("="*80)

    # Create temporary logger
    import tempfile
    temp_dir = tempfile.mkdtemp()
    logger = PredictionLogger(log_dir=temp_dir)

    # Create responder with prediction logger
    print("\n📊 Creating streaming responder with prediction logger...")
    responder = StreamingResponder(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        prediction_logger=logger,
        max_new_tokens=256,
        temperature=0.7
    )

    print("   ✓ Responder initialized with prediction logging")

    # Generate response (might trigger hallucination)
    print("\n🔄 Generating response (might hallucinate)...")
    result = responder.generate_response_streaming(
        user_text="How are you feeling?",
        system_prompt="You are SAGE, a helpful assistant."
    )

    print(f"\n✅ Generation complete:")
    print(f"   Response: '{result['full_response'][:80]}...'")
    print(f"   Chunks: {result['chunk_count']}")
    print(f"   Tokens: {result['tokens_generated']}")
    print(f"   Time: {result['total_time']:.2f}s")

    # Check if hallucination was captured
    if logger.pending_prediction:
        print(f"\n📝 Hallucination captured!")
        print(f"   Model response: '{logger.pending_prediction['model_response'][:60]}...'")
        print(f"   Predicted user: '{logger.pending_prediction['predicted_user_response'][:60]}...'")

        # Complete the triplet
        print(f"\n🗣️ Simulating user response...")
        logger.log_actual_response("That's interesting, tell me more.")

        # Verify triplet
        stats = logger.get_stats()
        print(f"\n📊 Logger stats:")
        print(f"   Total predictions: {stats['total_predictions']}")
        print(f"   Complete: {stats['complete_predictions']}")
        print(f"   Avg similarity: {stats['avg_similarity']:.2%}")

        return True
    else:
        print(f"\n📝 No hallucination detected in this run")
        print(f"   (This is expected - hallucinations don't always occur)")
        return True


def test_export_training_data():
    """Test exporting predictions as training data."""
    print("\n" + "="*80)
    print("TEST 4: Export Training Data")
    print("="*80)

    # Create temporary logger with sample data
    import tempfile
    temp_dir = tempfile.mkdtemp()
    logger = PredictionLogger(log_dir=temp_dir)

    # Create several triplets
    print("\n📊 Creating sample prediction triplets...")
    test_triplets = [
        {
            'model': "Happiness feels like fulfillment.",
            'predicted': "User: How does that work?",
            'actual': "Tell me more about fulfillment."
        },
        {
            'model': "Do you have other feelings?",
            'predicted': "User: Not really.",
            'actual': "Yes, I'm curious about sadness."
        },
        {
            'model': "That's a great question.",
            'predicted': "User: Can you explain?",
            'actual': "Can you give examples?"
        }
    ]

    for i, triplet in enumerate(test_triplets, 1):
        logger.capture_hallucination(
            model_response=triplet['model'],
            predicted_user_response=triplet['predicted'],
            context={'triplet': i}
        )
        logger.log_actual_response(triplet['actual'])
        print(f"   ✓ Triplet {i} created")

    # Export
    print("\n📦 Exporting training data...")
    export_path = logger.export_training_data()

    if export_path:
        print(f"   ✓ Exported to: {export_path}")

        # Verify export
        with open(export_path, 'r') as f:
            training_data = json.load(f)

        print(f"\n✅ Export validation:")
        print(f"   Total examples: {len(training_data['examples'])}")
        print(f"   Avg similarity: {training_data['metadata']['avg_similarity']:.2%}")

        # Show sample
        if training_data['examples']:
            sample = training_data['examples'][0]
            print(f"\n📝 Sample example:")
            print(f"   Assistant: '{sample['assistant_response'][:50]}...'")
            print(f"   Predicted: '{sample['predicted_user_response'][:50]}...'")
            print(f"   Actual: '{sample['actual_user_response'][:50]}...'")
            print(f"   Similarity: {sample['similarity']:.2%}")

        return True
    else:
        print(f"   ❌ Export failed!")
        return False


if __name__ == "__main__":
    print("\n🧪 TESTING PREDICTION LOGGING SYSTEM")
    print("="*80)

    results = []

    # Run tests
    try:
        results.append(("Hallucination Extraction", test_hallucination_extraction()))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Hallucination Extraction", False))

    try:
        results.append(("Prediction Triplet Creation", test_prediction_triplet_creation()))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Prediction Triplet Creation", False))

    try:
        results.append(("Streaming with Logger", test_streaming_with_prediction_logger()))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Streaming with Logger", False))

    try:
        results.append(("Export Training Data", test_export_training_data()))
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Export Training Data", False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print(f"\n📊 Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
