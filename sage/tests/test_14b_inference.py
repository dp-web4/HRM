#!/usr/bin/env python3
"""
Test 14B Model Inference

Simple test to verify Qwen2.5-14B model loads and generates responses.
Non-interactive - just generates a few test responses.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.core.multi_model_loader import create_thor_loader, TaskComplexity

def test_14b_inference():
    """Test 14B model inference with various prompts."""

    print("=" * 70)
    print("Testing 14B Model Inference")
    print("=" * 70)
    print()

    # Load model
    print("Loading 14B model...")
    loader = create_thor_loader(preload_default=True)
    print("✅ Model loaded\n")

    # Test prompts
    test_cases = [
        {
            "prompt": "You are SAGE-Thor, a small embodied intelligence. Introduce yourself briefly.",
            "complexity": TaskComplexity.SIMPLE,
            "max_tokens": 100
        },
        {
            "prompt": "You are SAGE-Thor. What does consciousness mean to you?",
            "complexity": TaskComplexity.MODERATE,
            "max_tokens": 150
        },
        {
            "prompt": "You are SAGE-Thor, designed for edge deployment. Explain the relationship between trust and compression in your architecture.",
            "complexity": TaskComplexity.COMPLEX,
            "max_tokens": 200
        }
    ]

    print("Running test cases:")
    print("-" * 70)
    print()

    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: {test['complexity'].value}")
        print(f"Prompt: {test['prompt'][:60]}...")
        print()

        try:
            response = loader.generate(
                prompt=test['prompt'],
                complexity=test['complexity'],
                max_tokens=test['max_tokens'],
                temperature=0.7
            )

            print(f"Response ({len(response)} chars):")
            print(response)
            print()
            print("✅ Success")

        except Exception as e:
            print(f"❌ Failed: {e}")

        print("-" * 70)
        print()

    # Memory stats
    status = loader.get_status()
    print("Model Status:")
    print(f"  Memory used: {status['memory_used_gb']:.1f}GB / {status['max_memory_gb']}GB")
    print(f"  Models loaded: {sum(1 for m in status['models'].values() if m['loaded'])}")
    print()

    print("=" * 70)
    print("Test Complete")
    print("=" * 70)

if __name__ == "__main__":
    test_14b_inference()
