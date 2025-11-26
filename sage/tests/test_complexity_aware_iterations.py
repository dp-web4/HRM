#!/usr/bin/env python3
"""
Complexity-Aware IRP Iterations Test

Session 18 - Voice Latency Optimization

Session 17 revealed:
- Simple factual queries: 3.4s (fast)
- Complex philosophical queries: 48s (slow)
- LLM inference is 90% of latency

Hypothesis: Query complexity detection can enable adaptive iteration counts:
- Simple queries: 1 iteration (fast response)
- Complex queries: 3 iterations (quality response)

Expected outcome:
- 50% of queries could be "simple" → ~3s response
- 50% of queries "complex" → ~40s response
- Average latency improvement: 30-40%
"""

import sys
import time
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

# Add sage to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class QueryResult:
    """Result of a single query"""
    query: str
    query_type: str
    predicted_complexity: str
    actual_iterations: int
    response: str
    inference_time: float
    response_length: int


def classify_query_complexity(query: str) -> Tuple[str, int]:
    """
    Classify query complexity and return recommended IRP iterations.

    Returns: (complexity_level, recommended_iterations)

    Classification heuristics:
    - SIMPLE (1 iteration): Factual, math, definitions, greetings
    - MEDIUM (2 iterations): Explanations, comparisons, opinions
    - COMPLEX (3 iterations): Philosophy, consciousness, meta-cognitive, emotional
    """
    query_lower = query.lower().strip()

    # SIMPLE patterns (factual, quick answers)
    simple_patterns = [
        r'^what is \d',              # "What is 2+2?"
        r'^what is the capital',     # "What is the capital of France?"
        r'^who (is|was|are)',        # "Who is Albert Einstein?"
        r'^when (did|was|is)',       # "When was Python created?"
        r'^how many',                # "How many continents are there?"
        r'^define ',                 # "Define democracy"
        r'^hello',                   # Greetings
        r'^hi\b',                    # Greetings
        r'^(good )?(morning|afternoon|evening)', # Greetings
        r'^what year',               # "What year did..."
        r'^\d+\s*[\+\-\*\/]\s*\d+',  # Math expressions
        r'^list ',                   # "List the planets"
        r'^name ',                   # "Name three..."
        r'^true or false',           # Boolean questions
        r'^yes or no',               # Boolean questions
    ]

    for pattern in simple_patterns:
        if re.search(pattern, query_lower):
            return "simple", 1

    # COMPLEX patterns (philosophical, meta-cognitive)
    complex_patterns = [
        r'consciousness',            # Consciousness questions
        r'aware(ness)?',             # Awareness questions
        r'feel(ing)?s?',             # Feelings/emotions
        r'experience',               # Experiential questions
        r'meaning of',               # Meaning questions
        r'purpose of',               # Purpose questions
        r'what do you think',        # Opinion on complex topics
        r'philosophy',               # Philosophy
        r'exist(ence|ential)',       # Existential questions
        r'soul',                     # Soul/spirit
        r'mind\b',                   # Mind questions
        r'belief',                   # Belief questions
        r'moral(ity)?',              # Moral questions
        r'ethic(s|al)',              # Ethics
        r'what makes you',           # Self-reflection
        r'how do you know',          # Epistemological
        r'can you (truly|really)',   # Meta-cognitive probing
        r'relationship between',     # Complex relationships
        r'paradox',                  # Paradoxes
        r'understand(ing)?',         # Understanding
        r'knowledge',                # Knowledge
        r'truth',                    # Truth
    ]

    for pattern in complex_patterns:
        if re.search(pattern, query_lower):
            return "complex", 3

    # Default to MEDIUM
    return "medium", 2


def get_temperature() -> float:
    """Read Jetson thermal zone temperature"""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return int(f.read().strip()) / 1000.0
    except:
        return 0.0


def run_complexity_test():
    """Test complexity-aware iteration selection"""

    print("=" * 80)
    print("COMPLEXITY-AWARE IRP ITERATIONS TEST")
    print("=" * 80)
    print()
    print("Hardware: Jetson Orin Nano 8GB")
    print("Purpose: Validate query complexity detection for voice latency optimization")
    print()

    # Test queries with expected classifications
    test_queries = [
        # Simple (expected: 1 iteration)
        ("What is 2 + 2?", "simple"),
        ("What is the capital of France?", "simple"),
        ("Hello, how are you?", "simple"),
        ("Define photosynthesis.", "simple"),

        # Medium (expected: 2 iterations)
        ("Explain how neural networks learn.", "medium"),
        ("Compare Python and JavaScript.", "medium"),
        ("What are the benefits of meditation?", "medium"),
        ("Describe the water cycle.", "medium"),

        # Complex (expected: 3 iterations)
        ("What is consciousness and do you have it?", "complex"),
        ("How do you experience this conversation?", "complex"),
        ("What is the relationship between knowledge and belief?", "complex"),
        ("Are you truly aware or just simulating awareness?", "complex"),
    ]

    # Test classification accuracy
    print("-" * 80)
    print("CLASSIFICATION TEST")
    print("-" * 80)
    print(f"{'Query':<50} {'Expected':<10} {'Predicted':<10} {'Iters':<6} {'Match'}")
    print("-" * 80)

    correct = 0
    for query, expected in test_queries:
        predicted, iterations = classify_query_complexity(query)
        match = "✓" if predicted == expected else "✗"
        if predicted == expected:
            correct += 1

        query_display = query[:48] + ".." if len(query) > 50 else query
        print(f"{query_display:<50} {expected:<10} {predicted:<10} {iterations:<6} {match}")

    accuracy = correct / len(test_queries) * 100
    print("-" * 80)
    print(f"Classification Accuracy: {correct}/{len(test_queries)} ({accuracy:.0f}%)")
    print()

    # Now test actual inference with adaptive iterations
    print("=" * 80)
    print("INFERENCE TEST: Fixed vs Adaptive Iterations")
    print("=" * 80)

    from sage.irp.plugins.llm_impl import ConversationalLLM

    # Test subset (one of each type)
    inference_queries = [
        ("What is 2 + 2?", "simple", 1),
        ("Explain how neural networks learn.", "medium", 2),
        ("What is consciousness?", "complex", 3),
    ]

    model_path = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"

    # Test 1: Fixed 3 iterations (baseline)
    print()
    print("Test 1: Fixed 3 Iterations (Baseline)")
    print("-" * 80)

    conv_fixed = ConversationalLLM(model_path=model_path, irp_iterations=3)
    fixed_results: List[QueryResult] = []

    for query, qtype, _ in inference_queries:
        start = time.perf_counter()
        response, _ = conv_fixed.respond(query)
        elapsed = time.perf_counter() - start

        result = QueryResult(
            query=query,
            query_type=qtype,
            predicted_complexity=qtype,
            actual_iterations=3,
            response=response,
            inference_time=elapsed,
            response_length=len(response)
        )
        fixed_results.append(result)
        print(f"  {qtype:<8}: {elapsed:.1f}s ({len(response)} chars)")

    fixed_total = sum(r.inference_time for r in fixed_results)
    fixed_avg = fixed_total / len(fixed_results)
    print(f"  Total: {fixed_total:.1f}s, Average: {fixed_avg:.1f}s")

    # Clean up
    del conv_fixed
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Test 2: Adaptive iterations
    print()
    print("Test 2: Adaptive Iterations")
    print("-" * 80)

    adaptive_results: List[QueryResult] = []

    for query, qtype, recommended_iters in inference_queries:
        # Create model with recommended iterations
        conv_adaptive = ConversationalLLM(model_path=model_path, irp_iterations=recommended_iters)

        start = time.perf_counter()
        response, _ = conv_adaptive.respond(query)
        elapsed = time.perf_counter() - start

        result = QueryResult(
            query=query,
            query_type=qtype,
            predicted_complexity=qtype,
            actual_iterations=recommended_iters,
            response=response,
            inference_time=elapsed,
            response_length=len(response)
        )
        adaptive_results.append(result)
        print(f"  {qtype:<8}: {elapsed:.1f}s ({recommended_iters} iters, {len(response)} chars)")

        # Clean up
        del conv_adaptive
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    adaptive_total = sum(r.inference_time for r in adaptive_results)
    adaptive_avg = adaptive_total / len(adaptive_results)
    print(f"  Total: {adaptive_total:.1f}s, Average: {adaptive_avg:.1f}s")

    # Analysis
    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print()
    print("Per-Query Comparison:")
    print(f"{'Type':<10} {'Fixed 3':<12} {'Adaptive':<12} {'Speedup':<12} {'Iters'}")
    print("-" * 60)

    for fixed, adaptive in zip(fixed_results, adaptive_results):
        speedup = fixed.inference_time / adaptive.inference_time if adaptive.inference_time > 0 else 0
        speedup_str = f"{speedup:.1f}x" if speedup > 1 else f"{speedup:.2f}x"
        print(f"{fixed.query_type:<10} {fixed.inference_time:<12.1f} {adaptive.inference_time:<12.1f} {speedup_str:<12} {adaptive.actual_iterations}")

    total_speedup = fixed_total / adaptive_total if adaptive_total > 0 else 0
    print("-" * 60)
    print(f"{'TOTAL':<10} {fixed_total:<12.1f} {adaptive_total:<12.1f} {total_speedup:.1f}x")

    # Savings calculation
    time_saved = fixed_total - adaptive_total
    percent_saved = (time_saved / fixed_total) * 100 if fixed_total > 0 else 0

    print()
    print("Latency Savings:")
    print(f"  Time saved: {time_saved:.1f}s ({percent_saved:.0f}%)")
    print(f"  Fixed 3 average: {fixed_avg:.1f}s")
    print(f"  Adaptive average: {adaptive_avg:.1f}s")

    # Verdict
    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()

    if percent_saved > 20:
        print(f"✓ EFFECTIVE: Adaptive iterations save {percent_saved:.0f}% latency")
        print(f"  - Simple queries run faster with fewer iterations")
        print(f"  - Complex queries still get full quality")
        print(f"  - Recommended: Implement complexity detection in ConversationalLLM")
    elif percent_saved > 10:
        print(f"⚠ MODERATE: Adaptive iterations save {percent_saved:.0f}% latency")
        print(f"  - Some improvement, but limited")
    else:
        print(f"✗ NOT EFFECTIVE: Only {percent_saved:.0f}% improvement")
        print(f"  - Complexity detection doesn't significantly help")

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    # Print sample responses for quality check
    print()
    print("Sample Responses (Adaptive):")
    print("-" * 80)
    for r in adaptive_results:
        print(f"Q: {r.query}")
        print(f"A: {r.response[:150]}...")
        print(f"[{r.actual_iterations} iters, {r.inference_time:.1f}s]")
        print()


if __name__ == "__main__":
    run_complexity_test()
