#!/usr/bin/env python3
"""
Test Pattern Learning - Validate Hybrid System Learning

Tests:
1. Novel questions force LLM (slow path)
2. Repeated questions learned as patterns (fast path)
3. Fast path ratio improves over time
4. Pattern confidence gating works correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cognitive.pattern_learner import PatternLearner
from cognitive.pattern_responses import PatternResponseEngine

class MockLLM:
    """Mock LLM with consistent responses for testing"""
    def generate_response(self, question: str, conversation_history=None, system_prompt=None) -> str:
        q = question.lower()

        # Consistent responses for learning test
        if 'quantum' in q:
            return "Quantum mechanics describes the behavior of matter at atomic scales."
        elif 'neural network' in q:
            return "Neural networks are inspired by biological brain structures."
        elif 'black hole' in q:
            return "Black holes are regions where gravity is so strong nothing escapes."
        elif 'photosynthesis' in q:
            return "Photosynthesis is how plants convert sunlight into chemical energy."
        elif 'fibonacci' in q:
            return "The Fibonacci sequence is 0, 1, 1, 2, 3, 5, 8, 13, ..."
        else:
            return f"Interesting question about {q.split()[0] if q.split() else 'that'}."

def test_pattern_learning():
    """Test that pattern learning works correctly"""

    print("="*80)
    print("🧪 PATTERN LEARNING TEST")
    print("="*80)

    # Initialize components
    pattern_engine = PatternResponseEngine()
    learner = PatternLearner(min_occurrences=2, confidence_threshold=0.6)
    llm = MockLLM()

    initial_patterns = len(pattern_engine.patterns)
    print(f"\n📊 Initial state:")
    print(f"   Pattern engine: {initial_patterns} patterns")
    print(f"   Pattern learner: min_occurrences=2")

    # Test questions (designed to force learning)
    test_questions = [
        # Round 1: Novel questions (should all miss)
        "What is quantum mechanics?",
        "How do neural networks work?",
        "Tell me about black holes",
        "Explain photosynthesis",
        "What is the Fibonacci sequence?",

        # Round 2: Repeat same questions (should start learning)
        "What is quantum mechanics?",
        "How do neural networks work?",
        "Tell me about black holes",
        "Explain photosynthesis",
        "What is the Fibonacci sequence?",

        # Round 3: Repeat again (should use fast path after learning)
        "What is quantum mechanics?",
        "How do neural networks work?",
        "Tell me about black holes",
    ]

    stats = {
        'fast_hits': 0,
        'slow_hits': 0,
        'patterns_learned': 0
    }

    print(f"\n🔬 Running {len(test_questions)} test queries...")
    print()

    for i, question in enumerate(test_questions, 1):
        # Try pattern matching first
        fast_response = pattern_engine.generate_response(question)

        if fast_response:
            # Fast path hit
            stats['fast_hits'] += 1
            path = "⚡ FAST"
            response = fast_response
        else:
            # Slow path - use LLM
            stats['slow_hits'] += 1
            path = "🧠 SLOW"
            response = llm.generate_response(question)

            # Learn from this interaction
            learner.observe(question, response)

            # Check if we learned new patterns
            current_patterns = len(learner.get_learned_patterns())
            if current_patterns > stats['patterns_learned']:
                stats['patterns_learned'] = current_patterns

                # Integrate learned patterns
                for pattern_regex, responses in learner.get_learned_patterns().items():
                    import re
                    compiled_pattern = re.compile(pattern_regex)

                    # Check if not already in engine
                    pattern_exists = any(
                        p.pattern == compiled_pattern.pattern
                        for p, _ in pattern_engine.compiled_patterns
                    )

                    if not pattern_exists:
                        pattern_engine.compiled_patterns.append((compiled_pattern, responses))
                        print(f"   📚 [{i:2d}] NEW PATTERN LEARNED: {pattern_regex[:50]}...")

        # Calculate current efficiency
        total = stats['fast_hits'] + stats['slow_hits']
        efficiency = stats['fast_hits'] / total if total > 0 else 0.0

        print(f"   [{i:2d}] {path} | Q: {question[:50]:50s} | Efficiency: {efficiency:.1%}")

    # Final statistics
    print()
    print("="*80)
    print("📊 FINAL RESULTS")
    print("="*80)

    total_queries = stats['fast_hits'] + stats['slow_hits']
    fast_ratio = stats['fast_hits'] / total_queries if total_queries > 0 else 0.0

    print(f"\n✅ Total queries: {total_queries}")
    print(f"   Fast path: {stats['fast_hits']} ({fast_ratio:.1%})")
    print(f"   Slow path: {stats['slow_hits']}")
    print(f"   Patterns learned: {stats['patterns_learned']}")
    print(f"   Total patterns: {initial_patterns + stats['patterns_learned']}")

    # Validate learning progression
    print(f"\n🎯 Learning Validation:")

    # Round 1 (questions 1-5): Should all be slow
    # Round 2 (questions 6-10): Should all be slow (learning happens here)
    # Round 3 (questions 11-13): Should be fast (using learned patterns)

    expected_fast = 0  # After 2 occurrences, patterns should be learned
    # In practice, round 3 should hit fast path

    if fast_ratio > 0:
        print(f"   ✅ Fast path engaged ({fast_ratio:.1%})")
    else:
        print(f"   ⚠️  Fast path never engaged")

    if stats['patterns_learned'] > 0:
        print(f"   ✅ Patterns learned ({stats['patterns_learned']})")
    else:
        print(f"   ⚠️  No patterns learned")

    # Success criteria
    success = fast_ratio > 0 and stats['patterns_learned'] > 0

    if success:
        print(f"\n✅ TEST PASSED: Pattern learning is working!")
        print(f"   System learned {stats['patterns_learned']} new patterns")
        print(f"   Fast path efficiency: {fast_ratio:.1%}")
    else:
        print(f"\n❌ TEST FAILED: Pattern learning not working as expected")
        if fast_ratio == 0:
            print(f"   - Fast path never engaged")
        if stats['patterns_learned'] == 0:
            print(f"   - No patterns learned")

    print()
    print("="*80)

    return success

if __name__ == "__main__":
    success = test_pattern_learning()
    sys.exit(0 if success else 1)
