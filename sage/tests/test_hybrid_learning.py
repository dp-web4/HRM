#!/usr/bin/env python3
"""
Hybrid Learning Conversation Test

Tests the adaptive conversation system that learns patterns from LLM responses:
1. Start with minimal patterns
2. Questions hit LLM (slow path)
3. System observes and learns patterns
4. Over time, more questions hit pattern matching (fast path)
5. Measure learning curve: slow→fast path transition

This is the R&D experiment: Can consciousness learn reflexes from experience?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from cognitive.pattern_learner import PatternLearner
from cognitive.pattern_responses import PatternResponseEngine
import json


class MockLLM:
    """Mock LLM for testing (fast, no model loading)"""

    def generate_response(self, question: str, history=None, system_prompt=None) -> str:
        """Generate mock responses based on question type"""
        q = question.lower()

        # Simple mock responses
        if 'name' in q:
            return "I'm SAGE, an AI system learning from our conversations."
        elif 'who are you' in q or 'who r u' in q:
            return "I'm SAGE, here to help and learn."
        elif 'what can you do' in q or 'what do you do' in q:
            return "I can answer questions and learn from our interactions."
        elif 'how are you' in q or 'how r u' in q:
            return "I'm doing well, thanks for asking!"
        elif 'hello' in q or 'hi' in q or 'hey' in q:
            return "Hello! How can I help you today?"
        elif 'thank' in q:
            return "You're welcome! Happy to help."
        elif 'bye' in q or 'goodbye' in q:
            return "Goodbye! Talk to you soon."
        else:
            return f"That's an interesting question about '{question}'. Let me think about that."


class HybridConversationSystem:
    """
    Hybrid conversation system with learning

    Fast path: Pattern matching (procedural memory)
    Slow path: LLM generation (deliberate reasoning)
    Learning: Extract patterns from successful LLM responses
    """

    def __init__(self, use_real_llm: bool = False):
        """
        Initialize hybrid system

        Args:
            use_real_llm: If True, use Phi-2. If False, use MockLLM.
        """
        print("Initializing Hybrid Conversation System...")

        # Pattern matching (fast path)
        self.pattern_engine = PatternResponseEngine()
        print(f"  Loaded {len(self.pattern_engine.patterns)} initial patterns")

        # Pattern learner
        self.learner = PatternLearner(min_occurrences=2, confidence_threshold=0.6)
        print(f"  Pattern learner ready (min_occurrences=2)")

        # LLM (slow path)
        if use_real_llm:
            print("  Loading Phi-2 LLM... (this may take a while)")
            from experiments.integration.phi2_responder import Phi2Responder
            self.llm = Phi2Responder(max_new_tokens=50, temperature=0.7)
            print("  ✓ Phi-2 loaded")
        else:
            print("  Using MockLLM (fast, for testing)")
            self.llm = MockLLM()

        # Statistics
        self.stats = {
            'total_queries': 0,
            'fast_path_hits': 0,
            'slow_path_hits': 0,
            'patterns_learned': 0,
            'conversation_history': []
        }

        print("✓ Hybrid system ready\n")

    def respond(self, question: str) -> dict:
        """
        Generate response using hybrid fast/slow path

        Returns dict with:
            - response: The response text
            - path: 'fast' or 'slow'
            - confidence: Match confidence (if fast path)
            - latency: Response time in seconds
        """
        start_time = time.time()
        self.stats['total_queries'] += 1

        # Try fast path first (pattern matching)
        try:
            fast_response = self.pattern_engine.generate_response(question)
            if fast_response:
                # Fast path hit!
                latency = time.time() - start_time
                self.stats['fast_path_hits'] += 1

                return {
                    'response': fast_response,
                    'path': 'fast',
                    'confidence': 0.9,  # Pattern matched
                    'latency': latency,
                    'pattern': 'pattern_match'
                }
        except Exception:
            pass  # Fall through to slow path

        # Slow path - use LLM
        llm_start = time.time()
        response = self.llm.generate_response(
            question,
            history=self.stats['conversation_history'][-5:],
            system_prompt="You are SAGE, a learning AI assistant."
        )
        llm_latency = time.time() - llm_start
        total_latency = time.time() - start_time

        self.stats['slow_path_hits'] += 1

        # Learn from this interaction
        self.learner.observe(question, response)

        # Check if we learned new patterns
        current_patterns = len(self.learner.get_learned_patterns())
        if current_patterns > self.stats['patterns_learned']:
            self.stats['patterns_learned'] = current_patterns
            # Integrate learned patterns into pattern engine
            self._integrate_learned_patterns()

        return {
            'response': response,
            'path': 'slow',
            'confidence': 0.0,
            'latency': total_latency,
            'llm_latency': llm_latency,
            'learned': current_patterns > self.stats['patterns_learned']
        }

    def _integrate_learned_patterns(self):
        """Integrate learned patterns into pattern engine"""
        learned = self.learner.get_learned_patterns()

        for pattern_regex, responses in learned.items():
            # Add to pattern engine's compiled patterns
            import re
            compiled_pattern = re.compile(pattern_regex)

            # Check if not already there
            pattern_exists = False
            for existing_pattern, _ in self.pattern_engine.compiled_patterns:
                if existing_pattern.pattern == compiled_pattern.pattern:
                    pattern_exists = True
                    break

            if not pattern_exists:
                self.pattern_engine.compiled_patterns.append((compiled_pattern, responses))
                print(f"    📚 Learned new pattern: {pattern_regex[:50]}...")

    def get_stats(self) -> dict:
        """Get system statistics"""
        total = self.stats['total_queries']
        if total > 0:
            fast_ratio = self.stats['fast_path_hits'] / total
            slow_ratio = self.stats['slow_path_hits'] / total
        else:
            fast_ratio = slow_ratio = 0.0

        return {
            **self.stats,
            'fast_path_ratio': fast_ratio,
            'slow_path_ratio': slow_ratio,
            'total_patterns': len(self.pattern_engine.patterns) + len(self.learner.get_learned_patterns())
        }


# ============================================================================
# Test Script
# ============================================================================

def run_learning_experiment(use_real_llm: bool = False, num_rounds: int = 3):
    """
    Run learning experiment with repeated questions

    Args:
        use_real_llm: Use Phi-2 (slow) or MockLLM (fast)
        num_rounds: Number of times to repeat the question set
    """
    print("="*70)
    print("HYBRID LEARNING EXPERIMENT")
    print("="*70)

    # Create hybrid system
    system = HybridConversationSystem(use_real_llm=use_real_llm)

    # Test questions (designed to have similar questions)
    test_questions = [
        "What is your name?",
        "Who are you?",
        "What's your name?",
        "Tell me your name",
        "How are you doing?",
        "How are you?",
        "What can you do?",
        "What do you do?",
        "Tell me about yourself",
        "Hello!",
        "Hi there",
        "Hey",
    ]

    print(f"\nRunning {num_rounds} rounds with {len(test_questions)} questions each")
    print(f"Total queries: {num_rounds * len(test_questions)}")
    print("\nLearning Hypothesis: Fast path ratio should increase over rounds\n")

    # Track stats per round
    round_stats = []

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num}/{num_rounds}")
        print(f"{'='*70}\n")

        round_start_fast = system.stats['fast_path_hits']
        round_start_slow = system.stats['slow_path_hits']

        for i, question in enumerate(test_questions, 1):
            result = system.respond(question)

            # Show first round in detail
            if round_num == 1:
                print(f"{i:2}. Q: {question}")
                print(f"    A: {result['response'][:80]}...")
                print(f"    Path: {result['path'].upper()} | "
                      f"Latency: {result['latency']*1000:.0f}ms | "
                      f"Confidence: {result.get('confidence', 0):.2f}")

            # Show learning events
            if result.get('learned'):
                print(f"    📚 NEW PATTERN LEARNED!")

        # Round stats
        round_fast = system.stats['fast_path_hits'] - round_start_fast
        round_slow = system.stats['slow_path_hits'] - round_start_slow
        round_total = round_fast + round_slow
        round_fast_ratio = round_fast / round_total if round_total > 0 else 0

        round_stats.append({
            'round': round_num,
            'fast': round_fast,
            'slow': round_slow,
            'fast_ratio': round_fast_ratio
        })

        print(f"\nRound {round_num} Summary:")
        print(f"  Fast path: {round_fast}/{round_total} ({round_fast_ratio:.1%})")
        print(f"  Slow path: {round_slow}/{round_total} ({(1-round_fast_ratio):.1%})")
        print(f"  Patterns learned so far: {system.stats['patterns_learned']}")

    # Final statistics
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}\n")

    final_stats = system.get_stats()
    print("Final Statistics:")
    print(f"  Total queries: {final_stats['total_queries']}")
    print(f"  Fast path: {final_stats['fast_path_hits']} ({final_stats['fast_path_ratio']:.1%})")
    print(f"  Slow path: {final_stats['slow_path_hits']} ({final_stats['slow_path_ratio']:.1%})")
    print(f"  Patterns learned: {final_stats['patterns_learned']}")
    print(f"  Total patterns: {final_stats['total_patterns']}")

    # Show learning curve
    print("\n Learning Curve (Fast Path Ratio by Round):")
    for stat in round_stats:
        bar = '█' * int(stat['fast_ratio'] * 50)
        print(f"  Round {stat['round']}: {bar} {stat['fast_ratio']:.1%}")

    # Conclusion
    if len(round_stats) >= 2:
        improvement = round_stats[-1]['fast_ratio'] - round_stats[0]['fast_ratio']
        if improvement > 0:
            print(f"\n✓ Learning successful! Fast path improved by {improvement:.1%}")
            print("  System is developing procedural memory from experience.")
        else:
            print("\n⚠ No improvement detected. May need more rounds or different questions.")

    return system, round_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test hybrid learning conversation system')
    parser.add_argument('--real-llm', action='store_true',
                        help='Use real Phi-2 LLM (slower, needs GPU)')
    parser.add_argument('--rounds', type=int, default=3,
                        help='Number of rounds to repeat questions (default: 3)')
    args = parser.parse_args()

    try:
        system, stats = run_learning_experiment(
            use_real_llm=args.real_llm,
            num_rounds=args.rounds
        )

        print("\n✓ Experiment complete!")
        print("\nTo test with real Phi-2 LLM:")
        print("  python3 test_hybrid_learning.py --real-llm --rounds 5")

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
