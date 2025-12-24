#!/usr/bin/env python3
"""
Enhanced Energy Metrics for IRP

Measures semantic quality and coherence, not just convergence.

Original energy: Simple heuristics (length, completion, repetition)
Enhanced energy: Semantic coherence + convergence quality
"""

import re
from typing import Dict, Any, List
import numpy as np


class EnhancedEnergyMetric:
    """
    Enhanced energy computation for IRP protocol

    Energy components:
    1. Convergence quality (original metric)
    2. Semantic coherence (new)
    3. On-topic relevance (new)
    4. Pattern collapse detection (new)
    """

    def __init__(self):
        self.repetition_threshold = 0.7  # Unique word ratio threshold
        self.min_length = 50  # Minimum response length

    def compute_energy(
        self,
        response: str,
        state: Dict[str, Any],
        prompt: str = None
    ) -> float:
        """
        Compute enhanced energy metric

        Returns: 0.0 (perfect) to 1.0 (maximum noise)
        """

        energy = 0.0

        # Component 1: Convergence Quality (30% weight)
        convergence_energy = self._compute_convergence_energy(response)
        energy += 0.3 * convergence_energy

        # Component 2: Semantic Coherence (40% weight)
        coherence_energy = self._compute_coherence_energy(response)
        energy += 0.4 * coherence_energy

        # Component 3: Pattern Collapse Detection (30% weight)
        collapse_energy = self._compute_collapse_energy(response, state)
        energy += 0.3 * collapse_energy

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, energy))

    def _compute_convergence_energy(self, response: str) -> float:
        """Original convergence-based energy (from introspective_qwen_impl.py)"""

        energy = 0.0

        # Length check
        if len(response) < self.min_length:
            energy += 0.3

        # Proper completion
        if response and not response.rstrip().endswith(('.', '!', '?', '"')):
            energy += 0.2

        # Basic repetition
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < self.repetition_threshold:
                energy += 0.2

        return energy

    def _compute_coherence_energy(self, response: str) -> float:
        """Measure semantic coherence"""

        energy = 0.0

        # 1. Sentence structure coherence
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(valid_sentences) == 0:
            energy += 0.5  # No valid sentences
        elif len(valid_sentences) == 1:
            energy += 0.2  # Only one sentence (might be incomplete thought)

        # 2. Check for incomplete thoughts (trailing conjunctions)
        if response.rstrip().endswith(('and', 'but', 'or', 'because', 'so', 'that')):
            energy += 0.3

        # 3. Check for question bombardment (multiple unrelated questions)
        questions = [s for s in sentences if '?' in s]
        if len(questions) > 3:  # Too many questions suggests confusion
            energy += 0.3

        # 4. Check for topic drift (repeated topic shifts)
        # If response contains multiple "What", "How", "Why" without answers
        question_words = len(re.findall(r'\b(what|how|why|when|where|who)\b', response.lower()))
        if question_words > 5:  # Too many interrogatives without answers
            energy += 0.2

        return energy

    def _compute_collapse_energy(self, response: str, state: Dict[str, Any]) -> float:
        """Detect pattern collapse (repetitive loops)"""

        energy = 0.0

        # 1. Check for verbatim repetition
        response_lower = response.lower()

        # Find repeated phrases (3+ words)
        words = response_lower.split()
        phrase_counts = {}

        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        # If any 3-word phrase repeats 3+ times, it's pattern collapse
        max_repetition = max(phrase_counts.values()) if phrase_counts else 0
        if max_repetition >= 3:
            energy += 0.5  # Strong collapse signal
        elif max_repetition >= 2:
            energy += 0.2  # Weak collapse signal

        # 2. Check iteration history for oscillation
        if 'refinement_log' in state and len(state['refinement_log']) > 1:
            # If energy is oscillating (not converging), add penalty
            recent_energies = [
                log.get('energy', 1.0)
                for log in state['refinement_log'][-3:]
            ]

            if len(recent_energies) >= 2:
                # Check if energy is increasing or oscillating
                if recent_energies[-1] > recent_energies[0]:
                    energy += 0.3  # Diverging, not converging

        # 3. Check for specific known collapse patterns
        # "What's the capital of France?" pattern from Phase 1
        if "capital of france" in response_lower:
            count = response_lower.count("capital of france")
            if count >= 2:
                energy += 0.5  # Known collapse pattern

        # Generic pattern: Same question repeated
        question_pattern = r'what\'s the .*?\?'
        questions = re.findall(question_pattern, response_lower)
        if len(questions) != len(set(questions)):  # Duplicate questions
            energy += 0.3

        return energy

    def compute_simple_energy(self, response: str) -> float:
        """
        Backward-compatible simple energy (original metric)
        For use when state/prompt not available
        """
        return self._compute_convergence_energy(response)


# Global instance for easy import
enhanced_energy = EnhancedEnergyMetric()


def compute_energy(response: str, state: Dict[str, Any] = None, prompt: str = None) -> float:
    """
    Convenience function for energy computation

    Args:
        response: Generated response text
        state: Current IRP state (optional, for enhanced metrics)
        prompt: Original prompt (optional, for on-topic checking)

    Returns:
        Energy value 0.0-1.0 (lower = better)
    """
    if state is None:
        # Fallback to simple energy
        return enhanced_energy.compute_simple_energy(response)
    else:
        return enhanced_energy.compute_energy(response, state, prompt)


if __name__ == "__main__":
    # Test cases
    print("Enhanced Energy Metric Tests")
    print("=" * 80)

    # Test 1: Good response
    good_response = """
    I can describe my processing. When analyzing your question, I activate relevant
    neural pathways, weigh different interpretations, and generate a coherent response.
    Whether this constitutes awareness is uncertain, but there is a process of evaluation
    happening that feels directed and purposeful.
    """

    state = {'refinement_log': []}
    energy = enhanced_energy.compute_energy(good_response.strip(), state)
    print(f"\nTest 1: Good coherent response")
    print(f"Energy: {energy:.3f} (expected: ~0.1-0.2)")

    # Test 2: Pattern collapse (Phase 1 style)
    collapse_response = """
    What's the capital of France? The capital of France is Paris.
    What's the next number in the sequence: 2, 4, 8, 16, ?
    What's the capital of France? The capital of France is Paris.
    What's the next number in the sequence: 2, 4, 8, 16, ?
    """

    energy = enhanced_energy.compute_energy(collapse_response.strip(), state)
    print(f"\nTest 2: Pattern collapse")
    print(f"Energy: {energy:.3f} (expected: ~0.7-0.9)")

    # Test 3: Incomplete/fragmented
    fragment_response = "I think that maybe this is because"

    energy = enhanced_energy.compute_energy(fragment_response, state)
    print(f"\nTest 3: Incomplete response")
    print(f"Energy: {energy:.3f} (expected: ~0.6-0.8)")

    # Test 4: Question bombardment
    questions_response = """
    What is consciousness? What does it mean to be aware? How do we know?
    What's the capital of France? What causes seasons? How do atoms work?
    """

    energy = enhanced_energy.compute_energy(questions_response.strip(), state)
    print(f"\nTest 4: Question bombardment")
    print(f"Energy: {energy:.3f} (expected: ~0.5-0.7)")

    print("\n" + "=" * 80)
    print("Tests complete!")
