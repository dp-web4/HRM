"""
Fast Path Handoff Mechanism

Implements the "hmm..." acknowledgment when fast path recognizes
it should defer to slow path for complex reasoning.

This mirrors human behavior during deep thought - when someone speaks
to you while you're concentrating, there's a simple performatory loop
that does context-adjacent acknowledgments: "hmm...", "let me think...",
"interesting..."

The fast path learns to recognize:
1. Complexity indicators (questions with "why", "how", "what if")
2. Philosophical/abstract language
3. Multi-part questions
4. Novel topics not in training data
5. Uncertainty in pattern matching

When detected, respond with acknowledgment and hand off to slow path.
"""

from typing import Tuple, Optional
import re


class FastPathHandoff:
    """
    Manages handoff from fast path to slow path with acknowledgment.

    The fast path should:
    1. Quickly assess if it can handle the query confidently
    2. If uncertain or complex, acknowledge and defer to slow path
    3. Provide immediate feedback ("hmm...") while slow path processes
    """

    def __init__(self):
        """Initialize handoff detection"""
        # Complexity indicators
        self.complexity_keywords = [
            'why', 'how', 'explain', 'understand', 'meaning',
            'purpose', 'relationship', 'difference', 'compare',
            'philosophy', 'consciousness', 'nature of', 'what if'
        ]

        # Abstract/conceptual language
        self.abstract_keywords = [
            'essence', 'fundamental', 'principle', 'concept',
            'theory', 'abstract', 'metaphor', 'analogy'
        ]

        # Handoff acknowledgments (natural, human-like)
        self.handoff_responses = [
            "Hmm...",
            "Let me think about that...",
            "That's an interesting question...",
            "I need to consider this carefully...",
            "Good question. Let me reflect...",
            "Hmm, that's complex...",
            "Interesting. Give me a moment..."
        ]

        # Track handoff statistics
        self.handoff_count = 0
        self.handoff_reasons = {}

    def should_handoff(self, text: str, confidence: float) -> Tuple[bool, Optional[str]]:
        """
        Determine if fast path should hand off to slow path.

        Args:
            text: Input text from user
            confidence: Confidence score from fast path pattern matching

        Returns:
            Tuple of (should_handoff, reason)
        """
        text_lower = text.lower()

        # Reason 1: Low confidence in pattern match
        if confidence < 0.6:
            return True, "low_confidence"

        # Reason 2: Complexity keywords present
        complexity_count = sum(1 for kw in self.complexity_keywords if kw in text_lower)
        if complexity_count >= 2:
            return True, "complexity_indicators"

        # Reason 3: Abstract/philosophical language
        abstract_count = sum(1 for kw in self.abstract_keywords if kw in text_lower)
        if abstract_count >= 1:
            return True, "abstract_language"

        # Reason 4: Multi-part question (multiple question marks or "and")
        if text.count('?') > 1 or (('?' in text) and (' and ' in text_lower)):
            return True, "multi_part_question"

        # Reason 5: Long, complex sentence structure
        if len(text) > 150 and '?' in text:
            return True, "long_complex_query"

        # Reason 6: Question words combined with abstract concepts
        question_words = ['what', 'why', 'how', 'when', 'where', 'who']
        has_question = any(qw in text_lower for qw in question_words)
        has_abstract = any(ak in text_lower for ak in self.abstract_keywords)
        if has_question and has_abstract:
            return True, "question_with_abstraction"

        return False, None

    def get_handoff_acknowledgment(self, reason: Optional[str] = None) -> str:
        """
        Get appropriate handoff acknowledgment.

        Args:
            reason: Reason for handoff (optional, for context-aware responses)

        Returns:
            Acknowledgment string
        """
        import random

        # Track handoff
        self.handoff_count += 1
        if reason:
            self.handoff_reasons[reason] = self.handoff_reasons.get(reason, 0) + 1

        # Select acknowledgment (could be made context-aware)
        return random.choice(self.handoff_responses)

    def format_handoff_response(self, acknowledgment: str) -> dict:
        """
        Format handoff response with metadata.

        Returns:
            Dictionary with response and metadata for slow path
        """
        return {
            'response': acknowledgment,
            'handoff': True,
            'path': 'fast_to_slow_handoff',
            'immediate_feedback': True,  # User gets immediate acknowledgment
            'slow_path_needed': True     # Signal to invoke slow path
        }

    def get_statistics(self) -> dict:
        """Get handoff statistics"""
        return {
            'total_handoffs': self.handoff_count,
            'reasons': dict(self.handoff_reasons)
        }


class FastPathConfidenceEstimator:
    """
    Estimates confidence for fast path responses.

    Eventually, this will be replaced by the tiny model's internal
    confidence scores. For now, uses heuristics.
    """

    def __init__(self):
        """Initialize confidence estimator"""
        self.known_patterns = set()  # Track learned patterns

    def estimate_confidence(self,
                          text: str,
                          matched_pattern: Optional[str],
                          response: str) -> float:
        """
        Estimate confidence in fast path response.

        Args:
            text: Input text
            matched_pattern: Pattern that matched (None if no match)
            response: Generated response

        Returns:
            Confidence score (0.0 - 1.0)
        """
        if matched_pattern is None:
            return 0.0

        confidence = 0.5  # Base confidence

        # Factor 1: Pattern has been successful before
        if matched_pattern in self.known_patterns:
            confidence += 0.2

        # Factor 2: Input closely matches pattern (exact vs fuzzy)
        if self._is_exact_match(text, matched_pattern):
            confidence += 0.2

        # Factor 3: Response is concise and clear
        if len(response) < 50 and '?' not in response:
            confidence += 0.1

        # Factor 4: No uncertainty markers in response
        uncertainty_markers = ['maybe', 'perhaps', 'not sure', 'think']
        if not any(marker in response.lower() for marker in uncertainty_markers):
            confidence += 0.1

        return min(1.0, confidence)

    def _is_exact_match(self, text: str, pattern: str) -> bool:
        """Check if text exactly matches pattern (not just fuzzy match)"""
        # Simplified check (would be more sophisticated with actual regex)
        return text.lower().strip() in pattern.lower()

    def register_successful_pattern(self, pattern: str) -> None:
        """Register a pattern that produced a successful response"""
        self.known_patterns.add(pattern)
