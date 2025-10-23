#!/usr/bin/env python3
"""
Pattern Learner - Extracts Patterns from LLM Responses

Learns new patterns by analyzing successful LLM responses:
1. Identify question type (keywords, structure)
2. Extract response template
3. Create regex pattern for similar questions
4. Add to runtime pattern engine

This enables the system to develop "reflexive" responses from experience.
"""

import re
from typing import List, Tuple, Dict, Any
from collections import Counter
import json


class PatternLearner:
    """
    Learns conversation patterns from LLM responses

    Strategy:
    - Analyze question structure (wh-words, keywords, intent)
    - Extract response essence (template with placeholders)
    - Generate regex pattern for similar questions
    - Track pattern usage and confidence
    """

    def __init__(self, min_occurrences: int = 2, confidence_threshold: float = 0.7):
        """
        Initialize pattern learner

        Args:
            min_occurrences: Minimum times a pattern must occur before adding
            confidence_threshold: Minimum confidence to use learned pattern
        """
        self.min_occurrences = min_occurrences
        self.confidence_threshold = confidence_threshold

        # Track question-response pairs
        self.training_data: List[Tuple[str, str]] = []

        # Learned patterns (pattern_regex -> responses)
        self.learned_patterns: Dict[str, List[str]] = {}

        # Pattern usage statistics
        self.pattern_stats: Dict[str, Dict[str, Any]] = {}

        # Question clustering (similar questions)
        self.question_clusters: List[List[str]] = []

    def observe(self, question: str, response: str, metadata: Dict[str, Any] = None):
        """
        Observe a successful LLM response for learning

        Args:
            question: User question
            response: LLM response
            metadata: Optional metadata (confidence, latency, etc.)
        """
        self.training_data.append((question.lower().strip(), response.strip()))

        # Try to extract pattern immediately if we see repeating structure
        self._try_extract_pattern(question, response)

    def _try_extract_pattern(self, question: str, response: str):
        """
        Attempt to extract a learnable pattern from question-response pair

        Strategy:
        1. Identify question type (what/who/when/how/why/can/do/etc.)
        2. Extract keywords
        3. Check if similar questions exist in history
        4. If pattern emerges (2+ similar), create regex and template
        """
        q_lower = question.lower().strip()

        # Identify question type
        question_type = self._classify_question(q_lower)

        # Find similar questions in history
        similar = self._find_similar_questions(q_lower)

        if len(similar) >= self.min_occurrences:
            # We have enough similar questions - extract pattern
            pattern_regex = self._generate_pattern_regex(similar)
            response_template = self._extract_response_template(
                [r for q, r in self.training_data if q in similar]
            )

            if pattern_regex and response_template:
                # Add learned pattern
                if pattern_regex not in self.learned_patterns:
                    self.learned_patterns[pattern_regex] = []
                    self.pattern_stats[pattern_regex] = {
                        'occurrences': 0,
                        'successful_matches': 0,
                        'confidence': 0.5,  # Start with medium confidence
                        'question_type': question_type,
                        'source': 'learned'
                    }

                if response_template not in self.learned_patterns[pattern_regex]:
                    self.learned_patterns[pattern_regex].append(response_template)

                self.pattern_stats[pattern_regex]['occurrences'] += 1

    def _classify_question(self, question: str) -> str:
        """Classify question by type"""
        q = question.lower()

        if re.match(r'^(what|what\'s|whats)', q):
            return 'what'
        elif re.match(r'^(who|who\'s|whos)', q):
            return 'who'
        elif re.match(r'^(when|when\'s|whens)', q):
            return 'when'
        elif re.match(r'^(where|where\'s|wheres)', q):
            return 'where'
        elif re.match(r'^(why|how come)', q):
            return 'why'
        elif re.match(r'^(how|how\'s|hows)', q):
            return 'how'
        elif re.match(r'^(can|could|would|will|should)', q):
            return 'modal'
        elif re.match(r'^(do|does|did|are|is|was)', q):
            return 'yes_no'
        elif re.match(r'^(tell me|show me|explain)', q):
            return 'command'
        else:
            return 'unknown'

    def _find_similar_questions(self, question: str) -> List[str]:
        """
        Find similar questions in training data using simple similarity

        Similarity based on:
        - Shared keywords (nouns, verbs)
        - Same question type
        - Similar length
        """
        q_keywords = self._extract_keywords(question)
        q_type = self._classify_question(question)
        q_len = len(question.split())

        similar = []

        for past_q, _ in self.training_data:
            past_keywords = self._extract_keywords(past_q)
            past_type = self._classify_question(past_q)
            past_len = len(past_q.split())

            # Check similarity
            shared_keywords = len(q_keywords & past_keywords)
            same_type = (q_type == past_type)
            similar_length = abs(q_len - past_len) <= 3

            if same_type and shared_keywords >= 1 and similar_length:
                similar.append(past_q)

        return similar

    def _extract_keywords(self, text: str) -> set:
        """Extract meaningful keywords from text (simple approach)"""
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'can', 'may', 'might', 'must', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'its', 'our', 'their'
        }

        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if w not in stop_words and len(w) > 2}

    def _generate_pattern_regex(self, similar_questions: List[str]) -> str:
        """
        Generate a regex pattern that matches similar questions

        Strategy:
        - Find common words across all similar questions
        - Create flexible pattern with optional words
        """
        if not similar_questions:
            return None

        # Find common keywords
        all_keywords = [self._extract_keywords(q) for q in similar_questions]

        if not all_keywords:
            return None

        # Keywords present in most questions
        keyword_counts = Counter()
        for keywords in all_keywords:
            keyword_counts.update(keywords)

        # Keep keywords present in at least 50% of questions
        common_keywords = [
            kw for kw, count in keyword_counts.items()
            if count >= len(similar_questions) * 0.5
        ]

        if not common_keywords:
            # Fallback: use question type pattern
            q_type = self._classify_question(similar_questions[0])
            type_patterns = {
                'what': r'(?i)what.{0,30}',
                'who': r'(?i)who.{0,30}',
                'when': r'(?i)when.{0,30}',
                'where': r'(?i)where.{0,30}',
                'why': r'(?i)why.{0,30}',
                'how': r'(?i)how.{0,30}',
                'modal': r'(?i)(can|could|would|will).{0,30}',
                'command': r'(?i)(tell me|show me|explain).{0,30}',
            }
            return type_patterns.get(q_type, r'.{0,50}')

        # Build pattern with common keywords (order-independent)
        # Use lookaheads for flexible matching
        pattern_parts = [f'(?=.*{re.escape(kw)})' for kw in common_keywords]
        pattern = r'(?i)' + ''.join(pattern_parts) + r'.{0,100}'

        return pattern

    def _extract_response_template(self, responses: List[str]) -> str:
        """
        Extract a response template from similar responses

        Strategy:
        - If responses are very similar, use common structure
        - Otherwise, pick the most frequent response style
        """
        if not responses:
            return None

        # For now, use the most common response
        # (could be more sophisticated - extract common phrases, etc.)
        response_counter = Counter(responses)
        most_common = response_counter.most_common(1)[0][0]

        return most_common

    def get_learned_patterns(self) -> Dict[str, List[str]]:
        """Get all learned patterns (for integration with PatternResponseEngine)"""
        return self.learned_patterns

    def get_pattern_confidence(self, pattern: str) -> float:
        """Get confidence score for a learned pattern"""
        if pattern in self.pattern_stats:
            stats = self.pattern_stats[pattern]
            # Confidence increases with successful uses
            base_confidence = 0.5
            usage_bonus = min(0.3, stats['occurrences'] * 0.05)
            return min(0.95, base_confidence + usage_bonus)
        return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'total_observations': len(self.training_data),
            'learned_patterns': len(self.learned_patterns),
            'pattern_details': {
                pattern: {
                    **stats,
                    'response_count': len(self.learned_patterns[pattern])
                }
                for pattern, stats in self.pattern_stats.items()
            }
        }

    def save_patterns(self, filepath: str):
        """Save learned patterns to file"""
        data = {
            'learned_patterns': self.learned_patterns,
            'pattern_stats': self.pattern_stats,
            'training_data': self.training_data
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_patterns(self, filepath: str):
        """Load learned patterns from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.learned_patterns = data.get('learned_patterns', {})
        self.pattern_stats = data.get('pattern_stats', {})
        self.training_data = [tuple(pair) for pair in data.get('training_data', [])]


# Test the pattern learner
if __name__ == "__main__":
    print("="*60)
    print("Testing Pattern Learner")
    print("="*60)

    learner = PatternLearner(min_occurrences=2)

    # Simulate learning from LLM responses
    test_cases = [
        ("What is your name?", "I'm SAGE, an AI assistant."),
        ("What's your name?", "My name is SAGE."),
        ("Who are you?", "I'm SAGE, here to help."),
        ("Tell me about yourself", "I'm SAGE, an AI system designed to assist you."),
        ("What do you do?", "I help answer questions and assist with tasks."),
        ("What can you do?", "I can answer questions, provide information, and help solve problems."),
    ]

    print("\nObserving question-response pairs...")
    for q, r in test_cases:
        print(f"  Q: {q}")
        print(f"  A: {r}")
        learner.observe(q, r)

    print(f"\n{learner.get_stats()}")

    # Check learned patterns
    patterns = learner.get_learned_patterns()
    print(f"\nLearned {len(patterns)} patterns:")
    for pattern, responses in patterns.items():
        print(f"\n  Pattern: {pattern}")
        print(f"  Responses: {responses}")
        print(f"  Confidence: {learner.get_pattern_confidence(pattern):.2f}")
