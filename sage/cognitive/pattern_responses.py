"""
Pattern-Based Cognitive Response Engine

Fast (<1ms) pattern matching for common conversational patterns.
Handles greetings, acknowledgments, status queries, and simple interactions.

For complex queries that don't match patterns, returns None to signal
that deeper processing (LLM or human) is needed.

This is NOT a chatbot - it's a fast cognitive shortcut for obvious responses.
"""

import re
import random
import time
from typing import Optional, Dict, List, Tuple


class PatternResponseEngine:
    """
    Fast pattern-based cognitive responses

    Uses regex patterns to match common conversational inputs
    and generate contextually appropriate responses.
    """

    def __init__(self):
        """Initialize pattern engine with response templates"""

        # Pattern categories (pattern → response templates)
        self.patterns = {
            # Greetings
            r'\b(hello|hi|hey|greetings)\b': [
                "Hello! I'm listening.",
                "Hi there! How can I help?",
                "Hey! What's on your mind?",
                "Greetings! I'm ready to talk."
            ],

            # Good morning/evening
            r'\bgood (morning|afternoon|evening)\b': [
                "Good {0}! How are you?",
                "Good {0}! What would you like to discuss?",
                "{0} to you too!"
            ],

            # How are you
            r'\bhow (are|r) (you|u)\b': [
                "I'm operating normally. All systems functional.",
                "I'm doing well, thank you. How about you?",
                "All systems running smoothly. What can I do for you?"
            ],

            # What are you doing
            r'\bwhat.*?(doing|up to)\b': [
                "I'm listening and ready to help.",
                "Monitoring sensors and waiting for input.",
                "Standing by for conversation."
            ],

            # Simple acknowledgments
            r'^\s*(okay|ok|alright|sure|yeah|yep|yes)\s*$': [
                "Understood.",
                "Got it.",
                "Acknowledged.",
                "Noted."
            ],

            # Thank you
            r'\b(thanks|thank you|thx)\b': [
                "You're welcome!",
                "Happy to help!",
                "Anytime!",
                "My pleasure."
            ],

            # Questions about SAGE
            r'\bwhat (are|is) (you|sage)\b': [
                "I'm SAGE - a real-time cognition system running on this Jetson device.",
                "SAGE: Situation-Aware Governance Engine. I orchestrate attention and resources.",
                "I'm an attention orchestrator that manages sensors, cognition, and effectors."
            ],

            # Status/health check
            r'\b(status|health|check)\b': [
                "All systems operational. Audio sensor active, cognitive layer responsive.",
                "Status: Green. Sensors online, ATP available, ready for tasks.",
                "Systems nominal. Standing by."
            ],

            # Can you hear me
            r'\bcan (you|u) hear( me)?\b': [
                "Yes, I can hear you clearly.",
                "Loud and clear!",
                "Audio sensor receiving your speech."
            ],

            # Test/testing
            r'\b(test|testing)\b': [
                "Test acknowledged. All systems responding.",
                "Testing successful. I'm receiving your input.",
                "Test confirmed. Audio and cognition working."
            ],

            # Simple farewells
            r'\b(bye|goodbye|see you|talk later)\b': [
                "Goodbye! Talk to you soon.",
                "See you later!",
                "Farewell!",
                "Take care!"
            ],

            # Repeat that / what
            r'^\s*(what|huh|pardon|repeat)\s*[\?]?\s*$': [
                "Could you please repeat that?",
                "I didn't catch that. Could you say it again?",
                "Sorry, I missed that. What did you say?"
            ],

            # REMOVED catch-all question pattern - let complex questions go to LLM
            # Questions deserve thoughtful LLM responses, not generic fast-path replies
        }

        # Compile patterns for efficiency
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), responses)
            for pattern, responses in self.patterns.items()
        ]

        # Conversation state
        self.last_match = None
        self.match_count = {}

        # Statistics
        self.total_queries = 0
        self.matched_queries = 0
        self.total_processing_time = 0.0

    def generate_response(self, user_input: str) -> Optional[str]:
        """
        Generate response from pattern matching

        Args:
            user_input: User's speech transcription

        Returns:
            Response string if pattern matched, None if no match (needs deeper processing)
        """
        start_time = time.time()
        self.total_queries += 1

        # Normalize input
        text = user_input.strip()

        # Try to match patterns
        for pattern, responses in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                # Track matches for variety
                pattern_str = pattern.pattern
                self.match_count[pattern_str] = self.match_count.get(pattern_str, 0) + 1

                # Select response (avoid repetition)
                if len(responses) > 1:
                    # Rotate through responses
                    idx = self.match_count[pattern_str] % len(responses)
                    response_template = responses[idx]
                else:
                    response_template = responses[0]

                # Format response with match groups if needed
                try:
                    if match.groups():
                        response = response_template.format(*match.groups())
                    else:
                        response = response_template
                except (IndexError, KeyError):
                    response = response_template

                # Update stats
                self.matched_queries += 1
                self.total_processing_time += (time.time() - start_time)
                self.last_match = pattern_str

                return response

        # No pattern matched - signal need for deeper processing
        self.total_processing_time += (time.time() - start_time)
        return None

    def get_stats(self) -> Dict:
        """Get pattern matching statistics"""
        return {
            'total_queries': self.total_queries,
            'matched_queries': self.matched_queries,
            'match_rate': self.matched_queries / self.total_queries if self.total_queries > 0 else 0.0,
            'avg_processing_time_ms': (self.total_processing_time / self.total_queries * 1000) if self.total_queries > 0 else 0.0,
            'pattern_counts': self.match_count.copy(),
            'last_match': self.last_match
        }

    def add_pattern(self, pattern: str, responses: List[str]):
        """
        Add custom pattern at runtime

        Args:
            pattern: Regex pattern string
            responses: List of possible response templates
        """
        compiled = re.compile(pattern, re.IGNORECASE)
        self.compiled_patterns.append((compiled, responses))

    def clear_stats(self):
        """Reset statistics"""
        self.total_queries = 0
        self.matched_queries = 0
        self.total_processing_time = 0.0
        self.match_count.clear()
        self.last_match = None

    def get_mrh_profile(self) -> Dict[str, str]:
        """
        Get Markov Relevancy Horizon profile for pattern matching

        Pattern matching operates at:
        - deltaR (spatial): local - no external resources, pure local regex
        - deltaT (temporal): ephemeral - instant responses, no state persistence
        - deltaC (complexity): simple - direct pattern matching, no reasoning

        Returns:
            MRH profile dict
        """
        return {
            'deltaR': 'local',
            'deltaT': 'ephemeral',
            'deltaC': 'simple'
        }


class CognitiveRouter:
    """
    Routes queries to appropriate cognitive processor

    Fast path: Pattern matching (~1ms)
    Slow path: Returns None, signaling need for LLM/human
    """

    def __init__(self):
        """Initialize cognitive router"""
        self.pattern_engine = PatternResponseEngine()

        # Future: Could add other processors
        # self.llm_engine = LocalLLMEngine()
        # self.knowledge_base = KnowledgeBase()

        self.stats = {
            'pattern_hits': 0,
            'pattern_misses': 0,
            'fallback_requests': 0
        }

    def process(self, user_input: str) -> Tuple[Optional[str], str]:
        """
        Process user input and generate response

        Args:
            user_input: User's speech transcription

        Returns:
            (response_text, method_used)
            - response_text: Generated response or None if needs fallback
            - method_used: 'pattern', 'llm', 'fallback', etc.
        """
        # Try pattern matching first (fast path)
        response = self.pattern_engine.generate_response(user_input)

        if response:
            self.stats['pattern_hits'] += 1
            return (response, 'pattern')

        # Pattern didn't match - needs deeper processing
        self.stats['pattern_misses'] += 1
        self.stats['fallback_requests'] += 1

        # Could add LLM here later
        # if self.llm_engine:
        #     response = self.llm_engine.generate(user_input)
        #     return (response, 'llm')

        # No processors available - return None
        return (None, 'fallback')

    def get_stats(self) -> Dict:
        """Get router statistics"""
        pattern_stats = self.pattern_engine.get_stats()
        return {
            **self.stats,
            'pattern_details': pattern_stats
        }
