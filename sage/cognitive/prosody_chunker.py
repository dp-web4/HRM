#!/usr/bin/env python3
"""
Prosody-Aware Chunking for Natural Speech Synthesis

Implements breath-group aligned text chunking based on linguistic research.
Replaces primitive punctuation-based chunking with prosodic boundary detection.

Key insight: Chunk at prosodic boundaries (clauses, phrases) that align with
biological breath groups (10-15 words, 2-4 seconds), not arbitrary punctuation.

References:
- Lieberman (1966) - Breath groups as fundamental speech units
- PMC2945274 - Breath group parameters (12.4 words avg, 2.3s duration)
- PMC4240966 - Breath groups align with grammatical junctures (94%)
"""

import re
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ProsodicChunk:
    """
    A chunk of speech aligned with prosodic structure.

    Represents a natural unit of speech production that:
    1. Aligns with breath groups (respiratory constraints)
    2. Respects linguistic boundaries (clauses, phrases)
    3. Reduces cognitive load (working memory constraints)
    """
    text: str
    boundary_type: str  # "IP" (intonational), "ip" (intermediate), "BREATH", "NATURAL"
    word_count: int
    estimated_duration: float  # seconds (based on average speaking rate)

    # Prosodic features for TTS optimization
    boundary_tone: str  # "L-L%" (final fall), "L-H%" (continuation rise), etc.
    continuation: bool  # True if more content coming

    def get_tts_hints(self) -> dict:
        """
        Generate TTS hints for natural prosody.

        Returns metadata that TTS systems can use to:
        - Insert appropriate pauses
        - Adjust boundary intonation
        - Reset pitch for new phrases
        """
        return {
            'pause_after': self._get_pause_duration(),
            'boundary_tone': self.boundary_tone,
            'pitch_reset': not self.continuation
        }

    def _get_pause_duration(self) -> int:
        """
        Pause duration in milliseconds based on boundary type.

        Follows SSML conventions:
        - Sentence: 800-1200ms
        - Clause: 400-500ms
        - Natural break: 200-300ms
        """
        if self.boundary_type == "IP":
            return 800  # Intonational phrase (sentence)
        elif self.boundary_type == "ip":
            return 400  # Intermediate phrase (clause)
        elif "BREATH" in self.boundary_type:
            return 300  # Forced breath (overflow)
        else:
            return 200  # Natural pause (minimal)


class ProsodyAwareChunker:
    """
    Detects prosodic boundaries for natural speech chunking.

    Uses linguistic patterns to identify natural break points that align
    with breath groups, rather than relying on punctuation alone.

    Breathing constraints (from research):
    - Min breath group: 5 words (~1.5 seconds)
    - Target breath group: 12 words (~3 seconds)
    - Max breath group: 18 words (~4.5 seconds)
    """

    def __init__(
        self,
        min_phrase_words: int = 5,
        target_phrase_words: int = 12,
        max_phrase_words: int = 18,
        speaking_rate: float = 3.8  # words per second (spontaneous speech average)
    ):
        """
        Initialize prosodic chunker with breath group parameters.

        Args:
            min_phrase_words: Minimum words before considering a break
            target_phrase_words: Preferred breath group size
            max_phrase_words: Maximum words before forced break
            speaking_rate: Words per second for duration estimation
        """
        self.min_phrase_words = min_phrase_words
        self.target_phrase_words = target_phrase_words
        self.max_phrase_words = max_phrase_words
        self.speaking_rate = speaking_rate

    def is_prosodic_boundary(
        self,
        buffer: str,
        new_chunk: str = ""
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if we've crossed a prosodic boundary.

        Checks in priority order:
        1. Sentence boundaries (Intonational Phrases)
        2. Clause boundaries (Intermediate Phrases)
        3. Breath group overflow (forced safety net)
        4. Natural break points at target size

        Args:
            buffer: Current accumulated text
            new_chunk: Newly added text (unused, for future extensions)

        Returns:
            (is_boundary, boundary_type) tuple
        """
        if not buffer.strip():
            return (False, None)

        word_count = len(buffer.strip().split())

        # Priority 1: Sentence boundaries (Intonational Phrase)
        # These are ALWAYS natural breath group boundaries
        if self._is_sentence_end(buffer):
            return (True, "IP")

        # Priority 2: Clause boundaries (Intermediate Phrase)
        # Only check after minimum phrase size to avoid tiny chunks
        if word_count >= self.min_phrase_words:
            if self._is_clause_boundary(buffer):
                return (True, "ip")

        # Priority 3: Breath group overflow (safety net)
        # Prevent unbounded buffering by forcing break at max size
        if word_count >= self.max_phrase_words:
            return (True, f"BREATH({word_count}w)")

        # Priority 4: Target breath group size with natural break
        # Emit at target size if there's a natural pause point
        if word_count >= self.target_phrase_words:
            if self._has_natural_break(buffer):
                return (True, "NATURAL")

        # No boundary detected - continue accumulating
        return (False, None)

    def _is_sentence_end(self, text: str) -> bool:
        """
        Check if text ends with sentence boundary.

        Excludes common abbreviations that use periods but aren't sentence ends.
        """
        text = text.strip()
        if not text:
            return False

        # Check for terminal punctuation
        if not re.search(r'[.!?]\s*$', text):
            return False

        # Exclude abbreviations (Dr., Mr., Mrs., etc.)
        abbrevs = [
            'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.',
            'etc.', 'e.g.', 'i.e.', 'vs.', 'Inc.', 'Ltd.', 'Co.'
        ]
        for abbrev in abbrevs:
            if text.endswith(abbrev):
                return False

        return True

    def _is_clause_boundary(self, text: str) -> bool:
        """
        Detect clause boundaries beyond simple punctuation.

        Identifies grammatical junctures where speakers naturally pause:
        - Coordinating conjunctions (and, but, or, so)
        - Subordinating clauses (when, if, because, although)
        - Relative clauses (which, who, that, where)

        Research shows 94% of breaths occur at these boundaries.
        """
        text = text.strip()

        # Coordinating conjunctions after comma
        # "I went to the store, and then I came home"
        #                      ↑ natural breath point
        if re.search(r',\s+(and|but|or|so|yet|for|nor)\s+\w', text):
            return True

        # Subordinating clause starters
        # "I'll help you, if you need it"
        #              ↑ natural breath point
        if re.search(r',\s+(when|if|because|although|while|since|unless|until)\s+\w', text):
            return True

        # Relative clauses
        # "The system, which processes audio, is running"
        #           ↑ natural breath point
        if re.search(r',\s+(which|who|that|where|whose)\s+\w', text):
            return True

        # Semicolons are strong clause boundaries
        if ';' in text:
            return True

        return False

    def _has_natural_break(self, text: str) -> bool:
        """
        Check for natural pause points at target breath group size.

        Identifies locations where pausing would feel natural:
        - After prepositional phrases
        - After introductory phrases
        - Between list items
        """
        text = text.strip()

        # Prepositional phrases
        # "I'm working on the project [break] with my team"
        if re.search(r'\s+(in|on|at|by|with|from|to|for|about|during|after|before)\s+\w+\s*$', text):
            return True

        # Introductory discourse markers
        # "However, [natural pause] the results were unexpected"
        if re.search(r'^(However|Therefore|Moreover|Furthermore|Additionally|Meanwhile|Nevertheless),', text):
            return True

        # List items (series of commas suggests enumeration)
        # "I need milk, eggs, bread, [natural break]"
        if text.count(',') >= 2:
            return True

        # After time expressions
        if re.search(r'(today|tomorrow|yesterday|now|then|currently|recently),', text):
            return True

        return False

    def find_nearest_graceful_break(self, text: str, max_lookback: int = 5) -> int:
        """
        Find nearest graceful break point when forced to split.

        When buffer overflows, looks backward to find a natural boundary
        rather than breaking mid-phrase.

        Args:
            text: Text to analyze
            max_lookback: Maximum words to look backward

        Returns:
            Index of nearest graceful break point (word count)
        """
        words = text.split()

        # Look backwards from end for break points
        for i in range(len(words) - 1, max(0, len(words) - max_lookback), -1):
            partial = ' '.join(words[:i])

            # Check if this location is a natural boundary
            if self._is_clause_boundary(partial):
                return i
            if self._has_natural_break(partial):
                return i

        # No natural break found - return full length
        return len(words)

    def create_chunk(
        self,
        text: str,
        boundary_type: str,
        is_final: bool = False
    ) -> ProsodicChunk:
        """
        Create a ProsodicChunk with computed metadata.

        Args:
            text: The chunk text
            boundary_type: Type of prosodic boundary
            is_final: Whether this is the final chunk

        Returns:
            ProsodicChunk with prosodic metadata
        """
        word_count = len(text.strip().split())
        estimated_duration = word_count / self.speaking_rate

        # Determine boundary tone based on finality
        # L-L% = falling tone (final)
        # L-H% = rising tone (continuation)
        boundary_tone = "L-L%" if is_final else "L-H%"

        return ProsodicChunk(
            text=text.strip(),
            boundary_type=boundary_type,
            word_count=word_count,
            estimated_duration=estimated_duration,
            boundary_tone=boundary_tone,
            continuation=not is_final
        )


# Convenience function for backward compatibility
def is_sentence_complete(text: str) -> bool:
    """Legacy function - use ProsodyAwareChunker instead."""
    chunker = ProsodyAwareChunker()
    return chunker._is_sentence_end(text)
