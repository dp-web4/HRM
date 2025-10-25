"""
SNARC-Optimized Context Window Memory Manager

Treats the LLM's context window as a SNARC-filtered short-term memory buffer.
Experiences are scored for salience (Surprise, Novelty, Arousal, Reward, Conflict)
and prioritized for retention in the limited context window.

High-salience experiences are extracted to long-term storage.
Low-salience experiences age out as context fills up.
"""

import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque


@dataclass
class ConversationTurn:
    """Single conversation turn with metadata"""
    speaker: str  # "User" or "Assistant"
    text: str
    timestamp: float
    salience_score: float = 0.0  # SNARC score (0.0 - 1.0)
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SNARCMemoryManager:
    """
    SNARC-based memory manager for context window optimization.

    Uses 5D salience scoring to prioritize which conversation turns
    remain in the context window as it approaches capacity.
    """

    def __init__(self, max_tokens: int = 127000, tokens_per_turn: int = 50):
        """
        Initialize SNARC memory manager.

        Args:
            max_tokens: Maximum tokens available for conversation history
            tokens_per_turn: Average tokens per conversation turn
        """
        self.max_tokens = max_tokens
        self.tokens_per_turn = tokens_per_turn
        self.max_turns = max_tokens // tokens_per_turn  # ~2540 turns!

        # Conversation buffer (most recent turns, SNARC-filtered)
        self.conversation_buffer: List[ConversationTurn] = []

        # Long-term memory (high-salience extractions)
        self.long_term_memory: List[ConversationTurn] = []

        # Statistics
        self.total_turns = 0
        self.extracted_to_longterm = 0

    def add_turn(self, speaker: str, text: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a conversation turn with automatic salience scoring.

        Args:
            speaker: "User" or "Assistant"
            text: Text content
            metadata: Optional metadata (path, latency, etc.)
        """
        turn = ConversationTurn(
            speaker=speaker,
            text=text,
            timestamp=time.time(),
            metadata=metadata or {}
        )

        # Calculate SNARC salience
        turn.salience_score = self._calculate_salience(turn)

        # Add to buffer
        self.conversation_buffer.append(turn)
        self.total_turns += 1

        # Manage buffer capacity
        self._manage_buffer_capacity()

    def _calculate_salience(self, turn: ConversationTurn) -> float:
        """
        Calculate SNARC salience score (0.0 - 1.0).

        SNARC dimensions:
        - Surprise: Unexpected responses, errors, novel patterns
        - Novelty: First-time topics, unique phrases
        - Arousal: Emotional content, exclamations, questions
        - Reward: Positive feedback, learning moments, breakthroughs
        - Conflict: Contradictions, corrections, disagreements

        Args:
            turn: Conversation turn to score

        Returns:
            Salience score (0.0 - 1.0)
        """
        scores = {}

        text_lower = turn.text.lower()

        # SURPRISE: Unexpected events, errors, edge cases
        surprise_keywords = ['error', 'unexpected', 'strange', 'weird', 'wow', 'what', 'huh']
        scores['surprise'] = min(1.0, sum(1 for kw in surprise_keywords if kw in text_lower) * 0.3)

        # NOVELTY: New topics, unique content
        # (Simplified: questions and longer responses tend to be novel)
        is_question = '?' in turn.text
        is_long = len(turn.text) > 100
        scores['novelty'] = 0.5 if is_question else (0.3 if is_long else 0.1)

        # AROUSAL: Emotional content, emphasis
        arousal_keywords = ['!', 'amazing', 'terrible', 'love', 'hate', 'excited', 'frustrated']
        arousal_count = sum(turn.text.count(kw) for kw in arousal_keywords)
        scores['arousal'] = min(1.0, arousal_count * 0.4)

        # REWARD: Learning, success, positive feedback
        reward_keywords = ['learned', 'understand', 'makes sense', 'got it', 'thank', 'great',
                          'excellent', 'perfect', 'milestone', 'breakthrough']
        scores['reward'] = min(1.0, sum(1 for kw in reward_keywords if kw in text_lower) * 0.4)

        # CONFLICT: Disagreements, corrections, confusion
        conflict_keywords = ['no', 'wrong', 'incorrect', 'actually', 'correction', 'disagree',
                            'but', 'however', 'mistake', 'issue', 'problem']
        scores['conflict'] = min(1.0, sum(1 for kw in conflict_keywords if kw in text_lower) * 0.3)

        # Metadata bonuses
        if turn.metadata.get('path') == 'slow':
            # Slow path responses used LLM reasoning - likely more salient
            scores['novelty'] = min(1.0, scores['novelty'] + 0.2)

        if turn.metadata.get('learned'):
            # Pattern learning occurred - high salience!
            scores['reward'] = min(1.0, scores['reward'] + 0.5)

        # Combined SNARC score (weighted average)
        salience = (
            scores['surprise'] * 0.25 +
            scores['novelty'] * 0.20 +
            scores['arousal'] * 0.15 +
            scores['reward'] * 0.25 +
            scores['conflict'] * 0.15
        )

        return min(1.0, salience)

    def _manage_buffer_capacity(self) -> None:
        """
        Manage buffer capacity by extracting high-salience memories
        and aging out low-salience ones when approaching capacity.
        """
        if len(self.conversation_buffer) <= self.max_turns:
            return  # Still have room

        # Extract high-salience experiences to long-term memory
        high_salience_threshold = 0.7
        high_salience_turns = [
            turn for turn in self.conversation_buffer
            if turn.salience_score >= high_salience_threshold
        ]

        # Add to long-term memory (avoid duplicates)
        existing_timestamps = {turn.timestamp for turn in self.long_term_memory}
        for turn in high_salience_turns:
            if turn.timestamp not in existing_timestamps:
                self.long_term_memory.append(turn)
                self.extracted_to_longterm += 1

        # Sort buffer by salience (descending)
        self.conversation_buffer.sort(key=lambda t: t.salience_score, reverse=True)

        # Keep top N turns (most salient)
        keep_count = int(self.max_turns * 0.9)  # Keep 90% of capacity
        self.conversation_buffer = self.conversation_buffer[:keep_count]

        # Re-sort by timestamp to maintain chronological order
        self.conversation_buffer.sort(key=lambda t: t.timestamp)

    def get_context_for_llm(self, include_longterm: bool = True) -> List[Tuple[str, str]]:
        """
        Get conversation history formatted for LLM context.

        Args:
            include_longterm: Whether to include salient long-term memories

        Returns:
            List of (speaker, text) tuples for LLM prompt
        """
        context = []

        # Add salient long-term memories as summary
        if include_longterm and self.long_term_memory:
            # Take most recent N high-salience memories
            recent_longterm = sorted(self.long_term_memory,
                                    key=lambda t: t.timestamp,
                                    reverse=True)[:10]

            if recent_longterm:
                # Add as context marker
                context.append(("System", "[Salient past experiences recalled from long-term memory:]"))
                for turn in reversed(recent_longterm):  # Chronological order
                    context.append((turn.speaker, turn.text))
                context.append(("System", "[Current conversation continues:]"))

        # Add current conversation buffer
        for turn in self.conversation_buffer:
            context.append((turn.speaker, turn.text))

        return context

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            'total_turns': self.total_turns,
            'buffer_size': len(self.conversation_buffer),
            'buffer_capacity': self.max_turns,
            'buffer_utilization': len(self.conversation_buffer) / self.max_turns,
            'longterm_memories': len(self.long_term_memory),
            'extracted_count': self.extracted_to_longterm,
            'avg_buffer_salience': sum(t.salience_score for t in self.conversation_buffer) / len(self.conversation_buffer) if self.conversation_buffer else 0.0,
            'avg_longterm_salience': sum(t.salience_score for t in self.long_term_memory) / len(self.long_term_memory) if self.long_term_memory else 0.0,
        }

    def save_longterm_to_disk(self, filepath: str) -> None:
        """Save long-term memories to disk (for persistence across sessions)"""
        import json

        data = [{
            'speaker': turn.speaker,
            'text': turn.text,
            'timestamp': turn.timestamp,
            'salience_score': turn.salience_score,
            'metadata': turn.metadata
        } for turn in self.long_term_memory]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_longterm_from_disk(self, filepath: str) -> None:
        """Load long-term memories from disk (for cross-session continuity)"""
        import json
        import os

        if not os.path.exists(filepath):
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.long_term_memory = [
            ConversationTurn(
                speaker=item['speaker'],
                text=item['text'],
                timestamp=item['timestamp'],
                salience_score=item['salience_score'],
                metadata=item.get('metadata', {})
            )
            for item in data
        ]
