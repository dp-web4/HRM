"""
SNARC Integration for LLM IRP Plugin

Connects conversational learning with salience-based memory:
- Score conversation exchanges using 5D SNARC
- Filter salient exchanges for training
- Enable selective learning from high-value interactions

Based on Sprout's successful implementation (November 2025).
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class DialogueExchange:
    """A single question-answer exchange in conversation."""
    question: str
    answer: str
    irp_info: Optional[Dict] = None
    timestamp: Optional[float] = None


class DialogueSNARC:
    """
    5D Salience scoring for dialogue exchanges.

    Dimensions:
    - Surprise: Unexpected questions or novel topics
    - Novelty: New information or perspectives
    - Arousal: Engaging or thought-provoking content
    - Reward: Valuable or insightful exchanges
    - Conflict: Paradoxes, contradictions, or challenging questions
    """

    def __init__(self):
        """Initialize SNARC scorer with dimension weights."""
        # Equal weighting for all dimensions
        self.weights = {
            'surprise': 0.2,
            'novelty': 0.2,
            'arousal': 0.2,
            'reward': 0.2,
            'conflict': 0.2
        }

    def score_exchange(
        self,
        exchange: DialogueExchange,
        conversation_history: Optional[List[DialogueExchange]] = None
    ) -> Dict[str, float]:
        """
        Score a dialogue exchange on 5 SNARC dimensions.

        Args:
            exchange: The exchange to score
            conversation_history: Prior exchanges for context

        Returns:
            Dict with scores for each dimension plus total
        """
        question = exchange.question
        answer = exchange.answer
        irp_info = exchange.irp_info

        # 1. SURPRISE: Novel question patterns or unexpected topics
        surprise = self._score_surprise(question, conversation_history)

        # 2. NOVELTY: New vocabulary or concepts introduced
        novelty = self._score_novelty(question, answer, conversation_history)

        # 3. AROUSAL: Engaging complexity and depth
        arousal = self._score_arousal(question, answer)

        # 4. REWARD: Quality and value of response
        reward = self._score_reward(answer, irp_info)

        # 5. CONFLICT: Paradoxes, contradictions, meta-cognition
        conflict = self._score_conflict(question, answer)

        # Total weighted salience
        total_salience = (
            self.weights['surprise'] * surprise +
            self.weights['novelty'] * novelty +
            self.weights['arousal'] * arousal +
            self.weights['reward'] * reward +
            self.weights['conflict'] * conflict
        )

        return {
            'surprise': surprise,
            'novelty': novelty,
            'arousal': arousal,
            'reward': reward,
            'conflict': conflict,
            'total_salience': total_salience
        }

    def _score_surprise(
        self,
        question: str,
        history: Optional[List[DialogueExchange]]
    ) -> float:
        """Score surprise: how unexpected is this question?"""
        if not history:
            return 0.0  # First exchange has no baseline

        # Check for topic shifts
        question_words = set(question.lower().split())

        # Compare with recent history
        recent_topics = set()
        if history:
            for ex in history[-3:]:  # Last 3 exchanges
                recent_topics.update(ex.question.lower().split())

        # Surprise = proportion of new words
        if not recent_topics:
            return 0.0

        new_words = question_words - recent_topics
        surprise = len(new_words) / len(question_words)

        return min(1.0, surprise)

    def _score_novelty(
        self,
        question: str,
        answer: str,
        history: Optional[List[DialogueExchange]]
    ) -> float:
        """Score novelty: new vocabulary or concepts introduced."""
        # Measure vocabulary richness
        all_words = (question + " " + answer).lower().split()
        unique_words = set(all_words)

        # Type-token ratio (lexical diversity)
        if len(all_words) == 0:
            return 0.0

        ttr = len(unique_words) / len(all_words)

        # Novelty keywords (meta-cognitive, epistemic, philosophical)
        novel_keywords = {
            'understand', 'know', 'aware', 'conscious', 'process',
            'think', 'believe', 'certain', 'uncertain', 'paradox',
            'meta', 'self', 'reference', 'aware', 'cognition'
        }

        keyword_count = sum(1 for word in unique_words if word in novel_keywords)
        keyword_score = min(1.0, keyword_count / 3)  # 3+ keywords = max score

        # Combine TTR and keyword presence
        novelty = 0.5 * ttr + 0.5 * keyword_score

        return min(1.0, novelty)

    def _score_arousal(self, question: str, answer: str) -> float:
        """Score arousal: complexity and depth of exchange."""
        # Question complexity (length, punctuation, question words)
        q_words = question.split()
        q_length_score = min(1.0, len(q_words) / 20)  # 20+ words = complex

        question_words = {'what', 'why', 'how', 'when', 'where', 'who'}
        has_question_word = any(w in question.lower() for w in question_words)

        # Answer depth (length, structure)
        a_words = answer.split()
        a_length_score = min(1.0, len(a_words) / 50)  # 50+ words = thorough

        # Combine factors
        arousal = (
            0.3 * q_length_score +
            0.2 * (1.0 if has_question_word else 0.0) +
            0.5 * a_length_score
        )

        return min(1.0, arousal)

    def _score_reward(self, answer: str, irp_info: Optional[Dict]) -> float:
        """Score reward: quality and value of response."""
        # Response quality indicators
        a_words = answer.split()

        # Length (good answers are detailed but not excessive)
        ideal_length = 50
        length_diff = abs(len(a_words) - ideal_length)
        length_score = max(0.0, 1.0 - length_diff / ideal_length)

        # Structure (has punctuation, capitalization)
        has_punctuation = any(c in answer for c in '.!?')
        has_capital = any(c.isupper() for c in answer)
        structure_score = 1.0 if (has_punctuation and has_capital) else 0.5

        # IRP convergence (if available)
        irp_score = 0.5  # Default if no IRP info
        if irp_info:
            # Good convergence = low energy
            final_energy = irp_info.get('final_energy', 1.0)
            irp_score = 1.0 - min(1.0, final_energy)

        # Combine factors
        reward = (
            0.3 * length_score +
            0.2 * structure_score +
            0.5 * irp_score
        )

        return min(1.0, reward)

    def _score_conflict(self, question: str, answer: str) -> float:
        """Score conflict: paradoxes, meta-cognition, self-reference."""
        # Meta-cognitive keywords
        meta_keywords = {
            'aware', 'know', 'understand', 'think', 'believe',
            'certain', 'uncertain', 'accurate', 'true', 'false',
            'paradox', 'contradiction', 'self', 'myself', 'you',
            'generate', 'process', 'create', 'discover'
        }

        text = (question + " " + answer).lower()
        meta_count = sum(1 for keyword in meta_keywords if keyword in text)

        # Self-reference patterns
        self_ref_patterns = [
            'you know', 'you understand', 'you are', 'you can',
            'i am', 'i know', 'i think', 'your answer',
            'this conversation', 'my response'
        ]

        self_ref_count = sum(1 for pattern in self_ref_patterns if pattern in text)

        # Question words indicating uncertainty
        uncertainty_words = {'uncertain', 'maybe', 'might', 'could', 'perhaps'}
        uncertainty_count = sum(1 for word in uncertainty_words if word in text)

        # Combine signals
        conflict = min(1.0, (
            0.4 * min(1.0, meta_count / 3) +
            0.4 * min(1.0, self_ref_count / 2) +
            0.2 * min(1.0, uncertainty_count / 2)
        ))

        return conflict

    def is_salient(
        self,
        exchange: DialogueExchange,
        threshold: float = 0.15,
        history: Optional[List[DialogueExchange]] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Determine if exchange is salient enough for memory storage.

        Args:
            exchange: Exchange to evaluate
            threshold: Minimum total salience score
            history: Conversation history for context

        Returns:
            Tuple of (is_salient, scores_dict)
        """
        scores = self.score_exchange(exchange, history)
        is_salient = scores['total_salience'] >= threshold

        return is_salient, scores


class ConversationalMemory:
    """
    Memory system for conversational learning.

    Stores:
    - All exchanges (full conversation history)
    - Salient exchanges (filtered by SNARC)
    - Training data (formatted for sleep-cycle training)
    """

    def __init__(self, salience_threshold: float = 0.15):
        """
        Initialize conversational memory.

        Args:
            salience_threshold: Minimum salience to store exchange
        """
        self.snarc_scorer = DialogueSNARC()
        self.salience_threshold = salience_threshold

        # Storage
        self.all_exchanges: List[DialogueExchange] = []
        self.salient_exchanges: List[Tuple[DialogueExchange, Dict]] = []  # (exchange, scores)

    def record_exchange(
        self,
        question: str,
        answer: str,
        irp_info: Optional[Dict] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Record an exchange and determine if it's salient.

        Args:
            question: User's question
            answer: Model's response
            irp_info: IRP refinement info

        Returns:
            Tuple of (is_salient, scores)
        """
        exchange = DialogueExchange(
            question=question,
            answer=answer,
            irp_info=irp_info
        )

        # Store in full history
        self.all_exchanges.append(exchange)

        # Score with SNARC
        is_salient, scores = self.snarc_scorer.is_salient(
            exchange,
            threshold=self.salience_threshold,
            history=self.all_exchanges[:-1]  # Exclude current
        )

        # Store if salient
        if is_salient:
            self.salient_exchanges.append((exchange, scores))

        return is_salient, scores

    def get_salient_for_training(self) -> List[Dict[str, str]]:
        """
        Get salient exchanges formatted for training.

        Returns:
            List of dicts with 'question' and 'answer' keys
        """
        return [
            {
                'question': ex.question,
                'answer': ex.answer
            }
            for ex, scores in self.salient_exchanges
        ]

    def get_statistics(self) -> Dict[str, any]:
        """Get memory statistics."""
        total = len(self.all_exchanges)
        salient = len(self.salient_exchanges)
        capture_rate = (salient / total * 100) if total > 0 else 0.0

        avg_salience = 0.0
        if self.salient_exchanges:
            saliences = [scores['total_salience'] for _, scores in self.salient_exchanges]
            avg_salience = sum(saliences) / len(saliences)

        return {
            'total_exchanges': total,
            'salient_exchanges': salient,
            'capture_rate': capture_rate,
            'avg_salience': avg_salience
        }


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("SNARC Integration Test")
    print("="*70)

    # Initialize memory
    memory = ConversationalMemory(salience_threshold=0.15)

    # Test exchanges
    exchanges = [
        ("What is knowledge?", "Knowledge is information that has been acquired through learning or experience."),
        ("Can you know with certainty?", "Absolute certainty is difficult to achieve in most domains of knowledge."),
        ("Are you aware of this conversation?", "I process each exchange but don't maintain continuous awareness between conversations."),
        ("What's 2+2?", "4"),
    ]

    for question, answer in exchanges:
        is_salient, scores = memory.record_exchange(question, answer)

        print(f"\nQ: {question[:50]}...")
        print(f"Salience: {scores['total_salience']:.3f} | Salient: {is_salient}")
        print(f"  Surprise={scores['surprise']:.3f}, Novelty={scores['novelty']:.3f}, "
              f"Arousal={scores['arousal']:.3f}, Reward={scores['reward']:.3f}, Conflict={scores['conflict']:.3f}")

    # Statistics
    stats = memory.get_statistics()
    print("\n" + "="*70)
    print(f"Total exchanges: {stats['total_exchanges']}")
    print(f"Salient exchanges: {stats['salient_exchanges']}")
    print(f"Capture rate: {stats['capture_rate']:.1f}%")
    print(f"Avg salience: {stats['avg_salience']:.3f}")
    print("="*70)
