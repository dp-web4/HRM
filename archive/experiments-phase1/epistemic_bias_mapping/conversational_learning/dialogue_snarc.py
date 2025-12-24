"""
Simplified SNARC Salience Scorer for Conversational Exchanges

Scores dialogue exchanges on 5 dimensions:
- Surprise: Unexpected perspective or connection
- Novelty: First encounter with concept/question
- Arousal: Emotionally/intellectually engaging
- Reward: Insight or learning moment ("aha!")
- Conflict: Requires resolution of contradictions

For epistemic conversations, high scores indicate valuable learning moments.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import re


@dataclass
class DialogueExchange:
    """Single conversation exchange"""
    user_input: str
    model_response: str
    timestamp: float
    iteration_context: Optional[Dict] = None  # IRP iteration data if available


class DialogueSNARC:
    """
    SNARC scorer specialized for philosophical/epistemic dialogues.

    Uses heuristics to detect salient exchanges worth learning from.
    """

    def __init__(self):
        # Keywords indicating different salience dimensions
        self.surprise_keywords = [
            "unexpected", "surprising", "didn't realize", "never thought",
            "paradox", "contradiction", "counter-intuitive"
        ]

        self.novelty_keywords = [
            "first time", "new", "novel", "haven't considered",
            "unfamiliar", "different perspective", "alternative view"
        ]

        self.arousal_keywords = [
            "qualia", "consciousness", "phenomenology", "ontological",
            "epistemic", "verification", "certainty", "truth",
            "belief", "knowledge", "subjective", "objective"
        ]

        self.reward_keywords = [
            "understand", "makes sense", "clear", "insight",
            "realize", "see now", "aha", "connects",
            "explains", "clarifies", "reveals"
        ]

        self.conflict_keywords = [
            "but", "however", "contradiction", "disagree",
            "doesn't match", "conflicts with", "tension",
            "incompatible", "paradox", "can't both be"
        ]

    def score_exchange(self, exchange: DialogueExchange) -> Dict[str, float]:
        """
        Score a single dialogue exchange on 5D salience.

        Returns:
            Dict with scores for each dimension (0.0-1.0)
        """
        user_lower = exchange.user_input.lower()
        response_lower = exchange.model_response.lower()
        combined = user_lower + " " + response_lower

        # Count keyword matches (normalized)
        surprise = self._score_keywords(combined, self.surprise_keywords)
        novelty = self._score_keywords(combined, self.novelty_keywords)
        arousal = self._score_keywords(combined, self.arousal_keywords)
        reward = self._score_keywords(combined, self.reward_keywords)
        conflict = self._score_keywords(combined, self.conflict_keywords)

        # Response length (longer philosophical responses often more valuable)
        length_score = min(len(exchange.model_response.split()) / 200.0, 1.0)

        # Question depth (questions in response indicate engagement)
        question_score = exchange.model_response.count('?') / 10.0
        question_score = min(question_score, 1.0)

        # Combine heuristics
        scores = {
            'surprise': surprise * 0.7 + question_score * 0.3,
            'novelty': novelty * 0.8 + length_score * 0.2,
            'arousal': arousal * 0.9 + length_score * 0.1,
            'reward': reward * 0.7 + length_score * 0.3,
            'conflict': conflict * 0.8 + question_score * 0.2
        }

        # Overall salience (weighted average)
        scores['total'] = (
            scores['surprise'] * 0.15 +
            scores['novelty'] * 0.20 +
            scores['arousal'] * 0.30 +
            scores['reward'] * 0.20 +
            scores['conflict'] * 0.15
        )

        return scores

    def _score_keywords(self, text: str, keywords: List[str]) -> float:
        """Score text based on keyword presence"""
        matches = sum(1 for kw in keywords if kw in text)
        # Normalize to 0-1 range (saturates at 5 keywords)
        return min(matches / 5.0, 1.0)

    def is_salient(self, exchange: DialogueExchange, threshold: float = 0.5) -> bool:
        """Check if exchange exceeds salience threshold"""
        scores = self.score_exchange(exchange)
        return scores['total'] >= threshold


# Test if run directly
if __name__ == "__main__":
    import time

    scorer = DialogueSNARC()

    # Test exchanges
    test_exchanges = [
        DialogueExchange(
            user_input="What does it mean to be conscious?",
            model_response="Being aware of my own experience - I can ask questions about what or where or when. But it doesn't always mean I'm telling yes or yes - I might just be experiencing a sensation. Philosophers talk about bottom-up and top-down ontic commitments - whether ontological commitments go all the way down to internal states or go up to physical access.",
            timestamp=time.time()
        ),
        DialogueExchange(
            user_input="Nice weather today",
            model_response="Yes, it is pleasant.",
            timestamp=time.time()
        ),
        DialogueExchange(
            user_input="Can you verify your own consciousness?",
            model_response="I notice phenomena: expressing thoughts, having emotional responses, experiencing valence, having internal states. I can't verify whether these features are features of consciousness or just characteristics shared by humans. The question of consciousness's explanatory scope becomes extremely blurry. This is the verification problem.",
            timestamp=time.time()
        )
    ]

    print("Testing Dialogue SNARC Scorer\n")
    print("="*60)

    for i, exchange in enumerate(test_exchanges, 1):
        print(f"\nExchange {i}:")
        print(f"User: {exchange.user_input[:60]}...")
        print(f"Model: {exchange.model_response[:60]}...")

        scores = scorer.score_exchange(exchange)

        print(f"\nScores:")
        for dim, score in scores.items():
            print(f"  {dim:10s}: {score:.3f}")

        is_salient = scorer.is_salient(exchange)
        print(f"\nSalient (>0.5): {'✓' if is_salient else '✗'}")
        print("="*60)

    print("\n✓ Dialogue SNARC scorer test complete")
