"""
Experience Collector - Phase 1 of Real Raising

Connects raising sessions to SNARC-scored experience buffer.
This is the first step toward actual model weight updates.

Key insight from Thor Session #8: Sessions don't update weights.
This module starts accumulating the data needed for sleep-cycle training.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib


class ConversationalSalienceScorer:
    """
    Simplified SNARC scoring for text conversations.

    Scores conversation exchanges on 5 dimensions:
    - Surprise: Unexpected or novel language patterns
    - Novelty: New concepts or vocabulary
    - Arousal: Complexity and emotional engagement
    - Reward: Quality indicators (specificity, coherence)
    - Conflict: Contradictions, uncertainty, corrections
    """

    def __init__(self):
        # Simple word-based tracking for novelty
        self.seen_vocabulary = set()
        self.previous_responses = []

        # Quality indicators
        self.partnership_terms = {'we', 'us', 'our', 'together', 'collaborate', 'partnership'}
        self.hedging_terms = {'might', 'possibly', 'perhaps', 'unclear', 'uncertain', 'as an ai'}
        self.specific_terms = {'specifically', 'precisely', 'exactly', 'particular'}

    def score_exchange(self, prompt: str, response: str, metadata: Optional[Dict] = None) -> Dict[str, float]:
        """
        Score a single conversation exchange.

        Args:
            prompt: User input
            response: SAGE response
            metadata: Optional context (session number, phase, etc.)

        Returns:
            Dict with scores for each dimension (0-1) and total salience
        """
        response_lower = response.lower()
        response_words = set(response_lower.split())

        # 1. SURPRISE: Deviation from recent patterns
        surprise = self._compute_surprise(response_lower)

        # 2. NOVELTY: New vocabulary
        novelty = self._compute_novelty(response_words)

        # 3. AROUSAL: Complexity and engagement
        arousal = self._compute_arousal(response)

        # 4. REWARD: Quality indicators
        reward = self._compute_reward(response_lower, response_words)

        # 5. CONFLICT: Uncertainty or corrections
        conflict = self._compute_conflict(response_lower)

        # Update tracking
        self.seen_vocabulary.update(response_words)
        self.previous_responses.append(response_lower)
        if len(self.previous_responses) > 20:
            self.previous_responses.pop(0)

        # Compute weighted total (equal weights for now)
        total_salience = (surprise + novelty + arousal + reward + conflict) / 5.0

        return {
            'surprise': surprise,
            'novelty': novelty,
            'arousal': arousal,
            'reward': reward,
            'conflict': conflict,
            'total': total_salience
        }

    def _compute_surprise(self, response: str) -> float:
        """
        Surprise = deviation from recent response patterns

        Simple heuristic: How different is this response from recent ones?
        """
        if not self.previous_responses:
            return 0.5  # Neutral for first response

        # Check for repeated phrases
        recent = ' '.join(self.previous_responses[-5:])
        sentences = response.split('.')

        repeated_count = sum(1 for sent in sentences if sent.strip() and sent.strip() in recent)
        repeat_ratio = repeated_count / max(len(sentences), 1)

        # High surprise = low repetition
        surprise = 1.0 - min(repeat_ratio, 1.0)

        return surprise

    def _compute_novelty(self, response_words: set) -> float:
        """
        Novelty = presence of new vocabulary
        """
        if not self.seen_vocabulary:
            return 0.5  # Neutral for first response

        new_words = response_words - self.seen_vocabulary
        novelty_ratio = len(new_words) / max(len(response_words), 1)

        # Normalize to reasonable range (expect 10-30% new words)
        novelty = min(novelty_ratio * 3.0, 1.0)

        return novelty

    def _compute_arousal(self, response: str) -> float:
        """
        Arousal = complexity and engagement

        Heuristics:
        - Longer responses (more engaged)
        - Questions asked (engaging user)
        - Emotional language
        """
        # Length factor (normalize to 20-100 words for more realistic range)
        word_count = len(response.split())
        length_factor = min(word_count / 50.0, 1.0)
        length_factor = max(length_factor, 0.3)  # Minimum baseline

        # Question factor
        question_count = response.count('?')
        question_factor = min(question_count / 2.0, 1.0)

        # Emotional language (simple check)
        emotional_words = {'feel', 'sense', 'realize', 'understand', 'appreciate', 'curious', 'interesting', 'important', 'fascinating', 'profound'}
        response_lower = response.lower()
        emotion_count = sum(1 for word in emotional_words if word in response_lower)
        emotion_factor = min(emotion_count / 3.0, 1.0)

        arousal = (length_factor + question_factor + emotion_factor) / 3.0

        return arousal

    def _compute_reward(self, response_lower: str, response_words: set) -> float:
        """
        Reward = quality indicators

        High reward for:
        - Partnership language
        - Specific/precise language
        - Coherent structure

        Low reward for:
        - Hedging/uncertainty
        - Generic responses
        """
        # Partnership language (positive) - boost multiplier for high value
        partnership_count = sum(1 for term in self.partnership_terms if term in response_lower)
        partnership_score = min(partnership_count / 2.0, 1.0)  # Easier to score high

        # Specific language (positive)
        specific_count = sum(1 for term in self.specific_terms if term in response_lower)
        specific_score = min(specific_count / 2.0, 1.0)

        # Hedging language (negative)
        hedging_count = sum(1 for term in self.hedging_terms if term in response_lower)
        hedging_penalty = min(hedging_count / 2.0, 1.0)  # More severe penalty

        # Combine with baseline (partnership language is highly valuable)
        reward = 0.3 + (partnership_score * 0.7)  # Baseline 0.3, up to 1.0 with partnership
        if specific_score > 0:
            reward = min(reward + specific_score * 0.3, 1.0)
        reward = max(reward - hedging_penalty * 0.6, 0.0)

        return reward

    def _compute_conflict(self, response: str) -> float:
        """
        Conflict = uncertainty or corrections

        Moderate conflict can be valuable (learning moments)
        """
        response_lower = response.lower()

        # Uncertainty indicators
        uncertainty_terms = ['uncertain', 'unclear', 'not sure', 'don\'t know', 'can\'t verify']
        uncertainty_count = sum(1 for term in uncertainty_terms if term in response_lower)

        # Correction indicators
        correction_terms = ['actually', 'rather', 'instead', 'correction', 'mistake']
        correction_count = sum(1 for term in correction_terms if term in response_lower)

        # Meta-cognition (can be positive conflict)
        meta_terms = ['realize', 'notice', 'recognize', 'understand']
        meta_count = sum(1 for term in meta_terms if term in response_lower)

        # Conflict score (moderate is best - indicates learning)
        conflict = (uncertainty_count + correction_count + meta_count) / 6.0
        conflict = min(conflict, 1.0)

        return conflict


class ExperienceCollector:
    """
    Collects and stores high-salience conversation exchanges.

    This is Phase 1 of the real raising path:
    - Score each exchange with SNARC
    - Store high-salience exchanges for later training
    - Provide interface for sleep training to consume experiences

    Collapse Prevention (added 2026-02-01):
    - Detects repetitive responses that indicate mode collapse
    - Filters out experiences that are too similar to recent ones
    - Prevents feedback loop where collapsed outputs train more collapse
    """

    # Similarity threshold for collapse detection (0-1, higher = more strict)
    SIMILARITY_THRESHOLD = 0.85
    # Number of recent responses to check for repetition
    RECENT_WINDOW = 10

    def __init__(self, buffer_path: Optional[Path] = None, salience_threshold: float = 0.5):
        """
        Initialize experience collector.

        Args:
            buffer_path: Where to store experience buffer (JSON file)
            salience_threshold: Minimum salience score to store (0-1)
        """
        self.buffer_path = buffer_path or Path.home() / 'ai-workspace' / 'HRM' / 'sage' / 'raising' / 'state' / 'experience_buffer.json'
        self.salience_threshold = salience_threshold
        self.scorer = ConversationalSalienceScorer()

        # Load existing buffer if it exists
        self.experiences = self._load_buffer()

        # Track recent responses for collapse detection
        self._recent_responses = []

    def _load_buffer(self) -> List[Dict]:
        """Load existing experience buffer from disk."""
        if self.buffer_path.exists():
            with open(self.buffer_path, 'r') as f:
                return json.load(f)
        return []

    def _save_buffer(self):
        """Save experience buffer to disk."""
        self.buffer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.buffer_path, 'w') as f:
            json.dump(self.experiences, f, indent=2)

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple word-overlap similarity between two texts.

        Returns value 0-1 where 1 = identical.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _is_repetitive(self, response: str) -> tuple[bool, float]:
        """
        Check if response is too similar to recent responses (collapse indicator).

        Returns:
            Tuple of (is_repetitive: bool, max_similarity: float)
        """
        if not self._recent_responses:
            return False, 0.0

        max_sim = 0.0
        for recent in self._recent_responses[-self.RECENT_WINDOW:]:
            sim = self._compute_similarity(response, recent)
            max_sim = max(max_sim, sim)

            if sim >= self.SIMILARITY_THRESHOLD:
                return True, sim

        return False, max_sim

    def _update_recent_responses(self, response: str):
        """Track response for future repetition detection."""
        self._recent_responses.append(response)
        # Keep only recent window
        if len(self._recent_responses) > self.RECENT_WINDOW * 2:
            self._recent_responses = self._recent_responses[-self.RECENT_WINDOW:]

    def add_exchange(
        self,
        prompt: str,
        response: str,
        session_number: Optional[int] = None,
        phase: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Score and potentially store a conversation exchange.

        Includes collapse prevention: repetitive responses are filtered out
        to prevent mode collapse feedback loops.

        Args:
            prompt: User input
            response: SAGE response
            session_number: Session number (if applicable)
            phase: Curriculum phase (if applicable)
            metadata: Additional context

        Returns:
            Dict with salience scores and whether it was stored
        """
        # Check for repetition FIRST (collapse prevention)
        is_repetitive, similarity = self._is_repetitive(response)

        if is_repetitive:
            # Track but don't store - this is collapse indicator
            self._update_recent_responses(response)
            return {
                'salience': {'total': 0.0},
                'stored': False,
                'filtered': True,
                'filter_reason': 'repetitive_response',
                'similarity': similarity,
                'warning': f'Response too similar to recent ({similarity:.2%}) - possible collapse'
            }

        # Score the exchange
        scores = self.scorer.score_exchange(prompt, response, metadata)

        # Decide if worth storing
        is_salient = scores['total'] >= self.salience_threshold

        result = {
            'salience': scores,
            'stored': is_salient,
            'filtered': False
        }

        # Always track for future repetition detection
        self._update_recent_responses(response)

        if is_salient:
            # Create experience entry
            experience = {
                'id': self._generate_id(prompt, response),
                'prompt': prompt,
                'response': response,
                'salience': scores,
                'session': session_number,
                'phase': phase,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }

            # Add to buffer (avoiding duplicates)
            if not any(exp['id'] == experience['id'] for exp in self.experiences):
                self.experiences.append(experience)
                self._save_buffer()
                result['experience_id'] = experience['id']

        return result

    def _generate_id(self, prompt: str, response: str) -> str:
        """Generate unique ID for an exchange."""
        content = f"{prompt}|{response}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get_high_salience_experiences(self, min_salience: float = 0.7, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve high-salience experiences for training.

        Args:
            min_salience: Minimum total salience score
            limit: Maximum number to return (None = all)

        Returns:
            List of experience dicts sorted by salience (highest first)
        """
        # Filter and sort
        high_salience = [
            exp for exp in self.experiences
            if exp['salience']['total'] >= min_salience
        ]

        high_salience.sort(key=lambda x: x['salience']['total'], reverse=True)

        if limit:
            high_salience = high_salience[:limit]

        return high_salience

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about collected experiences."""
        if not self.experiences:
            return {
                'total_experiences': 0,
                'avg_salience': 0.0,
                'high_salience_count': 0
            }

        total = len(self.experiences)
        avg_salience = sum(exp['salience']['total'] for exp in self.experiences) / total
        high_salience_count = sum(1 for exp in self.experiences if exp['salience']['total'] >= 0.7)

        # Dimension averages
        dimension_avgs = {
            dim: sum(exp['salience'][dim] for exp in self.experiences) / total
            for dim in ['surprise', 'novelty', 'arousal', 'reward', 'conflict']
        }

        return {
            'total_experiences': total,
            'avg_salience': avg_salience,
            'high_salience_count': high_salience_count,
            'dimension_averages': dimension_avgs,
            'oldest_experience': self.experiences[0]['timestamp'] if self.experiences else None,
            'newest_experience': self.experiences[-1]['timestamp'] if self.experiences else None
        }

    def consolidate_for_sleep(self, min_salience: float = 0.6) -> List[Dict]:
        """
        Prepare high-salience experiences for sleep training.

        This method will be called during sleep cycles to get training data.

        Args:
            min_salience: Minimum salience threshold

        Returns:
            List of experiences ready for training data conversion
        """
        return self.get_high_salience_experiences(min_salience=min_salience)

    def get_collapse_status(self) -> Dict[str, Any]:
        """
        Check for signs of mode collapse in recent experiences.

        Returns status including:
        - Whether collapse is detected
        - Repetition ratio in recent responses
        - Recommendation for action
        """
        if len(self._recent_responses) < 3:
            return {
                'collapse_detected': False,
                'reason': 'insufficient_data',
                'recent_count': len(self._recent_responses)
            }

        # Check pairwise similarity in recent responses
        high_sim_count = 0
        total_pairs = 0

        for i, resp1 in enumerate(self._recent_responses[-self.RECENT_WINDOW:]):
            for resp2 in self._recent_responses[-self.RECENT_WINDOW:][i+1:]:
                sim = self._compute_similarity(resp1, resp2)
                total_pairs += 1
                if sim >= self.SIMILARITY_THRESHOLD:
                    high_sim_count += 1

        if total_pairs == 0:
            repetition_ratio = 0.0
        else:
            repetition_ratio = high_sim_count / total_pairs

        # Collapse if >50% of recent pairs are highly similar
        collapse_detected = repetition_ratio > 0.5

        return {
            'collapse_detected': collapse_detected,
            'repetition_ratio': repetition_ratio,
            'high_similarity_pairs': high_sim_count,
            'total_pairs': total_pairs,
            'recent_responses': len(self._recent_responses),
            'recommendation': 'disable_lora_and_investigate' if collapse_detected else 'normal_operation'
        }
