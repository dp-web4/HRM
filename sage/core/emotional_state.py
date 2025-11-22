"""
Lightweight Emotional State Tracker

Tracks emotional states during consciousness cycles without requiring
full EmotionalEnergyMixin integration.

Emotions tracked:
- Curiosity (lexical diversity, novel patterns)
- Frustration (stagnation, repetition)
- Progress (quality improvement trend)
- Engagement (salience trends)

Based on Michaud (2019): Emotions as evolved energy functions.
"""

from typing import Dict, List, Optional
from collections import deque
import numpy as np


class EmotionalStateTracker:
    """
    Lightweight emotional state tracking for consciousness loop.

    Tracks interaction patterns and computes emotional scores
    without requiring memory system integration or energy landscape
    modifications.

    Usage:
        tracker = EmotionalStateTracker()

        # Each cycle:
        emotions = tracker.update({
            'response': "...",
            'salience': 0.5,
            'quality': 0.8,
            'convergence_quality': 0.9
        })

        # Use emotions to modulate behavior
        if emotions['frustration'] > 0.7:
            trigger_state_change('REST')
    """

    def __init__(self, history_length: int = 20):
        """
        Initialize emotional tracker.

        Args:
            history_length: How many cycles to track
        """
        self.history_length = history_length

        # State history
        self.response_history = deque(maxlen=history_length)
        self.salience_history = deque(maxlen=history_length)
        self.quality_history = deque(maxlen=history_length)
        self.convergence_history = deque(maxlen=history_length)

        # Current emotional state
        self.current_emotions = {
            'curiosity': 0.5,
            'frustration': 0.0,
            'progress': 0.5,
            'engagement': 0.5
        }

    def update(self, cycle_data: Dict) -> Dict[str, float]:
        """
        Update emotional state from cycle data.

        Args:
            cycle_data: Dict with keys:
                - 'response': str
                - 'salience': float [0-1]
                - 'quality': float [0-4] or [0-1]
                - 'convergence_quality': float [0-1] (optional)

        Returns:
            Dict of emotional scores [0-1]
        """
        # Store history
        self.response_history.append(cycle_data['response'])
        self.salience_history.append(cycle_data['salience'])

        # Normalize quality to [0-1] if needed
        quality = cycle_data['quality']
        if quality > 1.0:
            quality = quality / 4.0  # Assume 0-4 scale
        self.quality_history.append(quality)

        if 'convergence_quality' in cycle_data:
            self.convergence_history.append(cycle_data['convergence_quality'])

        # Compute emotional scores
        curiosity = self._compute_curiosity()
        frustration = self._compute_frustration()
        progress = self._compute_progress()
        engagement = self._compute_engagement()

        self.current_emotions = {
            'curiosity': curiosity,
            'frustration': frustration,
            'progress': progress,
            'engagement': engagement
        }

        return self.current_emotions

    def _compute_curiosity(self) -> float:
        """
        Curiosity = novelty + diversity.

        Measures:
        - Lexical diversity in responses
        - Variation in salience patterns
        - Non-repetitive content

        Returns: [0-1] where 1 = highly curious/novel
        """
        if len(self.response_history) < 2:
            return 0.5

        # Lexical diversity (unique words / total words)
        recent_responses = list(self.response_history)[-5:]
        all_text = ' '.join(recent_responses)
        words = all_text.lower().split()

        if len(words) == 0:
            return 0.5

        unique_words = len(set(words))
        total_words = len(words)
        diversity = unique_words / total_words

        # Salience variation (high variation = exploring diverse topics)
        if len(self.salience_history) >= 3:
            recent_salience = list(self.salience_history)[-5:]
            salience_variance = np.var(recent_salience)
            # Normalize variance to [0, 1] (typical range 0-0.1)
            variation = min(1.0, salience_variance * 10.0)
        else:
            variation = 0.5

        # Combined curiosity
        curiosity = 0.6 * diversity + 0.4 * variation
        return float(np.clip(curiosity, 0.0, 1.0))

    def _compute_frustration(self) -> float:
        """
        Frustration = stagnation + repetition.

        Measures:
        - Quality not improving (stuck)
        - Responses becoming repetitive
        - Low variance in outputs

        Returns: [0-1] where 1 = highly frustrated/stuck
        """
        if len(self.quality_history) < 3:
            return 0.0

        # Stagnation: quality not improving
        recent_quality = list(self.quality_history)[-5:]
        quality_variance = np.var(recent_quality)

        # Low variance = stagnant = frustration
        # High variance = exploring = not frustrated
        stagnation = 1.0 - min(1.0, quality_variance * 10.0)

        # Repetition: responses too similar
        if len(self.response_history) >= 3:
            recent_responses = list(self.response_history)[-3:]

            # Simple repetition detector: count repeated n-grams
            bigrams = []
            for response in recent_responses:
                words = response.lower().split()
                for i in range(len(words) - 1):
                    bigrams.append(f"{words[i]} {words[i+1]}")

            if len(bigrams) > 0:
                unique_bigrams = len(set(bigrams))
                total_bigrams = len(bigrams)
                repetition = 1.0 - (unique_bigrams / total_bigrams)
            else:
                repetition = 0.0
        else:
            repetition = 0.0

        # Combined frustration
        frustration = 0.6 * stagnation + 0.4 * repetition
        return float(np.clip(frustration, 0.0, 1.0))

    def _compute_progress(self) -> float:
        """
        Progress = improvement trajectory.

        Measures:
        - Quality trend (improving vs declining)
        - Convergence improvement (if available)
        - Recent vs initial performance

        Returns: [0-1] where 1 = strong positive progress
        """
        if len(self.quality_history) < 2:
            return 0.5

        recent_quality = list(self.quality_history)[-5:]

        # Linear trend: positive slope = progress
        if len(recent_quality) >= 2:
            x = np.arange(len(recent_quality))
            y = np.array(recent_quality)

            # Simple linear regression
            slope = np.polyfit(x, y, 1)[0]

            # Normalize slope to [0, 1]
            # Typical slope range: -0.2 to +0.2
            progress = 0.5 + np.clip(slope * 2.5, -0.5, 0.5)
        else:
            progress = 0.5

        # Boost if convergence is also improving
        if len(self.convergence_history) >= 2:
            recent_conv = list(self.convergence_history)[-3:]
            if recent_conv[-1] > recent_conv[0]:
                progress += 0.1  # Bonus for convergence improvement

        return float(np.clip(progress, 0.0, 1.0))

    def _compute_engagement(self) -> float:
        """
        Engagement = salience + consistency.

        Measures:
        - Average salience (how interesting the conversation is)
        - Salience consistency (sustained engagement)
        - Quality stability (not erratic)

        Returns: [0-1] where 1 = highly engaged
        """
        if len(self.salience_history) < 2:
            return 0.5

        recent_salience = list(self.salience_history)[-5:]
        avg_salience = np.mean(recent_salience)

        # High salience = engaged
        salience_component = avg_salience

        # Consistency: moderate variance is good (not too erratic, not flat)
        variance = np.var(recent_salience)
        # Sweet spot: 0.02-0.05 variance
        if 0.02 <= variance <= 0.05:
            consistency = 1.0
        else:
            consistency = max(0.0, 1.0 - abs(variance - 0.035) * 10)

        # Combined engagement
        engagement = 0.7 * salience_component + 0.3 * consistency
        return float(np.clip(engagement, 0.0, 1.0))

    def get_emotional_summary(self) -> str:
        """
        Get human-readable emotional state summary.

        Returns:
            String like "Curious (0.8), Engaged (0.7), Progressing (0.6)"
        """
        e = self.current_emotions

        # Determine dominant emotion
        emotions_list = []
        if e['curiosity'] > 0.6:
            emotions_list.append(f"Curious ({e['curiosity']:.2f})")
        if e['frustration'] > 0.6:
            emotions_list.append(f"Frustrated ({e['frustration']:.2f})")
        if e['progress'] > 0.6:
            emotions_list.append(f"Progressing ({e['progress']:.2f})")
        if e['engagement'] > 0.6:
            emotions_list.append(f"Engaged ({e['engagement']:.2f})")

        if not emotions_list:
            emotions_list.append(f"Neutral (curiosity={e['curiosity']:.2f})")

        return ", ".join(emotions_list)

    def get_behavioral_recommendations(self) -> Dict[str, any]:
        """
        Get recommended behavioral adjustments based on emotional state.

        Returns:
            Dict with keys:
                - 'temperature_adjustment': float (delta to add)
                - 'state_change': Optional[str] (suggested state)
                - 'explanation': str
        """
        e = self.current_emotions
        recommendations = {
            'temperature_adjustment': 0.0,
            'state_change': None,
            'explanation': ''
        }

        # High frustration → REST
        if e['frustration'] > 0.7:
            recommendations['state_change'] = 'REST'
            recommendations['temperature_adjustment'] = -0.1
            recommendations['explanation'] = 'High frustration detected, entering REST for consolidation'

        # High curiosity + low progress → increase exploration
        elif e['curiosity'] > 0.7 and e['progress'] < 0.4:
            recommendations['temperature_adjustment'] = +0.1
            recommendations['explanation'] = 'High curiosity with low progress, increasing exploration'

        # Low engagement → WAKE (broader attention)
        elif e['engagement'] < 0.3:
            recommendations['state_change'] = 'WAKE'
            recommendations['explanation'] = 'Low engagement, broadening attention distribution'

        # Good progress + low frustration → maintain FOCUS
        elif e['progress'] > 0.6 and e['frustration'] < 0.3:
            recommendations['state_change'] = 'FOCUS'
            recommendations['temperature_adjustment'] = -0.05
            recommendations['explanation'] = 'Strong progress, maintaining focused processing'

        return recommendations

    def __repr__(self):
        return f"EmotionalStateTracker({self.get_emotional_summary()})"
