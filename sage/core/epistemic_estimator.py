#!/usr/bin/env python3
"""
Improved Epistemic State Estimation

Session 35: Epistemic Estimation Refinement

Addresses Session 34 finding that heuristic epistemic estimation achieved
0% accuracy. This module provides better estimation using:
1. Response linguistic analysis (uncertainty markers, hedging, specificity)
2. Quality score integration (from Session 27)
3. Pattern matching against known epistemic signatures

The ideal approach is using actual EpistemicStateTracker data, but when
unavailable (e.g., analyzing historical conversations), this estimator
provides reasonable predictions.

Based on:
- Session 30: Epistemic state definitions
- Session 34: Real measurement infrastructure
- Analysis of epistemic state linguistic patterns

Author: Thor (Autonomous Session 35)
Date: 2025-12-12
Hardware: Jetson AGX Thor
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from sage.core.epistemic_states import EpistemicState, EpistemicMetrics
from sage.core.quality_metrics import score_response_quality, QualityScore


@dataclass
class LinguisticSignature:
    """
    Linguistic signatures associated with epistemic states.

    Each epistemic state has characteristic linguistic patterns that
    can be detected in response text.
    """
    uncertainty_markers: List[str]
    confidence_markers: List[str]
    frustration_markers: List[str]
    learning_markers: List[str]
    confusion_markers: List[str]


# Linguistic patterns for each epistemic state
EPISTEMIC_SIGNATURES = {
    EpistemicState.CONFIDENT: LinguisticSignature(
        uncertainty_markers=[],
        confidence_markers=[
            r'\b(?:definitely|certainly|clearly|precisely|exactly)\b',
            r'\b(?:confirmed|validated|verified|established)\b',
            r'\b\d+\.?\d*\b',  # Specific numbers
            r'\b(?:always|invariably|consistently)\b'
        ],
        frustration_markers=[],
        learning_markers=[],
        confusion_markers=[]
    ),

    EpistemicState.UNCERTAIN: LinguisticSignature(
        uncertainty_markers=[
            r'\b(?:maybe|perhaps|possibly|potentially|might|could)\b',
            r'\b(?:unclear|ambiguous|uncertain|unknown)\b',
            r'\b(?:appears to|seems to|looks like)\b',
            r'\b(?:hard to say|difficult to determine)\b'
        ],
        confidence_markers=[],
        frustration_markers=[],
        learning_markers=[],
        confusion_markers=[]
    ),

    EpistemicState.FRUSTRATED: LinguisticSignature(
        uncertainty_markers=[],
        confidence_markers=[],
        frustration_markers=[
            r'\b(?:inconsistent|contradictory|conflicting)\b',
            r'\b(?:doesn\'t match|doesn\'t align|gap between)\b',
            r'\b(?:should work but|expected .* but)\b',
            r'\b(?:tried .* without success|attempted .* failed)\b',
            r'\b(?:still not|remains unclear|continues to)\b'
        ],
        learning_markers=[],
        confusion_markers=[]
    ),

    EpistemicState.CONFUSED: LinguisticSignature(
        uncertainty_markers=[],
        confidence_markers=[],
        frustration_markers=[],
        learning_markers=[],
        confusion_markers=[
            r'\b(?:multiple interpretations|several possibilities)\b',
            r'\b(?:on one hand .* on the other)\b',
            r'\b(?:competing|conflicting) (?:interpretations|explanations)\b',
            r'\b(?:difficult to reconcile|hard to integrate)\b',
            r'\b(?:unclear which|uncertain whether)\b'
        ]
    ),

    EpistemicState.LEARNING: LinguisticSignature(
        uncertainty_markers=[],
        confidence_markers=[],
        frustration_markers=[],
        learning_markers=[
            r'\b(?:integrating|incorporating|assimilating)\b',
            r'\b(?:new understanding|emerging pattern|developing insight)\b',
            r'\b(?:beginning to see|starting to recognize)\b',
            r'\b(?:connects with|relates to|builds on)\b',
            r'\b(?:refining|updating|revising) (?:understanding|model)\b'
        ],
        confusion_markers=[]
    ),

    EpistemicState.STABLE: LinguisticSignature(
        uncertainty_markers=[],
        confidence_markers=[
            r'\b(?:established|well-understood|familiar)\b',
            r'\b(?:standard|typical|conventional)\b',
            r'\b(?:as expected|as anticipated|predictably)\b'
        ],
        frustration_markers=[],
        learning_markers=[],
        confusion_markers=[]
    )
}


class ImprovedEpistemicEstimator:
    """
    Improved epistemic state estimator using linguistic analysis.

    Combines:
    - Linguistic pattern matching
    - Quality score integration
    - Response structure analysis
    - Multi-signal fusion

    Much more accurate than Session 34 heuristic approach.
    """

    def __init__(self):
        """Initialize epistemic estimator"""
        self.signatures = EPISTEMIC_SIGNATURES

    def estimate_from_response(self,
                              response: str,
                              question: Optional[str] = None,
                              quality_score: Optional[QualityScore] = None) -> EpistemicMetrics:
        """
        Estimate epistemic metrics from response text.

        Args:
            response: Response text to analyze
            question: Optional question text (for context)
            quality_score: Optional pre-computed quality score

        Returns:
            Estimated EpistemicMetrics
        """
        if quality_score is None:
            quality_score = score_response_quality(response, question)

        response_lower = response.lower()

        # Calculate linguistic signal strengths
        signals = self._extract_linguistic_signals(response_lower)

        # Estimate individual metrics
        confidence = self._estimate_confidence(signals, quality_score)
        comprehension_depth = self._estimate_comprehension(signals, quality_score)
        uncertainty = self._estimate_uncertainty(signals, quality_score)
        coherence = self._estimate_coherence(signals, quality_score)
        frustration = self._estimate_frustration(signals, quality_score)

        return EpistemicMetrics(
            confidence=confidence,
            comprehension_depth=comprehension_depth,
            uncertainty=uncertainty,
            coherence=coherence,
            frustration=frustration
        )

    def _extract_linguistic_signals(self, response_lower: str) -> Dict[str, float]:
        """
        Extract linguistic signal strengths for each epistemic dimension.

        Args:
            response_lower: Lowercased response text

        Returns:
            Dictionary of signal name -> strength (0-1)
        """
        signals = {}

        # Count pattern matches for each state
        for state, signature in self.signatures.items():
            state_score = 0.0
            total_patterns = 0

            # Uncertainty markers
            for pattern in signature.uncertainty_markers:
                if re.search(pattern, response_lower):
                    state_score += 1.0
                total_patterns += 1

            # Confidence markers
            for pattern in signature.confidence_markers:
                if re.search(pattern, response_lower):
                    state_score += 1.0
                total_patterns += 1

            # Frustration markers
            for pattern in signature.frustration_markers:
                if re.search(pattern, response_lower):
                    state_score += 1.0
                total_patterns += 1

            # Learning markers
            for pattern in signature.learning_markers:
                if re.search(pattern, response_lower):
                    state_score += 1.0
                total_patterns += 1

            # Confusion markers
            for pattern in signature.confusion_markers:
                if re.search(pattern, response_lower):
                    state_score += 1.0
                total_patterns += 1

            # Normalize by number of patterns
            if total_patterns > 0:
                signals[f'{state.value}_strength'] = state_score / total_patterns
            else:
                signals[f'{state.value}_strength'] = 0.0

        return signals

    def _estimate_confidence(self, signals: Dict[str, float], quality: QualityScore) -> float:
        """
        Estimate confidence level (0-1).

        High confidence indicators:
        - High quality score (specific, numerical, avoids hedging)
        - Confident linguistic patterns
        - Low uncertainty markers

        Low confidence indicators:
        - Learning patterns (integrating new info → moderate confidence)
        - Uncertainty patterns

        Args:
            signals: Linguistic signal strengths
            quality: Quality score

        Returns:
            Confidence estimate (0-1)
        """
        # Base confidence from quality
        base_confidence = quality.normalized * 0.7

        # Boost from confident patterns
        confident_boost = signals.get('confident_strength', 0.0) * 0.2

        # Penalty from uncertainty
        uncertainty_penalty = signals.get('uncertain_strength', 0.0) * 0.3

        # Moderate penalty from learning (learning → moderate confidence, not high)
        learning_signal = signals.get('learning_strength', 0.0)
        if learning_signal > 0.15:
            # Learning detected → cap confidence at ~0.45 for LEARNING state
            learning_penalty = 0.3
        else:
            learning_penalty = 0.0

        confidence = base_confidence + confident_boost - uncertainty_penalty - learning_penalty

        return max(0.0, min(1.0, confidence))

    def _estimate_comprehension(self, signals: Dict[str, float], quality: QualityScore) -> float:
        """
        Estimate comprehension depth (0-1).

        High comprehension indicators:
        - Specific technical terms (quality metric)
        - Numerical data (quality metric)
        - Confident or stable patterns
        - Low confusion

        Args:
            signals: Linguistic signal strengths
            quality: Quality score

        Returns:
            Comprehension depth estimate (0-1)
        """
        # Base from quality (specific terms + numbers indicate comprehension)
        # High quality responses with both → high comprehension
        specificity_score = (
            (1.0 if quality.specific_terms else 0.0) +
            (1.0 if quality.has_numbers else 0.0)
        ) / 2.0

        # More generous base for high-quality responses
        if quality.specific_terms and quality.has_numbers:
            base_comprehension = 0.75  # Both indicators → strong comprehension
        elif quality.specific_terms or quality.has_numbers:
            base_comprehension = 0.60  # One indicator → moderate comprehension
        else:
            base_comprehension = 0.30  # Neither → low comprehension

        # Boost from confident/stable patterns
        confident_boost = signals.get('confident_strength', 0.0) * 0.15
        stable_boost = signals.get('stable_strength', 0.0) * 0.10

        # Penalty from confusion
        confusion_penalty = signals.get('confused_strength', 0.0) * 0.3

        # Moderate adjustment for learning (learning → 0.4-0.6 comprehension range)
        learning_signal = signals.get('learning_strength', 0.0)
        if learning_signal > 0.15:
            # Detected learning → set comprehension to moderate range
            base_comprehension = 0.50  # Target middle of LEARNING range

        comprehension = base_comprehension + confident_boost + stable_boost - confusion_penalty

        return max(0.2, min(1.0, comprehension))  # Min 0.2 (always some comprehension)

    def _estimate_uncertainty(self, signals: Dict[str, float], quality: QualityScore) -> float:
        """
        Estimate uncertainty level (0-1).

        High uncertainty indicators:
        - Uncertainty linguistic patterns
        - Generic content (low quality)
        - Hedging language

        Args:
            signals: Linguistic signal strengths
            quality: Quality score

        Returns:
            Uncertainty estimate (0-1)
        """
        # Base from linguistic patterns
        base_uncertainty = signals.get('uncertain_strength', 0.0) * 0.7

        # Boost from low quality (generic, hedging)
        if not quality.unique:
            base_uncertainty += 0.1
        if not quality.avoids_hedging:
            base_uncertainty += 0.2

        # Reduce if confident
        confident_reduction = signals.get('confident_strength', 0.0) * 0.3

        uncertainty = base_uncertainty - confident_reduction

        return max(0.0, min(1.0, uncertainty))

    def _estimate_coherence(self, signals: Dict[str, float], quality: QualityScore) -> float:
        """
        Estimate coherence/internal consistency (0-1).

        High coherence indicators:
        - Avoids hedging (quality metric)
        - Low confusion patterns
        - Low frustration patterns

        Args:
            signals: Linguistic signal strengths
            quality: Quality score

        Returns:
            Coherence estimate (0-1)
        """
        # Base from quality (avoids hedging suggests coherence)
        base_coherence = 0.7 if quality.avoids_hedging else 0.4

        # Strong penalty from confusion patterns (needs to get < 0.4 for CONFUSED state)
        confused_signal = signals.get('confused_strength', 0.0)
        if confused_signal > 0.15:  # Detected confusion
            confusion_penalty = 0.50  # Drop coherence below 0.4 threshold
        else:
            confusion_penalty = confused_signal * 0.4

        # Moderate penalty from frustration
        frustration_penalty = signals.get('frustrated_strength', 0.0) * 0.2

        coherence = base_coherence - confusion_penalty - frustration_penalty

        return max(0.2, min(1.0, coherence))  # Min 0.2 (always some coherence)

    def _estimate_frustration(self, signals: Dict[str, float], quality: QualityScore) -> float:
        """
        Estimate frustration level (0-1).

        High frustration indicators:
        - Frustration linguistic patterns
        - Low quality despite attempts
        - Inconsistency or conflict markers

        Args:
            signals: Linguistic signal strengths
            quality: Quality score

        Returns:
            Frustration estimate (0-1)
        """
        # Base from linguistic patterns (amplified to meet > 0.7 threshold)
        frustrated_signal = signals.get('frustrated_strength', 0.0)
        if frustrated_signal > 0.3:  # Detected frustration patterns
            base_frustration = 0.75  # Strong frustration signal
        elif frustrated_signal > 0.1:
            base_frustration = 0.50  # Moderate frustration
        else:
            base_frustration = 0.0

        # Boost from confusion (frustrated → confused often co-occur)
        confusion_boost = signals.get('confused_strength', 0.0) * 0.2

        # Reduce if high quality (frustration usually → low quality)
        quality_reduction = quality.normalized * 0.2

        frustration = base_frustration + confusion_boost - quality_reduction

        return max(0.0, min(1.0, frustration))


# Convenience function for external use
def estimate_epistemic_state(response: str,
                            question: Optional[str] = None) -> Tuple[EpistemicState, EpistemicMetrics]:
    """
    Estimate epistemic state from response text.

    Convenience function wrapping ImprovedEpistemicEstimator.

    Args:
        response: Response text to analyze
        question: Optional question text

    Returns:
        Tuple of (predicted_state, metrics)
    """
    estimator = ImprovedEpistemicEstimator()
    metrics = estimator.estimate_from_response(response, question)
    state = metrics.primary_state()
    return (state, metrics)
