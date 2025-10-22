"""
SNARC Salience Detectors

Individual components for 5D salience assessment:
- SurpriseDetector: Prediction error
- NoveltyDetector: Memory comparison
- ArousalDetector: Signal magnitude
- RewardEstimator: Goal relevance
- ConflictDetector: Cross-sensor disagreement
"""

from .surprise import SurpriseDetector
from .novelty import NoveltyDetector
from .arousal import ArousalDetector
from .reward import RewardEstimator
from .conflict import ConflictDetector

__all__ = [
    'SurpriseDetector',
    'NoveltyDetector',
    'ArousalDetector',
    'RewardEstimator',
    'ConflictDetector'
]
