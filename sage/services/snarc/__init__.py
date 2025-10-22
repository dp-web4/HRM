"""
SNARC System Service - "Sensor of Sensors"

Assesses salience across entire sensor field using 5D framework:
- Surprise: Deviation from prediction
- Novelty: Difference from past experiences
- Arousal: Intensity/urgency
- Reward: Goal relevance
- Conflict: Cross-sensor disagreement

Provides attention recommendations to SAGE kernel.
"""

from .data_structures import SalienceReport, SalienceBreakdown, Outcome
from .snarc_service import SNARCService

__all__ = ['SalienceReport', 'SalienceBreakdown', 'Outcome', 'SNARCService']
