"""
SNARC Data Structures

Core data types for salience assessment and reporting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class CognitiveStance(Enum):
    """Recommended cognitive stance based on salience pattern"""
    CURIOUS_UNCERTAINTY = "curious-uncertainty"  # Novel patterns, high surprise/novelty
    CONFIDENT_EXECUTION = "confident-execution"  # Known routines, low surprise/novelty
    SKEPTICAL_VERIFICATION = "skeptical-verification"  # High conflict, suspicious
    EXPLORATORY = "exploratory"  # High arousal, moderate novelty
    FOCUSED_ATTENTION = "focused-attention"  # High reward relevance


@dataclass
class SalienceBreakdown:
    """
    5D Salience Assessment

    Each dimension scored 0.0-1.0:
    - surprise: Deviation from prediction (prediction error)
    - novelty: Difference from past experiences (memory comparison)
    - arousal: Intensity/urgency of signal (magnitude)
    - reward: Relevance to current goals (value estimation)
    - conflict: Cross-sensor disagreement (coherence check)
    """
    surprise: float  # 0.0-1.0
    novelty: float  # 0.0-1.0
    arousal: float  # 0.0-1.0
    reward: float  # 0.0-1.0
    conflict: float  # 0.0-1.0

    def __post_init__(self):
        """Validate all scores are in valid range"""
        for field_name, value in [
            ('surprise', self.surprise),
            ('novelty', self.novelty),
            ('arousal', self.arousal),
            ('reward', self.reward),
            ('conflict', self.conflict)
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value}")

    def total(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute weighted total salience

        Args:
            weights: Optional dict mapping dimension names to weights.
                    If None, uses equal weights (0.2 each).

        Returns:
            Total salience score (0.0-1.0)
        """
        if weights is None:
            weights = {
                'surprise': 0.2,
                'novelty': 0.2,
                'arousal': 0.2,
                'reward': 0.2,
                'conflict': 0.2
            }

        total = (
            self.surprise * weights.get('surprise', 0.0) +
            self.novelty * weights.get('novelty', 0.0) +
            self.arousal * weights.get('arousal', 0.0) +
            self.reward * weights.get('reward', 0.0) +
            self.conflict * weights.get('conflict', 0.0)
        )

        return min(max(total, 0.0), 1.0)  # Clamp to 0-1


@dataclass
class SalienceReport:
    """
    SNARC's attention recommendation to SAGE kernel

    Provides:
    - Which sensor/region to focus on
    - How salient (importance score)
    - Why salient (breakdown by dimension)
    - What cognitive stance to use
    - Relevant memories to load
    - Confidence in assessment
    """
    focus_target: str  # Sensor ID or region to attend to
    salience_score: float  # Overall importance (0.0-1.0)
    salience_breakdown: SalienceBreakdown  # 5D breakdown
    suggested_stance: CognitiveStance  # Recommended cognitive mode
    relevant_memories: List[Any] = field(default_factory=list)  # From SNARC memory
    confidence: float = 0.5  # How certain about this assessment (0.0-1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context

    def __post_init__(self):
        """Validate report"""
        if not 0.0 <= self.salience_score <= 1.0:
            raise ValueError(f"salience_score must be 0.0-1.0, got {self.salience_score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'focus_target': self.focus_target,
            'salience_score': self.salience_score,
            'salience_breakdown': {
                'surprise': self.salience_breakdown.surprise,
                'novelty': self.salience_breakdown.novelty,
                'arousal': self.salience_breakdown.arousal,
                'reward': self.salience_breakdown.reward,
                'conflict': self.salience_breakdown.conflict,
                'total': self.salience_breakdown.total()
            },
            'suggested_stance': self.suggested_stance.value,
            'confidence': self.confidence,
            'num_relevant_memories': len(self.relevant_memories),
            'metadata': self.metadata
        }


@dataclass
class Outcome:
    """
    Result of kernel's action based on SNARC recommendation

    Used for outcome-based learning: did this salience assessment
    lead to useful action?
    """
    success: bool  # Was the action successful?
    reward: float  # Numeric reward signal (higher = better)
    metrics: Dict[str, float] = field(default_factory=dict)  # Additional metrics
    description: str = ""  # Human-readable outcome description

    def __post_init__(self):
        """Set default reward if not provided"""
        if self.reward is None:
            self.reward = 1.0 if self.success else 0.0


@dataclass
class SensorOutput:
    """
    Standardized sensor output from IRP plugins

    All sensors (vision, audio, etc.) produce this format
    so SNARC can process uniformly.
    """
    sensor_id: str  # Unique identifier (e.g., "vision_camera_0", "audio_mic_1")
    timestamp: float  # When this was captured
    data: Any  # Actual sensor data (tensor, vector, etc.)
    sensor_type: str  # "vision", "audio", "proprioception", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info

    def magnitude(self) -> float:
        """
        Compute magnitude of signal for arousal detection

        Returns:
            Normalized magnitude (0.0-1.0)
        """
        import torch
        import numpy as np

        if isinstance(self.data, torch.Tensor):
            return min(float(torch.norm(self.data).item()) / 10.0, 1.0)
        elif isinstance(self.data, np.ndarray):
            return min(float(np.linalg.norm(self.data)) / 10.0, 1.0)
        elif isinstance(self.data, (int, float)):
            return min(abs(float(self.data)) / 10.0, 1.0)
        else:
            return 0.5  # Default moderate magnitude


@dataclass
class SNARCMemory:
    """
    Memory entry stored by SNARC

    Unlike full SNARC-SAGE memory bridge, this is SNARC's
    internal memory of past salience assessments and outcomes.
    """
    assessment: SalienceReport  # The salience assessment made
    outcome: Optional[Outcome]  # Result (if available)
    timestamp: float  # When this occurred
    sensor_snapshot: Dict[str, Any]  # Sensor state at time

    def was_useful(self) -> bool:
        """Did this assessment lead to positive outcome?"""
        if self.outcome is None:
            return False
        return self.outcome.success and self.outcome.reward > 0.5
