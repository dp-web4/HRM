#!/usr/bin/env python3
"""
Universal Chunking Framework - Cross-Modal Processing for SAGE

Extends breath-based prosodic chunking principles to ALL modalities:
- Vision (attention-based fixation groups)
- Audio (breath-based prosodic groups)
- Motion (kinematic phrases with velocity zero-crossings)
- Memory (episodic boundaries with 7±2 working memory)
- Language (clause-aligned generation chunks)
- Control (goal-hierarchy action sequences)

Key Innovation: Every modality has "prosody" - rhythmic structure defining
natural boundaries. Universal constraints apply across all modalities:
- Duration: 2-4 seconds (attention window)
- Size: 5-9 discrete items (working memory capacity)
- Boundaries: Natural transitions in modality's prosodic structure

This provides the foundation for true cross-modal intelligence.

References:
- Breath-based chunking research (PMC2945274, Lieberman 1966)
- Vision: Yarbus (1967), Henderson (2003) - fixation-saccade structure
- Motion: Sosnik et al. (2004) - velocity zero-crossings as boundaries
- Memory: Miller (1956) - 7±2 working memory capacity
- Universal constraints: Doupe & Kuhl (1999) - hierarchical chunking across species

Author: Claude (Implementation from design doc UNIVERSAL_CHUNKING_ARCHITECTURE.md)
Date: October 26, 2025
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple, Dict


@dataclass
class ChunkTrustMetrics:
    """
    Trust metrics for chunk quality assessment.

    Trust measures: How reliable is this chunk?
    - Confidence: Detection/generation confidence
    - Consistency: Internal coherence
    - Completeness: Boundary detection quality
    - Fidelity: Compression/reconstruction error

    Trust guides ATP (attention) allocation:
    - High trust → Use chunk confidently
    - Low trust → Verify or re-process
    """
    confidence: float    # 0.0-1.0, detection/generation confidence
    consistency: float   # 0.0-1.0, internal coherence
    completeness: float  # 0.0-1.0, boundary quality (natural vs forced)
    fidelity: float      # 0.0-1.0, compression/reconstruction quality

    def compute_overall_trust(self) -> float:
        """
        Weighted combination of trust dimensions.

        Weights based on importance for decision-making:
        - Confidence: 35% (most important - can we trust the content?)
        - Consistency: 25% (is it coherent internally?)
        - Completeness: 25% (did we chunk at natural boundary?)
        - Fidelity: 15% (how well preserved through compression?)

        Returns:
            Overall trust score 0.0-1.0
        """
        weights = {
            'confidence': 0.35,
            'consistency': 0.25,
            'completeness': 0.25,
            'fidelity': 0.15
        }

        return (
            self.confidence * weights['confidence'] +
            self.consistency * weights['consistency'] +
            self.completeness * weights['completeness'] +
            self.fidelity * weights['fidelity']
        )

    def to_dict(self) -> Dict[str, float]:
        """Export as dictionary"""
        return {
            'confidence': self.confidence,
            'consistency': self.consistency,
            'completeness': self.completeness,
            'fidelity': self.fidelity,
            'overall': self.compute_overall_trust()
        }


@dataclass
class ChunkSalienceMetrics:
    """
    SNARC-based salience metrics for chunk importance.

    Extended SNARC 5D + Prosodic:
    - Surprise: Unexpected content (prediction error)
    - Novelty: New vs. familiar patterns
    - Arousal: Intensity/energy (modality-specific)
    - Reward: Goal-relevance, value
    - Conflict: Ambiguity, contradiction
    - Prosodic: Boundary importance (major > minor > micro)

    Salience guides attention allocation:
    - High salience → Allocate more ATP
    - Low salience → Background processing
    """
    surprise: float   # 0.0-1.0, prediction error
    novelty: float    # 0.0-1.0, familiarity inverse
    arousal: float    # 0.0-1.0, intensity/energy
    reward: float     # 0.0-1.0, goal-relevance
    conflict: float   # 0.0-1.0, ambiguity
    prosodic: float   # 0.0-1.0, boundary importance

    def compute_overall_salience(self) -> float:
        """
        Weighted combination of salience dimensions.

        Base SNARC (5D) is averaged equally, then modulated by
        prosodic salience (boundary importance amplifies base).

        Formula: base_snarc * (0.7 + prosodic * 0.3)
        - Prosodic acts as multiplicative gate
        - Major boundaries (0.9) amplify salience
        - Micro boundaries (0.3) dampen salience

        Returns:
            Overall salience score 0.0-1.0
        """
        # Base SNARC (equal weights)
        snarc_base = (
            self.surprise + self.novelty + self.arousal +
            self.reward + self.conflict
        ) / 5.0

        # Modulated by prosodic salience (multiplicative)
        # Prosodic salience amplifies base salience
        return snarc_base * (0.7 + self.prosodic * 0.3)

    def to_dict(self) -> Dict[str, float]:
        """Export as dictionary"""
        return {
            'surprise': self.surprise,
            'novelty': self.novelty,
            'arousal': self.arousal,
            'reward': self.reward,
            'conflict': self.conflict,
            'prosodic': self.prosodic,
            'overall': self.compute_overall_salience()
        }


@dataclass
class UniversalChunk:
    """
    Universal chunk representation across all modalities.

    Every chunk carries:
    - Content (modality-specific data)
    - Boundaries (prosodic markers)
    - Trust (quality metrics)
    - Salience (SNARC scores)

    This enables:
    - Cross-modal comparison (trust × salience)
    - ATP allocation (attention budget)
    - Memory consolidation (high-salience chunks persist)
    - Quality control (low-trust chunks verified)
    """
    # Core attributes
    content: Any          # Modality-specific data
    modality: str         # "vision", "audio", "memory", "language", "control", "motion"
    timestamp: float
    duration: float       # seconds

    # Boundary metadata (prosodic structure)
    boundary_type: str    # "major", "minor", "micro", "forced"
    chunk_size: int       # Number of discrete items
    continuation: bool    # Whether more chunks expected

    # Trust metrics (quality + confidence)
    trust_score: float    # 0.0-1.0, from ChunkTrustMetrics

    # SNARC salience (5D + prosodic)
    salience_score: float  # 0.0-1.0, combined salience

    # Optional breakdowns and metadata (must come after required fields)
    trust_breakdown: Optional[ChunkTrustMetrics] = None
    salience_breakdown: Optional[ChunkSalienceMetrics] = None
    prosody: Optional[Any] = None  # e.g., ProsodicMetadata, VisualProsody, MotorProsody
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_priority(self) -> float:
        """
        Compute priority for attention allocation.

        Priority = trust × salience
        - High trust, high salience → Top priority
        - Low trust → Needs verification
        - Low salience → Background processing

        Returns:
            Priority score 0.0-1.0
        """
        return self.trust_score * self.salience_score

    def needs_verification(self, threshold: float = 0.5) -> bool:
        """Check if chunk needs verification (low trust)"""
        return self.trust_score < threshold

    def is_high_salience(self, threshold: float = 0.7) -> bool:
        """Check if chunk is high salience (important)"""
        return self.salience_score >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary (for logging/debugging)"""
        return {
            'modality': self.modality,
            'timestamp': self.timestamp,
            'duration': self.duration,
            'boundary_type': self.boundary_type,
            'chunk_size': self.chunk_size,
            'continuation': self.continuation,
            'trust_score': self.trust_score,
            'trust_breakdown': self.trust_breakdown.to_dict() if self.trust_breakdown else None,
            'salience_score': self.salience_score,
            'salience_breakdown': self.salience_breakdown.to_dict() if self.salience_breakdown else None,
            'priority': self.get_priority(),
            'needs_verification': self.needs_verification(),
            'is_high_salience': self.is_high_salience(),
            'metadata': self.metadata
        }


class UniversalChunker(ABC):
    """
    Abstract base class for modality-specific chunkers.

    All chunkers must implement:
    - Boundary detection (prosodic structure)
    - Trust assessment (quality metrics)
    - Salience computation (SNARC + prosodic)

    Subclasses implement modality-specific logic:
    - VisionChunker: Fixation-based boundaries
    - AudioChunker: Breath-based boundaries
    - MotionChunker: Velocity zero-crossings
    - MemoryChunker: Episode boundaries
    - LanguageChunker: Clause boundaries
    - ControlChunker: Sub-goal boundaries
    """

    def __init__(
        self,
        modality: str,
        min_chunk_size: int,
        target_chunk_size: int,
        max_chunk_size: int,
        chunk_duration: Tuple[float, float]  # (min, max) seconds
    ):
        """
        Initialize universal chunker.

        Args:
            modality: Modality name ("vision", "audio", etc.)
            min_chunk_size: Minimum items before considering break
            target_chunk_size: Preferred chunk size
            max_chunk_size: Maximum items before forced break
            chunk_duration: (min_seconds, max_seconds) duration range
        """
        self.modality = modality
        self.min_chunk_size = min_chunk_size
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_duration, self.max_duration = chunk_duration

    @abstractmethod
    def detect_boundary(self, buffer: Any, new_item: Any) -> Tuple[bool, str]:
        """
        Detect prosodic boundary in modality-specific stream.

        Checks for natural boundaries in the stream:
        - Vision: Scene changes, object transitions, fixation clusters
        - Audio: Sentence ends, clause boundaries, breath groups
        - Motion: Goal completions, velocity zero-crossings
        - Memory: Context shifts, temporal gaps
        - Language: Clause boundaries, sentence ends
        - Control: Sub-goal completions, action sequences

        Args:
            buffer: Current accumulated items
            new_item: Newly added item

        Returns:
            (is_boundary, boundary_type) tuple
            - is_boundary: True if boundary detected
            - boundary_type: "major", "minor", "micro", or "forced"
        """
        pass

    @abstractmethod
    def compute_trust(self, chunk_content: Any) -> ChunkTrustMetrics:
        """
        Compute trust metrics for chunk.

        Modality-specific trust factors:
        - Vision: Detection confidence, optical flow consistency
        - Audio: Transcription confidence, prosodic coherence
        - Motion: Planning confidence, movement smoothness
        - Memory: Retrieval confidence, semantic clustering
        - Language: Generation probability, perplexity
        - Control: Planning confidence, action feasibility

        Args:
            chunk_content: Content to assess

        Returns:
            ChunkTrustMetrics with 4D trust scores
        """
        pass

    @abstractmethod
    def compute_salience(self, chunk_content: Any) -> ChunkSalienceMetrics:
        """
        Compute SNARC salience for chunk.

        Modality-specific salience factors:
        - Vision: Visual surprise (edge density), motion arousal
        - Audio: Semantic surprise, prosodic arousal
        - Motion: Trajectory deviation, movement intensity
        - Memory: Recall surprise, emotional intensity
        - Language: Perplexity, semantic intensity
        - Control: Plan deviation, goal value

        All modalities include prosodic salience (boundary importance).

        Args:
            chunk_content: Content to assess

        Returns:
            ChunkSalienceMetrics with 6D salience scores
        """
        pass

    @abstractmethod
    def extract_prosody(self, chunk_content: Any) -> Any:
        """
        Extract modality-specific prosodic features.

        Returns modality-specific prosody object:
        - Vision: VisualProsody (fixation count, saccade pattern)
        - Audio: ProsodicMetadata (F0 contour, energy)
        - Motion: MotorProsody (peak velocity, smoothness)
        - Memory: MemoryProsody (episode duration, coherence)
        - Language: LanguageProsody (clause type, complexity)
        - Control: ControlProsody (goal hierarchy, action sequence)

        Args:
            chunk_content: Content to analyze

        Returns:
            Modality-specific prosody object
        """
        pass

    def create_chunk(
        self,
        content: Any,
        boundary_type: str,
        chunk_size: int,
        duration: float,
        continuation: bool = True
    ) -> UniversalChunk:
        """
        Create universal chunk with full metadata.

        Computes trust, salience, and prosody automatically.

        Args:
            content: Chunk content
            boundary_type: Type of boundary detected
            chunk_size: Number of items in chunk
            duration: Chunk duration in seconds
            continuation: Whether more chunks expected

        Returns:
            UniversalChunk with complete metadata
        """
        # Extract prosodic features
        prosody = self.extract_prosody(content)

        # Compute trust metrics
        trust_metrics = self.compute_trust(content)
        trust_score = trust_metrics.compute_overall_trust()

        # Compute salience metrics
        salience_metrics = self.compute_salience(content)
        salience_score = salience_metrics.compute_overall_salience()

        return UniversalChunk(
            content=content,
            modality=self.modality,
            timestamp=time.time(),
            duration=duration,
            boundary_type=boundary_type,
            chunk_size=chunk_size,
            continuation=continuation,
            trust_score=trust_score,
            trust_breakdown=trust_metrics,
            salience_score=salience_score,
            salience_breakdown=salience_metrics,
            prosody=prosody,
            metadata={}
        )


# Utility functions for cross-modal coordination

def allocate_attention(
    chunks: List[UniversalChunk],
    total_atp: float
) -> List[Tuple[UniversalChunk, float]]:
    """
    Allocate ATP budget based on trust and salience.

    High-trust, high-salience chunks get more attention.
    Low-trust chunks trigger verification/re-processing.

    Args:
        chunks: List of chunks to allocate attention to
        total_atp: Total ATP budget available

    Returns:
        List of (chunk, atp_allocation) tuples
    """
    if not chunks:
        return []

    allocations = []

    # Compute priority scores (trust × salience)
    priorities = [chunk.get_priority() for chunk in chunks]

    # Normalize to ATP budget
    total_priority = sum(priorities)
    if total_priority == 0:
        # Equal allocation if all priorities are zero
        equal_share = total_atp / len(chunks)
        for chunk in chunks:
            allocations.append((chunk, equal_share))
    else:
        for chunk, priority in zip(chunks, priorities):
            atp_allocation = (priority / total_priority) * total_atp
            allocations.append((chunk, atp_allocation))

    return allocations


def evict_lowest_priority_chunk(buffer: List[UniversalChunk]) -> Optional[UniversalChunk]:
    """
    Evict chunk with lowest combined trust × salience score.

    Used for buffer management when capacity is reached.

    Args:
        buffer: Buffer of chunks

    Returns:
        Chunk to evict (lowest priority), or None if buffer empty
    """
    if not buffer:
        return None

    return min(buffer, key=lambda c: c.get_priority())


def group_by_time(
    chunks: List[UniversalChunk],
    temporal_window: float
) -> Dict[float, List[UniversalChunk]]:
    """
    Group chunks by timestamp for cross-modal fusion.

    Chunks occurring within temporal_window are considered co-occurring
    and can be fused for cross-modal analysis.

    Args:
        chunks: List of chunks from different modalities
        temporal_window: Time window in seconds

    Returns:
        Dictionary mapping representative timestamps to chunk groups
    """
    if not chunks:
        return {}

    # Sort by timestamp
    sorted_chunks = sorted(chunks, key=lambda c: c.timestamp)

    groups = {}
    current_group = []
    current_timestamp = sorted_chunks[0].timestamp

    for chunk in sorted_chunks:
        if chunk.timestamp - current_timestamp <= temporal_window:
            # Within window - add to current group
            current_group.append(chunk)
        else:
            # Outside window - start new group
            if current_group:
                groups[current_timestamp] = current_group
            current_group = [chunk]
            current_timestamp = chunk.timestamp

    # Add final group
    if current_group:
        groups[current_timestamp] = current_group

    return groups
