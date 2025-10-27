#!/usr/bin/env python3
"""
Audio Chunker - Breath-Based Speech Chunking for Universal Framework

Wraps the existing ProsodyAwareChunker to integrate with the universal
chunking framework. Demonstrates the integration pattern for modality-specific
chunkers.

This chunker:
1. Uses existing prosodic boundary detection (breath groups)
2. Computes trust from transcription and prosodic coherence
3. Computes SNARC salience from semantic and prosodic features
4. Produces UniversalChunk objects compatible with cross-modal fusion

Author: Claude
Date: October 26, 2025
"""

import numpy as np
from typing import Tuple, Any, Optional, Dict
from dataclasses import dataclass

from cognitive.prosody_chunker import (
    ProsodyAwareChunker,
    ProsodicChunk as LegacyProsodicChunk
)
from cognitive.universal_chunking import (
    UniversalChunker,
    ChunkTrustMetrics,
    ChunkSalienceMetrics,
    UniversalChunk
)


@dataclass
class AudioContent:
    """
    Content for audio chunks.

    Contains both text and audio data for trust/salience computation.
    """
    text: str
    audio: Optional[np.ndarray] = None  # Audio waveform (if available)
    transcription_confidence: float = 0.85  # Whisper confidence (placeholder)
    prosodic_metadata: Optional[Dict[str, Any]] = None


class AudioChunker(UniversalChunker):
    """
    Audio chunking with prosodic boundaries (breath groups).

    Integrates existing ProsodyAwareChunker with universal framework:
    - Boundary detection: Delegates to ProsodyAwareChunker
    - Trust: Transcription confidence + prosodic coherence
    - Salience: Semantic surprise + prosodic arousal

    Chunks: 5-18 words (target 12), 1.5-4.5 seconds
    """

    def __init__(
        self,
        min_phrase_words: int = 5,
        target_phrase_words: int = 12,
        max_phrase_words: int = 18,
        speaking_rate: float = 3.8  # words/sec
    ):
        """
        Initialize audio chunker with breath group parameters.

        Args:
            min_phrase_words: Minimum words before considering break
            target_phrase_words: Preferred breath group size
            max_phrase_words: Maximum words before forced break
            speaking_rate: Words per second for duration estimation
        """
        # Initialize base class
        super().__init__(
            modality="audio",
            min_chunk_size=min_phrase_words,
            target_chunk_size=target_phrase_words,
            max_chunk_size=max_phrase_words,
            chunk_duration=(1.5, 4.5)  # 1.5-4.5 seconds
        )

        # Create underlying prosody chunker
        self.prosody_chunker = ProsodyAwareChunker(
            min_phrase_words=min_phrase_words,
            target_phrase_words=target_phrase_words,
            max_phrase_words=max_phrase_words,
            speaking_rate=speaking_rate
        )

        self.speaking_rate = speaking_rate

    def detect_boundary(
        self,
        buffer: str,
        new_chunk: str = ""
    ) -> Tuple[bool, str]:
        """
        Detect prosodic boundary in speech stream.

        Delegates to ProsodyAwareChunker for boundary detection:
        1. Sentence boundaries (Intonational Phrases - IP)
        2. Clause boundaries (Intermediate Phrases - ip)
        3. Breath group overflow (forced safety net)
        4. Natural break points at target size

        Args:
            buffer: Current accumulated text
            new_chunk: Newly added text (unused)

        Returns:
            (is_boundary, boundary_type) tuple
            - is_boundary: True if boundary detected
            - boundary_type: "major" (IP), "minor" (ip), "micro" (natural), "forced" (overflow)
        """
        # Delegate to prosody chunker
        is_boundary, prosodic_boundary_type = self.prosody_chunker.is_prosodic_boundary(
            buffer, new_chunk
        )

        if not is_boundary:
            return (False, None)

        # Map prosodic boundary types to universal types
        boundary_map = {
            'IP': 'major',           # Sentence boundary (Intonational Phrase)
            'ip': 'minor',           # Clause boundary (Intermediate Phrase)
            'NATURAL': 'micro',      # Natural pause (prepositional phrase, etc.)
        }

        # Handle BREATH(N) boundaries (overflow)
        if prosodic_boundary_type and prosodic_boundary_type.startswith('BREATH'):
            return (True, 'forced')

        # Map to universal boundary type
        universal_type = boundary_map.get(prosodic_boundary_type, 'micro')
        return (True, universal_type)

    def compute_trust(self, chunk_content: Any) -> ChunkTrustMetrics:
        """
        Compute trust for audio chunk.

        Trust factors:
        - Confidence: Transcription confidence (Whisper logprobs)
        - Consistency: Prosodic coherence (F0 continuity, energy smoothness)
        - Completeness: Boundary quality (natural vs forced)
        - Fidelity: Audio quality (SNR, no clipping)

        Args:
            chunk_content: AudioContent or str

        Returns:
            ChunkTrustMetrics with 4D trust scores
        """
        # Extract content
        if isinstance(chunk_content, AudioContent):
            text = chunk_content.text
            audio = chunk_content.audio
            transcription_conf = chunk_content.transcription_confidence
            prosody = chunk_content.prosodic_metadata or {}
        else:
            text = str(chunk_content)
            audio = None
            transcription_conf = 0.85  # Default Whisper confidence
            prosody = {}

        # 1. Confidence: Transcription confidence
        confidence = transcription_conf

        # 2. Consistency: Prosodic coherence
        # If we have prosodic metadata, use it
        if 'prosodic_coherence' in prosody:
            consistency = prosody['prosodic_coherence']
        elif audio is not None:
            # Compute from audio (F0 variance, energy smoothness)
            consistency = self._compute_prosodic_coherence(audio)
        else:
            # Default - assume reasonably coherent
            consistency = 0.75

        # 3. Completeness: Boundary quality
        # This is set during chunk creation based on boundary type
        # Major (IP) = 1.0, Minor (ip) = 0.8, Micro (natural) = 0.6, Forced = 0.3
        boundary_type = prosody.get('boundary_type', 'micro')
        completeness = {
            'major': 1.0,
            'minor': 0.8,
            'micro': 0.6,
            'forced': 0.3
        }.get(boundary_type, 0.5)

        # 4. Fidelity: Audio quality
        if 'snr' in prosody:
            # SNR provided - use it
            snr = prosody['snr']
            fidelity = min(1.0, snr / 20.0)  # 20dB = perfect
        elif audio is not None:
            # Compute SNR from audio
            snr = self._compute_snr(audio)
            fidelity = min(1.0, snr / 20.0)
        else:
            # No audio - assume decent quality
            fidelity = 0.8

        return ChunkTrustMetrics(
            confidence=confidence,
            consistency=consistency,
            completeness=completeness,
            fidelity=fidelity
        )

    def compute_salience(self, chunk_content: Any) -> ChunkSalienceMetrics:
        """
        Compute SNARC salience for audio chunk.

        Audio SNARC:
        - Surprise: Semantic surprise (language model perplexity)
        - Novelty: Topic novelty (vs. conversation history)
        - Arousal: Prosodic arousal (pitch + energy variance)
        - Reward: Goal relevance (task-dependent)
        - Conflict: Semantic ambiguity (multiple interpretations)
        - Prosodic: Boundary importance (major > minor > micro)

        Args:
            chunk_content: AudioContent or str

        Returns:
            ChunkSalienceMetrics with 6D salience scores
        """
        # Extract content
        if isinstance(chunk_content, AudioContent):
            text = chunk_content.text
            audio = chunk_content.audio
            prosody = chunk_content.prosodic_metadata or {}
        else:
            text = str(chunk_content)
            audio = None
            prosody = {}

        # 1. Surprise: Semantic surprise
        # TODO: Compute from language model perplexity
        surprise = prosody.get('semantic_surprise', 0.5)

        # 2. Novelty: Topic novelty
        # TODO: Compute from conversation history
        novelty = prosody.get('topic_novelty', 0.5)

        # 3. Arousal: Prosodic arousal
        if 'prosodic_arousal' in prosody:
            arousal = prosody['prosodic_arousal']
        elif audio is not None:
            # Compute from audio (pitch variance + energy variance)
            arousal = self._compute_prosodic_arousal(audio)
        else:
            # Default - moderate arousal
            arousal = 0.5

        # 4. Reward: Goal relevance
        # Task-dependent - use metadata if available
        reward = prosody.get('goal_relevance', 0.5)

        # 5. Conflict: Semantic ambiguity
        # TODO: Compute from multiple interpretations
        conflict = prosody.get('semantic_ambiguity', 0.3)

        # 6. Prosodic: Boundary importance
        boundary_type = prosody.get('boundary_type', 'micro')
        prosodic_salience = {
            'major': 0.9,   # Sentence boundary (IP)
            'minor': 0.7,   # Clause boundary (ip)
            'micro': 0.5,   # Natural pause
            'forced': 0.3   # Overflow (unnatural)
        }.get(boundary_type, 0.5)

        return ChunkSalienceMetrics(
            surprise=surprise,
            novelty=novelty,
            arousal=arousal,
            reward=reward,
            conflict=conflict,
            prosodic=prosodic_salience
        )

    def extract_prosody(self, chunk_content: Any) -> Dict[str, Any]:
        """
        Extract prosodic features for audio chunk.

        Returns metadata compatible with existing ProsodicChunk but
        also includes additional features for trust/salience computation.

        Args:
            chunk_content: AudioContent or str

        Returns:
            Dictionary with prosodic features
        """
        # Extract content
        if isinstance(chunk_content, AudioContent):
            text = chunk_content.text
            audio = chunk_content.audio
            existing_prosody = chunk_content.prosodic_metadata or {}
        else:
            text = str(chunk_content)
            audio = None
            existing_prosody = {}

        # Start with any existing prosodic metadata
        prosody = existing_prosody.copy()

        # Add text-based features
        word_count = len(text.strip().split())
        estimated_duration = word_count / self.speaking_rate

        prosody.update({
            'word_count': word_count,
            'estimated_duration': estimated_duration,
        })

        # Add audio-based features if available
        if audio is not None:
            prosody.update({
                'prosodic_arousal': self._compute_prosodic_arousal(audio),
                'prosodic_coherence': self._compute_prosodic_coherence(audio),
                'snr': self._compute_snr(audio)
            })

        return prosody

    # Helper methods for audio analysis

    def _compute_prosodic_coherence(self, audio: np.ndarray) -> float:
        """
        Compute prosodic coherence from audio waveform.

        High coherence = smooth F0 contour, consistent energy
        Low coherence = erratic pitch, energy spikes

        Args:
            audio: Audio waveform

        Returns:
            Coherence score 0.0-1.0
        """
        # TODO: Implement proper F0 extraction and variance analysis
        # Placeholder - assume reasonably coherent
        return 0.75

    def _compute_prosodic_arousal(self, audio: np.ndarray) -> float:
        """
        Compute prosodic arousal from audio waveform.

        High arousal = high pitch variance, high energy variance
        Low arousal = monotone, low energy

        Args:
            audio: Audio waveform

        Returns:
            Arousal score 0.0-1.0
        """
        # TODO: Implement proper pitch/energy variance analysis
        # Placeholder - moderate arousal
        return 0.5

    def _compute_snr(self, audio: np.ndarray) -> float:
        """
        Compute signal-to-noise ratio from audio waveform.

        Args:
            audio: Audio waveform

        Returns:
            SNR in dB
        """
        # TODO: Implement proper SNR estimation
        # Placeholder - assume decent quality
        return 15.0  # 15 dB

    def create_chunk_from_text(
        self,
        text: str,
        boundary_type: str,
        is_final: bool = False,
        audio: Optional[np.ndarray] = None,
        prosodic_metadata: Optional[Dict[str, Any]] = None
    ) -> UniversalChunk:
        """
        Convenience method to create chunk from text.

        Wraps text in AudioContent and delegates to create_chunk.

        Args:
            text: Chunk text
            boundary_type: Universal boundary type ("major", "minor", "micro", "forced")
            is_final: Whether this is the final chunk
            audio: Optional audio waveform
            prosodic_metadata: Optional prosodic metadata

        Returns:
            UniversalChunk with full metadata
        """
        # Create audio content
        metadata = prosodic_metadata or {}
        metadata['boundary_type'] = boundary_type

        content = AudioContent(
            text=text,
            audio=audio,
            prosodic_metadata=metadata
        )

        # Compute duration
        word_count = len(text.strip().split())
        duration = word_count / self.speaking_rate

        # Create universal chunk
        return self.create_chunk(
            content=content,
            boundary_type=boundary_type,
            chunk_size=word_count,
            duration=duration,
            continuation=not is_final
        )
