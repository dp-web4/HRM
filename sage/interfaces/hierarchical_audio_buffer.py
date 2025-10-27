#!/usr/bin/env python3
"""
Hierarchical Audio Buffer - Memory-Efficient Audio Management with Prosodic Segmentation

Replaces disk-based audio storage with intelligent tiered buffering:
- Tier 1: Rolling capture buffer (5s, always in memory)
- Tier 2: Prosodic segments (breath-aligned, awaiting transcription)
- Tier 3: Transcribed prosodic chunks (audio discarded, prosody preserved)

Phase 2 Enhancement: Adds prosodic boundary detection for natural speech chunking
Storage: 19GB/day → 3.4MB maximum (99.98% reduction)
"""

import numpy as np
import threading
import time
from collections import deque
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ProsodicMetadata:
    """Prosodic features extracted from speech segment"""
    boundary_type: Optional[str] = None  # "IP", "ip", "NATURAL", "BREATH"
    word_count: Optional[int] = None
    estimated_duration: Optional[float] = None
    boundary_tone: Optional[str] = None  # "L-L%", "L-H%"
    continuation: bool = True  # Whether more content is expected

    # Emotional prosody (for SNARC salience)
    pitch_mean: Optional[float] = None  # Average F0
    pitch_variance: Optional[float] = None  # F0 variation (affect indicator)
    energy_mean: Optional[float] = None  # Volume
    energy_variance: Optional[float] = None
    speaking_rate: Optional[float] = None  # Words per second

    # Prosodic salience indicators (complement SNARC's 5D)
    prosodic_surprise: Optional[float] = None  # Unexpected pitch/energy patterns
    prosodic_arousal: Optional[float] = None  # Energy + pitch variance

    def compute_prosodic_salience(self) -> float:
        """
        Compute prosodic contribution to SNARC salience.

        Prosody carries emotional content that complements semantic salience:
        - High pitch variance = excitement/surprise
        - High energy variance = emphasis/importance
        - Pitch extremes = emotional arousal

        Returns salience score 0.0-1.0
        """
        if not all([self.pitch_variance, self.energy_variance, self.pitch_mean]):
            return 0.5  # Neutral if no prosodic data

        # Normalize components (assuming typical ranges)
        pitch_var_norm = min(1.0, self.pitch_variance / 50.0)  # Max 50Hz variance
        energy_var_norm = min(1.0, self.energy_variance / 20.0)  # Max 20dB variance

        # Combine: arousal = energy + pitch variation
        arousal = (energy_var_norm + pitch_var_norm) / 2.0

        # Boundary type importance (IP > ip > NATURAL > BREATH)
        boundary_weight = {
            'IP': 0.9,  # Sentence boundaries are important
            'ip': 0.7,  # Clause boundaries moderately important
            'NATURAL': 0.5,  # Natural breaks are neutral
            'BREATH': 0.3,  # Forced breaks less important
        }.get(self.boundary_type, 0.5)

        # Combined prosodic salience
        salience = (arousal * 0.6) + (boundary_weight * 0.4)

        return salience


@dataclass
class SpeechSegment:
    """A segment of detected speech awaiting transcription"""
    audio: np.ndarray  # Raw audio samples
    timestamp: float
    duration: float
    salience: Optional[float] = None  # SNARC salience score
    prosody: Optional[ProsodicMetadata] = None  # Prosodic features
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Partial transcription (for boundary detection)
    partial_text: Optional[str] = None
    confidence: Optional[float] = None

    def compute_combined_salience(self) -> float:
        """
        Combine SNARC semantic salience with prosodic salience.

        Returns combined salience score 0.0-1.0
        """
        semantic_salience = self.salience if self.salience is not None else 0.5
        prosodic_salience = self.prosody.compute_prosodic_salience() if self.prosody else 0.5

        # Weighted combination (70% semantic, 30% prosodic)
        # Prosody modulates but doesn't dominate content
        combined = (semantic_salience * 0.7) + (prosodic_salience * 0.3)

        return combined


class HierarchicalAudioBuffer:
    """
    Three-tier audio buffer for intelligent memory management.

    Tier 1: Rolling Capture (Circular Buffer)
    - Fixed 5-second window
    - Constantly overwriting oldest
    - Captures everything, retains nothing permanently

    Tier 2: Speech Segments (Queue)
    - Up to 10 detected speech segments
    - Awaiting transcription
    - SNARC-based eviction (future)

    Tier 3: Transcriptions (Text Storage)
    - Only the transcribed text
    - Audio discarded after transcription
    - Permanent storage in SNARC memory
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        rolling_duration: float = 5.0,
        max_speech_segments: int = 10,
        frame_duration_ms: int = 30,
        enable_prosodic_segmentation: bool = True
    ):
        """
        Initialize hierarchical audio buffer.

        Args:
            sample_rate: Audio sample rate (Hz)
            rolling_duration: Duration of rolling capture buffer (seconds)
            max_speech_segments: Maximum speech segments to retain
            frame_duration_ms: Frame duration in milliseconds
            enable_prosodic_segmentation: Enable prosodic boundary detection
        """
        self.sample_rate = sample_rate
        self.rolling_duration = rolling_duration
        self.max_speech_segments = max_speech_segments
        self.frame_duration_ms = frame_duration_ms
        self.enable_prosodic_segmentation = enable_prosodic_segmentation

        # Calculate buffer sizes
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.rolling_capacity = int(rolling_duration * sample_rate / self.frame_size)

        # Tier 1: Rolling capture buffer (circular)
        self.rolling_buffer = deque(maxlen=self.rolling_capacity)
        self.rolling_lock = threading.Lock()

        # Tier 2: Speech segments queue
        self.speech_segments: List[SpeechSegment] = []
        self.speech_lock = threading.Lock()

        # Prosodic chunker (if enabled)
        self.prosody_chunker = None
        if enable_prosodic_segmentation:
            try:
                from cognitive.prosody_chunker import ProsodyAwareChunker
                self.prosody_chunker = ProsodyAwareChunker(
                    min_phrase_words=5,
                    target_phrase_words=12,
                    max_phrase_words=18
                )
            except ImportError:
                print("  ⚠️  Prosodic chunker not available (optional)")

        # Stats
        self.stats = {
            'total_frames': 0,
            'speech_segments_captured': 0,
            'speech_segments_evicted': 0,
            'transcriptions_completed': 0,
            'bytes_saved_vs_disk': 0,
            'prosodic_segments_created': 0
        }

        print(f"📼 Hierarchical Audio Buffer initialized")
        print(f"   Tier 1 (Rolling): {rolling_duration}s = {self.rolling_capacity} frames")
        print(f"   Tier 2 (Speech): {max_speech_segments} segments max")
        print(f"   Frame size: {self.frame_size} samples ({frame_duration_ms}ms)")
        if self.prosody_chunker:
            print(f"   ✓ Prosodic segmentation enabled (breath-aligned chunks)")

    # =========================================================================
    # Tier 1: Rolling Capture Buffer
    # =========================================================================

    def push_frame(self, audio_frame: np.ndarray) -> None:
        """
        Push audio frame to rolling capture buffer.

        This is called continuously for ALL audio. Oldest frames are
        automatically overwritten when buffer is full.

        Args:
            audio_frame: Audio samples (numpy array)
        """
        with self.rolling_lock:
            self.rolling_buffer.append(audio_frame)
            self.stats['total_frames'] += 1

    def get_rolling_buffer(self, duration_seconds: Optional[float] = None) -> np.ndarray:
        """
        Get audio from rolling capture buffer.

        Args:
            duration_seconds: How many seconds to retrieve (None = all)

        Returns:
            Numpy array of audio samples
        """
        with self.rolling_lock:
            if not self.rolling_buffer:
                return np.array([], dtype=np.int16)

            if duration_seconds is None:
                # Return entire buffer
                frames = list(self.rolling_buffer)
            else:
                # Calculate how many frames to retrieve
                num_frames = int(duration_seconds * self.sample_rate / self.frame_size)
                num_frames = min(num_frames, len(self.rolling_buffer))
                # Get most recent N frames
                frames = list(self.rolling_buffer)[-num_frames:]

            # Concatenate all frames
            return np.concatenate(frames) if frames else np.array([], dtype=np.int16)

    def get_rolling_duration(self) -> float:
        """Get current duration of audio in rolling buffer (seconds)."""
        with self.rolling_lock:
            num_frames = len(self.rolling_buffer)
            return num_frames * self.frame_size / self.sample_rate

    # =========================================================================
    # Tier 2: Speech Segments
    # =========================================================================

    def promote_to_speech(self, duration_seconds: float = None, metadata: Dict[str, Any] = None) -> SpeechSegment:
        """
        Promote rolling buffer content to speech segment queue.

        Called when VAD detects speech. Copies audio from rolling buffer
        to speech segment for transcription.

        Args:
            duration_seconds: How much audio to capture (default: all)
            metadata: Additional metadata for this segment

        Returns:
            Created SpeechSegment
        """
        # Get audio from rolling buffer
        audio = self.get_rolling_buffer(duration_seconds)

        if len(audio) == 0:
            raise ValueError("No audio in rolling buffer to promote")

        # Create speech segment
        segment = SpeechSegment(
            audio=audio,
            timestamp=time.time(),
            duration=len(audio) / self.sample_rate,
            metadata=metadata or {}
        )

        with self.speech_lock:
            # Add to queue
            self.speech_segments.append(segment)
            self.stats['speech_segments_captured'] += 1

            # Evict oldest if over capacity (FIFO for now, SNARC-based later)
            while len(self.speech_segments) > self.max_speech_segments:
                evicted = self.speech_segments.pop(0)
                self.stats['speech_segments_evicted'] += 1
                # Calculate how much disk space we saved
                self.stats['bytes_saved_vs_disk'] += len(evicted.audio) * 2  # 16-bit = 2 bytes

        return segment

    def get_next_speech_segment(self) -> Optional[SpeechSegment]:
        """
        Get next speech segment for transcription (FIFO).

        Returns:
            SpeechSegment or None if queue is empty
        """
        with self.speech_lock:
            if not self.speech_segments:
                return None
            return self.speech_segments.pop(0)

    def get_speech_segment_count(self) -> int:
        """Get number of speech segments waiting for transcription."""
        with self.speech_lock:
            return len(self.speech_segments)

    # =========================================================================
    # Prosodic Segmentation (Phase 2 Enhancement)
    # =========================================================================

    def extract_prosodic_features(self, audio: np.ndarray) -> ProsodicMetadata:
        """
        Extract basic prosodic features from audio segment.

        This is a lightweight feature extractor that doesn't require
        full transcription. Uses simple signal processing for:
        - Energy (volume) patterns
        - Pitch estimation (zero-crossing rate proxy)
        - Temporal dynamics

        For full prosodic analysis, use Whisper's attention weights.

        Args:
            audio: Raw audio samples (np.ndarray, int16)

        Returns:
            ProsodicMetadata with extracted features
        """
        if len(audio) == 0:
            return ProsodicMetadata()

        # Convert to float for processing
        audio_float = audio.astype(np.float32) / 32768.0

        # Energy (RMS amplitude)
        energy = np.sqrt(np.mean(audio_float ** 2))
        energy_db = 20 * np.log10(energy + 1e-10)  # Convert to dB

        # Energy variance (indicates emphasis patterns)
        frame_size = int(self.sample_rate * 0.1)  # 100ms frames
        num_frames = len(audio_float) // frame_size
        if num_frames > 0:
            frames = audio_float[:num_frames * frame_size].reshape(num_frames, frame_size)
            frame_energy = np.sqrt(np.mean(frames ** 2, axis=1))
            energy_variance = np.var(frame_energy)
        else:
            energy_variance = 0.0

        # Pitch estimation (zero-crossing rate - rough proxy)
        # High ZCR ≈ high pitch, low ZCR ≈ low pitch
        zero_crossings = np.where(np.diff(np.sign(audio_float)))[0]
        zcr = len(zero_crossings) / len(audio_float) * self.sample_rate

        # Pitch variance (frame-based ZCR variance)
        if num_frames > 0:
            frame_zcr = []
            for i in range(num_frames):
                frame = audio_float[i*frame_size:(i+1)*frame_size]
                frame_zc = np.where(np.diff(np.sign(frame)))[0]
                frame_zcr.append(len(frame_zc) / len(frame) * self.sample_rate)
            pitch_variance = np.var(frame_zcr)
            pitch_mean = np.mean(frame_zcr)
        else:
            pitch_variance = 0.0
            pitch_mean = zcr

        # Create prosodic metadata
        prosody = ProsodicMetadata(
            pitch_mean=pitch_mean,
            pitch_variance=pitch_variance,
            energy_mean=energy_db,
            energy_variance=energy_variance,
            speaking_rate=None,  # Requires transcription
            prosodic_arousal=(pitch_variance / 100.0 + energy_variance) / 2.0  # Normalized
        )

        return prosody

    def add_prosodic_metadata(
        self,
        segment: SpeechSegment,
        partial_text: Optional[str] = None
    ) -> None:
        """
        Add prosodic metadata to speech segment.

        Args:
            segment: Speech segment to annotate
            partial_text: Optional partial transcription for boundary detection
        """
        # Extract prosodic features from audio
        segment.prosody = self.extract_prosodic_features(segment.audio)

        # If we have partial transcription, check for prosodic boundaries
        if partial_text and self.prosody_chunker:
            segment.partial_text = partial_text
            is_boundary, boundary_type = self.prosody_chunker.is_prosodic_boundary(partial_text)

            if is_boundary:
                segment.prosody.boundary_type = boundary_type
                segment.prosody.word_count = len(partial_text.split())
                segment.prosody.estimated_duration = segment.duration

                # Determine boundary tone (final vs continuation)
                # IP boundaries are final, others are continuation
                segment.prosody.boundary_tone = "L-L%" if boundary_type == "IP" else "L-H%"
                segment.prosody.continuation = boundary_type != "IP"

                self.stats['prosodic_segments_created'] += 1

    # =========================================================================
    # Tier 3: Transcription
    # =========================================================================

    def mark_transcription_complete(self, segment: SpeechSegment, transcription: str) -> None:
        """
        Mark segment as transcribed. Audio is discarded.

        Args:
            segment: The speech segment that was transcribed
            transcription: The text transcription
        """
        # Calculate storage savings
        audio_bytes = len(segment.audio) * 2  # 16-bit samples
        text_bytes = len(transcription.encode('utf-8'))
        savings = audio_bytes - text_bytes

        self.stats['transcriptions_completed'] += 1
        self.stats['bytes_saved_vs_disk'] += savings

        # Audio is garbage collected automatically when segment is discarded
        # Transcription should be stored in SNARC memory (handled by caller)

    # =========================================================================
    # Statistics and Monitoring
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.rolling_lock:
            rolling_frames = len(self.rolling_buffer)
            rolling_duration = self.get_rolling_duration()

        with self.speech_lock:
            speech_count = len(self.speech_segments)
            speech_bytes = sum(len(s.audio) * 2 for s in self.speech_segments)

        return {
            **self.stats,
            'rolling_buffer': {
                'frames': rolling_frames,
                'capacity': self.rolling_capacity,
                'utilization': rolling_frames / self.rolling_capacity if self.rolling_capacity > 0 else 0,
                'duration_seconds': rolling_duration,
                'memory_bytes': rolling_frames * self.frame_size * 2  # 16-bit
            },
            'speech_queue': {
                'segments': speech_count,
                'capacity': self.max_speech_segments,
                'utilization': speech_count / self.max_speech_segments if self.max_speech_segments > 0 else 0,
                'memory_bytes': speech_bytes
            },
            'storage_efficiency': {
                'bytes_saved': self.stats['bytes_saved_vs_disk'],
                'mb_saved': self.stats['bytes_saved_vs_disk'] / (1024 * 1024)
            }
        }

    def print_stats(self) -> None:
        """Print buffer statistics (for debugging)."""
        stats = self.get_stats()

        print("\n📊 Hierarchical Audio Buffer Stats:")
        print(f"   Total frames: {stats['total_frames']}")
        print(f"\n   Tier 1 (Rolling):")
        print(f"      Frames: {stats['rolling_buffer']['frames']}/{stats['rolling_buffer']['capacity']}")
        print(f"      Duration: {stats['rolling_buffer']['duration_seconds']:.2f}s")
        print(f"      Memory: {stats['rolling_buffer']['memory_bytes'] / 1024:.1f} KB")
        print(f"\n   Tier 2 (Speech):")
        print(f"      Segments: {stats['speech_queue']['segments']}/{stats['speech_queue']['capacity']}")
        print(f"      Memory: {stats['speech_queue']['memory_bytes'] / 1024:.1f} KB")
        print(f"\n   Tier 3 (Transcription):")
        print(f"      Completed: {stats['transcriptions_completed']}")
        print(f"      Evicted: {stats['speech_segments_evicted']}")
        print(f"\n   Storage Efficiency:")
        print(f"      Saved vs disk: {stats['storage_efficiency']['mb_saved']:.2f} MB")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create buffer
    buffer = HierarchicalAudioBuffer(
        sample_rate=16000,
        rolling_duration=5.0,
        max_speech_segments=10
    )

    # Simulate continuous audio capture
    print("\nSimulating continuous audio capture...")
    for i in range(200):  # 200 frames = 6 seconds
        # Generate fake audio frame (30ms at 16kHz)
        fake_frame = np.random.randint(-32768, 32767, size=480, dtype=np.int16)
        buffer.push_frame(fake_frame)

        # Simulate speech detection at frame 100
        if i == 100:
            print("   [VAD] Speech detected! Promoting to Tier 2...")
            segment = buffer.promote_to_speech(duration_seconds=2.0)
            print(f"   [TIER 2] Captured {segment.duration:.2f}s segment")

        # Simulate transcription at frame 150
        if i == 150:
            print("   [TRANSCRIPTION] Processing speech segment...")
            segment = buffer.get_next_speech_segment()
            if segment:
                transcription = "Hello, this is a test transcription."
                buffer.mark_transcription_complete(segment, transcription)
                print(f"   [TIER 3] Transcribed: '{transcription}'")
                print(f"   [TIER 3] Audio discarded, saved {len(segment.audio)*2} bytes")

    # Print final stats
    buffer.print_stats()

    print("\n✅ Hierarchical buffer demonstration complete!")
    print("   No files written to disk - everything in memory!")
