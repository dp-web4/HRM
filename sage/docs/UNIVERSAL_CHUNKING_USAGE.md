# Universal Chunking Framework - Usage Guide

**Date**: October 26, 2025
**Status**: ✅ Base Framework Implemented
**Tests**: 22/22 passing

---

## Quick Start

The universal chunking framework provides a common interface for chunking all SAGE modalities (audio, vision, motion, memory, language, control) with built-in trust and salience metrics.

### Basic Usage

```python
from cognitive.audio_chunker import AudioChunker

# Create chunker
chunker = AudioChunker(
    min_phrase_words=5,
    target_phrase_words=12,
    max_phrase_words=18
)

# Detect boundary
text_buffer = "I went to the store, and then I came home."
is_boundary, boundary_type = chunker.detect_boundary(text_buffer)

if is_boundary:
    # Create universal chunk
    chunk = chunker.create_chunk_from_text(
        text=text_buffer,
        boundary_type=boundary_type,
        is_final=False
    )

    # Access chunk properties
    print(f"Modality: {chunk.modality}")  # "audio"
    print(f"Trust: {chunk.trust_score:.2f}")  # 0.0-1.0
    print(f"Salience: {chunk.salience_score:.2f}")  # 0.0-1.0
    print(f"Priority: {chunk.get_priority():.2f}")  # trust × salience
```

---

## Core Classes

### 1. UniversalChunk

Universal representation of a chunk across all modalities.

```python
chunk = UniversalChunk(
    content="chunk content",
    modality="audio",  # "vision", "motion", "memory", "language", "control"
    timestamp=time.time(),
    duration=2.5,  # seconds
    boundary_type="major",  # "major", "minor", "micro", "forced"
    chunk_size=12,  # number of items
    continuation=True,
    trust_score=0.75,
    salience_score=0.65
)

# Priority computation
priority = chunk.get_priority()  # trust × salience

# Quality checks
needs_verification = chunk.needs_verification()  # trust < 0.5
is_important = chunk.is_high_salience()  # salience >= 0.7
```

### 2. ChunkTrustMetrics

4D trust assessment for chunk quality.

```python
trust = ChunkTrustMetrics(
    confidence=0.8,    # Detection/generation confidence
    consistency=0.7,   # Internal coherence
    completeness=0.9,  # Boundary quality (natural vs forced)
    fidelity=0.6       # Compression/reconstruction quality
)

overall_trust = trust.compute_overall_trust()  # Weighted 0.0-1.0
```

**Trust weights**:
- Confidence: 35% (most important)
- Consistency: 25%
- Completeness: 25%
- Fidelity: 15%

### 3. ChunkSalienceMetrics

6D SNARC salience for chunk importance.

```python
salience = ChunkSalienceMetrics(
    surprise=0.6,   # Prediction error
    novelty=0.5,    # New vs. familiar
    arousal=0.7,    # Intensity/energy
    reward=0.4,     # Goal-relevance
    conflict=0.3,   # Ambiguity
    prosodic=0.8    # Boundary importance
)

overall_salience = salience.compute_overall_salience()  # 0.0-1.0
```

**Salience formula**: `base_snarc * (0.7 + prosodic * 0.3)`
Prosodic salience acts as multiplicative gate.

---

## Creating Custom Chunkers

All modality-specific chunkers inherit from `UniversalChunker`:

```python
from cognitive.universal_chunking import UniversalChunker, ChunkTrustMetrics, ChunkSalienceMetrics

class VisionChunker(UniversalChunker):
    """Chunk visual attention into fixation-aligned episodes"""

    def __init__(self):
        super().__init__(
            modality="vision",
            min_chunk_size=1,       # 1 fixation minimum
            target_chunk_size=4,    # 4 fixations target
            max_chunk_size=7,       # 7 fixations max (working memory)
            chunk_duration=(0.5, 3.0)  # 0.5-3.0 seconds
        )

    def detect_boundary(self, buffer, new_item):
        """Detect visual attention boundaries"""
        # Scene changes, object transitions, attention capacity
        # ...
        return (is_boundary, boundary_type)

    def compute_trust(self, chunk_content):
        """Compute trust from detection confidence, optical flow"""
        return ChunkTrustMetrics(
            confidence=0.85,  # Detection confidence
            consistency=0.80,  # Optical flow coherence
            completeness=0.90,  # Natural boundary
            fidelity=0.75  # VAE reconstruction
        )

    def compute_salience(self, chunk_content):
        """Compute SNARC salience from visual features"""
        return ChunkSalienceMetrics(
            surprise=0.6,  # Edge density, color novelty
            novelty=0.5,   # New objects
            arousal=0.7,   # Motion energy, color saturation
            reward=0.4,    # Task relevance
            conflict=0.3,  # Detection uncertainty
            prosodic=0.8   # Scene > Object > Fixation
        )

    def extract_prosody(self, chunk_content):
        """Extract visual prosodic features"""
        return {
            'fixation_count': 4,
            'saccade_pattern': 'exploratory',
            'gaze_duration': 1.5
        }
```

---

## Cross-Modal Utilities

### ATP Allocation

Allocate attention budget based on chunk priority (trust × salience):

```python
from cognitive.universal_chunking import allocate_attention

chunks = [chunk1, chunk2, chunk3]  # Mixed modalities
total_atp = 100.0

# Allocate attention proportionally
allocations = allocate_attention(chunks, total_atp)

for chunk, atp in allocations:
    print(f"{chunk.modality}: {atp:.2f} ATP")
    if chunk.needs_verification():
        print(f"  → Needs verification (low trust)")
```

### Buffer Eviction

Evict lowest-priority chunk when buffer is full:

```python
from cognitive.universal_chunking import evict_lowest_priority_chunk

buffer = [chunk1, chunk2, chunk3, chunk4]

# Evict chunk with lowest trust × salience
evicted = evict_lowest_priority_chunk(buffer)
buffer.remove(evicted)
```

### Temporal Grouping

Group chunks by timestamp for cross-modal fusion:

```python
from cognitive.universal_chunking import group_by_time

chunks = [audio_chunk, vision_chunk, motion_chunk]
temporal_window = 0.5  # seconds

# Group co-occurring chunks
groups = group_by_time(chunks, temporal_window)

for timestamp, chunk_group in groups.items():
    print(f"Time {timestamp:.2f}:")
    for chunk in chunk_group:
        print(f"  {chunk.modality}: salience={chunk.salience_score:.2f}")
```

---

## Integration with SAGE

### Attention Allocation

```python
def sage_cycle():
    # Process modality streams
    audio_chunk = audio_chunker.process_stream(audio_data)
    vision_chunk = vision_chunker.process_stream(vision_data)

    # Collect recent chunks
    chunks = [audio_chunk, vision_chunk]

    # Allocate ATP budget
    allocations = allocate_attention(chunks, total_atp=100.0)

    # Process high-priority chunks first
    for chunk, atp in sorted(allocations, key=lambda x: x[1], reverse=True):
        if chunk.is_high_salience():
            # High-salience event - allocate extra resources
            process_with_attention(chunk, atp * 1.5)
        elif chunk.needs_verification():
            # Low-trust - verify before processing
            verified_chunk = verify_chunk(chunk)
            process_with_attention(verified_chunk, atp)
        else:
            # Normal processing
            process_with_attention(chunk, atp)
```

---

## File Structure

```
sage/cognitive/
├── universal_chunking.py       # Base framework (587 lines)
├── audio_chunker.py            # Audio wrapper (347 lines)
├── prosody_chunker.py          # Existing breath-based chunking (323 lines)
└── (future chunkers)
    ├── vision_chunker.py       # Fixation-based chunking
    ├── motion_chunker.py       # Kinematic chunking
    ├── memory_chunker.py       # Episode chunking
    ├── language_chunker.py     # Clause chunking
    └── control_chunker.py      # Goal-hierarchy chunking

sage/tests/
└── test_universal_chunking_base.py  # Comprehensive tests (22/22 passing)

sage/docs/
├── UNIVERSAL_CHUNKING_ARCHITECTURE.md  # Full design (1,573 lines)
├── UNIVERSAL_CHUNKING_USAGE.md         # This file
├── BREATH_BASED_CHUNKING_RESEARCH_REPORT.md  # Research validation
└── PROSODIC_CHUNKING_IMPLEMENTATION.md  # Audio implementation
```

---

## Implementation Status

### ✅ Complete

- **Base Framework**: UniversalChunk, ChunkTrustMetrics, ChunkSalienceMetrics, UniversalChunker
- **Utilities**: allocate_attention, evict_lowest_priority_chunk, group_by_time
- **Tests**: 22 comprehensive tests (all passing)
- **Audio Chunker**: Integration with existing prosody_chunker.py
- **Documentation**: Design doc, usage guide, research validation

### 🚧 In Progress

- **UnifiedChunkingPipeline**: Cross-modal orchestration
- **Cross-modal fusion**: Salience amplification from multi-modal agreement

### ⏳ Planned (Phase 4 Roadmap)

- **Phase 4.1**: Vision Chunker (fixation-based, 3-5 days)
- **Phase 4.2**: Motion Chunker (kinematic, 3-5 days)
- **Phase 4.3**: Memory/Language/Control Chunkers (2-3 days each)
- **Phase 4.4**: Cross-Modal Fusion (2-3 days)
- **Phase 4.5**: Trust + SNARC Integration with SAGE Core (2-3 days)

---

## Next Steps

1. ✅ **Test audio chunker** with real speech data
2. **Implement UnifiedChunkingPipeline** for cross-modal coordination
3. **Start Phase 4.1** (Vision Chunker) when ready
4. **Integrate with SAGE core** for ATP-based attention allocation

---

**Generated**: October 26, 2025
**Author**: Claude
**Base Implementation Complete**: Universal chunking framework operational and tested
