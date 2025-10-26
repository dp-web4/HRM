# Hierarchical Audio Buffer Architecture - GPU Mailbox Integration

## Problem Statement

Current audio sensor implementation dumps **19GB of raw audio files** to `/tmp`, causing:
- Disk space exhaustion
- Unnecessary I/O overhead
- No intelligent filtering
- No SNARC salience integration

## Solution: Tiered GPU Mailbox Buffer

Use the existing GPU mailbox tiling architecture for intelligent audio buffering with SNARC-based promotion.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ TIER 1: Rolling Capture Buffer (GPU Mailbox - Short Term)          │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ Size: 5 seconds (80K samples @ 16kHz)                               │
│ Location: GPU unified memory (Jetson) or pinned CPU memory          │
│ Behavior: Circular buffer, always overwriting oldest                │
│ Purpose: Capture audio stream continuously, low latency             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ VAD Detection
                    (Speech detected? Promote to Tier 2)
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ TIER 2: Speech Segment Buffer (GPU Mailbox - Salient)              │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ Size: 10 speech segments (up to 10s each = 100s total)             │
│ Location: GPU unified memory with PBM mailbox                       │
│ Behavior: Queue with SNARC-based eviction (low salience dropped)   │
│ Purpose: Hold speech for transcription and salience evaluation     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ SNARC Salience
                    (High salience? Transcribe and store text)
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ TIER 3: SAGE Attention (Text + High Salience Metadata)             │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ Size: Transcriptions only (text, ~100 bytes per utterance)         │
│ Location: SNARC memory + consciousness snapshots                    │
│ Behavior: Permanent storage with context history                   │
│ Purpose: Long-term memory, conversation continuity                 │
│                                                                      │
│ Audio discarded after transcription (saves 99.9% storage)          │
└─────────────────────────────────────────────────────────────────────┘
```

## Implementation Strategy

### 1. GPU Mailbox Audio Buffers

Use the existing `tiling_mailbox_torch_extension_v2/`:

**Tier 1 - Rolling Capture (Peripheral Broadcast Mailbox)**:
```python
# Initialize PBM for circular audio buffer
audio_capture_mailbox = PeripheralBroadcastMailbox(
    record_size=480,      # 30ms frames @ 16kHz
    capacity=167,         # 5 seconds = 167 frames
    device='cuda'         # Unified memory on Jetson
)

# Continuous capture (overwrite oldest)
def audio_capture_thread():
    while running:
        frame = read_audio_frame()  # 30ms of audio
        audio_capture_mailbox.push(frame)  # Overwrites oldest automatically
```

**Tier 2 - Speech Segments (Focus Tensor Mailbox)**:
```python
# Initialize FTM for speech segment queue
speech_mailbox = FocusTensorMailbox(
    max_tensors=10,       # Up to 10 speech segments
    device='cuda'
)

# VAD detection promotes to speech buffer
def vad_check_thread():
    while running:
        if vad_detected_speech():
            # Copy rolling buffer to speech segment
            segment = audio_capture_mailbox.get_last_n_frames(167)  # 5s
            segment_tensor = torch.from_numpy(segment).cuda()

            # Push to speech mailbox with metadata
            speech_mailbox.push(segment_tensor, metadata={
                'timestamp': time.time(),
                'duration': 5.0,
                'salience': None  # To be computed
            })
```

### 2. SNARC Salience Evaluation

Integrate with SNARC memory for intelligent eviction:

```python
def evaluate_speech_salience(segment_tensor, metadata):
    """
    Compute SNARC salience for speech segment

    Uses 5D salience (Surprise, Novelty, Arousal, Reward, Conflict)
    """
    # Quick transcription for context
    transcription = whisper_transcribe(segment_tensor)

    # SNARC evaluation
    salience = snarc_memory.calculate_salience(
        text=transcription,
        audio_features=extract_prosody(segment_tensor),
        context=recent_conversation_history
    )

    return {
        'text': transcription,
        'salience': salience,
        'timestamp': metadata['timestamp']
    }

def snarc_eviction_policy():
    """
    When speech mailbox full, evict lowest salience segment
    """
    if speech_mailbox.count() >= 10:
        # Find lowest salience segment
        segments = speech_mailbox.get_all_with_metadata()
        lowest = min(segments, key=lambda s: s['metadata']['salience'])

        # Discard low-salience audio (don't transcribe)
        speech_mailbox.remove(lowest['id'])
        print(f"Evicted low-salience audio: salience={lowest['metadata']['salience']:.2f}")
```

### 3. Transcription Pipeline

Only transcribe high-salience speech:

```python
def transcription_worker():
    """
    Asynchronous transcription of high-salience speech
    """
    while running:
        # Get next speech segment
        segment = speech_mailbox.pop()
        if segment is None:
            time.sleep(0.1)
            continue

        # Evaluate salience
        result = evaluate_speech_salience(segment['tensor'], segment['metadata'])

        if result['salience'] > SALIENCE_THRESHOLD:
            # High salience - store transcription, discard audio
            sage_memory.add_turn("User", result['text'])

            # Free GPU memory (audio no longer needed)
            del segment['tensor']

            print(f"Transcribed: '{result['text'][:50]}...' (salience={result['salience']:.2f})")
        else:
            # Low salience - discard everything
            print(f"Skipped low-salience speech (salience={result['salience']:.2f})")
```

## Storage Savings

### Before (Current System):
```
5 minute conversation @ 16kHz 16-bit:
= 5 * 60 * 16000 * 2 bytes
= 9.6 MB per minute
= 576 MB per hour
= 13.8 GB per day
```

### After (Hierarchical Buffer):
```
Tier 1 (Rolling): 5s * 32KB = 160 KB (constant, never grows)
Tier 2 (Speech):  10 segments * 320 KB = 3.2 MB (max, usually less)
Tier 3 (Text):    100 turns * 100 bytes = 10 KB

Total: ~3.4 MB maximum (vs 13.8 GB per day)
Reduction: 99.98%
```

## Performance Benefits

### Memory Efficiency:
- **19 GB → 3.4 MB** (5700x reduction)
- Constant memory footprint (circular buffers)
- No disk I/O for rejected audio

### Latency Improvements:
- **Tier 1**: 0ms (always in memory)
- **Tier 2**: <10ms (GPU memory copy)
- **Tier 3**: ~100ms (transcription only when needed)

### SNARC Integration:
- Only transcribe salient speech (save compute)
- Prosody features available for salience
- Context-aware filtering

## GPU Mailbox Benefits

Why use the existing mailbox architecture:

1. **Zero-copy tensor handoff** (FTM)
   - Audio stays on GPU
   - No CPU↔GPU transfers
   - Direct to Whisper model

2. **Synchronization built-in**
   - Count-based pops prevent race conditions
   - Thread-safe by design
   - Tested on RTX 4090, Jetson Orin

3. **Hierarchical tiling**
   - Natural fit for tiered buffers
   - Each tier is a separate mailbox
   - Proven architecture

4. **Already implemented and tested**
   - 2.9 tiles/sec on RTX 2060 SUPER
   - 55-60x faster on Jetson Orin Nano
   - Production-ready

## Implementation Plan

### Phase 1: Basic Tier Structure ✅ (Design)
- [x] Define mailbox parameters
- [x] Design tier promotion logic
- [ ] Create AudioBufferManager class

### Phase 2: GPU Mailbox Integration
- [ ] Initialize PBM for Tier 1 (rolling capture)
- [ ] Initialize FTM for Tier 2 (speech segments)
- [ ] Implement VAD promotion (Tier 1→2)

### Phase 3: SNARC Salience
- [ ] Integrate SNARC salience calculation
- [ ] Implement eviction policy (low salience)
- [ ] Add prosody feature extraction

### Phase 4: Transcription Pipeline
- [ ] Async transcription worker
- [ ] Salience-based filtering
- [ ] Text storage (discard audio)

### Phase 5: Production Integration
- [ ] Replace current StreamingAudioSensor temp files
- [ ] Add monitoring/stats
- [ ] Tune salience thresholds

## Code Location

New files to create:
```
interfaces/
├── hierarchical_audio_buffer.py    # Main AudioBufferManager
├── audio_mailbox.py                # GPU mailbox wrapper
├── audio_snarc_filter.py           # SNARC salience for audio
└── test_hierarchical_buffer.py     # Unit tests
```

Existing to modify:
```
interfaces/
└── streaming_audio_sensor.py       # Replace temp file logic
```

## Key Insight

**Audio is ephemeral, transcription is permanent.**

Current system treats all audio equally (save everything).
New system applies SAGE's core principle: **attention is selective**.

SNARC determines what deserves transcription, GPU mailbox ensures efficient processing, consciousness persistence keeps what matters.

This is the same pattern as KV-cache consciousness:
- Ephemeral: Audio in GPU memory (like attention states)
- Salient: Transcribed speech (like important context)
- Persistent: Text in snapshots (like conversation history)

**Biological parallel**: Human auditory memory
- Echoic memory: ~5s rolling buffer (Tier 1)
- Working memory: Current speech segments (Tier 2)
- Long-term memory: Understood meaning (Tier 3)

We're not storing audio - we're storing understanding.
