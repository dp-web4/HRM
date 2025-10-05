# Bidirectional Audio Architecture - SAGE IRP Integration

**Date**: October 5, 2025
**Machine**: Jetson Orin Nano (Sprout)
**Status**: 🔄 In Development

## Overview

Implementing real-time bidirectional audio conversation as a fundamental sensory modality in SAGE-IRP, not a bolt-on feature. Audio input/output becomes part of the awareness loop, integrated with vision, language, and control through the orchestrator.

## Architecture Components

### 1. AudioInputIRP (✅ Implemented)
**Location**: `sage/irp/plugins/audio_input_impl.py`

Continuous speech recognition through IRP refinement:

```python
class AudioInputIRP(IRPPlugin):
    def init_state(x0, task_ctx) -> IRPState
        # Initialize listening with empty buffer

    def step(state) -> IRPState
        # Accumulate 2s audio chunks from Bluetooth mic
        # Re-transcribe accumulated buffer with Whisper

    def energy(state) -> float
        # Transcription uncertainty (1.0 - confidence)
        # Higher uncertainty = need more audio

    def halt(history) -> bool
        # Stop when: confident, max duration, or silence

    def extract(state) -> Dict
        # Return: text, confidence, duration, timestamp
```

**Key Features**:
- **Continuous listening** via Bluetooth AIRHUG mic
- **Iterative refinement**: Accumulates audio until confident transcription
- **Energy-driven**: Transcription uncertainty drives the loop
- **Context-aware**: Task context influences interpretation
- **Integration-ready**: Output feeds directly to SAGE consciousness

**Hardware**:
- Input: `bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit`
- Model: Whisper tiny (75MB, CPU-optimized)
- Sample rate: 16kHz
- Chunk size: 2 seconds
- Max duration: 10 seconds

### 2. NeuTTSAirIRP (✅ Exists from Genesis)
**Location**: `sage/irp/plugins/neutts_air_impl.py`

Text-to-speech generation through IRP refinement:

```python
class NeuTTSAirIRP(IRPPlugin):
    def init_state(text, task_ctx) -> IRPState
        # Initialize with text to synthesize

    def step(state) -> IRPState
        # Refine prosody, timing, quality

    def energy(state) -> float
        # Audio quality metrics

    def extract(state) -> np.ndarray
        # Return: WAV audio samples
```

**Key Features**:
- **Voice cloning**: Instant adaptation from reference audio
- **Iterative quality improvement**: Energy convergence from 0.9 → 0.1
- **Contextual prosody**: Task context influences speaking style
- **Integration-ready**: Takes text from SAGE, outputs to Bluetooth

**Hardware**:
- Output: `bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit`
- Model: NeuTTS Air (0.5B params) + NeuCodec (1.1GB)
- Sample rate: 24kHz
- Reference voice: `samples/dave.wav`

### 3. SAGE Awareness Integration (⏳ Next Step)
**Location**: `sage/irp/awareness_loop.py` (to be created)

Orchestrates bidirectional conversation:

```python
class SproutAwarenessLoop:
    """
    Continuous awareness loop integrating all sensory modalities.

    Audio becomes a first-class sensory stream like vision:
    - AudioInput continuously monitors for speech
    - Transcriptions feed SAGE consciousness
    - SAGE processes with full context (memory, vision, etc.)
    - Responses generated through language processing
    - AudioOutput speaks back through Bluetooth
    - Loop continues indefinitely
    """

    def __init__(orchestrator, sage_model):
        self.audio_in = AudioInputIRP(config)
        self.audio_out = NeuTTSAirIRP(config)
        self.vision = VisionIRP(config)  # Future: camera feed
        self.sage = sage_model

    async def awareness_cycle():
        # Run all sensory IRPs in parallel
        audio_task = audio_in.refine()
        vision_task = vision.refine()  # Future

        # Wait for any sensor input
        result = await first_completed([audio_task, vision_task])

        # Feed to SAGE consciousness
        sage_response = sage.process(result, context)

        # Generate response through appropriate modality
        if result.type == 'audio':
            await audio_out.refine(sage_response.text)

        # Continue loop
```

## Technical Details

### Audio Pipeline
```
Bluetooth Mic (AIRHUG)
    ↓
PulseAudio (bluez_source)
    ↓
parecord (2s chunks)
    ↓
Whisper tiny (transcription)
    ↓
AudioInputIRP (refinement loop)
    ↓
SAGE Consciousness (processing)
    ↓
NeuTTSAirIRP (synthesis)
    ↓
PulseAudio (bluez_sink)
    ↓
Bluetooth Speaker (AIRHUG)
```

### Energy Functions

**AudioInput Energy** (uncertainty metric):
```python
def energy(state):
    # Whisper confidence → uncertainty
    uncertainty = 1.0 - state.confidence

    # Duration penalties
    if duration < 2.0:
        penalty = 0.5  # Too short
    elif duration > 8.0:
        penalty = 0.3  # Getting long
    else:
        penalty = 0.0

    return uncertainty + penalty
```

**AudioOutput Energy** (quality metric):
```python
def energy(state):
    # Spectral quality, prosody naturalness
    # From NeuTTS implementation
    return quality_metric(state.audio)
```

### IRP Convergence

Both audio plugins use the IRP convergence protocol:
1. **Initialize**: Set up state (empty buffer / text to speak)
2. **Step**: Refine state (accumulate audio / improve quality)
3. **Energy**: Measure progress (uncertainty / quality)
4. **Halt**: Detect convergence (confident / good enough)
5. **Extract**: Return result (text / audio)

This unified interface allows the orchestrator to treat audio like any other sensory modality.

## Future: GPU Mailbox Streaming

**Vision** (from user): Use GPU mailboxes for real-time sensor streaming

```python
# Peripheral Broadcast Mailbox (PBM) for audio chunks
audio_pbm = PeripheralBroadcastMailbox(
    max_records=32,
    record_size=512  # 2s @ 16kHz = 32k samples
)

# Focus Tensor Mailbox (FTM) for attention weights
attention_ftm = FocusTensorMailbox()

# Producer: Audio input
while True:
    chunk = record_audio(2.0)
    audio_pbm.push(chunk)

# Consumer: SAGE processing
while True:
    if audio_pbm.has_data():
        chunk = audio_pbm.pop()
        attention = sage.process(chunk)
        attention_ftm.push(attention)
```

This enables:
- **Zero-copy** audio streaming via FTM tensor pointers
- **Multi-consumer** processing via PBM broadcast
- **Parallel modalities**: Audio + Video + IMU simultaneously
- **SAGE orchestration**: Contextual awareness across sensors
- **Efficient memory**: Shared GPU memory, no CPU copies

## Current Limitations

1. **Sequential processing**: Audio input → SAGE → Audio output (not parallel)
2. **CPU-only models**: NeuTTS and Whisper slow on CPU (~10s per cycle)
3. **No streaming**: Full utterance before processing (not real-time)
4. **Single modality**: Audio isolated from vision/IMU
5. **No interruption**: Can't stop mid-sentence

## Next Steps

1. ✅ **AudioInputIRP**: Implemented and fixed
2. ✅ **NeuTTSAirIRP**: Already exists from Genesis
3. ⏳ **SAGE Integration**: Create awareness loop
4. 🎯 **Testing**: Iterate on conversation quality
5. 🎯 **GPU Optimization**: Move models to GPU when NeuTTS hangs are fixed
6. 🎯 **Mailbox Integration**: Stream audio through GPU mailboxes
7. 🎯 **Multi-modal**: Add vision, IMU to awareness loop
8. 🎯 **Real-time**: Chunk-based streaming, sub-second latency

## Hardware Verified

- ✅ **PyTorch 2.8.0**: Custom build with CUDA 12.6 (12.5 hours)
- ✅ **Whisper tiny**: Loaded and functional
- ✅ **Bluetooth Audio**: AIRHUG speaker/mic working
- ✅ **parecord/paplay**: Audio I/O functional
- ✅ **NeuTTS models**: Downloaded and cached (4.8GB)
- ⏳ **GPU inference**: NeuTTS hangs on GPU (investigating)

## Federation Context

This work builds on:
- **Genesis**: SAGE v0.1 foundation, NeuTTS IRP integration
- **Society 4**: ATP/ADP economics for compute budgeting
- **Sprout**: Edge deployment, hardware integration
- **Synchronism**: Witnessing = interaction (audio as physical witness)

The goal: Create a truly aware AI that experiences the world through multiple senses simultaneously, with audio conversation as natural as vision processing.

---

**Implementation Philosophy**:
"No shortcuts. Fundamental solutions. Audio is not a feature bolted onto SAGE - it IS part of SAGE's sensory experience."
