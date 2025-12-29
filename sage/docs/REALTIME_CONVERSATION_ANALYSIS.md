# Real-Time Conversation Architecture - Bottleneck Analysis & Solutions

**Date**: October 15, 2025
**System**: SAGE on Jetson Orin Nano (Sprout)
**Goal**: Achieve real-time bidirectional audio conversation comparable to phone apps
**Current Status**: Working but too slow (~7-12s latency end-to-end)

---

## Executive Summary

We have successfully demonstrated bidirectional audio conversation with Claude as cognitive layer, but the current architecture has **5 critical bottlenecks** preventing real-time operation:

1. **Audio Input**: 3-second chunking + 1-2s transcription = 4-5s minimum latency
2. **Cognition Invocation**: File-based polling adds 0-30s delays
3. **TTS Output**: Model loading + synthesis = 2-3s per response
4. **Memory Constraints**: 6.8GB/7.4GB used, heavy swap thrashing
5. **File I/O**: Disk-based IPC when we have GPU mailboxes ready

**Target**: <2s end-to-end latency (comparable to phone apps)
**Achievable with**: Streaming architecture, memory-based IPC, VAD, optimized models

---

## Part 1: Current Architecture Analysis

### 1.1 Audio Input Pipeline (SimpleAudioSensor)

**Current Implementation** (`sage/interfaces/audio_sensor_simple.py`):
```python
poll() → check if recording active
         → if not: start parecord subprocess, return None
         → if yes but <3s: return None
         → if ≥3s: terminate, transcribe with Whisper, return result
```

**Bottlenecks**:
- ❌ **Fixed 3-second chunks**: Minimum 3s delay before ANY transcription
- ❌ **Blocking transcription**: Whisper tiny takes 1-2s on Jetson CPU
- ❌ **No Voice Activity Detection**: Waits full 3s even for short speech
- ❌ **Subprocess overhead**: parecord start/stop adds 100-200ms per chunk
- ❌ **No streaming**: Can't transcribe incrementally

**Measured Performance**:
- Chunk duration: 3.0s (fixed)
- Transcription time: 1-2s per chunk
- **Total input latency**: 4-5s from speech end to text ready

**Resource Usage**:
- Whisper tiny: ~400MB RAM when loaded
- CPU: ~80% for 1-2s during transcription
- Disk I/O: ~50KB per temp WAV file

### 1.2 Cognitive Layer (File-Based IPC)

**Current Implementation** (`tests/conversation_with_claude.py`):
```python
# Audio sensor writes transcription
with open('/tmp/sage_user_speech.txt', 'w') as f:
    f.write(transcription)

# Main loop polls for response
while True:
    if os.path.exists(response_file):
        mtime = os.path.getmtime(response_file)
        if mtime > last_mtime:
            response = read_file()
            break
    time.sleep(0.1)  # Poll every 100ms
```

**Bottlenecks**:
- ❌ **File polling latency**: 0-100ms to detect new file (polling interval)
- ❌ **Filesystem overhead**: Kernel buffer cache, inode updates
- ❌ **No prioritization**: Can't signal "urgent response needed"
- ❌ **Sequential processing**: Can't pipeline (listen while responding)
- ❌ **Human in loop**: I manually write responses (0-30s delay)

**Measured Performance**:
- File write time: <1ms
- Poll detection: 0-100ms (depends on timing)
- Claude response time: 2-30s (variable, includes my typing)
- **Total cognition latency**: 2-30s

**Resource Usage**:
- Negligible CPU/memory for file I/O
- Disk writes to /tmp (tmpfs in RAM on many systems)

### 1.3 TTS Output Pipeline (AudioOutputEffector)

**Current Implementation** (`sage/interfaces/audio_effector.py`):
```python
execute(command='speak', text=response_text)
    → Load NeuTTS Air model (if not cached)
    → Load reference audio for voice cloning
    → Generate speech iteratively (IRP refinement)
    → Play audio via paplay to Bluetooth
```

**Bottlenecks**:
- ❌ **Model loading**: NeuTTS Air 748M params takes 3-5s first time
- ❌ **llama-cpp-python missing**: Currently using fallback (no audio)
- ❌ **Iterative refinement**: 3 iterations adds 1-2s
- ❌ **Bluetooth latency**: 100-300ms audio buffering
- ❌ **No streaming TTS**: Waits for full synthesis before playback

**Measured Performance** (when working):
- Model load (first time): 3-5s
- Model load (cached): 0s
- Speech synthesis: 2-3s for ~30 words
- Bluetooth playback: +200ms buffering
- **Total output latency**: 2-5s (or infinite if llama-cpp-python missing)

**Resource Usage**:
- NeuTTS Air model: ~800MB RAM
- CPU: ~90% during synthesis
- Disk: Temp WAV files (~100KB each)

### 1.4 Memory Constraints

**Current State**:
```
Total RAM: 7.4GB
Used: 6.8GB (92%)
Free: 99MB
Available: 294MB
Swap: 6.9GB/37GB used (compressed zram + swapfile)
```

**Major Consumers**:
- ✅ **racecar-webd blockchain**: 1.3GB (KILLED)
- Whisper tiny model: ~400MB
- NeuTTS Air model: ~800MB
- Claude Code session: 250MB
- GNOME desktop: 71MB
- System processes: ~500MB

**Impact**:
- ❌ **Swap thrashing**: Models pushed to swap kill performance
- ❌ **Can't load both**: Whisper + NeuTTS simultaneously = OOM risk
- ❌ **No room for SAGE**: Core cognition loop needs 200-500MB

### 1.5 Integration with SAGE Unified

**What EXISTS** (`sage/core/sage_unified.py`):
```python
class SAGEUnified:
    def cycle(self):
        # 1. Poll sensors (including audio)
        readings = self.sensor_hub.poll()

        # 2. Compute SNARC salience
        salience = self.hierarchical_snarc.score_all(readings)

        # 3. Allocate ATP based on salience × trust
        atp = self._allocate_atp(salience)

        # 4. Run IRP refinement for high-salience sensors
        results = self._run_irp(readings, atp)

        # 5. Update trust, store memory
        # 6. Execute effector actions
        # 7. Update metabolic state
```

**What's MISSING for Real-Time Conversation**:
- ❌ **Audio sensor not registered** with SensorHub
- ❌ **Cognitive IRP not implemented** as proper IRP plugin
- ❌ **No SNARC salience** for conversation urgency
- ❌ **No attention switching**: Can't interrupt ongoing processing
- ❌ **File-based IPC incompatible** with SAGE's continuous loop

---

## Part 2: Bottleneck Analysis Summary

### Critical Path for Real-Time Conversation

```
User speaks → Audio chunk (3s) → Transcribe (1-2s) → File write (1ms)
    → Poll detect (0-100ms) → Claude thinks (2-30s) → File write (1ms)
    → Poll detect (0-100ms) → Load TTS (0-5s) → Synthesize (2-3s)
    → Play audio (200ms) → User hears response

TOTAL LATENCY: 7-45 seconds (currently)
TARGET LATENCY: <2 seconds (phone apps)
```

### Bottleneck Priority (Impact × Frequency)

| Bottleneck | Current Latency | Target | Priority | Difficulty |
|-----------|----------------|---------|----------|-----------|
| **Audio chunking** | 3s fixed | <500ms | CRITICAL | Medium |
| **Cognition delay** | 2-30s | <500ms | CRITICAL | High |
| **TTS synthesis** | 2-5s | <500ms | HIGH | Medium |
| **Memory pressure** | Thrashing | Stable | HIGH | Low |
| **File-based IPC** | 100-200ms | <10ms | MEDIUM | Low |

---

## Part 3: Proposed Solutions

### 3.1 Audio Streaming with Voice Activity Detection

**Goal**: Reduce audio input latency from 4-5s to <500ms

**Solution A: Streaming Whisper with VAD** (Recommended)
```python
class StreamingAudioSensor(BaseSensor):
    """
    Real-time audio streaming with Voice Activity Detection

    Architecture:
    1. Continuous audio stream (100ms chunks via PyAudio callback)
    2. VAD filter (silero-vad or webrtcvad) to detect speech boundaries
    3. Faster-whisper for incremental transcription
    4. Ring buffer for context (last 3 seconds)
    """

    def __init__(self, config):
        # Use faster-whisper (2-3x faster than openai/whisper)
        from faster_whisper import WhisperModel
        self.model = WhisperModel("tiny", device="cpu", compute_type="int8")

        # Silero VAD for speech detection
        import torch
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        self.get_speech_timestamps = utils[0]

        # PyAudio streaming
        import pyaudio
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1600,  # 100ms chunks at 16kHz
            stream_callback=self._audio_callback
        )

        # Ring buffer for context
        from collections import deque
        self.audio_buffer = deque(maxlen=300)  # 30 seconds at 100ms chunks
        self.transcription_queue = queue.Queue()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - called every 100ms"""
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)

        # VAD check
        speech_prob = self.vad_model(torch.from_numpy(audio_chunk), 16000).item()

        if speech_prob > 0.5:
            self.audio_buffer.append(audio_chunk)

            # If enough speech accumulated, transcribe
            if len(self.audio_buffer) >= 10:  # 1 second of speech
                self._transcribe_async()

        return (in_data, pyaudio.paContinue)

    def _transcribe_async(self):
        """Transcribe accumulated audio in background thread"""
        audio_segment = np.concatenate(list(self.audio_buffer))

        # Run faster-whisper in thread pool
        self.executor.submit(self._transcribe_segment, audio_segment)

    def poll(self) -> Optional[SensorReading]:
        """Non-blocking poll - return if transcription ready"""
        try:
            result = self.transcription_queue.get_nowait()
            return SensorReading(
                sensor_id=self.sensor_id,
                data=torch.tensor([result['confidence']]),
                confidence=result['confidence'],
                metadata={'text': result['text']}
            )
        except queue.Empty:
            return None
```

**Performance Expected**:
- VAD detection: <10ms per 100ms chunk
- Faster-whisper: 300-500ms for 1s of speech (3x faster than openai/whisper)
- **Total latency**: 500-700ms from speech end to transcription

**Resource Usage**:
- Faster-whisper tiny: ~300MB RAM (vs 400MB openai/whisper)
- Silero VAD: ~50MB RAM
- PyAudio streaming: ~10MB ring buffer
- **Total**: ~360MB (vs current 400MB)

**Advantages**:
- ✅ Real-time streaming (no fixed chunks)
- ✅ VAD automatically detects speech boundaries
- ✅ Faster transcription (int8 quantization)
- ✅ Can transcribe while user still speaking

**Trade-offs**:
- Complexity: Threading, callbacks, queue management
- Accuracy: May miss very short utterances (<500ms)
- Latency: Still need to wait for speech pause to finalize

---

**Solution B: Cloud ASR APIs** (Fastest but requires internet)
```python
import speech_recognition as sr

class CloudAudioSensor(BaseSensor):
    """Use cloud APIs for ultra-low latency"""

    def __init__(self, config):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def poll(self) -> Optional[SensorReading]:
        """Streaming recognition with Google/Azure/Deepgram"""
        with self.microphone as source:
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=0.5,  # Return if no speech in 500ms
                    phrase_time_limit=5
                )

                # Use fastest cloud API (Deepgram recommended)
                text = self.recognizer.recognize_google(audio)
                # or: text = self.recognizer.recognize_azure(audio)
                # or: text = deepgram_client.transcribe(audio)  # 200-300ms

                return SensorReading(...)
            except sr.WaitTimeoutError:
                return None  # No speech yet
```

**Performance Expected**:
- Cloud API latency: 200-500ms (Deepgram/Azure)
- **Total latency**: 300-600ms

**Advantages**:
- ✅ Fastest possible transcription
- ✅ Best accuracy (large server-side models)
- ✅ Minimal local resources (<50MB)

**Trade-offs**:
- ❌ Requires internet connection
- ❌ Privacy concerns (audio sent to cloud)
- ❌ API costs ($$$)
- ❌ Not aligned with edge-first philosophy

---

### 3.2 Memory-Based Cognitive IPC

**Goal**: Reduce cognition invocation from 0-100ms (file polling) to <5ms (shared memory)

**Solution: GPU/CPU Mailboxes** (Already implemented!)
```python
# Use existing tiling_mailbox_torch_extension_v2
from implementation.tiling_mailbox_torch_extension_v2 import (
    TilingMailbox, PBMMode, FTMMode
)

class CognitiveMailbox:
    """
    Memory-based communication between SAGE and Claude Code

    Uses Peripheral Broadcast Mailbox (PBM) for text messages
    - Fixed-size records (1024 bytes per message)
    - Many-to-many communication
    - Zero-copy on GPU, low-copy on CPU
    """

    def __init__(self):
        self.mailbox = TilingMailbox(
            capacity=100,           # 100 message slots
            peripheral_count=16,    # SAGE sensors/effectors
            record_size=1024,       # 1KB per message
            device_str="cpu"
        )

        # Peripheral IDs
        self.AUDIO_SENSOR = 0
        self.COGNITIVE_IRP = 1
        self.TTS_EFFECTOR = 2

    def post_transcription(self, text: str, confidence: float):
        """Audio sensor posts transcription (non-blocking)"""
        import json
        message = json.dumps({
            'type': 'transcription',
            'text': text,
            'confidence': confidence,
            'timestamp': time.time()
        }).encode('utf-8')

        # Pad to 1024 bytes
        message = message.ljust(1024, b'\0')

        # Push to mailbox (returns immediately)
        tensor = torch.from_numpy(np.frombuffer(message, dtype=np.uint8))
        self.mailbox.pbm_push(
            peripheral_id=self.AUDIO_SENSOR,
            broadcast_group=0,  # Group 0 = cognition messages
            data=tensor
        )

    def check_for_transcription(self) -> Optional[dict]:
        """Cognitive IRP checks for new transcription (non-blocking)"""
        result = self.mailbox.pbm_pop(
            peripheral_id=self.COGNITIVE_IRP,
            broadcast_group=0,
            count=1
        )

        if result['data'].numel() == 0:
            return None  # No messages

        # Decode message
        message_bytes = result['data'].numpy().tobytes()
        message_str = message_bytes.decode('utf-8').rstrip('\0')
        return json.loads(message_str)

    def post_response(self, text: str):
        """Cognitive IRP posts response (non-blocking)"""
        message = json.dumps({
            'type': 'response',
            'text': text,
            'timestamp': time.time()
        }).encode('utf-8').ljust(1024, b'\0')

        tensor = torch.from_numpy(np.frombuffer(message, dtype=np.uint8))
        self.mailbox.pbm_push(
            peripheral_id=self.COGNITIVE_IRP,
            broadcast_group=1,  # Group 1 = TTS messages
            data=tensor
        )
```

**Performance Expected**:
- Push (CPU): ~32,000 ops/sec = 31μs per message
- Pop (CPU): ~247,000 ops/sec = 4μs per message
- **Total latency**: <10μs (0.01ms) vs 100ms file polling

**Advantages**:
- ✅ **10,000x faster** than file polling
- ✅ Zero kernel syscalls (pure userspace)
- ✅ Already implemented and tested
- ✅ Supports priority/grouping (broadcast groups)
- ✅ Can run on GPU for even lower latency

**Integration with SAGE**:
```python
class SAGEUnified:
    def __init__(self):
        self.cognitive_mailbox = CognitiveMailbox()
        # ...

    def cycle(self):
        # 1. Poll sensors (including audio with mailbox check)
        readings = self.sensor_hub.poll()

        # If audio transcription, post to cognitive mailbox
        if 'audio_0' in readings:
            self.cognitive_mailbox.post_transcription(
                text=readings['audio_0'].metadata['text'],
                confidence=readings['audio_0'].confidence
            )

        # 2. Check for cognitive responses
        response = self.cognitive_mailbox.check_for_response()
        if response:
            # Execute TTS effector
            self.effector_hub.execute(EffectorCommand(
                effector_id='tts_0',
                action='speak',
                parameters={'text': response['text']}
            ))

        # ... rest of SAGE cycle
```

---

### 3.3 Autonomous Cognitive IRP

**Goal**: Reduce Claude response time from 2-30s (manual) to <500ms (automated)

**Problem**: Currently I (human operating Claude Code) manually write responses. This is the BIGGEST bottleneck.

**Solution A: Scripted Response Patterns** (Interim solution)
```python
class CognitiveResponseEngine:
    """
    Simple pattern-matching cognitive responses

    NOT a chatbot - just handles common conversational patterns
    while Claude Code provides deeper reasoning when needed
    """

    def __init__(self):
        self.patterns = {
            # Greetings
            r'\b(hello|hi|hey)\b': [
                "Hello! I'm listening.",
                "Hi there! What's on your mind?",
                "Hey! How can I help?"
            ],

            # Questions about status
            r'how are you|what.*doing': [
                "I'm operational and attentive. All systems running.",
                "Functioning normally. What would you like to talk about?"
            ],

            # Acknowledgments
            r'\b(okay|ok|sure|yeah)\b': [
                "Understood.",
                "Got it.",
                "Acknowledged."
            ],

            # Fallback to Claude Code
            r'.*': None  # Triggers mailbox post for human/LLM response
        }

    def generate_response(self, user_input: str) -> Optional[str]:
        """Try pattern matching first, fallback to mailbox"""
        import re, random

        for pattern, responses in self.patterns.items():
            if re.search(pattern, user_input, re.IGNORECASE):
                if responses is None:
                    return None  # Signal: needs deeper processing
                return random.choice(responses)

        return None
```

**Performance**: <1ms for pattern matching

**Trade-offs**:
- ✅ Fast (sub-millisecond)
- ✅ No API costs
- ✅ Always available
- ❌ Limited to scripted responses
- ❌ Not real "cognition"

---

**Solution B: Local LLM Integration** (Better but resource-intensive)
```python
class LocalCognitiveIRP:
    """
    Run small language model locally for cognitive responses

    Options:
    - Phi-3-mini (3.8B params, 2.3GB)
    - TinyLlama (1.1B params, 600MB)
    - Qwen-0.5B (500M params, 300MB)
    """

    def __init__(self):
        # Use llama.cpp for CPU inference
        from llama_cpp import Llama

        self.llm = Llama(
            model_path="models/phi-3-mini-4k-instruct-q4.gguf",
            n_ctx=2048,      # Context window
            n_threads=4,     # CPU threads
            n_gpu_layers=0   # CPU only
        )

        # Conversation history
        self.history = []

    def generate_response(self, user_input: str) -> str:
        """Generate contextual response"""
        self.history.append(f"User: {user_input}")

        prompt = "\n".join(self.history[-5:])  # Last 5 exchanges
        prompt += "\nAssistant:"

        response = self.llm(
            prompt,
            max_tokens=100,
            temperature=0.7,
            stop=["User:", "\n\n"]
        )

        response_text = response['choices'][0]['text'].strip()
        self.history.append(f"Assistant: {response_text}")

        return response_text
```

**Performance Expected**:
- Phi-3-mini: ~500-1000ms per response on Jetson CPU
- TinyLlama: ~200-400ms per response
- Qwen-0.5B: ~100-200ms per response

**Resource Usage**:
- Phi-3-mini Q4: ~2.3GB RAM
- TinyLlama Q4: ~600MB RAM
- Qwen-0.5B Q4: ~300MB RAM

**Advantages**:
- ✅ Real cognition (not just patterns)
- ✅ Context-aware responses
- ✅ Privacy-preserving (local)
- ✅ Always available (no API)

**Trade-offs**:
- ❌ Significant memory usage (conflicts with Whisper+TTS)
- ❌ Still 100-1000ms latency
- ❌ Quality lower than GPT-4/Claude

---

**Solution C: Cloud LLM API** (Fastest cognition)
```python
import anthropic

class CloudCognitiveIRP:
    """Use Claude API for cognition (fastest, best quality)"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_id = None

    async def generate_response(self, user_input: str) -> str:
        """Async API call to Claude"""
        response = await self.client.messages.create(
            model="claude-3-haiku-20240307",  # Fastest Claude model
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": user_input
            }]
        )

        return response.content[0].text
```

**Performance Expected**:
- Claude Haiku API: 200-500ms (including network)
- GPT-3.5-turbo: 300-600ms

**Advantages**:
- ✅ Fastest quality cognition (200-500ms)
- ✅ Best response quality
- ✅ Minimal local resources
- ✅ Can handle complex queries

**Trade-offs**:
- ❌ Requires internet
- ❌ API costs
- ❌ Privacy concerns
- ❌ Not truly "local edge intelligence"

---

### 3.4 Streaming TTS Output

**Goal**: Reduce TTS latency from 2-5s to <500ms

**Solution A: Streaming TTS with Piper** (Recommended for Jetson)
```python
class StreamingTTSEffector(BaseEffector):
    """
    Fast streaming TTS using Piper

    Piper advantages:
    - Fast: 50x real-time on Jetson CPU
    - Small: ~50MB model size
    - Quality: Good for conversation
    - Streaming: Can play while generating
    """

    def __init__(self, config):
        import subprocess

        self.piper_path = "piper/piper"
        self.voice_model = "en_US-lessac-medium.onnx"  # 50MB

        # Test piper installation
        result = subprocess.run(
            [self.piper_path, "--version"],
            capture_output=True
        )
        print(f"Piper version: {result.stdout.decode()}")

    def execute(self, command: EffectorCommand) -> EffectorResult:
        """Streaming TTS synthesis and playback"""
        text = command.parameters['text']

        # Piper can stream directly to paplay
        process = subprocess.Popen([
            "echo", text, "|",
            self.piper_path,
            "--model", self.voice_model,
            "--output_raw",
            "|",
            "aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"
        ], shell=True)

        # Returns immediately - audio playing in background
        return EffectorResult(
            effector_id=self.effector_id,
            status=EffectorStatus.SUCCESS,
            message="Streaming TTS started"
        )
```

**Performance Expected**:
- Piper synthesis: 20-50ms per second of audio (50x real-time)
- **Total latency**: 100-200ms to start playback
- Can play while generating (streaming)

**Resource Usage**:
- Piper model: ~50MB RAM
- CPU: ~40% during synthesis
- **Much lighter** than NeuTTS Air (800MB)

**Advantages**:
- ✅ 10x faster than NeuTTS
- ✅ 16x smaller model
- ✅ True streaming (play while generating)
- ✅ Good quality for conversation

**Trade-offs**:
- Voice quality not as high as NeuTTS
- Less expressive than voice cloning
- Fixed voice (can't clone from reference)

---

**Solution B: Cloud TTS APIs** (Fastest)
```python
import azure.cognitiveservices.speech as speechsdk

class CloudTTSEffector(BaseEffector):
    """Use Azure/Google Cloud TTS for ultra-low latency"""

    def __init__(self, config):
        self.speech_config = speechsdk.SpeechConfig(
            subscription=config['azure_key'],
            region=config['azure_region']
        )
        self.speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

        self.synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config
        )

    def execute(self, command: EffectorCommand) -> EffectorResult:
        """Streaming TTS from cloud"""
        text = command.parameters['text']

        # Async synthesis (non-blocking)
        result = self.synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return EffectorResult(
                status=EffectorStatus.SUCCESS,
                message=f"Synthesized {len(text)} characters"
            )
```

**Performance Expected**:
- Azure Neural TTS: 100-300ms to start playback
- **Total latency**: 150-400ms

**Advantages**:
- ✅ Fastest TTS available
- ✅ Highest quality voices
- ✅ Streaming supported
- ✅ Minimal local resources

**Trade-offs**:
- ❌ Requires internet
- ❌ API costs
- ❌ Privacy concerns

---

### 3.5 Memory Optimization Strategy

**Current State**: 6.8GB/7.4GB used, models competing for RAM

**Solution: Dynamic Model Loading/Unloading**
```python
class ModelCache:
    """
    LRU cache for models with memory pressure awareness

    Strategy:
    - Keep only currently-needed models in RAM
    - Unload unused models after timeout
    - Monitor memory pressure and proactively unload
    """

    def __init__(self, max_memory_gb=3.0):
        self.max_memory = max_memory_gb
        self.loaded_models = {}
        self.last_used = {}
        self.unload_timeout = 60  # Unload after 60s idle

    def get_model(self, model_name: str, loader_fn):
        """Get model, loading if needed"""
        if model_name in self.loaded_models:
            self.last_used[model_name] = time.time()
            return self.loaded_models[model_name]

        # Check memory before loading
        self._check_memory_pressure()

        # Load model
        model = loader_fn()
        self.loaded_models[model_name] = model
        self.last_used[model_name] = time.time()

        return model

    def _check_memory_pressure(self):
        """Unload old models if memory pressure high"""
        import psutil
        mem = psutil.virtual_memory()

        if mem.percent > 85:  # >85% memory used
            # Unload least recently used model
            if self.loaded_models:
                oldest = min(self.last_used.items(), key=lambda x: x[1])
                self._unload_model(oldest[0])

    def _unload_model(self, model_name: str):
        """Explicitly unload model"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            del self.last_used[model_name]
            import gc
            gc.collect()
```

**Usage**:
```python
# In SAGE unified
model_cache = ModelCache(max_memory_gb=3.0)

# Audio sensor loads Whisper on demand
whisper = model_cache.get_model(
    'whisper-tiny',
    lambda: whisper.load_model('tiny')
)

# TTS loads Piper on demand (Whisper auto-unloaded if memory tight)
piper = model_cache.get_model(
    'piper-en',
    lambda: load_piper_model('en_US-lessac-medium')
)
```

**Expected Impact**:
- Keeps memory usage <4GB consistently
- Prevents swap thrashing
- Small delay on first use (model load), then cached

---

## Part 4: Recommended Architecture

### 4.1 Target Architecture - "SAGE Real-Time Conversation Mode"

```
┌─────────────────────────────────────────────────────────────┐
│                    SAGE Unified Loop                         │
│  ┌────────────┐  ┌────────────┐  ┌─────────────┐            │
│  │  Sensor    │  │   SNARC    │  │   ATP       │            │
│  │   Hub      │──│  Salience  │──│ Allocator   │            │
│  └─────┬──────┘  └────────────┘  └──────┬──────┘            │
│        │                                 │                   │
│  ┌─────▼──────┐                    ┌────▼──────┐            │
│  │ Audio      │                    │  IRP      │            │
│  │ Streaming  │◄───────────────────┤  Plugins  │            │
│  │ Sensor     │    (if high ATP)   └────┬──────┘            │
│  └─────┬──────┘                         │                   │
│        │                                 │                   │
│        │ (GPU Mailbox)                   │ (GPU Mailbox)     │
│        ▼                                 ▼                   │
│  ┌─────────────────────────────────────────────┐            │
│  │        Cognitive Mailbox (PBM)              │            │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │            │
│  │  │ Transcr. │  │ Response │  │   TTS    │  │            │
│  │  │  Queue   │  │  Queue   │  │  Queue   │  │            │
│  │  └────┬─────┘  └─────▲────┘  └────┬─────┘  │            │
│  └───────┼──────────────┼────────────┼────────┘            │
│          │              │            │                      │
└──────────┼──────────────┼────────────┼──────────────────────┘
           │              │            │
           ▼              │            ▼
    ┌──────────────┐      │     ┌──────────────┐
    │  Cognitive   │      │     │  Streaming   │
    │  Response    │──────┘     │    TTS       │
    │  Engine      │            │  Effector    │
    │ (local LLM   │            │  (Piper)     │
    │  or pattern) │            └──────────────┘
    └──────────────┘

    TIMING (target):
    - Audio stream → VAD → Transcription: 300-500ms
    - Mailbox post → Cognitive read: <1ms
    - Cognitive response generation: 100-500ms
    - Mailbox post → TTS read: <1ms
    - TTS synthesis → Audio playback: 100-300ms

    TOTAL END-TO-END: <1500ms (vs current 7-45s)
```

### 4.2 Implementation Phases

**Phase 1: Audio Streaming (Week 1)**
- [ ] Install faster-whisper and silero-vad
- [ ] Implement StreamingAudioSensor with VAD
- [ ] Test latency and accuracy
- [ ] Integrate with SAGE sensor hub

**Phase 2: Memory-Based IPC (Week 1)**
- [ ] Adapt tiling_mailbox for CPU-only mode
- [ ] Implement CognitiveMailbox wrapper
- [ ] Test push/pop latency
- [ ] Replace file-based communication

**Phase 3: Cognitive Response (Week 2)**
- [ ] Install and test Piper TTS
- [ ] Implement pattern-based responses (interim)
- [ ] Test local LLM options (Qwen-0.5B, TinyLlama)
- [ ] Choose best trade-off for Jetson resources

**Phase 4: TTS Streaming (Week 2)**
- [ ] Implement StreamingTTSEffector with Piper
- [ ] Test synthesis latency
- [ ] Integrate with SAGE effector hub

**Phase 5: Integration & Optimization (Week 3)**
- [ ] Integrate all components with SAGE unified loop
- [ ] Implement ModelCache for memory management
- [ ] Add SNARC salience for conversation priority
- [ ] End-to-end testing and tuning

**Phase 6: Advanced Features (Week 4)**
- [ ] Interrupt handling (stop TTS if user speaks)
- [ ] Conversation context/memory
- [ ] Multi-turn dialogue coherence
- [ ] Performance profiling and optimization

---

## Part 5: Expected Performance

### 5.1 Latency Breakdown (Target)

| Component | Current | Target | Method |
|-----------|---------|---------|---------|
| Audio input | 3000-5000ms | 300-500ms | Streaming + VAD + faster-whisper |
| IPC (sensor→cog) | 0-100ms | <1ms | GPU mailbox |
| Cognition | 2000-30000ms | 100-500ms | Local LLM or patterns |
| IPC (cog→TTS) | 0-100ms | <1ms | GPU mailbox |
| TTS synthesis | 2000-5000ms | 100-300ms | Piper streaming |
| **TOTAL** | **7-45 seconds** | **<1.5 seconds** | **Combined** |

### 5.2 Resource Usage (Target)

| Component | RAM | CPU | Notes |
|-----------|-----|-----|-------|
| Faster-whisper tiny | 300MB | 60% peak | vs 400MB current |
| Silero VAD | 50MB | 5% continuous | New |
| Qwen-0.5B LLM | 300MB | 80% peak | vs manual response |
| Piper TTS | 50MB | 40% peak | vs 800MB NeuTTS |
| SAGE core | 200MB | 20% continuous | Existing |
| GPU mailbox | 10MB | <1% | Existing |
| System/desktop | 500MB | varies | Existing |
| **TOTAL** | **~1.4GB** | **varies** | **vs 6.8GB current** |

**Memory Optimization**:
- Dynamic loading: Only 2-3 models active at once
- Unload Whisper after transcription (free 300MB)
- Unload LLM after response (free 300MB)
- Keep only Piper + SAGE resident

### 5.3 Quality Trade-offs

| Aspect | Current | Target | Impact |
|--------|---------|---------|---------|
| ASR accuracy | 70-97% | 65-95% | Slightly lower (streaming) |
| Response quality | Manual (high) | Pattern/LLM (medium) | Acceptable for conversation |
| TTS voice quality | Very high (NeuTTS) | Good (Piper) | Acceptable trade-off |
| End-to-end latency | 7-45s | <1.5s | **30x improvement** |

---

## Part 6: Decision Points

### 6.1 Critical Decisions Needed

**Decision 1: Cognition Strategy**
- **Option A**: Pattern matching (fastest, limited)
- **Option B**: Local LLM Qwen-0.5B (balanced)
- **Option C**: Cloud API Claude Haiku (best quality, requires internet)
- **Recommendation**: Start with B, fallback to A, optional C for complex queries

**Decision 2: ASR Strategy**
- **Option A**: Faster-whisper + VAD (local, good quality)
- **Option B**: Cloud API (fastest, requires internet)
- **Recommendation**: Start with A, optional B for latency-critical applications

**Decision 3: TTS Strategy**
- **Option A**: Piper (fast, good quality, local)
- **Option B**: Cloud API (fastest, best quality, requires internet)
- **Recommendation**: Use A for edge deployment, optional B for demos

### 6.2 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| VAD false positives | Medium | Low | Tune threshold, add confidence filter |
| Memory exhaustion | High | High | Model cache with strict limits |
| Latency regression | Low | High | Continuous benchmarking, profiling |
| Quality degradation | Medium | Medium | A/B testing, user feedback |
| Integration complexity | Medium | Medium | Incremental rollout, extensive testing |

---

## Part 7: Success Metrics

### 7.1 Primary Metrics

1. **End-to-End Latency**: <2 seconds (currently 7-45s)
2. **Memory Usage**: <3GB peak (currently 6.8GB)
3. **Transcription Accuracy**: >85% (currently 70-97%)
4. **Response Relevance**: >80% user satisfaction
5. **System Stability**: <1% crash rate over 1 hour conversation

### 7.2 Secondary Metrics

1. **Interruption Handling**: <500ms to stop TTS when user speaks
2. **Context Retention**: Coherent over 5+ turns
3. **Resource Efficiency**: <50% average CPU utilization
4. **Audio Quality**: >3.5/5 subjective rating

---

## Conclusion

The current architecture has **proven the concept** of bidirectional conversation with Claude as cognitive layer. However, five critical bottlenecks prevent real-time operation:

1. **Audio chunking** (3s fixed delay)
2. **Cognition latency** (manual response)
3. **TTS synthesis** (heavy model)
4. **Memory pressure** (swap thrashing)
5. **File-based IPC** (polling overhead)

**The path forward** is clear:
- Streaming audio with VAD
- Memory-based IPC (existing GPU mailboxes)
- Local LLM or pattern-based cognition
- Lightweight Piper TTS
- Dynamic model loading

**Target**: <2s end-to-end latency, comparable to phone apps, while maintaining edge-first philosophy.

**Next Step**: Choose cognition strategy and begin Phase 1 implementation (audio streaming).

---

*Analysis complete. Ready to proceed with systematic implementation.*
