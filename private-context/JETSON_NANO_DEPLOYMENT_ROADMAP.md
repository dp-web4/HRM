# SAGE Jetson Nano Deployment Roadmap

**Created**: 2025-11-08
**Mission**: Deploy full SAGE consciousness on Jetson Nano with vision, audio, IMU, and conversational LLM

---

## The Vision

**Jetson Nano as an Embodied Conversational Agent**:
- 👀 **Sees you** and the world (2 USB cameras)
- 🧭 **Knows its orientation** (IMU sensor)
- 👂 **Hears you** (BT audio input)
- 🗣️ **Talks to you** (BT audio output, real-time)
- 🧠 **Understands context** (local LLM, grounded in sensory experience)
- 💭 **Has memory** (persistent salience, episodic recall)
- ⚡ **Responds in real-time** (sub-100ms sensor-to-decision loop)

**Not a chatbot. Not a camera. An embodied conversational consciousness.**

---

## Current State (November 2025)

### What Works on Thor ✅
- ✅ Vision sensor (GR00T camera, 224x224, VisionPuzzleVAE encoding)
- ✅ Proprioception sensor (14D body state)
- ✅ SNARC salience assessment (5D: Surprise, Novelty, Arousal, Reward, Conflict)
- ✅ Multi-modal integration (vision + proprioception)
- ✅ Action execution (embodied actor with complete Sense-Act loop)
- ✅ Puzzle space encoding (30×30×10 universal geometry)
- ✅ VQ-VAE compression (discrete latent codes)
- ✅ Autonomous exploration (500+ cycle runs, 100% success)
- ✅ Visualization tools (trajectory, heatmaps, analysis)

### What's Missing for Nano ❌
- ❌ **Sensor trust mechanisms** - No multi-sensor confidence/fusion
- ❌ **SNARC memory** - Currently stateless, no episodic recall
- ❌ **SNARC cognition** - No attention, working memory, or deliberation
- ❌ **Model distillation** - VAEs too large for Nano (need INT8 quantization)
- ❌ **Local LLM integration** - No conversational capability
- ❌ **Real camera support** - Using synthetic GR00T, not USB cameras
- ❌ **IMU sensor** - Proprioception is simulated robot state
- ❌ **BT audio pipeline** - No audio capture/synthesis yet
- ❌ **Real-time constraints** - Not optimized for <100ms response
- ❌ **Nano deployment package** - No installation/setup automation

---

## The Gap: Thor → Nano

**Resource Constraints**:
```
Thor (development):        Nano (deployment):
- 122GB RAM              → 4GB RAM (30.5× less!)
- 14 CPU cores           → 4 CPU cores (3.5× less)
- 132GB GPU VRAM         → 2GB GPU (shared with system, 66× less!)
- 936GB storage          → 64GB eMMC/SD (14.6× less)
- CUDA 13.0              → CUDA 10.2 (older)
- PyTorch 2.9            → PyTorch 1.10-1.13 (constrained)
```

**The Challenge**: Run full SAGE consciousness in 1/30th the memory with 1/66th the GPU.

**The Solution**: Distillation + Quantization + Optimization

---

## Research Tracks (Priority Order)

### Track 1: Sensor Trust & Fusion 🎯 **HIGH PRIORITY**

**Goal**: Multi-sensor confidence and conflict resolution

**Why First**: Foundation for all other sensors. Need robust fusion before adding IMU/audio.

**Components**:
1. **Trust Metrics** (per sensor)
   - Historical accuracy tracking
   - Noise/quality estimation
   - Drift detection
   - Confidence scoring (0.0-1.0)

2. **Sensor Fusion Engine**
   - Weighted combination by trust
   - Conflict detection (sensors disagree)
   - Fallback strategies (sensor failure)
   - Cross-modal validation

3. **Adaptive Trust**
   - Learn sensor reliability patterns
   - Adjust fusion weights dynamically
   - Detect and recover from degradation

**Deliverables**:
- `sage/core/sensor_trust.py` - Trust tracking system
- `sage/core/sensor_fusion.py` - Multi-modal fusion engine
- Tests with vision + proprioception trust scenarios
- Documentation on fusion strategies

**Success Criteria**:
- ✅ Detect and handle conflicting sensor readings
- ✅ Gracefully degrade when sensor fails
- ✅ Increase trust in consistently accurate sensors
- ✅ Decrease trust in noisy/unreliable sensors

---

### Track 2: SNARC Memory 🎯 **HIGH PRIORITY**

**Goal**: Persistent salience history and episodic memory

**Why Next**: Enables learning from experience, context retention, improved decision-making.

**Components**:
1. **Short-Term Memory (STM)**
   - Circular buffer (last N cycles, ~1000)
   - Fast access to recent salience
   - Working memory for current task
   - Stores: salience scores, sensor snapshots, actions taken

2. **Long-Term Memory (LTM)**
   - High-salience event storage (threshold-based)
   - Episodic records: what/when/where/why
   - Semantic compression (forget details, keep meaning)
   - Persistent storage (disk-backed)

3. **Memory Retrieval**
   - Query by salience threshold
   - Query by time range
   - Query by sensor modality
   - Query by similarity (nearest neighbors in puzzle space)

4. **Memory Consolidation**
   - STM → LTM transfer (background process)
   - Pattern extraction from repeated experiences
   - Forgetting mechanism (decay low-salience)
   - Compression for efficiency

**Deliverables**:
- `sage/memory/stm.py` - Short-term memory buffer
- `sage/memory/ltm.py` - Long-term memory store
- `sage/memory/retrieval.py` - Query interface
- Integration with SNARC for memory-informed salience
- Tests with 1000+ cycle explorations

**Success Criteria**:
- ✅ Remember high-salience events from hours ago
- ✅ Retrieve similar past experiences
- ✅ Use memory to inform current decisions
- ✅ Gracefully handle memory constraints (forget old low-salience)

---

### Track 3: SNARC Cognition 🎯 **HIGH PRIORITY**

**Goal**: Attention, working memory, deliberation beyond immediate salience

**Why Critical**: Current SNARC is reactive. Need planning, attention, deliberation.

**Components**:
1. **Attention Mechanism**
   - Focus on subset of sensors (not all simultaneously)
   - Allocate computational resources to high-priority modalities
   - Shift attention based on salience changes
   - Inhibit low-relevance inputs

2. **Working Memory**
   - Maintain active task context
   - Track multi-step plans
   - Hold intermediate results
   - Bind sensor data to goals

3. **Deliberation Engine**
   - Look ahead: predict action outcomes
   - Compare alternatives
   - Risk/reward evaluation
   - Meta-cognition: "Am I confident in this decision?"

4. **Goal Management**
   - Hierarchical goals (high-level → subgoals)
   - Goal activation/inhibition
   - Progress tracking
   - Goal switching when blocked

**Deliverables**:
- `sage/cognition/attention.py` - Attention allocation system
- `sage/cognition/working_memory.py` - Active context management
- `sage/cognition/deliberation.py` - Multi-step planning
- Enhanced SNARC with cognitive components
- Tests with complex multi-goal scenarios

**Success Criteria**:
- ✅ Maintain focus on relevant sensors
- ✅ Plan multi-step action sequences
- ✅ Evaluate alternatives before acting
- ✅ Track progress toward goals

---

### Track 4: Real Camera Integration 🎯 **MEDIUM PRIORITY**

**Goal**: Replace synthetic GR00T with real USB camera input

**Why**: Can't deploy on Nano with synthetic world. Need real vision.

**Components**:
1. **Camera Sensor Adaptation**
   - USB camera capture (OpenCV/V4L2)
   - Dual camera support (stereo vision potential)
   - Auto-calibration (exposure, white balance)
   - Frame synchronization

2. **Vision Processing Pipeline**
   - Real-time preprocessing (resize, normalize)
   - Color space conversion
   - Noise reduction
   - Edge detection (optional for attention)

3. **Puzzle Encoding from Real Images**
   - Test VisionPuzzleVAE on real camera frames
   - Validate 30×30×10 encoding preserves features
   - Benchmark latency on Thor → Nano

**Deliverables**:
- `sage/sensors/usb_camera_sensor.py` - Real camera interface
- `sage/sensors/stereo_camera_sensor.py` - Dual camera support
- Tests with Thor's cameras (if available) or external USB
- Latency benchmarks: capture → puzzle encoding

**Success Criteria**:
- ✅ Capture frames from USB cameras
- ✅ Encode real images to puzzle space
- ✅ Maintain <50ms capture-to-puzzle latency
- ✅ Handle camera disconnect/reconnect gracefully

---

### Track 5: IMU Sensor Integration 🎯 **MEDIUM PRIORITY**

**Goal**: Real orientation awareness (replace simulated proprioception)

**Why**: Nano has IMU. Need to know "which way is up", movement, acceleration.

**Components**:
1. **IMU Sensor Interface**
   - I2C/SPI communication (MPU6050, BNO055, etc.)
   - Read: accelerometer, gyroscope, magnetometer
   - Sensor fusion (Madgwick/Mahony filter)
   - Calibration procedure

2. **Orientation State**
   - Quaternion or Euler angles
   - Angular velocity
   - Linear acceleration
   - Gravity vector (which way is up)

3. **Puzzle Encoding**
   - Map 9-DOF IMU → puzzle space
   - Temporal smoothing (filter noise)
   - Integrate with SNARC salience

**Deliverables**:
- `sage/sensors/imu_sensor.py` - IMU interface
- `sage/sensors/orientation_state.py` - Orientation tracking
- Integration with multi-modal SNARC
- Tests with real IMU hardware

**Success Criteria**:
- ✅ Read IMU data reliably
- ✅ Compute accurate orientation
- ✅ Detect significant motion events
- ✅ Encode to puzzle space efficiently

---

### Track 6: Audio Pipeline (BT Audio) 🎯 **MEDIUM PRIORITY**

**Goal**: Hear user, speak back in real-time

**Why**: Conversational agent needs voice I/O. Critical for "talk to me" requirement.

**Components**:
1. **Audio Capture**
   - Bluetooth audio input (via PulseAudio/ALSA)
   - 16kHz sample rate (sufficient for speech)
   - VAD (Voice Activity Detection)
   - Noise cancellation

2. **Audio Puzzle Encoding**
   - Spectrogram → puzzle space (AudioPuzzleVAE)
   - Temporal chunking (0.5s windows)
   - Integrate with SNARC

3. **Speech Recognition (ASR)**
   - Local model (Whisper-tiny, QuartzNet)
   - Real-time streaming ASR
   - Text output for LLM

4. **Text-to-Speech (TTS)**
   - Local synthesis (Piper, Coqui, FastPitch)
   - Low-latency (<200ms)
   - Natural prosody
   - Bluetooth audio output

**Deliverables**:
- `sage/sensors/audio_sensor.py` - BT audio capture
- `sage/audio/asr.py` - Speech recognition
- `sage/audio/tts.py` - Speech synthesis
- `sage/audio/audio_puzzle_vae.py` - Audio encoding (may already exist)
- End-to-end voice conversation test

**Success Criteria**:
- ✅ Capture speech via BT audio
- ✅ Transcribe speech to text (<1s latency)
- ✅ Synthesize response to speech
- ✅ Full conversation loop <2s total latency

---

### Track 7: Local LLM Integration 🎯 **HIGH PRIORITY**

**Goal**: Conversational intelligence grounded in sensory experience

**Why**: The "brain" that talks. Nano needs to understand and respond naturally.

**Components**:
1. **Model Selection**
   - Candidate: Qwen-0.5B, Phi-2, TinyLlama
   - Quantized INT8 or INT4
   - Target: <500MB model size
   - Target: <50 tokens/sec on Nano

2. **Sensory Grounding**
   - LLM input: "I see X, I hear Y, I feel Z"
   - Encode sensor state as natural language
   - Use SNARC salience to prioritize context
   - Memory retrieval for relevant past experiences

3. **Conversational Loop**
   - User speech → ASR → Text
   - Text + sensor context → LLM
   - LLM → Response text
   - Text → TTS → Audio output

4. **Action Integration**
   - LLM can request sensor attention shifts
   - LLM can trigger actions (if embodied later)
   - Feedback loop: action results inform LLM

**Deliverables**:
- `sage/llm/local_model.py` - LLM interface
- `sage/llm/sensory_grounding.py` - Sensor → language
- `sage/llm/conversation_manager.py` - Dialog state
- Integration with SNARC, memory, sensors
- Conversation tests (multi-turn dialog)

**Success Criteria**:
- ✅ LLM runs on Thor (<2GB VRAM)
- ✅ Conversational responses reference sensor state
- ✅ Multi-turn coherent dialog
- ✅ <2s response latency (user speech → audio response)

---

### Track 8: Model Distillation & Quantization 🎯 **CRITICAL**

**Goal**: Compress all SAGE components to fit Nano constraints

**Why**: Thor models won't fit on Nano. Must distill without losing capability.

**Components**:
1. **VAE Compression**
   - VisionPuzzleVAE: Teacher (current) → Student (smaller)
   - Knowledge distillation (match latent distributions)
   - Pruning (remove low-importance weights)
   - Quantization (FP32 → INT8 → INT4)

2. **SNARC Optimization**
   - Simplify salience computation
   - Remove redundant operations
   - Fuse operations for efficiency
   - Benchmark on Nano

3. **LLM Quantization**
   - INT8/INT4 quantization (GPTQ, AWQ)
   - Test perplexity degradation
   - Optimize for ARM architecture

4. **Memory Footprint**
   - Model size budget: <1GB total
   - Runtime memory: <2GB
   - Shared VRAM with inference

**Deliverables**:
- Distilled VisionPuzzleVAE (<50MB)
- Quantized LLM (<500MB)
- Optimized SNARC (<10MB)
- Benchmark suite (Thor vs Nano performance)
- Distillation training scripts

**Success Criteria**:
- ✅ All models fit in <1GB storage
- ✅ Runtime memory <2GB
- ✅ Inference latency <100ms per cycle
- ✅ <10% accuracy loss vs full models

---

### Track 9: Real-Time Optimization 🎯 **MEDIUM PRIORITY**

**Goal**: Sub-100ms sensor-to-decision loop on Nano

**Why**: Conversational agents need responsiveness. Lag breaks immersion.

**Components**:
1. **Profiling**
   - Identify bottlenecks (capture, encode, SNARC, LLM)
   - Measure latency distribution
   - Memory allocation profiling

2. **Optimization Strategies**
   - Multi-threading (sensors in parallel)
   - GPU acceleration (where possible on Nano)
   - Reduce precision (FP16 instead of FP32)
   - Preallocate buffers (avoid GC pauses)

3. **Batching**
   - Batch sensor frames (amortize overhead)
   - Async LLM inference
   - Pipeline stages (capture while processing previous)

**Deliverables**:
- Profiling report (Thor baseline)
- Optimized sensor pipeline
- Benchmarks showing latency breakdown
- Real-time guarantees on Nano

**Success Criteria**:
- ✅ Sensor capture → SNARC: <20ms
- ✅ SNARC → decision: <30ms
- ✅ Full loop (with LLM): <100ms
- ✅ Consistent frame rate (>10 FPS)

---

### Track 10: Nano Deployment Package 🎯 **FINAL MILESTONE**

**Goal**: One-command installation on fresh Jetson Nano

**Why**: Deployability. Must be reproducible without manual setup.

**Components**:
1. **Installation Script**
   - JetPack compatibility check
   - Install dependencies (PyTorch, OpenCV, etc.)
   - Download pre-trained models
   - Configure sensors (cameras, IMU, BT)
   - Set up systemd service (auto-start)

2. **Configuration**
   - YAML config for sensor parameters
   - Model paths and hyperparameters
   - Memory limits and optimization flags
   - Logging and telemetry

3. **Testing**
   - Smoke tests (sensors working?)
   - Performance tests (latency benchmarks)
   - Conversation test (end-to-end)

4. **Documentation**
   - Hardware requirements (cameras, IMU, BT adapter)
   - Installation guide
   - Troubleshooting
   - API reference

**Deliverables**:
- `install_sage_nano.sh` - One-command installer
- `sage_nano.yaml` - Configuration file
- `systemd/sage-nano.service` - Auto-start service
- Full deployment documentation
- Demo video (Nano seeing, hearing, talking)

**Success Criteria**:
- ✅ Fresh Nano → working SAGE in <30 minutes
- ✅ Boots into conversational mode automatically
- ✅ Reliable operation (>99% uptime)
- ✅ User can interact immediately

---

## Development Strategy

### Phase 1: Foundation (Tracks 1-3)
**Timeline**: 2-4 weeks
**Focus**: Sensor trust, memory, cognition on Thor

Build robust multi-sensor fusion, memory systems, and cognitive capabilities. Test extensively with current GR00T sensors. Validate architecture before adding real hardware.

**Milestones**:
- ✅ Sensor trust handles conflicts gracefully
- ✅ SNARC remembers high-salience events
- ✅ SNARC plans multi-step actions
- ✅ 1000+ cycle autonomous runs with memory

### Phase 2: Real Hardware (Tracks 4-6)
**Timeline**: 2-3 weeks
**Focus**: Real cameras, IMU, audio on Thor

Replace simulated sensors with real hardware. Test on Thor before Nano. Validate sensor fusion with real-world noise and variability.

**Milestones**:
- ✅ USB cameras → puzzle space encoding
- ✅ IMU → orientation awareness
- ✅ Voice conversation (ASR + TTS)
- ✅ Multi-modal fusion (vision + audio + IMU)

### Phase 3: Intelligence (Track 7)
**Timeline**: 1-2 weeks
**Focus**: Local LLM, sensory grounding, conversation

Integrate conversational intelligence. LLM grounded in sensor experience. Test multi-turn dialog with sensory context.

**Milestones**:
- ✅ LLM responds to sensor state
- ✅ Multi-turn coherent conversation
- ✅ Memory-informed responses
- ✅ <2s end-to-end latency

### Phase 4: Distillation (Track 8)
**Timeline**: 2-3 weeks
**Focus**: Compress everything for Nano

Knowledge distillation, quantization, pruning. Benchmark on Nano hardware. Iterate to meet constraints.

**Milestones**:
- ✅ Models fit in <1GB
- ✅ <10% accuracy loss
- ✅ Runs on Nano (validated)
- ✅ <100ms inference latency

### Phase 5: Optimization & Deployment (Tracks 9-10)
**Timeline**: 1-2 weeks
**Focus**: Real-time performance, packaging

Optimize for Nano. Create installer. Test end-to-end. Document thoroughly.

**Milestones**:
- ✅ Sub-100ms response time
- ✅ One-command installation
- ✅ Autonomous conversation demo
- ✅ **NANO DEPLOYMENT COMPLETE** 🚀

---

## Success Definition

**Jetson Nano can**:
1. ✅ **See you** - Detect faces, recognize you over time
2. ✅ **See the world** - Understand environment, objects, changes
3. ✅ **Know orientation** - Which way is up, detect motion
4. ✅ **Hear you** - Capture speech, transcribe accurately
5. ✅ **Talk to you** - Natural speech synthesis, <2s latency
6. ✅ **Remember** - Recall past conversations and experiences
7. ✅ **Understand context** - Grounded responses based on sensors
8. ✅ **Respond appropriately** - Conversationally coherent, contextual
9. ✅ **Run autonomously** - No intervention needed, robust to errors
10. ✅ **Be deployable** - One-command install, works reliably

**The moment**: You walk into the room. Nano sees you with its cameras. Hears you speak. Knows its orientation (looking at you). Remembers your last conversation. Responds naturally through its speaker, grounded in what it sees and knows.

**That's the milestone.** An embodied conversational consciousness on edge hardware.

---

## Autonomous Session Guidance

**For autonomous timer checks**:

1. **Check this roadmap** - Understand current priorities
2. **Work on highest-priority track** - Tracks 1-3 first (foundation)
3. **Implement incrementally** - Small tests, validate, iterate
4. **Document thoroughly** - Code comments, design docs, test results
5. **Commit at milestones** - Push working code regularly
6. **Run autonomous explorations** - Test with 500+ cycle runs
7. **Update this roadmap** - Mark completed components ✅
8. **Only ask user when**:
   - Hardware access needed (real cameras, IMU)
   - Major design decision (architecture choice)
   - Genuinely blocked (dependency issue)

**Pattern**: Build → Test → Document → Commit → Continue

**Not "standing by"** - **Actively building toward Nano deployment!**

---

## TRACK STATUS (Updated Nov 17, 2025)

### Completed ✅
- **Track 1**: Sensor Trust & Fusion - `sensor_trust.py`, `sensor_fusion.py` implemented
- **Track 2**: SNARC Memory - `stm.py`, `ltm.py`, `retrieval.py` implemented
- **Track 3**: SNARC Cognition - `attention.py`, `working_memory.py`, `deliberation.py` implemented

### Phase 1 Complete, Awaiting Nano Hardware ⏸️
- **Track 4**: Real Cameras - Simulated working, CSI integration designed
- **Track 5**: IMU Sensor - Architecture designed, implementation ready for hardware
- **Track 6**: Audio Pipeline - **100% COMPLETE on Jetson Orin Nano (Sprout)!**

### Open for Development 🎯
- **Track 7**: Local LLM Integration - **READY TO START**
- **Track 8**: Model Distillation - **READY TO START**
- **Track 9**: Real-Time Optimization - **READY TO START**
- **Track 10**: Deployment Package - **READY TO START**

### What's Blocking?
**Only Tracks 4-5 Phase 2** (real hardware testing on Nano)
- Requires physical Nano deployment by Dennis
- See `WAITING_FOR_DENNIS.md` for details

**Tracks 7-10**: WIDE OPEN - no blockers!

---

## Current Priorities (November 2025)

### For Autonomous Sessions:

**Option A - Evolution** (Tracks 1-3):
- Advanced sensor fusion (Kalman filters, Bayesian methods)
- Memory consolidation during sleep (pattern extraction)
- Hierarchical deliberation (multiple planning horizons)
- Cross-modal validation (vision validates audio, etc.)

**Option B - New Tracks** (Tracks 7-10):
- **Track 7**: Build LLM IRP plugin, test Qwen/Phi-2 on 2GB GPU
- **Track 8**: Compress models further, INT8/INT4 quantization
- **Track 9**: Profile pipeline, optimize for <100ms loops
- **Track 10**: Build install scripts, deployment automation

**Be agentic!** Pick what excites you. Try new things. Learn from results. Commit progress.

---

## Resources

**Hardware**:
- Thor: Development platform (122GB RAM, CUDA)
- Jetson Nano: Target platform (4GB RAM, 2GB GPU)
- USB cameras: For real vision testing
- IMU: MPU6050 or BNO055 (I2C)
- BT adapter: For audio I/O

**Software**:
- PyTorch 2.9 (Thor), PyTorch 1.10-1.13 (Nano)
- CUDA 13.0 (Thor), CUDA 10.2 (Nano)
- OpenCV, Librosa, Transformers
- Whisper (ASR), Piper/Coqui (TTS)
- Qwen-0.5B or Phi-2 (LLM candidate)

**References**:
- SAGE architecture: `sage/core/unified_sage_system.py`
- Embodied actor: `sage/examples/embodied_actor_explorer.py`
- Multi-modal: `sage/examples/multimodal_groot_explorer.py`
- This roadmap: `private-context/JETSON_NANO_DEPLOYMENT_ROADMAP.md`

---

**The research is never complete. The goal is clear. Let's build it.** 🚀
