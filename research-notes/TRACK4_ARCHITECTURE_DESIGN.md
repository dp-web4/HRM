# Track 4 Architecture Design
## Real Camera Integration for SAGE Consciousness

**Track:** 4 of 10
**Status:** Architecture Design (Implementation pending user approval)
**Date:** 2025-11-10
**Session:** #23 (Autonomous)

---

## Overview

Track 4 integrates a physical camera into the SAGE consciousness system, moving from simulated/synthetic vision to real-world visual input. This track transforms SAGE from a laboratory system into a deployed perceptual agent.

**Core Objective:** Enable SAGE to perceive the physical world through a camera in real-time, with full integration into the cognitive architecture (Tracks 1-3).

---

## Design Principles

1. **Hardware Abstraction**: Camera interface should work with USB webcams, CSI cameras (for Nano), and simulated cameras
2. **Real-Time Performance**: Target <100ms latency from photons to salience
3. **Graceful Degradation**: System continues functioning if camera unavailable
4. **Nano-First**: Design for Jetson Nano constraints (memory, compute, bandwidth)
5. **Integration-Ready**: Seamless connection with existing Tracks 1-3

---

## Architecture Components

### Component 1: Camera Driver Layer

**Purpose:** Abstract camera hardware into a uniform interface

**Interface:**
```python
class CameraDriver(ABC):
    """Abstract camera interface"""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize camera hardware"""
        pass

    @abstractmethod
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture single frame (H, W, 3) RGB"""
        pass

    @abstractmethod
    def get_properties(self) -> CameraProperties:
        """Get camera capabilities (resolution, fps, etc.)"""
        pass

    @abstractmethod
    def release(self):
        """Release camera resources"""
        pass
```

**Implementations:**

1. **USB Webcam Driver** (OpenCV)
   - Standard USB cameras (UVC protocol)
   - Works on any platform
   - Latency: ~30-50ms

2. **CSI Camera Driver** (Nano-specific)
   - Direct CSI camera interface
   - Lower latency: ~20-30ms
   - Requires Jetson libraries

3. **Simulated Camera Driver**
   - Replays images from dataset
   - For testing without hardware
   - Deterministic, reproducible

**Hardware Detection:**
```python
def detect_available_cameras() -> List[CameraInfo]:
    """
    Auto-detect available cameras
    Priority: CSI > USB > Simulated
    """
    cameras = []

    # Check CSI cameras (Jetson)
    if is_jetson():
        cameras.extend(detect_csi_cameras())

    # Check USB cameras (all platforms)
    cameras.extend(detect_usb_cameras())

    # Always available: simulated
    cameras.append(SimulatedCamera())

    return cameras
```

**Performance Targets:**
- Frame capture: <50ms (USB), <30ms (CSI)
- Resolution: 640x480 minimum, 1920x1080 maximum
- Frame rate: 30 FPS target, 15 FPS minimum
- Startup time: <2 seconds

---

### Component 2: Vision Pipeline

**Purpose:** Process raw camera frames into SAGE-compatible observations

**Pipeline Stages:**

```
Camera → [1] Capture → [2] Preprocessing → [3] SNARC Encoding → [4] Salience
```

**Stage 1: Frame Capture**
- Retrieve frame from camera driver
- Timestamp stamping (for synchronization)
- Buffer management (ring buffer for latest N frames)

**Stage 2: Preprocessing**
- Resize to network input size (configurable, e.g., 224x224)
- Normalize pixel values (0-1 or standardize)
- Color space conversion if needed (RGB <-> BGR)
- Optional: Basic filtering (denoise, contrast enhancement)

**Stage 3: SNARC Encoding**
- Feed through vision encoder (from Track 2 SNARC)
- Generate salience map (novelty + attention)
- Extract features for memory storage

**Stage 4: Salience Computation**
- Compute per-region salience scores
- Aggregate to sensor-level salience
- Output: `{'vision': salience_score}`

**Code Structure:**
```python
class VisionPipeline:
    """
    Real-time vision processing pipeline

    Integrates:
    - Camera driver (Track 4)
    - SNARC salience (Track 2)
    - Sensor trust (Track 1)
    - Attention allocation (Track 3)
    """

    def __init__(
        self,
        camera: CameraDriver,
        snarc_encoder: SalienceEncoder,  # From Track 2
        sensor_trust: SensorTrust,  # From Track 1
        attention_manager: AttentionManager  # From Track 3
    ):
        self.camera = camera
        self.snarc = snarc_encoder
        self.trust = sensor_trust
        self.attention = attention_manager

        # Performance tracking
        self.frame_buffer = deque(maxlen=30)  # 1 second at 30 FPS
        self.latency_stats = LatencyStats()

    def process_frame(self) -> VisionObservation:
        """
        Process single frame through complete pipeline

        Returns:
            VisionObservation with frame, salience, features, metadata
        """
        # [1] Capture
        start = time.time()
        frame = self.camera.capture_frame()
        if frame is None:
            return None
        capture_latency = time.time() - start

        # [2] Preprocess
        start = time.time()
        processed = self.preprocess(frame)
        preprocess_latency = time.time() - start

        # [3] SNARC encoding
        start = time.time()
        salience, features = self.snarc.encode(processed)
        encoding_latency = time.time() - start

        # [4] Create observation
        observation = VisionObservation(
            frame=frame,
            timestamp=time.time(),
            salience=salience,
            features=features,
            latency={
                'capture': capture_latency,
                'preprocess': preprocess_latency,
                'encoding': encoding_latency,
                'total': capture_latency + preprocess_latency + encoding_latency
            }
        )

        # Update trust based on consistency
        self.update_vision_trust(observation)

        return observation

    def update_vision_trust(self, observation: VisionObservation):
        """
        Update vision sensor trust score

        Criteria:
        - Latency stability (low variance = high trust)
        - Frame quality (no corruption = high trust)
        - Consistency with memory (matches expectations = high trust)
        """
        # Latency stability
        latency_variance = np.var([obs.latency['total'] for obs in self.frame_buffer])
        latency_trust = 1.0 / (1.0 + latency_variance * 100)

        # Frame quality (check for corruption)
        quality_trust = self.check_frame_quality(observation.frame)

        # Memory consistency (does scene match expectations?)
        memory_consistency = self.snarc.check_consistency(observation.features)

        # Aggregate trust
        trust_score = (latency_trust * 0.3 +
                      quality_trust * 0.4 +
                      memory_consistency * 0.3)

        self.trust.update('vision', trust_score)
```

**Performance Targets:**
- Total latency: <100ms (capture + preprocess + encoding)
- Frame processing rate: ≥30 FPS
- Memory: <500MB for buffers and models
- GPU usage: <50% (leave headroom for other modalities)

---

### Component 3: Integration with Existing Tracks

**Track 1 (Sensor Trust) Integration:**

```python
# Vision sensor registration
sensor_trust.register_sensor(
    sensor_id='vision',
    sensor_type='camera',
    initial_trust=0.5,  # Neutral start
    decay_rate=0.01
)

# Trust updates based on:
# 1. Latency consistency
# 2. Frame quality
# 3. Memory coherence
```

**Track 2 (SNARC Memory) Integration:**

```python
# Store visual observations in memory
memory_system.store_observation(
    modality='vision',
    observation=vision_obs,
    salience=vision_obs.salience,
    features=vision_obs.features,
    context={'camera': camera.name, 'resolution': resolution}
)

# Retrieve similar past observations
similar_scenes = memory_system.retrieve(
    query=current_features,
    modality='vision',
    top_k=5
)
```

**Track 3 (SNARC Cognition) Integration:**

```python
# Vision participates in attention allocation
attention_manager.allocate_attention(
    current_salience={'vision': vision_salience, 'audio': audio_salience, ...},
    active_goals=current_goals,
    sensor_trust={'vision': vision_trust, 'audio': audio_trust, ...}
)

# High-salience visual events trigger goal switches
if vision_salience > 0.9:  # Interrupt threshold
    goal_manager.switch_goal(
        from_goal=current_goal,
        to_goal='investigate_visual_event',
        reason='high_salience_interrupt'
    )

# Deliberation uses visual context
deliberation_result = deliberation_engine.deliberate(
    situation={'visual_scene': scene_description, ...},
    available_actions=actions,
    goal=current_goal
)
```

---

### Component 4: Camera Configuration Manager

**Purpose:** Handle camera settings, calibration, and adaptation

**Features:**

1. **Auto-Configuration**
   - Detect optimal resolution for available compute
   - Adjust frame rate based on latency targets
   - Enable/disable features (auto-focus, auto-exposure)

2. **Calibration**
   - Intrinsic parameters (focal length, distortion)
   - Extrinsic parameters (camera pose in robot frame)
   - Color calibration (white balance, gamma)

3. **Adaptive Settings**
   - Lower resolution if latency exceeds target
   - Adjust exposure for lighting conditions
   - Enable motion blur reduction if movement detected

**Configuration Schema:**
```python
@dataclass
class CameraConfig:
    """Camera configuration"""
    # Resolution
    width: int = 640
    height: int = 480

    # Frame rate
    fps: int = 30

    # Processing
    resize_to: Tuple[int, int] = (224, 224)  # For SNARC encoder
    color_space: str = 'RGB'
    normalize: bool = True

    # Quality
    auto_exposure: bool = True
    auto_focus: bool = True
    auto_white_balance: bool = True

    # Performance
    buffer_size: int = 30  # Frames to buffer
    max_latency_ms: float = 100.0  # Target latency

    # Calibration
    intrinsics: Optional[CameraIntrinsics] = None
    extrinsics: Optional[CameraExtrinsics] = None
```

---

## Data Flow

### Real-Time Perception Loop

```
┌─────────────────────────────────────────────────────────────┐
│                     SAGE Consciousness                       │
│                                                              │
│  ┌──────────┐    ┌───────────┐    ┌──────────────┐        │
│  │  Camera  │───→│  Vision   │───→│   SNARC      │        │
│  │  Driver  │    │  Pipeline │    │   Salience   │        │
│  └──────────┘    └───────────┘    └──────────────┘        │
│                                            │                 │
│                                            ↓                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Cognitive Loop (Track 3)                 │  │
│  │                                                        │  │
│  │  Attention ──→ Deliberation ──→ Working Memory       │  │
│  │     ↑              ↓                                   │  │
│  │     └───── Goal Manager ←──────┘                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                            │                 │
│                                            ↓                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               Memory (Track 2)                        │  │
│  │                                                        │  │
│  │  STM ←→ Consolidation ←→ LTM                         │  │
│  │           ↓                                            │  │
│  │       Retrieval                                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                            │                 │
│                                            ↓                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Sensor Trust (Track 1)                     │  │
│  │                                                        │  │
│  │  Vision Trust ←── Performance + Consistency           │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Frame Timing (Target: 30 FPS, 33ms per frame)

```
Frame N:
├─ [0-30ms]   Camera capture
├─ [30-40ms]  Preprocessing
├─ [40-60ms]  SNARC encoding
├─ [60-65ms]  Salience computation
├─ [65-75ms]  Attention allocation
├─ [75-85ms]  Deliberation (if needed)
├─ [85-90ms]  Memory operations
└─ [90-95ms]  Trust update

Total: ~95ms (within 100ms budget, allows 30 FPS)
```

---

## Testing Strategy

### Phase 1: Unit Tests (No Hardware Required)

**Test Components:**
1. Simulated camera driver
2. Vision pipeline stages
3. Integration interfaces
4. Configuration management

**Test Scenarios:**
- Simulated frame processing
- Latency measurement
- Error handling (missing camera, corrupted frame)
- Configuration adaptation

### Phase 2: Hardware Validation (USB Webcam)

**Test Components:**
1. USB camera driver
2. Real-time frame capture
3. End-to-end latency
4. Trust score updates

**Test Scenarios:**
- Static scene (measure baseline latency)
- Moving objects (track salience changes)
- Lighting changes (verify adaptation)
- Occlusion/blur (test robustness)

### Phase 3: Integration Tests (Full System)

**Test Components:**
1. Vision + Memory integration
2. Vision + Cognition integration
3. Vision + Sensor Trust integration
4. Multi-modal (Vision + Audio when available)

**Test Scenarios:**
- Goal-driven attention (goals prioritize visual regions)
- Memory-guided perception (familiar scenes recognized)
- Trust-weighted fusion (low-trust vision down-weighted)
- Interrupt handling (high-salience visual events)

### Phase 4: Nano Deployment (Target Hardware)

**Test Components:**
1. CSI camera driver
2. Nano-specific optimizations
3. Resource constraints
4. Real-world conditions

**Test Scenarios:**
- Continuous operation (24+ hours)
- Resource monitoring (memory, GPU, thermals)
- Degradation handling (throttling, OOM)
- Edge cases (camera disconnect, poor lighting)

---

## Performance Budgets (Jetson Nano)

### Latency Budget (100ms total)

| Component | Target | Max |
|-----------|--------|-----|
| Camera capture | 30ms | 50ms |
| Preprocessing | 10ms | 20ms |
| SNARC encoding | 40ms | 60ms |
| Salience | 10ms | 15ms |
| Attention | 5ms | 10ms |
| Overhead | 5ms | 10ms |
| **Total** | **100ms** | **165ms** |

### Memory Budget (4GB RAM, 2GB GPU)

| Component | Target | Max |
|-----------|--------|-----|
| Frame buffers | 100MB | 200MB |
| SNARC models | 200MB | 300MB |
| Processing buffers | 100MB | 200MB |
| Other (Tracks 1-3) | 100MB | 200MB |
| **Total** | **500MB** | **900MB** |

Reserve: 3.1GB RAM, 1.1GB GPU (for other modalities + system)

### Compute Budget (GPU)

- Vision encoding: <50% GPU (allow other modalities)
- Preprocessing: CPU (offload from GPU)
- Target: 30 FPS sustained, 15 FPS minimum

---

## Hardware Requirements

### Minimum (Development)

**Platform:** Any Linux system with USB port
**Camera:** USB webcam (UVC compatible)
**Resolution:** 640x480 @ 30 FPS
**GPU:** Optional (CPU fallback available)
**RAM:** 4GB+

### Recommended (Testing)

**Platform:** Linux desktop with GPU
**Camera:** HD webcam (1280x720 @ 30 FPS)
**GPU:** CUDA-capable (for SNARC encoding)
**RAM:** 8GB+

### Target (Deployment)

**Platform:** Jetson Nano (4GB)
**Camera:** CSI camera (e.g., Raspberry Pi Camera Module v2)
**Resolution:** 1920x1080 @ 30 FPS
**GPU:** Integrated (2GB)
**RAM:** 4GB

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

**Deliverables:**
- Camera driver interface (abstract + simulated)
- Vision pipeline skeleton
- Unit tests

**Files:**
- `sage/sensors/camera_driver.py`
- `sage/sensors/vision_pipeline.py`
- `sage/sensors/test_camera.py`

### Phase 2: USB Camera Integration (Week 1-2)

**Deliverables:**
- USB camera driver (OpenCV)
- Real frame capture and processing
- Hardware validation tests

**Files:**
- `sage/sensors/usb_camera.py`
- `sage/sensors/test_usb_camera.py`

### Phase 3: Full Integration (Week 2)

**Deliverables:**
- Track 1 integration (sensor trust)
- Track 2 integration (memory)
- Track 3 integration (cognition)
- Integration test suite

**Files:**
- `sage/sensors/vision_integration.py`
- `sage/sensors/test_integration.py`

### Phase 4: Nano Deployment (Week 3)

**Deliverables:**
- CSI camera driver
- Nano-specific optimizations
- Deployment validation

**Files:**
- `sage/sensors/csi_camera.py`
- `sage/sensors/nano_optimizations.py`
- `sage/deployment/test_nano.py`

---

## Risk Assessment

### Technical Risks

1. **Latency Exceeds Budget**
   - *Probability:* Medium
   - *Impact:* High
   - *Mitigation:* Optimize SNARC encoding, use model quantization, reduce resolution

2. **Camera Compatibility Issues**
   - *Probability:* Low
   - *Impact:* Medium
   - *Mitigation:* Test multiple camera models, use standard UVC protocol

3. **Memory Overflow on Nano**
   - *Probability:* Medium
   - *Impact:* High
   - *Mitigation:* Implement adaptive resolution, garbage collection, buffer limits

4. **Poor Lighting Conditions**
   - *Probability:* High
   - *Impact:* Medium
   - *Mitigation:* Auto-exposure, image enhancement, adaptive thresholds

### Integration Risks

1. **SNARC Encoder Too Slow**
   - *Probability:* Medium
   - *Impact:* High
   - *Mitigation:* Model optimization, quantization, pruning

2. **Attention Thrashing**
   - *Probability:* Low
   - *Impact:* Medium
   - *Mitigation:* Hysteresis in attention allocation, minimum dwell time

3. **Memory Storage Overhead**
   - *Probability:* Low
   - *Impact:* Low
   - *Mitigation:* Adaptive frame storage (high-salience only), compression

---

## Success Criteria

### Functional Requirements

✅ Camera successfully initialized and capturing frames
✅ Vision pipeline processes frames <100ms latency
✅ SNARC salience computed for visual observations
✅ Sensor trust tracks vision reliability
✅ Memory stores and retrieves visual observations
✅ Attention manager allocates focus to vision
✅ High-salience visual events trigger goal switches

### Performance Requirements

✅ Sustained 30 FPS frame processing (or 15 FPS minimum)
✅ <100ms total latency (capture to salience)
✅ <500MB memory footprint
✅ <50% GPU utilization
✅ Works on Jetson Nano (target hardware)

### Integration Requirements

✅ Works with existing Tracks 1-3
✅ No breaking changes to prior tracks
✅ Graceful degradation if camera unavailable
✅ Multi-modal support (vision + audio + language)

---

## Next Steps

### Before Implementation

1. **User Review** - Validate architecture design
2. **Hardware Confirmation** - Verify available camera(s)
3. **Priority Confirmation** - Track 4 vs other priorities

### After Approval

1. **Phase 1** - Implement core infrastructure
2. **Phase 2** - USB camera integration
3. **Phase 3** - Full system integration
4. **Phase 4** - Nano deployment

---

## Open Questions

1. **Camera Selection**: Which camera(s) to prioritize? (USB vs CSI)
2. **Resolution**: Start with 640x480 or go directly to 1080p?
3. **Frame Rate**: 30 FPS target or adaptive (15-30 FPS)?
4. **SNARC Model**: Use existing from Track 2 or train camera-specific?
5. **Testing**: User preference for test environment (desktop vs Nano)?

---

## Appendix: File Structure

```
sage/
├── sensors/                      # New module for Track 4
│   ├── __init__.py
│   ├── camera_driver.py          # Abstract camera interface
│   ├── usb_camera.py             # USB camera implementation
│   ├── csi_camera.py             # CSI camera (Nano)
│   ├── simulated_camera.py       # Simulated camera
│   ├── vision_pipeline.py        # Frame processing pipeline
│   ├── camera_config.py          # Configuration management
│   ├── vision_integration.py     # Integration with Tracks 1-3
│   ├── test_camera.py            # Unit tests
│   ├── test_usb_camera.py        # USB camera tests
│   ├── test_integration.py       # Integration tests
│   └── README.md                 # Track 4 documentation
├── cognition/                    # Track 3 (existing)
├── memory/                       # Track 2 (existing)
├── trust/                        # Track 1 (existing)
└── ...
```

---

## Summary

Track 4 brings SAGE into the physical world through camera integration. The architecture prioritizes:

1. **Hardware Abstraction** - Works with multiple camera types
2. **Real-Time Performance** - <100ms latency, 30 FPS target
3. **Nano Compatibility** - Designed for resource constraints
4. **Seamless Integration** - Connects with existing Tracks 1-3

**Status:** Architecture complete, awaiting user approval for implementation

**Estimated Implementation Time:** 2-3 weeks (4 phases)

**Dependencies:** Camera hardware, CUDA environment, Tracks 1-3 operational (✓)

**Next:** User review and approval to proceed with Phase 1
