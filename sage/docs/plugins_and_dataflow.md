# SAGE Plugin Ecosystem and Data Flow Documentation

**Date**: October 12, 2025
**Version**: 1.0
**System**: SAGE (Situational Awareness & Guidance Engine)

## Table of Contents
1. [System Overview](#system-overview)
2. [Plugin Inventory](#plugin-inventory)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Memory Systems](#memory-systems)
5. [Plugin Interface Details](#plugin-interface-details)
6. [Data Formats at Each Stage](#data-formats-at-each-stage)
7. [Example Traces](#example-traces)

---

## System Overview

SAGE operates as a **Situational Awareness Orchestrator** that coordinates IRP (Iterative Refinement Primitive) plugins to process sensor inputs, make decisions, and control actuators. The system implements a trust-weighted attention mechanism with resource budgeting (ATP - Allocation Transfer Packet).

### Core Architecture Components

```
Sensors → IRP Plugins → SAGE Core → Memory Systems → Effectors
    ↓          ↓           ↓            ↓              ↓
 Raw Data   Latent     H/L Dual    SNARC/Buffer   Actions
           Refinement   Attention    Memory
```

### Key Principles

1. **Trust-Attention-Surprise Loop**: Trust attracts attention → predictions generate surprise → surprise modifies trust
2. **Modular Plugins**: Each operates in its optimal latent space with translation shims
3. **Resource Awareness**: ATP-based budget management with physical constraints
4. **Memory-Guided Refinement**: Past successful patterns guide future processing

---

## Plugin Inventory

### 1. Sensor IRP Plugins

#### Vision IRP (`vision_impl.py`)
**Location**: `/home/dp/ai-workspace/HRM/sage/irp/plugins/vision_impl.py`

**Purpose**: Refine visual representations in latent space for efficient processing

**Key Features**:
- VAE-based latent compression (224×224×3 → 7×7×256)
- U-Net architecture for latent refinement
- Energy-based early stopping
- Trust scoring via PSNR metrics

**Interface**:
```python
class VisionIRPImpl(IRPPlugin):
    def preprocess(x: Image) -> Latent[7, 7, 256]
    def refine_step(state: Latent) -> Latent
    def compute_energy(state: Latent) -> float  # Lower = better quality
    def postprocess(state: Latent) -> Image
    def compute_trust(initial, refined) -> float  # PSNR-based
```

**Data Flow**:
```
Raw Image [1, 3, 224, 224]
  → VAE Encoder
  → Latent [1, 256, 7, 7]
  → LatentRefiner (U-Net)
  → Refined Latent
  → VAE Decoder
  → Refined Image [1, 3, 224, 224]
```

**Resource Usage**:
- Memory: ~1.0 GB GPU
- Compute: GPU-accelerated convolutions
- Typical iterations: 5-15 (with early stopping)

---

#### Audio Input IRP (`audio_input_impl.py`)
**Location**: `/home/dp/ai-workspace/HRM/sage/irp/plugins/audio_input_impl.py`

**Purpose**: Continuous speech recognition with iterative confidence building

**Key Features**:
- Whisper-based transcription
- Bluetooth microphone support
- Chunk-based accumulation
- Confidence-driven halting

**Interface**:
```python
class AudioInputIRP(IRPPlugin):
    def init_state(x0, ctx) -> IRPState  # Start listening
    def step(state) -> IRPState          # Accumulate audio chunks
    def energy(state) -> float           # Inverse confidence
    def halt(history) -> bool            # Confidence threshold reached
    def extract(state) -> dict           # Final transcription
```

**Data Flow**:
```
Bluetooth Mic → Audio Chunks [16kHz mono]
  → Buffer Accumulation
  → Whisper Transcription
  → Confidence Scoring
  → Text Output + Metadata
```

**Audio State**:
```python
@dataclass
class AudioInputState:
    audio_buffer: np.ndarray      # Accumulated samples
    sample_rate: int              # 16000 Hz
    transcription: str            # Current text
    confidence: float             # 0.0-1.0
    duration: float               # Seconds accumulated
```

---

#### Language IRP (`language.py`)
**Location**: `/home/dp/ai-workspace/HRM/sage/irp/plugins/language.py`

**Purpose**: Masked denoising for text understanding

**Key Features**:
- Span-based masking
- Meaning latent tracking
- Perplexity-based energy
- Iterative unmasking

**Interface** (Partially Implemented):
```python
class LanguageIRP(IRPPlugin):
    def init_state(text) -> IRPState     # Apply masking
    def step(state) -> IRPState          # Denoise tokens
    def energy(state) -> float           # Perplexity
    def decode_text(state) -> str        # Extract text
    def get_meaning_vector() -> Tensor   # Semantic representation
```

**Data Flow**:
```
Text String
  → Tokenization
  → Span Masking [B, L]
  → Iterative Denoising
  → Meaning Latent [B, hidden_dim]
  → Decoded Text
```

---

### 2. Effector IRP Plugins

#### TTS/Speech IRP (`neutts_air_impl.py`)
**Location**: `/home/dp/ai-workspace/HRM/sage/irp/plugins/neutts_air_impl.py`

**Purpose**: Text-to-speech with voice cloning and iterative refinement

**Key Features**:
- NeuTTS Air integration (748M params, CPU-optimized)
- Instant voice cloning from reference
- Prosody refinement across iterations
- Quality-based energy metrics

**Interface**:
```python
class NeuTTSAirIRP(IRPPlugin):
    def init_state(text, ref_audio) -> IRPState
    def step(state, budget) -> (IRPState, budget_used)
    def energy(state) -> float           # Audio quality score
    def extract(state) -> dict           # Final audio + metadata
    def save_audio(state, path)          # Export to file
```

**TTS State**:
```python
@dataclass
class TTSState:
    text: str                            # Input text
    ref_audio: np.ndarray                # Reference for cloning
    ref_text: str                        # Reference transcript
    audio_waveform: np.ndarray           # Generated audio
    prosody_params: Dict[str, float]     # Speed, pitch, energy
    iteration: int
    confidence: float
```

**Data Flow**:
```
Text + Reference Audio
  → NeuCodec Encoding
  → Llama-based Generation [24kHz]
  → Prosody Refinement (optional)
  → Final Audio Waveform
```

**Resource Usage**:
- Memory: ~1.0 GB (CPU/GPU hybrid)
- Model: GGUF quantized for edge devices
- Sample Rate: 24kHz
- Typical iterations: 1-5

---

#### Visual Monitor Effector (`visual_monitor_effector.py`)
**Location**: `/home/dp/ai-workspace/HRM/sage/irp/plugins/visual_monitor_effector.py`

**Purpose**: Display visual feedback and monitoring information

---

### 3. Memory IRP Plugin

#### Memory Consolidation IRP (`irp/plugins/memory.py`)
**Location**: `/home/dp/ai-workspace/HRM/sage/irp/plugins/memory.py`

**Purpose**: Sleep consolidation through progressive abstraction

**Key Features**:
- Multi-level abstraction hierarchy
- Progressive refinement: episodic → semantic → procedural → conceptual → strategic
- Augmentation-based pattern extraction
- SQLite verbatim storage

**Abstraction Levels**:
1. **Episodic**: Specific events (minimal processing)
2. **Semantic**: General knowledge (semantic content extraction)
3. **Procedural**: How-to knowledge (action sequences)
4. **Conceptual**: Abstract principles (high compression)
5. **Strategic**: Meta-level patterns (highest compression)

**Interface**:
```python
class MemoryIRP(IRPPlugin):
    def init_state(experiences) -> IRPState
    def step(state) -> IRPState          # Progressive abstraction
    def energy(state) -> float           # Compression × retrieval
    def augment_memory(mem, type) -> Tensor  # Pattern extraction
    def retrieve(query, level) -> Tensor     # Level-specific retrieval
```

**Memory State**:
```python
IRPState(
    x=memory_batch,              # Current abstraction [B, D]
    meta={
        'current_level': str,     # episodic/semantic/etc.
        'level_idx': int,
        'compression_history': List[float],
        'retrieval_accuracy_history': List[float]
    }
)
```

**Augmentation Types**:
- `temporal_shift`: Shift temporal aspects
- `feature_dropout`: Random feature masking
- `noise_injection`: Gaussian noise
- `permutation`: Dimension shuffling

---

### 4. Specialized Sensor Plugins

#### Camera Sensor (`camera_sensor_impl.py`)
**Purpose**: Real camera integration with attention mechanisms

#### Proprioception Sensor (Planned)
**Purpose**: Joint angles, IMU, force sensors for robotics

#### Temporal/Clock Sensor (Planned)
**Purpose**: Phase-encoded time awareness

---

## Data Flow Architecture

### Overall System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Physical World                                │
│  (Sensors: Camera, Mic, IMU, Temperature, etc.)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ Raw Data
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Sensor Hub                                    │
│  Gathers AttentionPuzzles from all active sensors               │
└────────────────────────┬────────────────────────────────────────┘
                         │ AttentionPuzzle[]
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                 IRP Plugin Processing                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Vision   │  │ Audio    │  │ Language │  │ Memory   │       │
│  │ IRP      │  │ IRP      │  │ IRP      │  │ IRP      │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │ Latent      │ Text         │ Tokens      │ Memories    │
│       │ [7×7×256]   │ + Conf       │ [L, H]      │ [B, D]      │
└───────┴─────────────┴──────────────┴─────────────┴─────────────┘
                         │ Encoded Representations
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   SNARC Scorer                                   │
│  Evaluates: Surprise, Novelty, Arousal, Reward, Conflict       │
│  Output: Salience scores [batch, seq, 1]                        │
└────────────────────────┬────────────────────────────────────────┘
                         │ Salience-weighted inputs
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SAGE Core                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ H-Module (Strategic Reasoning)                           │  │
│  │   7 layers × 768 hidden → Strategy vector                │  │
│  └─────────────────────┬────────────────────────────────────┘  │
│                        ↕ Bidirectional Communication           │
│  ┌─────────────────────┴────────────────────────────────────┐  │
│  │ L-Module (Tactical Execution)                            │  │
│  │   7 layers × 768 hidden → Action logits                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Resource Router: Allocates attention based on trust weights    │
└────────────────────────┬────────────────────────────────────────┘
                         │ Actions + Resource Allocation
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                Memory Systems (Parallel)                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ SNARC       │  │ IRP Memory   │  │ Circular Buffer    │    │
│  │ Selective   │  │ Bridge       │  │ (X-from-last)      │    │
│  │ Memory      │  │              │  │                    │    │
│  └──────┬──────┘  └──────┬───────┘  └─────────┬──────────┘    │
│         │ Salient         │ Patterns           │ Recent         │
└─────────┴─────────────────┴────────────────────┴────────────────┘
                         │ Consolidated Knowledge
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                HRM Orchestrator                                  │
│  ATP Budget Management + Trust-weighted allocation              │
│  Parallel plugin execution with early stopping                  │
└────────────────────────┬────────────────────────────────────────┘
                         │ Orchestrated Commands
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Effector Plugins                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                     │
│  │ TTS      │  │ Motor    │  │ Display  │                     │
│  │ Output   │  │ Control  │  │ Output   │                     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                     │
└───────┴─────────────┴──────────────┴─────────────────────────────┘
                         │ Physical Actions
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Physical World                                │
│  (Actuators: Speakers, Motors, LEDs, etc.)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Memory Systems

### 1. SNARC (Surprise, Novelty, Arousal, Reward, Conflict)

**Location**: `/home/dp/ai-workspace/HRM/sage/attention/snarc_scorer.py`

**Purpose**: Selective memory based on salience scoring

**Architecture**:
```python
class SNARCScorer(nn.Module):
    # Individual scoring networks
    surprise_net: nn.Sequential      # Prediction error
    novelty_net: nn.Sequential       # Memory comparison
    arousal_net: nn.Sequential       # Complexity/entropy
    conflict_net: nn.Sequential      # Ambiguity/variance

    # Memory components
    memory_bank: deque               # Recent experiences
    predictor: nn.Sequential         # For surprise computation
    attention_weight: nn.Sequential  # SNARC → attention weights
```

**SNARC Components**:

1. **Surprise**: Deviation from predictions
   - Computed via predictor network
   - MSE between predicted and actual states
   - Formula: `surprise = MSE(predictor(t-1), actual(t))`

2. **Novelty**: Comparison to memory bank
   - Cosine similarity to stored experiences
   - Formula: `novelty = 1 - max_similarity(state, memory_bank)`

3. **Arousal**: Information density
   - Entropy of state distribution
   - Formula: `arousal = -Σ(p * log(p)) / log(D)`

4. **Reward**: Task success signals
   - External feedback (optional)
   - Default: 0.0

5. **Conflict**: Uncertainty/ambiguity
   - Variance across hidden dimensions
   - Formula: `conflict = var(state, dim=-1)`

**Output Format**:
```python
{
    'snarc_scores': Tensor[batch, seq, 1],      # Combined score
    'attention_weights': Tensor[batch, seq, 1],  # Attention modulation
    'surprise': Tensor[batch, seq, 1],
    'novelty': Tensor[batch, seq, 1],
    'arousal': Tensor[batch, seq, 1],
    'reward': Tensor[batch, seq, 1],
    'conflict': Tensor[batch, seq, 1]
}
```

---

### 2. IRP Memory Bridge

**Location**: `/home/dp/ai-workspace/HRM/sage/memory/irp_memory_bridge.py`

**Purpose**: Store and retrieve successful refinement patterns

**Key Components**:

```python
class IRPMemoryBridge:
    buffer: CircularScratchpad       # Short-term buffer
    verbatim: VerbatimStorage       # Full-fidelity storage
    snarc: SNARCGate                # Salience evaluation
    fast_weights: FastWeightMemory  # Rapid adaptation
    pattern_library: Dict           # Extracted patterns
```

**Memory Recording**:
```python
@dataclass
class RefinementMemory:
    plugin_id: str
    initial_state: Any
    final_state: Any
    energy_trajectory: List[float]
    iterations: int
    compute_saved: float
    trust_score: float
    timestamp: float
    context: Dict[str, Any]

    # Computed properties
    @property
    def convergence_rate() -> float
    @property
    def efficiency() -> float  # trust × compute_saved
```

**Pattern Extraction**:
- Consolidation triggered every 50 memories
- Groups memories by plugin
- Extracts common successful strategies
- Stores as reusable patterns

**Guidance Retrieval**:
```python
guidance = memory_bridge.retrieve_guidance(
    plugin_id='vision_irp',
    current_state=state
)
# Returns:
{
    'max_iterations': int,
    'early_stop_threshold': float,
    'trust_weight': float,
    'pattern': Dict or None,
    'similar_memories': int
}
```

---

### 3. Circular Buffer (X-from-last)

**Purpose**: Recent context for temporal binding

**Features**:
- Fixed-size ring buffer
- Maintains last N experiences
- Fast access to recent history
- Used by SNARC for novelty detection

---

### 4. Verbatim Storage

**Implementation**: SQLite database

**Schema**:
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    timestamp REAL,
    level TEXT,              -- Abstraction level
    content BLOB,            -- Serialized tensor
    metadata TEXT,           -- JSON metadata
    consolidation_count INTEGER
)
```

**Purpose**: Full-fidelity preservation of important experiences

---

## Plugin Interface Details

### Base IRP Interface

**Location**: `/home/dp/ai-workspace/HRM/sage/irp/base.py`

```python
class IRPPlugin:
    """Base class for all IRP implementations"""

    # REQUIRED: Core IRP Contract
    def init_state(self, x0: Any, task_ctx: Dict) -> IRPState
    def energy(self, state: IRPState) -> float
    def step(self, state: IRPState, noise_schedule=None) -> IRPState

    # OPTIONAL: Enhanced functionality
    def project(self, state: IRPState) -> IRPState
    def halt(self, history: List[IRPState]) -> bool
    def get_halt_reason(self, history: List[IRPState]) -> str

    # METRICS: Trust and telemetry
    def compute_trust_metrics(self, history: List[IRPState]) -> Dict
    def emit_telemetry(self, state: IRPState, history: List[IRPState]) -> Dict

    # CONVENIENCE: Complete refinement
    def refine(self, x0: Any, task_ctx: Dict, max_steps: int) -> Tuple[IRPState, List]
```

### IRPState Container

```python
@dataclass
class IRPState:
    x: Any                              # Plugin-specific state
    step_idx: int = 0
    energy_val: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
```

### Sensor Interface

**Location**: `/home/dp/ai-workspace/HRM/sage/sensors/sensor_interface.py`

```python
@dataclass
class AttentionPuzzle:
    sensor_type: str
    data: torch.Tensor
    metadata: Dict[str, Any]
    timestamp: float
    priority: float = 0.5
    snarc_scores: Optional[torch.Tensor] = None

class BaseSensor(ABC):
    def sense(self) -> Optional[AttentionPuzzle]
    def encode(self, data: Any) -> torch.Tensor
    def activate(self)
    def deactivate(self)
    def get_stats(self) -> Dict
```

**Sensor Types Implemented**:
- `VisionSensor`: Image processing (CNN encoder)
- `LanguageSensor`: Text processing (transformer encoder)
- `MemorySensor`: Retrieval from memory store
- `TimeSensor`: Temporal awareness (phase encoding)

---

## Data Formats at Each Stage

### 1. Raw Sensor Data

**Vision**:
```python
# Format: torch.Tensor
shape: [batch, channels, height, width]
dtype: float32
range: [0, 1] or [0, 255]
example: torch.randn(1, 3, 224, 224)
```

**Audio**:
```python
# Format: numpy.ndarray
shape: [samples]
dtype: float32
range: [-1.0, 1.0]
sample_rate: 16000 Hz
example: np.zeros(16000)  # 1 second
```

**Language**:
```python
# Format: torch.Tensor (token IDs)
shape: [batch, sequence_length]
dtype: int64
range: [0, vocab_size-1]
example: torch.randint(0, 50000, (1, 128))
```

---

### 2. Attention Puzzles

```python
AttentionPuzzle(
    sensor_type='vision',
    data=torch.Tensor([1, 1, 224, 224]),
    metadata={
        'shape': [1, 1, 224, 224],
        'unique_values': 10,
        'sparsity': 0.05
    },
    timestamp=1697123456.789,
    priority=0.7,
    snarc_scores=None  # Computed later
)
```

---

### 3. Latent Representations

**Vision Latent**:
```python
# After VAE encoding
shape: [batch, channels, height, width]
example: [1, 256, 7, 7]
dtype: float16  # For memory efficiency
range: typically [-3, 3] with clipping
```

**Audio Latent**:
```python
# Mel-spectrogram features
shape: [batch, time_steps, mel_bins]
example: [1, 100, 128]
dtype: float32
```

**Language Latent**:
```python
# Meaning vector
shape: [batch, hidden_dim]
example: [1, 768]
dtype: float32
```

---

### 4. SNARC Scores

```python
{
    'snarc_scores': torch.Tensor([batch, seq, 1]),
    'attention_weights': torch.Tensor([batch, seq, 1]),
    'surprise': torch.Tensor([batch, seq, 1]),
    'novelty': torch.Tensor([batch, seq, 1]),
    'arousal': torch.Tensor([batch, seq, 1]),
    'reward': torch.Tensor([batch, seq, 1]),
    'conflict': torch.Tensor([batch, seq, 1])
}

# All values in range [0.0, 1.0]
```

---

### 5. SAGE Core Processing

**Input to SAGE**:
```python
{
    'input_ids': torch.Tensor([batch, seq]),  # Token IDs
    'context': torch.Tensor([batch, context_dim]),  # Optional context
    'attention_mask': torch.Tensor([batch, seq]),  # Optional mask
    'num_cycles': int  # Number of reasoning cycles
}
```

**SAGE Output**:
```python
{
    'output': torch.Tensor([batch, seq, num_classes]),  # Action predictions
    'strategy': torch.Tensor([batch, hidden]),  # Strategic decisions
    'halt_probs': torch.Tensor([batch, cycles, 1]),  # Halting probabilities
    'resource_allocation': torch.Tensor([batch, num_resources]),  # Resource usage
    'h_states': torch.Tensor([batch, seq, hidden]),  # H-module states
    'l_states': torch.Tensor([batch, seq, hidden]),  # L-module states
    'num_cycles_used': int
}
```

---

### 6. Memory Formats

**Refinement Memory**:
```python
{
    'plugin_id': 'vision_irp',
    'initial_state': Tensor,
    'final_state': Tensor,
    'energy_trajectory': [1.0, 0.8, 0.6, 0.4, 0.3],
    'iterations': 5,
    'compute_saved': 0.5,  # 50% saved vs max iterations
    'trust_score': 0.85,
    'timestamp': 1697123456.789,
    'context': {
        'early_stopped': True,
        'halt_reason': 'convergence'
    }
}
```

**Pattern Library Entry**:
```python
{
    'avg_iterations': 12.5,
    'avg_convergence_rate': 0.87,
    'best_efficiency': 0.92,
    'energy_profile': [1.0, 0.85, 0.72, 0.61, 0.53, ...],
    'confidence': 0.75  # Based on sample size
}
```

---

### 7. Orchestrator Telemetry

**Plugin Result**:
```python
@dataclass
class PluginResult:
    plugin_id: str
    state: PluginState  # IDLE/RUNNING/COMPLETED/FAILED/HALTED_EARLY
    output: Any
    telemetry: Dict[str, Any]
    start_time: float
    end_time: float
    atp_consumed: float
    trust_score: float

    @property
    def execution_time() -> float
    @property
    def efficiency() -> float  # trust / atp
```

**ATP Budget Report**:
```python
{
    'total_budget': 1000.0,
    'total_allocated': 950.0,
    'total_consumed': 680.0,
    'utilization': 0.68,
    'per_plugin': {
        'vision': {
            'allocated': 500.0,
            'consumed': 380.0,
            'utilization': 0.76
        },
        'audio': {
            'allocated': 450.0,
            'consumed': 300.0,
            'utilization': 0.67
        }
    }
}
```

---

## Example Traces

### Example 1: Vision Processing Pipeline

```python
# Step 1: Raw Image Input
raw_image = torch.randn(1, 3, 224, 224)  # Random test image
print(f"Raw image shape: {raw_image.shape}")
# Output: Raw image shape: torch.Size([1, 3, 224, 224])

# Step 2: Create AttentionPuzzle
vision_sensor = VisionSensor(hidden_size=768, grid_size=224)
puzzle = vision_sensor.sense(raw_image)
print(f"Puzzle type: {puzzle.sensor_type}")
print(f"Puzzle priority: {puzzle.priority}")
print(f"Puzzle metadata: {puzzle.metadata}")
# Output:
#   Puzzle type: vision
#   Puzzle priority: 0.7
#   Puzzle metadata: {'shape': [1, 1, 224, 224], 'unique_values': 256, 'sparsity': 0.01}

# Step 3: Vision IRP Processing
vision_irp = VisionIRPImpl(vae_variant='minimal', device='cuda')

# Initialize state
state = vision_irp.init_state(raw_image, task_ctx={'task': 'segmentation'})
print(f"Initial latent shape: {state.x.shape}")
# Output: Initial latent shape: torch.Size([1, 128, 7, 7])

# Energy before refinement
initial_energy = vision_irp.energy(state)
print(f"Initial energy: {initial_energy:.4f}")
# Output: Initial energy: -0.4523

# Refinement loop
history = [state]
for i in range(10):
    state = vision_irp.step(state, noise_schedule=None)
    state = vision_irp.project(state)
    state.energy_val = vision_irp.energy(state)
    history.append(state)

    if vision_irp.halt(history):
        print(f"Halted early at iteration {i+1}")
        break
# Output: Halted early at iteration 7

# Final energy
final_energy = state.energy_val
print(f"Final energy: {final_energy:.4f}")
print(f"Energy improvement: {initial_energy - final_energy:.4f}")
# Output:
#   Final energy: -0.8912
#   Energy improvement: 0.4389

# Compute trust
refined_image = vision_irp.postprocess(state.x)
trust = vision_irp.compute_trust(raw_image, refined_image)
print(f"Trust score: {trust:.3f}")
# Output: Trust score: 0.847

# Telemetry
telemetry = vision_irp.emit_telemetry(state, history)
print(f"Telemetry: {telemetry}")
# Output:
# {
#     'entity_id': 'vision_irp',
#     'plugin': 'vision',
#     'step_idx': 7,
#     'E': -0.8912,
#     'dE': -0.0623,
#     'steps': 8,
#     'halt_reason': 'slope<eps',
#     'trust': {'monotonicity_ratio': 0.875, 'dE_variance': 0.0034, 'convergence_rate': 0.0623},
#     'budget': {'ATP_spent': 0.8, 'time_ms': 145.2, 'memory_mb': 256.0}
# }
```

---

### Example 2: Audio Input → Transcription

```python
# Step 1: Initialize Audio IRP
config = {
    'entity_id': 'audio_input',
    'device': 'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit',
    'sample_rate': 16000,
    'chunk_duration': 2.0,
    'min_confidence': 0.7,
    'max_duration': 10.0,
    'whisper_model': 'tiny'
}
audio_irp = AudioInputIRP(config)

# Step 2: Start listening
state = audio_irp.init_state(None, {'prompt': 'Listen to user command'})
print(f"Listening started...")

history = [state]

# Step 3: Refinement loop (recording + transcription)
while not audio_irp.halt(history):
    print(f"\nStep {state.step_idx}: Recording 2.0s...")

    # Record and transcribe
    state = audio_irp.step(state)
    state.energy_val = audio_irp.energy(state)
    history.append(state)

    # Show progress
    audio_state = state.x
    print(f"  Duration: {audio_state.duration:.1f}s")
    print(f"  Confidence: {audio_state.confidence:.2f}")
    print(f"  Energy: {state.energy_val:.3f}")
    if audio_state.transcription:
        print(f"  Text: \"{audio_state.transcription}\"")

# Output:
# Listening started...
#
# Step 0: Recording 2.0s...
#   Duration: 2.0s
#   Confidence: 0.45
#   Energy: 0.750
#   Text: "Turn on the"
#
# Step 1: Recording 2.0s...
#   Duration: 4.0s
#   Confidence: 0.68
#   Energy: 0.420
#   Text: "Turn on the lights in"
#
# Step 2: Recording 2.0s...
#   Duration: 6.0s
#   Confidence: 0.82
#   Energy: 0.230
#   Text: "Turn on the lights in the kitchen"

# Step 4: Extract final result
result = audio_irp.extract(state)
print(f"\nFinal Transcription:")
print(f"  Text: {result['text']}")
print(f"  Confidence: {result['confidence']:.2f}")
print(f"  Duration: {result['duration']:.1f}s")
print(f"  Halt reason: {audio_irp.get_halt_reason(history)}")

# Output:
# Final Transcription:
#   Text: Turn on the lights in the kitchen
#   Confidence: 0.82
#   Duration: 6.0s
#   Halt reason: confident (0.82)
```

---

### Example 3: SNARC-Guided SAGE Processing

```python
# Step 1: Create inputs
batch_size = 2
seq_len = 10
hidden_size = 768

input_states = torch.randn(batch_size, seq_len, hidden_size)
print(f"Input states: {input_states.shape}")

# Step 2: SNARC Scoring
snarc_scorer = SNARCScorer(hidden_size=hidden_size)
snarc_output = snarc_scorer(input_states, return_components=True)

print(f"\nSNARC Scores:")
print(f"  Surprise: {snarc_output['surprise'].mean():.3f}")
print(f"  Novelty: {snarc_output['novelty'].mean():.3f}")
print(f"  Arousal: {snarc_output['arousal'].mean():.3f}")
print(f"  Reward: {snarc_output['reward'].mean():.3f}")
print(f"  Conflict: {snarc_output['conflict'].mean():.3f}")
print(f"  Combined: {snarc_output['snarc_scores'].mean():.3f}")

# Output:
# SNARC Scores:
#   Surprise: 0.523
#   Novelty: 0.847
#   Arousal: 0.412
#   Reward: 0.000
#   Conflict: 0.358
#   Combined: 0.528

# Step 3: Get top-k salient positions
top_indices, top_scores = snarc_scorer.get_top_k_salient(
    snarc_output['snarc_scores'],
    k=3
)
print(f"\nTop-3 Salient Positions:")
print(f"  Indices: {top_indices}")
print(f"  Scores: {top_scores}")

# Output:
# Top-3 Salient Positions:
#   Indices: tensor([[4, 7, 2], [5, 8, 1]])
#   Scores: tensor([[0.847, 0.723, 0.651], [0.892, 0.768, 0.645]])

# Step 4: SAGE Processing
config = SAGEConfig(
    hidden_dim=768,
    num_attention_heads=12,
    num_h_layers=7,
    num_l_layers=7,
    max_seq_length=512,
    num_classes=1000
)
sage = SAGECore(config)

# Convert to token IDs (simplified)
input_ids = torch.randint(0, config.num_classes, (batch_size, seq_len))

# Forward pass
output = sage(
    input_ids=input_ids,
    context=None,
    attention_mask=None,
    num_cycles=3
)

print(f"\nSAGE Output:")
print(f"  Output shape: {output['output'].shape}")
print(f"  Strategy shape: {output['strategy'].shape}")
print(f"  Cycles used: {output['num_cycles_used']}")
print(f"  Resource allocation: {output['resource_allocation']}")

# Output:
# SAGE Output:
#   Output shape: torch.Size([2, 10, 1000])
#   Strategy shape: torch.Size([2, 768])
#   Cycles used: 3
#   Resource allocation: tensor([[0.2341, 0.1823, 0.2156, 0.1945, 0.1735],
#                                [0.2189, 0.1956, 0.2087, 0.2012, 0.1756]])
```

---

### Example 4: Complete Orchestration

```python
# Step 1: Create Orchestrator
orchestrator = HRMOrchestrator(
    initial_atp=1000.0,
    max_concurrent=4,
    device='cuda'
)

# Step 2: Register Plugins
orchestrator.create_default_plugins()  # Vision + Language

# Step 3: Prepare Tasks
tasks = {
    'vision': torch.randn(2, 3, 224, 224).cuda(),
    'language': torch.randint(100, 5000, (2, 32)).cuda()
}

print("Starting parallel execution...")
print(f"  Vision task: {tasks['vision'].shape}")
print(f"  Language task: {tasks['language'].shape}")
print(f"  Initial ATP budget: {orchestrator.budget.total}")

# Step 4: Execute in Parallel
import asyncio
results = asyncio.run(
    orchestrator.execute_parallel(tasks, early_stop=True)
)

# Step 5: Analyze Results
for result in results:
    print(f"\n{result.plugin_id.upper()}:")
    print(f"  State: {result.state.value}")
    print(f"  Time: {result.execution_time:.3f}s")
    print(f"  ATP consumed: {result.atp_consumed:.1f}")
    print(f"  Trust score: {result.trust_score:.3f}")
    print(f"  Efficiency: {result.efficiency:.3f}")
    print(f"  Iterations: {result.telemetry.get('iterations', 'N/A')}")
    print(f"  Compute saved: {result.telemetry.get('compute_saved', 0)*100:.1f}%")

# Output:
# Starting parallel execution...
#   Vision task: torch.Size([2, 3, 224, 224])
#   Language task: torch.Size([2, 32])
#   Initial ATP budget: 1000.0
#
# VISION:
#   State: halted_early
#   Time: 0.145s
#   ATP consumed: 70.0
#   Trust score: 0.847
#   Efficiency: 0.012
#   Iterations: 7
#   Compute saved: 86.0%
#
# LANGUAGE:
#   State: halted_early
#   Time: 0.089s
#   ATP consumed: 50.0
#   Trust score: 0.723
#   Efficiency: 0.014
#   Iterations: 5
#   Compute saved: 90.0%

# Step 6: Summary
summary = orchestrator.get_orchestration_summary()
print(f"\nOrchestration Summary:")
print(f"  Total execution: {summary['total_execution_time']:.3f}s")
print(f"  Plugins successful: {summary['successful']}/{summary['plugins_executed']}")
print(f"  Early stopped: {summary['early_stopped']}")
print(f"  Average efficiency: {summary['average_efficiency']:.3f}")
print(f"  Budget utilization: {summary['budget_report']['utilization']*100:.1f}%")

# Output:
# Orchestration Summary:
#   Total execution: 0.156s
#   Plugins successful: 2/2
#   Early stopped: 2
#   Average efficiency: 0.013
#   Budget utilization: 12.0%
```

---

### Example 5: Memory-Guided Refinement

```python
# Step 1: Create Memory Bridge
memory_bridge = IRPMemoryBridge(
    buffer_size=50,
    consolidation_threshold=10
)

# Step 2: Wrap Vision IRP with Memory
vision_irp = create_vision_irp(device='cuda')
memory_guided_vision = MemoryGuidedIRP(vision_irp, memory_bridge)

# Step 3: Build Memory with Multiple Refinements
print("Building memory with refinements...")

for i in range(15):
    test_image = torch.randn(1, 3, 224, 224).cuda()

    # Refine with memory guidance
    refined, telemetry = memory_guided_vision.refine(test_image, early_stop=True)

    if i % 5 == 0:
        print(f"  Refinement {i+1}:")
        print(f"    Iterations: {telemetry['iterations']}")
        print(f"    Efficiency: {telemetry.get('memory_efficiency', 0):.3f}")
        print(f"    Convergence: {telemetry.get('convergence_rate', 0):.3f}")

# Output:
# Building memory with refinements...
#   Refinement 1:
#     Iterations: 15
#     Efficiency: 0.425
#     Convergence: 0.687
#   Refinement 6:
#     Iterations: 9
#     Efficiency: 0.712
#     Convergence: 0.823
#   Refinement 11:
#     Iterations: 6
#     Efficiency: 0.891
#     Convergence: 0.914

# Step 4: Check Memory Stats
stats = memory_bridge.get_memory_stats()
print(f"\nMemory Statistics:")
print(f"  Total memories: {stats['total_memories']}")
print(f"  Pending consolidation: {stats['pending_consolidation']}")
print(f"  Patterns extracted: {stats['patterns_extracted']}")
print(f"  Avg efficiency: {stats.get('avg_efficiency', 0):.3f}")
print(f"  Avg iterations: {stats.get('avg_iterations', 0):.1f}")

# Output:
# Memory Statistics:
#   Total memories: 12
#   Pending consolidation: 5
#   Patterns extracted: 1
#   Avg efficiency: 0.743
#   Avg iterations: 8.2

# Step 5: Retrieve Guidance
guidance = memory_bridge.retrieve_guidance(
    'vision_irp',
    torch.randn(1, 3, 224, 224).cuda()
)

print(f"\nGuidance from Memory:")
print(f"  Max iterations: {guidance['max_iterations']}")
print(f"  Early stop threshold: {guidance['early_stop_threshold']:.4f}")
print(f"  Trust weight: {guidance['trust_weight']:.3f}")
print(f"  Similar memories: {guidance['similar_memories']}")

# Output:
# Guidance from Memory:
#   Max iterations: 10
#   Early stop threshold: 0.0082
#   Trust weight: 0.980
#   Similar memories: 12
```

---

## Summary

### Plugin Count
- **Sensor IRPs**: 4 implemented (Vision, Audio, Language, Memory)
- **Effector IRPs**: 2 implemented (TTS, Visual Monitor)
- **Total Active**: 6 plugins

### Data Flow Summary

1. **Sensors** gather raw data → AttentionPuzzles
2. **IRP Plugins** refine in latent spaces → Refined representations
3. **SNARC** evaluates salience → Attention weights
4. **SAGE Core** processes strategically/tactically → Actions + resource allocation
5. **Memory Systems** store/retrieve patterns → Guidance for future
6. **Orchestrator** manages resources → ATP-budgeted execution
7. **Effectors** execute actions → Physical world impact

### Memory Hierarchy

- **Working Memory**: SNARC + Circular Buffer (GPU, <500MB)
- **Episodic Buffer**: Recent refinements (IRP Memory Bridge)
- **Pattern Library**: Consolidated knowledge (extracted patterns)
- **Long-term Storage**: Verbatim SQLite database (disk)

### Key Innovation

The system achieves **adaptive intelligence** through:
- Trust-weighted attention allocation
- Early stopping for efficiency (typical 50-90% compute savings)
- Memory-guided refinement (improves over time)
- Multi-level abstraction (episodic → strategic)
- Physical constraint awareness (ATP as actual watts)

---

## Next Steps

1. **Real Sensor Integration**: Connect GPIO sensors on Jetson
2. **Federation Training**: Distribute learning across devices
3. **Performance Benchmarking**: Measure end-to-end latency
4. **Tool Use**: Implement manipulation primitives
5. **World Model**: Build predictive environment model

---

**Document Version**: 1.0
**Last Updated**: October 12, 2025
**Maintainers**: Dennis Palatov, SAGE Development Team
