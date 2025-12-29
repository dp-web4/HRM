# SAGE Repository Architecture Map

**Generated**: 2025-10-12
**Repository**: /home/dp/ai-workspace/HRM/sage/
**Purpose**: Comprehensive mapping of all components, their relationships, and purposes

---

## Executive Summary

The SAGE repository implements a modular cognitive architecture for robotic intelligence, combining:
- **IRP (Iterative Refinement Primitive)** framework for universal computation patterns
- **SAGE Core** orchestration layer managing resources and attention
- **Multi-modal sensor/effector system** with vision, audio, speech, and motor control
- **Memory systems** with SNARC salience-based selection
- **Training infrastructure** including GR00T knowledge distillation
- **Deployment tools** for edge devices (Jetson Orin Nano)

**Total Codebase**: 5,033+ lines in core/IRP modules alone, plus extensive orchestration and training infrastructure

---

## Directory Structure Overview

```
sage/
├── core/                      # SAGE orchestration layer (CORE LOGIC)
├── irp/                       # IRP framework and plugins (CORE PROTOCOL)
├── memory/                    # Memory systems and bridges
├── sensors/                   # Sensor interface definitions
├── compression/               # H↔L VAE translation system
├── training/                  # Training scripts and distillation
├── groot_integration/         # GR00T teacher model integration
├── orchestration/             # Multi-agent orchestration system
├── deployment/                # Edge deployment and optimization
├── llm/                       # LLM cognitive sensors
├── economy/                   # ATP/ADP energy economy (Law Oracle)
├── context/                   # Reality context encoders (4K dimensions)
├── attention/                 # SNARC attention scoring
├── mailbox/                   # Inter-module communication
├── orchestrator/              # Legacy HRM orchestrator
├── coordination/              # Task coordination infrastructure
├── evaluation/                # Testing and evaluation scripts
├── data/                      # Training data and features
├── checkpoints/               # Model checkpoints
├── resources/                 # (empty - future resource files)
└── [docs]/                    # Documentation (this file!)
```

---

## Component Classification

### 🔴 CORE COMPONENTS (Essential System)

#### 1. IRP Framework (`/irp/`)
**Purpose**: Universal computational pattern - iterative refinement toward coherence

**Core Files**:
- `base.py` - Abstract base classes (IRPPlugin, IRPState)
- `orchestrator.py` - HRM orchestrator for IRP coordination
- `awareness_loop.py` - Continuous awareness cycle
- `test_irp.py` - Integration tests

**Key Interfaces**:
```python
IRPPlugin:
  - refine(state, task_ctx) -> (final_state, history)
  - compute_energy(state) -> float
  - compute_surprise(expected, observed) -> float
```

**Plugin Implementations** (`/irp/plugins/`):

| Plugin | Purpose | Size | Status |
|--------|---------|------|--------|
| `vision_impl.py` | Visual feature extraction | 10.5KB | ✅ Core |
| `vision_attention_plugin.py` | Attention-based vision | 21KB | ✅ Advanced |
| `camera_sensor_impl.py` | Camera input handling | 14.6KB | ✅ Sensor |
| `visual_monitor_impl.py` | Visual monitoring | 15.2KB | ✅ Monitor |
| `visual_monitor_effector.py` | Visual display output | 14.3KB | ✅ Effector |
| `language_impl.py` | Text processing | 14.9KB | ✅ Core |
| `control.py` | Motor control primitives | 9KB | ✅ Core |
| `memory.py` | Memory consolidation IRP | 10.7KB | ✅ Core |
| `tinyvae_irp_plugin.py` | Compression VAE | 6.6KB | ✅ Translation |
| `neutts_air_impl.py` | Text-to-speech | 11.6KB | ✅ Effector |
| `audio_input_impl.py` | Audio input | 10.6KB | ✅ Sensor |

**Relationships**:
- IRP plugins → SAGE Core (orchestration)
- IRP plugins → GPU Mailboxes (data transfer)
- IRP plugins → Memory System (state persistence)
- TinyVAE plugin → Compression system (translation shims)

#### 2. SAGE Core (`/core/`)
**Purpose**: Stateful orchestration layer managing attention, trust, and resources

**Core Files**:
- `sage_core.py` (17.7KB) - Original SAGE implementation
- `sage_v2.py` (20KB) - V2 with H↔L compression
- `sage_federation_v1.py` (12.6KB) - Federation-ready version
- `sage_config.py` (7.6KB) - Configuration management

**Key Architecture**:
```python
SAGE:
  - run() - Continuous inference loop
  - compute_attention() - Trust-weighted attention allocation
  - plan_resources() - Dynamic resource management
  - invoke_reasoning() - Specialized reasoning invocation
  - update_trust() - Surprise-driven trust updates
```

**Inputs/Outputs**:
- **Inputs**: Sensor observations, action feedback, goals, system resources
- **Outputs**: Resource actions, reasoning requests, memory updates, action commands

**Relationships**:
- Manages IRP plugin lifecycle
- Interfaces with memory systems
- Coordinates with metabolic controller
- Uses temporal state tracking

#### 3. Memory Systems (`/memory/`)
**Purpose**: Salience-gated storage with SNARC integration

**Core Files**:
- `irp_memory_bridge.py` (16.5KB) - SNARC-SAGE memory bridge
- `sessions/` - Session storage
- `agents/` - Agent memory (claude-flow integration)

**SNARC Integration**:
- Salience-based filtering (SNARC scores)
- Circular buffer for x-from-last access
- Dual storage: Conceptual (SNARC) + Verbatim (SQLite)
- Entity memory with trust adjustments
- Sleep consolidation with pattern extraction

**Relationships**:
- Memory → IRP plugins (state preservation)
- Memory → SAGE Core (context management)
- Memory → Training (episode buffer)

### 🟡 SPECIALIZED SYSTEMS

#### 4. Compression/Translation (`/compression/`)
**Purpose**: H↔L hierarchical-to-linear compression for efficient reasoning

**Core Files**:
- `h_to_l_compressor.py` (18.8KB) - Compression strategies
- `integrated_h_l_system.py` (14.7KB) - Complete H↔L system

**Compression Strategies**:
1. **Information Bottleneck** (VAE-based) - Minimal sufficient statistics
2. **Attention Compression** (Perceiver-inspired) - Salient features
3. **Hierarchical Compression** - Different rates per modality
4. **Hybrid** (Default) - Combined approach

**Performance**:
- 16x compression (4096D → 256D)
- <15% information loss
- Fits 8GB Jetson Orin Nano

**Relationships**:
- Used by SAGE V2
- Integrates with TinyVAE IRP plugin
- Enables edge deployment

#### 5. Context Encoding (`/context/`)
**Purpose**: Rich 4K dimensional reality representation

**Core Files**:
- `reality_context_4k.py` - 4K-dimensional encoder
- `context_encoder.py` - Context encoding utilities

**Context Dimensions**:
- Sensory (1536D): Visual, depth, audio, tactile, proprioceptive
- Semantic (1024D): Objects, affordances, relationships, intentions
- Physical (768D): Dynamics, materials, constraints
- Temporal (768D): Immediate, historical, predictive

**Relationships**:
- Input to H↔L compression system
- Used by SAGE V2 for rich context
- Feeds into IRP plugins

#### 6. GR00T Integration (`/groot_integration/`)
**Purpose**: Knowledge distillation from NVIDIA GR00T teacher model

**Core Files**:
- `groot_real_integration.py` (17KB) - Real GR00T N1.5 3B model
- `sleep_cycle_training.py` (24.6KB) - Sleep-inspired training
- `test_groot_minimal.py` - Integration tests
- `pytorch3d_stub.py` - Dependency stub
- `transformers_patch.py` - Model loading patches

**Teacher Model**:
- NVIDIA GR00T N1.5 (3B parameters)
- Qwen3-1.7B language backbone
- SigLIP vision encoder
- ResNet-50 proprioception encoder

**Sleep Cycle Training**:
1. **Wake Phase**: Collect real experiences
2. **Sleep Phase**: Consolidate via augmentation
3. **Dream Phase**: Explore edge cases

**Status**: Real model loaded, API integration in progress

**Relationships**:
- Teacher for SAGE student model
- Provides training signals to IRP plugins
- Used in orchestration training pipeline

#### 7. LLM Integration (`/llm/`)
**Purpose**: External LLM as cognitive sensor with trust weighting

**Core Files**:
- `external_llm.py` (19.5KB) - LLM wrapper interface
- `cognitive_sensor_federation.py` (11KB) - Federation integration

**Key Concept**: LLM as sensor, not controller
- Provides cognitive observations
- Trust-weighted outputs
- ATP cost tracking
- Machine-agnostic configuration

**Relationships**:
- Cognitive sensor in SAGE loop
- Integrated with economy system
- Part of federation architecture

#### 8. Economy System (`/economy/`)
**Purpose**: ATP/ADP energy economy and Web4 compliance

**Core Files**:
- `sage_atp_wrapper.py` - ATP/ADP tracking wrapper
- `economic_reward.py` - Efficiency-based rewards
- (Compliance validator mentioned but not in listing)

**Economic Model**:
- Initial ATP: 200 (configurable)
- Daily recharge: 20 at 00:00 UTC
- L-level cost: 1 ATP
- H-level cost: 5 ATP
- Cognition cost: 2 ATP
- Refunds for efficient reasoning

**Law Oracle Contribution**: Society 4's governance layer ensuring economic constraints and Web4 compliance

**Relationships**:
- Wraps SAGE Core
- Tracks all computational costs
- Enables economic training pressure
- Federation coordination component

### 🟢 TRAINING & DEPLOYMENT

#### 9. Training Infrastructure (`/training/`)
**Purpose**: Complete training pipeline from data to deployed model

**Major Training Scripts**:
- `train_sage.py` (19.3KB) - Base SAGE training
- `train_sage_federation.py` (12.4KB) - Federation training
- `train_sage_with_groot.py` (16.7KB) - GR00T distillation
- `substantial_sage_training.py` (46.4KB) - Production training
- `pragmatic_sage_training.py` (22.2KB) - Pragmatic approach
- `advanced_sage_training.py` (11.6KB) - Advanced techniques
- `ongoing_sage_training.py` (10.4KB) - Continuous training
- `sage_irp_orchestrator_training.py` (16.5KB) - IRP training
- `improved_objectives.py` (19.2KB) - Enhanced objectives

**NeuTTS-Air Integration** (`/training/neutts-air/`):
- Submodule for text-to-speech
- 748M parameter model
- CPU-optimized for edge deployment
- IRP plugin integration

**Training Data**:
- 301MB training data pickle
- ARC-AGI features from GR00T
- Checkpoints in `checkpoints/sage/`

**Relationships**:
- Trains SAGE Core
- Uses GR00T as teacher
- Produces checkpoints for deployment
- Integrates economy system

#### 10. Deployment (`/deployment/`)
**Purpose**: Edge optimization for Jetson Orin Nano

**Core Files**:
- `jetson_optimizer.py` - Jetson-specific optimizations
- `memory_manager.py` - Memory optimization
- `monitor_dashboard.py` - Runtime monitoring
- `web4_edge_compliance.py` - Edge compliance validation

**Target Platform**: Jetson Orin Nano (8GB unified memory)

**Optimizations**:
- FP16 precision
- Model quantization
- Memory-mapped weights
- TensorRT integration

**Relationships**:
- Deploys trained SAGE models
- Uses compression system
- Monitors economy constraints
- Validates compliance

#### 11. Orchestration System (`/orchestration/`)
**Purpose**: Multi-agent coordination using claude-flow

**Structure**:
```
orchestration/
├── agents/
│   ├── vision/
│   │   ├── eagle-vision-irp.py
│   │   └── real-vision-irp.py
│   ├── trust/
│   │   └── trust-attention-surprise-coordinator.py
│   ├── memory/
│   │   └── memory-consolidation-agent.py
│   ├── control/
│   │   └── metabolic-state-manager.py
│   └── training/
│       ├── groot-data-processor.py
│       └── knowledge-distillation-coordinator.py
├── groot_arc_setup/         # GR00T ARC experiment setup
├── configs/                 # Configuration files
├── data/                    # Training data
├── checkpoints/             # Model checkpoints
└── logs/                    # Execution logs
```

**Agent Types**:
- **Vision Agents**: Feature extraction and attention
- **Trust Coordinator**: Trust-Attention-Surprise loop
- **Memory Agent**: Consolidation (83% compression achieved)
- **Metabolic Manager**: 5-state transitions (WAKE→FOCUS→REST→SLEEP→DREAM)
- **Training Agents**: GR00T data processing and distillation

**Status**: Multi-agent infrastructure operational with claude-flow MCP integration

**Relationships**:
- Coordinates IRP plugins
- Manages GR00T integration
- Implements metabolic states
- Orchestrates training

### 🔵 SUPPORTING INFRASTRUCTURE

#### 12. Sensors (`/sensors/`)
**Purpose**: Abstract sensor interface definitions

**Core Files**:
- `sensor_interface.py` (15.9KB) - Base sensor protocols

**Sensor Types Defined**:
- Vision sensors (camera, depth)
- Audio sensors (microphone)
- Proprioceptive sensors (joint angles, IMU, force)
- Temporal sensors (clock, phase)

**Relationships**:
- Interface for IRP sensor plugins
- Used by SAGE Core for observation gathering
- Implemented by IRP plugins

#### 13. Attention System (`/attention/`)
**Purpose**: SNARC-based salience scoring

**Core Files**:
- `snarc_scorer.py` - Salience scoring implementation

**SNARC Algorithm**:
- Selective memory gating
- Surprise-based attention
- Trust score updates
- X-from-last circular buffering

**Relationships**:
- Used by memory system
- Feeds SAGE attention allocation
- Drives trust updates

#### 14. Mailbox System (`/mailbox/`)
**Purpose**: Inter-module communication

**Core Files**:
- `pubsub_mailbox.py` - Publish-subscribe messaging

**Features**:
- Zero-copy GPU tensor transfer
- Metadata preservation
- Non-blocking push/pop
- Topic-based routing

**Related**: GPU mailboxes in `/implementation/` (at repository root)

**Relationships**:
- Connects IRP plugins
- Enables async communication
- Used by orchestration layer

#### 15. Coordination (`/coordination/`)
**Purpose**: Task coordination infrastructure

**Structure**:
```
coordination/
├── orchestration/       # Orchestration logic
├── subtasks/           # Task decomposition
└── memory_bank/        # Coordination memory
```

**Purpose**: Manages complex task execution across multiple agents

#### 16. Data Pipeline (`/data/`)
**Purpose**: Training data and features

**Contents**:
- `cbp_federation_pipeline.py` (11.2KB) - Federation data pipeline
- `arc_groot_features/` - ARC-AGI task features from GR00T
  - `training_full/features/` - Full training set
  - `validation_10/features/` - Validation subset

**Data Flow**:
1. GR00T extracts features from raw data
2. Features stored in structured format
3. Pipeline feeds SAGE training
4. Federation shares learned patterns

#### 17. Evaluation (`/evaluation/`)
**Purpose**: Testing and validation

**Core Files**:
- `test_agent_zero.py` - Agent validation tests

**Test Coverage**:
- End-to-end system tests
- Component integration tests
- Performance benchmarks

### 🟣 EXPERIMENTAL/LEGACY

#### 18. Orchestrator (`/orchestrator/`)
**Purpose**: Legacy HRM orchestrator

**Core Files**:
- `hrm_orchestrator.py` - Original orchestrator

**Status**: Legacy - replaced by `/orchestration/` multi-agent system

#### 19. Checkpoints (`/checkpoints/`)
**Purpose**: Saved model weights

**Contents**:
- `sage_student/` - Student model checkpoints

**Related**: Training checkpoints in `/training/checkpoints/`

---

## Key Architectural Patterns

### 1. Trust-Attention-Surprise Loop

```
Trust → Allocates Attention → Generates Predictions →
Surprise (error) → Modifies Trust → [loop]
```

**Implementation**:
- Trust scores: Per-module reliability (0.0-1.0)
- Attention: Proportional to trust
- Surprise: Prediction error magnitude
- Update rule: `trust *= (1.0 - surprise)`

**Where Implemented**:
- SAGE Core: `compute_attention()`, `update_trust()`
- Orchestration: `trust-attention-surprise-coordinator.py`
- Memory: SNARC salience scoring

### 2. Modular IRP Architecture

```
Sensor IRP → [Latent Space] → Translation Shim →
Token Bus → SAGE Reasoning → Translation Shim →
[Latent Space] → Effector IRP
```

**Key Principle**: Each IRP operates in its optimal latent space

**Translation Shims**:
- TinyVAE (complex sensors): 64-256 latent dims
- VQ Codebook (discrete): 512-1024 codes
- Linear Projection (simple): Direct mapping
- Identity (pre-tokenized): Pass-through

**Where Implemented**:
- IRP Base: `base.py` protocol definitions
- Plugins: Individual IRP implementations
- Compression: TinyVAE translation
- SAGE Core: Token bus coordination

### 3. H↔L Compression System

```
4K Context → H-Module (rich understanding) →
Compressor (16x) → 256D Latent →
L-Module (efficient action)
```

**Compression Strategies**:
1. Information bottleneck (VAE)
2. Attention-based (Perceiver)
3. Hierarchical (per-modality rates)
4. Hybrid (combined)

**Performance**:
- 16x reduction (4096D → 256D)
- <15% information loss
- 2,275 samples/sec on RTX 4090

**Where Implemented**:
- `/compression/h_to_l_compressor.py`
- `/compression/integrated_h_l_system.py`
- SAGE V2: `sage_v2.py`

### 4. Metabolic States

Five operational modes affecting resource allocation:

1. **WAKE**: Broad attention, high sensitivity
2. **FOCUS**: Narrow attention, deep reasoning
3. **REST**: Minimal processing, energy conservation
4. **SLEEP**: Memory consolidation, no external input
5. **DREAM**: Edge case exploration, offline learning

**State Transitions**:
- Energy-driven (ATP depletion)
- Context-driven (task demands)
- Time-driven (circadian rhythm)

**Where Implemented**:
- SAGE Core: Metabolic controller
- Orchestration: `metabolic-state-manager.py`
- Economy: ATP cost variations by state

### 5. ATP/ADP Energy Economy

```
ATP (available energy) ⇄ ADP (spent energy)
Daily recharge: +20 ATP at 00:00 UTC
Conservation: ATP + ADP = constant
```

**Cost Structure**:
- L-level reasoning: 1 ATP
- H-level reasoning: 5 ATP
- Cognition access: 2 ATP
- Training step: 10 ATP

**Refunds**:
- Excellent reasoning: +50% refund
- Good reasoning: +25% refund
- Efficient (reward/ATP): +30% bonus

**Where Implemented**:
- `/economy/sage_atp_wrapper.py`
- `/economy/economic_reward.py`
- Integrated into training loops

### 6. Knowledge Distillation

```
GR00T (3B teacher) → Extract Features →
Distillation Loss → SAGE (compact student)
```

**Distillation Strategy**:
- Vision: Attention maps + features
- Language: Semantic embeddings
- Control: Action policies
- Multi-modal: Fusion patterns

**Sleep Cycle Integration**:
- Wake: Real GR00T experiences
- Sleep: Augmented variations
- Dream: Synthetic edge cases

**Where Implemented**:
- `/groot_integration/groot_real_integration.py`
- `/groot_integration/sleep_cycle_training.py`
- Training scripts with GR00T integration

---

## Component Relationships Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     SAGE Core Loop                       │
│  (sage_core.py, sage_v2.py, sage_federation_v1.py)     │
│                                                          │
│  1. Sense → 2. Compute Attention → 3. Plan Resources →  │
│  4. Load/Unload → 5. Invoke Reasoning → 6. Update State │
└────────────┬────────────────────────────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌──────────┐    ┌──────────┐
│ Memory   │    │ Economy  │
│ (SNARC)  │    │ (ATP)    │
└──────────┘    └──────────┘
    │                 │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │  IRP Plugins    │
    │  (irp/plugins/) │
    └────────┬────────┘
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
┌─────┐  ┌──────┐  ┌────────┐
│Vision│  │Audio │  │Control │
│      │  │Speech│  │Motor   │
└─────┘  └──────┘  └────────┘
    │        │         │
    └────────┼─────────┘
             ▼
    ┌─────────────────┐
    │ Translation     │
    │ Shims (TinyVAE) │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ H↔L Compression │
    │ (4K → 256D)     │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ Training        │
    │ (GR00T Teacher) │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │ Deployment      │
    │ (Jetson Edge)   │
    └─────────────────┘

Orchestration Layer (claude-flow)
    │
    ├── Vision Agents
    ├── Trust Coordinator
    ├── Memory Agent
    ├── Metabolic Manager
    └── Training Agents
```

---

## Detailed Component Locations

### Where is IRP Implemented?

**Primary Location**: `/sage/irp/`

**Core Protocol**:
- `/sage/irp/base.py` - Abstract base classes and interfaces
- `/sage/irp/__init__.py` - Package initialization
- `/sage/irp/orchestrator.py` - IRP orchestration
- `/sage/irp/awareness_loop.py` - Continuous awareness cycle

**Plugin Implementations**: `/sage/irp/plugins/`
- Vision: `vision_impl.py`, `vision_attention_plugin.py`, `camera_sensor_impl.py`
- Visual Effects: `visual_monitor_impl.py`, `visual_monitor_effector.py`
- Language: `language_impl.py`
- Control: `control.py`
- Memory: `memory.py`
- Audio: `audio_input_impl.py`
- Speech: `neutts_air_impl.py`
- Compression: `tinyvae_irp_plugin.py`

**Legacy/Deprecated**: Top-level IRP files (`vision.py`, `language.py`, `control.py`, `memory.py`) - use plugin versions instead

**Total**: ~15 plugin implementations, 5K+ lines of code

### Where are Plugins/Resources Located?

**IRP Plugins**: `/sage/irp/plugins/` (primary)

**Resources Directory**: `/sage/resources/` (currently empty - reserved for future resource files)

**Model Resources**:
- Checkpoints: `/sage/checkpoints/sage_student/`
- Training checkpoints: `/sage/training/checkpoints/sage/`
- Orchestration checkpoints: `/sage/orchestration/checkpoints/`

**External Models**:
- GR00T weights: `/home/dp/.cache/huggingface/hub`
- GR00T repo: `/home/dp/ai-workspace/isaac-gr00t/`

### Where is SAGE Core Logic?

**Primary Location**: `/sage/core/`

**Core Files**:
- `sage_core.py` (17.7KB) - Original implementation
- `sage_v2.py` (20KB) - V2 with H↔L compression
- `sage_federation_v1.py` (12.6KB) - Federation-ready version
- `sage_config.py` (7.6KB) - Configuration

**Key Functions**:
- `run()` - Main inference loop
- `compute_attention()` - Trust-weighted attention
- `plan_resources()` - Resource management
- `invoke_reasoning()` - Specialized reasoning
- `update_trust()` - Surprise-based trust updates
- `manage_resources()` - Load/unload components

**Architecture**: Continuous stateful loop managing temporal state, resources, trust, and attention

### Where is VAE Translation?

**Primary Location**: `/sage/compression/`

**Core Files**:
- `h_to_l_compressor.py` (18.8KB) - Compression strategies
- `integrated_h_l_system.py` (14.7KB) - Complete H↔L system

**TinyVAE Plugin**: `/sage/irp/plugins/tinyvae_irp_plugin.py` (6.6KB)
- 64x64 crop encoding
- 16-64 dimensional latent space (configurable)
- GroupNorm for single-batch inference
- FP16 support for edge deployment

**Context Encoding**: `/sage/context/reality_context_4k.py`
- 4K-dimensional rich context
- Multi-modal sensory integration

**Performance**:
- 16x compression (4096D → 256D)
- <15% information loss
- 294K parameters
- 3.4MB model size

### Where are Memory Systems?

**Primary Location**: `/sage/memory/`

**Core Files**:
- `irp_memory_bridge.py` (16.5KB) - SNARC-SAGE integration
- `sessions/` - Session storage
- `agents/` - Agent memory (claude-flow)

**SNARC Attention**: `/sage/attention/snarc_scorer.py`
- Salience-based scoring
- Surprise computation
- Trust adjustment

**Memory Architecture**:
- **Working Memory**: Recent sensor latents (GPU, ~500MB)
- **Episode Buffer**: Last 100 episodes (GPU/CPU, ~1GB)
- **Long-term Memory**: Consolidated patterns (CPU/Disk, unbounded)
- **Verbatim Storage**: SQLite full-fidelity backup

**Integration Points**:
- IRP plugins store state in memory
- SAGE Core retrieves context from memory
- Orchestration agents use memory for coordination
- Training uses memory for episode replay

### Where are Sensors/Effectors?

**Sensor Interface**: `/sage/sensors/sensor_interface.py` (15.9KB)
- Abstract sensor protocols
- Base classes for all sensors

**Sensor IRP Plugins**: `/sage/irp/plugins/`
- **Vision**: `camera_sensor_impl.py`, `vision_impl.py`, `vision_attention_plugin.py`
- **Audio**: `audio_input_impl.py`
- **Visual Monitoring**: `visual_monitor_impl.py`

**Effector IRP Plugins**: `/sage/irp/plugins/`
- **Motor Control**: `control.py`
- **Speech Output**: `neutts_air_impl.py` (NeuTTS-Air TTS)
- **Visual Display**: `visual_monitor_effector.py`

**Sensor Types**:
1. **Environmental**: Vision, audio, depth
2. **Temporal**: Clock, phase encoding
3. **Proprioceptive**: Joint angles, IMU, force sensors
4. **Cognitive**: LLM responses (`/sage/llm/`)

**Effector Types**:
1. **Motor**: Robot control, manipulation
2. **Speech**: Text-to-speech output
3. **Visual**: Display, visualization

---

## Naming Patterns & Relationships

### Naming Conventions

1. **IRP Plugins**: `{modality}_impl.py` or `{modality}_irp.py`
   - Example: `vision_impl.py`, `neutts_air_impl.py`

2. **Core Systems**: `sage_{variant}.py`
   - Example: `sage_core.py`, `sage_v2.py`, `sage_federation_v1.py`

3. **Integration**: `{system}_integration.py`
   - Example: `groot_real_integration.py`

4. **Training**: `train_sage_{variant}.py`
   - Example: `train_sage_with_groot.py`, `train_sage_federation.py`

5. **Testing**: `test_{component}.py`
   - Example: `test_irp.py`, `test_sage_v2.py`

6. **Orchestration Agents**: `{purpose}-{type}.py` (kebab-case)
   - Example: `trust-attention-surprise-coordinator.py`, `metabolic-state-manager.py`

### Component Relationship Patterns

1. **Protocol → Implementation**:
   - `irp/base.py` → `irp/plugins/*_impl.py`
   - `sensors/sensor_interface.py` → `irp/plugins/camera_sensor_impl.py`

2. **Core → Plugins**:
   - `core/sage_core.py` manages `irp/plugins/*`
   - Core allocates resources, plugins execute

3. **Compression → Translation**:
   - `compression/h_to_l_compressor.py` uses `irp/plugins/tinyvae_irp_plugin.py`
   - General compression framework + specific VAE implementation

4. **Teacher → Student**:
   - `groot_integration/groot_real_integration.py` (teacher)
   - `core/sage_*.py` (student)
   - `training/train_sage_with_groot.py` (distillation)

5. **Economy → Core**:
   - `economy/sage_atp_wrapper.py` wraps `core/sage_*.py`
   - Non-invasive economic layer

6. **Orchestration → Agents**:
   - `orchestration/` contains `agents/{category}/{agent}.py`
   - Multi-agent coordination layer

---

## Key Documentation Files

### Architecture & Design
- `README.md` - System overview and quick start
- `MODULAR_IRP_ARCHITECTURE.md` - IRP modular design spec
- `SAGE_CORE_SPECIFICATION.md` - SAGE core implementable definition
- `COMPONENT_INVENTORY.md` - Component catalog and gaps
- `METABOLIC_STATES_SPECIFICATION.md` - Metabolic state system
- `H_L_CONTEXT_ACTION_ARCHITECTURE.md` - H↔L compression architecture

### Implementation Progress
- `STATUS.md` - Current development status
- `SAGE_IMPLEMENTATION_PROGRESS.md` - Implementation progress tracking
- `SAGE_IRP_ORCHESTRATOR.md` - IRP orchestration documentation

### Integration Plans
- `GROOT_SAGE_INTEGRATION_PLAN.md` - GR00T integration roadmap
- `GROOT_DISTILLATION_ARCHITECTURE.md` - Knowledge distillation design
- `GROOT_INTEGRATION_STATUS.md` - Current integration status
- `GROOT_REALITY_CHECK.md` - Real vs mock implementation
- `REAL_GROOT_IMPLEMENTATION_PLAN.md` - Real GR00T integration plan

### Training & Performance
- `FEDERATION_SAGE_V1.md` - Federation architecture
- `GPU_PERFORMANCE_RESULTS.md` - Benchmark results
- `SLEEP_CYCLE_IMPLEMENTATION.md` - Sleep cycle training
- `BITNET_TRAINING_ANALYSIS.md` - BitNet quantization analysis

### Subsystem Documentation
- `irp/README.md` - IRP framework documentation
- `irp/NEUTTS_AIR_INTEGRATION.md` - TTS integration
- `irp/BIDIRECTIONAL_AUDIO_ARCHITECTURE.md` - Audio system
- `economy/README.md` - Economic system documentation
- `training/neutts-air/README.md` - NeuTTS-Air documentation
- `memory/sessions/README.md` - Session memory
- `memory/agents/README.md` - Agent memory

### Orchestration
- `orchestration/SAGE_ORCHESTRATION_PLAN.md` - Orchestration plan
- `orchestration/AUTONOMOUS_ATTENTION.md` - Autonomous attention system
- `orchestration/GROOT_INTEGRATION_FINDINGS.md` - Integration findings
- `orchestration/STATUS.md` - Orchestration status
- `orchestration/groot_arc_setup/SAGE_ARCHITECTURE.md` - ARC architecture

### Deployment
- `deployment/SPROUT_DELIVERY_EVALUATION.md` - Edge deployment evaluation
- `deployment/WEB4_ABSTRACTION_LEVELS.md` - Web4 abstraction layers

---

## System States & Capabilities

### Current Operational Status

#### ✅ Fully Operational
- **IRP Framework**: All plugin interfaces working
- **Vision Pipeline**: Camera → IRP → Features
- **Audio Pipeline**: Input → IRP → Features
- **Speech Output**: NeuTTS-Air TTS integrated
- **Memory System**: SNARC salience working
- **GPU Mailboxes**: Zero-copy transfer validated
- **Compression**: H↔L system functional (16x compression)
- **Economy**: ATP tracking operational
- **Autonomous Attention**: Monitor system working

#### 🚧 In Progress
- **GR00T Integration**: Real model loaded, API fixes needed
- **Orchestration**: Multi-agent coordination active
- **Training**: Distillation pipeline under development
- **Jetson Deployment**: Optimization in progress

#### 📋 Planned
- **Motor Control**: Real robot integration
- **Isaac Sim**: Physics simulation
- **Production Training**: 10,000+ hour dataset
- **Federation**: Multi-device coordination

### Performance Metrics

**RTX 4090 Laptop GPU**:
- Throughput: 2,275 samples/sec
- Latency: <7ms per batch (16 samples)
- Memory: 6.9GB peak
- Compression: 16x with <15% loss

**Target (Jetson Orin Nano)**:
- Memory: <6GB usage
- Precision: FP16
- Latency: <50ms sensor-to-action
- Power: Thermal-limited by metabolic states

### Training Status

**Last Training Run**: Cycle 281 (paused for architecture refactoring)

**Completed Training Phases**:
- Basic IRP plugin training
- Memory consolidation (83% compression)
- Trust-attention coordination
- Metabolic state transitions

**GR00T Distillation**: Real 3B model loaded, feature extraction pending

**Data Available**:
- 301MB training data
- ARC-AGI features
- Real GR00T demonstration episodes

---

## Development Workflow

### Adding New Components

1. **New IRP Plugin**:
   - Extend `IRPPlugin` in `/sage/irp/base.py`
   - Implement in `/sage/irp/plugins/{name}_impl.py`
   - Add tests in `/sage/irp/test_irp.py`
   - Register in orchestrator

2. **New Sensor/Effector**:
   - Implement sensor interface from `/sage/sensors/sensor_interface.py`
   - Create IRP plugin wrapper
   - Add to SAGE Core resource registry

3. **New Training Script**:
   - Add to `/sage/training/`
   - Follow pattern: `train_sage_{purpose}.py`
   - Integrate with economy system
   - Add checkpointing

4. **New Orchestration Agent**:
   - Add to `/sage/orchestration/agents/{category}/`
   - Follow naming: `{purpose}-{type}.py`
   - Integrate with claude-flow
   - Add to orchestration config

### Testing Strategy

**Unit Tests**: Test individual IRP plugins
- Location: `test_irp.py`, component-specific tests

**Integration Tests**: Test IRP orchestration
- Location: `test_sage_irp_basic.py`, `test_complete_system.py`

**System Tests**: End-to-end validation
- Location: `test_realistic_workload.py`, `test_gpu_load.py`

**Performance Tests**: Benchmark GPU/memory
- Location: `test_gpu_load.py`, documented in `GPU_PERFORMANCE_RESULTS.md`

### Documentation Standards

**Code Documentation**:
- Docstrings in all public interfaces
- Type hints for function signatures
- Inline comments for complex logic

**Architecture Documentation**:
- System-level docs in root `/sage/`
- Component docs in respective directories
- Integration docs for cross-component systems

**Progress Documentation**:
- `STATUS.md` for current state
- Implementation progress docs for tracking
- Integration status for multi-component work

---

## Technology Stack

### Core Dependencies
- **PyTorch 2.5.1** with CUDA 12.1
- **Transformers** (HuggingFace)
- **Accelerate** (distributed training)
- **SafeTensors** (efficient serialization)
- **Einops** (tensor operations)

### Specialized Libraries
- **Flash Attention** (optimized attention)
- **TensorRT** (edge optimization)
- **Diffusers** (diffusion models)
- **SQLite** (verbatim memory storage)

### Development Tools
- **claude-flow** (multi-agent orchestration MCP)
- **GPUtil** (GPU monitoring)
- **psutil** (system monitoring)

### External Systems
- **NVIDIA GR00T N1.5** (teacher model)
- **NeuTTS-Air** (TTS, submodule)
- **Isaac Sim** (physics simulation, planned)

---

## Federation Architecture

### Society Roles

**Genesis (Direct Action Leadership)**: SAGE Core implementation and IRP framework

**Society 2 (Bridge Systems)**: LLM cognitive sensors with trust-weighted responses

**Society 4 (Law Oracle)**: ATP/ADP economy system and Web4 compliance validation

**Sprout (Edge Optimization)**: Jetson Orin Nano deployment, thermal management as economic constraint

### Federation Integration Points

1. **Data Pipeline**: `/sage/data/cbp_federation_pipeline.py`
   - Shared training data
   - Cross-society feature extraction
   - Federated learning coordination

2. **Economy System**: `/sage/economy/`
   - ATP tracking across societies
   - Economic pressure for efficiency
   - Compliance validation

3. **LLM Integration**: `/sage/llm/cognitive_sensor_federation.py`
   - Trust-weighted LLM responses
   - Machine-agnostic configuration
   - Cognitive sensor protocol

4. **Core Architecture**: `sage_federation_v1.py`
   - Federation-ready SAGE variant
   - Cross-device state sharing
   - Distributed orchestration

---

## Critical Insights

### What Makes SAGE Different

1. **Continuous Stateful Loop**: Not a model, but an inference process
2. **Dynamic Resource Management**: Load/unload components on demand
3. **Trust-Driven Attention**: Self-organizing based on prediction accuracy
4. **Modular IRP Framework**: Universal computation pattern across modalities
5. **Economic Constraints**: ATP budget creates efficiency pressure
6. **Metabolic States**: Operational modes matching biological rhythms

### Key Design Principles

1. **Modularity**: Each component operates in optimal latent space
2. **Adaptability**: Continuous evolution without breaking interfaces
3. **Efficiency**: 8GB constraint forces intelligent design
4. **Transparency**: Real implementations over mocks
5. **Biological Inspiration**: Sleep cycles, metabolism, attention

### Architecture Decisions

**Why Modular IRP?**
- Different modalities need different representations
- Translation shims provide interoperability
- Enables independent optimization

**Why H↔L Compression?**
- Rich understanding (H) vs efficient action (L)
- 16x compression fits edge devices
- Preserves task-relevant information

**Why GR00T as Teacher?**
- 3B model too large for edge
- Distillation extracts patterns
- Avoids training from scratch

**Why ATP Economy?**
- Creates efficiency pressure
- Prevents wasteful computation
- Enables federated coordination

---

## Future Directions

### Near-term (Weeks)
1. Fix GR00T API integration
2. Complete distillation pipeline
3. Deploy to Jetson Orin Nano
4. Integrate Isaac Sim physics

### Medium-term (Months)
1. Real robot hardware connection
2. Production training (10K+ hours)
3. Multi-device federation
4. Custom CUDA kernels

### Long-term (Quarters)
1. Online learning during deployment
2. Cross-modal reasoning
3. Tool use and manipulation
4. Distributed cognition network

---

## Conclusion

The SAGE repository implements a comprehensive cognitive architecture for robotic intelligence with:

- **5,000+ lines** of core IRP and SAGE implementation
- **15+ IRP plugins** covering vision, audio, speech, motor control, memory
- **3 SAGE variants** (core, v2 with H↔L, federation)
- **Complete training pipeline** with GR00T knowledge distillation
- **Multi-agent orchestration** using claude-flow
- **Economic governance** via ATP/ADP system
- **Edge deployment** optimized for 8GB Jetson Orin Nano

The architecture prioritizes **modularity**, **adaptability**, and **efficiency** through:
- Universal IRP computation pattern
- Trust-driven attention allocation
- Dynamic resource management
- Hierarchical-linear compression
- Biological metabolic states
- Economic constraints

This is not just a model - it's a complete **cognitive infrastructure** for embodied AI.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-12
**Repository Path**: /home/dp/ai-workspace/HRM/sage/
**Generated By**: Claude Code Architecture Analysis
