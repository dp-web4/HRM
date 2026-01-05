# SAGE Component Readiness Map

**Date**: October 12, 2025
**Purpose**: Detailed inventory of existing SAGE components and their integration readiness
**Status**: Complete system analysis for SAGESystem integration

---

## Executive Summary

**Reality Check**: The SAGE components exist and are largely functional, but they are **NOT integrated into a unified system**. Each component lives in isolation. We have the parts, but no assembly.

**Key Finding**: The gap is not in the components themselves, but in the **orchestration layer** that connects them into the living system described in `SAGE_WORKING_VISION.md`.

---

## Component Status Matrix

| Component | Exists? | Location | Usable? | Integration Effort |
|-----------|---------|----------|---------|-------------------|
| SNARC Scorer | ✅ | `/sage/attention/snarc_scorer.py` | ⚠️ Needs API adaptation | Medium |
| IRP Base Framework | ✅ | `/sage/irp/base.py` | ✅ Ready | Low |
| IRP Plugins (15+) | ✅ | `/sage/irp/plugins/` | ✅ Ready | Low |
| HRM Orchestrator | ✅ | `/sage/orchestrator/hrm_orchestrator.py` | ⚠️ Needs wiring | Medium |
| Memory Systems | ✅ | `/sage/memory/` | ⚠️ Needs integration | Medium |
| Metabolic Controller | ✅ | `/sage/orchestration/agents/control/` | ⚠️ Standalone | Medium |
| VAE Translation | ✅ | `/sage/compression/` + `/sage/irp/plugins/tinyvae_irp_plugin.py` | ✅ Ready | Low |
| SAGE Core (H-L modules) | ✅ | `/sage/core/sage_core.py` | ⚠️ Wrong abstraction | High |
| **SAGESystem (unified loop)** | ❌ | **Missing** | N/A | **CRITICAL GAP** |

---

## 1. SNARC Scorer

### Location
`/home/dp/ai-workspace/HRM/sage/attention/snarc_scorer.py`

### Current Interface
```python
class SNARCScorer(nn.Module):
    def __init__(self, hidden_size: int = 768, memory_size: int = 1000)

    def forward(
        self,
        input_states: torch.Tensor,  # [batch, seq, hidden]
        context: Optional[torch.Tensor] = None,
        task_success: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Returns:
        # - snarc_scores: Combined SNARC scores [batch, seq, 1]
        # - attention_weights: Attention importance weights [batch, seq, 1]
        # - (optional) surprise, novelty, arousal, reward, conflict
```

### Capabilities
- ✅ Computes 5D SNARC (Surprise, Novelty, Arousal, Reward, Conflict)
- ✅ Learnable neural networks for each dimension
- ✅ Memory bank for novelty assessment (1000 element deque)
- ✅ Attention weight computation
- ✅ Top-K salient position extraction
- ✅ Attention biasing capability

### Interface Mismatch
**Problem**: Expects `torch.Tensor` inputs with shape `[batch, seq, hidden]`, but vision document describes:
```python
observations = {
    'vision': torch.tensor([3, 480, 640]),  # Raw image
    'audio': torch.tensor([16000]),         # Raw audio
    'clock': 1728764892.134                # Timestamp
}
```

**Gap**: Need an **observation encoder** that converts multi-modal raw inputs into the hidden state format SNARC expects.

### Integration Requirements
1. **Create**: `ObservationEncoder` - Maps raw sensor data → hidden states
2. **Create**: Adapter layer for modality-specific encoding
3. **Modify**: SNARC to accept dict of observations or create wrapper
4. **Wire**: SNARC output → AttentionAllocator input

### Recommended Approach
```python
class ObservationEncoder:
    def __init__(self, hidden_size=768):
        self.encoders = {
            'vision': VisionEncoder(output_dim=hidden_size),
            'audio': AudioEncoder(output_dim=hidden_size),
            'clock': TemporalEncoder(output_dim=hidden_size)
        }

    def encode(self, observations: Dict[str, Any]) -> torch.Tensor:
        # Convert each modality to hidden_size
        encoded = []
        for modality, data in observations.items():
            encoded.append(self.encoders[modality](data))
        return torch.stack(encoded, dim=1)  # [batch, num_modalities, hidden]
```

### Status: ⚠️ **Ready but needs API adaptation**

---

## 2. IRP Plugin Framework

### Location
`/home/dp/ai-workspace/HRM/sage/irp/base.py`

### Current Interface
```python
class IRPPlugin:
    def __init__(self, config: Dict[str, Any])

    # Core IRP Contract (must override)
    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState
    def energy(self, state: IRPState) -> float
    def step(self, state: IRPState, noise_schedule: Any = None) -> IRPState

    # Optional overrides
    def project(self, state: IRPState) -> IRPState
    def halt(self, history: List[IRPState]) -> bool

    # Convenience
    def refine(self, x0: Any, task_ctx: Dict[str, Any] = None,
               max_steps: Optional[int] = None) -> tuple[IRPState, List[IRPState]]
```

### Capabilities
- ✅ Complete IRP protocol implementation
- ✅ IRPState container with metadata
- ✅ Energy-based convergence detection
- ✅ Trust metrics computation (monotonicity, variance, convergence rate)
- ✅ Telemetry emission for Web4 integration
- ✅ Halt condition with configurable epsilon and K parameters

### Current Usage Pattern
Perfect! Vision document describes exactly this:
```python
results = {
    'vision': PluginResult(
        output=latent_64d,
        energy_trajectory=[2.34, 1.89, 1.56, 1.44, 1.41, 1.40],
        iterations=6,
        atp_used=600,
        converged=True
    )
}
```

### Integration Requirements
**NONE** - Interface matches vision document perfectly.

### Status: ✅ **Ready for direct use**

---

## 3. IRP Plugins

### Location
`/home/dp/ai-workspace/HRM/sage/irp/plugins/`

### Available Plugins (15+)
1. **Vision**:
   - `vision_impl.py` - VisionIRPImpl with VAE latent refinement
   - `vision_attention_plugin.py` - SNARC-integrated vision
   - `tinyvae_irp_plugin.py` - TinyVAE (64D latent, Jetson-optimized)
   - `camera_sensor_impl.py` - Camera sensor interface

2. **Audio**:
   - `audio_input_impl.py` - Audio input processing
   - `neutts_air_impl.py` - TTS (NeuTTS Air integration)

3. **Language**:
   - `language_impl.py` - Language processing
   - `language.py` - Language base plugin

4. **Memory**:
   - `memory.py` - Memory IRP plugin

5. **Control**:
   - `control.py` - Control/action plugin

6. **Monitoring**:
   - `visual_monitor_impl.py` - Visual output monitoring
   - `visual_monitor_effector.py` - Visual display effector

### Example: VisionIRPImpl Interface
```python
class VisionIRPImpl(IRPPlugin):
    def refine(self, x: Any, early_stop: bool = True) -> Tuple[Any, Dict[str, Any]]:
        # Returns: (refined_image, telemetry)
        # telemetry = {
        #     'iterations': 6,
        #     'final_energy': -2.34,
        #     'energy_delta': -0.94,
        #     'trust': 0.85,
        #     'early_stopped': True,
        #     'compute_saved': 0.88
        # }
```

### Integration Requirements
1. **Wire**: Plugin loading/unloading in ResourceManager
2. **Create**: Plugin registry/discovery system
3. **Add**: ATP budget tracking per plugin (partially exists in orchestrator)
4. **Implement**: Dynamic plugin import (lazy loading)

### Status: ✅ **Ready, just need dynamic loading**

---

## 4. HRM Orchestrator

### Location
`/home/dp/ai-workspace/HRM/sage/orchestrator/hrm_orchestrator.py`

### Current Interface
```python
class HRMOrchestrator:
    def __init__(
        self,
        initial_atp: float = 1000.0,
        max_concurrent: int = 4,
        reallocation_interval: float = 0.1,
        device: Optional[torch.device] = None
    )

    def register_plugin(self, plugin_id: str, plugin: IRPPlugin, initial_trust: float = 1.0)

    async def execute_plugin(
        self, plugin_id: str, input_data: Any, early_stop: bool = True
    ) -> PluginResult

    async def execute_parallel(
        self, tasks: Dict[str, Any], early_stop: bool = True
    ) -> List[PluginResult]
```

### Capabilities
- ✅ ATP (Allocation Transfer Packet) budget management
- ✅ Trust-weighted resource allocation
- ✅ Concurrent plugin execution (asyncio)
- ✅ Dynamic budget reallocation
- ✅ Plugin state tracking
- ✅ Efficiency reporting

### Interface Match with Vision
Vision document shows:
```python
results = sage.orchestrator.run_cycle(targets, required, atp_budget=1000)
```

Current interface uses:
```python
results = await orchestrator.execute_parallel(tasks, early_stop=True)
```

### Gap Analysis
**Missing**:
- `run_cycle()` method that integrates with attention targets
- Connection to resource_manager for plugin loading
- Integration with metabolic state for ATP budget adjustment
- Synchronous wrapper (vision document doesn't use async)

**Exists but disconnected**:
- ATP budget system ✅
- Trust tracking ✅
- Parallel execution ✅

### Integration Requirements
1. **Add**: `run_cycle(targets, required_plugins, atp_budget)` method
2. **Modify**: Support both sync and async execution
3. **Wire**: Connect to ResourceManager for plugin lifecycle
4. **Add**: Metabolic state awareness for budget scaling

### Status: ⚠️ **Functional but needs wiring layer**

---

## 5. Memory Systems

### Location
`/home/dp/ai-workspace/HRM/sage/memory/`

### Available Systems

#### 5.1 IRP Memory Bridge
**File**: `irp_memory_bridge.py`

```python
class IRPMemoryBridge:
    def __init__(
        self,
        buffer_size: int = 100,
        snarc_capacity: int = 1000,
        consolidation_threshold: int = 50,
        device: Optional[torch.device] = None
    )

    def record_refinement(
        self, plugin_id: str, initial_state: Any,
        final_state: Any, energy_trajectory: List[float],
        telemetry: Dict[str, Any]
    ) -> RefinementMemory

    def retrieve_guidance(
        self, plugin_id: str, current_state: Any, k: int = 5
    ) -> Dict[str, Any]

    def consolidate()  # Sleep phase - pattern extraction
```

**Capabilities**:
- ✅ RefinementMemory storage (plugin execution history)
- ✅ Pattern extraction via consolidation
- ✅ Guidance retrieval for similar tasks
- ✅ SNARC integration (mock if real SNARC unavailable)
- ✅ Circular buffer for recent memories
- ✅ Verbatim storage capability

#### 5.2 Memory Components from Vision
Vision document describes 4 parallel systems:
1. **SNARC Memory** - Selective storage via salience
2. **IRP Memory Bridge** - Refinement pattern library ✅ (exists)
3. **Circular Buffer** - X-from-last context
4. **Verbatim Storage** - SQLite full records

### Gap Analysis
**Missing**:
- Unified `MemorySystem` class that coordinates all 4 subsystems
- Circular buffer standalone implementation
- SQLite verbatim storage implementation
- Integration with SNARC scorer for salience-based storage

**Exists**:
- IRP Memory Bridge ✅
- Basic SNARC mock in memory bridge

### Integration Requirements
1. **Create**: `MemorySystem` unified interface
2. **Create**: `CircularBuffer` implementation
3. **Create**: `VerbatimStorage` with SQLite backend
4. **Wire**: All 4 systems into single `memory_system.update()` call

### Status: ⚠️ **Core exists, needs unification**

---

## 6. Metabolic Controller

### Location
`/home/dp/ai-workspace/HRM/sage/orchestration/agents/control/metabolic-state-manager.py`

### Current Interface
```python
class MetabolicState(Enum):
    WAKE = "WAKE"      # Normal operation, broad attention
    FOCUS = "FOCUS"    # High performance on specific task
    REST = "REST"      # Recovery and maintenance
    DREAM = "DREAM"    # Consolidation and exploration
    CRISIS = "CRISIS"  # Emergency response mode

class MetabolicStateManager:
    def __init__(self, config: Dict = None)

    def start()  # Begins async state management loop
    def stop()

    def submit_event(self, event: Dict[str, Any])
    def get_state_config(self) -> StateConfig
    def get_status(self) -> Dict[str, Any]
```

### StateConfig Structure
```python
@dataclass
class StateConfig:
    name: MetabolicState
    energy_consumption_rate: float  # Energy per time unit
    attention_breadth: int          # Number of simultaneous focuses
    surprise_sensitivity: float     # Threshold for surprise detection
    exploration_rate: float         # Random exploration probability
    max_duration: float             # Maximum time in this state
    transition_conditions: Dict[str, Any]
```

### Capabilities
- ✅ 5 metabolic states with complete configs
- ✅ EnergyManager with consumption and recharge
- ✅ Autonomous state transition logic
- ✅ Event-driven updates (surprise, performance, error, attention)
- ✅ Threading for background operation
- ✅ Transition history tracking

### Interface Match with Vision
Vision document:
```python
sage.metabolic_controller.update(energy, fatigue, stress)
```

Current interface:
```python
manager.submit_event({"type": "performance", "value": 0.6})
```

### Gap Analysis
**Missing**:
- Integration with SAGESystem main loop
- Direct coupling to ATP budget (states affect budget but not connected)
- Attention breadth doesn't affect orchestrator
- No connection to plugin loading decisions

**Exists but standalone**:
- Complete state machine ✅
- Energy management ✅
- Transition logic ✅

### Integration Requirements
1. **Add**: `update(energy, fatigue, stress)` method for sync updates
2. **Wire**: State config → Orchestrator ATP budget scaling
3. **Wire**: `attention_breadth` → Attention allocator
4. **Wire**: Energy consumption → Actual plugin ATP usage

### Status: ⚠️ **Complete but operating in isolation**

---

## 7. VAE Translation Layer

### Locations
- `/home/dp/ai-workspace/HRM/sage/compression/h_to_l_compressor.py`
- `/home/dp/ai-workspace/HRM/sage/irp/plugins/tinyvae_irp_plugin.py`
- `/home/dp/ai-workspace/HRM/models/vision/tiny_vae_32.py`

### 7.1 TinyVAE IRP Plugin

```python
class TinyVAEIRP(IRPPlugin):
    def __init__(self, config=None):
        # latent_dim=64, input_channels=3
        # Jetson-optimized: depthwise separable convs

    def refine(self, x: Any, early_stop: bool = True) -> Tuple[Any, Dict[str, Any]]:
        # Returns: (latent_64d, telemetry)
        # Single-pass VAE encode (no iterative refinement)
```

**Architecture**:
- Input: 64×64 RGB image
- Latent: 64D vector
- Compression: 192× (12,288 → 64)
- Parameters: ~294K (Jetson-friendly)
- Uses depthwise separable convolutions

### 7.2 H→L Context Compressor

```python
class HToLCompressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 4096,    # H-module context
        output_dim: int = 256,     # L-module actions
        compression_type: str = "hybrid"
    )

    def forward(self, context: torch.Tensor, return_metrics: bool = False) -> Dict
    def compress(self, context: torch.Tensor) -> torch.Tensor
    def decompress(self, compressed: torch.Tensor) -> torch.Tensor
```

**Compression Strategies**:
1. **Information Bottleneck** - Variational compression with KL divergence
2. **Attention Compressor** - Cross-attention with learnable latent codes
3. **Hierarchical Compressor** - Different ratios for sensory/semantic/physical/temporal
4. **Hybrid** - Combines all three

**Compression Ratio**: 16× (4096D → 256D)

### Capabilities
- ✅ Multiple compression strategies
- ✅ Reconstruction decoder for quality measurement
- ✅ Metrics: reconstruction loss, information retained, sparsity, mutual information
- ✅ Jetson-optimized TinyVAE for vision
- ✅ Cross-modal translation via shared latent spaces

### Integration Requirements
**For TinyVAE**:
- Already integrated as IRP plugin ✅
- Just needs registration in ResourceManager

**For H→L Compressor**:
- Not directly used in vision document loop
- Relevant for H↔L module communication in SAGECore
- Not critical for initial minimal loop

### Status: ✅ **Ready for use**

---

## 8. SAGE Core (H-L Modules)

### Location
`/home/dp/ai-workspace/HRM/sage/core/sage_core.py`

### Architecture
```python
class SAGECore(nn.Module):
    def __init__(self, config: SAGEConfig):
        self.h_module = HModule(config)          # ~45M params
        self.l_module = LModule(config)          # ~45M params
        self.communication = BidirectionalCommunication(config)  # ~10M params
        self.halt_predictor = ...
        self.resource_router = ...

    def forward(
        self, input_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_cycles: Optional[int] = None
    ) -> Dict[str, torch.Tensor]
```

### Capabilities
- ✅ 100M parameter H↔L architecture
- ✅ Iterative reasoning cycles
- ✅ Bidirectional communication
- ✅ Halt prediction
- ✅ Resource routing head
- ✅ Strategic ↔ Tactical coordination

### **CRITICAL FINDING: Wrong Abstraction**

This is a **trainable neural model** for abstract reasoning (like ARC-AGI), NOT the runtime orchestrator described in the vision document.

**Vision document says**:
> "SAGE is not a model - it's a loop"
> "SAGE = The kernel (scheduler, resource manager, learner)"
> "IRP = The API (standard interface for plugins)"

**SAGECore is**:
- A transformer-based reasoning model
- Takes token IDs and produces action logits
- Designed for training on ARC-AGI style tasks
- NOT a resource orchestrator

### What We Actually Need
```python
class SAGESystem:  # NOT SAGECore
    def __init__(self, config):
        self.sensor_hub = SensorHub(config.sensors)
        self.snarc_scorer = SNARCScorer()
        self.attention_allocator = AttentionAllocator()
        self.resource_planner = ResourcePlanner()
        self.resource_manager = ResourceManager(config.plugins)
        self.orchestrator = HRMOrchestrator()
        self.memory_system = MemorySystem()
        self.trust_tracker = TrustTracker()
        self.effector_hub = EffectorHub(config.effectors)
        self.metabolic_controller = MetabolicController()

    def run(self):
        while True:
            self._cycle()
```

### Integration Status
**DO NOT USE SAGECore for runtime loop** - It's a reasoning model, not a scheduler.

### Status: ❌ **Exists but wrong abstraction for integration**

---

## CRITICAL GAPS - What's Missing

### 1. SAGESystem - The Unified Loop ❌❌❌
**File**: Does not exist
**Purpose**: Top-level orchestrator that runs continuous cycle
**Effort**: HIGH - This is the main integration work

```python
class SAGESystem:
    """The living system that coordinates everything"""

    def __init__(self, config): ...

    def _cycle(self):
        """Single cycle of cognition"""
        # 1. Sensing
        observations = self.sensor_hub.poll()

        # 2. Salience Evaluation
        salience = self.snarc_scorer.evaluate(observations, predictions)

        # 3. Attention Allocation
        targets = self.attention_allocator.allocate(salience, self.metabolic.state)

        # 4. Resource Planning
        required = self.resource_planner.plan(targets, self.resource_manager.active)

        # 5. Resource Loading
        self.resource_manager.update(required)

        # 6. IRP Plugin Execution
        results = self.orchestrator.run_cycle(targets, required, atp_budget=1000)

        # 7. Memory Update
        self.memory_system.update(observations, salience, results)

        # 8. Trust Update
        self.trust_tracker.update(results)

        # 9. Action Execution
        self.effector_hub.execute(actions)

        # 10. Metabolic State Update
        self.metabolic_controller.update(energy, fatigue, stress)
```

### 2. SensorHub ❌
**Purpose**: Unified sensor interface
**Effort**: MEDIUM

```python
class SensorHub:
    def __init__(self, sensor_configs):
        self.sensors = {
            name: self._load_sensor(cfg)
            for name, cfg in sensor_configs.items()
        }

    def poll(self) -> Dict[str, torch.Tensor]:
        return {
            name: sensor.read()
            for name, sensor in self.sensors.items()
        }
```

### 3. AttentionAllocator ❌
**Purpose**: Salience → priority mapping
**Effort**: LOW

```python
class AttentionAllocator:
    def allocate(self, salience: Dict, metabolic_state: MetabolicState):
        # Sort by salience
        # Apply metabolic state attention breadth
        # Return prioritized targets
        pass
```

### 4. ResourcePlanner ❌
**Purpose**: Targets → plugins decision
**Effort**: LOW

```python
class ResourcePlanner:
    def plan(self, targets, active_resources):
        # Determine which plugins needed
        # Check memory constraints
        # Return list of required plugins
        pass
```

### 5. ResourceManager ❌
**Purpose**: Dynamic plugin loading/unloading
**Effort**: MEDIUM

```python
class ResourceManager:
    def __init__(self, plugin_configs):
        self.available = {name: cfg for name, cfg in plugin_configs.items()}
        self.active = {}

    def update(self, required):
        # Load new plugins
        # Unload unused plugins
        # Track memory usage
        pass
```

### 6. TrustTracker ❌
**Purpose**: Behavior → trust scores
**Effort**: LOW

```python
class TrustTracker:
    def __init__(self):
        self.scores = {}

    def update(self, results):
        # Analyze energy trajectories
        # Compute monotonicity, efficiency
        # Update trust scores
        pass
```

### 7. EffectorHub ❌
**Purpose**: Unified effector interface (mirrors SensorHub)
**Effort**: MEDIUM

```python
class EffectorHub:
    def __init__(self, effector_configs):
        self.effectors = {
            name: self._load_effector(cfg)
            for name, cfg in effector_configs.items()
        }

    def execute(self, actions):
        for name, command in actions.items():
            self.effectors[name].execute(command)
```

### 8. Unified MemorySystem ❌
**Purpose**: Coordinate 4 parallel memory systems
**Effort**: MEDIUM

```python
class MemorySystem:
    def __init__(self):
        self.snarc_memory = SNARCMemory()
        self.irp_bridge = IRPMemoryBridge()
        self.circular_buffer = CircularBuffer()
        self.verbatim_storage = VerbatimStorage()

    def update(self, observations, salience, results):
        # Update all 4 subsystems
        pass
```

---

## Integration Roadmap

### Phase 1: Minimal Loop (2-3 days)
**Goal**: Get a single cycle running end-to-end

**Create**:
1. `SAGESystem` skeleton class
2. `SensorHub` with mock sensors (random tensors)
3. `AttentionAllocator` basic implementation
4. `ResourcePlanner` simple version (always load what's needed)
5. `ResourceManager` basic (no dynamic loading yet, just registry)

**Wire**:
1. SNARC Scorer with ObservationEncoder adapter
2. HRMOrchestrator with new `run_cycle()` method
3. Metabolic state as read-only (doesn't affect behavior yet)
4. Single IRP plugin (TinyVAE vision)

**Success Criteria**:
- Loop executes continuously without crashing
- Sensors → SNARC → Attention → Plugins → Console output
- Can see cycle telemetry printed

### Phase 2: Dynamic Resource Management (2-3 days)
**Goal**: Plugins load/unload on demand

**Enhance**:
1. ResourceManager with lazy plugin loading
2. ResourcePlanner with memory constraint checking
3. Multiple IRP plugins available
4. Plugin discovery/registration system

**Wire**:
1. ResourceManager ↔ HRMOrchestrator
2. Memory tracking (GPU/CPU usage)
3. Plugin lifecycle management

**Success Criteria**:
- Plugins load only when needed
- Plugins unload when unused
- Memory stays within 8GB limit (Jetson target)

### Phase 3: Memory & Trust (2 days)
**Goal**: System learns and adapts

**Create**:
1. Unified `MemorySystem`
2. `TrustTracker` implementation
3. `CircularBuffer` standalone
4. `VerbatimStorage` with SQLite

**Wire**:
1. Memory system receives cycle data
2. Trust scores update after each plugin execution
3. ATP allocation responds to trust
4. Memory guides future refinements

**Success Criteria**:
- Trust scores change over time
- ATP budget adapts to performance
- Memory influences decisions (guidance retrieval)

### Phase 4: Real Sensors/Effectors (3-4 days)
**Goal**: Connect to hardware

**Create**:
1. Real camera sensor adapter
2. Real TTS effector (NeuTTS Air already exists)
3. Clock/temporal sensor
4. `EffectorHub` implementation

**Wire**:
1. Camera → ObservationEncoder → SNARC
2. Plugin results → EffectorHub → TTS
3. Test on development machine first
4. Deploy to Jetson

**Success Criteria**:
- SAGE processes real camera frames
- SAGE generates speech output
- System runs continuously for >1 hour

### Phase 5: Metabolic Integration (1-2 days)
**Goal**: Adaptive behavior through states

**Wire**:
1. Metabolic state config → Orchestrator ATP budget
2. Attention breadth → AttentionAllocator
3. Energy consumption → Actual plugin usage
4. State transitions affect system behavior

**Success Criteria**:
- FOCUS state narrows attention (1 target)
- WAKE state broadens attention (5 targets)
- REST state reduces ATP budget
- Automatic transitions based on energy/surprise

---

## Component Dependency Graph

```
SAGESystem (MISSING - CRITICAL)
├── SensorHub (MISSING) → Observations
│   └── ObservationEncoder (MISSING) → Hidden States
├── SNARCScorer (EXISTS ⚠️) → Salience Scores
│   └── Memory Bank (EXISTS ✅)
├── AttentionAllocator (MISSING) → Attention Targets
├── ResourcePlanner (MISSING) → Required Plugins
├── ResourceManager (MISSING) → Plugin Lifecycle
├── HRMOrchestrator (EXISTS ⚠️) → Plugin Execution
│   ├── ATPBudget (EXISTS ✅)
│   ├── IRP Plugins (15+ EXISTS ✅)
│   └── Trust Weights (EXISTS ✅)
├── MemorySystem (PARTIAL)
│   ├── SNARC Memory (MISSING)
│   ├── IRP Bridge (EXISTS ✅)
│   ├── Circular Buffer (MISSING)
│   └── Verbatim Storage (MISSING)
├── TrustTracker (MISSING)
├── EffectorHub (MISSING)
└── MetabolicController (EXISTS ⚠️ standalone)
```

**Legend**:
- ✅ = Ready to use
- ⚠️ = Exists but needs wiring/adaptation
- ❌ = Missing completely

---

## Key Insights

### 1. The Components Are Good
The IRP framework, plugins, orchestrator, and metabolic states are well-designed and largely complete. No major rewrites needed.

### 2. The Integration Is Missing
The gap is not in component quality but in the orchestration layer that connects them. We need the "kernel" that makes this an operating system.

### 3. SAGECore Is a Red Herring
The 100M parameter model in `/sage/core/sage_core.py` is NOT what we need for runtime. It's a trainable reasoning model, not a runtime scheduler. Don't confuse the two.

### 4. The Vision Is Clear
`SAGE_WORKING_VISION.md` describes exactly what needs to be built. Use it as the specification.

### 5. This Is Achievable
Most components exist. The missing pieces are straightforward Python classes (not complex ML models). 10-15 days of focused integration work.

---

## Recommended Next Steps

1. **Start with Phase 1** - Get a single cycle running with mock sensors
2. **Use vision document as spec** - Every line of that document should map to working code
3. **Don't create new components** - Wire what exists first
4. **Test incrementally** - Each phase should produce runnable demos
5. **Target Jetson from day 1** - Keep memory/compute constraints in mind

---

## File Manifest

### Existing Components (Ready)
- `/sage/irp/base.py` - IRP framework ✅
- `/sage/irp/plugins/tinyvae_irp_plugin.py` - Vision plugin ✅
- `/sage/irp/plugins/neutts_air_impl.py` - TTS plugin ✅
- `/sage/orchestrator/hrm_orchestrator.py` - Orchestrator ⚠️
- `/sage/memory/irp_memory_bridge.py` - Memory bridge ✅
- `/sage/attention/snarc_scorer.py` - SNARC ⚠️
- `/sage/orchestration/agents/control/metabolic-state-manager.py` - Metabolic ⚠️

### To Be Created
- `/sage/core/sage_system.py` - Main integration ❌
- `/sage/sensors/sensor_hub.py` - Sensor interface ❌
- `/sage/sensors/observation_encoder.py` - Multi-modal encoder ❌
- `/sage/attention/attention_allocator.py` - Priority mapping ❌
- `/sage/resources/resource_planner.py` - Plugin selection ❌
- `/sage/resources/resource_manager.py` - Dynamic loading ❌
- `/sage/trust/trust_tracker.py` - Behavior analysis ❌
- `/sage/effectors/effector_hub.py` - Effector interface ❌
- `/sage/memory/memory_system.py` - Unified memory ❌
- `/sage/memory/circular_buffer.py` - Circular buffer ❌
- `/sage/memory/verbatim_storage.py` - SQLite storage ❌

---

**End of Component Readiness Map**

*The parts exist. Now we build the assembly.*
