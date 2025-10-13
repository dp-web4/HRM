# SAGE Integration Architecture

**Date**: October 12, 2025
**Purpose**: Detailed architectural blueprint for SAGESystem integration

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SAGESystem                                      │
│                         (Main Orchestration Loop)                           │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         Continuous Cycle                                │ │
│  │                                                                          │ │
│  │  1. Sensing → 2. Salience → 3. Attention → 4. Planning → 5. Loading   │ │
│  │     ↓            ↓            ↓              ↓             ↓            │ │
│  │  6. Execution → 7. Memory → 8. Trust → 9. Actions → 10. Metabolic     │ │
│  │                                                              ↓            │ │
│  │                                                           Loop →        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Layer 1: Input (Sensing)
```
┌──────────────────────────────────────────────────────────────┐
│                       SensorHub                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Sensor Registry                                      │   │
│  │  - camera: CameraSensor                              │   │
│  │  - audio: AudioSensor                                │   │
│  │  - clock: TemporalSensor                             │   │
│  │  - proprioception: ProprioceptionSensor (future)    │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  poll() → Dict[str, Tensor]                                 │
│  {                                                           │
│    'vision': Tensor[3, 480, 640],                           │
│    'audio': Tensor[16000],                                  │
│    'clock': float                                           │
│  }                                                           │
└──────────────────────────────────────────────────────────────┘
```

### Layer 2: Encoding
```
┌──────────────────────────────────────────────────────────────┐
│                   ObservationEncoder                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Modality-Specific Encoders                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │   Vision    │  │    Audio    │  │   Temporal  │ │   │
│  │  │   Encoder   │  │   Encoder   │  │   Encoder   │ │   │
│  │  │   (CNN)     │  │   (Conv1D)  │  │  (Embed)    │ │   │
│  │  │  → 768D     │  │   → 768D    │  │   → 768D    │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  encode(observations) → Tensor[batch, modalities, 768]      │
└──────────────────────────────────────────────────────────────┘
```

### Layer 3: Salience Evaluation
```
┌──────────────────────────────────────────────────────────────┐
│                      SNARCScorer                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  5D SNARC Components                                  │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │   │
│  │  │Surprise │  │ Novelty │  │ Arousal │             │   │
│  │  │  Net    │  │   Net   │  │   Net   │             │   │
│  │  └─────────┘  └─────────┘  └─────────┘             │   │
│  │  ┌─────────┐  ┌─────────┐                           │   │
│  │  │ Reward  │  │Conflict │                           │   │
│  │  │   Net   │  │   Net   │                           │   │
│  │  └─────────┘  └─────────┘                           │   │
│  │                                                       │   │
│  │  Memory Bank (1000 states)                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  forward() → {                                               │
│    'snarc_scores': Tensor[batch, modalities, 1],            │
│    'attention_weights': Tensor[batch, modalities, 1],       │
│    'surprise': ..., 'novelty': ..., etc.                    │
│  }                                                           │
└──────────────────────────────────────────────────────────────┘
```

### Layer 4: Attention Allocation
```
┌──────────────────────────────────────────────────────────────┐
│                   AttentionAllocator                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Inputs:                                              │   │
│  │  - salience_scores: Dict[modality, float]           │   │
│  │  - metabolic_state: MetabolicState                   │   │
│  │  - state_config: StateConfig                         │   │
│  │                                                       │   │
│  │  Logic:                                               │   │
│  │  1. Sort modalities by salience                      │   │
│  │  2. Take top-K (K = attention_breadth from state)   │   │
│  │  3. Compute priority weights                         │   │
│  │  4. Create AttentionTarget for each                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  allocate() → List[AttentionTarget]                         │
│  [                                                           │
│    AttentionTarget(                                          │
│      modality='vision',                                      │
│      priority=0.7,                                           │
│      data=observation_data                                   │
│    ),                                                        │
│    ...                                                       │
│  ]                                                           │
└──────────────────────────────────────────────────────────────┘
```

### Layer 5: Resource Planning
```
┌──────────────────────────────────────────────────────────────┐
│                    ResourcePlanner                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Inputs:                                              │   │
│  │  - attention_targets: List[AttentionTarget]         │   │
│  │  - active_resources: Dict[plugin_id, Plugin]        │   │
│  │  - available_resources: Dict[plugin_id, Config]     │   │
│  │                                                       │   │
│  │  Logic:                                               │   │
│  │  1. Map targets to required plugin types            │   │
│  │     vision → vision_irp                              │   │
│  │     audio → audio_irp                                │   │
│  │  2. Check memory constraints (8GB Jetson)           │   │
│  │  3. Prioritize by attention weights                  │   │
│  │  4. Decide keep/load/unload                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  plan() → ResourcePlan                                       │
│  {                                                           │
│    'keep': ['vision_irp', 'memory_irp'],                   │
│    'load': ['audio_irp'],                                   │
│    'unload': ['language_irp']                               │
│  }                                                           │
└──────────────────────────────────────────────────────────────┘
```

### Layer 6: Resource Management
```
┌──────────────────────────────────────────────────────────────┐
│                   ResourceManager                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Plugin Registry                                      │   │
│  │  available: Dict[plugin_id, PluginConfig]           │   │
│  │  active: Dict[plugin_id, IRPPlugin]                 │   │
│  │                                                       │   │
│  │  Memory Tracker                                       │   │
│  │  - current_usage: 2.3 GB                             │   │
│  │  - limit: 8.0 GB                                     │   │
│  │  - per_plugin_usage: Dict[plugin_id, float]         │   │
│  │                                                       │   │
│  │  Lifecycle Management                                 │   │
│  │  ┌───────┐  ┌─────┐  ┌─────────┐  ┌──────────┐    │   │
│  │  │ Load  │→ │Init │→ │Register │→ │Track Mem │    │   │
│  │  └───────┘  └─────┘  └─────────┘  └──────────┘    │   │
│  │  ┌─────────┐  ┌──────┐  ┌────────┐                │   │
│  │  │ Unload  │← │Cleanup│← │Save State│               │   │
│  │  └─────────┘  └──────┘  └────────┘                │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  update(plan) → Success/Failure                              │
└──────────────────────────────────────────────────────────────┘
```

### Layer 7: Plugin Orchestration
```
┌──────────────────────────────────────────────────────────────┐
│                    HRMOrchestrator                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ATP Budget                                           │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │ total: 1000.0                                │    │   │
│  │  │ allocated: {vision: 600, memory: 400}       │    │   │
│  │  │ consumed: {vision: 450, memory: 200}        │    │   │
│  │  │ trust_weights: {vision: 0.85, memory: 0.72}│    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │                                                       │   │
│  │  Plugin Execution (Parallel)                         │   │
│  │  ┌───────────────┐  ┌───────────────┐              │   │
│  │  │  Vision IRP   │  │  Memory IRP   │              │   │
│  │  │  ───────────  │  │  ───────────  │              │   │
│  │  │  init_state() │  │  init_state() │              │   │
│  │  │  step(0)      │  │  step(0)      │              │   │
│  │  │  energy()     │  │  halt() ✓     │              │   │
│  │  │  step(1)      │  │               │              │   │
│  │  │  ...          │  │  Result:      │              │   │
│  │  │  halt() ✓     │  │  - None       │              │   │
│  │  │               │  │  - 1 iter     │              │   │
│  │  │  Result:      │  │  - 100 ATP    │              │   │
│  │  │  - latent_64d │  │               │              │   │
│  │  │  - 6 iters    │  │               │              │   │
│  │  │  - 600 ATP    │  │               │              │   │
│  │  └───────────────┘  └───────────────┘              │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  run_cycle() → List[PluginResult]                           │
└──────────────────────────────────────────────────────────────┘
```

### Layer 8: Memory System
```
┌──────────────────────────────────────────────────────────────┐
│                     MemorySystem                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Four Parallel Subsystems                             │   │
│  │                                                       │   │
│  │  1. SNARC Memory (Selective)                         │   │
│  │     ┌──────────────────────────────────────┐        │   │
│  │     │ If salience > threshold:             │        │   │
│  │     │   store(key, value, salience)        │        │   │
│  │     │ Priority queue by salience           │        │   │
│  │     └──────────────────────────────────────┘        │   │
│  │                                                       │   │
│  │  2. IRP Memory Bridge (Pattern Library) ✅          │   │
│  │     ┌──────────────────────────────────────┐        │   │
│  │     │ Store refinement trajectories        │        │   │
│  │     │ Extract patterns during consolidation│        │   │
│  │     │ Retrieve guidance for similar tasks  │        │   │
│  │     └──────────────────────────────────────┘        │   │
│  │                                                       │   │
│  │  3. Circular Buffer (Recent Context)                │   │
│  │     ┌──────────────────────────────────────┐        │   │
│  │     │ Fixed-size ring buffer (100 cycles) │        │   │
│  │     │ X-from-last retrieval                │        │   │
│  │     │ Temporal context window              │        │   │
│  │     └──────────────────────────────────────┘        │   │
│  │                                                       │   │
│  │  4. Verbatim Storage (Full Records)                 │   │
│  │     ┌──────────────────────────────────────┐        │   │
│  │     │ SQLite database                       │        │   │
│  │     │ Full cycle records with blobs         │        │   │
│  │     │ Query by timestamp/cycle_id           │        │   │
│  │     └──────────────────────────────────────┘        │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  update(observations, salience, results) → void             │
└──────────────────────────────────────────────────────────────┘
```

### Layer 9: Trust Tracking
```
┌──────────────────────────────────────────────────────────────┐
│                     TrustTracker                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Per-Plugin Trust Scores                              │   │
│  │  scores: Dict[plugin_id, TrustScore]                 │   │
│  │                                                       │   │
│  │  Trust Metrics                                        │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │ Monotonicity:                                │    │   │
│  │  │   energy[i+1] < energy[i] for all i?       │    │   │
│  │  │   → monotonicity_ratio: 0.0-1.0             │    │   │
│  │  │                                              │    │   │
│  │  │ Efficiency:                                  │    │   │
│  │  │   iterations / max_iterations                │    │   │
│  │  │   → lower is better                          │    │   │
│  │  │                                              │    │   │
│  │  │ Stability:                                   │    │   │
│  │  │   variance(energy_deltas)                    │    │   │
│  │  │   → lower is better                          │    │   │
│  │  │                                              │    │   │
│  │  │ Convergence:                                 │    │   │
│  │  │   total_energy_decrease / iterations         │    │   │
│  │  │   → higher is better                         │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  │                                                       │   │
│  │  Trust Score Calculation                             │   │
│  │  trust = (0.4 * monotonicity +                      │   │
│  │           0.3 * (1 - efficiency) +                   │   │
│  │           0.2 * (1 - normalized_stability) +         │   │
│  │           0.1 * normalized_convergence)              │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  update(results) → Dict[plugin_id, float]                   │
└──────────────────────────────────────────────────────────────┘
```

### Layer 10: Action Execution
```
┌──────────────────────────────────────────────────────────────┐
│                     EffectorHub                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Effector Registry                                    │   │
│  │  - speech: TTSEffector (NeuTTS Air)                  │   │
│  │  - display: VisualMonitorEffector                    │   │
│  │  - motor: MotorEffector (future)                     │   │
│  │  - log: LogEffector                                  │   │
│  │                                                       │   │
│  │  Action Queue                                         │   │
│  │  ┌──────────────────────────────────────┐           │   │
│  │  │ [{type: 'speech', text: '...'},     │           │   │
│  │  │  {type: 'display', image: tensor}]  │           │   │
│  │  └──────────────────────────────────────┘           │   │
│  │                                                       │   │
│  │  Execution                                            │   │
│  │  For each action:                                     │   │
│  │    effector = registry[action.type]                  │   │
│  │    effector.execute(action.data)                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  execute(actions) → List[ExecutionResult]                   │
└──────────────────────────────────────────────────────────────┘
```

### Layer 11: Metabolic Control
```
┌──────────────────────────────────────────────────────────────┐
│                  MetabolicController                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Current State: WAKE                                  │   │
│  │  ┌────────────────────────────────────────────┐     │   │
│  │  │ Config:                                     │     │   │
│  │  │ - energy_consumption_rate: 10.0            │     │   │
│  │  │ - attention_breadth: 5                     │     │   │
│  │  │ - surprise_sensitivity: 1.0                │     │   │
│  │  │ - exploration_rate: 0.1                    │     │   │
│  │  │ - max_duration: inf                        │     │   │
│  │  └────────────────────────────────────────────┘     │   │
│  │                                                       │   │
│  │  State Machine                                        │   │
│  │  ┌────────┐  high perf  ┌────────┐                 │   │
│  │  │  WAKE  │────────────→│ FOCUS  │                 │   │
│  │  └────────┘             └────────┘                 │   │
│  │      ↓ low energy           ↓ exhausted            │   │
│  │  ┌────────┐  recharged  ┌────────┐                │   │
│  │  │  REST  │←────────────│ CRISIS │                 │   │
│  │  └────────┘             └────────┘                 │   │
│  │      ↓ idle                                          │   │
│  │  ┌────────┐                                         │   │
│  │  │ DREAM  │ (consolidation)                         │   │
│  │  └────────┘                                         │   │
│  │                                                       │   │
│  │  Energy Manager                                       │   │
│  │  current: 94%, max: 100%, rate: 5.0/sec             │   │
│  │  consumption_history: [(time, amount), ...]         │   │
│  └──────────────────────────────────────────────────────┘   │
│                           ↓                                   │
│  get_state_config() → StateConfig                           │
│  update(energy, fatigue, stress) → Optional[Transition]    │
└──────────────────────────────────────────────────────────────┘
```

---

## Data Flow Architecture

### Single Cycle Data Flow
```
Raw Sensors
    ↓
Observations Dict
    ↓
[ObservationEncoder]
    ↓
Hidden States Tensor [batch, modalities, 768]
    ↓
[SNARCScorer]
    ↓
Salience Scores Dict {modality: score}
    ↓
[AttentionAllocator + MetabolicState]
    ↓
Attention Targets List[AttentionTarget]
    ↓
[ResourcePlanner]
    ↓
Resource Plan {keep, load, unload}
    ↓
[ResourceManager]
    ↓
Active Plugins Dict[plugin_id, Plugin]
    ↓
[HRMOrchestrator + ATP Budget]
    ↓
Plugin Results List[PluginResult]
    ↓
[MemorySystem] [TrustTracker]
    ↓              ↓
Updated Memory  Updated Trust Scores
    ↓              ↓
[Action Computation]
    ↓
Actions List[Action]
    ↓
[EffectorHub]
    ↓
External Effects (speech, display, etc.)
    ↓
[MetabolicController]
    ↓
Updated Metabolic State
    ↓
[Loop continues]
```

---

## Class Hierarchy

```
SAGESystem
├── SensorHub
│   ├── CameraSensor
│   ├── AudioSensor
│   └── TemporalSensor
├── ObservationEncoder
│   ├── VisionEncoder
│   ├── AudioEncoder
│   └── TemporalEncoder
├── SNARCScorer (nn.Module) ✅
│   ├── surprise_net
│   ├── novelty_net
│   ├── arousal_net
│   ├── conflict_net
│   └── memory_bank (deque)
├── AttentionAllocator
├── ResourcePlanner
├── ResourceManager
│   └── active: Dict[str, IRPPlugin]
├── HRMOrchestrator ✅
│   ├── ATPBudget
│   └── plugins: Dict[str, IRPPlugin]
├── MemorySystem
│   ├── SNARCMemory
│   ├── IRPMemoryBridge ✅
│   ├── CircularBuffer
│   └── VerbatimStorage
├── TrustTracker
├── EffectorHub
│   ├── TTSEffector
│   ├── VisualMonitorEffector
│   └── LogEffector
└── MetabolicController ✅
    ├── MetabolicState (Enum)
    ├── StateConfig
    └── EnergyManager
```

**Legend**:
- ✅ = Already implemented
- (no mark) = Needs implementation

---

## Interface Contracts

### SensorHub
```python
class SensorHub:
    def poll(self) -> Dict[str, Union[torch.Tensor, float]]:
        """Poll all sensors, return observations"""

class Sensor(ABC):
    @abstractmethod
    def read(self) -> Union[torch.Tensor, float]:
        """Read current sensor value"""
```

### ObservationEncoder
```python
class ObservationEncoder(nn.Module):
    def encode(self, observations: Dict[str, Any]) -> torch.Tensor:
        """
        Convert observations to hidden states
        Returns: [batch, num_modalities, hidden_size]
        """
```

### AttentionAllocator
```python
@dataclass
class AttentionTarget:
    modality: str
    priority: float  # 0.0-1.0
    data: Any

class AttentionAllocator:
    def allocate(
        self,
        salience_scores: torch.Tensor,
        attention_breadth: int
    ) -> List[AttentionTarget]:
        """Allocate attention based on salience and state"""
```

### ResourcePlanner
```python
@dataclass
class ResourcePlan:
    keep: List[str]    # Plugin IDs to keep loaded
    load: List[str]    # Plugin IDs to load
    unload: List[str]  # Plugin IDs to unload

class ResourcePlanner:
    def plan(
        self,
        targets: List[AttentionTarget],
        active: Dict[str, IRPPlugin]
    ) -> ResourcePlan:
        """Determine resource requirements"""
```

### ResourceManager
```python
class ResourceManager:
    def update(self, plan: ResourcePlan) -> bool:
        """Execute resource plan, return success"""

    def get_active(self) -> Dict[str, IRPPlugin]:
        """Get currently loaded plugins"""

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
```

### TrustTracker
```python
@dataclass
class TrustMetrics:
    monotonicity: float
    efficiency: float
    stability: float
    convergence: float
    overall: float

class TrustTracker:
    def update(self, results: List[PluginResult]) -> Dict[str, float]:
        """Update trust scores, return new scores"""

    def get_metrics(self, plugin_id: str) -> TrustMetrics:
        """Get detailed metrics for plugin"""
```

### EffectorHub
```python
@dataclass
class Action:
    type: str  # 'speech', 'display', 'log', etc.
    data: Any

class EffectorHub:
    def execute(self, actions: List[Action]) -> List[ExecutionResult]:
        """Execute actions through registered effectors"""

class Effector(ABC):
    @abstractmethod
    def execute(self, data: Any) -> ExecutionResult:
        """Execute action"""
```

---

## Configuration Schema

```python
@dataclass
class SAGEConfig:
    # Sensors
    sensors: Dict[str, SensorConfig] = field(default_factory=lambda: {
        'camera': {'type': 'camera', 'device': 0, 'resolution': (640, 480)},
        'audio': {'type': 'audio', 'sample_rate': 16000, 'channels': 1},
        'clock': {'type': 'temporal', 'format': 'unix'}
    })

    # SNARC
    snarc_hidden_size: int = 768
    snarc_memory_size: int = 1000

    # Attention
    max_attention_breadth: int = 10

    # Resources
    max_memory_gb: float = 8.0
    plugin_configs: Dict[str, PluginConfig] = field(default_factory=lambda: {
        'vision': {'type': 'tinyvae_irp', 'device': 'cuda'},
        'audio': {'type': 'audio_irp', 'device': 'cuda'},
        'language': {'type': 'language_irp', 'device': 'cuda'},
        'memory': {'type': 'memory_irp', 'device': 'cpu'}
    })

    # Orchestrator
    initial_atp: float = 1000.0
    max_concurrent_plugins: int = 4

    # Memory
    memory_buffer_size: int = 100
    snarc_capacity: int = 1000
    consolidation_threshold: int = 50
    verbatim_db_path: str = './sage_memory.db'

    # Metabolic
    initial_energy: float = 100.0
    recharge_rate: float = 5.0

    # Effectors
    effectors: Dict[str, EffectorConfig] = field(default_factory=lambda: {
        'speech': {'type': 'neutts_air', 'device': 'cpu'},
        'display': {'type': 'visual_monitor', 'device': 'cuda'},
        'log': {'type': 'console', 'verbose': True}
    })

    # Cycle
    target_cycle_time_ms: float = 500.0  # 2 Hz target
    enable_async: bool = True
```

---

## Testing Architecture

### Unit Test Structure
```
tests/
├── unit/
│   ├── test_sensor_hub.py
│   ├── test_observation_encoder.py
│   ├── test_snarc_scorer.py
│   ├── test_attention_allocator.py
│   ├── test_resource_planner.py
│   ├── test_resource_manager.py
│   ├── test_orchestrator.py
│   ├── test_memory_system.py
│   ├── test_trust_tracker.py
│   ├── test_effector_hub.py
│   └── test_metabolic_controller.py
├── integration/
│   ├── test_single_cycle.py
│   ├── test_multi_cycle.py
│   ├── test_plugin_lifecycle.py
│   ├── test_memory_persistence.py
│   └── test_state_transitions.py
├── hardware/
│   ├── test_camera_sensor.py
│   ├── test_tts_effector.py
│   └── test_jetson_deployment.py
└── performance/
    ├── test_cycle_timing.py
    ├── test_memory_usage.py
    └── test_stability.py
```

---

## Deployment Architecture

### Development Machine
```
Ubuntu 22.04 / WSL2
├── Python 3.12
├── PyTorch 2.3.0 + CUDA 12.1
├── 16GB+ RAM
└── NVIDIA GPU (any)
```

### Jetson Orin Nano (Target)
```
JetPack 6.0
├── Python 3.10
├── PyTorch 2.3.0 (Jetson build)
├── 8GB Unified Memory
├── CUDA 12.2
└── 1024-core Ampere GPU
```

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Cycle Time | < 500ms | Time per `_cycle()` execution |
| Memory Usage | < 7GB | Peak RAM on Jetson |
| Plugin Load Time | < 100ms | Time to load single plugin |
| Attention Latency | < 10ms | SNARC + Allocation |
| Trust Update | < 5ms | Per plugin trust computation |
| Memory Write | < 20ms | Full memory system update |
| Stability | > 1 hour | Continuous operation without crash |

---

**This architecture enables SAGE to operate as a living, adaptive system on edge devices.**
