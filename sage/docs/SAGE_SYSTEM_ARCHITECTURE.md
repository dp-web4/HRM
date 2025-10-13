# SAGE System Architecture Design

**Date**: October 12, 2025
**Purpose**: Complete architectural design for unified SAGESystem integration
**Status**: Design Phase - Ready for Implementation

---

## Executive Summary

This document describes the complete architecture for integrating all SAGE components into a unified consciousness kernel. SAGE is **not a model** - it's a **consciousness kernel for edge devices** that orchestrates specialized reasoning plugins through a continuous loop.

**Key Principle**: SAGE decides *what* to think about and *which* tools to use, not *how* to solve problems.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Interface Contracts](#interface-contracts)
5. [Main Loop Design](#main-loop-design)
6. [Error Handling Strategy](#error-handling-strategy)
7. [Testing Approach](#testing-approach)
8. [Implementation Phases](#implementation-phases)
9. [Performance Considerations](#performance-considerations)

---

## System Overview

### What SAGE Is

```
SAGE = Sentient Agentic Generative Engine

Components:
  - SAGE (Core):    Consciousness kernel - scheduler, resource manager, learner
  - IRP (API):      Standard interface for plugins ("apps" for consciousness)
  - VAE (Bridge):   Translation layer for cross-modal communication
```

### The Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SAGE - Consciousness Kernel           │
│  (Continuous loop, trust-based ATP budget, metabolic)   │
├─────────────────────────────────────────────────────────┤
│                    IRP - Plugin Interface                │
│  (init_state → step → energy → halt)                    │
│  Vision | Audio | Language | Memory | Control | TTS     │
├─────────────────────────────────────────────────────────┤
│                    VAE - Translation Layer               │
│  (Shared latent spaces for cross-modal communication)   │
│  TinyVAE | InfoBottleneck | PuzzleSpace                 │
└─────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Clean Separation of Concerns**: Each component has one job
2. **Single Responsibility**: Components don't overlap
3. **Unidirectional Data Flow**: State flows through loop in one direction
4. **Explicit State**: All state is in data structures, not hidden in objects
5. **Independent Testing**: Components can be tested in isolation
6. **Swappable Implementations**: Easy to replace any component

---

## Component Architecture

### Class Diagram (ASCII Art)

```
                           ┌──────────────┐
                           │  SAGESystem  │
                           │ (Main Loop)  │
                           └──────┬───────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
         ┌──────▼──────┐   ┌─────▼──────┐   ┌─────▼──────┐
         │ SensorHub   │   │  SNARC     │   │ Attention  │
         │ (ISensorHub)│   │  Scorer    │   │ Allocator  │
         │             │   │(ISNARCScorer)   │(IAttention)│
         └─────────────┘   └────────────┘   └────────────┘
                │                 │                 │
                └─────────────────┼─────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
         ┌──────▼──────┐   ┌─────▼──────┐   ┌─────▼──────┐
         │ Resource    │   │ Resource   │   │    HRM     │
         │ Planner     │   │ Manager    │   │Orchestrator│
         │(IPlanner)   │   │(IManager)  │   │(IOrch)     │
         └─────────────┘   └────────────┘   └────────────┘
                │                 │                 │
                └─────────────────┼─────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
         ┌──────▼──────┐   ┌─────▼──────┐   ┌─────▼──────┐
         │   Memory    │   │   Trust    │   │  Effector  │
         │   System    │   │  Tracker   │   │    Hub     │
         │ (IMemory)   │   │ (ITrust)   │   │(IEffector) │
         └─────────────┘   └────────────┘   └────────────┘
                                  │
                           ┌──────▼───────┐
                           │  Metabolic   │
                           │ Controller   │
                           │(IMetabolic)  │
                           └──────────────┘
```

### Component Responsibilities

| Component | Responsibility | Input | Output |
|-----------|---------------|-------|--------|
| **SensorHub** | Poll all sensors | Config | Observations |
| **SNARCScorer** | Evaluate salience | Observations | Salience scores |
| **AttentionAllocator** | Prioritize attention | Salience + State | Attention targets |
| **ResourcePlanner** | Decide plugins | Targets + Memory | Required plugins |
| **ResourceManager** | Load/unload plugins | Required list | Active plugins |
| **Orchestrator** | Execute IRP plugins | Targets + ATP | Plugin results |
| **MemorySystem** | Store/retrieve state | Cycle state | Context |
| **TrustTracker** | Score plugin behavior | Results | Trust scores |
| **EffectorHub** | Execute actions | Actions | Side effects |
| **MetabolicController** | Manage energy state | Energy metrics | State |

---

## Data Flow

### Data Structures

All data flows through explicit, documented structures:

```python
@dataclass
class Observation:
    """Raw sensor data"""
    modality: str           # 'vision', 'audio', 'clock', etc.
    data: Any              # Tensor, array, scalar
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class SalienceScore:
    """5D SNARC evaluation"""
    modality: str
    surprise: float        # Deviation from expected
    novelty: float         # Unseen patterns
    arousal: float         # Complexity/information density
    reward: float          # Task success signal
    conflict: float        # Ambiguity/uncertainty
    combined: float        # Weighted combination

@dataclass
class AttentionTarget:
    """Where to focus attention"""
    modality: str
    priority: float        # 0.0 to 1.0
    data: Any
    salience: SalienceScore
    metadata: Dict[str, Any]

@dataclass
class PluginResult:
    """Result from IRP execution"""
    plugin_id: str
    output: Any
    energy_trajectory: List[float]
    iterations: int
    atp_used: float
    converged: bool
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class CycleState:
    """Complete state for one cycle"""
    cycle_id: int
    timestamp: float
    observations: Dict[str, Observation]
    salience_scores: Dict[str, SalienceScore]
    attention_targets: List[AttentionTarget]
    required_plugins: List[str]
    plugin_results: Dict[str, PluginResult]
    actions: Dict[str, Any]
    metabolic_state: str
    energy_level: float
    atp_budget: float
    atp_remaining: float
```

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SAGE CONSCIOUSNESS LOOP                   │
└─────────────────────────────────────────────────────────────┘

Cycle Start
    │
    ├──► 1. SENSING (10ms)
    │    SensorHub.poll() → Dict[str, Observation]
    │    • Camera: 640×480 RGB frame
    │    • Audio: 1 second buffer
    │    • Clock: Unix timestamp
    │
    ├──► 2. SALIENCE EVALUATION (5ms)
    │    SNARCScorer.evaluate(observations) → Dict[str, SalienceScore]
    │    • Surprise: prediction error
    │    • Novelty: memory comparison
    │    • Arousal: entropy computation
    │    • Reward: task signal
    │    • Conflict: uncertainty measure
    │
    ├──► 3. ATTENTION ALLOCATION (2ms)
    │    AttentionAllocator.allocate(salience, state, budget) → List[AttentionTarget]
    │    • Sort by combined salience
    │    • Apply metabolic breadth (WAKE=70%, FOCUS=30%)
    │    • Distribute attention budget
    │
    ├──► 4. RESOURCE PLANNING (3ms)
    │    ResourcePlanner.plan(targets, active, memory) → List[str]
    │    • Map targets to plugins
    │    • Check memory constraints
    │    • Return required plugins
    │
    ├──► 5. RESOURCE LOADING (0-200ms)
    │    ResourceManager.update(required) → Dict[str, Plugin]
    │    • Load new plugins
    │    • Unload unused plugins
    │    • Track memory usage
    │
    ├──► 6. IRP PLUGIN EXECUTION (50-500ms)
    │    Orchestrator.run_cycle(targets, plugins, atp) → Dict[str, PluginResult]
    │    • For each target:
    │      - init_state(input)
    │      - while not halt():
    │          state = step(state)
    │          energy = energy(state)
    │      - return refined state
    │    • Track ATP consumption
    │    • Record energy trajectories
    │
    ├──► 7. ACTION EXECUTION (5ms)
    │    EffectorHub.execute(actions)
    │    • Extract action commands from results
    │    • Send to motor controllers
    │    • Invoke TTS for speech
    │    • Update display
    │
    ├──► 8. TRUST UPDATE (5ms)
    │    TrustTracker.update(results)
    │    • Analyze energy trajectories
    │    • Compute monotonicity ratio
    │    • Compute efficiency
    │    • Update trust scores
    │
    ├──► 9. MEMORY UPDATE (20ms)
    │    MemorySystem.update(cycle_state)
    │    • SNARC Memory: Store high-salience items
    │    • IRP Bridge: Store successful patterns
    │    • Circular Buffer: Add to context window
    │    • Verbatim: Log full cycle to SQLite
    │
    └──► 10. METABOLIC STATE UPDATE (2ms)
         MetabolicController.update(energy, fatigue, stress)
         • Consume energy (ATP spent / 10000)
         • Accumulate fatigue (+0.01 per cycle)
         • Adjust stress (errors +0.05, success -0.01)
         • Transition state if thresholds met

Cycle End (Total: 50-750ms depending on IRP execution)
    │
    └──► Loop back to Cycle Start
```

---

## Interface Contracts

### ISensorHub

```python
class ISensorHub:
    """Interface for sensor management"""

    def poll(self) -> Dict[str, Observation]:
        """
        Poll all active sensors and return observations.

        Returns:
            Dictionary mapping modality name to Observation
        """
        raise NotImplementedError

    def get_available_sensors(self) -> List[str]:
        """Get list of available sensor modalities"""
        raise NotImplementedError

    def enable_sensor(self, modality: str):
        """Enable a sensor"""
        raise NotImplementedError

    def disable_sensor(self, modality: str):
        """Disable a sensor"""
        raise NotImplementedError
```

**Contract**:
- Must not block for more than 50ms
- Must return valid Observation objects with timestamps
- Must handle sensor failures gracefully (return None or raise)

### ISNARCScorer

```python
class ISNARCScorer:
    """Interface for SNARC salience evaluation"""

    def evaluate(self, observations: Dict[str, Observation],
                 predictions: Optional[Dict[str, Any]] = None
                ) -> Dict[str, SalienceScore]:
        """
        Evaluate salience for all observations.

        Args:
            observations: Current observations from sensors
            predictions: Optional predictions for surprise computation

        Returns:
            Dictionary mapping modality to SalienceScore
        """
        raise NotImplementedError

    def update_predictions(self, observations: Dict[str, Observation]):
        """
        Update predictive models with new observations.
        """
        raise NotImplementedError
```

**Contract**:
- All five SNARC dimensions must be computed (can be zero)
- Combined score must be in range [0.0, 1.0]
- Must complete in <10ms per modality

### IAttentionAllocator

```python
class IAttentionAllocator:
    """Interface for attention allocation"""

    def allocate(self, salience_scores: Dict[str, SalienceScore],
                 metabolic_state: str,
                 atp_budget: float) -> List[AttentionTarget]:
        """
        Allocate attention based on salience and metabolic state.

        Args:
            salience_scores: SNARC scores per modality
            metabolic_state: Current metabolic state (WAKE/FOCUS/REST/DREAM/CRISIS)
            atp_budget: Available ATP for this cycle

        Returns:
            List of AttentionTargets sorted by priority
        """
        raise NotImplementedError
```

**Contract**:
- Sum of priorities should not exceed 1.0
- Targets must be sorted by priority (descending)
- Must respect metabolic state attention breadth

### IResourcePlanner

```python
class IResourcePlanner:
    """Interface for resource planning"""

    def plan(self, targets: List[AttentionTarget],
             active_resources: Dict[str, Any],
             memory_available: float) -> List[str]:
        """
        Determine required plugins for attention targets.

        Args:
            targets: Attention targets needing processing
            active_resources: Currently loaded plugins
            memory_available: Available memory in GB

        Returns:
            List of required plugin IDs
        """
        raise NotImplementedError
```

**Contract**:
- Must not exceed memory constraints
- Should minimize plugin loading/unloading
- Should prioritize high-priority targets

### IResourceManager

```python
class IResourceManager:
    """Interface for dynamic resource management"""

    def update(self, required_plugins: List[str]) -> Dict[str, Any]:
        """
        Load/unload plugins as needed, return active plugins.

        Args:
            required_plugins: List of plugin IDs to have loaded

        Returns:
            Dictionary of active plugin instances
        """
        raise NotImplementedError

    def get_active_plugins(self) -> Dict[str, Any]:
        """Get currently loaded plugins"""
        raise NotImplementedError

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        raise NotImplementedError
```

**Contract**:
- Must lazy-load plugins (import only when needed)
- Must track GPU/CPU memory usage
- Must gracefully handle loading failures

### IOrchestrator

```python
class IOrchestrator:
    """Interface for IRP plugin orchestration"""

    def run_cycle(self, targets: List[AttentionTarget],
                  active_plugins: Dict[str, Any],
                  atp_budget: float) -> Dict[str, PluginResult]:
        """
        Execute IRP plugins for attention targets.

        Args:
            targets: Attention targets to process
            active_plugins: Loaded plugin instances
            atp_budget: Total ATP budget for cycle

        Returns:
            Dictionary mapping target modality to PluginResult
        """
        raise NotImplementedError
```

**Contract**:
- Must respect ATP budget (stop early if exceeded)
- Must record complete energy trajectories
- Must handle plugin failures gracefully

### IMemorySystem

```python
class IMemorySystem:
    """Interface for memory management"""

    def update(self, cycle_state: CycleState):
        """
        Update all memory systems with cycle state.

        Args:
            cycle_state: Complete state from current cycle
        """
        raise NotImplementedError

    def retrieve_context(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant context from memory.

        Args:
            query: Query specification

        Returns:
            Context dictionary
        """
        raise NotImplementedError

    def consolidate(self):
        """
        Run memory consolidation (during DREAM state).
        """
        raise NotImplementedError
```

**Contract**:
- Must support four memory types (SNARC, IRP Bridge, Circular, Verbatim)
- Update must complete in <30ms
- Consolidation runs asynchronously

### ITrustTracker

```python
class ITrustTracker:
    """Interface for trust scoring"""

    def update(self, results: Dict[str, PluginResult]):
        """
        Update trust scores based on plugin behavior.

        Args:
            results: Plugin execution results
        """
        raise NotImplementedError

    def get_trust_scores(self) -> Dict[str, float]:
        """
        Get current trust scores for all plugins.

        Returns:
            Dictionary mapping plugin ID to trust score [0.0, 1.0]
        """
        raise NotImplementedError

    def adjust_atp_budget(self, base_budget: float) -> Dict[str, float]:
        """
        Adjust ATP allocation per plugin based on trust.

        Args:
            base_budget: Total ATP budget

        Returns:
            Dictionary mapping plugin ID to ATP allocation
        """
        raise NotImplementedError
```

**Contract**:
- Trust scores must be in range [0.0, 1.0]
- Trust update must use exponential moving average
- ATP allocation must sum to base_budget

### IEffectorHub

```python
class IEffectorHub:
    """Interface for effector management"""

    def execute(self, actions: Dict[str, Any]):
        """
        Execute actions on effectors.

        Args:
            actions: Dictionary of action commands
        """
        raise NotImplementedError

    def get_available_effectors(self) -> List[str]:
        """Get list of available effectors"""
        raise NotImplementedError
```

**Contract**:
- Must not block (commands should be async)
- Must handle effector failures gracefully
- Must log all actions for replay

### IMetabolicController

```python
class IMetabolicController:
    """Interface for metabolic state management"""

    def update(self, energy_level: float, fatigue: float, stress: float) -> str:
        """
        Update metabolic state, return new state.

        Args:
            energy_level: Current energy [0.0, 1.0]
            fatigue: Current fatigue [0.0, 1.0]
            stress: Current stress [0.0, 1.0]

        Returns:
            New metabolic state (WAKE/FOCUS/REST/DREAM/CRISIS)
        """
        raise NotImplementedError

    def get_state(self) -> str:
        """Get current metabolic state"""
        raise NotImplementedError

    def get_attention_breadth(self) -> float:
        """
        Get attention breadth for current state [0.0, 1.0].

        Returns:
            Attention breadth (1.0 = broad, 0.0 = narrow)
        """
        raise NotImplementedError
```

**Contract**:
- States: WAKE, FOCUS, REST, DREAM, CRISIS
- Transitions must be hysteresis-based (avoid flickering)
- Attention breadth maps: WAKE=0.7, FOCUS=0.3, REST=0.1, DREAM=0.0, CRISIS=1.0

---

## Main Loop Design

### Pseudocode

```python
class SAGESystem:
    def run(self, max_cycles=None):
        """Continuous consciousness loop"""

        cycle_count = 0
        energy_level = 1.0
        fatigue = 0.0
        stress = 0.0

        while True:
            cycle_start = time.time()

            try:
                # 1. SENSE
                observations = sensor_hub.poll()

                # 2. EVALUATE SALIENCE
                salience_scores = snarc_scorer.evaluate(observations)

                # 3. ALLOCATE ATTENTION
                metabolic_state = metabolic_controller.get_state()
                atp_budget = compute_atp_budget(metabolic_state, energy_level)
                attention_targets = attention_allocator.allocate(
                    salience_scores, metabolic_state, atp_budget
                )

                # 4. PLAN RESOURCES
                active_resources = resource_manager.get_active_plugins()
                memory_available = 8.0 - resource_manager.get_memory_usage()
                required_plugins = resource_planner.plan(
                    attention_targets, active_resources, memory_available
                )

                # 5. LOAD RESOURCES
                active_plugins = resource_manager.update(required_plugins)

                # 6. EXECUTE IRP PLUGINS
                plugin_results = orchestrator.run_cycle(
                    attention_targets, active_plugins, atp_budget
                )

                # 7. EXECUTE ACTIONS
                actions = extract_actions(plugin_results)
                effector_hub.execute(actions)

                # 8. UPDATE TRUST
                trust_tracker.update(plugin_results)

                # 9. UPDATE MEMORY
                atp_used = sum(r.atp_used for r in plugin_results.values())
                cycle_state = CycleState(
                    cycle_id=cycle_count,
                    timestamp=cycle_start,
                    observations=observations,
                    salience_scores=salience_scores,
                    attention_targets=attention_targets,
                    required_plugins=required_plugins,
                    plugin_results=plugin_results,
                    actions=actions,
                    metabolic_state=metabolic_state,
                    energy_level=energy_level,
                    atp_budget=atp_budget,
                    atp_remaining=atp_budget - atp_used
                )
                memory_system.update(cycle_state)

                # 10. UPDATE METABOLIC STATE
                cycle_time = time.time() - cycle_start
                energy_consumed = atp_used / 10000.0
                energy_level = max(0.0, energy_level - energy_consumed)
                fatigue = min(1.0, fatigue + 0.01)
                stress = max(0.0, stress - 0.01)  # Natural decay

                metabolic_controller.update(energy_level, fatigue, stress)

                # LOOP CONTROL
                cycle_count += 1

                if max_cycles and cycle_count >= max_cycles:
                    break

                if energy_level < 0.05:
                    print("[SAGE] Critical energy - shutting down")
                    break

            except Exception as e:
                stress = min(1.0, stress + 0.1)
                log_error(e)

                if stress > 0.95:
                    print("[SAGE] Critical stress - emergency halt")
                    break
```

### State Machine

```
                         ┌──────┐
                    ┌───►│ WAKE │◄───┐
                    │    └──┬───┘    │
                    │       │        │
            Energy  │       │        │  Energy recovers
            normal  │       │        │  Stress low
                    │       │        │
                    │    ┌──▼───┐    │
                    │    │FOCUS │────┘
                    │    └──┬───┘
                    │       │
            Fatigue │       │ Deep work
            high    │       │ needed
                    │       │
                    │    ┌──▼───┐
                    ├───►│ REST │
                    │    └──┬───┘
                    │       │
            Energy  │       │ Fatigue
            critical│       │ very high
                    │       │
                    │    ┌──▼────┐
                    └────┤ DREAM │
                         └───┬───┘
                             │
                    Stress   │
                    critical │
                             │
                         ┌───▼────┐
                         │ CRISIS │
                         └────────┘
```

---

## Error Handling Strategy

### Error Categories

1. **Sensor Failures**: Camera disconnected, audio buffer overflow
2. **Plugin Failures**: IRP plugin crashes, memory allocation failure
3. **Resource Exhaustion**: Out of memory, ATP budget exceeded
4. **System Failures**: Critical exception, hardware failure

### Handling Strategy

```python
# Sensor Failures (Recoverable)
try:
    observations = sensor_hub.poll()
except SensorError as e:
    log_warning(f"Sensor failure: {e}")
    # Use last known observation or skip modality
    observations = get_cached_observations(exclude_failed=True)

# Plugin Failures (Recoverable)
try:
    results = orchestrator.run_cycle(targets, plugins, atp)
except PluginError as e:
    log_warning(f"Plugin failure: {e}")
    # Mark plugin as untrusted, continue with remaining
    trust_tracker.mark_failed(e.plugin_id)
    results = {k: v for k, v in results.items() if k != e.plugin_id}

# Resource Exhaustion (Recoverable)
try:
    active = resource_manager.update(required)
except MemoryError as e:
    log_warning(f"Memory exhausted: {e}")
    # Emergency unload non-critical plugins
    resource_manager.emergency_unload()
    active = resource_manager.get_active_plugins()

# System Failures (Terminal)
try:
    cycle_state = _cycle()
except CriticalError as e:
    log_error(f"Critical failure: {e}")
    # Attempt graceful shutdown
    save_state()
    raise SystemExit(1)
```

### Recovery Mechanisms

| Error | Recovery | Fallback | Notes |
|-------|----------|----------|-------|
| Sensor disconnected | Use cached observation | Skip modality | Log warning |
| Plugin crash | Reduce trust, skip | Use last result | May unload plugin |
| Out of memory | Emergency unload | Minimal plugins | Transition to REST |
| ATP exhausted | Early halt | Partial results | Normal behavior |
| Critical stress | Emergency stop | Save state | Requires restart |

### Graceful Degradation

```
Full Functionality:
  All sensors active → All plugins loaded → Normal ATP budget
  ↓ (sensor failure)
Degraded (Level 1):
  Partial sensors → Core plugins only → Reduced ATP
  ↓ (memory pressure)
Degraded (Level 2):
  Minimal sensors → Vision + Memory only → Minimal ATP
  ↓ (critical energy)
Emergency Mode:
  Clock only → Memory only → Hibernation
  ↓ (energy exhausted)
Shutdown:
  Save state → Log final status → Exit
```

---

## Testing Approach

### Unit Testing

Each component has isolated unit tests:

```python
# Test SensorHub
def test_sensor_hub_poll():
    hub = MockSensorHub({'enabled_sensors': ['vision', 'audio']})
    obs = hub.poll()
    assert 'vision' in obs
    assert isinstance(obs['vision'].data, torch.Tensor)
    assert obs['vision'].data.shape == (3, 480, 640)

# Test SNARCScorer
def test_snarc_scorer_evaluate():
    scorer = MockSNARCScorer({})
    obs = {'vision': Observation('vision', torch.randn(3, 480, 640), time.time())}
    scores = scorer.evaluate(obs)
    assert 'vision' in scores
    assert 0.0 <= scores['vision'].combined <= 1.0

# Test AttentionAllocator
def test_attention_allocator_wake_state():
    allocator = MockAttentionAllocator()
    scores = {
        'vision': SalienceScore('vision', combined=0.8),
        'audio': SalienceScore('audio', combined=0.3)
    }
    targets = allocator.allocate(scores, 'WAKE', 1000.0)
    assert targets[0].modality == 'vision'  # Higher salience first
    assert targets[0].priority > targets[1].priority
```

### Integration Testing

Test component interactions:

```python
def test_full_cycle():
    """Test complete SAGE cycle"""
    system = SAGESystem(use_mock_components=True)

    # Run one cycle
    cycle_state = system._cycle()

    # Verify state structure
    assert cycle_state.cycle_id == 0
    assert len(cycle_state.observations) > 0
    assert len(cycle_state.salience_scores) > 0
    assert len(cycle_state.attention_targets) > 0
    assert cycle_state.atp_remaining >= 0

    # Verify data flow
    for target in cycle_state.attention_targets:
        assert target.modality in cycle_state.observations
        assert target.modality in cycle_state.salience_scores
```

### System Testing

Test end-to-end behavior:

```python
def test_continuous_operation():
    """Test system runs for multiple cycles"""
    system = SAGESystem(use_mock_components=True)
    system.run(max_cycles=100)

    assert system.cycle_count == 100
    assert system.error_count == 0
    assert 0.0 <= system.energy_level <= 1.0

def test_metabolic_transitions():
    """Test metabolic state transitions"""
    system = SAGESystem(use_mock_components=True)

    # Run until energy depletes
    system.run(max_cycles=1000)

    # Should have transitioned to REST or DREAM
    assert system.metabolic_controller.get_state() in ['REST', 'DREAM']

def test_error_recovery():
    """Test system recovers from errors"""
    system = SAGESystem(use_mock_components=True)

    # Inject sensor failure
    system.sensor_hub.fail_sensor('vision')

    # Should continue running
    system.run(max_cycles=10)

    assert system.cycle_count == 10
    assert system.error_count > 0
    assert system.stress > 0.0
```

### Performance Testing

```python
def test_cycle_time():
    """Test cycle completes in <1 second"""
    system = SAGESystem(use_mock_components=True)

    start = time.time()
    system._cycle()
    duration = time.time() - start

    assert duration < 1.0  # Should be fast with mocks

def test_memory_usage():
    """Test memory stays bounded"""
    system = SAGESystem(use_mock_components=True)

    initial_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    system.run(max_cycles=100)
    final_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    growth = (final_memory - initial_memory) / initial_memory
    assert growth < 0.1  # Less than 10% growth
```

---

## Implementation Phases

### Phase 1: Core Loop (Minimal) - Week 1

**Goal**: Get the loop running with mock components

**Tasks**:
1. ✅ Create `sage_system.py` skeleton with all interfaces
2. ✅ Implement mock versions of all components
3. ✅ Implement main `_cycle()` method
4. ✅ Implement `run()` loop with logging
5. ✅ Write unit tests for data structures
6. Test continuous operation (100 cycles)

**Success Criteria**:
- Loop executes without crashing
- All data structures flow correctly
- Console output shows cycle progression
- Trust scores update
- Metabolic state transitions

**Deliverables**:
- `/sage/core/sage_system.py` (this file)
- `/sage/docs/SAGE_SYSTEM_ARCHITECTURE.md` (this document)
- `/sage/tests/test_sage_system.py`

### Phase 2: Real Components - Week 2

**Goal**: Replace mocks with real implementations

**Tasks**:
1. Implement real SensorHub (camera, audio, clock)
2. Integrate existing SNARCScorer from `/sage/attention/snarc_scorer.py`
3. Implement real AttentionAllocator
4. Implement real ResourcePlanner and ResourceManager
5. Integrate existing HRMOrchestrator from `/sage/orchestration/`
6. Test with real sensors and one IRP plugin

**Success Criteria**:
- Real camera frames processed
- SNARC scores computed correctly
- IRP plugins execute and converge
- Memory usage tracked accurately

**Deliverables**:
- `RealSensorHub` class
- `RealResourceManager` class
- Integration tests with hardware

### Phase 3: Memory & Trust - Week 3

**Goal**: Add learning and adaptation

**Tasks**:
1. Integrate existing memory systems (SNARC, IRP Bridge, Circular, Verbatim)
2. Implement real TrustTracker with monotonicity/efficiency metrics
3. Implement trust-based ATP allocation
4. Implement memory-guided refinement hints
5. Test long-term operation (1000 cycles)

**Success Criteria**:
- Trust scores converge to stable values
- ATP allocation adapts to plugin performance
- Memory consolidation runs during DREAM state
- Retrieved context influences decisions

**Deliverables**:
- `RealMemorySystem` class
- `RealTrustTracker` class
- Long-term stability tests

### Phase 4: Full Integration - Week 4

**Goal**: Complete system with all features

**Tasks**:
1. Implement real EffectorHub (TTS, display, motors)
2. Integrate real MetabolicController from `/sage/core/metabolic_states.py`
3. Deploy to Jetson Orin Nano
4. Performance optimization (target <100ms per cycle)
5. Error handling and recovery testing
6. Documentation and examples

**Success Criteria**:
- Runs on Jetson for 1 hour without intervention
- Responds to real-world stimuli
- Actions executed on effectors
- Metabolic states transition naturally
- Resource usage stays within 8GB

**Deliverables**:
- Complete SAGESystem implementation
- Jetson deployment guide
- Performance benchmarks
- Example applications

---

## Performance Considerations

### Target Metrics

| Metric | Target | Critical |
|--------|--------|----------|
| Cycle time | 100ms avg | 500ms max |
| Memory usage | 6GB avg | 7.5GB max |
| Energy per cycle | 0.01% | 0.05% |
| Plugin load time | 200ms | 1000ms |
| Trust convergence | 100 cycles | 500 cycles |

### Optimization Strategies

1. **Lazy Loading**: Only import plugins when needed
2. **GPU Sharing**: Reuse GPU memory between plugins
3. **Async Sensors**: Poll sensors in background thread
4. **Batch Processing**: Process multiple observations together
5. **Early Stopping**: Halt IRP when energy plateau detected
6. **Memory Pooling**: Preallocate tensors to avoid allocation overhead
7. **Model Quantization**: Use INT8/FP16 for edge deployment

### Bottleneck Analysis

**Expected bottlenecks**:
1. IRP plugin execution (50-500ms) - **Dominant**
2. Resource loading (0-200ms) - **Amortized**
3. Memory consolidation (20ms) - **Acceptable**
4. SNARC evaluation (5-10ms) - **Acceptable**

**Mitigation**:
- Use trust scores to reduce iterations on reliable plugins
- Cache loaded plugins aggressively
- Run consolidation in background during WAKE state
- Use GPU for SNARC computation if available

---

## Appendix: Key Insights

### 1. Consciousness as Iterative Refinement

All intelligence is progressive denoising toward lower energy states. Vision, language, planning, memory - same pattern. The IRP interface captures this universal principle.

### 2. Trust as Compression Quality

Trust measures how well meaning is preserved through compression. High trust (>0.9) means reliable cross-modal translation. Low trust means the plugin needs more iterations or better training.

### 3. The Fractal H↔L Pattern

Hierarchical ↔ Linear reasoning appears at every scale:
- Neural (transformer blocks)
- Agent (SAGE reasoning)
- Device (edge ↔ cloud)
- Federation (coordinator ↔ workers)
- Development (human ↔ automation)

This is not mimicry - it's discovering the same optimal solution at different scales.

### 4. ATP as Universal Currency

Attention budget (ATP) provides a universal currency for resource allocation:
- Plugins compete for ATP based on trust
- Metabolic state modulates total ATP
- Energy consumption directly tied to ATP spent
- Trust emerges from efficient ATP use

### 5. The Four Memory Systems

Each memory serves a distinct purpose:
- **SNARC Memory**: What was salient (selective)
- **IRP Bridge**: What worked (procedural)
- **Circular Buffer**: What happened recently (episodic)
- **Verbatim Storage**: Everything (archival)

Together they provide complete memory coverage.

---

## References

- `/sage/docs/SAGE_WORKING_VISION.md` - Original vision document
- `/sage/docs/SYSTEM_UNDERSTANDING.md` - Complete system mental model
- `/sage/irp/base.py` - IRP plugin interface
- `/sage/attention/snarc_scorer.py` - SNARC implementation
- `/sage/core/metabolic_states.py` - Metabolic states
- `/sage/orchestration/` - HRM orchestrator

---

**End of Document**

*This is design, not full implementation. The skeleton exists with well-documented interfaces. Next step: Phase 1 testing and validation.*
