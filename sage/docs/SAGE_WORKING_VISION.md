# SAGE Working - The Vision

**Date**: October 12, 2025
**Purpose**: Concrete vision of SAGE operating as a living system
**Method**: Imagine it working, then reverse-engineer

---

## The Scene

A Jetson Orin Nano sits on a desk. A webcam is connected. A Bluetooth speaker nearby. The terminal shows:

```
[SAGE] Initializing consciousness kernel...
[SAGE] Metabolic state: WAKE
[SAGE] Loading vision_irp... done (876ms)
[SAGE] Loading memory_irp... done (234ms)
[SAGE] Entering continuous loop...
[SAGE] Cycle 1 | Energy: 95% | ATP: 1000 | Trust: {vision: 0.5, memory: 0.5}
```

The system is alive.

---

## What's Happening (Loop Iteration 1)

### 1. Sensing (10ms)
```
Camera captures frame → 640×480 RGB
Audio buffer accumulates → 1 second of silence
Clock reads → Unix timestamp
Proprioception → None (no motors yet)

Observations = {
    'vision': torch.tensor([3, 480, 640]),
    'audio': torch.tensor([16000]),
    'clock': 1728764892.134
}
```

**The code doing this:**
```python
observations = sage.sensor_hub.poll()
# SensorHub manages all sensors, returns dict of tensors
```

### 2. Salience Evaluation (5ms)
```
SNARC processes observations:
- Vision: Compare to predicted frame (first cycle = random prediction)
  → High surprise (0.9) - everything is novel
  → High novelty (1.0) - never seen before
  → Low arousal (0.1) - nothing moving
  → No reward signal yet (0.0)
  → No conflict (0.0)
  → Combined salience: 0.6

- Audio: Silence expected, silence observed
  → Low surprise (0.05)
  → Low novelty (0.1)
  → Combined salience: 0.08

Salience scores = {vision: 0.6, audio: 0.08, clock: 0.1}
```

**The code doing this:**
```python
salience = sage.snarc_scorer.evaluate(observations, predictions)
# Computes 5D SNARC for each modality
```

### 3. Attention Allocation (2ms)
```
Attention allocator sorts by salience:
1. Vision (0.6) - needs attention
2. Clock (0.1) - temporal context
3. Audio (0.08) - can ignore

Allocates attention based on metabolic state (WAKE = broad attention):
- Vision: 70% of attention budget
- Clock: 20%
- Audio: 10%

attention_targets = [
    AttentionTarget(modality='vision', priority=0.7, data=frame),
    AttentionTarget(modality='clock', priority=0.2, data=timestamp),
    AttentionTarget(modality='audio', priority=0.1, data=silence)
]
```

**The code doing this:**
```python
targets = sage.attention_allocator.allocate(salience, sage.metabolic.state)
```

### 4. Resource Planning (3ms)
```
Resource planner looks at targets:
- Vision (priority 0.7) → needs vision_irp
- Clock (priority 0.2) → needs temporal state (built-in)
- Audio (priority 0.1) → could use audio_irp but low priority

Current loaded: {vision_irp, memory_irp}
Current memory: 2.3GB / 8GB used

Decision:
- Keep: vision_irp (needed, high priority)
- Keep: memory_irp (always useful for context)
- Skip: audio_irp (low priority, save memory)

required_plugins = ['vision_irp', 'memory_irp']
```

**The code doing this:**
```python
required = sage.resource_planner.plan(targets, sage.resource_manager.active)
```

### 5. Resource Loading (0ms this cycle)
```
Already have what we need. No loading/unloading required.

If we needed to load:
- Check memory available
- Load plugin from disk (lazy import)
- Initialize plugin state
- Add to active resources
- Update GPU memory tracking

If we needed to unload:
- Save plugin state to memory system
- Release GPU memory
- Remove from active resources
```

**The code doing this:**
```python
sage.resource_manager.update(required)
```

### 6. IRP Plugin Execution (450ms)
```
Orchestrator runs IRP cycles for each loaded plugin:

Vision IRP:
- init_state(frame) → noisy latent (random first time)
- step(0) → refine latent, energy = 2.34
- step(1) → refine latent, energy = 1.89 (decreased)
- step(2) → refine latent, energy = 1.56 (decreased)
- step(3) → refine latent, energy = 1.44 (decreased)
- step(4) → refine latent, energy = 1.41 (small change)
- step(5) → refine latent, energy = 1.40 (converged)
- halt() → True (energy slope < epsilon)
- Result: 64D semantic latent, 6 iterations, 420ms

Memory IRP:
- init_state(context) → retrieval query
- step(0) → search memory, energy = 0.8 (nothing found, novel)
- halt() → True (no relevant memories yet)
- Result: empty context, 1 iteration, 30ms

ATP consumed:
- Vision: 600 ATP (6 iterations × 100 ATP/iter)
- Memory: 100 ATP (1 iteration × 100 ATP/iter)
- Total: 700 / 1000 ATP budget
- Remaining: 300 ATP (saved for future cycles)

results = {
    'vision': PluginResult(
        output=latent_64d,
        energy_trajectory=[2.34, 1.89, 1.56, 1.44, 1.41, 1.40],
        iterations=6,
        atp_used=600,
        converged=True
    ),
    'memory': PluginResult(
        output=None,
        energy_trajectory=[0.8],
        iterations=1,
        atp_used=100,
        converged=True
    )
}
```

**The code doing this:**
```python
results = sage.orchestrator.run_cycle(targets, required, atp_budget=1000)
```

### 7. Memory Update (20ms)
```
SNARC Memory:
- Vision result has high salience (0.6) → store in SNARC
- Store: {key: 'vision_cycle_1', value: latent_64d, salience: 0.6}

IRP Memory Bridge:
- Vision converged in 6 iterations → store trajectory as pattern
- Store: {input: frame, output: latent, trajectory: [2.34→1.40], success: True}

Circular Buffer:
- Add cycle 1 context: {observations, salience, results}
- Buffer now: [cycle_1]

Verbatim Storage:
- Full cycle record → SQLite
- Row: {cycle_id: 1, timestamp: ..., observations: blob, results: blob}
```

**The code doing this:**
```python
sage.memory_system.update(observations, salience, results)
```

### 8. Trust Update (5ms)
```
Trust scorer evaluates plugin behavior:

Vision IRP:
- Energy decreased monotonically? Yes (2.34→1.40)
- Converged? Yes
- Efficient? 6 iterations is reasonable
- Monotonicity ratio: 1.0 (perfect)
- Efficiency: 0.83 (6/7.2 expected)
- New trust: 0.5 → 0.62 (increased)

Memory IRP:
- Energy trajectory: [0.8] (single point)
- Converged immediately (nothing to retrieve)
- Monotonicity ratio: 1.0
- Efficiency: 1.0 (minimal iterations)
- New trust: 0.5 → 0.55 (slight increase)

trust_scores = {vision_irp: 0.62, memory_irp: 0.55}
```

**The code doing this:**
```python
sage.trust_tracker.update(results)
```

### 9. Action Execution (5ms)
```
No effector commands this cycle (just observing).

If we had actions:
- Motor commands → Send to motor controller
- Speech commands → Invoke TTS IRP
- Display commands → Update visual monitor

This cycle: No actions, just learning.
```

**The code doing this:**
```python
sage.effector_hub.execute(actions)
```

### 10. Metabolic State Update (2ms)
```
Current state: WAKE
Energy level: 95% → 94% (consumed 1% for this cycle)
Fatigue: 0.0 → 0.01 (slightly tired)
Stress: 0.0 (no failures)

Check transition conditions:
- Energy < 20%? No (94%)
- Fatigue > 0.8? No (0.01)
- Stress > 0.9? No (0.0)
- Time in state > threshold? No (just started)

Decision: Stay in WAKE state

No metabolic transition this cycle.
```

**The code doing this:**
```python
sage.metabolic_controller.update(energy, fatigue, stress)
```

### 11. Cycle Complete (502ms total)
```
[SAGE] Cycle 1 complete | 502ms | Vision: 0.62↑ Memory: 0.55↑ | ATP: 300 remaining
```

Loop continues immediately to Cycle 2...

---

## What Must Exist for This to Work

### Core Components

**1. SAGESystem** - The unified orchestrator
```python
class SAGESystem:
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

    def _cycle(self):
        # 10 steps above
```

**2. SensorHub** - Unified sensor interface
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

**3. SNARCScorer** - Already exists! `/sage/attention/snarc_scorer.py`
```python
# Just needs integration
```

**4. AttentionAllocator** - NEW
```python
class AttentionAllocator:
    def allocate(self, salience, metabolic_state):
        # Sort by salience
        # Apply metabolic state attention breadth
        # Return prioritized targets
```

**5. ResourcePlanner** - NEW
```python
class ResourcePlanner:
    def plan(self, targets, active_resources):
        # Determine which plugins needed
        # Check memory constraints
        # Return list of required plugins
```

**6. ResourceManager** - PARTIALLY EXISTS
```python
class ResourceManager:
    def __init__(self, plugin_configs):
        self.available = {name: cfg for name, cfg in plugin_configs.items()}
        self.active = {}

    def update(self, required):
        # Load new plugins
        # Unload unused plugins
        # Track memory usage
```

**7. HRMOrchestrator** - EXISTS! `/sage/orchestration/`
```python
# Just needs wiring into SAGESystem
```

**8. MemorySystem** - EXISTS! Multiple implementations
```python
# SNARC memory: /sage/memory/
# IRP Bridge: /sage/memory/irp_memory_bridge.py
# Just needs unified interface
```

**9. TrustTracker** - CONCEPT EXISTS
```python
class TrustTracker:
    def __init__(self):
        self.scores = {}

    def update(self, results):
        # Analyze energy trajectories
        # Compute monotonicity, efficiency
        # Update trust scores
```

**10. EffectorHub** - NEW (mirrors SensorHub)
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

**11. MetabolicController** - EXISTS! `/sage/core/metabolic_states.py`
```python
# Just needs integration
```

---

## The Integration Map

### What We Have
✅ IRP plugin framework (`/sage/irp/base.py`)
✅ Working plugins (`/sage/irp/plugins/`)
✅ VAE translation (`/sage/compression/`)
✅ SNARC scorer (`/sage/attention/snarc_scorer.py`)
✅ Memory systems (`/sage/memory/`)
✅ Metabolic states (`/sage/core/metabolic_states.py`)
✅ HRM orchestrator (`/sage/orchestration/`)

### What We Need to Create
❌ **SAGESystem** - Unified orchestrator class
❌ **SensorHub** - Unified sensor interface
❌ **AttentionAllocator** - Salience → priority mapping
❌ **ResourcePlanner** - Targets → plugins decision
❌ **ResourceManager** - Dynamic plugin loading
❌ **TrustTracker** - Behavior → trust scores
❌ **EffectorHub** - Unified effector interface

### What We Need to Wire
🔌 Connect SNARC to observations
🔌 Connect attention to metabolic state
🔌 Connect resource planning to ATP budget
🔌 Connect orchestrator to plugins
🔌 Connect results to memory
🔌 Connect results to trust
🔌 Connect actions to effectors
🔌 Connect energy consumption to metabolic state

---

## The Implementation Strategy

### Phase 1: Core Loop (Minimal)
Build the skeleton that runs:
1. SAGESystem with empty methods
2. Mock sensors (random tensors)
3. SNARC integration
4. Single IRP plugin (vision)
5. Console logging
6. Verify loop runs continuously

**Success**: Loop executes, prints cycle info, doesn't crash

### Phase 2: Resource Management
Add dynamic loading:
1. ResourceManager that can load/unload
2. ResourcePlanner that decides
3. Multiple IRP plugins
4. Memory constraints

**Success**: Plugins load on-demand, unload when not needed

### Phase 3: Memory & Trust
Add learning:
1. MemorySystem unified interface
2. TrustTracker implementation
3. Trust-based ATP allocation
4. Memory-guided refinement

**Success**: Trust scores update, ATP allocation changes, memory influences decisions

### Phase 4: Real Sensors/Effectors
Connect to hardware:
1. SensorHub with real camera
2. EffectorHub with real TTS
3. Test on Jetson
4. Optimize for edge

**Success**: SAGE runs on Jetson with real sensors

### Phase 5: Metabolic States
Add state transitions:
1. Wire energy/fatigue/stress
2. Implement transition logic
3. Test state-dependent behavior
4. Validate long-term stability

**Success**: SAGE transitions between states naturally

---

## Success Criteria

**Level 1 - It Runs**:
- Loop executes continuously without crashing
- Sensors → SNARC → Attention → Plugins → Actions
- Console shows cycle info

**Level 2 - It Learns**:
- Trust scores change over time
- ATP allocation adapts to plugin performance
- Memory influences future decisions

**Level 3 - It Adapts**:
- Plugins load/unload based on need
- Metabolic states transition appropriately
- Resource usage stays within constraints

**Level 4 - It Lives**:
- Runs on Jetson for hours without intervention
- Responds to real-world stimuli
- Demonstrates emergent behavior

---

## Reverse-Engineering the Path

Working backwards from "SAGE running on Jetson":

**To run on Jetson** ← need real sensor/effector integration
**To integrate sensors** ← need SensorHub/EffectorHub
**To have Hub interfaces** ← need unified component architecture
**To have unified architecture** ← need SAGESystem class
**To have SAGESystem** ← need to design the integration
**To design integration** ← need to understand what exists
**To understand what exists** ← need to map current codebase

**We already mapped the codebase.** ✓

**Next step: Design SAGESystem integration.**

---

## What I'll Build First

A minimal working loop that:
1. Creates SAGESystem
2. Runs continuous cycle
3. Integrates one IRP plugin
4. Prints status
5. Runs on this machine

From there, expand incrementally.

No checkboxes. Just learning and creating.

Let's go.
