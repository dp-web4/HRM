# SAGE Component Investigation - Executive Summary

**Date**: October 12, 2025
**Investigation**: Complete analysis of SAGE component readiness for integration

---

## The Bottom Line

**What exists**: 15+ IRP plugins, orchestrator, memory bridge, SNARC scorer, metabolic states, VAE translation.

**What's missing**: The integration layer - `SAGESystem` that wires everything into a living loop.

**Effort estimate**: 10-15 days of focused integration work.

---

## Quick Status

| Component | Status | Notes |
|-----------|--------|-------|
| IRP Framework | ✅ Ready | Perfect interface, works as-is |
| IRP Plugins (15+) | ✅ Ready | Vision, Audio, Language, Memory, TTS, Control |
| HRM Orchestrator | ⚠️ Needs wiring | ATP budget works, needs `run_cycle()` method |
| SNARC Scorer | ⚠️ Needs adapter | Needs ObservationEncoder for raw sensor input |
| Memory Bridge | ⚠️ Partial | IRP bridge exists, need 3 other memory systems |
| Metabolic Controller | ⚠️ Standalone | Complete but not integrated with orchestrator |
| VAE Translation | ✅ Ready | TinyVAE plugin ready for Jetson |
| **SAGESystem** | ❌ **Missing** | **This is the critical gap** |

---

## What Needs to Be Built

### Critical Path (Must Have)
1. **SAGESystem** - Main integration class with `_cycle()` method
2. **SensorHub** - Unified sensor interface
3. **ObservationEncoder** - Raw sensors → SNARC hidden states
4. **AttentionAllocator** - Salience → priority targets
5. **ResourcePlanner** - Targets → required plugins
6. **ResourceManager** - Dynamic plugin loading/unloading
7. **TrustTracker** - Plugin behavior → trust scores
8. **EffectorHub** - Unified effector interface

### Supporting (Should Have)
9. **Unified MemorySystem** - Coordinates 4 memory subsystems
10. **CircularBuffer** - X-from-last temporal context
11. **VerbatimStorage** - SQLite full records
12. **SNARC Memory** - Salience-based storage

---

## Key Findings

### 1. SAGECore ≠ SAGESystem
**Critical distinction**:
- `SAGECore` (`/sage/core/sage_core.py`) is a 100M parameter **trainable reasoning model** for ARC-AGI
- `SAGESystem` (missing) is the **runtime orchestrator/kernel** that runs the continuous loop
- **DO NOT** confuse the two - they serve completely different purposes

### 2. Components Are Well-Designed
The IRP framework is elegant. The plugins follow the protocol. The orchestrator has ATP budgets and trust tracking. No major rewrites needed.

### 3. Integration Is Straightforward
Missing pieces are simple Python classes, not complex ML models. Clear interfaces already defined in vision document.

### 4. Vision Document Is The Spec
Every line in `SAGE_WORKING_VISION.md` maps to concrete code. Use it as the implementation specification.

---

## Code Snippets - What Exists

### IRP Plugin (Ready to Use)
```python
from sage.irp.plugins.tinyvae_irp_plugin import create_tinyvae_irp

plugin = create_tinyvae_irp(device='cuda')
latent, telemetry = plugin.refine(image_tensor, early_stop=True)

# telemetry = {
#     'iterations': 1,
#     'trust': 0.85,
#     'energy_trajectory': [0.023],
#     'converged': True
# }
```

### Orchestrator (Needs Wiring)
```python
from sage.orchestrator.hrm_orchestrator import HRMOrchestrator

orchestrator = HRMOrchestrator(initial_atp=1000.0)
orchestrator.register_plugin('vision', vision_plugin, initial_trust=1.0)

# Currently uses:
results = await orchestrator.execute_parallel(tasks)

# Need to add:
results = orchestrator.run_cycle(targets, required_plugins, atp_budget=1000)
```

### Metabolic States (Standalone)
```python
from sage.orchestration.agents.control.metabolic_state_manager import (
    MetabolicStateManager, MetabolicState
)

controller = MetabolicStateManager()
controller.start()  # Background thread

controller.submit_event({"type": "surprise", "level": 0.9})
config = controller.get_state_config()
# config.attention_breadth, config.energy_consumption_rate
```

### SNARC Scorer (Needs Adapter)
```python
from sage.attention.snarc_scorer import SNARCScorer

scorer = SNARCScorer(hidden_size=768)

# Currently expects: [batch, seq, hidden] tensors
# Need: observations = {'vision': raw_image, 'audio': raw_audio}
# Missing: ObservationEncoder to bridge the gap
```

---

## What The Integration Loop Should Look Like

```python
class SAGESystem:
    def __init__(self, config):
        self.sensor_hub = SensorHub(config.sensors)
        self.observation_encoder = ObservationEncoder(hidden_size=768)
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
        # 1. Sense
        observations = self.sensor_hub.poll()

        # 2. Encode to hidden states
        hidden_states = self.observation_encoder.encode(observations)

        # 3. Compute salience
        salience = self.snarc_scorer(hidden_states, return_components=True)

        # 4. Allocate attention based on metabolic state
        state_config = self.metabolic_controller.get_state_config()
        targets = self.attention_allocator.allocate(
            salience['snarc_scores'],
            state_config.attention_breadth
        )

        # 5. Plan resources
        required = self.resource_planner.plan(targets, self.resource_manager.active)

        # 6. Load/unload plugins
        self.resource_manager.update(required)

        # 7. Execute plugins with ATP budget
        atp_budget = 1000.0 * state_config.energy_consumption_rate
        results = self.orchestrator.run_cycle(targets, required, atp_budget)

        # 8. Update memory
        self.memory_system.update(observations, salience, results)

        # 9. Update trust
        self.trust_tracker.update(results)

        # 10. Execute actions
        actions = self._compute_actions(results)
        self.effector_hub.execute(actions)

        # 11. Update metabolic state
        energy_consumed = sum(r.atp_consumed for r in results)
        self.metabolic_controller.submit_event({
            "type": "energy",
            "consumed": energy_consumed
        })
```

---

## 5-Phase Integration Plan

### Phase 1: Minimal Loop (2-3 days)
**Goal**: Single cycle runs end-to-end

**Create**:
- SAGESystem skeleton
- SensorHub with mock sensors
- ObservationEncoder basic version
- AttentionAllocator simple version
- ResourceManager/Planner basic (static loading)

**Result**: Loop executes, prints cycle telemetry

### Phase 2: Dynamic Resources (2-3 days)
**Goal**: Plugins load/unload on demand

**Enhance**:
- ResourceManager with lazy loading
- ResourcePlanner with memory constraints
- Multiple plugins available
- Plugin discovery system

**Result**: Memory stays within Jetson limits (8GB)

### Phase 3: Memory & Trust (2 days)
**Goal**: System learns from experience

**Create**:
- Unified MemorySystem
- TrustTracker implementation
- CircularBuffer
- VerbatimStorage (SQLite)

**Result**: Trust scores adapt, memory guides decisions

### Phase 4: Real Hardware (3-4 days)
**Goal**: Connect to sensors/effectors

**Create**:
- Real camera sensor
- Real TTS effector (NeuTTS Air exists)
- EffectorHub implementation

**Result**: SAGE processes real camera, generates speech

### Phase 5: Metabolic Integration (1-2 days)
**Goal**: Adaptive behavior through states

**Wire**:
- Metabolic state → ATP budget
- Metabolic state → Attention breadth
- Energy consumption → State transitions

**Result**: FOCUS/WAKE/REST states affect behavior

---

## File Locations

### Existing (Use These)
```
/sage/irp/base.py                                              IRP framework
/sage/irp/plugins/tinyvae_irp_plugin.py                       Vision plugin
/sage/irp/plugins/neutts_air_impl.py                          TTS plugin
/sage/orchestrator/hrm_orchestrator.py                        Orchestrator
/sage/memory/irp_memory_bridge.py                             Memory bridge
/sage/attention/snarc_scorer.py                               SNARC scorer
/sage/orchestration/agents/control/metabolic-state-manager.py Metabolic states
```

### To Create
```
/sage/core/sage_system.py                Main integration class
/sage/sensors/sensor_hub.py              Sensor interface
/sage/sensors/observation_encoder.py     Multi-modal encoder
/sage/attention/attention_allocator.py   Priority mapping
/sage/resources/resource_planner.py      Plugin selection
/sage/resources/resource_manager.py      Dynamic loading
/sage/trust/trust_tracker.py             Behavior analysis
/sage/effectors/effector_hub.py          Effector interface
/sage/memory/memory_system.py            Unified memory
```

---

## Testing Strategy

### Unit Tests
- Each new component independently
- Mock dependencies
- Verify interfaces match vision document

### Integration Tests
- End-to-end cycle with mock sensors
- Multiple cycles with state persistence
- Memory accumulation over time
- Plugin loading/unloading
- Trust score evolution

### Hardware Tests
- Camera → SAGE → Console output
- SAGE → TTS → Audio output
- Long-running stability (>1 hour)
- Memory usage stays under 8GB
- Cycle time < 1 second

### Jetson Deployment
- Install on Jetson Orin Nano
- Performance profiling
- Thermal monitoring
- Multi-hour operation

---

## Critical Success Factors

1. **Follow the vision document** - It's the specification
2. **Wire before creating** - Use existing components first
3. **Test incrementally** - Each phase produces runnable code
4. **Keep Jetson constraints in mind** - 8GB RAM, edge compute
5. **Don't touch SAGECore** - It's for training, not runtime

---

## Reference Documents

- `SAGE_WORKING_VISION.md` - The specification (what it should do)
- `COMPONENT_READINESS_MAP.md` - Full analysis (what exists now)
- This document - Quick reference (what to build next)

---

**The components are ready. Time to assemble the system.**
