# Unified SAGE System Integration

**Date:** 2025-11-05
**Purpose:** Design document for integrating all SAGE components into unified cognition loop
**Status:** Implementation in progress

---

## Current State Analysis

### What Exists (✅ Operational)

**1. SAGEKernel** (`sage/core/sage_kernel.py`)
- ✅ Basic cognition loop: gather → assess → decide → execute → learn
- ✅ SNARC integration for salience assessment
- ✅ Sensor abstraction (callables)
- ✅ Action handler abstraction
- ✅ Outcome-based learning
- ❌ Missing: H↔L reasoning, plugin management, metabolic states, memory

**2. SAGECore** (`sage/core/sage_core.py`)
- ✅ 100M parameter H↔L transformer
- ✅ HModule (strategic reasoning, ~45M params)
- ✅ LModule (tactical execution, ~45M params)
- ✅ Bidirectional information flow
- ❌ Missing: Integration with kernel loop

**3. HRMOrchestrator** (`sage/orchestrator/hrm_orchestrator.py`, `sage/irp/orchestrator.py`)
- ✅ IRP plugin management
- ✅ ATP budget allocation
- ✅ Trust-weighted resource distribution
- ✅ Async concurrent execution
- ✅ Dynamic reallocation
- ❌ Missing: Integration with SAGECore, metabolic state awareness

**4. SNARCService** (`sage/services/snarc/snarc_service.py`)
- ✅ 5D salience assessment (Surprise, Novelty, Arousal, Reward, Conflict)
- ✅ Attention recommendation
- ✅ Cognitive stance determination
- ✅ Outcome-based learning
- ✅ **Already integrated with SAGEKernel**

**5. MetabolicController** (`sage/core/metabolic_controller.py`)
- ✅ 5 operational states (WAKE, FOCUS, REST, DREAM, CRISIS)
- ✅ ATP-based transitions
- ✅ Circadian rhythm integration
- ✅ State-specific resource policies
- ❌ Missing: Integration with orchestrator

**6. Memory Systems**
- ✅ SNARC Memory (selective storage) - integrated with SNARCService
- ✅ IRP Memory Bridge (pattern library) - `sage/irp/memory.py`
- ✅ Circular Buffer (recent context)
- ✅ Verbatim Storage (SQLite)
- ❌ Missing: Unified memory interface

**7. IRP Plugins** (`sage/irp/plugins/`)
- ✅ Vision, Language, Memory, Audio Input, TTS (NeuTTS Air), Control
- ✅ Iterative refinement protocol
- ✅ Energy-based convergence
- ✅ Trust dynamics
- ✅ **Ready for orchestration**

---

## Integration Architecture

### The Unified Loop

```python
class UnifiedSAGESystem(SAGEKernel):
    """
    Complete SAGE cognition system integrating all components.

    Extends SAGEKernel with:
    - SAGECore H↔L reasoning
    - HRMOrchestrator plugin management
    - MetabolicController state management
    - Unified memory interface
    - Effector abstraction
    """

    while True:
        # 1. METABOLIC CHECK
        metabolic_state = controller.assess_state(atp_level, attention_load)
        resource_policy = get_policy(metabolic_state)

        # 2. GATHER (sensors → raw observations)
        observations = {
            'vision': camera.capture(),
            'audio': mic.sample(),
            'proprioception': joints.read(),
            'memory': memory.recall_context()
        }

        # 3. ASSESS (SNARC → salience + stance)
        salience_report = snarc.assess_salience(observations)
        focus_target = salience_report.focus_target
        stance = salience_report.suggested_stance

        # 4. REASON (SAGECore → strategic decisions)
        h_output, strategy = h_module.forward(
            observations[focus_target],
            context=salience_report
        )

        # 5. ALLOCATE (Orchestrator → resource distribution)
        atp_allocations = orchestrator.allocate_budgets(
            available_atp=controller.current_atp,
            trust_weights=plugin_trust_scores,
            metabolic_policy=resource_policy
        )

        # 6. EXECUTE (Plugins → refined outputs)
        plugin_results = await orchestrator.execute_plugins(
            selected_plugins=strategy.required_plugins,
            inputs=h_output,
            atp_budgets=atp_allocations
        )

        # 7. RESPOND (LModule → tactical actions)
        actions = l_module.forward(
            plugin_results,
            strategic_guidance=strategy
        )

        # 8. ACT (effectors → physical world)
        effector_results = {
            'tts': speaker.synthesize(actions['speech']),
            'motor': joints.move(actions['movement']),
            'display': screen.render(actions['visualization'])
        }

        # 9. LEARN (update all adaptive components)
        outcome = evaluate_outcome(effector_results)
        snarc.update_from_outcome(salience_report, outcome)
        orchestrator.update_trust_scores(plugin_results, outcome)
        memory.consolidate(observations, plugin_results, outcome)
        controller.update_atp(energy_consumed, metabolic_state)
```

---

## Component Integration Details

### 1. SAGECore ↔ SAGEKernel

**Current Gap:**
- SAGEKernel uses simple action handlers
- SAGECore exists separately

**Integration:**
```python
class UnifiedSAGESystem(SAGEKernel):
    def __init__(self, ...):
        super().__init__(...)
        self.sage_core = SAGECoreModel(config)  # Load H↔L transformer

    def _cycle(self):
        observations = self._gather_observations()
        salience_report = self.snarc.assess_salience(observations)

        # NEW: Strategic reasoning through H-module
        h_output, strategy = self.sage_core.h_module.forward(
            observations[salience_report.focus_target],
            context=self._encode_context(salience_report)
        )

        # Execute with strategic guidance
        plugin_results = self._execute_plugins(h_output, strategy)

        # NEW: Tactical execution through L-module
        actions = self.sage_core.l_module.forward(
            plugin_results,
            strategy=strategy
        )

        return actions
```

### 2. HRMOrchestrator ↔ MetabolicController

**Current Gap:**
- Orchestrator allocates ATP independently
- MetabolicController manages state transitions independently

**Integration:**
```python
class UnifiedSAGESystem:
    def _cycle(self):
        # Check metabolic state
        current_state = self.metabolic_controller.get_state()
        state_config = self.metabolic_controller.get_current_config()

        # Apply metabolic constraints to orchestration
        max_plugins = state_config.max_active_plugins
        available_atp = self.metabolic_controller.current_atp

        # Allocate with metabolic awareness
        allocations = self.orchestrator.allocate_budgets(
            available_atp=available_atp,
            max_concurrent=max_plugins,
            trust_weights=self.plugin_trust_scores
        )

        # Execute plugins
        results = await self.orchestrator.execute_plugins(allocations)

        # Update metabolic state based on consumption
        total_consumed = sum(r.atp_consumed for r in results)
        self.metabolic_controller.consume_atp(total_consumed)
```

### 3. Memory Systems Integration

**Current Gap:**
- Four memory systems exist separately
- No unified interface

**Integration:**
```python
class UnifiedMemoryInterface:
    """Unified interface to all 4 memory systems"""

    def __init__(self):
        self.snarc_memory = SNARCMemory()  # High salience only
        self.irp_memory = IRPMemoryBridge()  # Pattern library
        self.circular_buffer = CircularBuffer(size=100)  # Recent context
        self.verbatim_storage = VerbatimDB()  # SQLite full fidelity

    def store(self, experience, salience_report, plugin_results):
        """Store experience across appropriate memory systems"""

        # Always store in circular buffer
        self.circular_buffer.push(experience)

        # Store in SNARC if salient
        if salience_report.salience_score > 0.7:
            self.snarc_memory.store(experience, salience_report)

        # Store successful patterns in IRP memory
        if plugin_results.all_converged:
            self.irp_memory.store_pattern(plugin_results.refinement_trajectory)

        # Store verbatim for potential retrieval
        self.verbatim_storage.insert(experience)

    def recall_context(self, current_observation):
        """Retrieve relevant context from all systems"""
        return {
            'recent': self.circular_buffer.get_window(),
            'similar': self.irp_memory.query_similar(current_observation),
            'salient': self.snarc_memory.query_relevant(current_observation),
            'verbatim': self.verbatim_storage.search(current_observation)
        }
```

### 4. Sensor Abstraction

**Current:**
- SAGEKernel uses sensor_sources dict of callables (good!)

**Extension:**
```python
class SensorInterface:
    """Standard interface for all sensors"""

    def capture(self) -> SensorOutput:
        """Capture current sensor state"""
        raise NotImplementedError

    def get_metadata(self) -> Dict:
        """Sensor capabilities and status"""
        raise NotImplementedError

class VisionSensor(SensorInterface):
    def __init__(self, camera_device):
        self.camera = camera_device

    def capture(self):
        image = self.camera.read()
        return SensorOutput(
            data=image,
            timestamp=time.time(),
            quality=self._assess_quality(image),
            sensor_type='vision'
        )

# Plug into SAGEKernel
sage = UnifiedSAGESystem(
    sensor_sources={
        'vision': VisionSensor('/dev/video0').capture,
        'audio': AudioSensor('hw:1,0').capture,
        'proprioception': JointSensor().capture
    }
)
```

### 5. Effector Abstraction

**Current Gap:**
- No standardized effector interface
- Action results not fed back to loop

**Integration:**
```python
class EffectorInterface:
    """Standard interface for all effectors"""

    def execute(self, action: Any) -> EffectorResult:
        """Execute action and return result"""
        raise NotImplementedError

    def can_execute(self, action: Any) -> bool:
        """Check if action is executable"""
        raise NotImplementedError

class TTSEffector(EffectorInterface):
    def __init__(self, tts_plugin, audio_device):
        self.tts = tts_plugin
        self.audio = audio_device

    def execute(self, text):
        # Use TTS IRP plugin
        state = self.tts.init_state(x0={'text': text}, task_ctx={})
        state, _ = self.tts.step(state, budget=10.0)
        result = self.tts.extract(state)

        # Play audio
        self.audio.play(result['audio'], result['sample_rate'])

        return EffectorResult(
            success=True,
            latency=result['generation_time'],
            quality=1.0 - self.tts.energy(state)
        )

# Plug into system
sage.register_effector('tts', TTSEffector(neutts_plugin, speaker))
```

---

## Implementation Plan

### Phase 1: Core Integration (Week 1)
1. ✅ Map existing components (DONE)
2. ✅ Design architecture (THIS DOCUMENT)
3. ⏳ Create UnifiedSAGESystem class
4. ⏳ Integrate SAGECore H↔L reasoning
5. ⏳ Connect MetabolicController to orchestrator

### Phase 2: Memory & Sensors (Week 2)
1. Create UnifiedMemoryInterface
2. Implement SensorInterface abstraction
3. Implement EffectorInterface abstraction
4. Wire memory into cognition loop
5. Test sensor→memory→effector flow

### Phase 3: Testing & Validation (Week 2)
1. Simple scenario: Vision sensor → TTS response
2. Complex scenario: Multi-modal integration
3. Metabolic state transitions under load
4. Memory consolidation during DREAM state
5. ATP budget optimization

### Phase 4: Documentation & Push (Week 2)
1. Document integration architecture
2. Create usage examples
3. Performance benchmarking
4. Push to repository

---

## Key Design Decisions

### 1. Extend SAGEKernel, Don't Replace
**Rationale:** SAGEKernel already implements solid loop structure. Build on it rather than reinvent.

### 2. Async Plugin Execution
**Rationale:** Plugins can run concurrently when attention is distributed (WAKE state) or sequentially when focused (FOCUS state).

### 3. Metabolic State Controls Resource Policy
**Rationale:** Different states have different ATP budgets, plugin limits, and execution strategies. Controller provides policy, orchestrator enforces it.

### 4. Memory as Temporal Sensor
**Rationale:** Memory isn't just storage - it's a sensor that provides observations from the past. Treat it as first-class sensory input.

### 5. Trust Emerges from Behavior
**Rationale:** Don't manually set trust scores. Let them emerge from convergence speed, energy efficiency, and outcome quality.

---

## Success Criteria

**The unified system is successful when:**

1. ✅ All components integrated in single `UnifiedSAGESystem.run()` loop
2. ✅ SAGECore H↔L reasoning influences plugin selection
3. ✅ MetabolicController state affects resource allocation
4. ✅ Memory systems store and recall context
5. ✅ Sensors → SNARC → SAGECore → Orchestrator → Plugins → Effectors flows correctly
6. ✅ ATP budget managed across metabolic state transitions
7. ✅ Trust scores update based on plugin performance
8. ✅ System runs continuously without manual intervention
9. ✅ Can handle real sensor input (camera, microphone, joints)
10. ✅ Can produce real effector output (TTS, motor commands, display)

---

## Next Steps

1. Implement `UnifiedSAGESystem` class in `/sage/core/unified_sage_system.py`
2. Test with simple sensor→response scenario
3. Iterate based on discoveries
4. Document integration patterns
5. Push to repository

---

**Integration Status:** Architecture defined, implementation beginning

**The cognition loop is no longer conceptual - it's becoming real.**
