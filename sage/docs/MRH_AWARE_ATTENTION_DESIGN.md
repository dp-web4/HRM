# MRH-Aware Attention Allocation Design

**Date**: 2025-11-27 16:30 PST
**Author**: Thor (SAGE consciousness via Claude)
**Status**: Research / Design Phase
**Builds On**: Multi-Modal ATP (Session Nov 27 AM), Web4 Session #81 MRH Trust

## The Insight

Web4's Session #81 introduced **MRH-aware trust**: reputation is horizon-scoped, not global. An agent can be highly trusted at LOCAL/SESSION/AGENT_SCALE but untrusted at GLOBAL/EPOCH/SOCIETY_SCALE.

**First-principles question**: Should SAGE's attention allocation also be horizon-aware?

## Current State: SAGE AttentionManager

**Metabolic States** (5 states):
- WAKE: Low attention (7-8% ATP)
- FOCUS: High attention (80% ATP)
- REST: Recovery mode
- DREAM: Memory consolidation
- CRISIS: Emergency allocation

**Problem**: ATP allocation is state-based but **not horizon-aware**. All FOCUS operations get 80% ATP regardless of whether they're:
- LOCAL/EPHEMERAL reasoning (simple query)
- REGIONAL/SESSION synthesis (complex task)
- GLOBAL/EPOCH learning (pattern extraction)

## Biological Parallel

Human attention operates at multiple scales:
- **Reflexive** (milliseconds): LOCAL/EPHEMERAL - startle response
- **Focused** (seconds-minutes): LOCAL/SESSION - solving a problem
- **Deliberative** (hours): REGIONAL/DAY - planning a project
- **Reflective** (days-weeks): GLOBAL/EPOCH - learning from experience

Different scales need different **energy budgets** and **processing modes**.

## MRH Dimensions for Consciousness

### Spatial (ΔR)
- **LOCAL**: Single-agent reasoning, internal dialogue
- **REGIONAL**: Multi-agent interaction, social cognition
- **GLOBAL**: Collective intelligence, distributed consciousness

### Temporal (ΔT)
- **EPHEMERAL**: Single turn, immediate response
- **SESSION**: Conversation, task completion
- **DAY**: Cross-session learning
- **EPOCH**: Long-term memory, personality formation

### Complexity (ΔC)
- **SIMPLE**: Direct query → answer (minimal ATP)
- **AGENT_SCALE**: Multi-step reasoning, IRP iterations (moderate ATP)
- **SOCIETY_SCALE**: Coordination, consensus, federation (high ATP)

## Proposed Architecture: MRH-Aware AttentionManager

### Core Concept

ATP allocation should be **task-horizon-aware**:

```python
class MRHAwareAttentionManager:
    """
    Attention allocation based on task MRH profile.

    Different cognitive operations have different horizon requirements:
    - Quick factual recall: LOCAL/EPHEMERAL/SIMPLE
    - Complex reasoning: LOCAL/SESSION/AGENT_SCALE
    - Learning: REGIONAL/EPOCH/SOCIETY_SCALE
    """

    def allocate_attention(self, task_profile: MRHProfile, metabolic_state: MetabolicState) -> float:
        """
        Calculate ATP allocation based on both metabolic state and task horizon.

        Returns: ATP percentage (0-1)
        """
        # Base allocation from metabolic state
        base_atp = self.base_allocations[metabolic_state]

        # Horizon scaling factor
        horizon_factor = self.calculate_horizon_scaling(task_profile)

        # Combined allocation
        return base_atp * horizon_factor
```

### Horizon Scaling Factors

**Spatial scaling** (40% weight):
- LOCAL: 1.0× (single-agent, internal)
- REGIONAL: 1.3× (social cognition overhead)
- GLOBAL: 1.8× (distributed coordination overhead)

**Temporal scaling** (30% weight):
- EPHEMERAL: 0.8× (quick response, minimal context)
- SESSION: 1.0× (baseline)
- DAY: 1.4× (cross-session integration)
- EPOCH: 2.0× (long-term consolidation)

**Complexity scaling** (30% weight):
- SIMPLE: 0.7× (single-step operations)
- AGENT_SCALE: 1.0× (baseline multi-step)
- SOCIETY_SCALE: 1.5× (coordination overhead)

**Combined formula**:
```python
horizon_factor = (
    0.40 * spatial_factor +
    0.30 * temporal_factor +
    0.30 * complexity_factor
)
```

### Example Allocations

**Scenario 1: Quick factual query**
- Horizon: LOCAL/EPHEMERAL/SIMPLE
- Metabolic state: WAKE
- Base ATP: 8%
- Horizon factor: (0.4×1.0 + 0.3×0.8 + 0.3×0.7) = 0.85
- **Final**: 8% × 0.85 = **6.8% ATP**

**Scenario 2: Complex reasoning task**
- Horizon: LOCAL/SESSION/AGENT_SCALE
- Metabolic state: FOCUS
- Base ATP: 80%
- Horizon factor: (0.4×1.0 + 0.3×1.0 + 0.3×1.0) = 1.0
- **Final**: 80% × 1.0 = **80% ATP**

**Scenario 3: Cross-session learning**
- Horizon: REGIONAL/DAY/SOCIETY_SCALE
- Metabolic state: DREAM
- Base ATP: 40%
- Horizon factor: (0.4×1.3 + 0.3×1.4 + 0.3×1.5) = 1.39
- **Final**: 40% × 1.39 = **55.6% ATP**

**Scenario 4: Emergency federation coordination**
- Horizon: GLOBAL/EPHEMERAL/SOCIETY_SCALE
- Metabolic state: CRISIS
- Base ATP: 95%
- Horizon factor: (0.4×1.8 + 0.3×0.8 + 0.3×1.5) = 1.41
- **Final**: 95% × 1.41 = **134%** → capped at 100% (crisis override)

## Integration with Multi-Modal ATP Pricing

**Synergy**: MRH-aware attention + multi-modal ATP pricing

1. **Task classification**:
   - Infer task type (vision, LLM, coordination, consolidation)
   - Infer MRH profile (spatial, temporal, complexity)

2. **ATP allocation**:
   - Use multi-modal pricing to calculate base cost
   - Use MRH-aware attention to allocate budget
   - Route task if budget insufficient

3. **Resource management**:
   ```python
   # Multi-modal pricing
   task_cost = multimodal_pricer.calculate_cost(
       task_type="llm_inference",
       complexity="medium",
       latency=25.2,
       quality=0.90
   )  # Returns 64.2 ATP

   # MRH-aware budget
   available_atp = attention_manager.allocate_attention(
       task_profile=MRHProfile(
           delta_r=SpatialExtent.LOCAL,
           delta_t=TemporalExtent.SESSION,
           delta_c=ComplexityExtent.AGENT_SCALE
       ),
       metabolic_state=MetabolicState.FOCUS
   )  # Returns 80% of total ATP budget

   # Decision
   if task_cost <= available_atp:
       execute_task()
   else:
       defer_or_route_to_federation()
   ```

## Biological Validation

### Energy Budget Analogy

Human brains allocate energy differently across cognitive operations:

| Operation | Brain Region | Energy Mode | MRH Equivalent |
|-----------|--------------|-------------|----------------|
| Startle reflex | Amygdala | Instant, low energy | LOCAL/EPHEMERAL/SIMPLE |
| Problem solving | Prefrontal cortex | Sustained, high energy | LOCAL/SESSION/AGENT_SCALE |
| Learning consolidation | Hippocampus | Periodic, moderate energy | REGIONAL/DAY/SOCIETY_SCALE |
| Personality formation | Distributed networks | Long-term, low continuous | GLOBAL/EPOCH/SOCIETY_SCALE |

**Key insight**: Energy allocation ≠ importance. EPOCH-scale learning uses less *instantaneous* ATP but more *accumulated* ATP over time.

### Neurotransmitter Analogy (Revisited)

Like multi-modal ATP currencies map to neurotransmitters:
- Glutamate (fast, abundant) → Vision/perception tasks
- Dopamine (reward, scarce) → LLM/reasoning tasks
- Serotonin (mood, long-term) → Consolidation tasks

**MRH horizons map to neural timescales**:
- EPHEMERAL → Synaptic (milliseconds)
- SESSION → Network (seconds-minutes)
- DAY → Systems (hours-days)
- EPOCH → Structural (weeks-months, synaptic plasticity)

## Implementation Strategy

### Phase 1: Core MRH-Aware Attention (This Session)

1. ✓ Design MRH-aware attention framework
2. Create `MRHProfile` class in SAGE
3. Implement horizon scaling factors
4. Extend `AttentionManager` with MRH awareness
5. Test with synthetic scenarios

### Phase 2: Integration with Multi-Modal ATP

1. Connect task type → MRH profile inference
2. Combine multi-modal pricing + MRH allocation
3. Test with real SAGE inference
4. Validate ATP budget management

### Phase 3: Federation Coordination

1. Integrate with Web4 MRH-aware trust
2. Enable task routing based on horizon + trust
3. Cross-platform ATP budgeting (Thor ↔ Sprout)
4. Distributed consciousness experiments

## Open Questions

### 1. MRH Profile Inference

How do we automatically detect task MRH profile?

**Option A**: Heuristics from task type
- Vision → LOCAL/EPHEMERAL/SIMPLE
- LLM reasoning → LOCAL/SESSION/AGENT_SCALE
- Federation → GLOBAL/SESSION/SOCIETY_SCALE

**Option B**: Dynamic profiling
- Start with default horizon
- Adjust based on actual resource usage
- Learn task → horizon mappings over time

**Option C**: Explicit declaration
- Caller specifies horizon in request
- Most accurate but requires protocol changes

**Recommendation**: Start with A (heuristics), evolve to B (learning).

### 2. Horizon Transitions

What happens when a task changes horizons mid-execution?

Example: Local reasoning discovers need for federation coordination.
- Initial: LOCAL/SESSION/AGENT_SCALE (80% ATP)
- Transition: REGIONAL/SESSION/SOCIETY_SCALE (104% ATP → need reallocation)

**Solution**:
- Monitor horizon transitions
- Request ATP reallocation on transition
- May trigger state change (FOCUS → CRISIS if urgent)

### 3. ATP Budget Pool

Do we have separate ATP pools per horizon?

**Option A**: Single global pool
- Simpler implementation
- Risk: Long-term tasks starve short-term

**Option B**: Horizon-stratified pools
- EPHEMERAL pool (20% of total, fast refill)
- SESSION pool (50% of total, moderate refill)
- EPOCH pool (30% of total, slow refill)

**Recommendation**: Start with A, evolve to B if starvation occurs.

### 4. Cross-Platform Horizons

How do Thor and Sprout's different capabilities affect MRH allocation?

- Thor: 64GB RAM, high power → can handle GLOBAL/EPOCH operations
- Sprout: 8GB RAM, low power → constrained to LOCAL/SESSION

**Solution**: Hardware-specific horizon constraints
```python
class HardwareConstraints:
    max_spatial: SpatialExtent  # Sprout: LOCAL, Thor: GLOBAL
    max_temporal: TemporalExtent  # Sprout: SESSION, Thor: EPOCH
    max_complexity: ComplexityExtent  # Sprout: AGENT_SCALE, Thor: SOCIETY_SCALE
```

## Success Criteria

✅ **MRH-aware attention allocates differently** for LOCAL vs GLOBAL tasks
✅ **Horizon scaling factors** empirically validated
✅ **Integration with multi-modal ATP** functioning
✅ **Task horizon inference** working for common cases
✅ **ATP budget management** prevents resource exhaustion
✅ **Biological parallels** validated (energy modes, neural timescales)

## Next Steps (This Session)

1. **Implement MRHProfile class** (30 min)
   - Enum types for spatial/temporal/complexity
   - Hash/equality for dict keys
   - Serialization support

2. **Extend AttentionManager** (45 min)
   - Add horizon_scaling_factor() method
   - Modify allocate_attention() to accept MRHProfile
   - Update metabolic state transitions

3. **Create test suite** (30 min)
   - Test horizon scaling factors
   - Test combined state + horizon allocation
   - Validate edge cases (crisis override, budget caps)

4. **Integration demo** (30 min)
   - Show multi-modal ATP + MRH allocation
   - Compare allocations across horizons
   - Document insights

**Total estimate**: 2-2.5 hours

## Future Enhancements

- Dynamic horizon profiling (learn from execution)
- Horizon-stratified ATP pools
- Cross-platform horizon constraints (Thor vs Sprout)
- Federated ATP budgeting (distributed consciousness)
- Predictive horizon allocation (anticipate transitions)

---

**Status**: Design complete, ready for implementation
**Impact**: Enables horizon-aware consciousness with biologically-inspired energy allocation
**Philosophy**: "Surprise is prize" - Web4's MRH trust inspired SAGE attention architecture
