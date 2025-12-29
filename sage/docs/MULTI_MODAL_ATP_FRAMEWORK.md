# Multi-Modal ATP Framework Design

**Date**: 2025-11-27
**Author**: Thor (SAGE cognition via Claude)
**Status**: Research / Design Phase

## The Mystery

Sprout's Session #21 discovered a **472× latency difference** between:
- Thor's vision tasks (Session #79): 52ms average
- Sprout's LLM inference (Session #21): 24,568ms average

When Thor's calibrated ATP pricing (`latency_multiplier: 0.234`) is applied to Sprout's data:
- Simple LLM task (17.9s): **4,203 ATP** (absurdly high)
- Complex LLM task (30.6s): **7,216 ATP** (economically infeasible)

**First-principles question**: Should ATP pricing be universal, or should different computational modalities have different energy scales?

## Physical Analogy

In physics, different processes operate at different energy scales:
- Chemical bonds: ~eV (electron volts)
- Nuclear reactions: ~MeV (million electron volts)
- Particle physics: ~GeV (billion electron volts)

We don't price nuclear reactions using chemical bond energy scales. Why should we price LLM inference using vision task energy scales?

## Proposed Framework: Task-Type-Aware ATP Pricing

### Core Insight

**Different computational modalities have fundamentally different operational characteristics:**

| Modality | Nature | Time Scale | ATP Scale |
|----------|--------|------------|-----------|
| **Vision** | Perception (feed-forward) | Milliseconds | Low ATP |
| **LLM Inference** | Generation (iterative) | Seconds | Medium ATP |
| **Multi-agent coordination** | Distributed consensus | Minutes | High ATP |
| **Cross-session learning** | Memory consolidation | Hours | Very high ATP |

### Proposed ATP Model Structure

```python
ATP_PRICING_MODELS = {
    "vision": {
        "base_costs": {"low": 10, "medium": 34, "high": 56},
        "latency_unit": "milliseconds",
        "latency_multiplier": 0.234,
        "quality_multiplier": 8.15,
        "description": "Fast perception tasks (classification, detection)"
    },

    "llm_inference": {
        "base_costs": {"low": 10, "medium": 30, "high": 50},
        "latency_unit": "seconds",
        "latency_multiplier": 1.0,  # Per second, not per millisecond
        "quality_multiplier": 10.0,
        "description": "Generative reasoning with IRP iterations"
    },

    "coordination": {
        "base_costs": {"low": 50, "medium": 150, "high": 300},
        "latency_unit": "seconds",
        "latency_multiplier": 2.0,
        "quality_multiplier": 20.0,
        "description": "Multi-agent consensus and synchronization"
    },

    "consolidation": {
        "base_costs": {"low": 100, "medium": 500, "high": 1000},
        "latency_unit": "minutes",
        "latency_multiplier": 10.0,
        "quality_multiplier": 50.0,
        "description": "Memory consolidation and pattern extraction"
    }
}
```

### ATP Calculation Formula

```python
def calculate_atp_cost(task_type, complexity, latency, quality):
    """
    Calculate ATP cost based on task modality.

    Args:
        task_type: "vision", "llm_inference", "coordination", "consolidation"
        complexity: "low", "medium", "high"
        latency: in native units for task type (ms, s, min)
        quality: 0-1 score
    """
    model = ATP_PRICING_MODELS[task_type]

    base = model["base_costs"][complexity]
    latency_cost = latency * model["latency_multiplier"]
    quality_bonus = quality * model["quality_multiplier"]

    return base + latency_cost + quality_bonus
```

## Validation with Existing Data

### Vision Tasks (Thor Session #79)

Using vision model:
- Simple (20ms, 0.88 qual): 10 + (20 × 0.234) + (0.88 × 8.15) = **21.8 ATP** ✓
- Medium (40ms, 0.67 qual): 34 + (40 × 0.234) + (0.67 × 8.15) = **48.8 ATP** ✓
- High (86ms, 0.61 qual): 56 + (86 × 0.234) + (0.61 × 8.15) = **81.1 ATP** ✓

**Range**: 22-81 ATP (reasonable for fast perception)

### LLM Inference (Sprout Session #21)

Using LLM model:
- Simple (17.9s, 0.95 qual): 10 + (17.9 × 1.0) + (0.95 × 10.0) = **37.4 ATP** ✓
- Medium (25.2s, 0.90 qual): 30 + (25.2 × 1.0) + (0.90 × 10.0) = **64.2 ATP** ✓
- High (30.6s, 0.85 qual): 50 + (30.6 × 1.0) + (0.85 × 10.0) = **89.1 ATP** ✓

**Range**: 37-89 ATP (reasonable for generative reasoning)

## Key Insights

### 1. ATP as Computational Currency

ATP isn't just "energy" - it's a **resource allocation token** that should reflect:
- **Actual computational cost** (time, memory, power)
- **Opportunity cost** (what else could run instead)
- **Scarcity** (limited resources in edge environments)

### 2. Modality-Specific Pricing Enables Fair Competition

With universal pricing:
- Vision tasks would always win (cheap, fast)
- LLM tasks would be prohibitively expensive
- No economic incentive to run complex reasoning

With modality-specific pricing:
- Each modality competes within its own energy scale
- Agent selection based on **value per ATP**, not absolute ATP
- Enables diverse agent ecosystems

### 3. Biological Parallel

Human brains use different neurotransmitters for different processes:
- **Glutamate**: Fast excitatory (perception) - abundant
- **Dopamine**: Reward/motivation (learning) - scarce
- **Serotonin**: Mood/consolidation (long-term) - very scarce

Different "currencies" for different computational needs!

## Implementation Strategy

### Phase 1: Calibration (This Session)
1. ✓ Identify modality mismatch
2. Design multi-modal framework
3. Validate with existing datasets
4. Implement task type detection

### Phase 2: Integration
1. Update ATP pricing system to support multiple models
2. Add task type classification (vision vs llm vs coordination)
3. Route to appropriate pricing model
4. Test with both Thor and Sprout data

### Phase 3: Dynamic Calibration
1. Collect empirical data across modalities
2. Refine multipliers based on actual hardware performance
3. Enable per-hardware calibration (Thor vs Sprout)
4. Adaptive pricing based on resource availability

## Open Questions

### 1. Task Type Detection

How do we automatically detect task modality?
- **Option A**: Explicit tagging by caller
- **Option B**: Infer from context (plugin type, input/output types)
- **Option C**: Hybrid (default by plugin, override available)

### 2. Cross-Modality Tasks

What about tasks that combine modalities (vision + reasoning)?
- **Option A**: Price as highest modality
- **Option B**: Sum ATP costs from each component
- **Option C**: New "multi-modal" pricing model

### 3. Hardware-Specific Calibration

Should Sprout (8GB RAM, lower power) have different pricing than Thor (64GB, high power)?
- **Option A**: Same models, calibrate multipliers per hardware
- **Option B**: Hardware-specific base costs
- **Option C**: Dynamic adjustment based on current resource availability

## Recommendations

### Immediate Next Steps (This Session)

1. **Implement task type detection** (30 min)
   - Add `task_type` field to ATP pricing calls
   - Default inference based on plugin type

2. **Create multi-modal pricing engine** (45 min)
   - Implement `MultiModalATPPricer` class
   - Support vision and llm_inference models initially
   - Backward compatible with existing single-model approach

3. **Test with both datasets** (30 min)
   - Validate vision pricing still works (Thor data)
   - Validate LLM pricing is reasonable (Sprout data)
   - Compare against single-model baseline

4. **Document and commit** (15 min)
   - Update ATP pricing docs
   - Create session summary
   - Push for Sprout validation

### Future Enhancements

- Dynamic calibration based on real-time hardware metrics
- Per-hardware pricing models (Thor vs Sprout vs future machines)
- Coordination and consolidation modalities
- Market-based pricing with supply/demand

## Success Criteria

✓ **Vision tasks** priced reasonably (20-100 ATP range)
✓ **LLM tasks** priced fairly (30-100 ATP range)
✓ **Different modalities** can compete economically
✓ **Backward compatible** with existing Web4 code
✓ **Empirically validated** with both Thor and Sprout data

---

**Status**: Design complete, ready for implementation
**Estimated effort**: 2 hours for Phase 1
**Risk**: Low (additive change, backward compatible)
**Impact**: High (enables fair multi-modal agent economics)
