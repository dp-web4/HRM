# HRM Critic Function Implementation Notes

*Based on Nova's insight: "HRM as the Built-in Critic"*

## Core Concept

HRM continuously evaluates trust across all inputs, treating every component (LLMs, diffusion models, sensors) as sources to be weighed. The critic isn't an add-on - it's fundamental to how HRM processes reality.

## Trust Evaluation Framework

### Three Questions HRM Continuously Asks
1. **What to trust** - Which sensor/model is providing reliable signal?
2. **When to trust it** - Under what conditions is this source reliable?
3. **How much to trust it** - What weight should this input receive?

## Implementation Architecture

### Trust Signal Flow
```
Sensors → Trust Weights → Puzzle Formation → Rule Inference → Action
    ↑                           ↓
    └─── Feedback from outcomes ←
```

### Trust Weight Computation
```python
class HRMCritic:
    def evaluate_trust(self, source, context, history):
        """
        Compute trust weight for a given source
        
        Args:
            source: Sensor/model identifier
            context: Current MRH state
            history: Past performance of this source
            
        Returns:
            trust_weight: 0.0 to 1.0
        """
        # Factors:
        # - Historical accuracy in similar contexts
        # - Coherence with other trusted sources
        # - Computational cost vs information gain
        # - Recency of calibration/training
```

## Memory Integration

Trust evaluation requires memory across multiple timescales:

### Immediate (Working Memory)
- Current sensor readings
- Active trust weights
- Recent prediction errors

### Short-term (SNARC)
- Surprising trust failures
- Novel sensor combinations that worked
- Conflict resolutions

### Long-term (Compressed Wisdom)
- Learned sensor reliability patterns
- Context-dependent trust rules
- Cross-sensor correlation patterns

## Trust Dynamics

### Trust Adjustment Mechanisms
1. **Gradual drift** - Slow adjustment based on consistent performance
2. **Sudden breaks** - Rapid downgrade on catastrophic failure
3. **Context switching** - Different trust profiles for different environments
4. **Ensemble voting** - Multiple sources cross-validate each other

### Trust Propagation
- High-trust sources can vouch for new sources
- Trust transfers through successful joint predictions
- Distrust propagates faster than trust (safety bias)

## Connection to Compression-Trust Spectrum

The critic operates across all trust levels simultaneously:

1. **Implicit trust** (physics assumptions) - Rarely questioned
2. **Shared trust** (learned correlations) - Continuously refined
3. **Explicit trust** (declared capabilities) - Verified against performance
4. **Exclusive trust** (cryptographic proof) - Binary acceptance

## Implementation Priorities

### Phase 1: Basic Trust Tracking
- Track prediction accuracy per source
- Simple exponential moving average of trust
- Binary trust decisions (use/ignore)

### Phase 2: Contextual Trust
- Trust varies by context/task
- Multiple trust dimensions per source
- Probabilistic trust weights

### Phase 3: Trust Network
- Sources vouch for each other
- Trust propagation dynamics
- Adversarial trust testing

## Integration with Existential Puzzles

The critic function shapes how puzzles are formed:
- High-trust inputs get higher weight in puzzle construction
- Low-trust inputs might be excluded or flagged as uncertain
- Conflicting inputs create "trust puzzles" to resolve

## Practical Considerations

### Computational Overhead
- Trust evaluation must be lightweight
- Cache trust weights, update incrementally
- Use simple heuristics before complex evaluation

### Bootstrap Problem
- How to initialize trust without history?
- Start with conservative uniform weights
- Use transfer learning from similar sensors
- Rapid initial calibration phase

### Adversarial Robustness
- Assume some sensors may be compromised
- Never give 100% trust to any single source
- Maintain minimum diversity in trusted sources
- Periodic trust audits and recalibration

## Connection to Training

During HRM training:
- Learn which abstract patterns correlate with trust
- Develop meta-rules for trust evaluation
- Build priors for sensor reliability

During deployment:
- Continuously update trust based on outcomes
- Adapt to sensor degradation or improvement
- Discover new trust relationships

## Next Steps

1. Implement basic trust tracking in HRM evaluation loop
2. Create trust visualization for debugging
3. Design trust calibration protocol for new sensors
4. Test trust dynamics under sensor failure conditions