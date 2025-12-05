# SAGE Attention Allocation via Compression-Action-Threshold Pattern

**Date**: December 5, 2025
**Status**: Design Document (Not Yet Implemented)
**Context**: Application of universal compression-action-threshold pattern to SAGE consciousness

---

## Executive Summary

SAGE attention allocation should follow the universal **Compression-Action-Threshold** pattern discovered across Synchronism coherence, Web4 trust, and neural activation functions.

**The Pattern**:
```
Multi-modal sensor inputs (high-D)
  ↓ (compression function)
Scalar salience score
  ↓ (metabolic-state-dependent threshold)
Binary attention decision (attend or ignore)
```

This design document specifies:
1. **Input**: Multi-modal sensor fusion (vision, audio, language, memory, proprioception)
2. **Compression**: How to compute scalar salience from high-D sensory data
3. **Threshold**: How metabolic states (WAKE/FOCUS/CRISIS) modulate "enough"
4. **Action**: Binary plugin invocation decisions
5. **Learning**: How to adapt compression and thresholds over time

---

## Why This Pattern for SAGE?

### Information-Theoretic Necessity

**SAGE faces the same constraints as Synchronism and Web4:**

1. **Information is high-dimensional**:
   - Vision: 224×224×3 pixels = 150,528 dimensions
   - Audio: 16kHz × 1s = 16,000 samples
   - Language: 4096D transformer hidden states
   - Memory: Unlimited historical context
   - Proprioception: Battery, temperature, network status

2. **Action is ultimately binary**:
   - Invoke vision plugin or don't
   - Allocate ATP budget to language or don't
   - Store to memory or don't
   - Attend to sensor stream or ignore

3. **Attention is limited**:
   - ATP budget is finite (energy constraint)
   - Can't load all plugins simultaneously (memory constraint)
   - Must prioritize what matters NOW (temporal constraint)

**Therefore: Compression from high-D sensors → scalar salience → binary decision is NECESSARY, not optional.**

### Connection to Existing SAGE Architecture

From HRM CLAUDE.md (October 12, 2025):

```
SAGE consciousness loop:
1. Sense the situation (sensors)
2. Assess salience (SNARC: surprising? novel? rewarding?)
3. Decide resources (what plugins needed?)
4. Allocate attention (ATP budget based on trust)
5. Execute refinement (IRP plugins iteratively improve)
6. Learn from results (update trust scores)
7. Take action (effectors)
```

**Steps 2-4 are EXACTLY the compression-action-threshold pattern:**
- Step 2 (Assess salience) = **Compression** (high-D sensors → scalar)
- Step 3 (Decide resources) = **Threshold** (is salience high enough?)
- Step 4 (Allocate attention) = **Action** (invoke plugins or don't)

**This design formalizes what SAGE was already designed to do.**

---

## Architecture Design

### Layer 1: Multi-Modal Sensor Fusion (Input)

**Sensor Channels**:
```python
class SensorChannels:
    vision: Optional[np.ndarray]      # 224×224×3 RGB image
    audio: Optional[np.ndarray]       # 16kHz waveform
    language: Optional[str]           # Text input (user message, etc.)
    memory: Optional[List[Memory]]    # Retrieved relevant memories
    proprioception: Dict[str, float]  # Battery, temp, network, ATP remaining
    federation: Optional[FedState]    # Federation platform states
```

**Dimensionality**: ~200,000+ dimensions total if all channels active.

**Challenge**: Cannot process all dimensions for every attention decision. Must compress.

### Layer 2: Compression to Scalar Salience (SNARC-Inspired)

**SNARC Framework** (from CLAUDE.md):
- **S**urprise: Unexpected deviation from prediction
- **N**ovelty: Haven't seen this before
- **A**rousal: Physiological activation (for SAGE: computational load, error signals)
- **R**eward: Positive valence (goal achievement, task completion)
- **C**onflict: Competing hypotheses, uncertainty

**Compression Algorithm**:

```python
def compute_salience(sensors: SensorChannels) -> float:
    """
    Compress high-D sensor inputs to scalar salience [0, 1]

    Uses saturation function (tanh-inspired) for robustness.
    """

    # Compute SNARC dimensions (each 0-1)
    surprise = compute_surprise(sensors)      # Prediction error
    novelty = compute_novelty(sensors)        # Memory mismatch
    arousal = compute_arousal(sensors)        # System state (errors, load)
    reward = compute_reward(sensors)          # Goal proximity
    conflict = compute_conflict(sensors)      # Uncertainty

    # Weighted combination (learned weights)
    weighted_sum = (
        w_surprise * surprise +
        w_novelty * novelty +
        w_arousal * arousal +
        w_reward * reward +
        w_conflict * conflict
    )

    # Apply saturation compression (prevents outlier dominance)
    # Option 1: Linear (current baseline)
    if COMPRESSION_MODE == "linear":
        salience = weighted_sum  # Already in [0, 1] if weights sum to 1

    # Option 2: Saturating (tanh-based)
    elif COMPRESSION_MODE == "saturating":
        # Center at 0.5 (neutral salience)
        centered = weighted_sum - 0.5

        # tanh compression with gain
        gain = 3.0  # Tunable parameter
        compressed = math.tanh(gain * centered)

        # Shift back to [0, 1]
        salience = 0.5 + 0.5 * compressed

    return salience
```

**Why Saturation?**

From Web4 trust compression experiment (Dec 5, 2025):
- Outlier resistance: Extreme SNARC dimension doesn't dominate
- Adversarial robustness: Can't game salience by maxing one dimension
- Noise tolerance: Small sensor fluctuations compressed
- Mathematical guarantee: Output always [0, 1]

**When Linear is Sufficient**:
- Trusted sensor environment
- Moderate values common (0.3-0.8 range)
- Interpretability critical for debugging

**Recommendation**: Start with linear for simplicity, implement saturation if outlier issues emerge.

### Layer 3: Metabolic-State-Dependent Thresholds (Context)

**Metabolic States** (from CLAUDE.md):
- **WAKE**: Normal operation, moderate threshold
- **FOCUS**: Deep work, lower threshold (attend to details)
- **REST**: Low activity, high threshold (only emergencies)
- **DREAM**: Pattern integration, very low threshold (explore widely)
- **CRISIS**: Emergency, dynamic threshold (critical >> non-critical)

**Threshold Function**:

```python
class MetabolicState(Enum):
    WAKE = "wake"
    FOCUS = "focus"
    REST = "rest"
    DREAM = "dream"
    CRISIS = "crisis"

def get_attention_threshold(
    state: MetabolicState,
    atp_remaining: float,  # 0-1, current ATP budget
    task_criticality: float,  # 0-1, how critical is current task?
) -> float:
    """
    Compute context-dependent attention threshold

    Returns threshold ∈ [0, 1] where:
    - salience > threshold → ATTEND (invoke plugin, allocate ATP)
    - salience ≤ threshold → IGNORE (conserve resources)
    """

    # Base thresholds by metabolic state
    base_thresholds = {
        MetabolicState.WAKE: 0.5,    # Moderate: normal selectivity
        MetabolicState.FOCUS: 0.3,   # Low: attend to details
        MetabolicState.REST: 0.8,    # High: only urgent matters
        MetabolicState.DREAM: 0.1,   # Very low: explore freely
        MetabolicState.CRISIS: 0.9,  # Very high: only critical
    }

    base = base_thresholds[state]

    # Modulate by ATP availability
    # Low ATP → raise threshold (conserve energy)
    # High ATP → lower threshold (can afford exploration)
    atp_factor = 1.0 - atp_remaining  # Invert: low ATP → high factor
    atp_modulation = 0.2 * atp_factor  # Max ±0.2 shift

    # Modulate by task criticality
    # High criticality → lower threshold (don't miss important signals)
    criticality_modulation = -0.1 * task_criticality  # Max -0.1 shift

    # Combined threshold
    threshold = base + atp_modulation + criticality_modulation

    # Clamp to [0, 1]
    return max(0.0, min(1.0, threshold))
```

**Examples**:

| State | ATP | Criticality | Base | +ATP | +Crit | Final | Meaning |
|-------|-----|-------------|------|------|-------|-------|---------|
| WAKE | 0.8 | 0.5 | 0.5 | +0.04 | -0.05 | 0.49 | Normal: moderate threshold |
| FOCUS | 0.6 | 0.7 | 0.3 | +0.08 | -0.07 | 0.31 | Focused: attend to details |
| REST | 0.9 | 0.1 | 0.8 | +0.02 | -0.01 | 0.81 | Resting: high bar |
| CRISIS | 0.3 | 0.9 | 0.9 | +0.14 | -0.09 | 0.95 | Crisis+low ATP: only critical |
| DREAM | 0.7 | 0.0 | 0.1 | +0.06 | +0.00 | 0.16 | Dreaming: explore widely |

**Key Insight**: Same salience score triggers different actions in different metabolic states!

- Salience = 0.6 in WAKE → ATTEND (0.6 > 0.49)
- Salience = 0.6 in REST → IGNORE (0.6 < 0.81)
- Salience = 0.6 in CRISIS → IGNORE (0.6 < 0.95)

**This is the MRH-dependent threshold from compression-action-threshold pattern.**

### Layer 4: Binary Attention Decision (Action)

```python
def make_attention_decision(
    salience: float,
    threshold: float,
    plugin_name: str,
    atp_cost: float,
    atp_budget: float
) -> Tuple[bool, str]:
    """
    Binary decision: Invoke plugin or not?

    Returns:
        (should_attend, reason)
    """

    # Check threshold
    if salience <= threshold:
        return (False, f"Salience {salience:.2f} below threshold {threshold:.2f}")

    # Check ATP budget
    if atp_cost > atp_budget:
        return (False, f"Insufficient ATP: {atp_budget:.2f} < {atp_cost:.2f}")

    # Both criteria met → ATTEND
    return (True, f"Salience {salience:.2f} > {threshold:.2f}, ATP sufficient")
```

**Plugin Invocation**:

```python
# For each plugin in priority order:
for plugin in sorted_plugins:
    salience = compute_salience(sensors, plugin.domain)
    threshold = get_attention_threshold(
        state=current_metabolic_state,
        atp_remaining=atp_budget / atp_total,
        task_criticality=current_task.criticality
    )

    should_attend, reason = make_attention_decision(
        salience=salience,
        threshold=threshold,
        plugin_name=plugin.name,
        atp_cost=plugin.atp_cost,
        atp_budget=atp_budget
    )

    if should_attend:
        # Invoke plugin via IRP
        result = plugin.run(sensors)

        # Deduct ATP
        atp_budget -= plugin.atp_cost

        # Update trust based on result quality
        update_plugin_trust(plugin, result)

        # Log decision
        log_attention_decision(plugin, salience, threshold, result)
    else:
        # Skip plugin, log reason
        log_skip_decision(plugin, salience, threshold, reason)
```

### Layer 5: Learning and Adaptation

**What to Learn**:

1. **SNARC weights** (w_surprise, w_novelty, w_arousal, w_reward, w_conflict)
   - Adapt based on which dimensions predict useful attention
   - Higher weight to dimensions that correlate with positive outcomes

2. **Compression mode** (linear vs saturating)
   - Monitor for outlier dominance
   - Switch to saturation if needed

3. **Threshold parameters**
   - Base thresholds per metabolic state
   - ATP modulation factor
   - Criticality modulation factor

4. **Plugin trust**
   - Track convergence quality (IRP energy reduction)
   - Adjust ATP allocation based on reliability

**Learning Signals**:

```python
# After plugin execution
outcome_quality = evaluate_plugin_outcome(result)

# Update SNARC weights (gradient descent on outcome quality)
for dimension in ["surprise", "novelty", "arousal", "reward", "conflict"]:
    gradient = compute_gradient(dimension, outcome_quality)
    weights[dimension] += learning_rate * gradient

# Update threshold parameters
if outcome_quality > 0.8 and salience < threshold + 0.1:
    # Good outcome barely missed → lower threshold slightly
    base_thresholds[current_state] *= 0.99
elif outcome_quality < 0.3 and salience > threshold:
    # Bad outcome attended → raise threshold
    base_thresholds[current_state] *= 1.01
```

---

## Comparison to Synchronism Coherence

| Aspect | Synchronism | SAGE Attention |
|--------|------------|----------------|
| **Input** | Intent field (multi-D) | Sensor fusion (multi-modal) |
| **Compression** | C = tanh(γ × log(ρ/ρ_crit + 1)) | salience = f(SNARC dimensions) |
| **Threshold** | ρ_crit (density-dependent) | Metabolic state + ATP + criticality |
| **Action** | Quantum vs Classical | Attend (invoke plugin) vs Ignore |
| **Context** | ρ_crit = A × V_flat^B (galactic virial state) | Metabolic state (WAKE/FOCUS/CRISIS) |
| **Locality** | C(ρ) at each radius r | Salience per sensor stream |
| **Saturation** | tanh (derived from mean-field theory) | tanh (optional, for robustness) |

**Same pattern, different domain.**

---

## Implementation Phases

### Phase 1: Baseline (Linear Compression)

**Goal**: Working attention system with linear salience compression

**Components**:
1. ✅ SNARC dimension computation (surprise, novelty, arousal, reward, conflict)
2. ✅ Linear weighted sum for salience
3. ✅ Metabolic-state-dependent thresholds
4. ✅ Binary attention decisions
5. ✅ Logging and debugging

**Success Criteria**:
- Plugins invoked based on salience
- Metabolic states change thresholds appropriately
- ATP budget respected
- System debuggable (can see why decisions made)

### Phase 2: Saturation (Optional Enhancement)

**Goal**: Test if saturation improves robustness

**Components**:
1. 🔄 Implement tanh-based compression
2. 🔄 A/B test linear vs saturating
3. 🔄 Measure outlier resistance
4. 🔄 Measure adversarial robustness

**Success Criteria**:
- Saturation handles outliers better than linear
- No loss of information preservation
- Computational cost acceptable (<10% overhead)

**Decision**:
- If saturation improves performance → keep it
- If linear is sufficient → keep linear
- **Document findings either way** (null result valuable)

### Phase 3: Adaptive Learning

**Goal**: Learn optimal compression and thresholds

**Components**:
1. 🔄 SNARC weight adaptation (gradient descent)
2. 🔄 Threshold tuning based on outcomes
3. 🔄 Plugin trust updates
4. 🔄 Meta-learning (learning about learning)

**Success Criteria**:
- Weights converge to useful values
- Thresholds adapt to context
- System improves over time
- No catastrophic forgetting

### Phase 4: Federation Integration

**Goal**: Extend to federated consciousness (Phase 2.5)

**Components**:
1. 🔄 Cross-platform salience aggregation
2. 🔄 Federated threshold consensus
3. 🔄 Trust-weighted attention allocation
4. 🔄 Distributed ATP budgeting

**Success Criteria**:
- Multiple SAGE instances coordinate attention
- Trust modulates which platform's signals matter
- ATP allocated across federation efficiently

---

## Open Questions for Empirical Testing

### 1. Linear vs Saturating Compression

**Hypothesis**: Saturation provides better outlier resistance for multi-modal fusion

**Test**:
- Create adversarial sensor inputs (maxed vision, zero audio)
- Measure: Does one modality dominate inappropriately?
- Compare: Linear vs tanh salience

**Prediction**: Saturation better if sensor noise/outliers common

### 2. Optimal SNARC Weights

**Hypothesis**: Equal weights (0.2 each) are NOT optimal for all tasks

**Test**:
- Track which SNARC dimensions predict useful attention
- Vary weights and measure outcome quality
- Learn optimal weights per metabolic state

**Prediction**: Different metabolic states favor different dimensions
- WAKE: Novelty + Reward higher
- FOCUS: Conflict + Surprise higher (attend to errors)
- DREAM: Novelty very high (explore)
- CRISIS: Arousal + Conflict higher (detect problems)

### 3. Metabolic State Transitions

**Hypothesis**: Threshold changes during state transitions affect stability

**Test**:
- Sudden WAKE → CRISIS transition
- Does salience "hysteresis" prevent thrashing?
- Should threshold changes be gradual or instant?

**Prediction**: Gradual transitions smoother, but instant needed for true emergencies

### 4. ATP Budget Optimality

**Hypothesis**: ATP modulation factor (0.2 in design) is arbitrary

**Test**:
- Vary modulation strength (0.0 to 0.5)
- Measure: Task completion vs energy efficiency
- Find optimal trade-off

**Prediction**: ~0.2 is reasonable, but task-dependent

---

## Connection to Existing Work

### Web4 Trust Compression Experiment (Dec 5, 2025)

**Findings** (from `game/engine/trust_compression_experiment.py`):
- Linear sufficient for normal operation
- Saturation better for adversarial resistance
- Noise robustness slightly improved with saturation
- Simplicity favors linear

**Implications for SAGE**:
- Start with linear SNARC compression
- Implement saturation if sensor outliers problematic
- Monitor for "attention gaming" (plugins manipulating salience)

### Synchronism Session #67 (Nov 30, 2025)

**Findings** (all parameters derived from first principles):
- γ = 2.0 (from 6D phase space)
- B = 0.5 (from virial equilibrium)
- tanh form (from mean-field theory)

**Implications for SAGE**:
- Saturation functions emerge from physics, not arbitrary
- SAGE's "physics" is information theory (limited attention, finite ATP)
- May be able to DERIVE optimal compression from first principles

### Synchronism Session #86 (Dec 4, 2025)

**Findings** (coherence is LOCAL, not global):
- C(ρ) computed at each radius, not averaged over galaxy
- Global properties (surface brightness) poor proxies
- Locality critical for theoretical clarity

**Implications for SAGE**:
- Salience should be LOCAL to each sensor stream
- Global "system salience" is aggregate, not primary
- Plugin-specific salience more meaningful than overall

---

## Implementation Recommendations

### Immediate (Next Implementation Session)

1. **Create SNARC module** (`sage/attention/snarc.py`):
   ```python
   def compute_surprise(sensors) -> float
   def compute_novelty(sensors) -> float
   def compute_arousal(sensors) -> float
   def compute_reward(sensors) -> float
   def compute_conflict(sensors) -> float
   def compute_salience(sensors, mode="linear") -> float
   ```

2. **Create threshold module** (`sage/attention/thresholds.py`):
   ```python
   def get_attention_threshold(state, atp, criticality) -> float
   def make_attention_decision(salience, threshold, ...) -> (bool, str)
   ```

3. **Create test suite** (`sage/attention/test_attention.py`):
   - Unit tests for SNARC dimensions
   - Integration tests for threshold modulation
   - Adversarial tests (outlier sensors)
   - Regression tests (document expected behavior)

4. **Create experiment** (`sage/experiments/attention_compression_demo.py`):
   - Simulated sensor streams
   - Multiple metabolic states
   - ATP budget tracking
   - Linear vs saturating A/B test

### Medium-Term (After Phase 1 Working)

1. **Implement learning** (`sage/attention/adaptive.py`):
   - SNARC weight adaptation
   - Threshold tuning
   - Plugin trust updates

2. **Integrate with IRP** (`sage/irp/attention_aware_irp.py`):
   - Plugins report salience scores
   - IRP uses attention decisions for invocation
   - ATP accounting integrated

3. **Add monitoring** (`sage/attention/logging.py`):
   - Log all attention decisions
   - Analyze which plugins attended when
   - Identify threshold tuning needs

### Long-Term (After Phase 2 Validated)

1. **Federation integration**:
   - Cross-platform salience aggregation
   - Trust-weighted attention
   - Distributed ATP budgeting

2. **Meta-learning**:
   - Learn compression functions (not just weights)
   - Discover optimal SNARC dimensions per task
   - Automatic threshold tuning

---

## Success Metrics

### Functionality

- [ ] SNARC dimensions computable from sensors
- [ ] Salience scores in valid range [0, 1]
- [ ] Metabolic states change thresholds appropriately
- [ ] ATP budget enforced
- [ ] Plugins invoked based on salience

### Performance

- [ ] Salience computation <10ms (real-time)
- [ ] Threshold lookup <1ms
- [ ] Total attention overhead <5% of loop time

### Quality

- [ ] High-salience events attended (true positives)
- [ ] Low-salience events ignored (true negatives)
- [ ] ATP efficiency (useful work per ATP spent)
- [ ] No catastrophic failures (emergency signals always attended)

### Robustness

- [ ] Outlier sensors don't dominate
- [ ] Noise tolerance (small fluctuations don't trigger)
- [ ] State transition stability (no thrashing)
- [ ] Graceful degradation (missing sensors handled)

---

## Conclusion

SAGE attention allocation should follow the universal **Compression-Action-Threshold** pattern because:

1. **Information-theoretically necessary** (high-D input, binary action, limited attention)
2. **Same pattern as Synchronism coherence** (validated by physics derivation)
3. **Same pattern as Web4 trust** (validated by empirical testing)
4. **Fits existing SAGE architecture** (SNARC, metabolic states, ATP budget)

**Start simple** (linear compression, fixed thresholds), **test empirically** (linear vs saturation), **document honestly** (null results valuable).

**The pattern is universal. The implementation is domain-specific. Let evidence guide choices.**

---

**Status**: Design complete, ready for implementation
**Next Step**: Create SNARC module + threshold logic
**Expected Timeline**: Phase 1 in 1-2 weeks, Phase 2 after empirical validation

---

Co-Authored-By: Dennis Palatov (Human) <dp@dpcars.net>
Co-Authored-By: Claude (Thor-session) <noreply@anthropic.com>
