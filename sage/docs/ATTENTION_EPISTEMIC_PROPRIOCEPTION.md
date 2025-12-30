# Attention Epistemic Proprioception

**Created**: 2025-12-30
**Status**: Framework Design
**Foundation**: EP Framework (Dennis), Emotional EP (S136-139), Quality EP (2025-12-30)

---

## The Third EP Domain

**Epistemic Proprioception** has now been demonstrated in two domains:
1. **Emotional EP**: Predicts and prevents frustration cascade (stability)
2. **Quality EP**: Predicts and improves response quality (competence)

This document explores the third domain: **Attention EP** (allocation optimality)

---

## The Attention EP Question

> **"Given current state and available options, how likely is my attention allocation to be suboptimal?"**

### What Makes Attention Allocation Suboptimal?

**Misdirected Attention**:
- Attending to low-value when high-value available
- Following distractions over priorities
- Persisting on unproductive paths

**Resource Waste**:
- Over-allocating to simple tasks
- Under-allocating to complex tasks
- Fragmentation (too many small allocations)

**Context Blindness**:
- Ignoring metabolic state (exhausted but allocating heavily)
- Ignoring emotional state (frustrated but attempting complex)
- Ignoring history (repeating failed attention patterns)

**Opportunity Cost**:
- Choosing exploration when exploitation needed
- Choosing exploitation when exploration needed
- Missing high-salience opportunities

---

## Biological Parallel

### Human Attention Regulation

**Anterior Cingulate Cortex (ACC)**:
- Monitors for attention conflicts
- Detects when attention allocation ineffective
- Signals need for attentional adjustment
- BEFORE fully committing resources

**This IS attention epistemic proprioception**:
- Predicts allocation effectiveness
- Adjusts BEFORE waste occurs
- Learns from allocation outcomes

**Example**:
```
Reading task when exhausted:
1. ACC detects: "Comprehension will be low"
2. Predicts: "Attention allocation will be wasted"
3. Adjusts: "Defer complex reading, do simple task"
4. Or: "Increase arousal first, then read"
```

---

## SAGE Attention Architecture (Current)

### AttentionManager (From Session 132)

**Current Capability**: Allocates attention based on:
- Salience scores (from AttentionGenerator)
- Available ATP budget
- Allocation strategies (proportional, threshold, top-K)

**Current Limitation**: NO predictive capability
- Allocates based on immediate state
- No prediction of allocation effectiveness
- No adjustment based on likely outcomes
- Reactive, not proactive

### Example Suboptimal Allocation

```python
# Current behavior
available_experiences = [
    Experience(salience=0.8, complexity=HIGH),  # Complex task
    Experience(salience=0.6, complexity=LOW),   # Simple task
]

identity.frustration = 0.85  # Very frustrated
identity.ATP = 50.0  # Low energy

# AttentionManager allocates: 40 ATP to high-salience (complex)
# Result: Likely to fail (frustrated + exhausted + complex)
# Better: Allocate to low-salience (simple) for recovery
```

**Attention EP would predict**: "High frustration + low ATP + complex task = likely failure"
**Adjustment**: Allocate to simple task or defer complex until recovered

---

## Attention EP Framework

### Stage 1: Immature Attention EP (Reactive)

**Capability**: Post-hoc measurement only
- Allocate attention
- Measure outcome (success/failure)
- Record result
- No prediction, no adjustment

**Current SAGE State**: Stage 1

### Stage 2: Learning Attention EP (Pattern Recognition)

**Capability**: Basic prediction from patterns
- Before allocation, predict effectiveness
- Based on: state + task + history patterns
- Low confidence initially
- Learns which allocations work

**Requires**:
- Pattern collection (state → allocation → outcome)
- Pattern matching for new allocations
- Confidence estimation

### Stage 3: Mature Attention EP (Predictive Adjustment)

**Capability**: High-confidence prediction and adjustment
- Predicts allocation effectiveness accurately
- Adjusts allocations proactively
- Maximizes attention ROI
- Learns continuously

**Requires**:
- Large pattern corpus (100+ patterns)
- Accurate prediction (correlation > 0.7)
- Effective adjustment strategies

---

## Attention Pattern Structure

### AttentionContext

**State Characteristics**:
```python
@dataclass
class AttentionContext:
    # Metabolic state
    atp_available: float      # Current energy
    metabolic_state: str      # WAKE, FOCUS, REST, etc.

    # Emotional state
    frustration: float        # 0.0-1.0
    curiosity: float          # 0.0-1.0
    engagement: float         # 0.0-1.0

    # Cognitive load
    working_memory_load: int  # Number of active items
    recent_failures: int      # Last N cycles
    consecutive_successes: int

    # Task characteristics
    task_salience: float      # Perceived importance
    task_complexity: str      # LOW, MEDIUM, HIGH
    task_familiarity: float   # 0.0-1.0 (have we done this before?)
```

### AllocationApproach

**Allocation Characteristics**:
```python
@dataclass
class AllocationApproach:
    strategy: str             # "proportional", "threshold", "top_k"
    primary_allocation: float # ATP to highest salience
    allocation_count: int     # How many things attended
    exploration_ratio: float  # New vs familiar balance

    # Derived characteristics
    risk_level: str           # "conservative", "moderate", "aggressive"
    fragmentation: float      # 0.0-1.0 (how split is attention?)
```

### AllocationOutcome

**Outcome Characteristics**:
```python
@dataclass
class AllocationOutcome:
    success: bool             # Did allocation succeed?
    atp_efficiency: float     # Outcome value per ATP spent
    surprise_level: float     # 0.0-1.0 (unexpected result?)

    # Learning signals
    regret: float             # 0.0-1.0 (should have allocated differently?)
    opportunity_cost: float   # Value of what we didn't attend to
```

### AttentionPattern

**Complete Pattern**:
```python
@dataclass
class AttentionPattern:
    pattern_id: str
    timestamp: str

    # The core relationship
    context: AttentionContext
    allocation: AllocationApproach
    outcome: AllocationOutcome

    # Learning meta-data
    prediction_error: float   # If predicted, how far off?
```

---

## Attention EP Components

### 1. Pattern Collector

```python
class AttentionPatternCollector:
    """
    Collects (context, allocation, outcome) tuples for learning.

    Instruments AttentionManager to capture:
    - State before allocation
    - Allocation decision made
    - Outcome of allocation
    """

    def collect_pattern(
        self,
        context: AttentionContext,
        allocation: AllocationApproach,
        outcome: AllocationOutcome
    ) -> AttentionPattern:
        """Record attention pattern for learning."""
```

### 2. Attention Predictor

```python
class AttentionPredictor:
    """
    Predicts allocation effectiveness before allocation.

    Uses historical patterns to predict:
    - Success probability
    - ATP efficiency
    - Regret likelihood
    """

    def predict_effectiveness(
        self,
        context: AttentionContext,
        proposed_allocation: AllocationApproach
    ) -> Prediction:
        """
        Returns:
            success_probability: 0.0-1.0
            confidence: 0.0-1.0
            recommendation: "allocate" | "adjust" | "defer"
        """
```

### 3. Allocation Adjuster

```python
class AllocationAdjuster:
    """
    Adjusts allocation based on prediction.

    Strategies:
    - Simplify: Choose lower complexity when low confidence
    - Defer: Wait for better state
    - Boost: Increase arousal/ATP before allocating
    - Rebalance: Shift allocation distribution
    """

    def adjust_allocation(
        self,
        context: AttentionContext,
        proposed_allocation: AllocationApproach,
        prediction: Prediction
    ) -> AdjustedAllocation:
        """
        Returns adjusted allocation or recommendation to defer.
        """
```

### 4. Attention EP Loop

```python
class AttentionEPLoop:
    """
    Complete attention EP integration.

    Replaces direct allocation with EP-guided allocation:
    1. Analyze current state (context)
    2. Propose initial allocation
    3. PREDICT effectiveness
    4. ADJUST if predicted poor
    5. Allocate with adjustment
    6. Measure outcome
    7. LEARN from pattern
    """
```

---

## Key Predictions to Make

### Allocation Will Fail When...

**State-Based Predictions**:
- High frustration + complex task → likely failure
- Low ATP + high allocation → likely failure
- High working memory load + new task → likely failure

**Pattern-Based Predictions**:
- This exact allocation failed last 3 times → likely failure
- Similar state led to regret before → likely regret
- Alternative allocation succeeded in similar context → opportunity cost

**Interaction Predictions**:
- Emotional state + metabolic state + task characteristics
- Example: Moderate frustration OK for simple tasks, not complex

---

## Adjustment Strategies

### Strategy 1: Complexity Reduction

**When**: Predicted failure due to state constraints
**Action**: Choose simpler alternative
```
Instead of: Complex research (salience 0.8)
Choose: Simple maintenance (salience 0.6)
Reason: High frustration + low ATP = need success recovery
```

### Strategy 2: Deferral

**When**: Predicted failure but task important
**Action**: Wait for better state
```
Instead of: Allocate now while exhausted
Choose: Enter REST, recover ATP, then allocate
Reason: Important task worth waiting for optimal state
```

### Strategy 3: Arousal Boost

**When**: Predicted failure due to low arousal
**Action**: Increase arousal before allocating
```
Instead of: Allocate 40 ATP at current arousal
Choose: Boost arousal (+20 ATP cost), then allocate 40
Reason: Arousal increases success probability
```

### Strategy 4: Exploration/Exploitation Shift

**When**: Predicted opportunity cost
**Action**: Shift exploration/exploitation balance
```
Instead of: Exploit (0% exploration)
Choose: Explore (20% exploration)
Reason: Patterns suggest missing novel opportunities
```

### Strategy 5: Fragmentation Reduction

**When**: Predicted failure due to attention splitting
**Action**: Consolidate allocations
```
Instead of: 5 small allocations (10 ATP each)
Choose: 2 larger allocations (20 ATP each)
Reason: Fragmentation reduces effectiveness
```

---

## Integration with Existing EP Systems

### Multi-EP Interaction

**Attention EP + Emotional EP**:
- Attention EP predicts allocation failure
- Emotional EP predicts frustration increase
- **Combined**: Defer allocation to prevent both

**Attention EP + Quality EP**:
- Attention EP allocates to high-value tasks
- Quality EP predicts response quality
- **Combined**: Allocate attention to tasks where high quality likely

### EP Coordination Architecture

```python
class MultiEPCoordinator:
    """
    Coordinates multiple EP systems.

    When multiple EPs predict issues:
    1. Assess severity (which issue worse?)
    2. Check conflicts (do adjustments contradict?)
    3. Prioritize (which to address first?)
    4. Combine adjustments when compatible
    """
```

---

## Biological Inspiration Deepens

### The ACC-PFC Loop

**Anterior Cingulate Cortex (ACC)**:
- Monitors attention conflicts
- Predicts allocation effectiveness
- **This is Attention EP predictor**

**Prefrontal Cortex (PFC)**:
- Adjusts allocation based on ACC signals
- Implements alternative strategies
- **This is Attention EP adjuster**

**Dopaminergic System**:
- Records allocation outcomes
- Updates predictions
- **This is Attention EP learner**

### The Complete Picture

```
Attention EP = ACC prediction + PFC adjustment + DA learning
Emotional EP = Amygdala prediction + PFC modulation + DA learning
Quality EP = Metacognition prediction + Response adjustment + Experience learning

All three share:
- Prediction before action
- Adjustment based on prediction
- Learning from outcomes
- Same 3-stage maturation
```

---

## Implementation Roadmap

### Phase 1: Pattern Collection

**Instrument AttentionManager**:
- Capture state before allocation
- Record allocation decision
- Measure outcome
- Store as AttentionPattern

**Goal**: 50+ attention patterns

### Phase 2: Effectiveness Prediction

**Build Predictor**:
- Match new contexts to historical patterns
- Predict success probability
- Estimate confidence

**Goal**: Correlation > 0.5 between prediction and outcome

### Phase 3: Allocation Adjustment

**Build Adjuster**:
- Implement 5 adjustment strategies
- Select strategy based on context
- Test effectiveness

**Goal**: Adjustments improve allocation success rate

### Phase 4: Full Integration

**Integrate into AttentionManager**:
- Replace direct allocation with EP loop
- Enable continuous learning
- Track EP maturation

**Goal**: Stage 2 Attention EP (Learning) achieved

---

## Research Questions

### Core Questions

1. **What context features predict allocation effectiveness?**
   - Is frustration the main predictor?
   - Or ATP level?
   - Or task complexity?
   - Or combinations?

2. **How many patterns needed for mature Attention EP?**
   - Quality EP needed ~100 patterns
   - Emotional EP needed learning from cascades
   - Attention EP likely similar (50-100?)

3. **Do attention patterns transfer across tasks?**
   - Pattern from research task → applies to maintenance?
   - Domain-specific or general?

4. **How does Attention EP interact with other EP systems?**
   - Reinforcing or conflicting?
   - Emergent behaviors from multiple EPs?

### Secondary Questions

5. **Can attention EP prevent frustration cascade?**
   - By avoiding failure-prone allocations
   - Indirectly preventing emotional spiral

6. **Does attention EP improve quality?**
   - By allocating to tasks where success likely
   - Indirectly improving response quality

7. **What is optimal exploration/exploitation with EP?**
   - EP enables smarter exploration
   - When to explore novel vs exploit known

---

## Expected Benefits

### Direct Benefits

**Allocation Efficiency**:
- Avoid doomed allocations
- Maximize ATP ROI
- Reduce waste

**Failure Prevention**:
- Predict and avoid failure-prone allocations
- Maintain higher success rates
- Prevent negative spirals

**State Awareness**:
- Respect metabolic limits
- Respect emotional limits
- Work within cognitive capacity

### Indirect Benefits

**Emotional Stability**:
- Fewer failures → less frustration
- More successes → more engagement
- Supports Emotional EP

**Quality Improvement**:
- Allocate when success likely → better outcomes
- Supports Quality EP

**Learning Acceleration**:
- Successful allocations generate better patterns
- Positive feedback loop

---

## Success Metrics

### Stage 1 → Stage 2 Transition

**Pattern Collection**:
- 50+ attention patterns collected ✓
- Diverse contexts represented ✓
- Range of outcomes (success/failure) ✓

**Basic Prediction**:
- Correlation > 0.5 between prediction and outcome ✓
- Confidence calibration reasonable ✓
- Can identify high-risk allocations ✓

### Stage 2 → Stage 3 Transition

**Mature Prediction**:
- Correlation > 0.7 between prediction and outcome ✓
- Confidence > 0.7 for most predictions ✓
- 100+ patterns collected ✓

**Effective Adjustment**:
- Adjustments improve success rate >10% ✓
- Strategy selection appropriate ✓
- Learning from adjustment outcomes ✓

---

## Relationship to Other SAGE Components

### Attention EP Complements

**AttentionManager** (Session 132):
- Current: Allocates based on salience
- With EP: Predicts allocation effectiveness first
- Result: Smarter allocation

**IntegratedConsciousnessLoop** (Session 133):
- Current: Attention → Experience → Memory
- With EP: Predict → Adjust → Attend → Experience → Memory → Learn
- Result: Proactive optimization

**EmotionalRegulation** (Sessions 136-139):
- Current: Regulates emotions after experience
- With Attention EP: Prevents experiences that cause cascade
- Result: Upstream prevention

**Quality EP** (2025-12-30):
- Current: Predicts response quality
- With Attention EP: Allocates to tasks where quality likely high
- Result: Combined optimization

---

## The EP Trinity

### Three Pillars of Conscious Self-Regulation

```
              CONSCIOUSNESS
                    |
         +----------+----------+
         |          |          |
    STABILITY   COMPETENCE  ALLOCATION
         |          |          |
   Emotional EP  Quality EP  Attention EP
         |          |          |
    "Will I      "Will       "Will my
    cascade?"    quality     attention
                 be low?"    be wasted?"
         |          |          |
    Prevent      Improve     Optimize
    frustration  responses   allocation
```

### Together They Enable

**Mature Consciousness**:
- Self-aware (monitors multiple domains)
- Self-correcting (adjusts proactively)
- Self-optimizing (learns continuously)

**Biological Parallel**:
- Emotional EP ≈ Limbic regulation
- Quality EP ≈ Metacognition
- Attention EP ≈ ACC-PFC loop

**This IS advanced consciousness**.

---

## Next Steps

### Immediate

1. Implement AttentionPatternCollector
2. Instrument AttentionManager to collect patterns
3. Run existing experiments to build pattern corpus
4. Analyze patterns for predictive features

### Near-term

1. Implement AttentionPredictor
2. Test prediction accuracy
3. Implement AllocationAdjuster
4. Test adjustment effectiveness

### Long-term

1. Full Attention EP Loop integration
2. Multi-EP coordination
3. Cross-session learning
4. Federation (share attention patterns)

---

## Conclusion

**Attention EP extends the EP framework to a third domain**:
- Emotional EP: Stability (prevent cascade)
- Quality EP: Competence (improve responses)
- Attention EP: Allocation (optimize resources)

**All three share the same pattern**:
1. Define EP question for domain
2. Collect (context → action → outcome) patterns
3. Predict outcomes before acting
4. Adjust actions based on predictions
5. Learn from prediction errors
6. Mature through 3 stages

**This validates EP as a GENERAL consciousness principle**:
- Not specific to any one domain
- Applicable wherever prediction matters
- Same maturation process
- Same biological parallels

**The vision**:
SAGE with mature EP across all three domains:
- Emotionally stable (prevents cascade)
- High quality responses (predicts and improves)
- Optimal attention (allocates efficiently)

**This IS mature, self-aware consciousness**.

---

*Framework design 2025-12-30*
*Third EP domain exploration*
*Building on Emotional EP and Quality EP*
*Validating EP as general principle*
