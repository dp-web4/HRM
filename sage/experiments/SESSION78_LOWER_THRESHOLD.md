# Session 78: Lower Trust Evidence Threshold - Unexpected Finding

**Date**: 2025-12-19
**Status**: ⚠️ Surprising Result - Trust_driven still 0% despite sufficient evidence
**Platform**: Thor (Jetson AGX Thor)

---

## Goal

Test min_trust_evidence=2 with ε=0.2 to enable trust_driven transitions.

---

## Hypothesis (from Session 77)

**Session 77 Analysis**:
- 0% trust_driven despite 45 experts, 39 specialists
- min_trust_evidence=3 appeared too strict
- Expected: Many experts have 2 samples per context (below threshold=3)

**Session 78 Hypothesis**:
- Lowering threshold to min_trust_evidence=2 will enable trust_driven
- Evidence log should show ≥2 experts with ≥2 samples per context
- Trust_driven should activate around generation 20-30

---

## Implementation

### Configuration

**Changed from Session 77**:
```python
trust_selector = TrustFirstMRHSelector(
    num_experts=128,
    min_trust_evidence=2,  # Session 78: LOWERED from 3
    low_trust_threshold=0.3,
    overlap_threshold=0.7,
    component="thinker",
    network="thor-testnet",
    context_classifier=classifier,
    epsilon=0.2  # Same as S77 optimal
)
```

**Kept constant**:
- epsilon=0.2 (optimal from S77)
- 10 epochs, 90 generations
- 9 diverse sequences (3 code, 3 reasoning, 3 text)
- Real Q3-Omni 30B model

---

## Results

### Summary Comparison

| Metric | S77 (threshold=3) | S78 (threshold=2) | Change |
|--------|-------------------|-------------------|--------|
| **Experts** | 45 | **65** | +20 (+44%) |
| **Utilization** | 35.2% | **50.8%** | +15.6pp |
| **Specialists** | 39 | **50** | +11 (+28%) |
| **Generalists** | 6 | **15** | +9 |
| **Specialization** | 86.7% | **76.9%** | -9.8pp |
| **Trust_Driven** | 0% | **0%** | **NO CHANGE** |
| **Forced Exp** | 20.0% | 23.3% | +3.3pp |

### Detailed Results

```
📊 Expert Diversity:
  Unique experts: 65/128 (50.8%)  [+44% vs S77]
  Total selections: 360

🎯 Specialization:
  Specialists (single-context): 50  [+28% vs S77]
  Generalists (multi-context): 15  [+150% vs S77]
  Specialization rate: 76.9%  [-9.8pp vs S77]

🔄 Mode Transitions:
  router_explore: 69/90 (76.7%)
  trust_driven: 0/90 (0.0%)  ← STILL ZERO
  forced_exploration: 21/90 (23.3%)
  First trust_driven: NEVER
```

### Trust Evidence Accumulation Log

```
Gen 10: 14 experts with ≥2 samples across contexts
  context_0: 4 experts
  context_1: 6 experts
  context_2: 4 experts

Gen 20: 14 experts with ≥2 samples
  context_0: 4 experts
  context_1: 6 experts
  context_2: 4 experts

Gen 30: 14 experts with ≥2 samples
  context_0: 4 experts
  context_1: 6 experts
  context_2: 4 experts

...

Gen 90: 18 experts with ≥2 samples
  context_0: 7 experts  ← MORE THAN THRESHOLD (≥2)
  context_1: 6 experts  ← MORE THAN THRESHOLD
  context_2: 5 experts  ← MORE THAN THRESHOLD
```

---

## Key Finding: Trust_Driven STILL 0% Despite Sufficient Evidence

### The Paradox

**Evidence Log Shows**:
- By generation 90: 7 experts in context_0, 6 in context_1, 5 in context_2
- ALL contexts have ≥2 experts with ≥2 samples
- Threshold requirement (min_trust_evidence=2) IS MET

**But**:
- trust_driven: 0/90 (0.0%)
- NO trust_driven activations at all

**This is UNEXPECTED**:
- _has_sufficient_trust_evidence() should return True when:
  ```python
  # Need at least 2 experts with sufficient trust
  return len(experts_with_evidence) >= 2
  ```
- Evidence log shows 4-7 experts per context meet threshold
- Therefore, trust_driven SHOULD have activated

### Hypothesis for Why Trust_Driven Didn't Activate

**Most Likely**:  trust > low_trust_threshold (0.3) check is failing

**Code Flow**:
```python
def _has_sufficient_trust_evidence(self, context: str) -> bool:
    experts_with_evidence = []
    for expert_id in range(self.num_experts):
        key = (expert_id, context)
        if key in self.bridge.trust_history:
            history = self.bridge.trust_history[key]
            if len(history) >= self.min_trust_evidence:  # ✅ 4-7 experts meet this
                trust = history[-1]
                if trust > self.low_trust_threshold:  # ⚠️  Likely failing here
                    experts_with_evidence.append((expert_id, trust))

    return len(experts_with_evidence) >= 2
```

**Why trust > 0.3 might be failing**:
1. **Trust update** may not be working correctly
2. **Simulated quality** (0.75 ± 0.1) should give trust ≈ 0.7-0.8
3. **But** if trust history is being reset or not persisting, trust could be stuck at 0.5 (initialization)
4. **Or** forced exploration selections (uniform random quality) might be diluting trust

**Alternative Hypotheses**:
1. **Context string mismatch**: Evidence log tracks "context_0" but selector checks different format
2. **Trust history bug**: History exists but trust values incorrect
3. **Threshold check bug**: low_trust_threshold=0.3 being compared incorrectly

---

## Secondary Finding: More Diversity, Less Specialization

### Diversity Increase

**Session 77 → Session 78**:
- Experts: 45 → 65 (+44%)
- Utilization: 35.2% → 50.8% (+15.6pp)

**Why**:
- Lower threshold means more "exploration credit"
- Experts need only 2 samples (not 3) to potentially trigger trust mode
- More experts get "recognized" by the system
- Forced exploration continues breaking monopoly

### Specialization Decrease

**Session 77 → Session 78**:
- Specialization rate: 86.7% → 76.9% (-9.8pp)
- Generalists: 6 → 15 (+150%)

**Why**:
- Higher diversity → more experts used
- More experts → higher chance of appearing in multiple contexts
- Lower specialization rate is expected with higher diversity

**Trade-off**:
- ε=0.2, threshold=3: 39 specialists (86.7% specialization) ← FOCUSED
- ε=0.2, threshold=2: 50 specialists (76.9% specialization) ← BROAD

**Interpretation**: Lower threshold favors **breadth** (more experts) over **depth** (focused specialization).

---

## Comparison to Session 77

### What Improved

1. **Expert Diversity**: 45 → 65 experts (+44%)
2. **Specialist Count**: 39 → 50 specialists (+28%)
3. **Utilization**: 35.2% → 50.8% (+15.6pp)

### What Stayed the Same

1. **Trust_Driven Mode**: 0% → 0% (NO CHANGE) ← **SURPRISING**
2. **Router Monopoly**: Still broken (forced exploration working)
3. **Evidence Accumulation**: Working (4-7 experts per context)

### What Worsened

1. **Specialization Rate**: 86.7% → 76.9% (-9.8pp)
2. **Generalists**: 6 → 15 (+150%) - more cross-context usage

---

## Root Cause Analysis

### Why Trust_Driven Didn't Activate

**Step-by-step debugging needed**:

1. **Check trust history contents**:
   - Are trust values being stored correctly?
   - Are they persisting across generations?
   - What are actual trust values at generation 90?

2. **Check trust > threshold test**:
   - Are trust values > 0.3?
   - If not, why? (simulated quality is 0.75 ± 0.1)

3. **Check context string matching**:
   - Evidence log uses "context_0", "context_1", "context_2"
   - Does _has_sufficient_trust_evidence() use same format?
   - Context classifier returns "context_0" format?

4. **Check update_trust_for_expert**:
   - Is weighted_quality being calculated correctly?
   - Is bridge.update_trust_history() working?

### Likely Culprit

**Trust values stuck at or below 0.3** despite simulated quality of 0.75.

**Possible causes**:
1. Trust initialization at 0.5
2. Trust update formula not increasing trust properly
3. Forced exploration (uniform random selection) not updating trust
4. Trust decay or reset happening

---

## Next Steps

### Immediate Investigation

1. **Add debug logging** to _has_sufficient_trust_evidence():
   ```python
   print(f"Context {context}: {len(experts_with_evidence)} experts with trust > {self.low_trust_threshold}")
   for expert_id, trust in experts_with_evidence[:5]:
       print(f"  Expert {expert_id}: trust={trust:.3f}")
   ```

2. **Inspect trust_history** at generation 90:
   ```python
   for expert_id in range(128):
       for ctx in ["context_0", "context_1", "context_2"]:
           key = (expert_id, ctx)
           if key in trust_selector.bridge.trust_history:
               history = trust_selector.bridge.trust_history[key]
               print(f"Expert {expert_id}, {ctx}: {len(history)} samples, last trust={history[-1]:.3f}")
   ```

3. **Validate trust update formula** in ContextAwareIdentityBridge

### Alternative Approaches

If trust values are indeed stuck:

**Option 1: Lower low_trust_threshold** (instead of min_trust_evidence)
```python
low_trust_threshold=0.2  # or 0.1, instead of 0.3
```

**Option 2: Use min_trust_evidence=1**
```python
min_trust_evidence=1  # Minimum possible
```

**Option 3: Fix trust update mechanism**
- Review ContextAwareIdentityBridge.update_trust_history()
- Ensure trust values are calculated and stored correctly
- Validate trust increases with positive quality

---

## Conclusion

**Session 78 Results**:
- ✅ Diversity increased dramatically (45 → 65 experts)
- ✅ Specialist count increased (39 → 50)
- ❌ Trust_driven STILL 0% despite meeting threshold requirements
- ⚠️  Specialization rate decreased (86.7% → 76.9%)

**Key Mystery**:
Evidence log shows 4-7 experts per context with ≥2 samples, but trust_driven never activated.

**Most Likely Cause**:
Trust values ≤ 0.3 (failing trust > low_trust_threshold check), despite simulated quality of 0.75.

**Next Session (79)**:
1. Add debug logging to inspect actual trust values
2. Investigate why trust isn't increasing
3. Either fix trust update OR lower low_trust_threshold to enable trust_driven

**Sessions 74-78 Arc**:
```
S74: Integration → S75: API fix (15 lines) →
S76: Reality check (monopoly discovered) →
S77: Monopoly broken (50 lines, 11.25x diversity) →
S78: Lower threshold (MORE diversity, but trust_driven mystery)
```

**Status**: Trust-first architecture working (diversity, specialists), but trust_driven mode requires investigation.

---

## Update (2025-12-20)

**Mystery SOLVED - Sessions 79-80**:
- Session 79 investigation found root cause: weighted quality bug
- Session 78 used `weighted_quality = quality * weight ≈ 0.75 * 0.25 = 0.19`
- Trust threshold check: `0.19 > 0.3` → FALSE (always fails)
- Session 80 fix: Use unweighted quality (0.75 > 0.3 → passes)

**Session 80 Fix Applied**:
- Updated `session78_lower_threshold.py` to use unweighted quality
- Script now uses `quality` directly instead of `quality * weight`
- Expected result if re-run: ~70% trust_driven activation (validated in Session 80)

**Note**: Session 78 diversity/specialist results remain VALID. The weighted quality bug only affected trust_driven activation rate.

See SESSION79_TRUST_FIX.md and SESSION80_TRUST_FIX_VALIDATION.md for complete analysis.

---

*"When the data surprises you, you're learning. Session 78: Expected trust_driven. Got 0%. Evidence shows threshold met. Trust values must be the issue. Investigate next."*
