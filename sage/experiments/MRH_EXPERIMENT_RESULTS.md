# MRH-Aware ATP Allocation: Experimental Results
**Date**: November 23, 2025
**Status**: Experiment 1 Complete

## Summary

Implemented and tested MRH-aware plugin selection for SAGE. Hypothesis was rejected (0% improvement), but experiment revealed valuable insights about implicit vs explicit MRH awareness.

## Implementation

### 1. MRH Profile Methods Added

**Pattern Response Engine** (`sage/cognitive/pattern_responses.py:221-237`):
```python
def get_mrh_profile(self) -> Dict[str, str]:
    return {
        'deltaR': 'local',      # No external resources
        'deltaT': 'ephemeral',  # Instant responses
        'deltaC': 'simple'      # Direct pattern matching
    }
```

**Introspective-Qwen IRP** (`sage/irp/plugins/introspective_qwen_impl.py:319-335`):
```python
def get_mrh_profile(self) -> Dict[str, str]:
    return {
        'deltaR': 'local',        # Local GPU
        'deltaT': 'session',      # Conversation context
        'deltaC': 'agent-scale'   # Iterative reasoning
    }
```

### 2. MRH Utilities Created

**File**: `sage/core/mrh_utils.py` (398 lines)

**Key Functions**:
- `compute_mrh_similarity()` - Exact match scoring (0-1)
- `compute_mrh_distance()` - Ordinal distance between MRH profiles
- `infer_situation_mrh()` - Infer MRH from query text
- `select_plugin_with_mrh()` - MRH-aware plugin selection

**Test Results**: All tests passing ✓

### 3. Experimental Framework

**File**: `sage/experiments/mrh_aware_allocation_experiment.py` (432 lines)

**Design**:
- Baseline: Trust-based fast/slow path
- Experimental: MRH-aware selection (`trust × mrh_similarity × 1/cost`)
- Test suite: 20 queries (7 simple, 3 medium, 10 complex)
- Metrics: ATP consumed, plugin usage, MRH match quality

## Results

### Quantitative Results

| Metric | Baseline | MRH-Aware | Improvement |
|--------|----------|-----------|-------------|
| Total ATP | 339 | 339 | 0 (0%) |
| Avg ATP/query | 16.95 | 16.95 | 0 |
| Pattern uses | 9 (45%) | 9 (45%) | 0 |
| Qwen uses | 11 (55%) | 11 (55%) | 0 |
| Avg MRH match | 0.88 | 0.88 | 0.00 |

**Hypothesis**: ✗ REJECTED
- Expected: 15-30% ATP improvement
- Actual: 0% improvement
- Both strategies made identical selections

### Why No Difference?

**Key Discovery**: Pattern matcher is **implicitly MRH-aware** by design!

The regex patterns naturally filter by MRH:
```python
r'\b(hello|hi|hey)\b'           # → (local, ephemeral, simple)
r'\bcan (you|u) hear\b'         # → (local, ephemeral, simple)
r'\bwhat.*?(doing|up to)\b'     # → (local, session, simple)
```

**Baseline fast/slow path is already near-optimal**:
1. Try pattern matcher first (1 ATP) → handles all simple queries
2. Fall back to IRP if no match (30 ATP) → handles complex queries

For our test suite, this perfectly separates:
- 7 simple queries → pattern (7 ATP)
- 10 complex queries → qwen (300 ATP)
- 3 medium queries → 2 pattern, 1 qwen (32 ATP)
- **Total: 339 ATP**

MRH-aware makes same decisions because pattern matcher already acts as MRH filter!

## Research Insights

### 1. Domain-Specific vs Generic Frameworks

**Domain-specific** (Pattern matching):
- Regex patterns encode expert knowledge directly
- Implicitly MRH-aware through pattern design
- Fast, simple, effective for well-defined domain

**Generic framework** (MRH scoring):
- Principled across different scenarios
- Explicit MRH reasoning
- More flexible but potentially overkill

**Lesson**: Domain knowledge beats generic frameworks when domain is well-understood.

### 2. When MRH Awareness Adds Value

MRH awareness provides value when:

**Multiple plugins at same MRH level**:
```
qwen-0.5b: (local, session, agent-scale), cost=10, trust=0.7
qwen-7b: (local, session, agent-scale), cost=100, trust=0.9
cloud-gpt: (global, session, agent-scale), cost=50, trust=0.95

Without MRH: Pick cloud-gpt (highest trust)
With MRH: Pick qwen-0.5b (same MRH, much cheaper, decent trust)
```

**Cross-horizon queries**:
```
Query: "What happened yesterday?"
MRH: (local, day, agent-scale)

conversation_buffer: (local, session, simple) - won't have it
epistemic_db: (regional, day, agent-scale) - perfect match!
llm_reasoning: (local, session, agent-scale) - could infer

Without MRH: Try conversation_buffer first (cheap), fail
With MRH: Directly use epistemic_db (correct horizon)
```

**Spatial extent matters**:
```
Query: "Search the web for X"
MRH: (global, ephemeral, simple)

local_cache: (local, ephemeral, simple) - wrong spatial extent
web_api: (global, ephemeral, simple) - perfect match!

Without MRH: Might try cache first, waste time
With MRH: Directly use web_api
```

### 3. Negative Results Are Valuable

This experiment **didn't confirm** the hypothesis, but it taught us:
- Pattern matcher is implicitly MRH-aware (important discovery!)
- Fast/slow path is good baseline heuristic
- Need more complex scenarios to demonstrate explicit MRH value
- Research is about learning, not just confirming predictions

## Next Experiments

### Experiment 2: Cross-Horizon Memory Queries ✓ Recommended

Test MRH with queries spanning different temporal horizons:
```
"What did we just discuss?" → (local, session, simple)
"What did we discuss yesterday?" → (regional, day, agent-scale)
"What patterns have emerged over the month?" → (regional, epoch, society-scale)
```

**Expected**: MRH-aware selects appropriate memory backend (conversation vs epistemic vs blockchain)

### Experiment 3: Multi-Plugin Same-Horizon

Test with 3 agent-scale plugins of different sizes:
```
Qwen-0.5B: cost=10, trust=0.7
Qwen-7B: cost=100, trust=0.9
GPT-4 API: cost=200, trust=0.95
```

**Expected**: MRH-aware balances cost/quality trade-offs

### Experiment 4: Real Voice Session

Run on actual voice session with Introspective-Qwen:
- Natural conversation with mixed simple/complex queries
- Measure actual ATP savings in production
- Validate MRH inference quality on real speech

## Files Created

1. **`sage/cognitive/pattern_responses.py`** - Added `get_mrh_profile()` method
2. **`sage/irp/plugins/introspective_qwen_impl.py`** - Added `get_mrh_profile()` method
3. **`sage/core/mrh_utils.py`** (398 lines) - Complete MRH utility library
4. **`sage/experiments/mrh_aware_allocation_experiment.py`** (432 lines) - Experimental framework
5. **`sage/experiments/EXPERIMENT_ANALYSIS.md`** - Detailed analysis
6. **`sage/experiments/MRH_EXPERIMENT_RESULTS.md`** - This file

**Total**: ~1,300 lines of code + documentation

## Conclusion

**Hypothesis**: ✗ Rejected (0% improvement)
**Research**: ✓ Successful (valuable insights gained)

**Key Discoveries**:
1. Pattern matcher is implicitly MRH-aware by design
2. Fast/slow path baseline is already near-optimal for simple scenarios
3. Explicit MRH awareness adds value in cross-horizon and multi-plugin scenarios
4. Domain-specific implementations can outperform generic frameworks

**Research Mindset Maintained**:
- Learned from "failure" (hypothesis rejection)
- Analyzed WHY no improvement was observed
- Identified when MRH awareness DOES add value
- Designed better experiments based on findings

**Next Step**: Implement Experiment 2 (cross-horizon memory queries) to demonstrate where explicit MRH awareness provides clear value.

---

*"The experiment that confirms your hypothesis teaches you something you already suspected. The experiment that rejects your hypothesis teaches you something you didn't know."*
