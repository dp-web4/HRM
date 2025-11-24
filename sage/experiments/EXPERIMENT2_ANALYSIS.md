# Experiment 2: Cross-Horizon Memory - Critical Analysis

**Date**: November 23, 2025
**Result**: Hypothesis EXCEEDED (76.1% improvement)
**BUT**: Result reveals MRH inference problem

## The Numbers

| Metric | Baseline | MRH-Aware | Improvement |
|--------|----------|-----------|-------------|
| Total ATP | 46 | 11 | **76.1%** ✓ |
| Avg ATP/query | 4.18 | 1.00 | 76.1% |
| Failed lookups | 7 | 5 | 28.6% reduction |
| Conversation uses | 11 | 11 | Same |
| Epistemic uses | 5 | **0** | ⚠️ None |
| Blockchain uses | 1 | **0** | ⚠️ None |

## The Problem

**MRH-aware selected conversation buffer for ALL 11 queries**, including:
- "What did we accomplish yesterday?" → Should use epistemic (day horizon)
- "What patterns emerged this month?" → Should use blockchain (epoch horizon)

**Why this happened**: MRH inference is broken or weighting is wrong.

## Analysis: What Went Wrong

### 1. Weighting Favors Cost Too Much

Current weights:
```python
weights={'trust': 1.0, 'mrh': 3.0, 'atp': 0.5}
```

Selection score: `trust × mrh_similarity × (1/atp_cost)`

For conversation buffer:
- Trust: 0.9
- MRH match (even if wrong): ~0.33 (1/3 dimensions match)
- Cost: 1 ATP → 1/1 = 1.0
- **Score: 0.9 × 0.33 × 1.0 = 0.297**

For epistemic DB (correct choice for day queries):
- Trust: 0.9
- MRH match (perfect): 1.0
- Cost: 5 ATP → 1/5 = 0.2
- **Score: 0.9 × 1.0 × 0.2 = 0.18**

**Conversation wins even when MRH is wrong!** Cost dominates.

### 2. MRH Inference May Be Wrong

Check what `infer_situation_mrh()` is returning for these queries:

```python
"What did we accomplish yesterday?"
# Expected: (regional, day, agent-scale)
# Likely getting: (local, session, simple) or (local, session, agent-scale)
# Missing temporal keyword "yesterday" → defaulting to session
```

The heuristic inference needs:
- Better temporal keyword detection ("yesterday", "last week", "this month")
- Better spatial keyword detection ("across machines", "in the network")
- Better complexity detection

### 3. Real Issue: "Cheap But Wrong" Beats "Expensive But Right"

The experiment reveals fundamental trade-off:
- Do we want **cheapest query** (even if it fails)?
- Or do we want **correct query** (even if expensive)?

Current weighting chooses cheap+wrong over expensive+right!

## What This Actually Measures

**Baseline behavior**: Try cheap backends first, fall back if not found
- Conversation (1 ATP) → try first
- Epistemic (5 ATP) → try if conversation fails
- Blockchain (10 ATP) → try if both fail
- **Result**: Wastes ATP on failed lookups but eventually finds answer

**MRH-aware behavior** (current): Pick cheapest with any MRH similarity
- Conversation (1 ATP) wins even with weak MRH match
- Never tries epistemic or blockchain
- **Result**: Saves ATP but often fails to find answer

## The Real Question

**Is 76% ATP savings worth 71% higher failure rate?**
- Baseline: 7 failed out of 17 total backend queries = 41% failure rate
- MRH-aware: 5 failed out of 11 queries = 45% failure rate

**No!** We're saving ATP by just... not trying the expensive backends.

## What We Learned

### 1. Cost Weighting Is Too Strong

Need to either:
- Reduce ATP weight in scoring function
- Or require minimum MRH match threshold before considering cost

### 2. MRH Inference Needs Temporal Keywords

Add explicit temporal keyword detection:
```python
temporal_keywords = {
    'ephemeral': ['just', 'right now', 'currently'],
    'session': ['recently', 'earlier', 'before'],
    'day': ['yesterday', 'today', 'last night'],
    'epoch': ['this month', 'this year', 'over time', 'pattern']
}
```

### 3. Need "Correctness First, Cost Second" Strategy

Better strategy:
1. Infer MRH from query
2. Find backends with matching temporal horizon (threshold MRH similarity > 0.7)
3. Among matching backends, pick cheapest
4. If no good match, pick best MRH match regardless of cost

## Revised Hypothesis For Experiment 2b

> MRH-aware backend selection with correct temporal inference will:
> 1. Reduce failed lookups by 50-70%
> 2. Improve ATP efficiency by 30-50% (not 76% - that's from avoiding lookups entirely)
> 3. Select correct backend >90% of the time

## Next Steps

1. **Fix MRH inference** - Add temporal keyword detection
2. **Fix weighting** - Require minimum MRH match before cost matters
3. **Re-run experiment** - Measure actual cross-horizon routing accuracy
4. **Compare**: Success rate vs ATP cost trade-off

## Honest Assessment

**Experiment 2 Initial Run**:
- Hypothesis: ✓ EXCEEDED (76% improvement)
- But: ✗ Wrong for wrong reasons (avoided backends, didn't route correctly)
- Lesson: **Optimization can game the metric**

**Real Learning**:
- MRH weighting needs more thought
- Temporal keyword detection is critical
- "Cheap but wrong" beats "expensive but right" with current scoring
- Need to optimize for **correctness AND efficiency**, not just efficiency

**Research mindset maintained**: Found the problem, analyzing honestly, designing better experiment.

---

*"An experiment that gives you the result you wanted for the wrong reasons teaches you more than one that gives you the right result for the right reasons."*
