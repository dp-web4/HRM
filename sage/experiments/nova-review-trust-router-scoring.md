# Trust-Router Scoring Shape for Nova Review

**Date**: 2025-12-22 (Updated with Session 90)
**Purpose**: Present current trust-based expert routing implementation for stability/thrashing analysis
**Request**: Identify remaining failure modes and stability issues

---

## Architecture Evolution

### Phase 1: Trust-First Conditional (Session 72)

Replaced weighted blending with **conditional trust-first logic**:

```
OLD (Sessions 64-67):  selection = α × router + (1-α) × trust
NEW (Session 72+):     if has_trust → pure_trust else free_router
```

Result: 3.4x more expert diversity (17 → 58 experts utilized).

### Phase 2: Resource-Aware Trust (Session 90)

Integrated your feedback - **trust = permission to consume scarce shared resources**:

```
NEW (Session 90):  permission = expertise × cheapness × persistence
```

Result: **+1033 generation speedup** (Gen 1166 → Gen 133 for first trust activation).

---

## Selection Flow (Pseudocode)

### Base Flow (Sessions 72-89)

```python
def select_experts(router_logits, context, k=8):

    # 1. FORCED EXPLORATION (ε-greedy, Session 77)
    if random() < epsilon:  # epsilon = 0.2
        return random_sample(all_experts, k)

    # 2. CHECK TRUST EVIDENCE
    has_evidence = has_sufficient_trust_evidence(context)

    # 3. CONDITIONAL SELECTION
    if has_evidence:
        return trust_driven_selection(context, k)
    else:
        return router_explore_selection(router_logits, k)
```

### Resource-Aware Flow (Session 90)

```python
def select_expert(layer, context):
    """Select using resource-aware permission score."""

    for each available expert:
        # 1. EXPERTISE (reputation + internal quality)
        expertise = 0.4 * reputation + 0.6 * internal_quality

        # 2. RESOURCE COST (cheapness)
        if expert_is_hot:
            resource_cost = 0.0  # Already loaded, free
        elif swaps_budget_exhausted:
            resource_cost = 10.0  # Extreme penalty, block swap
        else:
            resource_cost = swap_cost * 0.3 + bandwidth_cost * 0.2

        cheapness = 1.0 / (1.0 + resource_cost)

        # 3. PERSISTENCE (hysteresis)
        if expert_is_hot:
            persistence = 1.0 + hysteresis_boost  # +20%
        else:
            persistence = 1.0

        # COMPOSITE PERMISSION
        permission = expertise * cheapness * persistence

    return expert_with_highest_permission
```

---

## Trust Evidence Check

```python
def has_sufficient_trust_evidence(context) -> bool:
    """Need ≥2 experts with sufficient trust history in this context."""

    experts_with_evidence = []
    for expert_id in range(num_experts):
        key = (expert_id, context)
        if key in trust_history:
            history = trust_history[key]
            if len(history) >= min_trust_evidence:  # 3 samples
                trust = history[-1]  # Most recent trust value
                if trust > low_trust_threshold:     # 0.3
                    experts_with_evidence.append((expert_id, trust))

    return len(experts_with_evidence) >= 2
```

**Parameters**:
- `min_trust_evidence = 3` - Need 3+ observations to trust
- `low_trust_threshold = 0.3` - Minimum trust to count as "evidence"

---

## Trust-Driven Selection

```python
def trust_driven_selection(context, k):
    """100% trust, 0% router influence."""

    # Get trust scores for all experts in this context
    trust_scores = [get_context_trust(eid, context) for eid in range(num_experts)]

    # Select top-k by trust alone
    top_k = argsort(trust_scores)[-k:][::-1]

    selected = []
    for expert_id in top_k:
        if trust_scores[expert_id] < low_trust_threshold:
            # MRH substitution: find better alternative
            alt = find_mrh_alternative(expert_id, context)
            if alt:
                selected.append(alt)
            else:
                selected.append(expert_id)
        else:
            selected.append(expert_id)

    # Normalize trust scores for mixing weights
    weights = softmax(trust_scores[top_k])

    return selected, weights
```

---

## Router Exploration (Cold Start)

```python
def router_explore_selection(router_logits, k):
    """100% router, 0% trust - pure exploration."""

    top_k = argsort(router_logits)[-k:][::-1]
    weights = softmax(router_logits[top_k])

    return top_k, weights
```

---

## Trust Retrieval

```python
def get_context_trust(expert_id, context) -> float:
    key = (expert_id, context)

    if key in trust_history:
        history = trust_history[key]
        if history:
            return history[-1]  # Most recent value only

    # Fallback to reputation DB
    rep = reputation_db.get_reputation(expert_id)
    if rep:
        return rep.get_context_trust(context, default=0.5)

    return 0.5  # Default neutral trust
```

---

## Trust Update

```python
def update_trust_for_expert(expert_id, context, quality):
    """Called after each generation with measured quality."""
    key = (expert_id, context)
    if key not in trust_history:
        trust_history[key] = []
    trust_history[key].append(quality)  # Just append, no smoothing
```

**Quality measurement**: Based on perplexity, coherence, task_quality metrics.

---

## MRH Alternative Finding

```python
def find_mrh_alternative(expert_id, context, all_experts):
    """Find expert with high context overlap and better trust."""

    alternatives = []
    for other in all_experts:
        if other == expert_id:
            continue

        overlap = compute_context_overlap(expert_id, other)

        if overlap >= overlap_threshold:  # 0.7
            alt_trust = get_context_trust(other, context)
            current_trust = get_context_trust(expert_id, context)

            if alt_trust > current_trust:
                alternatives.append((other, alt_trust, overlap))

    return max(alternatives, key=lambda x: x[1]) if alternatives else None
```

**Context overlap**: Cosine similarity of context distribution vectors.

---

## Current Parameters

### Base Parameters (Sessions 72-89)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| epsilon | 0.2 | Forced random exploration probability |
| min_trust_evidence | 3 | Samples needed before trust-driven mode |
| low_trust_threshold | 0.3 | Minimum trust to count |
| overlap_threshold | 0.7 | MRH pairing threshold |
| k (experts per selection) | 8 | How many experts selected |
| num_experts | 128 | Total expert pool (Q3-Omni) |

### Resource-Aware Parameters (Session 90)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| max_hot_experts | 64 | LRU cache size |
| hysteresis_boost | 0.2 | +20% trust boost for loaded experts |
| switching_cost_weight | 0.3 | Weight of swap cost in permission |
| memory_cost_weight | 0.2 | Weight of bandwidth cost in permission |
| max_swaps_per_gen | 8 | Budget limit prevents thrashing |

---

## Trust Storage Structure

```python
# Trust history: Dict[(expert_id, context), List[float]]
trust_history = {
    (42, "context_0"): [0.65, 0.72, 0.68, 0.71],
    (42, "context_1"): [0.55, 0.48],
    (99, "context_0"): [0.82, 0.85, 0.88, 0.84, 0.87],
    ...
}
```

- History is a **simple list** of quality values
- **No explicit decay**
- **No exponential smoothing**
- Uses **most recent value** for selection decisions

---

## Results So Far

### Session 77: ε-Greedy Exploration
- ε=0.0: 0.4% trust_driven (router monopoly)
- ε=0.1: 7.1% trust_driven
- ε=0.2: 27.4% trust_driven (optimal)
- ε=0.3: 24.3% trust_driven
- Validated across all 48 layers (Sessions 80-82)

### Session 87: Multi-Dimensional Trust
- +27% improvement with simulated conversational signals
- Signal coverage ~33% (27 signals / 810 selections)

### Session 88: Real Conversation Data
- 0% improvement
- Signal coverage 2.7% (22 signals / 810 selections)
- **~40x sparser than simulated data**

### Session 90: Resource-Aware Trust (Your Feedback Integrated)

| Metric | Before (S89) | After (S90) | Change |
|--------|--------------|-------------|--------|
| First trust activation | Gen 1166 | Gen 133 | **+1033 gen faster** |
| Cache hit rate | N/A | 80% | Hysteresis working |
| Expert churn | N/A | 0.197 swaps/sel | Stable routing |
| Swap denials | N/A | 33 | Budget limiting thrash |

**Key finding**: Hysteresis creates positive feedback loop - experts stay loaded, accumulate trust faster, reach activation 8x sooner.

---

## What We're NOW Doing (Session 90)

1. **✅ Hysteresis/stickiness** - +20% boost for loaded experts
2. **✅ Switching cost** - Swap cost weighted in permission score
3. **✅ Budgeted exploration** - Max 8 swaps/generation
4. **✅ Memory traffic cost** - Bandwidth contention in score

## What We're Still NOT Doing

1. **No decay** - Old trust values persist indefinitely
2. **No smoothing** - Raw quality values stored directly
3. **No prefetching** - Selection doesn't predict future needs
4. **No temporal windowing** - All history weighted equally
5. **No trust vs skill distinction** - Quality directly = trust
6. **No two-stage routing** - No expert families → individuals
7. **No regret tracking** - Not measuring "wanted but unavailable"

---

## Known Issues

1. **Cold Start Monopoly** (solved by ε-greedy): Without exploration, router creates monopoly that trust can't break.

2. **Sparse Signal Problem** (Session 88): Real conversational feedback is ~40x sparser than needed for effective trust building.

3. **Potential Thrashing** (not yet observed): No mechanism to prevent rapid expert switching on small quality variations.

---

## Questions for Nova

### Already Addressed (Session 90)
- ~~Hysteresis~~ → +20% boost implemented
- ~~Switching cost~~ → Weighted in permission score
- ~~Budgeted exploration~~ → Max 8 swaps/gen

### Remaining Questions

1. **Trust Decay**: Should old observations decay? What rate? Or should we use windowed history?

2. **Smoothing**: Should trust updates use EMA? What α would prevent twitchiness while remaining responsive?

3. **Trust vs Skill**: Our "trust" is really "recent performance". Is this conflation causing issues?

4. **Hysteresis Tuning**: Is +20% boost the right magnitude? Should it vary by context or expert age?

5. **Two-Stage Routing**: Should we implement expert families → individuals to reduce "want unavailable" thrash?

6. **Regret Tracking**: How should we use regret metrics to tune the system? What threshold indicates policy failure?

7. **Signal Sparsity**: With real signals at 4% coverage, what hybrid approach best leverages sparse ground truth?

8. **Parameter Sensitivity**: How sensitive is the 1033-gen speedup to the specific parameter values (hysteresis=0.2, swap_budget=8)?

---

## Code Locations

- **Base selector**: `sage/core/trust_first_mrh_selector.py`
- **Resource-aware selector**: `sage/experiments/session90_trust_as_resource_permission.py`
- **Trust storage**: `sage/web4/context_aware_identity_bridge.py`
- **Quality bridge**: `sage/core/quality_reputation_bridge.py`
- **Session docs**: `sage/docs/SESSION72.md` through `SESSION90.md`

---

## Summary for Review

**What we implemented based on your feedback**:
- Hysteresis (+20% boost) → 80% cache hit rate
- Switching cost → Stable 0.197 swaps/selection
- Budgeted exploration (8/gen) → 33 swap denials
- Memory traffic cost → Part of permission score

**Result**: 1033 generation speedup for first trust activation

**What we're asking**:
1. Are there remaining stability failure modes we're missing?
2. Is the permission formula (expertise × cheapness × persistence) the right composition?
3. What decay/smoothing would help without over-regularizing?
4. Should we add two-stage routing and/or regret tracking next?

---

*Ready for your analysis of remaining failure modes and parameter tuning.*
