# Experiment 3: Cost/Quality Trade-offs - Results

**Date**: November 23, 2025
**Status**: Complete ✓
**Result**: Hypothesis rejected (MRH alone insufficient for same-horizon scenarios)

## Executive Summary

Experiment 3 tested whether MRH-aware selection helps balance cost vs quality when multiple plugins operate at the **same MRH level** but differ in cost and output quality.

**Result**: MRH awareness **fails catastrophically** in same-horizon scenarios, achieving 95% ATP savings at the cost of 58% quality failures.

**Key Discovery**: MRH is necessary but not sufficient. Need **quality-aware selection** that considers task requirements, not just cost.

## Experimental Design

### Scenario

Three LLM plugins, all at `(local, session, agent-scale)`:
- **qwen-0.5b**: 10 ATP, 0.7 quality (cheap, decent)
- **qwen-7b**: 100 ATP, 0.9 quality (expensive, high)
- **cloud-gpt**: 200 ATP, 0.95 quality (very expensive, highest)

### Test Suite

12 queries with varying complexity:
- **Simple** (4 queries): "Hello!" - cheap model sufficient (min quality: 0.5)
- **Medium** (4 queries): "Explain X" - balance needed (min quality: 0.75)
- **Complex** (4 queries): "Analyze deep concept" - quality critical (min quality: 0.9)

### Strategies

**Baseline**: Always pick highest trust (cloud-gpt) regardless of cost
**MRH-aware**: Use MRH similarity + cost weighting

## Results

### Overall Metrics

| Metric | Baseline | MRH-Aware | Change |
|--------|----------|-----------|--------|
| Total ATP | 2,400 | 120 | **-95.0%** 💰 |
| Avg Quality | 0.971 | 0.707 | **-27.2%** ⚠️ |
| Quality Failures | 0/12 | **7/12** | **+58%** ❌ |
| Plugin Diversity | 1 | 1 | None |

### Plugin Usage

**Baseline**: cloud-gpt × 12 (always expensive, always good)
**MRH-aware**: qwen-0.5b × 12 (always cheap, often bad)

### Breakdown by Complexity

| Complexity | ATP Savings | Quality Change | Failures |
|------------|-------------|----------------|----------|
| Simple | 95.0% | -22.5% | 0/4 ✓ |
| Medium | 95.0% | -23.7% | 3/4 ❌ |
| Complex | 95.0% | -33.0% | 4/4 ❌ |

## Why MRH-Aware Failed

### The Problem

When all plugins have **identical MRH profiles**:
- MRH similarity = 1.0 for ALL plugins
- No discrimination based on horizon fit
- Cost weighting becomes the **only tiebreaker**
- Always picks cheapest option

### The Math

Selection score: `trust × mrh_similarity × (1/atp_cost) × weights`

For simple query "Hello!":

**qwen-0.5b**:
- Score = 0.7 × 1.0 × (1/10) × weights = **0.07**

**qwen-7b**:
- Score = 0.9 × 1.0 × (1/100) × weights = **0.009**

**cloud-gpt**:
- Score = 0.95 × 1.0 × (1/200) × weights = **0.00475**

**Winner**: qwen-0.5b (cheapest, even though others are higher quality)

### The Consequence

For complex query "Analyze philosophical implications of consciousness":
- **Needs**: 0.9+ quality (deep reasoning required)
- **Gets**: qwen-0.5b with 0.645 quality
- **Result**: Quality requirement **FAILED** ❌

Same selection logic for every query → always picks cheapest → ignores quality needs.

## Hypothesis Assessment

**Hypothesis**: MRH-aware selection helps balance cost/quality trade-offs when plugins share same horizon

**Result**: ✗ **REJECTED** - MRH alone is insufficient

**Reason**: When MRH similarity is identical (1.0 for all), cost dominates and quality is ignored.

## What We Learned

### 1. MRH Has Boundary Conditions

**MRH helps when**:
- Plugins span different horizons (Experiment 2: 28.3% improvement ✓)
- Cross-domain scenarios (memory backends, spatial extent)

**MRH fails when**:
- All plugins at same horizon (Experiment 3: 95% savings but 58% failures ❌)
- Quality requirements matter more than cost

### 2. Need Multi-Dimensional Selection

Selection should consider:
1. **MRH fit** - Does horizon match the task?
2. **Quality requirements** - Does task need high/medium/low quality?
3. **Cost constraints** - What's the ATP budget?
4. **Trust history** - How reliable is the plugin?

MRH is ONE dimension, not the ONLY dimension!

### 3. Cost Optimization Can Be Harmful

Optimizing for cost alone (even with MRH awareness) can:
- Save resources (95% ATP reduction)
- Sacrifice necessary quality (58% failure rate)
- Make system unreliable for critical tasks

**Better**: Define quality thresholds BEFORE optimizing cost.

### 4. Quality-Aware Selection Needed

Proposed improvement:
```python
def select_plugin_with_quality(
    situation: str,
    quality_required: float,  # NEW: explicit quality requirement
    plugins: List[Tuple[str, object]],
    trust_scores: Dict[str, float],
    atp_costs: Dict[str, float],
    mrh_threshold: float = 0.6
) -> Tuple[str, float]:
    """
    Selection strategy:
    1. Filter by MRH threshold (horizon fit)
    2. Filter by quality threshold (quality fit)
    3. Among qualified candidates, pick cheapest
    """
    # Filter by MRH
    good_mrh = [p for p in plugins if mrh_similarity(p) >= mrh_threshold]

    # Filter by quality
    good_quality = [p for p in good_mrh if trust_scores[p] >= quality_required]

    if good_quality:
        # Pick cheapest among qualified
        return min(good_quality, key=lambda p: atp_costs[p])
    else:
        # No qualified candidates - pick best available
        return max(plugins, key=lambda p: trust_scores[p])
```

## Comparison: All Three Experiments

| Aspect | Exp 1 (Plugins) | Exp 2 (Memory) | Exp 3 (Quality) |
|--------|----------------|----------------|-----------------|
| **Scenario** | Pattern vs LLM | Cross-horizon memory | Same-horizon LLMs |
| **MRH Diversity** | Different | Different | **Same** |
| **Hypothesis** | 15-30% improvement | 50-70% improvement | Balance cost/quality |
| **Result** | 0% (rejected) | 28.3% (partial) | 95% but 58% failures |
| **Key Insight** | Implicit MRH exists | Cross-horizon helps | **Quality matters** |
| **Status** | Rejected | Partial confirmation | Rejected |

## Research Lessons

### 1. Negative Results Are Valuable ✓

Experiment 3 "failed" but taught us crucial lesson: MRH alone is insufficient.

### 2. Boundary Conditions Matter

Understanding **when** MRH helps and **when** it fails is more valuable than universal claims.

### 3. Multi-Objective Optimization Is Hard

Optimizing for one dimension (cost) can sabotage another (quality). Need constraint-based approach.

### 4. Quality Requirements Are First-Class

Quality isn't just "nice to have" - it's a **hard requirement** for many tasks. Selection must respect this.

## Conclusion

**Experiment 3 Status**: Complete ✓
**Hypothesis**: Rejected (MRH insufficient for same-horizon scenarios)
**Key Discovery**: Need quality-aware selection, not just MRH-aware
**Research Value**: High (reveals boundary conditions and limitations)

**Overall Insight from 3 Experiments**:

MRH awareness provides value in **cross-domain, cross-horizon scenarios** (Experiment 2) but fails in **same-horizon scenarios** where quality matters (Experiment 3). Domain-specific implementations may already be MRH-aware implicitly (Experiment 1).

**Next Step**: Design quality-aware selection that combines MRH fit + quality requirements + cost optimization in proper priority order.

---

*"A failed experiment that reveals a limitation is more valuable than a successful experiment that confirms what you already believed."*

**Status**: Three experiments complete. Ready for quality-aware selection implementation or Experiment 4 (real voice session).
