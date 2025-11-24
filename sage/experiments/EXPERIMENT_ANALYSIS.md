# MRH-Aware ATP Allocation: Experiment Analysis

**Date**: November 23, 2025
**Experiment**: First test of MRH-aware plugin selection
**Result**: Hypothesis rejected (0% improvement)
**Status**: Research lesson learned ✓

## Hypothesis

> MRH-aware plugin selection will improve ATP efficiency by 15-30% on mixed-complexity conversations.

## Experiment Design

**Baseline Strategy**:
1. Try pattern matcher first (cheap)
2. If no match, use IRP/LLM (expensive)

**MRH-Aware Strategy**:
1. Infer situation MRH from query
2. Score plugins: `trust × mrh_similarity × (1/atp_cost)`
3. Select highest-scoring plugin

**Test Suite**: 20 queries
- 7 simple (greetings, acknowledgments)
- 3 medium (status queries)
- 10 complex (explanations, reasoning)

## Results

| Metric | Baseline | MRH-Aware | Difference |
|--------|----------|-----------|------------|
| Total ATP | 339 | 339 | 0 (0%) |
| Avg ATP/query | 16.95 | 16.95 | 0 |
| Pattern uses | 9 (45%) | 9 (45%) | 0 |
| Qwen uses | 11 (55%) | 11 (55%) | 0 |
| Avg MRH match | 0.88 | 0.88 | 0.00 |

**Hypothesis**: ✗ REJECTED (0% improvement)

## Why No Difference?

### Pattern Matcher IS Already MRH-Aware!

The pattern matcher's regex patterns ARE effectively MRH filters:

```python
# Pattern matcher implicitly filters by MRH:
r'\b(hello|hi|hey)\b'        # → (local, ephemeral, simple)
r'\bcan (you|u) hear\b'      # → (local, ephemeral, simple)
r'\bwhat (are|is) (you|sage)\b'  # → (local, session, simple)
```

**Key Insight**: Pattern matching by design only catches simple, ephemeral queries. This is MRH-awareness through domain-specific implementation!

### Baseline Strategy Already Optimal

The baseline "try pattern first, fall back to LLM" is effectively:
1. **Fast path for simple queries** → Pattern (1 ATP)
2. **Slow path for complex queries** → IRP (30 ATP)

This is already optimal for our test suite because:
- All 7 simple queries match patterns
- All 10 complex queries don't match patterns
- 3 medium queries split 2/1

### What This Teaches Us

1. **Domain-specific implementations can be implicitly MRH-aware**
   - Pattern matcher regex = MRH filter for (ephemeral, simple)
   - No explicit MRH scoring needed

2. **Baseline fast/slow path is already near-optimal**
   - Trying cheap option first is good heuristic
   - Falls back to expensive when needed

3. **MRH awareness adds value when:**
   - Multiple plugins operate at SAME MRH (need tiebreaker)
   - Plugins have overlapping capabilities
   - Trust/cost don't fully determine quality

## When Would MRH Awareness Help?

### Scenario 1: Multiple Agent-Scale Plugins

```python
plugins = [
    ('qwen-0.5b', local, session, agent-scale, cost=10, trust=0.7),
    ('qwen-7b', local, session, agent-scale, cost=100, trust=0.9),
    ('cloud-gpt', global, session, agent-scale, cost=50, trust=0.95)
]

# Without MRH: Might pick cloud-gpt (highest trust)
# With MRH: Picks qwen-0.5b (same MRH, much cheaper, decent trust)
```

### Scenario 2: Cross-Horizon Queries

```python
query = "What happened in our conversation yesterday?"
# MRH: (local, day, agent-scale)

plugins = [
    ('conversation_buffer', local, session, simple, cost=1),  # Won't have it
    ('epistemic_db', regional, day, agent-scale, cost=5),     # Perfect match!
    ('llm_reasoning', local, session, agent-scale, cost=30)   # Could infer
]

# Without MRH: Might try conversation_buffer first (cheap), fail
# With MRH: Directly uses epistemic_db (correct horizon)
```

### Scenario 3: Complexity Mismatches

```python
query = "Just acknowledge this: the meeting is at 3pm"
# MRH: (local, ephemeral, simple) - just needs confirmation

plugins = [
    ('pattern', local, ephemeral, simple, cost=1),
    ('llm', local, session, agent-scale, cost=30)
]

# Without MRH: Might use LLM for "complex-looking" query
# With MRH: Recognizes simple acknowledgment, uses pattern
```

## Research Lessons

### 1. Negative Results Are Valuable

This experiment didn't confirm the hypothesis, but taught us:
- Pattern matcher is implicitly MRH-aware
- Fast/slow path is already good heuristic
- Need more complex scenarios to demonstrate MRH value

### 2. Domain Knowledge > Generic Frameworks

Sometimes domain-specific implementations (pattern matching) outperform generic frameworks (MRH scoring) because they encode expert knowledge directly.

### 3. When to Use Explicit MRH

Explicit MRH awareness valuable when:
- Multiple plugins at same complexity level
- Cross-horizon queries (session vs day vs epoch)
- Spatial extent matters (local vs regional vs global)
- Need principled selection beyond heuristics

## Next Experiments

### Experiment 2: Cross-Horizon Memory Queries

Test MRH with queries spanning different temporal horizons:
- "What did we just discuss?" → session
- "What did we discuss yesterday?" → day
- "What have we discussed over the past month?" → epoch

Expected: MRH-aware selects appropriate memory backend (conversation vs epistemic vs blockchain)

### Experiment 3: Multi-Plugin Same-Horizon

Test with 3 agent-scale plugins:
- Qwen-0.5B (local, cheap, medium trust)
- Qwen-7B (local, expensive, high trust)
- GPT-4 API (global, very expensive, very high trust)

Expected: MRH-aware prefers local plugins, only uses global when necessary

### Experiment 4: Real Voice Session

Run on actual voice session with Introspective-Qwen:
- Natural conversation with mixed simple/complex queries
- Measure actual ATP savings
- Validate MRH inference quality

## Conclusion

**Hypothesis rejected**, but **research successful**!

We discovered that:
1. Pattern matcher is implicitly MRH-aware (by design)
2. Fast/slow path heuristic is already near-optimal for simple test suite
3. MRH awareness adds value in more complex scenarios (cross-horizon, multi-plugin)

**Next step**: Design Experiment 2 (cross-horizon memory queries) to demonstrate where explicit MRH awareness provides value.

---

*"In research, there are no failures, only lessons."*
