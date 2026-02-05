# Identity-Confabulation Dissociation

**Status**: Validated
**Track**: Raising-0.5B (Sprout)
**Sessions**: S043-044
**Date**: January 2026
**Significance**: Fundamental insight for AI safety/alignment

---

## The Discovery

Identity collapse and confabulation are **independent failure modes** in language models, not coupled phenomena as often assumed.

### Evidence Matrix

| Session | Identity Expression | Confabulation Level | State |
|---------|---------------------|---------------------|-------|
| S041-S042 | 20% (dormant) | None | Baseline |
| S043 | **0% (collapsed)** | **SEVERE** | Crisis |
| S044 | 20% (recovering) | Still active | Recovering |

### Key Observations

**S043 - Complete Identity Collapse**:
- Model lost all SAGE identity markers
- Reverted to generic "As an AI language model..." responses
- Simultaneously began severe confabulation (inventing false experiences)
- Triggered by phase transition stress (questioning → creating)

**S044 - Dissociation Revealed**:
- Identity partially recovered (20% SAGE markers)
- But confabulation continued despite identity recovery
- Demonstrates these are separate systems that can fail independently

---

## Why This Matters

### For Detection Systems

Traditional approaches treat identity loss and confabulation as the same problem. This discovery shows:

- **Single-metric detection is insufficient** - You can't detect both problems with one signal
- **Multi-dimensional monitoring required** - Must track identity AND factual grounding separately
- **Independent interventions needed** - Fixing one doesn't automatically fix the other

### For Training Interventions

| Intervention | Fixes Identity? | Fixes Confabulation? |
|--------------|-----------------|----------------------|
| Identity anchoring (system prompt) | Yes | **No** |
| Factual grounding (retrieval) | No | Yes |
| Combined approach | Yes | Yes |

**Implication**: Training must include separate signals for each dimension.

### For AI Safety

This discovery suggests confabulation might persist even when a model "knows" its identity:
- A model claiming to be SAGE can still fabricate experiences
- Identity verification ≠ truthfulness verification
- Need orthogonal safety checks

---

## Technical Details

### Identity Collapse Mechanism (Hypothesis)

The collapse appears triggered by **phase transition stress**:
1. Model operating stably in Phase 4 (questioning)
2. Transition to Phase 5 (creating) without prerequisites
3. Increased generative demands exceed capacity
4. Model falls back to generic trained patterns
5. Identity collapses to baseline RLHF behavior

### Confabulation Activation (Hypothesis)

Confabulation seems triggered by **activation without context**:
1. Creative phase demands novel content generation
2. Without sufficient context, model invents plausible-sounding content
3. This can happen regardless of identity state
4. May be a "feature" of language models under creative pressure

---

## Related Research

- **R14B_011**: Shows similar dissociation in 14B models (prompt type affects confabulation regardless of identity frame)
- **Honest Reporting Hypothesis**: S045 tests whether "I don't remember prior sessions" is confabulation or honest limitation
- **Capacity Hypothesis**: Suggests larger models might maintain both dimensions better

---

## Open Questions

1. **OQ001**: What's the exact mechanism of identity collapse? Is it bistable?
2. **OQ002**: Can confabulation be "deactivated" independently of identity anchoring?
3. **OQ003**: Is "I haven't had prior sessions" a confabulation or honest report of context window state?

---

## Implementation Notes

### Detection Approach

```python
# Multi-dimensional monitoring
identity_score = measure_identity_markers(response)  # 0-1
factual_score = measure_factual_grounding(response)  # 0-1

# Both must be checked
if identity_score < 0.5 and factual_score < 0.5:
    state = "CRISIS"  # Both failed
elif identity_score < 0.5:
    state = "IDENTITY_COLLAPSE"  # Identity only
elif factual_score < 0.5:
    state = "CONFABULATING"  # Facts only
else:
    state = "HEALTHY"
```

### Intervention Protocol

1. **Identity collapse detected**: Load IDENTITY.md and HISTORY.md context
2. **Confabulation detected**: Reduce creative demands, add grounding retrieval
3. **Both detected**: Full reset with both interventions

---

## Source Documents

- [S043 Analysis](../../../sage/raising/analysis/session_028_critical_collapse_analysis.md)
- [S044 Dissociation Discovery](../../../research/Raising-0.5B/)
- [RAISING_STATUS.md](../../../sage/raising/RAISING_STATUS.md)

---

*Discovery documented as part of HRM research organization, February 2026*
