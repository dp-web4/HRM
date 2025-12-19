# Session 74: Trust-First Real Model Integration

**Date**: 2025-12-19
**Status**: Integration Challenges Discovered
**Platform**: Thor (Jetson AGX Thor)

---

## Goal

Integrate trust-first conditional selector (Sessions 72-73) with real Q3-Omni model inference.

**Building On**:
- Session 72: Trust-first paradigm shift → 58 experts (3.4x)
- Session 73: Long-term validation → 104 experts, 51 specialists
- Legion S68: Cross-platform validation → 29 experts (3.6x)

**Expected**: Trust-first achieves higher diversity than weighted blend on real model.

---

## Implementation

Created `session74_trust_first_real_model.py` with:

1. **TrustFirstMRHSelector** instantiation with context classifier
2. **SelectiveLanguageModel** integration with trust_selector parameter
3. **Real expert extraction** using `get_selected_experts_from_model()`
4. **Diversity tracking** for specialists vs generalists
5. **Mode transition monitoring** (router_explore → trust_driven)

**Architecture**:
```python
# Trust-first selector
trust_selector = TrustFirstMRHSelector(
    num_experts=128,
    min_trust_evidence=3,
    low_trust_threshold=0.3,
    context_classifier=classifier
)

# Model with trust-first selector
model = SelectiveLanguageModel(
    extraction_dir=extraction_dir,
    num_layers=1,
    trust_selector=trust_selector  # Integration point
)
```

---

## Discovery: API Incompatibility

**Problem**: `TrustFirstSelectionResult` doesn't match expected MoE layer API.

**Error**:
```
AttributeError: 'TrustFirstSelectionResult' object has no attribute 'selection_scores'
```

**Root Cause**:
- `TrustFirstSelectionResult` (Session 72/73) returns: `selected_expert_ids`, `selection_mode`, `trust_scores`
- MoE layer expects: `selection_scores` attribute for weighting experts

**Location**: `sage/compression/selective_transformer_layer.py:419`
```python
selected_weights = torch.tensor(
    result.selection_scores,  # ← Expected but not present
    device=hidden_states.device,
    dtype=hidden_states.dtype
)
```

---

## Arch itecture Analysis

### TrustFirstSelectionResult (Session 72):

```python
@dataclass
class TrustFirstSelectionResult:
    selected_expert_ids: List[int]
    selection_mode: str  # "trust_driven" or "router_explore"
    trust_evidence: bool
    context: str
    trust_scores: List[float]  # Trust scores, not selection weights
    mrh_substitutions: int
    quality_checks: int
```

### Expected by MoE Layer:

```python
# Inferred from selective_transformer_layer.py
result.selected_expert_ids  # ✓ Present
result.selection_scores     # ✗ Missing - needs weights for expert mixing
```

---

## Solution Paths

### Option 1: Add `selection_scores` to TrustFirstSelectionResult

**Approach**: Modify `TrustFirstMRHSelector` to include selection weights.

```python
@dataclass
class TrustFirstSelectionResult:
    selected_expert_ids: List[int]
    selection_mode: str
    trust_evidence: bool
    context: str
    trust_scores: List[float]
    mrh_substitutions: int
    quality_checks: int
    selection_scores: List[float]  # NEW - weights for expert mixing
```

**Implementation**:
- In `_trust_driven_selection()`: Return trust scores as selection weights
- In `_router_explore_selection()`: Return router logits as selection weights
- Normalize to sum to 1.0

**Pros**:
- Minimal changes to TrustFirstMRHSelector
- Maintains trust-first semantics

**Cons**:
- Adds field that's not used by trust-first logic directly
- Need to ensure weights align with expert IDs correctly

### Option 2: Create Adapter Layer

**Approach**: Wrap TrustFirstMRHSelector with adapter that provides `selection_scores`.

```python
class TrustFirstAdapter:
    def __init__(self, trust_selector):
        self.trust_selector = trust_selector

    def select_experts(self, router_logits, context, k, input_embedding, all_expert_ids):
        result = self.trust_selector.select_experts(...)

        # Add selection_scores based on mode
        if result.selection_mode == "trust_driven":
            selection_scores = result.trust_scores
        else:  # router_explore
            selection_scores = router_logits[result.selected_expert_ids]

        # Return augmented result
        return TrustFirstSelectionResultWithScores(
            **dataclasses.asdict(result),
            selection_scores=selection_scores
        )
```

**Pros**:
- No changes to TrustFirstMRHSelector
- Clean separation of concerns
- Easy to swap out

**Cons**:
- Additional layer of indirection
- Need to maintain adapter code

### Option 3: Update MoE Layer to Handle Trust-First Results

**Approach**: Modify `SelectiveMoELayer` to detect `TrustFirstSelectionResult` and handle appropriately.

```python
# In SelectiveMoELayer.forward()
if isinstance(result, TrustFirstSelectionResult):
    # Trust-first: use trust_scores as weights
    if result.selection_mode == "trust_driven":
        selected_weights = torch.tensor(result.trust_scores, ...)
    else:
        # router_explore: use router logits
        selected_weights = router_logits[result.selected_expert_ids]
else:
    # Original behavior
    selected_weights = torch.tensor(result.selection_scores, ...)
```

**Pros**:
- Preserves trust-first semantics exactly
- No changes to TrustFirstMRHSelector API
- MoE layer becomes "trust-aware"

**Cons**:
- Adds conditional logic to MoE layer
- Tighter coupling between components

---

## Recommended Path

**Option 1** (Add `selection_scores`) is recommended:

1. **Simplest**: Single field addition
2. **Consistent**: All selectors provide weights
3. **Minimal Impact**: LocalChanges to one class
4. **Standards-Compliant**: Matches expected MoE API

### Implementation Steps:

1. Update `Trust FirstSelectionResult` dataclass (add `selection_scores` field)
2. Modify `_trust_driven_selection()` to populate `selection_scores` from trust
3. Modify `_router_explore_selection()` to populate `selection_scores` from router
4. Test with Session 74 script
5. Validate diversity improvements vs Session 70 (weighted blend)

---

## Next Steps

### Immediate:
1. ✅ Document integration challenge (this file)
2. Implement Option 1 (add `selection_scores`)
3. Complete Session 74 integration test
4. Measure diversity on real model

### Short-Term:
5. Compare trust-first vs weighted blend on real model
6. Scale to full 48 layers
7. Extended validation with diverse workloads

### Medium-Term:
8. Federation testing (Thor → Sprout)
9. Production deployment readiness
10. Cross-model trust transfer

---

## Key Insights

**Discovery**: Trust-first architecture (Sessions 72-73) was developed and validated on *simulation*, not integrated with real model inference pipeline.

**Implication**: API contracts between selector and MoE layer need alignment before production deployment.

**Learning**: Paradigm validation != Production integration. Both are necessary steps.

**Value**: This integration work reveals the gap between research prototypes and production systems, highlighting the importance of end-to-end testing.

---

## Files Created

- `sage/experiments/session74_trust_first_real_model.py` (~420 LOC)
- `sage/experiments/SESSION74_INTEGRATION_NOTES.md` (this file)

---

## Session Statistics

- **Duration**: ~2 hours
- **Tracks**: 1 (Trust-first real model integration)
- **Lines of Code**: ~420
- **Discovery**: API incompatibility between trust-first selector and MoE layer
- **Status**: Integration path identified, implementation pending

---

**Conclusion**: Session 74 successfully identified the integration challenge between trust-first selector (Sessions 72-73) and real model inference. Clear solution paths outlined. This represents crucial bridge from research validation to production deployment.

*"The gap between simulation and production is where real engineering happens. Session 74: Found the gap. Mapped the bridge."*
