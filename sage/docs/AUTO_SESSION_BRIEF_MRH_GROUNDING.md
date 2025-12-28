# Auto Session Brief: MRH Grounding Integration

**Date**: 2025-12-27
**Priority**: Medium (integration opportunity)
**Blocking**: No - enhancement, not prerequisite
**Related**: Fractal IRP Architecture, Sessions 95/129

---

## Context

Web4 has a new proposal for **Grounding** as a fifth MRH relationship type. This is directly relevant to SAGE/IRP work.

**Proposal Location**: `/Web4/proposals/MRH_GROUNDING_PROPOSAL.md`

---

## What Grounding Adds

Grounding captures **ephemeral operational presence** - where an entity IS and what it CAN do right now, distinct from identity (Binding), authorization (Pairing), or earned trust (Witnessing).

### Coherence Index (CI)

A real-time multiplier on trust computed from:
- Spatial coherence (plausible location given movement history)
- Capability coherence (advertised capabilities match hardware class)
- Temporal coherence (activity timing matches historical patterns)
- Relational coherence (interactions within usual MRH neighborhood)

---

## SAGE-Specific Extensions

The proposal includes SAGE-specific grounding context:

```python
class SAGEGroundingContext(GroundingContext):
    hardware_attestation: HardwareAttestation  # TPM/secure enclave proof
    model_state: {
        active_model: ModelID
        quantization: QuantizationLevel
        memory_pressure: float
    }
    federation_state: {
        connected_peers: [LCT]
        consensus_role: "leader" | "follower" | "observer"
        last_sync: ISO8601
    }
```

---

## Integration Opportunities

### 1. IRP Expert Grounding

Extend `ExpertDescriptor` to include grounding context:

```python
@dataclass
class ExpertDescriptor:
    # Existing fields...
    grounding: Optional[GroundingContext] = None
    coherence_index: Optional[float] = None  # Computed, not stored
```

### 2. Coherence-Aware Routing

Incorporate CI into expert selection:

```python
def score_expert(expert: ExpertDescriptor, context: TaskContext) -> float:
    # Existing scoring...

    # Add coherence factor
    if expert.coherence_index:
        if expert.coherence_index < 0.5:
            score *= expert.coherence_index  # Heavy penalty
        elif expert.coherence_index < 0.8:
            score *= 0.9  # Mild penalty

    return score
```

### 3. Federation Coherence

For multi-machine SAGE (Legion ↔ Thor ↔ Sprout):

```python
def federation_coherence(instances: [LCT]) -> float:
    """Are federated instances behaving coherently?"""
    # Check sync drift
    # Check leader consensus
    # Check partition indicators
```

### 4. Metabolic State as Grounding

Sessions 95/129 emotional/metabolic state IS a form of grounding:

```python
# Already have:
emotional_state: EmotionalStateAdvertisement
metabolic_mode: MetabolicState  # WAKE, FOCUS, REST, DREAM, CRISIS

# Maps to grounding:
grounding.capabilities.resource_state = {
    compute: metabolic_mode_to_compute(metabolic_mode),
    # ...
}
```

---

## Suggested Actions

1. **Read the full proposal** in Web4/proposals/MRH_GROUNDING_PROPOSAL.md
2. **Consider adding grounding to IRP descriptors** in next iteration
3. **Map emotional/metabolic state to grounding context** for unified model
4. **Implement federation coherence check** for distributed SAGE

---

## Not Required Now

This is an enhancement opportunity, not a blocker. The Fractal IRP v0.2 spec can proceed without grounding - it can be added in v0.3.

---

*Grounding completes the presence picture: not just who you ARE (LCT), but where you ARE and what you CAN do right now.*
