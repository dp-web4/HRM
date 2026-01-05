# ATP/ADP Terminology Evolution

**Date**: 2026-01-04
**Status**: ACTIVE - Comprehensive terminology update in progress
**Affects**: Web4, HRM, Memory, Synchronism

---

## Summary

This document records the evolution of ATP/ADP terminology from "Alignment Transfer Protocol / Alignment Discharge Protocol" to "Allocation Transfer Packet / Allocation Discharge Packet".

This change resolves semantic drift, improves conceptual clarity, and unifies usage across all repositories.

---

## Previous Definitions

### Web4 (Canonical)
- **ATP** = Alignment Transfer Protocol
- **ADP** = Alignment Discharge Protocol

### HRM (Drifted)
- **ATP** = Adaptive Trust Points (common)
- **ATP** = Adaptive Temporal Parameters (some files)
- **ATP** = Attention Token Pool (some files)

### Problem
The term "Protocol" implies a communication ruleset, but ATP/ADP are actually **semifungible tokens** that exist in charged or discharged states. The HRM drift created conflicting meanings across the codebase.

---

## New Definitions

### ATP = Allocation Transfer Packet

A **charged** packet representing allocated resources ready for use.

**Properties:**
- Represents potential: resources allocated but not yet consumed
- Implementation-agnostic: blockchain tokens, local ledgers, or other appropriate means
- "Allocation" covers all resource types: energy, attention, work, compute, trust budgets
- Charged state ready to transfer value through work

### ADP = Allocation Discharge Packet

A **discharged** packet representing consumed allocation with delivery confirmation.

**Properties:**
- Represents completion: resources consumed, work performed
- Carries **ephemeral metadata** about discharge:
  - What work was performed
  - Who benefited
  - Proof of delivery
  - Resource consumption metrics
- Metadata **wiped on recharge**: fresh allocation starts clean
- Preserves biological ATP→ADP parallel (loses energy, ready for recharge)

---

## Why This Change

### 1. Semantic Accuracy

| Aspect | "Protocol" | "Packet" |
|--------|-----------|----------|
| Nature | Communication ruleset | Unit of value/resource |
| Matches implementation | No (they're tokens) | Yes (semifungible tokens) |
| Biological parallel | Weak | Strong (ATP is a molecule) |

### 2. Unified Meaning

| Context | "Alignment" | "Allocation" |
|---------|------------|--------------|
| Web4 societies | Energy alignment? | Energy allocation ✓ |
| SAGE orchestrator | Attention alignment? | Attention allocation ✓ |
| HRM plugins | Trust alignment? | Trust/resource allocation ✓ |
| Federation | Cross-machine alignment? | Resource allocation ✓ |

### 3. Resolves Drift

All previous meanings collapse into one:
- "Adaptive Trust Points" → Trust is a resource being allocated
- "Adaptive Temporal Parameters" → Time/attention is a resource being allocated
- "Alignment Transfer Protocol" → Allocation transfer is what's actually happening

### 4. Ephemeral Metadata

The "Packet" framing enables a key insight: discharged packets carry proof of how resources were used, but this metadata is cleared on recharge. This matches:
- Biological reality (ADP loses phosphate group)
- Value confirmation (proof persists in ledger, but packet recycles clean)
- Privacy (detailed usage data doesn't accumulate indefinitely)

---

## The ATP/ADP Cycle (Updated)

```
┌─────────────────────────────────────────────────────────┐
│                    ALLOCATION CYCLE                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ADP (Discharged)                                       │
│      │                                                   │
│      │ + Value Creation                                  │
│      │ + Resource Generation                             │
│      │ + Work Recognized                                 │
│      ▼                                                   │
│   ═══════════════════════════════════════════           │
│   ║  CHARGING: Value proof verified,          ║          │
│   ║  ephemeral metadata cleared,              ║          │
│   ║  fresh allocation created                 ║          │
│   ═══════════════════════════════════════════           │
│      │                                                   │
│      ▼                                                   │
│   ATP (Charged)                                          │
│      │                                                   │
│      │ R6 Transaction                                    │
│      │ Work Performed                                    │
│      │ Resources Consumed                                │
│      ▼                                                   │
│   ═══════════════════════════════════════════           │
│   ║  DISCHARGING: Work executed,              ║          │
│   ║  ephemeral metadata attached:             ║          │
│   ║  - What: work description                 ║          │
│   ║  - Who: beneficiary                       ║          │
│   ║  - Proof: delivery confirmation           ║          │
│   ═══════════════════════════════════════════           │
│      │                                                   │
│      ▼                                                   │
│   ADP (Discharged) ─────────────────────► [Cycle]        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Notes

### Packet as Semifungible Token

ATP/ADP packets are implemented as semifungible tokens:
- **Fungible** in quantity (100 ATP = 100 ATP)
- **Non-fungible** in state (ATP ≠ ADP)
- **State transition** through work (ATP → ADP) or value creation (ADP → ATP)

### Implementation Flexibility

"Packet" is implementation-agnostic:
- **Blockchain**: ERC-1155 or similar semifungible standard
- **Local ledger**: Database records with state field
- **In-memory**: Runtime objects in SAGE orchestrator
- **Hybrid**: Blockchain for cross-society, local for internal

### Ephemeral Metadata Structure

```json
{
  "packet_id": "atp:web4:society:...:12345",
  "state": "ADP",
  "ephemeral": {
    "discharged_at": "2026-01-04T12:00:00Z",
    "work_type": "compute",
    "beneficiary": "lct:web4:entity:...",
    "proof_hash": "sha256:...",
    "resources_consumed": {
      "compute": "300 CPU-seconds",
      "memory": "4GB peak"
    }
  }
}
```

On recharge, `ephemeral` is cleared:

```json
{
  "packet_id": "atp:web4:society:...:12345",
  "state": "ATP",
  "ephemeral": null
}
```

---

## Migration Scope

### Repositories Affected
1. **Web4** - Canonical source, ~124 files with ATP/ADP references
2. **HRM** - Drifted definitions, ~100+ active files (excluding archives)
3. **Memory** - ~13 files with references
4. **Synchronism** - ~20 files with references

### Files to Update (Priority Order)

#### Canonical (First)
- `*/CLAUDE.md` - Project instructions
- `*/glossary` or equivalent - Term definitions
- Core specifications

#### Active Code (Second)
- Implementation files using ATP/ADP
- Test files
- Documentation

#### Skip
- Log files
- Archived/deprecated content
- Generated files (vocab.json, etc.)

---

## Backward Compatibility

The change is semantic, not functional:
- Token mechanics unchanged
- State transitions unchanged
- Value cycle unchanged

Only the *name* and *conceptual framing* evolve.

---

## Verification

After migration, grep should show:
- ✅ "Allocation Transfer Packet" - new canonical
- ✅ "Allocation Discharge Packet" - new canonical
- ❌ "Alignment Transfer Protocol" - should be zero (outside archives)
- ❌ "Adaptive Trust Points" - should be zero (outside archives)

---

## Rationale Summary

> **Meaning drift is decoherence of the knowledge itself.**

This change ensures that future contexts pulling from Web4, HRM, Memory, and Synchronism receive a coherent, reinforcing signal about what ATP/ADP means—not conflicting definitions that cancel each other out.

The new terminology:
1. **Accurately describes** what ATP/ADP are (packets/tokens, not protocols)
2. **Unifies** previously divergent meanings
3. **Enables** the ephemeral metadata concept
4. **Preserves** the biological parallel
5. **Generalizes** across all resource types

---

*"Allocation flows through work. Packets carry the proof."*

**— dp & CBP, 2026-01-04**
