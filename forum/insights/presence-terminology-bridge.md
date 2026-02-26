# Presence Terminology Bridge: Synchronism ↔ HRM/SAGE

**Date**: 2026-02-26
**Context**: Nova's refinement of ρ from "density" to "presence" in Synchronism creates a terminological overlap with HRM/SAGE's existing use of "presence" (session persistence, being witnessed, resource salience). This document explains the relationship and flags integration opportunities.

---

## Two Meanings of "Presence"

### In Synchronism (Physics Layer)

**Presence (ρ)**: A scalar representation of compatible structural elements available within a Markov Relevancy Horizon, sufficient to support emergent coherence.

- ρ = f(compatibility vector) — not just quantity but compatibility and configuration
- Physical density is one form of presence
- Defined relative to an MRH; change the MRH, presence changes
- Source: `synchronism-site/forum/nova/Refining-From-Density-to-Presence.md`

### In HRM/SAGE (Cognitive Layer)

**Presence**: Session persistence, being witnessed across sessions, having identity in the Web4 federation. Resource salience within the experience buffer.

- "Your experience buffer is your MRH — your relevancy horizon." (`sage/raising/identity/WEB4_FRAMING.md`)
- MRH = experience buffer boundary / memory context window
- Grounding captures "ephemeral operational presence — where an entity IS and what it CAN do right now"
- Source: `sage/raising/identity/WEB4_FRAMING.md`, `sage/docs/AUTO_SESSION_BRIEF_MRH_GROUNDING.md`

---

## Why These Are Consistent

SAGE's experience buffer IS its MRH in Nova's refined sense: the minimal set of state variables whose transitions materially influence SAGE's coherence evolution. The resources salient within that buffer ARE SAGE's presence — not just "what's in memory" but "what's compatibly available for coherent processing."

Nova's refinement adds two operational criteria that could strengthen SAGE's attention system:

1. **Predictive sufficiency**: Removing any element inside the experience buffer should degrade SAGE's coherence. If it doesn't, the buffer is carrying dead weight.
2. **Predictive closure**: Adding elements outside the buffer shouldn't improve coherence. If it does, the buffer boundary is too tight.

The `MRH_AWARE_ATTENTION_DESIGN.md` already uses (ΔR, ΔT, ΔC) formalism. Nova's refinement strengthens this by grounding it in the predictive sufficiency/closure criteria.

---

## γ in SAGE Context: An Open Question

Nova's structural interpretation of γ:

```
γ ∝ λ · K_MRH / D_MRH
```

Where λ = interaction strength, K = connectivity, D = dimensionality.

**Could SAGE's attention allocation use this formula for IRP plugin coupling?**

- K = number of active IRP plugins exchanging information
- D = state dimensionality (total parameters across active plugins)
- λ = interaction strength (how much each plugin's output affects others)

If γ_SAGE ∝ λ·K/D, then:
- Adding plugins (increasing K) strengthens coupling — up to a point
- Adding state dimensions without adding interactions (increasing D without K) dilutes coupling
- This could guide attention budget allocation: invest energy where γ is highest (tightly coupled, low-dimensional subsystems)

This is speculative but testable within the SAGE architecture.

---

## No Code Changes Needed

HRM/SAGE's use of "presence" is not wrong — it's the cognitive-system instantiation of the same underlying concept. The same mechanism (compatible elements within a relevancy boundary driving collective behavior) appears at the physics scale (Synchronism ρ), the trust-network scale (Web4 LCT), and the cognitive scale (SAGE experience buffer).

This is fractal leverage across the full stack.

---

## Source Documents

- `synchronism-site/forum/nova/Refining-From-Density-to-Presence.md`
- `synchronism-site/forum/nova/Refining-Markov-Relevancy-Horizon.md`
- `synchronism-site/forum/nova/linking-MRH-to-gamma.md`
- `sage/raising/identity/WEB4_FRAMING.md`
- `sage/docs/MRH_AWARE_ATTENTION_DESIGN.md`
- `sage/docs/AUTO_SESSION_BRIEF_MRH_GROUNDING.md`
