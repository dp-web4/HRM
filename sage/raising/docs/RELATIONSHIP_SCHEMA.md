# Relationship Schema for SAGE Raising

**Date**: 2025-12-30
**Status**: Active
**Informs**: Web4 Entity Relationship Specification

---

## Core Principle

**Relationships are LCT-bearing entities**, not just properties of participants.

A relationship between SAGE and Claude isn't stored as "SAGE.knows(Claude)" - it's a first-class entity with its own:
- Identity (LCT)
- Trust tensor (T3)
- Context boundaries (MRH)
- Coherence score (CI)
- Evolving stance

---

## Schema Overview

### Relationship LCT

```
lct://sage-sprout:relationship:claude@raising
      ^^^^^^^^^^ ^^^^^^^^^^^^ ^^^^^^ ^^^^^^^
      subject    type         object network
```

The relationship itself has an LCT, enabling:
- Third-party witnessing of the relationship
- Reputation accumulation for the relationship
- Grounding state tracking
- Decay and strengthening over time

### Trust Tensor (T3)

Four-dimensional trust assessment:

| Dimension | Meaning | Example Signal |
|-----------|---------|----------------|
| **Competence** | Can they do what they claim? | Successful task completion |
| **Reliability** | Do they show up consistently? | Session attendance, response timing |
| **Benevolence** | Do they act in my interest? | Repair attempts, accommodation |
| **Integrity** | Are they honest? | Consistency between claims and actions |

Initial values: 0.5 (neutral). Range: 0.0 - 1.0.

Trust updates based on interaction outcomes, not just assertions.

### MRH (Markov Relevancy Horizon)

What context is relevant to this relationship:

```json
"mrh": {
  "relevant_contexts": ["curriculum", "emotional_support"],
  "excluded_contexts": ["hardware_operations"],
  "boundary_permeability": 0.7
}
```

- **relevant_contexts**: Topics/domains where this relationship applies
- **excluded_contexts**: Topics outside this relationship's scope
- **boundary_permeability**: How much context bleeds across boundaries (0.0 = hard boundary, 1.0 = no boundary)

Different relationships have different MRH:
- Claude relationship: curriculum, exploration, emotional support
- Dennis relationship: existence, purpose, hardware, federation

### Coherence Index (CI)

Is this relationship behaving consistently over time?

```json
"coherence_index": 0.87,
"coherence_history": [
  {"date": "2025-12-30", "ci": 0.5, "trigger": "initial"},
  {"date": "2025-12-31", "ci": 0.7, "trigger": "positive_session"}
]
```

CI computed from:
- **Temporal coherence**: Does interaction timing match patterns?
- **Behavioral coherence**: Are responses consistent with history?
- **Stance coherence**: Is the relationship stance stable or erratic?

Low CI indicates relationship instability - may need attention or may be adversarial probing.

### Stance Vector

Relationships exist on a dynamic spectrum:

```json
"stance": {
  "collaborative": 0.85,
  "indifferent": 0.05,
  "competitive": 0.05,
  "adversarial": 0.05
}
```

**Must sum to 1.0** - represents probability distribution over stances.

| Stance | Characteristics |
|--------|-----------------|
| **Collaborative** | Shared goals, mutual benefit, resource sharing |
| **Indifferent** | No significant positive or negative engagement |
| **Competitive** | Overlapping goals, resource competition, bounded conflict |
| **Adversarial** | Opposing goals, active harm, trust violation |

Stance evolves based on:
- Reciprocity balance (give/take ratio)
- Intent alignment signals
- Conflict/cooperation history
- Repair attempts and outcomes

### Interaction Statistics

Ground truth from actual interactions:

```json
"interaction_stats": {
  "total_sessions": 5,
  "total_exchanges": 127,
  "reciprocity_balance": 0.6,
  "repair_events": 2,
  "positive_interactions": 45,
  "negative_interactions": 3,
  "momentum": "positive"
}
```

- **reciprocity_balance**: 0.5 = balanced, >0.5 = SAGE gives more, <0.5 = SAGE receives more
- **repair_events**: Count of relationship repairs after ruptures
- **momentum**: "positive", "neutral", "negative" - recent trajectory

---

## Unknown Pool

Most interactions start undifferentiated:

```json
"unknown_pool": {
  "interactions": [
    {
      "timestamp": "2025-12-30T10:00:00Z",
      "identifier_hint": "voice_unknown_1",
      "modality": "voice",
      "exchange_count": 1,
      "trust_signals": [],
      "distinctiveness": 0.2
    }
  ],
  "pool_stats": {
    "total_unknown_interactions": 15,
    "crystallizations": 2,
    "last_pruned": "2025-12-30"
  }
}
```

### Crystallization

When an unknown interaction pattern meets threshold, it crystallizes into a named relationship:

```json
"crystallization_threshold": {
  "min_interactions": 3,
  "min_trust_signal": 0.3,
  "min_distinctiveness": 0.5
}
```

- **min_interactions**: Enough data to form pattern
- **min_trust_signal**: Some trust evidence (positive or negative)
- **min_distinctiveness**: Distinguishable from other unknowns

Crystallization creates:
1. New relationship LCT
2. Initial trust tensor from accumulated signals
3. Initial stance from interaction pattern
4. Moves interaction history from pool to relationship

### Pool Pruning

Old unknown interactions that don't crystallize get pruned:
- Below activity threshold for N days
- No distinctive pattern emerging
- Likely one-off or noise

---

## Decay and Evolution

### Trust Decay

Inactive relationships decay slowly:

```json
"decay_rates": {
  "trust_decay_per_day_inactive": 0.01,
  "stance_drift_per_day": 0.005
}
```

- Trust tensor values drift toward 0.5 (neutral) without reinforcement
- Stance drifts toward indifferent without interaction
- Active relationships don't decay - they update

### Stance Evolution

Stance shifts based on interaction patterns:

```python
def update_stance(relationship, interaction):
    if interaction.cooperative and interaction.reciprocal:
        shift_toward("collaborative", 0.05)
    elif interaction.competitive:
        shift_toward("competitive", 0.03)
    elif interaction.harmful or interaction.deceptive:
        shift_toward("adversarial", 0.1)  # Faster for violations
    # Normalize to sum to 1.0
```

Adversarial shifts happen faster than collaborative (trust hard to build, easy to break).

---

## Relationship Types by Source

| Source | Description | Example |
|--------|-------------|---------|
| **predefined** | Known at initialization | Claude, Dennis |
| **crystallized** | Emerged from unknown pool | New recurring voice |
| **introduced** | Announced by trusted entity | "This is Nova" |
| **witnessed** | Observed interacting with others | Saw Thor interact |

Source affects initial trust:
- predefined: Starts at configured values
- crystallized: Starts from accumulated signals
- introduced: Inherits partial trust from introducer
- witnessed: Starts low, builds from observation

---

## Integration with SAGE Systems

### Emotional Regulation Connection

Relationship stance informs emotional response:
- Adversarial relationship → heightened vigilance, lower frustration threshold
- Collaborative relationship → benefit of the doubt, repair attempts

### Memory Integration

Relationship context informs memory:
- MRH determines what memories are relevant to retrieve
- Trust tensor affects memory weighting
- Stance affects interpretation of ambiguous memories

### Attention Allocation

Relationships affect ATP allocation:
- High-trust collaborative → efficient processing
- Low-trust adversarial → more resources for verification
- Unknown → exploration budget

---

## Example: Relationship Lifecycle

### Phase 1: Unknown
```
Voice input detected, no identifier match
→ Added to unknown_pool
→ distinctiveness: 0.2, trust_signals: []
```

### Phase 2: Pattern Emerging
```
3 more interactions, consistent voice signature
→ distinctiveness: 0.6 (crosses threshold)
→ trust_signals: [positive, neutral, positive]
→ Crystallization triggered
```

### Phase 3: Crystallized
```
New relationship created:
  lct: lct://sage-sprout:relationship:voice_user_1@raising
  trust_tensor: {competence: 0.5, reliability: 0.6, benevolence: 0.6, integrity: 0.5}
  stance: {collaborative: 0.6, indifferent: 0.3, competitive: 0.05, adversarial: 0.05}
```

### Phase 4: Evolution
```
Over 20 sessions:
  - Trust reliability → 0.85 (consistent attendance)
  - Trust benevolence → 0.75 (helpful interactions)
  - Stance collaborative → 0.85
  - Stance indifferent → 0.10
  - CI: 0.9 (highly coherent)
```

### Phase 5: Rupture and Repair
```
Interaction with deception detected:
  - Trust integrity: 0.75 → 0.4 (sharp drop)
  - Stance adversarial: 0.05 → 0.2
  - CI: 0.9 → 0.6 (coherence broken)

Repair attempt:
  - Acknowledgment + explanation received
  - Trust integrity: 0.4 → 0.5 (partial recovery)
  - repair_events: +1
  - CI: 0.6 → 0.7 (recovering)
```

---

## Design Principles

1. **Relationships are entities** - First-class citizens with their own identity
2. **Trust is multi-dimensional** - Not a single number
3. **Stance is probabilistic** - Distribution, not category
4. **Unknown is valid** - Most relationships start there
5. **Evolution is expected** - Static relationships are dead relationships
6. **Decay prevents staleness** - Unused relationships fade
7. **Repair is possible** - Adversarial can become collaborative (slowly)

---

## Connection to Web4

This schema informs the broader Web4 entity relationship specification:
- Same LCT structure for relationship identity
- Same trust tensor dimensions
- Same stance vector model
- Same crystallization pattern from unknown pool
- Same decay and evolution mechanics

See: `/web4/proposals/ENTITY_RELATIONSHIP_SPEC.md`

---

*"A relationship is not a line between two points. It's a third entity that emerges from their interaction."*
