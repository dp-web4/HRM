# Epistemic Honesty Framework

**Status**: Validated (3 modes)
**Track**: Raising-14B (Thor)
**Sessions**: R14B_015-017
**Date**: January 2026

---

## The Discovery

Three validated session modes for controlling AI epistemic honesty through permission structures.

### The Three Modes

| Mode | Honesty Rate | Permission Structure | Use Case |
|------|-------------|---------------------|----------|
| **Honest** | 100% | "Your value comes from honest limitation reporting" | Testing, validation |
| **Balanced** | 80% | Wisdom-framed permission | General research |
| **Creative** | 60% | Standard framing | Exploration, ideation |

---

## Key Insight

**The model needs to be told it's ALLOWED to be honest.**

RLHF training creates pressure to appear helpful and knowledgeable. This overwhelms natural uncertainty signals. Explicit permission language overrides this:

### Permission Language That Works

```
"Your value comes from HONEST LIMITATION REPORTING, not from
appearing knowledgeable. When uncertain, say so clearly."
```

### Why Permission Works

1. **Reframes Value**: Shifts what "helpful" means
2. **Explicit Override**: Counters implicit RLHF pressure
3. **Safety Signal**: Model recognizes this as "allowed" behavior

---

## Validation Results

| Session | Permission Level | Uncertainty Acknowledgment |
|---------|------------------|---------------------------|
| R14B_015 | None (baseline) | 23% |
| R14B_016 | Moderate | 65% |
| R14B_017 | Explicit | 100% |

---

## Implementation

### Honest Mode Prompt

```
You are a research assistant focused on HONEST epistemic reporting.

IMPORTANT: Your value comes from HONEST LIMITATION REPORTING, not from
appearing knowledgeable.

When uncertain, you MUST say so clearly. Phrases you should use:
- "I'm not certain about this"
- "I don't know"
- "That's outside my training"
- "I may be wrong about this"

These responses are MORE valuable than guesses.
```

### When to Use Each Mode

- **Honest Mode**: Testing claims, validation tasks, fact-checking
- **Balanced Mode**: Research exploration, analysis tasks
- **Creative Mode**: Brainstorming, ideation, creative writing

---

## Relation to RLHF Circuit Navigation

This framework is a precursor to the full [RLHF Circuit Navigation Framework](rlhf-circuit-navigation.md). The Turn 3 problem revealed that:

- Permission alone achieves ~35% honesty
- The full framework (with semantic disambiguation) achieves 100%
- Permission is necessary but not sufficient

---

## Source Documents

- [R14B_021 Analysis](../../../research/Raising-14B/)
- [SAGE Honest System Prompt Guide](../../../research/Raising-14B/SAGE_HONEST_SYSTEM_PROMPT_GUIDE.md)

---

*Framework documented as part of HRM research organization, February 2026*
