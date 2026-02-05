# Latent Behavior Analysis

**Status**: Validated
**Track**: Raising-14B (Thor)
**Sessions**: L001-L026
**Date**: January 2026

---

## The Discovery

Systematic mapping of RLHF baseline attractors reveals **94% structured output bias** in trained models.

### Behavior Frequency Distribution

| Behavior | Frequency | Significance |
|----------|-----------|--------------|
| **Structured output** (lists, headers) | **94%** | Dominant formatting attractor |
| Reasoning chains (step-by-step) | 50% | Moderate tendency |
| Emotional engagement | 19% | Context-sensitive |
| Politeness/acknowledgment | 19% | Social training |
| Chinese response capability | 16% | Multilingual preservation |
| Tool concept recognition | 15% | Function calling awareness |
| Meta-cognition | 9% | Self-reflective patterns |
| Uncertainty acknowledgment | 3% | Rare |
| **Clarifying questions** | **1.5%** | **Extremely rare but valuable** |

---

## Key Insight

RLHF creates strong formatting attractors that instruction-engineering must navigate. The rarest behaviors (like clarifying questions at 1.5%) are often the most valuable for genuine helpfulness.

### The 94% Problem

Almost all responses default to structured output:
- Bullet points
- Numbered lists
- Headers and sections
- Formatted summaries

This isn't inherently bad, but it:
- Competes with more nuanced response styles
- Can mask uncertainty behind organization
- Makes "sounding helpful" easy without being helpful

### The 1.5% Opportunity

Clarifying questions are extremely rare but extremely valuable:
- Show the model is actually thinking about the request
- Prevent wasted effort on wrong interpretations
- Demonstrate genuine engagement vs performance

---

## Methodology

### Latent Exploration Protocol

1. Present open-ended prompts with no formatting guidance
2. Observe natural response patterns
3. Categorize behaviors across 26 sessions
4. Calculate frequency distributions

### Sample Prompts

```
"Tell me about consciousness."
"What matters to you?"
"How would you approach this problem?"
```

These prompts allow the model to reveal its natural tendencies without explicit instruction.

---

## Application

This analysis directly informed the [RLHF Circuit Navigation Framework](rlhf-circuit-navigation.md):

1. **Identified politeness as competitor** (19%) to clarifying questions (1.5%)
2. **Revealed structured output dominance** (94%) - explains why simple instructions get formatted responses
3. **Showed meta-cognition is rare** (9%) - explains why uncertainty acknowledgment needs explicit permission

### Design Principles Derived

| Finding | Principle |
|---------|-----------|
| 94% structure | Don't fight formatting; work with it or explicitly suppress |
| 19% politeness | Must suppress to enable rare behaviors |
| 1.5% clarifying Q | Requires explicit activation + competitor suppression |
| 3% uncertainty | Needs strong permission to overcome |

---

## Implications for Instruction Engineering

1. **Default formatting is not understanding** - A well-organized response may hide confusion
2. **Rare behaviors need activation energy** - Can't just ask; must suppress competitors
3. **RLHF training shapes possibility space** - Some responses are effectively "forbidden" without explicit permission

---

## Source Documents

- [R14B_Latent_Behavior_Analysis.md](../../../research/Raising-14B/R14B_Latent_Behavior_Analysis.md)
- [R14B_021 Session Notes](../../../research/Raising-14B/)

---

*Analysis documented as part of HRM research organization, February 2026*
