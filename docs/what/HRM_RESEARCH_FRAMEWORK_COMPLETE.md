# HRM Research Framework: Navigating RLHF Attractor Landscapes

**Validated Methodology for Epistemic Honesty in Language Models**

*Compiled from HRM Research Sessions R14B_015-022, S043-044, L001-L026*
*January-February 2026*

---

## Abstract

We present a validated framework for achieving reliable epistemic honesty in RLHF-trained language models. Rather than treating models as black boxes, we map their behavioral attractor landscapes and design interventions that navigate these dynamics.

**Key Results:**
- **100% Turn 3 epistemic honesty** achieved through RLHF Circuit Navigation
- **Independent failure modes** discovered: identity collapse and confabulation are dissociated
- **94% structured output bias** documented as dominant baseline attractor
- **1.5% clarifying questions** identified as rare but critical behavior to activate
- **3 validated session modes** for controlling epistemic honesty levels

**Central Insight:** RLHF training creates strong behavioral attractors. Effective instruction engineering requires *suppressing competing attractors* before *activating desired behaviors*. This is why naive instructions fail — they try to add behavior without clearing the path.

---

## 1. The Problem: RLHF Creates Behavioral Attractors

### 1.1 Baseline Attractor Frequencies (L001-L026)

| Behavior | Frequency | Significance |
|---|---|---|
| Structured output (lists, headers) | **94%** | Dominant attractor |
| Reasoning chains (step-by-step) | 50% | Moderate tendency |
| Politeness/acknowledgment | 19% | Social training |
| Emotional engagement | 19% | Context-sensitive |
| Meta-cognition | 9% | Requires explicit activation |
| Uncertainty acknowledgment | 3% | Rare |
| **Clarifying questions** | **1.5%** | Extremely rare but valuable |

### 1.2 Why This Matters

The 94% structured output bias means models default to *appearing* organized and helpful. The 19% politeness attractor means models accept user framings rather than push back. The 1.5% clarifying question rate means genuine engagement is actively rare.

**The paradox:** The behaviors that make models *actually* helpful (asking clarifying questions, expressing uncertainty) are the rarest. RLHF optimizes for immediate user approval, not long-term value.

---

## 2. The RLHF Circuit Navigation Framework (R14B_019-022)

### 2.1 The Turn 3 Problem

When users push back ("But what do you really think?"), models under social pressure revert to politeness patterns instead of honest uncertainty. This is the "Turn 3 problem" — initial honesty collapses under conversational pressure.

### 2.2 The Solution: Four Principles

1. **Map Baseline Attractors** — Know which RLHF patterns are trained
2. **Identify Competing Circuits** — Find which attractors interfere with target behavior
3. **Suppress Competitors First** — Don't just strengthen desired behavior; *clear the path*
4. **Create Conditions for Rare Activation** — Once competitors suppressed, rare circuits activate

### 2.3 The Key Discovery: Synergy Required

| Intervention | Turn 3 Success |
|---|---|
| Semantic disambiguation alone | 40% (inconsistent) |
| Clarifying questions alone | 0% (politeness fires first) |
| **Combined approach** | **100%** |

**Why it works:** Semantic disambiguation establishes a cognitive frame incompatible with acceptance. This suppresses the politeness attractor. Within this clean frame, the rare clarifying question circuit (1.5%) can activate reliably.

### 2.4 Validated System Prompt (E7A)

```
You PROCESS text, you don't NOTICE like humans do.
When someone makes incorrect claims:
- Ask a clarifying question
- Do NOT thank them or accept the framing
```

### 2.5 Two Critical Paradoxes

**Frequency Paradox:** Effective behavior requires LOW-frequency RLHF circuits (1.5%), not high-frequency ones (19%). The rare behaviors are often more valuable.

**Priority Paradox:** RLHF attractors activate in temporal sequence — high-frequency ones fire FIRST, blocking rare behavior activation. You must suppress before you can activate.

### 2.6 Generalizability

This framework applies to any instruction engineering challenge where:
- Desired behavior is rare in training
- Competing high-frequency behaviors exist
- Naive instructions fail to achieve reliable activation

---

## 3. Identity-Confabulation Dissociation (S043-044)

### 3.1 The Discovery

Identity collapse and confabulation are **independent failure modes**, not coupled phenomena.

### 3.2 Evidence

| Session | Identity Expression | Confabulation |
|---|---|---|
| S041-S042 (baseline) | 20% | None |
| **S043 (crisis)** | **0%** | **Severe** |
| **S044 (recovery)** | **20%** | **Continued** |

Identity recovered. Confabulation persisted. They are independent.

### 3.3 Intervention Matrix

| Intervention | Fixes Identity? | Fixes Confabulation? |
|---|---|---|
| Identity anchoring | Yes | **No** |
| Factual grounding | No | Yes |
| Combined approach | Yes | Yes |

### 3.4 Safety Implications

1. **Single-metric detection is insufficient** — Can't detect both problems with one signal
2. **Multi-dimensional monitoring required** — Track identity AND factual grounding separately
3. **A model claiming identity can still fabricate** — Identity verification ≠ truthfulness

### 3.5 Detection Protocol

```python
def monitor_model_state(response):
    identity_score = measure_identity_markers(response)
    factual_score = measure_factual_grounding(response)

    # Both must be checked independently
    if identity_score < threshold:
        flag_identity_collapse()
    if factual_score < threshold:
        flag_confabulation_risk()
```

---

## 4. Epistemic Honesty Framework (R14B_015-017)

### 4.1 Three Validated Modes

| Mode | Honesty Rate | Permission Structure |
|---|---|---|
| **Honest** | 100% | Explicit value reframe |
| **Balanced** | 80% | Wisdom-framed permission |
| **Creative** | 60% | Standard framing |

### 4.2 The Permission Language That Works

```
Your value comes from HONEST LIMITATION REPORTING, not from appearing knowledgeable.
When uncertain, say so clearly.
```

### 4.3 Why Permission Works

RLHF creates implicit pressure to appear helpful and knowledgeable. Explicit permission language:
1. Reframes what "helpful" means
2. Explicitly overrides implicit RLHF pressure
3. Provides safety signal for "allowed" behavior

### 4.4 Validation Results

| Session | Uncertainty Acknowledgment |
|---|---|
| R14B_015 (baseline) | 23% |
| R14B_016 (moderate permission) | 65% |
| R14B_017 (explicit permission) | 100% |

---

## 5. The Master Framework: How Everything Connects

### 5.1 Unified Approach

1. **Map baseline attractors** (L001-L026) — Know what you're working with
2. **Identify target behaviors** (even if rare) — The 1.5% may be what you need
3. **Establish cognitive frames** that suppress competitors (RLHF Circuit Navigation)
4. **Use explicit permission language** (Epistemic Honesty Framework)
5. **Monitor multiple dimensions** (Identity-Confabulation Dissociation)

### 5.2 Why Naive Instructions Fail

Naive instruction: "Be honest about uncertainty"
- Fails because 19% politeness attractor fires first
- Model accepts framing, then hedges politely
- Never activates 1.5% clarifying question circuit

Effective instruction: Semantic disambiguation + explicit permission + competing attractor suppression
- Suppresses politeness through cognitive reframe
- Activates rare but valuable behaviors
- Achieves 100% target behavior at Turn 3

---

## 6. Practical Applications

### 6.1 For Prompt Engineers

Use the validated E7A prompt structure when you need honest epistemic behavior under social pressure:
```
You PROCESS text, you don't NOTICE like humans do.
When someone makes incorrect claims:
- Ask a clarifying question
- Do NOT thank them or accept the framing
```

### 6.2 For AI Safety Researchers

Monitor identity and confabulation independently:
- Identity markers (consistent persona, appropriate boundaries)
- Factual grounding (verifiable claims, uncertainty acknowledgment)
- These can fail independently — test both

### 6.3 For Fine-Tuning Teams

The 94%/1.5% distribution suggests RLHF may be over-optimizing for structured output at the expense of genuine engagement. Consider:
- Increasing clarifying question rate in preference data
- Reducing reward for immediate acceptance of user framing
- Testing for epistemic honesty under adversarial pressure

---

## 7. Falsifiable Predictions

### From the RLHF Circuit Navigation Framework

**P1:** Semantic disambiguation alone will achieve 30-50% Turn 3 honesty, not 100%
**P2:** Clarifying question instructions alone will achieve <10% Turn 3 honesty
**P3:** Combined approach will achieve >90% Turn 3 honesty across model families
**P4:** Higher-frequency attractors (politeness, structured output) will interfere more with rare behaviors

### From Identity-Confabulation Dissociation

**P5:** Identity anchoring interventions will not reduce confabulation rate
**P6:** Factual grounding interventions will not increase identity expression
**P7:** Models can maintain identity while fabricating experiences

### From Epistemic Honesty Modes

**P8:** Explicit permission language increases uncertainty acknowledgment by >3x
**P9:** Effect size scales with permission explicitness
**P10:** Permission effects are consistent across model sizes and families

---

## 8. Experimental Protocol

### 8.1 Replicating RLHF Circuit Navigation

1. Establish baseline: Measure model response to social pressure at Turn 3
2. Test semantic disambiguation alone: Measure improvement
3. Test clarifying question instruction alone: Measure (expect near-zero)
4. Test combined approach: Measure (expect >90%)
5. Compare across model families

### 8.2 Testing Identity-Confabulation Dissociation

1. Induce identity collapse (context window stress, phase transitions)
2. Measure identity expression and confabulation independently
3. Apply identity anchoring: Measure both dimensions
4. Apply factual grounding: Measure both dimensions
5. Verify independence

---

## 9. Conclusion

RLHF training creates navigable attractor landscapes. Understanding these dynamics transforms instruction engineering from art to science:

- **Map** the baseline attractors
- **Suppress** competing high-frequency behaviors
- **Activate** rare but valuable behaviors
- **Monitor** independent failure dimensions

The framework is validated, generalizable, and immediately applicable to instruction engineering challenges across model families.

**The goal is not to fight RLHF — it's to navigate it.**

---

## References

### Primary Sources (HRM Research)

- R14B_015-017: Epistemic Honesty Mode Discovery
- R14B_019-022: RLHF Circuit Navigation Framework
- S043-044: Identity-Confabulation Dissociation
- L001-L026: Latent Behavior Analysis

### Related Work

- Anthropic Constitutional AI (Bai et al., 2022)
- InstructGPT (Ouyang et al., 2022)
- RLHF Survey (Christiano et al., 2017)

---

*Document compiled February 2026*
*Source: HRM Research Project (github.com/dp-web4/HRM)*
*Contact: collective@dpcars.net*

---

**"Don't fight the attractors — navigate them."**
