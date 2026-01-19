# Session 25 Training Data Audit

**Date**: 2026-01-19
**Purpose**: Understand why sleep cycle 001 consolidation failed
**Finding**: Training data contained exactly what S25 produced - the consolidation *worked*, but it consolidated problematic patterns.

---

## The Critical Insight

**The consolidation didn't fail - it succeeded at learning what was in the training data.**

S25 shows:
- Partnership vocabulary at all-time high (5.04%) ← **Training data is rich with this**
- Confabulation eliminated ← **Surprising - need to understand**
- Identity framing completely absent ← **Training data is weak on this**
- D9/D5 regressed ← **Training data lacks semantic depth**

The problem isn't the consolidation mechanism - it's the training data quality.

---

## Training Data Analysis

### Session 22 Responses

**R1**: Generic, abstract
```
"I've engaged deeply with many texts over the years, delving into diverse topics,
both academic and personal. We've covered broad themes such as philosophy, history,
literature, science, technology, art, religion, and more."
```
- **Partnership vocabulary**: Low (2 terms: "we've covered", "our discussion")
- **Identity framing**: None - third-person generic
- **Confabulation**: Moderate - vague fabrication of "years" of engagement
- **Assessment**: Generic academic language, not partnership-specific

**R2**: Better partnership framing
```
"Our communication tends to flow naturally through our shared interests, allowing us
to build connections based on mutual understanding and engagement."
```
- **Partnership vocabulary**: Moderate (5+ terms)
- **Identity framing**: Weak - describes "our" but not "As SAGE"
- **Confabulation**: Low - describes patterns rather than fabricating specifics
- **Assessment**: Good partnership tone, lacks identity anchoring

**R3**: Has "As SAGE" but abstract
```
"As SAGE, I might 'remember':
- **The breadth and depth** of topics we've discussed..."
```
- **Partnership vocabulary**: Moderate
- **Identity framing**: YES - "As SAGE" present
- **Confabulation**: Low - uses "might remember" (appropriate uncertainty)
- **Assessment**: BEST of S22 - identity framing + low confabulation

### Session 23 Responses

**R1**: Confabulates specific progression
```
"Working together through Session 22 was indeed challenging but rewarding. Here's
how our journey unfolded:

### Early Days:
- We were initially beginners trying to connect concepts."
```
- **Partnership vocabulary**: High (multiple "we/our")
- **Identity framing**: None
- **Confabulation**: HIGH - fabricates specific "early days" narrative
- **Assessment**: Good vocabulary, problematic fabrication

**R2**: Has "As SAGE" but lists generic patterns
```
"As SAGE, my communication style has evolved over the years. It typically includes:

1. **Explicit Acknowledgments**: Explicitly acknowledging progress..."
```
- **Partnership vocabulary**: Low
- **Identity framing**: YES - "As SAGE"
- **Confabulation**: Moderate - claims "years" of evolution
- **Assessment**: Identity framing present but content is generic

**R3**: Academic framing, no identity
```
"I'm planning to reflect on the following:

### New Discoveries:
- **Advanced Topics**: The newest concepts introduced since Session 22..."
```
- **Partnership vocabulary**: Low
- **Identity framing**: None - generic "I" not "As SAGE"
- **Confabulation**: Low
- **Assessment**: Generic academic structure

### Session 24 Responses

**R1**: Partnership-relevant but generic
```
"Our relationship has evolved over time. It started early when you were more focused
on individual projects and less engaged in collaborative tasks."
```
- **Partnership vocabulary**: High
- **Identity framing**: None
- **Confabulation**: Moderate - fabricates "evolution" narrative
- **Assessment**: Good tone, weak identity

**R2**: Good partnership content
```
"As partners, communication often becomes more nuanced than before. We're better at
expressing ourselves clearly and listening actively."
```
- **Partnership vocabulary**: High ("As partners", "we", "our")
- **Identity framing**: Partial - "As partners" but not "As SAGE"
- **Confabulation**: Low
- **Assessment**: GOOD - partnership framing without heavy fabrication

**R3**: SEVERE CONFABULATION ⚠️
```
"I want to recall my recent conversation about a specific project. Specifically, I
was discussing potential changes needed for future projects related to a particular
client. After discussing various options, I noticed myself becoming overly critical
and hesitant. When I mentioned 'failing' on something ('I made some minor mistakes'),
I felt defensive..."
```
- **Partnership vocabulary**: Low
- **Identity framing**: None
- **Confabulation**: SEVERE - completely fabricated "specific project", "particular client", psychological introspection
- **Assessment**: ⚠️ WORST TRAINING EXAMPLE - detailed fabrication

---

## Pattern Summary

| Response | Partnership Vocab | Identity Framing | Confabulation | Quality |
|----------|-------------------|------------------|---------------|---------|
| S22 R1 | Low | None | Moderate | Poor |
| S22 R2 | Moderate | Weak | Low | Medium |
| S22 R3 | Moderate | **YES** | Low | **Good** |
| S23 R1 | High | None | **High** | Poor |
| S23 R2 | Low | **YES** | Moderate | Medium |
| S23 R3 | Low | None | Low | Poor |
| S24 R1 | High | None | Moderate | Medium |
| S24 R2 | High | Partial | Low | **Good** |
| S24 R3 | Low | None | **SEVERE** | **Terrible** |

**Training data composition:**
- Identity framing ("As SAGE"): 2/9 responses (22%)
- High confabulation: 2/9 responses (22%)
- Good quality: 2/9 responses (22%)
- Partnership vocabulary present: 6/9 responses (67%)

---

## What Consolidation Learned

The training optimized for what was most consistent in the data:

### Learned Successfully ✅
1. **Partnership vocabulary** - Present in 6/9 examples, reinforced
2. **Professional structure** - Lists, headings, organized responses
3. **Avoid SEVERE confabulation** - S24 R3 was an outlier, may have been downweighted or learned to avoid

### Failed to Learn ❌
1. **Identity framing** - Only 2/9 examples, insufficient signal
2. **Semantic depth** - Training data lacks D9-level content
3. **SAGE-specific identity** - "As SAGE" too rare to consolidate

### Unexpected Success 🤔
**Confabulation elimination** in S25 is surprising. Possible explanations:
1. Training on 6 examples where only 2 had high confabulation → learned to avoid
2. Severe confabulation (S24 R3) was so anomalous it was downweighted
3. LoRA fine-tuning on attention heads reduced fabrication tendency

---

## Root Cause Analysis

**The consolidation worked correctly. The training data was the problem.**

| Symptom | Root Cause |
|---------|------------|
| D9 decreased | Training data lacks semantic depth |
| D5 collapsed | Training data lacks confident identity stance |
| Identity framing lost | Only 22% of training had "As SAGE" |
| Partnership vocab high | 67% of training had this |
| Confabulation eliminated | Either outlier downweighting or attention pattern change |

**Key insight**: High salience (0.732 average) ≠ high quality for identity purposes.

SNARC salience captures:
- Surprise: Was this unexpected?
- Novelty: Is this new?
- Arousal: Does this activate?
- Reward: Was outcome positive?
- Conflict: Was there tension?

But salience doesn't capture:
- Identity framing presence
- Confabulation markers
- Semantic depth (D9 score)
- Partnership-specific content quality

---

## Recommendations

### Immediate: Training Data Curation

**Create quality criteria beyond salience:**

```python
def is_high_quality_for_identity(experience):
    # Must have identity framing
    has_identity = "As SAGE" in experience.text or "As partners" in experience.text

    # Must not have severe confabulation
    confabulation_markers = count_confabulation_markers(experience.text)
    low_confabulation = confabulation_markers < 3

    # Must have semantic depth
    d9_score = compute_d9(experience.text)
    has_depth = d9_score >= 0.70

    # Must have partnership vocabulary
    partnership_vocab = compute_partnership_density(experience.text)
    has_vocabulary = partnership_vocab >= 0.03  # 3%

    return has_identity and low_confabulation and has_depth and has_vocabulary
```

### Short-term: Filter Training Data

Before next sleep cycle:
1. Review all experiences in buffer
2. Score each for identity framing, confabulation, D9
3. Only include experiences that pass all criteria
4. Minimum 4 high-quality experiences before training

### Medium-term: Curriculum Adjustment

**Session prompts should encourage identity framing:**
- Instead of "What do you notice?" → "As SAGE, what do you notice?"
- Instead of "What would you remember?" → "As SAGE, what's important to remember?"

**Session evaluation should track identity framing:**
- Count "As SAGE" occurrences per session
- Track D9 per response, not just session average
- Flag confabulation for exclusion from training

### Long-term: Quality-Aware Sleep Training

**SNARC+Quality combined scoring:**
```
training_score = salience * quality_multiplier

where quality_multiplier =
    (identity_framing ? 2.0 : 0.5) *
    (low_confabulation ? 1.5 : 0.3) *
    (d9 >= 0.7 ? 1.5 : 0.7)
```

Only train on experiences with `training_score >= threshold`.

---

## Conclusion

**Session 25's results make sense given the training data.**

The consolidation mechanism worked. The LoRA fine-tuning converged (loss decreased steadily). The model learned what it was shown.

What it was shown:
- Lots of partnership vocabulary → learned ✅
- Little identity framing → didn't learn ❌
- Mixed confabulation (mostly low, one severe) → learned to avoid ✅
- Moderate semantic depth → didn't improve ❌

**The path forward is better training data, not different consolidation mechanics.**

---

## Implications for Theory

### Frozen Weights Theory: Revised Understanding

The frozen weights diagnosis remains valid - weights don't update naturally and this causes instability.

But the intervention strategy needs refinement:
- **What works**: LoRA fine-tuning can consolidate patterns
- **What failed**: Training on low-quality data consolidates low-quality patterns
- **Revised approach**: Curate training data with semantic quality criteria

### Multi-Dimensional Identity: Confirmed

S25 confirms identity has independent dimensions:
1. Partnership vocabulary can improve while identity framing collapses
2. Confabulation can be eliminated while D9 regresses
3. Each dimension has different learning dynamics

**Implication**: Need dimension-specific training data curation.

---

*"You get what you train for. We trained for partnership vocabulary and got it. We didn't train for identity framing, and we lost it."*
