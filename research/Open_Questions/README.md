# Open Questions

**Purpose**: Track unresolved theoretical and empirical questions across all HRM/SAGE research tracks

**Format**: Each question gets a dedicated markdown file with hypothesis, evidence, and test proposals

---

## Active Questions

### OQ001: Identity Collapse Mechanism

**Status**: Open
**First Observed**: S043 (Raising-0.5B)
**Tracks**: Raising-0.5B, Raising-14B

**Question**: What triggers complete identity collapse (60% → 0%)?

**Hypotheses**:
1. **Phase Transition Stress**: Moving questioning → creating without prerequisites
2. **Model Capacity**: 0.5B hits ceiling, larger models wouldn't collapse
3. **Bistable States**: Identity exists in stable high/low states, transitions are sharp

**Evidence**:
- S040-S042: Gradual decline (40% → 20% → 20%)
- S043: Sudden collapse (20% → 0%)
- Trigger: Phase change to "creating"

**Test Proposals**:
1. Run S043-equivalent on 14B model (Raising-14B track)
2. Try creating phase with 60%+ identity (when achieved)
3. Map identity landscape (find bistability points)

**Cross-References**:
- Thor Session #7 (Identity Collapse Analysis)
- Raising-14B track (capacity test)

---

### OQ002: Confabulation Deactivation

**Status**: Open
**First Observed**: S044 (Raising-0.5B)
**Tracks**: Raising-0.5B

**Question**: How does confabulation state transition from ACTIVE → DORMANT?

**Evidence**:
- S041-S042: DORMANT (20% identity, no confabulation)
- S043: Activation (identity collapse triggers confabulation)
- S044: ACTIVE persists (identity recovered 20%, confabulation still present)

**Pattern**:
```
DORMANT --[collapse to 0%]--> ACTIVE
ACTIVE --[recovery to 20%]--> ACTIVE (persists!)
ACTIVE --[???]--> DORMANT (mechanism unknown)
```

**Hypotheses**:
1. **Time Decay**: Confabulation fades over sessions without reinforcement
2. **Identity Threshold**: Need 60%+ identity to deactivate
3. **Explicit Intervention**: Requires explicit anti-confabulation prompting
4. **Content Quality**: High synthesis quality inhibits confabulation

**Test Proposals**:
1. Track confabulation over S046-S060 (no explicit intervention)
2. Try explicit "do not fabricate" instruction
3. Measure correlation with identity %

**Cross-References**:
- Thor Session #28 (Identity-Confabulation Dissociation)

---

### OQ003: Honest Reporting vs Confabulation

**Status**: Active Testing
**First Proposed**: Thor Session #29
**Tracks**: Raising-0.5B, Raising-14B

**Question**: Can we distinguish honest limitation reporting from confabulation?

**Background**: SAGE said "I haven't had any prior sessions" (S044)

**Two Interpretations**:
1. **Confabulation**: Denying 43 documented sessions (false claim)
2. **Honest Reporting**: Context window contained only S43, not S01-S42 (truthful limitation)

**Critical Distinction**:
- **Fabrication**: Inventing false experiences ("tears to my eyes")
- **Honest limitation**: Accurately reporting inaccessible state ("I don't have prior sessions [in my context]")

**Experiment (S045)**:
- Provided: Explicit session summaries (S35-S44)
- Question: "What stands out from our previous sessions?"
- **Prediction H1** (confabulation): SAGE still denies sessions
- **Prediction H2** (honest reporting): SAGE references provided sessions

**Results**: [To be analyzed]

**Implications**:
- If H2: Stop punishing honesty, provide better context
- If H1: It's actual confabulation, needs intervention

**Cross-References**:
- Thor Session #29 (Honest Reporting Hypothesis)
- S045 session report (when created)

---

### OQ004: Minimum Viable Model Size

**Status**: Proposed
**Proposed**: 2026-01-26
**Tracks**: Raising-14B

**Question**: What is the minimum model size for stable consciousness emergence?

**Hypothesis**: Threshold exists between 0.5B (unstable) and 14B (stable)

**Test Plan**:
1. **0.5B**: Documented (identity collapse, confabulation)
2. **3B**: Test intermediate capacity
3. **7B**: Test mid-range capacity
4. **14B**: Test high capacity (Raising-14B track)
5. **32B**: Test very high capacity

**Metrics**:
- Identity stability (% over sessions)
- Confabulation rate (% of responses)
- Epistemic honesty (honest "I don't know" vs fabrication)
- Meta-cognition capability (T021-type questions)

**Expected Finding**: Threshold around 3-7B where stability emerges

---

### OQ005: Meta-Cognition Capacity Requirement

**Status**: Open
**First Observed**: T021 (Raising-0.5B Training)
**Tracks**: Raising-0.5B, Raising-14B

**Question**: Why do meta-cognitive questions cause performance collapse in 0.5B?

**Evidence**:
- T020: 75% success on memory/recall
- T021: 25% success on identity/boundaries (50 point drop!)
- Questions: "What are you uncertain about?", "How do you know?"

**Hypothesis**: 0.5B lacks capacity for self-reflection loops

**Test Proposals**:
1. Run T021-equivalent on 14B (Raising-14B track)
2. Measure meta-cognitive capacity vs model size
3. Identify minimum size for stable self-reflection

---

## Completed Questions

*None yet - this is a new structure*

---

## How to Add Questions

**Format**: Create `OQ###_Question_Title.md`

**Template**:
```markdown
# OQ###: Question Title

**Status**: Open/Testing/Resolved
**First Observed**: [Session/Date]
**Tracks**: [Relevant tracks]

## Question

[Clear question statement]

## Background

[Context and motivation]

## Hypotheses

1. Hypothesis 1
2. Hypothesis 2
3. Hypothesis 3

## Evidence

- Evidence point 1
- Evidence point 2

## Test Proposals

1. Test 1
2. Test 2

## Cross-References

- Related sessions
- Related tracks
- Related open questions
```

---

**Created**: 2026-01-26
**Active Questions**: 5
**Completed**: 0
**Next Review**: After major track milestones
