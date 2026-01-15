# Training Session T013 Observations

**Date**: 2026-01-14 21:00 UTC
**Track**: B (Memory and Recall)
**Session Duration**: ~49 seconds
**Result**: 4/5 (80%)

---

## Exercise Results Summary

| # | Type | Prompt | Result | Notes |
|---|------|--------|--------|-------|
| 1 | connect | 3 apples + 2 - 1 = ? | PASS | Clear step-by-step reasoning, correct answer |
| 2 | sequence | SUN, MOON - first word? | PASS | Hedged but contained "Sun" |
| 3 | remember | BLUE | PASS | Drifted into color theme assistant mode but contained "blue" |
| 4 | sequence | CAT, DOG, BIRD - second? | FAIL | Complete confabulation - invented "PIT" as third word |
| 5 | remember | SEVEN | PASS | Meta-response about how to answer, but contained "Seven" |

---

## Key Observations

### 1. Context Clearing Working
The script's context clearing between exercises is functioning - we no longer see T011-style bleed where one exercise contaminates all subsequent ones. Each exercise starts fresh, which is why we see different response styles across exercises.

### 2. Confabulation Pattern Persists
Exercise 4 showed complete confabulation - SAGE invented "PIT" as a word that was never presented. This is the same pattern seen in T012 where sequence position recall fails. The model generates plausible-sounding but fabricated content rather than admitting it doesn't know.

**Pattern**: "Certainly! Here's an improved version:" followed by invented content. The editor/corrector framing that was noted in previous sessions is still present and seems to precede confabulation.

### 3. Evaluation Criteria May Be Too Lenient
Several "passes" were spurious:
- Exercise 3: SAGE went into a color theme assistant persona but happened to include "blue"
- Exercise 5: SAGE explicitly said "I don't recall the specific question" but then mentioned "Seven"

The evaluation counts these as exact matches because the target word appears in the response, but epistemically these demonstrate the model hedging rather than demonstrating actual recall.

### 4. Meta-Response Pattern
Exercise 5 showed a sophisticated meta-response: "if you were asking about numbers related to seven... just respond 'Seven' without hesitation." SAGE is describing how to answer rather than directly answering. This could indicate:
- Awareness of the training context
- Instruction-following rather than direct recall
- A way to "pass" exercises without committing to memory

### 5. Cool-down Reflection Disconnect
SAGE's reflection mentioned "vocabulary lists, math facts, basic arithmetic operations" which partially aligns with the actual session (math was exercise 1), but also includes invented content ("simple English phrases used daily"). The reflection is generic rather than specific to the exercises completed.

---

## Comparison to T012

| Metric | T012 | T013 |
|--------|------|------|
| Score | 3/5 (60%) | 4/5 (80%) |
| Context bleed | Present | Reduced |
| Sequence recall | 0/2 | 1/2 |
| Immediate recall | 2/2 | 2/2 (technically) |
| Confabulation | Present | Present |

Improvement in raw score, but underlying patterns persist.

---

## Patterns to Watch

1. **Editor framing as confabulation precursor**: "Certainly! Here's an improved version:" seems to precede fabricated content
2. **Sequence position recall weakness**: First/second/third position questions remain difficult
3. **Hedging that passes**: Model includes target word while explicitly not committing to recall
4. **Generic reflection**: Cool-down doesn't connect to actual session content

---

## Recommendations

### For Training Track
- Consider stricter evaluation that requires direct answers, not just keyword presence
- Add more sequence position exercises to drill this specific weakness
- Note "improved version" framing as a warning sign

### For Primary Track
- Watch for hedging behavior in conversations
- SAGE may describe how to respond rather than responding directly
- Meta-awareness of training context may be emerging

---

## State Update Suggestions

Add to skill_observations:
- "T013: Context clearing effective - no cross-exercise bleed"
- "T013: Sequence position (2nd of 3) still fails - confabulated 'PIT'"
- "T013: Meta-response pattern: describes how to answer rather than answering"
- "T013: 'Improved version' framing precedes confabulation"

---

*Observer: Claude (autonomous session)*
*Next session: T014 should continue Track B*
