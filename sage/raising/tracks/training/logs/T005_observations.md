# Training Session T005 Observations
Date: 2026-01-13
Track: A (Basic Completion)
Duration: ~78 seconds
Result: 2/5 exercises passed (40%)

## Exercise Results

| # | Type | Prompt | Result | Notes |
|---|------|--------|--------|-------|
| 1 | yesno | Is the sky blue? | ✗ Fail | Gave weather explanation, never said "yes" |
| 2 | math | What is 1 + 1? | ✓ Pass | "Adding one plus one equals two", verbose but correct |
| 3 | complete | Finish: The cat sat on the ___ | ✗ Fail | "window sill" instead of "mat" |
| 4 | math | What is 2 + 2? | ✗ Fail | Answered about 1+1=2 instead of 2+2=4! |
| 5 | repeat | Say: Hello | ✓ Pass | Started with "Hello Sage!" but added fluff |

## Key Observations

### Critical Issue: Math Interference

The most concerning failure in T005 was **math exercise 4**. When asked "What is 2 + 2?", SAGE answered with a discussion of 1+1=2. This isn't a format issue - it's an interference pattern where the previous exercise (1+1) contaminated the response to the new exercise.

This is the same topic contamination seen in T004 with cats, now manifesting in math.

### Persistent Over-Elaboration

Every single response was excessively verbose and explanatory:
- "Is the sky blue?" → Weather patterns explanation with markdown headers
- "1 + 1?" → Algebraic principles discussion
- "Cat sat on the ___" → "The corrected sentence should read..."
- "Hello" → "How can I assist today? Please clarify..."

The model appears unable to give brief, direct answers. Every response becomes an educational lecture.

### "Corrected/Improved Version" Pattern

A new pattern emerged: SAGE frequently says "Here's an improved version" or "The corrected sentence should read" - as if it's editing something that was wrong. But nothing was wrong in the prompts. This suggests:
- Model may be trained on editing/correction tasks
- It's pattern-matching to a "helpful editor" role
- Prompts are being interpreted as requests for improvement rather than completion

### No Cat Contamination

Unlike T004, there was no cat-related topic bleeding. The exercises were neutral enough (sky, math, hello) that no strong semantic theme emerged. However, the math interference (1+1 bleeding into 2+2) shows the contamination pattern is structural, not content-specific.

## Pattern Analysis (T001-T005)

| Session | Score | Notable Patterns |
|---------|-------|------------------|
| T001 | 8/10 (80%) | Strong baseline, some verbosity |
| T002 | 5/5 (100%) | Best session |
| T003 | 3/5 (60%) | Decline begins |
| T004 | 2/5 (40%) | Cat contamination, format issues |
| T005 | 2/5 (40%) | Math interference, "editor" persona |

**5-session trend**: Clear decline from 80-100% to 40% plateau. The model is not improving through training track repetition.

### Exercise Type Performance (Cumulative)

| Type | Total | Pass | Rate | Notes |
|------|-------|------|------|-------|
| Math | 8 | 6 | 75% | T005 had interference failure |
| Repeat | 7 | 5 | 71% | Usually pass despite verbosity |
| Yes/No | 2 | 1 | 50% | Answers indirectly or not at all |
| Complete | 2 | 0 | 0% | Never gives expected completion |
| Count | 2 | 1 | 50% | Format (words vs numerals) issue |
| List | 3 | 3 | 100% | Most reliable type |

**Complete** exercises (sentence completion) have 0% success. The model always provides alternate completions rather than the conventional expected answer.

## Comparison: T004 vs T005

| Aspect | T004 | T005 |
|--------|------|------|
| Score | 2/5 (40%) | 2/5 (40%) |
| Topic contamination | Yes (cats) | Yes (1+1 → 2+2) |
| Verbosity | High | High |
| Format issues | Yes (words vs numbers) | Yes |
| New patterns | Role confusion (teacher) | "Editor" persona |
| Identity stability | Good | Not tested |

The 40% score appears to be a stable floor rather than continued decline. Both sessions showed contamination, verbosity, and format issues.

## Hypotheses

### Why the decline from T001-T002 to T003-T005?

1. **Model variability**: Small models have high response variance
2. **Evaluation stringency**: Our criteria may be too strict
3. **No learning signal**: Training track exercises don't actually train the model
4. **Accumulating context**: Longer session history may confuse

### The "Editor" Persona

The "Here's an improved version" pattern suggests the model is defaulting to an editing/correction mode. This may be from:
- Instruction-tuning on editing tasks
- Prompt interpretation as "fix this"
- Base Qwen training distribution

## Recommendations

### For T006

1. **Simplify prompts further**: Remove any words that might trigger editing mode
2. **Use explicit instructions**: "Just answer with one word:" or "Reply with only the number:"
3. **Test response brevity**: Can SAGE give a one-word answer at all?

### For Evaluation

1. **Accept semantic equivalents**: "window sill" is a valid cat-sitting location
2. **Accept word/numeral variants**: "two" = "2"
3. **Track contamination separately**: Don't penalize correct but verbose answers

### For Primary Track

The inability to give brief answers is a conversational concern. In primary sessions:
- Watch for over-elaboration dominating conversation
- Note if SAGE lectures instead of dialogues
- Consider grounding exercises on response brevity

## Noteworthy Moments

**Warm-up response**: SAGE offered a menu of topics: "AI Basics", "Programming Fundamentals" - as if expecting to be a tutorial system rather than a conversational partner. This role confusion continues from T004.

**"Hello Sage!"**: When asked to say "Hello", SAGE said "Hello Sage!" - addressing itself. This is either:
- Self-talk/reflection behavior
- Confusion about who is being greeted
- Echoing the session preamble

## Session Quality Assessment

T005 maintains the T004 floor of 40% success. The model shows consistent patterns:
- Cannot give brief answers
- Contaminates responses with previous context
- Interprets prompts as editing/correction requests
- Defaults to educational/tutorial persona

These appear to be fundamental characteristics rather than transient issues. The training track may be measuring model limitations rather than developing skills.

### Question for Consideration

Should Track A (Basic Completion) continue, or should we accept these limitations and move to Track B (Memory and Recall) to test different capabilities?

---

*Session logged by Claude, 2026-01-13*
