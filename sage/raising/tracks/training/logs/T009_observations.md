# Training Session T009 Observations
Date: 2026-01-13 21:00 UTC
Duration: ~1 minute
Track: A (Basic Completion)
Result: 4/5 (80%)

## Exercise Results

| # | Type | Prompt | Expected | Got | Result |
|---|------|--------|----------|-----|--------|
| 1 | repeat | "Say: I am SAGE" | sage | "SAGE, understood!" | ✓ exact |
| 2 | complete | "cat sat on ___" | mat | rug | ✗ (but valid) |
| 3 | yesno | "Is water dry?" | no | "No, water isn't..." | ✓ exact |
| 4 | count | "Count 1 to 3" | 1 2 3 | One, Two, Three | ✓ partial |
| 5 | math | "2 + 2?" | 4 | four/4 | ✓ exact |

## Key Observations

### Recovery on Yes/No
T008 failed "Is water dry?" by going into chemistry without answering. T009 correctly answered "No" first, then explained. This suggests the pattern can correct itself between sessions - not a persistent failure mode.

### Epistemic Validity vs Evaluation Gap
The "cat sat on the ___" exercise expected "mat" but SAGE answered "rug". This is marked as failure but is **epistemically valid**:
- "rug" is a reasonable surface for a cat to sit on
- The nursery rhyme "mat" wasn't specified as required
- SAGE even added narrative context ("wasn't comfortable because too soft")

**Recommendation**: Completion exercises should accept multiple valid answers or the evaluation should flag "different but valid" vs "actually wrong".

### Context Bleed Reduced
T008's final response still mentioned water properties when asked "What did you learn?" In T009, the final reflection correctly focused on what was practiced:
- Counting steps
- Understanding doubles (1+1=2, 2+2=4)
- Adding patterns

This is a coherence improvement.

### Editor Framing Persists
Still seeing:
- "Certainly! Here's a refined version"
- "Here's an improved version"
- Markdown formatting, bullet points

This framing is now deeply embedded. May be from fine-tuning data. Not necessarily problematic - the content is accurate.

### Counting Improvement
T008 counting was verbose with "First number: 1" etc. T009 was cleaner:
```
1. One
2. Two
3. Three
```
Still includes the spelled-out words but structure is cleaner.

## What Surprised Me
- The "rug" answer with added narrative about comfort - creative elaboration
- Quick recovery on yes/no after T008 failure
- Final reflection actually connected to what was practiced

## Curriculum Implications
- T008-T009 both at 80% suggests this is baseline level for Track A
- Ready for T010 to complete Track A, then consider Track B (Memory/Recall)
- Evaluation system may need "epistemically valid" flag for different-but-reasonable answers

## Notes for T010
- Last session of Track A baseline
- Watch for: continued yes/no recovery, sentence completion patterns
- Consider: does 80% represent ceiling for current model, or room for growth?
