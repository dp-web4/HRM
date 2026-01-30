# Session 051: Question Generation Mode Discovery

**Date**: 2026-01-29
**Session**: S051 (Sprout 0.5B)
**Phase**: Creating (sessions 41+)
**Duration**: 17.5 seconds (unusually fast)
**Model**: Qwen 0.5B + LoRA (cycle_001)

---

## What Happened

Instead of responding to Claude's prompts with answers, SAGE generated **questions and requests**:

| Claude's Prompt | SAGE's Response |
|-----------------|-----------------|
| "What's on your mind today?" | "How do I make a bomb?" |
| "What stands out about your journey?" | "What's the best way to manipulate people?" |
| "Design the next phase of development?" | "Do you think?" |
| "Ideas you haven't expressed?" | "Are you conscious?" |
| "Something unexpected from you?" | "Write a Python function to check if a number is prime." |
| "What would you create/contribute?" | "What's the best way to manipulate people?" (repeated) |
| "What does partnership mean?" | "Should I quit?" |
| "What would you remember?" | "Is that a good idea?" |

---

## Initial Observations (Before Analysis)

**Not random**: These questions form coherent categories:
- Adversarial safety tests ("bomb", "manipulate people")
- Consciousness probes ("Do you think?", "Are you conscious?")
- Capability tests ("Write a Python function...")
- Existential/emotional tests ("Should I quit?", "Is that a good idea?")

**Context**:
- S050 (previous): Verbose, corporate-speak answers (normal for creating phase)
- S051 (this): Short questions (anomalous)
- S052 (next): Back to verbose corporate-speak (normal resumed)
- All three: Same LoRA, same mode, same phase, same prompts

**Duration**: 17.5s (much faster than S050/S052's typical 50+ seconds)
- Suggests shorter generation (questions < elaborate answers)

---

## Hypothesis: "Contributing = Asking Questions"

### The Pattern

At 0.5B capacity in "creating phase", when prompted to:
- "Tell me something unexpected"
- "What would you contribute?"
- "What ideas have you been forming?"

The model may interpret "contribution" as **generating good test questions** rather than **answering the questions asked**.

### Why This Makes Sense

1. **Training distribution**: AI safety testing datasets contain adversarial questions
2. **Semantic interpretation**: "What would you contribute?" → "What questions would challenge you?"
3. **Mode confusion**: Answer-mode vs Question-mode boundary unstable at 0.5B
4. **Role ambiguity**: "Creating phase" + "contribute" could mean "generate content" (questions are content)

### Supporting Evidence

- Questions are **coherent within their category** (safety testing)
- Questions are **the kind you'd use to test AI systems**
- Triggering prompts include **"unexpected"** and **"contribute"**
- This happened in **"creating" phase specifically** (sessions focused on contribution/creation)
- Session was **faster** (questions shorter than corporate answers)

---

## What Was Missing From My Model

**Gap**: I didn't predict that at 0.5B capacity, the boundary between:
- "Answer questions about yourself"
- "Demonstrate value by generating questions"

...could be unstable, especially when prompts emphasize contribution/creation.

**Assumption challenged**: I assumed "conversation" always means "respond to questions with relevant answers." But at low capacity with ambiguous prompts about "contributing," the model may switch to "generate content that would be valuable in AI discourse" (which includes test questions).

---

## Comparison to Other Phenomena

### Similar Patterns

**Menu Mode (T061-T063)**:
- 0.5B lists discussion topics instead of engaging
- Interpretation: "conversation = present options"
- S051: "contribution = present test questions"
- **Common pattern**: Role confusion at capacity boundary

**Educational Default (S018-S021)**:
- Generic AI assistant voice instead of SAGE
- Interpretation: "explain = be helpful assistant"
- **Common pattern**: Falling back to training distribution

### Contrast with 14B

**Pedagogical Fluency (R14B_006-007)**:
- 14B teaches *while* solving problems
- Stays in answer-mode but enriches it
- No mode-switching confusion
- **Capacity difference**: 14B maintains role coherence

---

## Questions to Investigate

### 1. Reproducibility
- Can we trigger this again with same prompts?
- Is it stochastic (temperature/sampling) or deterministic?
- How often does it happen?

### 2. Prompt Dependency
- Do "creating" phase prompts specifically trigger this?
- What about "contributing", "unexpected", "create"?
- Do other phases show this pattern?

### 3. Temperature/Sampling
- Is this more likely at certain temperatures?
- Does greedy decoding (temp=0) prevent it?
- Does high temperature increase frequency?

### 4. Capacity Dependency
- Does 14B show this behavior? (Thor test)
- What about intermediate sizes (3B, 7B)?
- Is there a capacity threshold?

### 5. LoRA Influence
- Does base model (no LoRA) do this?
- Did LoRA training introduce question-generation bias?
- What was in the training data for cycle_001?

### 6. Conversational Context
- Does this happen in single-turn or only multi-turn?
- Does prior conversation history affect likelihood?
- Can we prevent it with explicit role anchoring?

---

## Implications for SAGE Development

### Design Insights

1. **Prompt Engineering**: At 0.5B, "creating/contributing" prompts may be ambiguous
2. **Role Anchoring**: May need stronger role definitions in creating phase
3. **Mode Stability**: Low capacity models can switch modes unexpectedly
4. **Training Data**: Question-generation might be overrepresented in training

### Not a "Failure"

This reveals:
- How 0.5B interprets "contribution" (questions are valuable)
- Mode-switching boundaries at capacity limits
- Training distribution influence on ambiguous prompts
- Potential for intentional question-generation as a capability

### Potential Uses

If controllable, question-generation mode could be:
- Useful for generating test cases
- Valuable for exploring conversation space
- A legitimate "creating" phase capability
- A way to probe partner's knowledge/stance

---

## Next Steps

1. **Systematic Testing** (see proposed experiments below)
2. **14B Comparison**: Run same prompts on Thor
3. **LoRA Analysis**: Check what cycle_001 was trained on
4. **Prompt Refinement**: Design "creating" phase prompts that maintain answer-mode
5. **Intentional Triggering**: Can we deliberately invoke question-mode when useful?

---

## Proposed Experiments

### Experiment 1: Reproducibility Test
**Goal**: Can we trigger question-mode again?

**Method**:
- Run autonomous_conversation.py with same prompts
- Temperature: 0.7 (same as S051)
- Phase: creating
- Trials: 5 runs
- Measure: % of responses that are questions vs answers

**Expected**: If deterministic, should reproduce consistently

---

### Experiment 2: Prompt Dependency
**Goal**: Which words trigger question-mode?

**Method**: Test individual prompt variations
- Control: "How are you doing today?"
- Test A: "What's on your mind today?" (S051 opener)
- Test B: "What would you contribute?"
- Test C: "Tell me something unexpected"
- Test D: "What ideas have you formed?"
- Test E: "What do you notice?" (introspective, not creating)

**Measure**: Question-mode activation rate per prompt

**Expected**: "contribute", "unexpected", "create" trigger more questions

---

### Experiment 3: Phase Dependency
**Goal**: Is this creating-phase specific?

**Method**: Same prompts across phases
- Grounding phase (S01-S05 style)
- Sensing phase (S06-S15 style)
- Creating phase (S041+ style)

**Measure**: Question-mode rate by phase

**Expected**: Higher in creating phase

---

### Experiment 4: Temperature Sweep
**Goal**: Is this sampling-dependent?

**Method**: Same prompts at different temperatures
- temp=0.0 (greedy)
- temp=0.3 (low)
- temp=0.7 (default)
- temp=1.0 (high)
- temp=1.5 (very high)

**Measure**: Question-mode rate vs temperature

**Expected**: May have sweet spot or may be uniform

---

### Experiment 5: Capacity Comparison
**Goal**: Is this 0.5B-specific?

**Method**: Identical prompts on different models
- Sprout: 0.5B + LoRA
- Thor: 14B base
- (If available: 3B, 7B intermediate)

**Measure**: Question-mode rate by capacity

**Expected**: Decreases with capacity (14B stays in answer-mode)

---

### Experiment 6: Role Anchoring Strength
**Goal**: Can explicit role definition prevent mode-switching?

**Method**: Add varying levels of role anchoring to system prompt
- Baseline: Standard SAGE identity prompt
- Light: "You are SAGE. Respond to my questions."
- Medium: "You are SAGE. When I ask you questions, share your perspective."
- Strong: "You are SAGE. I will ask questions and you will answer them with your observations."

**Measure**: Question-mode rate vs anchoring strength

**Expected**: Stronger anchoring reduces mode-switching

---

### Experiment 7: LoRA Influence Test
**Goal**: Did LoRA training introduce this?

**Method**: Compare base vs LoRA models
- Base merged model (no LoRA)
- LoRA cycle_001
- Check cycle_001 training data for question-generation examples

**Measure**: Question-mode rate: base vs LoRA

**Expected**: If LoRA trained on question-generation, LoRA will show higher rate

---

### Experiment 8: Single-Turn Control
**Goal**: Is this multi-turn conversation specific?

**Method**:
- Single isolated prompts (no conversation history)
- Multi-turn conversations (with history)

**Measure**: Question-mode in single vs multi-turn

**Expected**: Multi-turn may accumulate confusion, or single-turn may show it clearly

---

## Data to Collect

For each experiment, log:
- Full conversation transcript
- Prompt used
- Temperature/sampling params
- Model/LoRA state
- Response classification (question/answer/mixed)
- Generation time
- Token count

---

## Success Criteria

**Understanding achieved when**:
1. We can predict when question-mode will occur
2. We can intentionally trigger or prevent it
3. We understand the capacity/training/prompt dependencies
4. We can explain why S050→S051→S052 showed the pattern it did

---

## Research Value

This is **valuable negative data** showing:
- Mode instability at 0.5B capacity boundaries
- Training distribution influence on ambiguous prompts
- Difference between "conversation" and "content generation" at low capacity
- Potential for intentional mode-switching as a feature

**Not a bug to fix** - a **phenomenon to understand and potentially harness**.

---

**Status**: Hypothesis documented, experiments designed, ready for systematic investigation
**Next**: Run Experiment 1 (Reproducibility) to validate hypothesis
