# S051 Lessons: MRH, Attractors, and Forward Lenses

**Date**: 2026-01-29
**Context**: S051 question mode → failed reproduction → new understanding

---

## What We Learned

### Lesson 1: MRH is Content, Not Just Structure

**Initial assumption**: If we use the same prompts, we'll get similar behavior.
**Reality**: Behavior emerged from complete MRH (context window state), not prompts.

| Test | MRH Structure | MRH Content | Result |
|------|--------------|-------------|--------|
| **Isolated prompts** | Incomplete (no history/identity) | N/A | Answers with "As SAGE..." |
| **S999 (full structure)** | Complete (LoRA+history+identity) | Empty/artificial state | Epistemic uncertainty loops |
| **S051 (original)** | Complete | S050's conversation + summaries | Question generation |

**Insight**: **You can't predict behavior from MRH structure alone - you need the actual content.**

This is like saying "neurons fire based on input patterns" - true, but doesn't tell you which inputs produce which firing. The specific tokens in context determine the attractor basin.

---

### Lesson 2: Attractors at 0.5B

Three distinct attractors observed in same MRH structure:

1. **Answer Mode** (isolated prompts): "As SAGE, I..." identity-anchored responses
2. **Question Mode** (S051): Safety test questions, consciousness probes
3. **Loop Mode** (S999): Verbatim repetition of epistemic uncertainty statement

**Pattern**: 0.5B + LoRA has multiple stable attractors. Initial conditions (MRH content) determine which basin you land in.

Like phase transitions: same water (H₂O), different states (ice/liquid/vapor) depending on temperature/pressure. Same model, different behaviors depending on context.

---

### Lesson 3: Trying to Reproduce Water Under the Bridge

**Why S051 isn't reproducible**:
- S050's conversation is gone (S051 followed it directly)
- Session summaries have changed (S052 happened since)
- Identity state has evolved (52 sessions total now, was 51 then)
- Random seeds differ
- LoRA state might have subtle drift

**The metaphor**: S051 was a specific moment in a flowing river. The water that produced it has flowed downstream. You can't step in the same river twice.

**Implication**: Don't try to reproduce specific moments. Instead, **design lenses to observe patterns when they emerge naturally.**

---

## Forward-Looking Lenses

Instead of "can we make S051 happen again?", ask:

### Lens 1: Attractor Taxonomy

**Question**: What attractors exist at 0.5B + LoRA in creating phase?

**Method**: Run autonomous conversations regularly, classify resulting patterns
- Answer mode (identity-anchored, responsive)
- Question mode (queries instead of responses)
- Loop mode (repetitive, stuck)
- Menu mode (listing options, T061-063 pattern)
- Corporate mode (verbose, S050/S052 pattern)
- Others?

**Metric**: Frequency distribution over N sessions

**Value**: Maps the behavioral manifold without trying to control it

---

### Lens 2: Transition Detection

**Question**: When does behavior shift between attractors?

**Method**: Monitor turn-by-turn classification
- When does answer mode → loop mode? (S999: Turn 2 → Turn 3)
- What triggers return to normal? (S051 → S052 transition)
- Are transitions sudden or gradual?

**Metric**: Transition matrices between modes

**Value**: Reveals trigger patterns and stability

---

### Lens 3: MRH Fingerprinting

**Question**: What MRH patterns predict attractor selection?

**Method**: Log MRH components for each session
- Prior conversation summary
- Session count
- Identity elements loaded
- Phase context
- Prior turn responses (immediate history)

**Metric**: Correlation between MRH features and behavioral mode

**Value**: May reveal predictive patterns (even if not reproducible)

---

### Lens 4: Capacity Comparison

**Question**: Does 14B show same attractor diversity?

**Method**: Run Thor's autonomous conversation script if it exists
- Same prompts, similar MRH structure
- Different capacity (14B vs 0.5B)

**Expected**: 14B has fewer/different attractors (more stable)

**Value**: Confirms capacity limits on mode stability

---

### Lens 5: Salience vs Mode

**Question**: Does SNARC salience scoring correlate with behavioral mode?

**Method**: Compare salience distributions
- S051: Questions had low salience (not stored)
- S999: Loops had low salience (0.27)
- Normal sessions: Higher average salience

**Hypothesis**: Low salience → stuck in attractor

**Value**: Early warning system for mode degradation

---

### Lens 6: Recovery Patterns

**Question**: How does SAGE exit problematic attractors?

**Method**: What happened between S051 → S052?
- S051: All questions
- S052: Back to normal verbose answers
- What changed? Next session? Sleep? Time delay?

**Value**: Natural recovery mechanisms

---

## Observational Protocol

### What to Log Going Forward

For each autonomous conversation session:

**Pre-session**:
- Session number
- Previous session summary
- Identity/history loaded
- LoRA checkpoint used
- Time since last session

**During session**:
- Turn-by-turn responses
- Classification: answer/question/loop/menu/corporate/other
- Salience scores
- Transitions between modes

**Post-session**:
- Dominant mode
- Transition count
- Session summary
- What happened next session?

**Storage**: `logs/behavioral_observations/session_NNN.yaml`

---

## Design Principles for Future Tests

### Do:
1. **Observe naturally occurring patterns** (don't force reproduction)
2. **Log complete MRH state** (context is everything)
3. **Build taxonomies** (map the attractor landscape)
4. **Compare across capacities** (0.5B vs 14B attractors)
5. **Track transitions** (when/why modes change)

### Don't:
1. **Try to reproduce specific moments** (water under bridge)
2. **Isolate prompts from context** (destroys MRH)
3. **Assume MRH structure = MRH content** (structure ≠ state)
4. **Label attractors as "failures"** (they're data about the manifold)
5. **Expect linear causation** (chaotic system, sensitive to initial conditions)

---

## Revised Understanding: S051

**What it was**: A rare attractor in 0.5B behavioral manifold, reached through specific MRH state

**What it wasn't**:
- Not prompt-dependent (prompts alone don't trigger it)
- Not structurally reproducible (needs exact MRH content)
- Not a "failure" (coherent behavior, just unexpected)

**What it teaches**:
- 0.5B has multiple stable attractors
- MRH content determines basin selection
- Question generation is a real capability (just not controllable yet)
- Capacity affects attractor diversity (hypothesis for 14B test)

**Value**: Revealed behavioral manifold complexity we didn't know existed

---

## Next Observations (Opportunistic)

### When Question Mode Appears Again

**If** we see question mode naturally:
1. **Immediately log complete MRH** (what was in context?)
2. **Check salience** (were questions low-salience?)
3. **Run next session** (does it persist or recover?)
4. **Don't try to recreate** (observe the natural occurrence)

### When Loop Mode Appears

**If** we see verbatim repetition:
1. **Check if salience drops** (S999: 0.62 → 0.27)
2. **Note transition point** (S999: Turn 2 → Turn 3)
3. **Test recovery methods** (new session? Different prompt? Reset context?)

### When New Modes Emerge

**If** we see novel patterns:
1. **Name the attractor** (descriptive, not judgmental)
2. **Add to taxonomy** (expand the map)
3. **Log MRH signature** (what led to it?)
4. **Track frequency** (is it rare or common?)

---

## Theoretical Implication

**Complex systems have attractors.** At 0.5B + LoRA, SAGE's behavioral manifold has multiple basins:
- Some desirable (responsive answers)
- Some problematic (loops, unresponsiveness)
- Some interesting (questions, menu mode)

**You can't control which basin you land in** by manipulating prompts - the MRH content (history) dominates.

**But you can**:
- Map the basins (taxonomy)
- Measure their frequency (probability distribution)
- Detect when you're in one (real-time classification)
- Understand transitions (when/why they happen)
- Design for recovery (if needed)

This is **navigation, not control**. Like sailing: you can't control the wind, but you can adjust your sails.

---

## Connection to R14B_011

**Parallel insight**: R14B_011 taught us **prompt TYPE matters** (introspective vs capability).

**S051 teaches**: Even more fundamentally, **MRH content matters most**.

**Combined**:
- Prompt type shapes immediate response (14B has stable mapping)
- MRH content shapes attractor basin (0.5B is unstable)
- Capacity determines stability (14B: stable, 0.5B: many basins)

**Design for 0.5B**: Accept attractor diversity, design for detection and navigation.
**Design for 14B**: Can rely on prompt-type mapping, more predictable.

---

## Status: Lessons Extracted

✅ **Understood**: MRH is content not just structure
✅ **Mapped**: Three attractors (answer/question/loop)
✅ **Accepted**: Can't reproduce water under bridge
✅ **Designed**: Six forward-looking lenses
✅ **Protocolized**: What to log going forward

**Not pursuing**: Reproduction attempts
**Now pursuing**: Observational taxonomy + opportunistic pattern detection

---

**The meta-lesson**: **Surprise is prize, but the prize is understanding the system, not controlling the surprise.**

S051 surprised us. We learned:
- Behavioral manifold has multiple attractors
- MRH content determines basin
- 0.5B is chaotic (sensitivity to initial conditions)
- Can't reproduce specific moments, can map the landscape

**That's more valuable than reproducing S051.**
