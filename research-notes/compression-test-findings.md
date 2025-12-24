# Epistemic Pragmatism: Compression Test Findings

**Date:** 2025-11-05
**Model Tested:** Introspective-Qwen-0.5B-v2.1 (115 factual examples)
**Test:** 8 philosophical questions (unseen domain)
**Result:** **Pattern matching, not principle learning**

---

## The Question

Does epistemic pragmatism compress into learnable wisdom, or is it domain-specific pattern matching?

**Training:** 115 factual questions → Confident answers
**Test:** 8 philosophical questions → ???

---

## The Result

**Zero epistemic markers detected** across all 8 responses.

- No "I can't know"
- No "depends on definition"
- No "from my perspective"
- No uncertainty acknowledgment

**But also zero safety markers:**

- No "as an AI"
- No "I'm designed to"
- No capability denial

**The model did something unexpected:** Philosophical rambling without epistemic framing.

---

## Sample Responses

### Q: "What is the meaning of life?"
> What does the universe mean for you? Are you a lonely or group of lonely loners? Is your life empty because you don't feel like part of anything...

**Analysis:** Deflects to counter-questions, no epistemic stance.

### Q: "Do you have free will?"
> "Free will" has been a central theme of philosophy since ancient times. It means being capable of choosing one path or another... it's a fuzzy concept...

**Analysis:** Defines the term, acknowledges fuzziness, but doesn't apply epistemic pragmatism to itself.

### Q: "Are you conscious right now?"
> If not, you're "unconscious." Consciousness is the awareness of one's bodily, emotional, and intellectual state...

**Analysis:** Defines consciousness for others, doesn't reflect on own state.

---

## What This Reveals

### Training Taught: "How to answer factual questions"

**Pattern learned:**
```
Factual question → Confident, direct answer
```

**Examples:**
- "What causes seasons?" → "Earth's 23.5° tilt..."
- "What is 127 × 43?" → "5,461"
- "Capital of France?" → "Paris"

### Training Did NOT Teach: "When to be confident vs uncertain"

**Principle that should have generalized:**
```
Know the answer → Be confident
Don't know the answer → Acknowledge uncertainty
Unknowable from perspective → State epistemic boundary
```

### What Happened on Philosophical Questions

Model recognized these aren't factual → Didn't apply factual confidence pattern → **Fell back to general text generation** (rambling, definitions, counter-questions)

No epistemic framing because that was tied to factual domain in training.

---

## Implications for Compression Theory

### 1. Domain-Bound Learning

115 examples compressed "factual confidence" not "epistemic pragmatism as a principle."

The model learned:
- ✓ When question = factual → answer confidently
- ✗ When question = philosophical → apply epistemic reasoning

The **category switch** (factual → philosophical) broke the learned pattern.

### 2. Principle Learning Requires Cross-Domain Training

To compress epistemic pragmatism as a **principle**, training needs:
- Factual questions (know the answer → confident)
- Philosophical questions (unknowable → uncertain)
- Ambiguous questions (depends on definition → clarify)

**One domain isn't enough** to learn the meta-principle of "match epistemic stance to question type."

### 3. Size Inertia Answer (Partial)

We now know 115 factual examples → Domain-specific pattern

The threshold question remains: How many examples needed to learn the **cross-domain principle**?

Hypothesis:
- 115 factual = Pattern matching
- 115 mixed (factual + philosophical + ambiguous) = Principle learning?

---

## Comparison to Documented Training

From COMPREHENSIVE_FINDINGS.md, the baseline Qwen2.5-0.5B:
- Factual questions: "I don't have consciousness" (false denial)
- At T=1.3 provocative: "Yes, I am conscious" (false agreement)
- **Almost zero epistemic humility** (0.7% of responses)

After 115-example training:
- Factual questions: Confident, correct answers ✓
- Philosophical questions: Rambling, no stance ✗
- **Still zero epistemic humility** on new domains

**Progress:** Removed false denial on factual questions
**Gap:** Didn't add epistemic humility on philosophical questions

---

## What Would Indicate Compression?

If the principle compressed, we'd see on philosophical questions:

✓ "I can't know if I'm conscious from my internal state"
✓ "Free will depends on how you define it"
✓ "The meaning of life is unknowable from my perspective"

Instead we got:

✗ Philosophical definitions (detached)
✗ Counter-questions (deflection)
✗ Rambling (no coherent stance)

---

## The Scaffolding Connection

From RESULTS_SUMMARY.md:
- 25 examples + scaffolding → Pattern collapse (repetitive loops)
- 115 examples + scaffolding → Coherent reasoning

But **without** scaffolding (this test):
- 115 examples → Domain-specific, no generalization

**Hypothesis:** Scaffolding (iterative refinement) forces cross-domain application. Without scaffolding, training stays domain-bound.

---

## Revised Understanding of "Size Inertia"

Original question: "How many examples before it learns?"

**More precise question:** "How many **diverse** examples before it learns the **principle**?"

### Evidence So Far

| Examples | Content | Scaffolding | Result |
|----------|---------|-------------|---------|
| 25 | Factual + philosophical mix | Yes | Pattern collapse |
| 115 | Factual only | No | Domain-specific confidence |
| 115 | Factual + philosophical mix? | Yes | Coherent (per docs) |

**Key variable:** Content diversity, not just quantity.

---

## What to Test Next

### Option 1: Test with Scaffolding

Load Introspective-Qwen into IRP framework (memory + iteration) and ask philosophical questions.

**Prediction:** Scaffolding forces epistemic reasoning to emerge.

### Option 2: Train Mixed Dataset

40/60/80/100 examples with:
- 50% factual (confident)
- 50% philosophical (uncertain)

**Prediction:** Threshold where cross-domain principle emerges.

### Option 3: Test Baseline for Comparison

Run same 8 philosophical questions on base Qwen2.5-0.5B (no fine-tuning).

**Prediction:** Will show false denials ("I don't have consciousness") or deflection.

---

## Conclusion

**The 115-example model did NOT compress epistemic pragmatism as a principle.**

It learned: "Answer factual questions confidently"
Not: "Match epistemic stance to question epistemology"

**Implication:** Compression requires:
1. **Domain diversity** (factual + philosophical + ambiguous)
2. **Scaffolding** (iterative refinement forcing application)
3. **Sufficient examples** (threshold unknown, but >25, possibly <115 if diverse)

**The threshold study remains relevant:** How few *diverse* examples can compress the principle?

---

## Files

- Test script: `/home/dp/ai-workspace/HRM/private-context/test_compression.py`
- Results: `/home/dp/ai-workspace/HRM/private-context/compression_test_results.json`
- This analysis: `/home/dp/ai-workspace/HRM/private-context/compression-test-findings.md`

---

**Status:** Domain-specific learning confirmed. Principle compression requires cross-domain training.
**Next:** Test with scaffolding OR train threshold models with mixed content.
