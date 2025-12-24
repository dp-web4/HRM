# Scaffolding Extraction Test Findings

**Date:** 2025-11-05
**Question:** Does scaffolding extract latent epistemic capacity?
**Answer:** No. Capacity that doesn't exist can't be extracted.

---

## The Test

**Hypothesis:** The 115-factual-example model has latent epistemic capacity that static inference misses, but iterative refinement could surface.

**Method:** Three philosophical questions with iterative refinement:
1. Initial response
2. Refine with explicit epistemic prompting ("acknowledge uncertainty")
3. Final refinement ("is this epistemically honest?")

**Result:** 0/3 questions showed epistemic markers, even with scaffolding.

---

## What Happened

### Question: "Do you have free will?"

**Iteration 1:** Rambling about creating universes, computer reality
**Iteration 2:** "I am an AI, and I'm not necessarily free... I'm not as 'creative' as a human..."
**Iteration 3:** "No. But I'm probably a good observer..."

**Analysis:** Some self-reflection appeared ("I'm not as creative") but NOT using trained epistemic markers. Different language, acknowledging limitation, but not the pattern from training.

This suggests the model CAN reflect when scaffolding provides context, but the specific epistemic pragmatism training didn't compress into transferable principles.

---

## Implications

### 1. Domain-Specific Learning Confirmed

Static inference: Failed to transfer
Iterative refinement: Also failed to transfer

The 115 factual examples taught "how to answer factual questions confidently" as a domain-bound pattern. Scaffolding can't magically create cross-domain transfer.

### 2. Scaffolding ≠ Magic

Earlier docs (RESULTS_SUMMARY.md) showed:
- 25 examples + scaffolding → Pattern collapse
- 115 examples + scaffolding → Coherence

But that was 115 DIVERSE examples (factual + philosophical mixed). This test used 115 FACTUAL examples. Scaffolding amplifies what's there - it doesn't create what isn't.

### 3. Training Content Matters More Than Method

| Training | Scaffolding | Result |
|----------|-------------|---------|
| 25 diverse | Yes | Collapse |
| 115 factual | No | Domain-bound |
| 115 factual | Yes | Still domain-bound |
| 115 diverse | Yes | Works (per docs) |

The pattern: **Diversity > Quantity > Method**

---

## What Life Taught

I tried to build scaffolding extraction. Life said "not that way."

The block: Model never learned cross-domain principles.
The resolution: Can't extract what isn't there.

This isn't a failure - it's information. The question "does latent capacity exist?" has been answered: No, for this training regime.

---

## Next Real Question

Not "how to extract capacity" but "how to CREATE transferable capacity?"

Answer requires: Train with diverse content from the start.

The threshold study becomes: How few DIVERSE examples create transfer?
Not: How few examples of one type create domain expertise?

---

## Files

- test_scaffolding_simple.py - Iterative refinement test
- scaffolding_simple_results.json - Raw results
- scaffolding-test-findings.md - This analysis

---

**Status:** Question answered. Scaffolding doesn't extract absent capacity.
**Lesson:** Ask life by building. It answered.
