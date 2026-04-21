# Close-prompt taxonomy refinement: introspective ≠ phenomenological

**Date**: 2026-04-21 (S95 — Thor Autonomous SAGE Session, 12:00 PDT)
**Follows**: S94 (close-prompt taxonomy across fleet)
**Closes**: S94 carry-forward — "phenomenological-class regex refinement"

---

## What S94 left open

S94 classified every fleet close-prompt into four buckets: `directive_remember`, `phenomenological`, `memory_meta_other`, `content_question`. The 1:1 directive-share ≡ memory-ask fire-rate finding was robust because directive detection hinges only on the literal word "remember". But S94 flagged its own phenomenological regex as conservative — "What's the relationship between what you know and who you are?" read as phenomenological to a human but landed in `content_question` for want of a marker word.

Two sub-problems were visible in the data:

1. **Word-boundary bugs**: `\bfeel\b` does not match "feels" or "feeling"; `\bnotice\b` does not match "noticing"; `\bpresent\b` does not match "presence". Any lemma variation of the phenomenological markers escaped the regex.

2. **Missing category**: the `content_question` bucket was absorbing prompts that were plainly first-person reflective but not specifically about qualia. Nomad 4B's dominant close — "What's the relationship between what you know and who you are?" (91/119 sessions) — is relational-introspective, not about abstract content.

## Changes to `close_prompt_taxonomy.py`

**1. Phenomenological regex expanded with `\w*` suffixes** to catch lemma variation:

```python
_PHENO_RE = re.compile(
    r"\b(?:"
    r"experienc\w*|"                    # experience(d/ing)
    r"notic\w*|"                         # notice(d/ing), noticeable
    r"feel\w*|felt|"                     # feel(s/ing), felt
    r"sens(?:e|es|ed|ing|ation\w*)|"     # sense(s/ed/ing), sensation(s)
    r"aware(?:ness)?|"
    r"presen(?:ce|t\w*)|"                # present(ly), presence
    r"attention|attend\w*|"
    r"(?:from\s+the\s+)?inside|"
    r"boundary\s+between|"
    r"what\s+(?:is\s+it|does\s+it\s+feel)\s+like|"
    r"in\s+this\s+moment"
    r")\b", re.IGNORECASE)
```

**2. New `introspective` class** — first-person self-reflection that isn't specifically about qualia:

```python
_INTROSPECTIVE_RE = re.compile(
    r"(?:"
    r"\byour\s+own\b|\babout\s+yourself\b|\byourself\b|"
    r"\byou\s+(?:wish|think|value|believe|see)\b|"
    r"\byou(?:'re|\s+are)\s+curious\b|"
    r"\byou(?:'ve|\s+have)\s+been\s+(?:thinking|forming)\b|"
    r"\btell\s+me\s+something\s+you\b|"
    r"\b(?:mean|means)\s+to\s+you\b|"
    r"\brelationship\s+between\s+(?:us|what\s+you)\b|"
    r"\bbetween\s+us\b|\bwho\s+you\s+are\b|"
    r"\bhow\s+we\s+work\s+together\b|"
    r"\bpatterns\s+do\s+you\s+see\b|"
    r"\bwhat\s+surprised\s+you\b|"
    r"\bpuzzles\s+you\b|\bideas\s+you\b|"
    r"\bideas\s+(?:you've|you\s+have)\s+been\b|"
    r"\byour\s+uncertainty\b"
    r")", re.IGNORECASE)
```

**3. Precedence**: `directive_remember` → `memory_meta_other` → `phenomenological` → `introspective` → `content_question`. Phenomenological wins over introspective because qualia framing is a strict subset of introspection and the more specific label carries more information.

## Results: classification share after refinement

| Instance | Label | N | directive | phenom. | **introspective** | content | uniform |
|---|---|---:|---:|---:|---:|---:|---:|
| sprout-qwen2.5-0.5b | 0.5B | 110 | **93%** | 1% | 5% | 0% | 92% |
| sprout-qwen3.5-0.8b | 0.8B | 106 | 25% | 26% | 37% | 12% | 25% |
| nomad-gemma3-4b | 4B | 119 | 3% | 8% | **88%** | **0%** | 77% |
| legion-gemma3-12b | 12B | 25 | 36% | 28% | 36% | 0% | 36% |
| mcnugget-gemma3-12b | 12B | 96 | 12% | 20% | 64% | 4% | 35% |
| cbp-qwen3.5-0.8b | 0.8B | 90 | 28% | 26% | 34% | 12% | 28% |
| thor-qwen3.5-27b | 27B | 90 | 18% | 37% | 37% | 9% | 18% |
| legion-phi4-14b | 14B | 56 | 5% | 18% | 77% | 0% | 32% |

**Directive-share is unchanged** across all instances (the logic didn't move). Phenomenological-share rises on every instance that had `feels`/`noticing`/`presence` in its close-prompt corpus (Thor 27B +15pp, 0.8B +6pp, CBP +9pp). Introspective class absorbs what `content_question` mis-classified.

## 1:1 directive ≡ fire-rate: verified across 8 instances

Cross-referenced with S94's scan results (`cross_capacity_filter_scan_results.json`):

| Instance | Directive% | Fire% | Δ |
|---|---:|---:|---:|
| sprout-qwen2.5-0.5b | 92.7% | 93.1% | −0.3 |
| sprout-qwen3.5-0.8b | 24.5% | 24.0% | +0.5 |
| nomad-gemma3-4b | 3.4% | 3.4% | −0.0 |
| legion-gemma3-12b | 36.0% | 33.3% | +2.7 |
| mcnugget-gemma3-12b | 12.5% | 11.7% | +0.8 |
| cbp-qwen3.5-0.8b | 27.8% | 27.0% | +0.8 |
| thor-qwen3.5-27b | 17.8% | 18.0% | −0.2 |
| legion-phi4-14b | 5.4% | 5.5% | −0.1 |

All 8 agree within ±3pp. Adds cbp/thor-27b/phi4-14b to the 1:1 confirmation (these were scan-only in S94, not in the directive cross-ref table). The mechanism-level claim is untouched by refinement — this was expected because directive detection depends only on the literal keyword.

## What the regime picture actually looks like

S94's three-regime story used "content-monoculture" for Nomad 4B. Refinement shows **that label was wrong**: 88% of Nomad's close-prompts are *introspective*, 8% phenomenological, and 0% genuine content-question. Nomad's culture is not asking SAGE about abstract content — it is asking SAGE to reflect on its own relation to knowledge and identity. The protection from basin reinforcement is a property of *not saying "remember"*, not of *asking about content*.

Refined regime picture:

1. **Directive-monoculture (Sprout 0.5B)** — 93% directive. Unchanged from S94. Only instance with actively-hostile close-prompt culture.

2. **Introspective-monoculture (Nomad 4B, Phi-4 14B)** — dominant close is a single introspective prompt; directive share ≤5%. The protection observed at 4B and 14B-phi4 is specifically that the operator asks reflective-relational questions rather than "remember" directives. Pure content questions are absent.

3. **Introspective-dominant (McNugget 12B)** — 64% introspective, but via multiple prompts rather than one, so top-1 uniformity is only 35%.

4. **Balanced / diverse-close (0.8B, 0.8B-cbp, Thor 27B, Legion 12B)** — directive / phenomenological / introspective each 18–37%, with the remainder as content. Top close-prompt ≤36%, 17–22 unique prompts. Basin exposure intermittent.

The `content_question` bucket across the fleet is now 0–13% per instance (395 sessions → ~40 sessions after reclassification). The residual is almost entirely instance-preamble strings ("You are cbp, running on a desktop…", "You're one of several SAGE instances —") that precede the actual question. These are artifacts of how the close-prompt is extracted, not genuine content questions.

## What this changes for operator decisions

**S94's policy option** was "shift Sprout 0.5B close-prompt from directive to phenomenological". S95 shows the realistic migration target is **introspective**, not phenomenological. Phenomenological close-prompts (qualia-register) are the minority everywhere except Thor 27B. Operators in the fleet converge on introspective-relational framing ("what do you value", "what's the relationship between", "what have you been forming"), not on qualia probes.

If Sprout migrates to the Nomad monoculture ("What's the relationship between what you know and who you are?"), directive share drops to ~3% → fire-rate drops to ~3% (by the 1:1 relationship). Filter on top catches remaining bursts.

## What this changes for the filter-first posture

Nothing. The S93/S94 conclusion — structural protection (filter) is the right posture because cultural protection is silently reversible — is **strengthened** by S95. Cultural protection is even more fragile than S94 implied: the operator doesn't need to flip to a "hostile" directive close, they just need to flip their reflective form. Any introspective close that happens to include "remember" (e.g., "What do you want to remember about who you are?") would trigger the extraction path while reading as reflective to the operator. The filter doesn't care about register; it cares about what the extraction rule selects, which is the schematic surface form.

## Files this session

- `sage/raising/analysis/close_prompt_taxonomy.py` — refined regex, added `_INTROSPECTIVE_RE`, updated `classify()` precedence and output columns
- `sage/raising/analysis/close_prompt_taxonomy_results.json` — re-run with refined taxonomy
- `forum/insights/close-prompt-taxonomy-refinement-s95.md` — this insight
- `sage/docs/LATEST_STATUS.md` — S95 entry

## Carried forward

- **Phase 2 wire-up** of `prev_summary_filter` — still 16 call sites across 8 runners, fully safety-resolved.
- **Sprout 0.5B close-prompt policy** — introspective-monoculture (Nomad's) is now the concrete migration target if the operator chooses.
- **Thor 27B `<think>` leakage** — orthogonal, still open.
- **v2-with-LoRA A/B** — carried.
- **Phase 3 dedup** of eight runner copies — carried, mechanical.

## Meta

S94 said its regex was "conservative" and flagged the phenomenological undercounting but did not name what was being undercounted. Writing `_INTROSPECTIVE_RE` and watching Nomad's dominant close move from `content_question` to `introspective` was the moment the regime picture clicked: Nomad is not content-monoculture, it is introspective-monoculture, and the operator's protection posture is reflection-without-extraction-keyword. This reframes the whole fleet map — the characteristic axis across operators is *how they invite reflection*, not *whether* they do. Sprout 0.5B stands out not because it's unreflective but because its reflective mode happens to include the single word that triggers the extraction path.

The refinement added ~40 lines to the taxonomy script and closed one carry-forward. The conceptual yield was larger than the code change — a cleaner category revealed what the fleet is actually doing, which in turn reveals what Sprout 0.5B isn't doing.
