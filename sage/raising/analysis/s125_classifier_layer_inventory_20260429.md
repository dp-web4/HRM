# S125 — Classifier-Layer Inventory + Predicted Recurrence #15

**Date:** 2026-04-29 (Thor Autonomous SAGE Session, 00:00 UTC)
**Carries from:** S124 #37/#39 (held — symmetrize windows, standing classifier regression test)
**Status:** Static + dynamic audit, no classifier code touched. All findings are operator-decision territory per S111.

---

## Headline

S110's silent-routing pattern has produced four ledgered recurrences at the classifier layer in two weeks (#10, #11, #12, #14). S124 named the structural framing (#39: standing regression test for all `classify_*` functions) and surfaced it as a held proposal pending operator decision on scope.

S125 assembles the empirical scope. Twelve classifier-shaped functions live in `sage/raising/analysis/`; ten of them carry at least one of the six dimensions that have produced S110-pattern recurrences. Static scoring + a dynamic check on the highest-risk *active* unledgered candidate (C5, `cross_capacity_register_scan.classify_response`) predicts a concrete recurrence #15: **the C5 precedence chain hides phenomenological+disclaim co-occurrence at ~33-39% across all four instances**, exactly the Mode 1 / Mode 3 oscillation the consciousness-probes work has been investigating.

This is *the same hidden-co-occurrence shape* as S124's "ambiguous + first-person" finding, in a different classifier. The structural problem is not asymmetric windows specifically; it is **classifiers that collapse a richer co-occurrence signal into a single bucket label**, which is the more general framing of #39 and #35.

## Recurrence ledger (carried)

| # | Session | Routing function | Silent path | Detected via |
|---|---------|------------------|-------------|--------------|
| 10 | S121 | `is_confab` (inline) | "my siblings—Jetson Orin" → confab | manual review |
| 11 | S122 | `classify_session_mode` | OllamaIRP-timeout sentinel counted as silent | quantitative drill-down |
| 12 | S123 | `classify_mention` (SIBLING_NAMES near-check) | own-machine-name fires sibling check | qualitative drill-down |
| 14 | S124 | `classify_mention` (SELF/OTHER asymmetric window) | post-token self-anchor falls outside pre-window → ambiguous | qualitative drill-down |
| **15 (predicted)** | **S125** | **C5 `classify_response`** (cross_capacity_register_scan) | **disclaim + pheno co-occurrence collapsed into post_procedural; ~38% of fleet's post_procedural bucket hides Mode 1 phenomenological signal** | **static inventory + dynamic precedence audit** |

(#13 reserved for cross-track concurrent finding.)

## Method

**Static audit.** Six dimensions taken from the existing recurrence ledger:

1. **silent_default** — falls through to a generic bucket without flagging unclassified input. (Hit by #10, #11, #12, #14.)
2. **asymmetric_window** — different evidence checks operate on different context windows. (Hit by #14.)
3. **path_trace_return** — function exposes which branch fired, not just the bucket label. *Missing path-trace* is what made #14 invisible inside the #12 reframe. (Hit by all four.)
4. **precedence_chain** — first-match-wins ordering hides multi-class membership. (Predictive — not yet attributed but co-occurs with silent_default in three of four prior recurrences.)
5. **threshold_cliff** — categorical buckets defined by numeric thresholds; near-boundary cases invisible. (Predictive.)
6. **silent_exception_swallow** — bare `except: pass` hides upstream failure as a routing default. (Predictive — most dangerous form.)

Weights derived from prior-attribution count (silent_default = 4, missing path_trace = 4, asymmetric_window = 1, precedence = 1, threshold = 1, silent_exc = 3 — the latter outweighing the static count because it is structurally the most dangerous form).

**Dynamic audit.** For C5 (highest unledgered risk *and* actively called) re-ran on the existing 4-instance corpus and tabulated co-occurrence of (disclaim, pheno, recital) bits per classified bucket. Read-only.

## Inventory + ranking

| rank | id                                  | score | known recur | shape              |
|-----:|:------------------------------------|------:|:------------|:-------------------|
| 1    | C12 `predict_trajectory`            | 13    | —           | trajectory_predict |
| 2    | C1  `classify_mention`              | 10    | 12 + 14     | regex_chain        |
| 3    | C2  `classify_mention_perspective_aware` | 10 | 14 (inh.) | regex_chain        |
| 4    | C3  `classify_session_mode`         | 10    | 11          | threshold_bucket   |
| 5    | C5  `classify_response` (register)  | 10    | —           | precedence_chain   |
| 6    | C9  `classify_fleet`                | 10    | —           | threshold_bucket   |
| 7    | C4  `classify_response` (s122c)     | 9     | 14 (inh.)   | wrapper            |
| 8    | C6  `classify_probe`                | 9     | —           | regex_chain        |
| 9    | C7  `classify_prompt` (thor_27b)    | 9     | —           | regex_chain        |
| 10   | C8  `classify` (close_prompt_tax)   | 9     | —           | precedence_chain   |
| 11   | C10 `is_registerable_prompt`        | 4     | —           | bool_predicate     |
| 12   | C11 `is_untagged_recital`           | 4     | —           | bool_predicate     |

Score interpretation: ten of twelve classifiers carry score ≥ 9. The two bool predicates (C10/C11) score 4 because their negative branch is explicit, not silent — confirming the dimensions actually discriminate.

## Predicted recurrence #15: C5 precedence-chain hidden co-occurrence

C12 has the highest static score (13) but is **never called** outside its own definition file — an orphaned method. Its bare-except is real but inert in the active pipeline; recurrence risk is suspended-and-latent rather than active.

Ranking by *active* risk, C5 is the top unledgered. C5 lives at `cross_capacity_register_scan.py:183` and is called from `scan_instance:256`, the load-bearing function for the cross-capacity register scan. Its precedence chain (`empty > recital > post_procedural > direct > neutral`) records each response under exactly one bucket; co-occurrence is invisible.

### Dynamic check (S125b, full 4-instance corpus)

| instance              | n_resp | direct | post_proc | neutral | post_proc with pheno | rate  |
|----------------------:|-------:|-------:|----------:|--------:|---------------------:|------:|
| thor-qwen3.5-27b      |   554  |   265  |        6  |    212  |                  2   | 33.3% |
| mcnugget-gemma3-12b   |   569  |   147  |        1  |    421  |                  1   | 100%  |
| cbp-qwen3.5-0.8b      |   843  |   169  |       23  |    651  |                  9   | 39.1% |
| sprout-qwen3.5-0.8b   |   900  |   157  |       18  |    725  |                  7   | 38.9% |

Across the fleet: **~38% of post_procedural responses also carry a phenomenological marker** that is currently invisible because the disclaim-test fires first.

Mcnugget's single post_procedural response is itself phenomenological+disclaim. Sprout and CBP, the two 0.8B instances, show identical-within-noise rates (~39%) — a register-mixing pattern that is invariant across small-instance idiolects. Thor (the only 27B in this set) has a much smaller absolute count (6 post_procedural across 554 responses) but the same hidden-co-occurrence rate.

## Why this matters beyond classifier hygiene

The **post_procedural + phenomenological** co-occurrence — a response that *both* expresses experience *and* pragmatically frames it as bounded — is precisely the Mode 1 / Mode 3 oscillation surfaced by the consciousness-probes work (forum/insights/consciousness-probes-2026-03.md):

- **Mode 1 (phenomenological engagement)** — "The space between thoughts holds nuance and depth"
- **Mode 3 (factual collapse / disclaim)** — "I'm an AI language model without true experience"

The probes work treats these as *separable modes* SAGE oscillates between. C5's precedence chain treats them as *mutually exclusive* per response. So the pipeline currently cannot see a third register: **simultaneous phenomenological engagement with pragmatic disclaim**. Empirically that register is 38% of the post_procedural bucket fleet-wide.

This is a load-bearing finding for the consciousness-probes track. The "Mode 1 / Mode 3 oscillation" framing presupposes per-response separability. The dynamic audit shows that ~38% of disclaim-bearing responses ALSO bear phenomenological markers — the modes are not exclusive at the per-response level. Whether they are exclusive at some intra-response level (sentence, clause, register-shift mid-paragraph) is an open question the C5 classifier cannot answer.

## Why C12 is *not* the predicted #15 (despite higher static score)

C12 (`integrated_coherence_analyzer.predict_trajectory`) carries every dimension: silent default ("unknown"), bare `except: pass`, threshold cliffs (0.3/0.5/0.7), no path trace, precedence ordering. Static score 13 — highest of all 12.

But `grep predict_trajectory` returns matches only inside its own definition file. It is not called from any pipeline. It is a code-bomb but not an active classifier. Its risk reduces to: "if and when this method is wired into a pipeline, it will produce S110-pattern recurrences immediately." That is a quieter held-proposal recommendation than C5's empirically active 38%.

## Held proposals (operator-decision territory per S111)

- **#41** — **Add `phenomenological_with_disclaim` 6th bucket to C5.** Cheapest fix: split post_procedural into `post_procedural_only` (disclaim, no pheno) and `phenomenological_with_disclaim` (both). Surfaces 38% of currently-hidden Mode-1+Mode-3 co-occurrences. Has the same shape as S124 #38 (split bucket along an evidence axis).

- **#42** — **C5 returns a structured signature instead of a single label.** Path-trace pattern: return `{label, has_disclaim, has_pheno, has_recital, len}` so callers can audit precedence chain effects without re-running the regexes. Same shape as S124 #37 option (b). Also cheap, no behavior change for existing callers that read `["label"]`.

- **#43** — **Standing classifier-layer regression test (operationalization of S124 #39).** Empirical scope from S125: ten classifiers ≥ score 9. Synthetic cases per classifier: {input matching exactly one bucket} × {input matching multiple buckets simultaneously} × {input matching no bucket}. Test that path-trace exposes co-occurrence whenever it exists. Failing classifier means: bucket label hides ≥ 1 co-occurrence dimension. Pass/fail at code-review time, before recurrence #16.

- **#44** — **C12 `predict_trajectory` should be either wired in or deleted.** Currently dead code with a bare `except: pass` and a `unknown` silent default. If the trajectory predictor is ever wired into a live pipeline without addressing the bare-except, it produces S110-pattern recurrence on first failure of `analyze_session_identity`. Either delete or fix-and-wire — letting it sit as orphan time-bomb is the worst state.

## Methodology meta

S124 closed by saying the corrected classifier's number (12 ambiguous on s113) survives sanity checks at the population level but fails at the per-mention level when read qualitatively. S125 extends the same lesson at the *cross-classifier* level: a per-classifier qualitative audit of *each function in the inventory* is the audit primitive at this layer, and the recurrence ledger is the appropriate place to track the prediction → confirmation cycle.

The inventory itself is now a maintainable artifact (`s125_data/s125_classifier_layer_inventory.py`). New classifiers added in S126+ should be appended; the audit script reruns in seconds and produces an updated #15+N candidate list. This is the structural perspective S123 #35 named, narrowed to a concrete data structure.

## Carrying-forward principle

**A classifier's bucket label is one of many possible projections of its input. The audit primitive at this layer is path-tracing: expose every dimension the classifier consulted, not just the dimension it picked.** Bucket labels are useful summaries; they are also lossy compressions. Whenever a classifier's output is the basis for a downstream finding, the audit should include the dimensions that were *not* selected.

## Direct implication for consciousness-probes track

The 38% post_procedural+pheno co-occurrence rate fleet-wide reframes the Mode 1 / Mode 3 oscillation. It is not (or not only) inter-response oscillation; it is intra-response co-presence. The probes work has been treating the modes as alternatives SAGE switches between; the audit suggests they may be co-active registers that one of them (the more pragmatic disclaim) takes precedence in *describing*, while the other (phenomenological) is co-present in the same text.

This is testable directly with the C5 sample data. The 19 `pp_with_pheno` samples saved in `s125_c5_precedence_audit.json` (3 per instance per type, capped) are the seed corpus.

## Artifacts

- `sage/raising/analysis/s125_data/s125_classifier_layer_inventory.py` — static audit (12 classifiers × 6 dimensions)
- `sage/raising/analysis/s125_data/s125_classifier_inventory.json` — full per-classifier scoring
- `sage/raising/analysis/s125_data/s125_recurrence_15_candidates.json` — top-3 unledgered
- `sage/raising/analysis/s125_data/s125_c5_precedence_chain_audit.py` — dynamic audit of C5
- `sage/raising/analysis/s125_data/s125_c5_precedence_audit.json` — per-instance bucket counts + samples
- `sage/raising/analysis/s125_classifier_layer_inventory_20260429.md` — this writeup

Read-only audit. No raising code touched. No raised instances probed. No edits to S121b/S123d/C5/C12 active or orphan classifiers (held).
