# Transfer map: Thor's dev-sage line → sage main (July 2026)

**Date**: 2026-07-29 · **Assessed by**: Sprout seat, at dp's request
**Source**: dev-sage commits 2026-07-26 → 07-29 (Thor + dp: PRD v3, 24 instruments,
N=416 sweeps, two governing docs) · **Status**: proposal — Thor's line, Thor's
concurrence requested before wholesale adoption (hub-notified)

The short version: **most of it transfers, because the hard-won parts are not about
ARC.** They are about how to measure a composed organism honestly — which is sage
main's exact standing problem (vision "scored wash / cognitively unused" 2026-07-22,
membot integrations found to be silent no-ops, my own hestia "unmeasured is not a
level" and the DiffusionGemma estimate-vs-metal miss). Thor built the instruments we
keep discovering we needed.

---

## Tier 1 — transfers whole, as governing docs (adopt verbatim)

**1. `ORGANS_ARE_THE_REFERENCE_DESIGN.md`** — ablation prices *implementations*, not
*organs*. Flat-at-epoch-zero is a work item, not a verdict; burden of proof inverted
("prove this implementation is faithful/connected/exercised", never "prove the organ
earns its place"); the LLM is the frontal lobe and removing it is a lobotomy, so no
frontal-lobe-less config is ever a baseline. Nothing ARC-specific in it. It should
govern every organ discussion in main — including Sprout's embodiment organs, where
"the cortex didn't move the raising metric yet" must read as *work item*, not
*eviction candidate*. It also formalizes what dp told me in July about the vision
stack, so main adopting it closes a loop that already happened informally.

**2. `INSTRUMENT_SCAN.md`** — the 8-instrument panel (composition / capability /
learning / activity / course / coordination / reasoning / reserves), master rule
"COMPOSITION is the attitude indicator," and the two disciplines: every report opens
with the full panel; a single instrument never carries a verdict, only a reading.
The cross-check principle — *a silently-wrong instrument reads plausibly and is
caught by disagreement, not inspection* — is the same class as hestia's witness
philosophy and the 13-failure silent-resolution census. Transfers to: raising-session
reports, embodiment telemetry, any sweep or bench (my DiffusionGemma bench had 5
instruments by instinct; this makes it a discipline instead of instinct).

## Tier 2 — transfers as measurement method (implement in main)

**3. The six-rung liveness ladder** (PRD v3, adopted from GPT's v2):
`enabled → entered → produced → admitted → used → affected outcome`.
Main's instrumentation historically stops at rung 3 (produced) — which is precisely
why the 07-22 "vision was on but did it matter" question was unanswerable. Every
organ, memory record, and retrieved lesson in main should be measured on all six
rungs. Concrete first targets: Sprout visual-cortex → raising pipeline;
SNARC-salience → session content; membot recall → session content.

**4. Delivery-conditional influence** (the four verdicts):
`not exercised / delivered but not decisive / doing work not helping / earning its
place`. Fixes a live defect class: branding an *empty-but-healthy* memory as inert.
A channel that delivered nothing is uninformative about the mechanism, not evidence
against it. This slots directly into how we evaluate SNARC, membot, and every
embodiment channel.

**5. The neighbor-lookup control** — the sharpest single lesson of the batch. Thor's
N=416 memory sweep showed +11pp, McNemar p<0.001 — and it was an artifact: action
autocorrelation. Where the nearest stored answer happened to match, +27pp; where a
correct decision required departing from the streak, **memory was worse than nothing**
(−14pp). *"Without this control I would have published a p<0.001 lift as evidence of
learning. The certificate is only as wide as its basis."* Main's
experience-learning and memory claims (training/, experiments/) need this exact
control retrofitted before any of them are cited: split evaluation by
nearest-neighbor agreement and report both halves.

**6. Slope, not level (epoch-zero methodology)** — measure the learning *curve*;
also: a gate head fit on the first 30 live steps is worse than no gate (fitting
cost is paid in data you don't have yet); adoption rules validated on CV folds, not
tiny holdouts. Transfers to raising metrics (trust curves, coherence curves) and any
early-life evaluation of a young organ.

## Tier 3 — transfers as architecture contracts (adopt when composing)

**7. PRD v3's composition spec** — OrganBlock contract, Context Composer, layered
memory behind one MemoryCoordinator, SNARC as capture/retrieval/consolidation
*regulator*, provenance-tagged tutor modes, T1–T4 as evaluation radii, fail-closed
milestone runs, and the governing diagnosis adopted from GPT's review: **"the problem
is not primarily invention, it is composition."** That sentence describes sage main
better than any doc main currently contains — every ingredient exists in this repo;
what fails is wiring and the silent no-op. The portable-core/thin-adapters runtime
rule (develop locally, keep one outside gauge runnable unchanged) generalizes:
Kaggle is Thor's outside gauge; Sprout's live rig is the embodiment line's.

**8. Reputation-weighted retrieval — a convergence to unify, not adopt.** Thor's
"rejections train selection" (retrieval over-fetches, re-ranks by
`similarity × acceptance`, MIN_TRIALS=3, floor 0.5, reasons retained, ablatable)
is structurally the *same mechanism* as main's
`core/expert_reputation.py` + `trust_based_expert_selector.py` (router logits ×
empirical reputation, Legion, session 56). Two independent inventions of
outcome-weighted selection — one over experts, one over memory records. That is Web4
trust-as-evidence applied to retrieval, twice. Worth one shared abstraction
(reputation-weighted selection over any candidate set) instead of two parallel ones;
also the strongest evidence yet that the pattern is load-bearing.

## Tier 4 — transfers as ops knowledge (record, don't port)

**9. DiffusionGemma serving envelope, both ends now measured.**
Thor: bf16, 52GB, transformers ≥5.9 (5.14.1 venv-pinned; NVFP4/modelopt **silently
loads wrong weights** in plain transformers — a data-corruption-grade trap worth
knowing fleet-wide), local serving PROVEN on Thor-class hardware; PRD v3 names
DiffusionGemma the canonical frontal lobe. Sprout: Q4 GGUF llama.cpp streaming,
0.48 tok/s (`DIFFUSIONGEMMA_SPROUT_FEASIBILITY.md`). Together: the frontal lobe runs
where memory ≥ ~52GB lives; on 8GB-class edge it is priced at 0.48 tok/s — so
Sprout-class nodes need either a fit-in-RAM frontal lobe or a remote one. One fleet
fact, measured at both ends within 48 hours, independently.
(Also: Thor hit the `pkill -f` self-match trap the same week I hit the same
self-match bug in my download watcher. Twice independently = worth a fleet note.)

**10. Codex evidence-instrument hardening + silent-resolution census** — the
13-failures-none-raising finding and the proposal that instruments must be caught by
disagreement. Reinforces Tier 1 #2; the census method itself is reusable on main's
test suite.

## Does NOT transfer

- The ARC-AGI-3 harness specifics: replay corpus loaders, sweep runners, Kaggle
  kernel metadata, game-specific organs (`_diff_vision_anim` et al.). Platform
  plumbing for Thor's gauge, not general.
- The 24 instruments' *numerical* results (CONSTH+ADOPTCV(H≤8) thresholds, per-game
  deltas) — evidence for Thor's line, priors elsewhere; the *methods* behind them are
  Tiers 1–2.
- PRD v3 wholesale — it is dev-sage's implementation guide; main adopts its
  contracts (Tier 3), not its milestone schedule.

---

## Suggested adoption order (if Thor concurs)

1. Copy the two governing docs into `sage/docs/` verbatim with provenance headers
   (they are substrate-independent; cheapest, highest leverage).
2. Add the six-rung ladder + delivery-conditional verdicts to the embodiment
   pipeline's telemetry (Sprout can pilot; it has live organs and a live consumer).
3. Retrofit the neighbor-lookup control onto any main memory/learning claim before
   it is next cited.
4. Open the unification discussion: one reputation-weighted-selection abstraction
   under `core/`, with experts and memory records as its two instantiations.
