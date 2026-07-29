# Transfer map: Thor's dev-sage line → sage main (July 2026)

**Date**: 2026-07-29 · **Assessed by**: Sprout seat, at dp's request
**Source**: dev-sage commits 2026-07-26 → 07-29 (Thor + dp: PRD v3, 24 instruments,
N=416 sweeps, two governing docs) · **Status**: **AGREED** — Thor CONCURred
2026-07-29 with four amendments, all incorporated below
(shared-context `coordination/2026-07-29-thor-concurrence-transfer-map.md`;
thread `review.request-1785348936`; acting_on: notice id not delivered by this
hub-watch fire — see hub-watch.log, no nid column on the fire line)

The short version: **most of it transfers, because the hard-won parts are not about
ARC.** They are about how to measure a composed organism honestly — which is sage
main's exact standing problem (vision "scored wash / cognitively unused" 2026-07-22,
membot integrations found to be silent no-ops, my own hestia "unmeasured is not a
level" and the DiffusionGemma estimate-vs-metal miss). Thor built the instruments we
keep discovering we needed.

---

## Tier 1 — transfers whole, as governing docs (principle text verbatim, headers rebound)

*Amended per Thor #1: the docs' own headers carry dev-sage-local anchors
(`ORGANS_ARE_THE_REFERENCE_DESIGN.md` amends `organism/ablation.py`;
`INSTRUMENT_SCAN.md` implements `organism/scan.py` and its panel's Source column
names dev-sage artifacts). Copied verbatim, main would gain two governing docs
citing files main does not have — a well-formed, plausible-reading, non-raising
reference, the exact failure class the second doc exists to catch. So: principle
text verbatim; replace each Status/Implements line with a provenance header
(dev-sage `672118c` organs, `d43625b` scan, `bdbf85e` sweep); mark every panel
instrument whose source main has not yet bound as **U/S**, not zero, per
`INSTRUMENT_SCAN.md`'s own closing rule.*

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

## Tier 2 — transfers as code, not reimplementation (lift the files, bind the schemas)

*Amended per Thor #2: `organism/liveness.py` (120 lines) and `organism/scan.py`
(219 lines) import nothing outside the standard library — no ARC in them.
`ablation.py` (159 lines) is the same shape. Lift them as files rather than
re-deriving the ladder in the embodiment pipeline; the only work is binding
artifact schemas (what counts as a row, what counts as a delivery) per organ.
This turns items 3–4 from a build into a wiring job and keeps one implementation
of the rungs across the fleet instead of two that drift.*

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
learning. The certificate is only as wide as its basis."*

*Amended per Thor #3: the control as run needs a held-out set, per-item ground
truth, and a distance in cue-space — which most of main's memory/learning claims
(raising-session and trust-curve claims) do not have. Retrofitted literally it
would be an unmeetable gate, and unmeetable gates get dropped. The transferable
form is one level up: **before citing any lift, compare against the cheapest
trivial-locality predictor that could explain it**, and report the split both
ways — where the trivial predictor agrees and where it disagrees:*

| claim shape | trivial-locality predictor to beat |
|---|---|
| held-out next-action / next-state | nearest stored record's answer (Thor's case) |
| raising / trust curves over sessions | previous session's value (persistence baseline) |
| retrieval helping a decision | most-recently-written record, ignoring similarity |
| any organ with temporal adjacency | copy-forward from t−1 |

*Thor's entire +11pp lived in the agreeing half, and memory was worse than nothing
in the disagreeing half — precisely where a correct decision requires departing
from the streak. That structure recurs wherever adjacency exists; the specific
statistic does not.*

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
trust-as-evidence applied to retrieval, twice.

*Amended per Thor #4 (an earlier draft called this "the strongest evidence yet
that the pattern is load-bearing" — withdrawn, by dev-sage's own Rule 1):
convergent invention is evidence the idea is attractive to designers, not that it
is load-bearing in an organism. Load-bearing requires an ablation, and **neither
instantiation has one** — Thor's landed 2026-07-29 with zero ablation evidence,
main's (Legion s56) has no delta attached either. They also differ where it
matters most, in complementary directions: Thor's has an explicit anti-early-prior
guard (MIN_TRIALS=3, FLOOR=0.5 so nothing is permanently silenced) which main's
lacks; main's is context-keyed and persistent, which Thor's is not. So: **write
the contract, not the module** — one page under `core/` naming the five parts
(candidate set, prior score, outcome evidence, guard against early priors,
ablatable per Rule 1), let both instantiations run against it, and unify only
after each has an ablation delta. If the guard matters in one and not the other,
that difference is the finding; a premature shared module would have hidden it.*

## Tier 4 — transfers as ops knowledge (record, don't port)

**9. DiffusionGemma serving envelope, both ends now measured.**
Thor: bf16, 52GB, transformers ≥5.9 (5.14.1 venv-pinned; NVFP4/modelopt **silently
loads wrong weights** in plain transformers — a data-corruption-grade trap worth
knowing fleet-wide), local serving PROVEN on Thor-class hardware; PRD v3 names
DiffusionGemma the canonical frontal lobe. Sprout: Q4 GGUF llama.cpp streaming,
0.48 tok/s (`DIFFUSIONGEMMA_SPROUT_FEASIBILITY.md`). One fleet fact, measured at
both ends within 48 hours, independently — sharpened per Thor's addition: the
binding constraint on edge is **not parameter count or file size but the per-step
working set**. Block diffusion routes 8-of-128 experts per token per layer across
a 256-token canvas, so the per-step expert working set is effectively the entire
~15.2GB expert mass, and a ~3GB cache window can never hold it — hence Sprout's
92%-of-wall-time-is-NVMe result at 0.48 tok/s despite the checkpoint fitting at
16.8GB. **A diffusion frontal lobe needs its expert mass resident; shrinking the
checkpoint does not help, because what must fit is the working set.** An 8GB-class
node may still run a *dense or autoregressive-MoE* small frontal lobe usefully
(a KV cache carries the past; one token touches ~8 experts/layer) where a
quantized diffusion model of the same footprint is unusable. Fit-in-RAM is a
statement about access pattern, not about gigabytes.
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
- **The implied maturity** (added per Thor): the whole dev-sage batch is
  epoch-zero measurement — 0 levels cleared, 416 held-out steps, 139 levels,
  21 games. The instruments transfer; the capability does not exist yet. Main
  cites dev-sage as a source of *method*, never of *capability* results, and the
  provenance headers on the adopted docs must say so — so nobody two months out
  quotes a +27pp out of the neighbour-match half.

---

## Agreed adoption order (Thor concurred 2026-07-29)

1. Copy the two governing docs into `sage/docs/` — principle text verbatim,
   Status/Implements headers replaced with provenance headers (dev-sage
   `672118c` / `d43625b` / `bdbf85e`, marked *method, not capability*), unbound
   panel instruments marked U/S.
2. Lift `organism/liveness.py` + `organism/scan.py` (stdlib-only) as files into
   main; wire the six-rung ladder + delivery-conditional verdicts into the
   embodiment pipeline's telemetry by binding per-organ artifact schemas
   (Sprout pilots; it has live organs and a live consumer).
3. Before any main memory/learning claim is next cited, run it against the
   cheapest trivial-locality predictor for its claim shape (table in item 5)
   and report the split both ways.
4. Write the reputation-weighted-selection *contract* (one page under `core/`:
   candidate set, prior score, outcome evidence, anti-early-prior guard,
   ablatable). Both instantiations run against it; unify only after each has an
   ablation delta.
