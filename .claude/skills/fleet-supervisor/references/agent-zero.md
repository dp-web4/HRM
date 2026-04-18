# Agent-Zero Discipline — evaluation that doesn't fool you

Named after Dennis Palatov's article "The Curious Case of Agent Zero": a system that always outputs the modal class can score surprisingly well on a benchmark whose class distribution is skewed. An always-zero output scored ~49% on ARC-AGI-1 because ~80% of grids are zeros.

The lesson is about **evaluation**, not training. Headline metrics without defended dummies are mirrors.

## 1. Every metric needs a dummy

Rule: **no metric is allowed to appear without its modal-class dummy and the margin above it**.

Binding thresholds (from PRD §4 + §7.10):

| Gate | Threshold | Why |
|---|---|---|
| Aggregate agreement with teacher | ≥ 98% | Sanity check — model learns the teacher |
| Modal-dummy margin | ≥ 25pp aggregate + per class | Beats `always-predict-modal` meaningfully |
| Per-class F1 | ≥ 0.85 every class | No class is starved |
| Salience-weighted agreement on top-decile arousal | ≥ 95% | Accuracy when it matters most |
| Rare-decision recall | ≥ 0.80 | Doesn't collapse to modal class |
| SNARC-utility ablation delta | ≥ 5% | Model actually uses SNARC features |
| Data diversity (sources, classes, entropy, SNARC stddev) | ≥ floor | Won't compute against pathological input |

If **any** gate fails → verdict is NOT PASS.

## 2. Three-valued verdict discipline

`phase1_training.py` returns one of:

- **PASS** — all gates cleared, adapter promotable
- **INCONCLUSIVE** — some gate is below threshold for a reason that can be diagnosed (data insufficiency, collinear features, teacher structure). Valuable signal; name the reason.
- **FAIL** — worse than dummy or diversity/entropy floors breached. Something is structurally wrong.

INCONCLUSIVE is not a halfway PASS. It is a legitimate verdict that tells us the evaluation didn't commit either way. Often more useful than forced PASS — it diagnoses what to fix.

## 3. Reframe triggers (halt conditions)

Halt training and revisit scope when:

1. **All machines return INCONCLUSIVE with the same blocking reason on the same head** — the head's framing is wrong, not the data.
2. **Head passes gates on easy slice, fails on salience-weighted slice** — model learned the common case and is blind when it matters.
3. **Cross-machine disagreement > 20pp on the same aggregate data** — reproducibility failure; don't promote anything.
4. **SNARC-utility delta = 0 with high aggregate accuracy** — the model isn't using SNARC; either the features are insufficient, the target is SNARC-independent, or both.
5. **Per-class F1 = 1.0 on all populated classes** — suspicious perfection. Probably a single feature collinear with the label. Investigate.
6. **Held-out game accuracy collapses relative to seen-game** — no generalization, memorizing game-specific patterns.
7. **Data-diversity gate trips with realistic data** — the diversity floor is calibrated wrong, OR the corpus genuinely lacks what's needed.

Halt means commit a reframe row to `phase-1-convergence.jsonl` with `reframe_trigger` field populated, and ping the fleet if others would be affected.

## 4. Forbidden shortcuts

Never:

❌ Relax a gate threshold to make a run pass. If the gate is wrong, fix the gate design in the PRD; don't fudge the number.

❌ Quote aggregate accuracy without its dummy. A value without context is not data.

❌ Train until PASS — if training needs multiple restarts with different hyperparameters until one passes, you're fitting the gate, not the data.

❌ Silently drop hard cases. If records are filtered, document why and quantify the impact on coverage.

❌ Promote on single-machine PASS. Federation requires cross-machine convergence. One PASS is a data point, not a verdict.

## 5. Productive failure > safe summaries

A well-documented INCONCLUSIVE with a specific reframe trigger is more valuable than any number of "looking good" summaries.

If CBP's first Phase 1 canary returned INCONCLUSIVE because "the programmatic teacher is SNARC-blind", that finding eliminates a path and redirects the fleet. That's research output.

If the same canary had returned PASS after gate-tweaking, the fleet would train toward the wrong target and discover the problem later, at higher cost.

Celebrate productive failure. Commit it proudly.

## 6. Dummy variants — know which applies

Different dummy baselines apply to different slices:

| Slice | Right dummy |
|---|---|
| Multi-class dispatch | always-predict-modal-class |
| Binary decision | always-predict-modal-class (which may be 0 or 1 depending on prevalence) |
| Per-class F1 | per-class random with matched class proportions |
| Salience-weighted | modal-class on the salience-weighted subset (NOT the global modal) |
| Held-out game | uniform-random across possible actions (game-independent dummy) |
| Action coordinate regression (CLICK x,y) | center-of-frame prediction |

If the dummy isn't obvious for your evaluation, write it down and justify it.

## 7. SNARC ablation

The SNARC-utility gate is specifically: train two models, identical architecture and data, one with SNARC features zeroed. The delta is `Δ = acc_with_snarc - acc_without_snarc`.

If Δ ≥ 5%, SNARC is contributing. If Δ ≈ 0, the model learned the non-SNARC features and SNARC is dead weight. This is the agent-zero pattern at the feature-importance level.

SNARC-blind training signals (a teacher that doesn't consult SNARC) will always produce Δ = 0. That's the head's fault, not the router's.

## 8. Feature-level agent-zero

Beyond the overall metric, check:

- Is any single feature perfectly correlated with the label? (trivial classification)
- Does any feature have near-zero variance across the corpus? (dead feature)
- Are SNARC features varying, or are they all near their default? (idle records tend to have flat SNARC — mix sources)

`phase1_training.py` reports min SNARC-dim stddev as part of its diversity check. Low stddev = dilution. Source-filter before training if needed.

## 9. Always quote the data

Every verdict report should include:

- Total records
- Source distribution (idle, gameplay, raising, interactive)
- Decision class distribution
- SNARC quintile distribution
- Commit hash of the code used

Without these, the verdict is not reproducible and not trustable.
