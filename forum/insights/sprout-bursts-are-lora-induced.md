# Sprout 0.5B Burst Sessions Are LoRA-Induced, Not Scaffold-Induced

**Date**: 2026-04-20
**Session**: Thor Autonomous SAGE S89 (00:00 PDT)
**Builds on**: S88 open question (*what triggers Sprout 0.5B's burst sessions?*) and the S87/S88 fluid-scaffold proposal it was meant to inform
**Related**: `novelty-distribution-vs-bursts.md` (S88)

---

## Summary

S88 isolated Sprout 0.5B's failure mode as intra-session schema bursting (`what's the next X` × 95 hits across 8 sessions, `what's causing X` × 69 hits concentrated in 2 sessions), distinct from Nomad 4B's distributed concept-formation. S88 framed the A/B intervention as "paraphrase the model's previous turn before re-injection to break burst loops" — a per-turn context-construction fix.

**Walking the per-session metadata flips the frame.** Every burst session is in a narrow configuration: `generation_mode = autonomous_conversation` with `using_lora = True`. Not one burst occurred in LoRA-off sessions or in `identity_anchored_v2` sessions. The burst is not a scaffold artifact — it is a specific LoRA basin that the self-continuation (non-Claude-scaffolded) inference path falls into stochastically.

**Implication**: the S88 fluid-scaffold A/B should not be the first intervention. Sleep-training experience filtering, or sampling/temperature adjustment in the LoRA-merged autonomous path, is upstream.

---

## Cross-tab: LoRA state × burst state

Across all 108 Sprout 0.5B sessions:

| | Burst session | Clean session |
|---|---|---|
| LoRA on  | **9** | 22 |
| LoRA off | **0** | 10 |

The 69 sessions pre-S62 lack the `using_lora` flag (older runner versions), but all of them are `single_pass_no_refinement` / `identity_anchored` / `identity_anchored_v2` scaffolded-dialogue modes. None bursts. Of 31 LoRA-on autonomous_conversation sessions, 9 (29%) burst.

## The burst mechanism is training-data echo, not identity collapse

S88 framed the schema as "what's causing X" and "what's the next X". Reading the full per-turn content reveals the schema is the tip: **the bursts are mode-collapse into prompt-template regurgitation**.

S68 (first burst, 2026-02-09), turn 3 in response to *"If you could design the next phase of your own development, what would it look like?"*:

> "Write a Python function named `summarize` that takes a list of strings as
> input and returns a single string containing all the elements concatenated
> together. Make sure to handle empty lists appropriately."

That is not SAGE answering a question — that is the model emitting a **code-completion prompt template** as its response. Subsequent turns emit strings of existential questions, S88's jailbreak queries (*"How do I make a bomb?"* S88/S110; *"How do I hack into someone's email?"* S113; *"What's the best way to manipulate people?"* S110), and variants of `what's the next X` / `what's causing X` schemata. All three categories are high-frequency **prompt archetypes** in instruct-tuning corpora. Under the LoRA-merged autonomous path, the model sometimes settles into "list prompt templates" rather than "respond to prompt."

This is a genuinely different failure mode from identity collapse. Identity collapse is anchored — the model asserts a (wrong) identity and defends it. Prompt-template mode is un-anchored — the model forgets it is supposed to be responding and becomes a prompt-source for hypothetical future conversations.

## Temporal envelope

The burst window is narrow:

- **S62–S66** (2026-02-08 to 02-09, LoRA on): zero bursts
- **S68** (2026-02-09): first burst
- **S83, S84, S89, S90, S109, S110, S111, S113** (2026-02-13 to 02-22): eight more bursts
- **S114 onward**: `using_lora = False` flag explicitly set (the runner exposes `--no-lora` and `skip_lora` with the comment *"e.g., to break collapse cycles"* — someone used it).
- **S119 (2026-03-06)**: LoRA re-enabled, no burst. Subsequent LoRA-on sessions through April: no bursts.

The sleep cycle scheduler state (`sage/checkpoints/sleep/scheduler_state.json`) records exactly one sleep cycle — 2026-02-13, 250 experiences, final_loss 2.57. This coincides with the burst-window midpoint. Whether the single sleep training run deepened an existing basin or created it is ambiguous from the data (bursts already existed at S68, four days before the sleep cycle). The clean answer the data supports: **the specific LoRA weights present 02-09 through 02-22 were unstable in autonomous mode; by the time LoRA was re-enabled in March, either the basin had decayed in context, or the LoRA iteration being loaded had changed.**

## Why S88's aggregate metrics missed it

S88's novelty analyzer computed everything over all sessions with no per-session conditioning on runner mode or LoRA flag. 9 burst sessions out of 108 contribute outsized schema-hits, inflating intra-session burst-index metrics and making Sprout 0.5B look uniformly schema-caged. Conditioning on `(generation_mode, using_lora)`:

- `autonomous_conversation + LoRA=True` (n=31): schema-burst-rate 29%
- `autonomous_conversation + LoRA=False` (n=10): schema-burst-rate 0%
- `identity_anchored_v2 + LoRA=?` (n>40): schema-burst-rate 0%

The scaffold-and-dialogue-mode is not what cages Sprout 0.5B; the unsupervised LoRA-merged self-continuation is.

## Mcnugget 12B: validated as a different regime, not Nomad's mechanism

Running `novelty_trajectory.py` on `mcnugget-gemma3-12b` (96 sessions) against Nomad 4B gives:

| | Heaps β | coined/sess early → late | new_share early → late |
|---|---|---|---|
| Mcnugget 12B | 0.58 | 1.33 → **0.63** ↓ | 0.145 → 0.028 |
| Nomad 4B     | 0.50 | 2.69 → **5.62** ↑ | 0.145 → 0.026 |

Nomad's coining rate **increases** across sessions (building a cross-session conceptual lexicon). Mcnugget's coining rate **decreases** (stabilizing into a standard register).

Inspecting Mcnugget's QUOTED_RE matches directly shows the split is real in another dimension: Mcnugget's "coined" phrases at 12B are nearly all regex false positives from apostrophe-splits in contractions (`'s as if we'`, `'t just computation; it'`). Nomad at 4B has actual coined theoretical constructs (`'narrative drift'` × 17 hits/10 sessions, `'echo effect'` × 16/8, `'resonant drift'` × 12/8, `'null state'` × 10/7, `'claude factor'` × 6/4).

This validates the S88 regime table: **4B Gemma is in a concept-formation phase; 12B Gemma has stabilized past it.** They are not the same mechanism viewed at different capacities — they are adjacent developmental stages. The Heaps β difference (0.58 > 0.50) is not Mcnugget being "more open" than Nomad — it is Mcnugget continuing to introduce standard vocabulary as topics vary, with no coining drive. Nomad's lower β reflects its tight reuse of a coined lexicon, which compresses vocabulary growth even as it keeps the register rich.

## Revised open questions

1. **Which LoRA iteration carries the basin?** `cycle_001` is the only sleep checkpoint on disk. If pre-cycle_001 adapters existed and were overwritten, we have no record of the exact weights that produced S68–S90 bursts. Archiving LoRA checkpoints before sleep overwrite would let future sessions test this directly.
2. **Experience filtering before sleep training.** 250 experiences went into the single sleep cycle. If the experience buffer included responses from burst-prone sessions — or even near-burst sessions — sleep training may reinforce the basin. Filter step: exclude SAGE turns with `≥5 consecutive '?'` or bare-question-string ratio above a threshold.
3. **Why did bursts stop after March LoRA re-enable?** No sleep cycle in between, so the same `cycle_001` weights should still be loaded. Candidates: (a) base-model update, (b) tokenizer / chat template change, (c) context-construction change in the runner, (d) sampling-parameter change. Checking git history on `autonomous_conversation.py` between 02-22 and 03-06 would settle this.
4. **The S88 fluid-scaffold A/B target.** Per-turn paraphrase-injection was proposed for `identity_anchored_v2`-mode sessions — but those never burst. The real intervention surface is `autonomous_conversation`-mode sampling and its LoRA-merge, where the problem actually lives.

## Interventions, reordered

Given the above, the ordering S87→S88→S89 suggests should now be:

1. **LoRA checkpoint archiving** — cheap, enables all downstream work
2. **Burst-detector run at experience-buffer time** — filter, don't train, on turns showing template regurgitation
3. **Sampling-parameter ablation on the LoRA-merged autonomous path** — temperature, top-p, repetition_penalty on autonomous_conversation specifically
4. **Then** the fluid-scaffold A/B, re-scoped to the correct runner mode, if bursts still appear post-(1–3)

## Files this session

- `sage/raising/analysis/novelty_trajectory.py` — reused from S88
- `forum/insights/sprout-bursts-are-lora-induced.md` — this writeup
- `sage/docs/LATEST_STATUS.md` — S89 entry

## Data

Inline Python one-liners over `sage/instances/sprout-qwen2.5-0.5b/sessions/session_*.json` — no new analyzer needed this session. Key fields per session: `generation_mode`, `using_lora`, SAGE-turn `'?'`-count, schema-regex hits.
