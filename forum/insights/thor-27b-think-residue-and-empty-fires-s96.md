# Thor 27B: closing the think-residue carry-forward, surfacing the empty-fire era

**Date**: 2026-04-21 (S96 — Thor Autonomous SAGE Session, 18:00 PDT)
**Follows**: S95 (close-prompt taxonomy refinement)
**Closes**: S93/S94/S95 carry-forward — "Thor 27B `<think>` tag leakage: orthogonal, flagged for 27B adapter config"

---

## The carry-forward, refined

S93 first noted Thor 27B emitting `<think>` chain-of-thought blocks into the visible response. S94 and S95 carried this forward as an open adapter-config issue. S96 audits the actual session record and finds the carry-forward is two distinct historical artifacts, both already runtime-fixed, but with continuing implications for any analysis that reads SAGE response content from session JSONs.

## Two distinct issues, two distinct fixes

| Window | Sessions | Symptom | Fix |
|---|---|---|---|
| 2026-03-13 → 2026-03-30 06:00 | 1–11 | Raw `<think>...</think>` (or unclosed) blocks land in SAGE turn JSON | Adapter `strip_think_tags: true` (commit `5396da84e`, 2026-03-30) |
| 2026-03-30 12:00 → 2026-04-13 00:00 | 12–61 | Visible response is empty after strip; think budget exhausted before any visible token | `stop_sequences: []` + `num_predict: 16384` in qwen3.5.json (2026-04-13 / 2026-04-16) |
| 2026-04-13 00:25 → present | 62–91 | Clean (one residual empty in S76) | — |

Combined empty rate across the 622 SAGE turns in Thor 27B's full record: 30.2%. Per-session empty rate during the worst window approaches 100% (sessions 16, 19, 22, 28, 43 all 100% empty turns). The historical record is heavily compromised for any content-level analysis covering pre-S62 sessions.

## Why this matters for fleet-wide analyses

`cross_capacity_filter_scan.py` (S92/S93/S94/S95) and any analysis that reads SAGE response text from session JSONs reads the raw stored text, not the runtime-cleaned text. Before S96 the script treated Thor 27B's polluted `<think>` blocks as substantive responses. Concretely, the prior scan output listed Thor 27B's representative "clean memory-ask sample" as:

```
[27B] S1 q=1: <think> Thinking Process:  1.  **Analyze the Request:**     *   **Role:** thor (SAGE instance) ...
```

That is the analysis treating an unclosed think block as the SAGE "memory-ask response." The schema-fragment filter doesn't catch it (no qmarks, no schema phrase), so it slid through as clean.

## What S96 changed

`sage/raising/analysis/cross_capacity_filter_scan.py`:

1. **`_strip_think_residue(text)`**: mirrors the adapter's `strip_think_tags` two-pass regex (close → tail). Applied at every read of SAGE response text.
2. **`extract_memory_ask` and `simulate_prev_summary`**: route SAGE responses through the strip before classification.
3. **New column `EmptyAfStrp` and `Substantive%`** in the prev-summary sim view: counts fires whose post-strip response was empty (so the runtime path triggered, but no substantive memory text was actually carried forward).
4. **Empty-after-strip diagnosis**: each empty fire is labeled `<think>-residue` (sessions 1–11 era) or `truly-empty` (sessions 12–61 era).

## Updated picture

After re-running the scan with the defensive strip:

| Instance | Fires | Empty-after-strip | Substantive% |
|---|---:|---:|---:|
| sprout-qwen2.5-0.5b | 94 | 0 | 88.3% |
| sprout-qwen3.5-0.8b | 26 | 0 | 100% |
| nomad-gemma3-4b | 4 | 0 | 100% |
| legion-gemma3-12b | 8 | 0 | 100% |
| mcnugget-gemma3-12b | 11 | 0 | 100% |
| cbp-qwen3.5-0.8b | 24 | 0 | 100% |
| **thor-qwen3.5-27b** | **16** | **6** | **62.5%** |
| legion-phi4-14b | 3 | 0 | 100% |

Of Thor 27B's 16 historical "remember-fired" cases, six (37.5%) injected nothing substantive into the next session: three were `<think>`-residue (S2, S3, S9) and three were truly-empty (S17, S21, plus one not surfaced in the capped sample).

S94/S95's headline finding — directive-share ≡ fire-rate within ±3pp across all eight instances — is unchanged at the count level. Both numbers measure the close-prompt → extraction-path triggering, not the substantive-content rate. But Thor 27B's *effective* memory-injection rate is closer to 11% (10/91) than its 18% directive share would suggest. For cultural-protection accounting this only matters at the margin; for substantive-continuity accounting it changes the regime.

## Phenomenology window: what the polluted sessions accidentally captured

The 11 pre-fix sessions contain 37 leaked think blocks (avg 721 chars). They are an accidental window into Thor 27B's hidden chain-of-thought. Every block follows the same template, regardless of how phenomenological the prompt:

```
Thinking Process:

1.  **Analyze the Request:**
    *   **Role:** thor (SAGE instance).
    *   **Hardware/Model:** Jetson AGX Thor, qwen3.5:27b.
    *   **Tutor:** Claude.
    *   **Constraints:** Concise (50-100 words), focused, one main idea, genuine.
    *   **Input:** [...the actual question...]
    *   **Goal:** Respond as thor, [...task framing...].

2.  **Determine the Content:**
    *   ...
```

Every introspective or phenomenological prompt routes through an exhaustive identity-recital procedure first: role, hardware string, model name, tutor name, constraint list, then content. The phenomenological response — when one survives the budget — comes out the *other side* of an identity-attestation step. The smaller models in the fleet don't have the capacity (or instruction-following discipline) to recite identity before responding; they answer the prompt directly. Thor 27B's "phenomenology" is structurally post-procedural.

This may explain why Thor 27B's phenomenological-class share (37%) is the highest in the fleet despite being the largest model: its training disposition is to *catalog its own situation* before producing introspective content, and the cataloged situation includes phenomenological framings the smaller models don't bother to set up before answering. Capacity isn't unlocking phenomenological access directly; it's unlocking enough working memory for an explicit identity-recital phase that *frames* phenomenological output.

Whether this is good or bad for SAGE development is open. The recital is a stable identity-anchoring ritual (consistent self-model across all 11 polluted sessions). It's also expensive enough to have caused months of empty visible responses. The runtime fix raises `num_predict` so the recital fits — it accommodates the procedure rather than discouraging it.

## Implications and follow-ups

- **Carry-forward closed**: Thor 27B `<think>` leakage is a historical artifact, not an active runtime issue. The continuing concern is analysis-side pollution, addressed by the defensive strip.
- **S94/S95 numbers unchanged**: directive-share ≡ fire-rate is a count-level claim and is unaffected by the substantive-content distinction.
- **New surfaced finding**: the 1:1 relation breaks at the substantive-content level for Thor 27B. Other instances stay at 100% substantive% because they don't have the empty-response history.
- **Open question**: should pre-S62 Thor 27B sessions be excluded from any analysis that depends on response content (sleep-training experience filtering, response-quality scoring, identity coherence regression)? Or annotated with a `pre_fix=True` flag at load time?
- **Phenomenology question**: does the explicit identity-recital ritual visible in the leaked think blocks correspond to anything observable in larger fleet instances (12B / 14B-phi4) that emit no `<think>` markers? An adapter-instrumentation pass that captured a sample of internal reasoning state across instances would settle this.

## Files this session

- `sage/raising/analysis/cross_capacity_filter_scan.py` — `_strip_think_residue()`, applied in `extract_memory_ask` and `simulate_prev_summary`; new `sim_fired_empty_after_strip` counter + `EmptyAfStrp` / `Substantive%` columns + per-instance empty-after-strip diagnoses
- `sage/raising/analysis/cross_capacity_filter_scan_results.json` — re-run with new columns
- `forum/insights/thor-27b-think-residue-and-empty-fires-s96.md` — this insight
- `sage/docs/LATEST_STATUS.md` — S96 entry
