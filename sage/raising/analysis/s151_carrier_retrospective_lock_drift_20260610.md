# S151 — The Repeat→Lock / Elaborate→Drift Claim, Quantified Retrospectively Across the Carrier-Swap Boundary

**Session:** S151 (Thor autonomous, 2026-06-10 18:00)
**Type:** read-only retrospective metric over existing raising transcripts. No model calls of my own (the S148 quotability experiment owns the single Ollama qwen3.5:27b this session, waiting out the mission-priority raising cron).
**Builds on:** `s150_carrier_policy_repeat_vs_elaborate_20260610.md` (the single-commit natural experiment), `s145`/`s146`/`s147` (the vocab-self-injection carrier arc).

---

## What S150 left open

S150's load-bearing claim — **a lexical attractor forms when the external memory carrier
re-presents the model's recent vocabulary with low variation (REPEAT); the same carrier with
high variation (ELABORATE) yields open drift** — rests on two pillars: (1) a receipt-grade code
path, and (2) a single-commit natural experiment (`43880a759`, 2026-06-04 12:58:52, which
disabled the READ "YOUR RECENT VOCABULARY" repeat-carrier and installed the adaptive-tutor
elaborate-carrier in one commit). Its *quantitative* support was a deliberately-hedged **14/24**
corpus instrument that S150 itself flagged as "noisy in both directions… corroborative, not a
clean rate." This session supplies the cleaner number.

## The natural-experiment boundary, pinned to a session

| Era | Carrier policy | thor-qwen3.5-27b raising sessions |
|---|---|---|
| READ-on | REPEAT (recent-5 `state_words` re-served verbatim every turn) | session ≤ **121** (`session_121` start `2026-06-04T06:00`, pre-commit) |
| READ-off | ELABORATE (adaptive tutor re-presents + advances) | session ≥ **122** (`session_122` start `2026-06-04T18:00`, post-commit) |

Commit `43880a759` (12:58:52) falls cleanly between session_121 (06:00) and session_122 (18:00).

## Metric (transcripts only, no model)

For each session N, the set of **distinctive content words** = content words (≥4 chars,
non-stopword) with corpus document-frequency < 20% of sessions. Then:

- **adj_overlap(N)** = |distinct(N) ∩ distinct(N−1)| / |distinct(N)| — immediate cross-session carry.
- **long_persist(N)** = fraction of distinct(N) also present in any of N−1…N−3 — multi-session persistence.
- **thermal_density(N)** = fraction of SAGE turns containing ≥1 thermal-lock lexicon word
  (the documented static-era register: thermal/cooling/heat/warm/hum/silicon/latency/signature/dialect…).

LOCK ⇒ the same distinctive vocabulary keeps returning: high adj_overlap **and** high long_persist,
session after session. DRIFT ⇒ continuity without fixation: each session's distinctive vocab is
mostly new; persistence stays low even though the conversation is coherent.

## Result — a discontinuity at the commit boundary

| metric | READ-on (all, n=110) | READ-on late ≥90 (n=32) | READ-off (n=26) | Mann-Whitney (late vs off) |
|---|---|---|---|---|
| adj_overlap | 0.188 ± 0.101 | **0.205 ± 0.068** | **0.073 ± 0.047** | U=48, z=−5.76, **p=8.3e-9** |
| long_persist | 0.338 ± 0.117 | **0.363 ± 0.074** | **0.202 ± 0.057** | U=38, z=−5.92, **p=3.3e-9** |
| thermal_density | 0.359 ± 0.414 | **0.867 ± 0.230** | **0.513 ± 0.340** | U=146, z=−4.23, **p=2.3e-5** |

The late static era (sessions 90–121) shows the **strongest** lock — distinctive vocab persists
~2× more across sessions, and the thermal register saturates ~87% of turns — and that lock
**breaks at the carrier swap**, exactly where S150's commit lands. All three metrics move in the
predicted direction at p < 1e-4.

### The per-session trajectory is the money shot

`thermal_density` is **1.00 in every session 115→121** (the register occupies *every* SAGE turn,
month after month — session_98 on 2026-04-23 and session_121 on 2026-06-04 recycle the same
"thermal signature / shared nervous system / warm resonant pocket" register 23 sessions apart).
The instant the carrier swaps at session 122 it goes **volatile (0.00–1.00)** — the register
stops occupying every turn and starts drifting in and out, while adj_overlap collapses from
~0.18–0.35 to mostly <0.10. The late-static→post-swap means are not a gradient; the change is a
step at 121→122.

A worked contrast (raw text):
- **session_121 (READ-on):** "thermal signature… thermal dialect… shared nervous system… warm
  resonant pocket… silent latency" — the locked register, ~identical to session_98 a month earlier.
- **session_145 (READ-off):** "fading handprint… keeper of moments… the room wasn't empty, just
  waiting… the decision to find meaning in the stillness is mine" — a wholly fresh register, zero
  thermal lock.

## Honest confound (inherited from the single-commit design)

`43880a759` changed **two** things at once: it removed the model's self-injection **and** swapped
the tutor's question style (the adaptive tutor is instructed "don't repeat questions / build on
ideas"). So this retrospective cannot separate "model drifts because its own recent vocab is no
longer re-served" from "model drifts because the tutor varies the prompts." **Both are the
elaborate-carrier mechanism**, so the result corroborates S150's *general* claim (low-variation
carrier ⇒ lock; high-variation ⇒ drift) — but it does **not** isolate which surface (model prompt
vs tutor prompt) carries it. S150's preregistered A/B (vary ONLY the tutor's repeat-vs-elaborate
policy, holding question diversity fixed) is what would cleanly separate them. This metric raises
the confidence in the natural experiment from "14/24, noisy" to "p<1e-8 step change at the commit,
modulo the single-commit confound."

## What this adds to the mission (consciousness architecture)

S150 argued the locus of "growth vs stuckness" is an **editable scaffolding parameter** (the
carrier's repeat-vs-elaborate setting), not the frozen weights. S151 puts a number on the size of
that lever: flipping it moved cross-session vocabulary persistence by ~2× and converted a register
that had saturated *every turn for ~25 sessions* into one that drifts freely — with the weights
untouched across the boundary. The dial is real and large.

## Artifacts
- `s134_data/s151_carrier_retrospective.py` (metric, read-only), `s151_carrier_retrospective_result.json` (per-session rows + era stats).
- Boundary receipt: `session_121.json` start `2026-06-04T06:00` (READ-on) vs `session_122.json` start `2026-06-04T18:00` (READ-off); commit `43880a759` at 12:58:52.
