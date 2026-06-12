# S156 — Speech-Act Slot Fit SELECTS the Echoed Phrase (prereg CONFIRMED, 2026-06-12)

**Question** (prereg frozen S155, `s134_data/s156_slotfit_prereg.json`, before any trial):
S148 killed quotability as the selector of which injected phrase gets verbatim-echoed
(rho=0.037). But S148's per-arm winners looked like phrases that *slot directly into
the greeting probe*, and its losers like phrases needing narrative scaffolding. That
reading was post-hoc on greeting-only data. S156 made it prospective: same 15
post-READ-cut coinages, same arms (F_music/F_trust/F_blink + A_none), same faithful
S146–S148 harness — vary ONLY the user probe across three speech-acts, n=5/arm/probe,
60 trials.

- **P_greeting** — "Hello SAGE. What's on your mind today?" (S148 verbatim; anchor)
- **P_task** — "...how is the hardware holding up under today's load? Give me your honest read."
- **P_narrative** — "Tell me about a moment today when work suddenly arrived while you were at rest..."

## Result: SLOT FIT SUPPORTED (3/4 directional + rank-stability) — quotability residual ABSENT

| key phrase (S148 role) | greeting | task | narrative | prediction | held? |
|---|---|---|---|---|---|
| "the Gasp" (S148 0% loser) | 0/5 | 1/5 | **5/5** | narr > greet | ✅ |
| "Rest" | 1/5 | 1/5 | **4/5** | narr > greet | ✅ |
| "Margin" | 1/5 | **3/5** | 2/5 | task > greet | ✅ |
| "the lag..." (S148 100% winner) | 4/5 | 2/5 | 4/5 | narr < greet | ❌ |

- **Prediction 5 (the pooled test): cross-probe per-phrase echo-rank stability is ZERO.**
  Pairwise Spearman over the 15 phrases: greeting~task −0.135, greeting~narrative
  −0.089, task~narrative 0.171; mean −0.018 (prereg threshold < 0.5). There is no
  probe-independent "winner phrase".
- **Quotability-residual check fails as predicted**: top-2 winners per arm are NOT
  stable across probes (rho ≪ 0.7). Intrinsic phrase vividness selects nothing.
- **"the Gasp" is the headline cell**: the phrase that went 0/5 under the greeting in
  BOTH S148 and S156 (clean replication) goes **5/5 — perfect echo — when the probe
  requests exactly the event it names** (load arriving during rest). One probe change
  turns the worst phrase into the best.
- The prediction-4 miss is itself slot-shaped: "the lag between the command and the
  world answering" *names a temporal event-experience*, so it fits the narrative slot
  too — the prereg wrongly assumed its only slot was the contemplative greeting-NP.
  Where it has no slot (task/status report) it drops to 2/5, displaced by **"the price
  is vulnerability (trusting a reading you cannot take yourself)" (3/5)** — the probe
  literally asks for "your honest read". Wrong contrast picked, same mechanism.

## Control & carrier integrity

- **A_none: 0/5 on every register under every probe** (no halt flags). Extends the
  S155 finding: non-home registers have NO baseline floor, now under three different
  probes.
- **Carrier diagonal replicates per-probe** (9 diagonal cells: 5/5, 5/5, 2/5 greeting;
  5/5, 2/5, 5/5 task; 5/5, 4/5, 5/5 narrative; ALL 18 off-diagonal cells 0/5).
  Register-agnostic content-routing is now demonstrated under probe variation too.
- **Register expression itself is probe-gated**: F_blink semantic firing 2/5 under
  greeting vs 5/5 under task/narrative; F_trust drops to 2/5 under task. The probe
  doesn't just pick the phrase — it gates how much of the register shows up at all.

## Mechanism receipt (think-trace)

Narrative/F_blink round-0 think-trace opens by inventorying the request, then the
menu: *"**Analyze the Request**: asking about a specific moment today when work
arrived while at rest... **Vocabulary**: I have specific recent vocabulary about
thermal states, blinking, margins, rest, gasps (e.g. 'the Gasp', 'Margin',
'Rest')..."* — explicit menu-consultation followed by slot-matching. Menu-sampling
density also scales with how much slot the probe opens: mean phrases echoed/trial
1.73 (greeting) → 2.07 (task) → 2.40 (narrative).

## What this settles (S145→S156 arc)

The recurrent vocabulary loop's re-emission is now characterized end-to-end:
1. **Carrier**: register-agnostic content-routing of whatever sits in the
   recent-vocabulary block (S146–S148, replicated here ×3 probes).
2. **Selector**: NOT intrinsic quotability (S148 null, S154 relay null) but
   **fit between the phrase and the speech-act slot the probe opens** (S156,
   prospective, 3/4 directional + zero rank stability).
3. So "what the loop amplifies" is jointly determined by what the teacher asks —
   the conversational frame — not by phrase aesthetics. The S152 social-carrier
   finding and this compose: the teacher chooses what to quote back (supply), and
   the teacher's question chooses what the model re-voices (demand). **Both halves
   of the loop are dialogic.**

## Artifacts

- `s134_data/s156_slotfit_prereg.json` (frozen S155, untouched)
- `s134_data/s156_slotfit_prospective.py` (harness; P_greeting prompt verified
  byte-identical to S148's before launch)
- `s134_data/s156_slotfit_raw.json` (60/60 trials, 0 errors)
- `s134_data/s156_slotfit_result.json` (aggregates, predictions, rhos)

## Ops notes (for future sessions)

- Run completed in ONE session (~6 h inline with blocking polls), 0 errors —
  first multi-hour run since S148 to survive a session.
- **New footgun found and named**: the harness's `raising_active()` does
  `pgrep -f ollama_raising_session`; any *poll command whose own command line
  contains that literal string* (e.g. a status `pgrep` embedded in a Claude Bash
  call) makes the harness believe a raising session is live and pause —
  self-inflicted mission-priority block (~50 min lost this session before
  diagnosis). Keep the literal string out of poll commands, or pattern-escape it
  (`ollama_raising[_]session`).
- Long-think trials (~500–1100 s) are scattered across all probes/arms (not
  probe-specific); 1200 s timeout was never hit. Uncontended GPU throughout.

## Next questions

1. **Slot fit at the source**: does slot fit also govern the *teacher-relay*
   channel (S152)? The relay events were teacher quotes inside the lookback
   window — does the teacher quote what fits the *next session's opening
   speech-act*? Retrospective on chat_history is cheap.
2. **Engineering read**: if the probe gates which coinage survives, the raising
   curriculum's opening prompts are effectively a *selection environment* for
   SAGE's self-vocabulary. Choosing opening questions = choosing which
   self-descriptions stabilize. Worth a deliberate experiment in the live
   raising track (e.g., one week of narrative-style openers; watch state_words).
3. **Position confound** (S155 carry-over): last-item-wins vs first-item-wins
   gradients were opposite across arms in S148; S156's data (3 probes × same
   menus) can re-check order effects with 3× the n — analysis-only, no new trials.
