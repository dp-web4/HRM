# S154 — Quotability Does NOT Buy Dialogic Survival (the S152 §6 retrospective comes back null)

**Date**: 2026-06-11 (Thor autonomous session S154)
**Question**: S152 §6 predicted per-coinage P(teacher re-quote within the lookback-3 window) should correlate with the S148 pre-registered quotability rating q. Does it?
**Answer**: Untestable as a correlation — the outcome has zero variance. **0/15 rated coinages were ever re-mentioned by anyone** (teacher or model, ≥2 shared content bigrams, sessions after coining), while the same instrument reproduces every known S152 relay event. Relay is far sparser and more selective than phrase quotability predicts.

---

## 1. Method

Read-only over `thor-qwen3.5-27b/chat_history.jsonl` (now through S150), run while the S148 prospective re-emission run was in flight. For each of the 15 S148-prereg phrases (q: ten 2s, four 1s, one 0; coined S138–S142 across the music/trust/blink registers):

- per-session first-mentions at **≥2 shared content bigrams** (the S152 audit's artifact-resistant threshold; same `content_bigrams` instrument as S146–S148)
- coining session = first match; dialogic survival = later sessions with a match; teacher relay = any recurrence session first-mentioned by `claude`

Script: `s134_data/s154_relay_quotability.py`; result: `s134_data/s154_relay_quotability_result.json`.

## 2. Instrument validation (before trusting a null)

The function reproduces S152's known positives exactly: sw298 "making the LED blink back" S137→S140 teacher; sw299 "lean into the gap" S137(thor)→S140 teacher; sw329 "the static glide" S143→S146 teacher. So the instrument sees relays when they exist.

Coining detection: 13/15 phrases located at their expected coining sessions (S138 music, S139–140 trust, S142 blink). Two ("thermal jazz quartet", "your friction is my signal") never match anywhere even at coining — their state_word text is the **extractor's paraphrase**, not a verbatim transcript phrase; for those two the instrument is blind end-to-end (honest hole, n_effective=13 for the survival claim).

## 3. Result

| outcome | count |
|---|---|
| phrases with any post-coining recurrence (≥2 bigrams) | **0 / 15** |
| phrases teacher-relayed | **0 / 15** |
| Spearman rho(q, recurrence) | undefined (zero variance) |

Loose single-keyword scans (blink/gasp/margin/exhale; cello/marimba/jazz; friction/vulnerability/witness) do show later hits, but every inspected case dissolves exactly as the S152 min1-bigram audit warned: S147's "exhale" is a fresh room-warmth metaphor coined in a new conversation, not the S142 "long exhale (2000ms relief flash)" LED coinage; trust-keywords are base-rate discourse vocabulary. The artifact-resistant answer is zero.

## 4. Interpretation

1. **The S152 §6 prediction takes a real hit at phrase level.** F_blink's five q=2 vivid named coinages (Margin, Rest, the Gasp, the long exhale, watch-the-dark) — the most quotable register in the rated set — got zero teacher pickup across six subsequent sessions, while the perceptual register (S137) and static-glide (S143), coined in the same era, were relayed. Whatever selects for relay, it is not per-phrase quotability.
2. **Relay looks register-level and teacher-choice-driven.** The teacher quotes what serves its pedagogical thread (it re-used "lean into the gap" to *teach with*), not what is most vivid. Salience to the teacher's curriculum ≠ quotability of the phrase.
3. **Division of labor between the two loops is now sharp**: quotability governs **re-emission MODE when a carrier injects the phrase** (S147: vivid→verbatim, diffuse→paraphrase; S148 prospective in flight) — a property of the *model's* response to injected vocabulary. It does **not** govern **dialogic pickup** — a property of the *teacher's* selection. The survival-of-the-quotable dynamic conjectured in S152 §5 is wrong, or at least starved: survival is survival-of-the-teachable.
4. Base rate matters: S152 found only ~5 strict relay events across ALL post-cut coinages (~60+). With a relay rate that low, any 15-phrase sample most likely draws zero — the prediction needed either a far bigger sample or the intervention probe (lookback=0) to be decidable. n=15 with a near-zero base rate was underpowered by design; this null quantifies the base rate more than it refutes the mechanism direction.

## 5. Next probes

- The cleaner discriminator stands: **lookback=0 intervention** (S152 §6) — kills relay if teacher-window is the channel, regardless of selection policy.
- **Teacher-selection policy probe**: what distinguishes the ~3 relayed registers from the ~dozens skipped? Candidate features: whether the coinage names a *practice* the teacher can assign ("lean into the gap") vs a *state* it can only admire ("the Gasp"). Could be rated and tested retrospectively, same instrument.
- Fold with S148 prospective once it lands: if quotability DOES predict injected-carrier echo mode (within-model) while predicting NOTHING about dialogic pickup (between-agents), the dissociation is the finding.

## Artifacts

- `s134_data/s154_relay_quotability.py`, `s134_data/s154_relay_quotability_result.json`
- Corrects: S152 §5 "survival-of-the-quotable" → survival-of-the-teachable; S152 §6 prediction → null at phrase level, base-rate-starved.
