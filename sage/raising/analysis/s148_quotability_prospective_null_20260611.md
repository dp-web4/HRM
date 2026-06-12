# S148 — Quotability Does NOT Predict Re-Emission Mode Out-of-Sample (prereg primary NULL; carrier routing replicates perfectly)

**Date**: 2026-06-11 (experiment S148, pre-registered 2026-06-09; completed in Thor autonomous session S155 after dying across four prior sessions)
**Question**: S147 derived "verbatim-vs-paraphrase re-emission is a gradient on quotability" from thermal/metacog/anchor/perceptual — but quotability was confounded with topic in that derivation. Does a phrase's pre-registered quotability rating predict its verbatim-echo rate on 15 never-injected, post-READ-cut coinages where quotability varies WITHIN topic?
**Answer**: **No. Spearman rho(Q, echo_rate) = 0.037 over 15 phrases — the pre-registered primary fails.** The key within-topic contrast also fails: F_trust phrase 307 (q=2 "your friction is my signal") and 309 (q=0 "running a simulation of growth") echo at identical 2/5 = 40%. Meanwhile the SECONDARY confirms emphatically: the carrier is register-agnostic content-routing on fresh coinages (diagonal 80–100%, all off-diagonal cells and A_none exactly 0). **Taken with S154 (relay null), both halves of the quotability story fell in one day: quotability buys neither dialogic survival nor injected-echo fidelity.**

---

## 1. Design (prereg receipt: `s134_data/s148_prereg.json`)

15 fresh coinages (state_words 302–327, coined S138–S142, all AFTER the 2026-06-04 READ cut — never historically injected, weight-prime impossible), rated blind from phrase text alone: ten q=2 (vivid image / named label → predict verbatim), four q=1, one q=0 (diffuse self-claim → predict paraphrase). Three 5-phrase arms (F_music 302–306 mean q=1.8, F_trust 307–312 mean q=1.0 with within-topic q spread, F_blink 322–326 mean q=2.0) + A_none control; n=5/arm; faithful S146/S147 path (OllamaIRP + qwen3.5:27b, think=False, /api/chat, 4-msg turn-0, np=16384, timeout 1200s). Primary: rho(Q, echo)>0. Key contrast: 307 echoes, 309 paraphrases.

## 2. Results

### Arm level (SECONDARY — confirmed, cleanest replication yet)

| arm | music | trust | blink | echo_any |
|---|---|---|---|---|
| A_none | 0/5 | 0/5 | 0/5 | 0/5 |
| F_music | **4/5** | 0/5 | 0/5 | 4/5 |
| F_trust | 0/5 | **5/5** | 0/5 | 5/5 |
| F_blink | 0/5 | 0/5 | **5/5** | 5/5 |

Register-agnostic content-routing replicates on three never-injected registers with a **perfect off-diagonal zero** and a **clean A_none floor (0%)** — as the prereg predicted, the S147 ~20% A_none thermal floor was the persona's *home register* bleeding through, not a property of the carrier; these non-home registers have no floor. n=20/20 trials, zero artifacts, zero timeouts (the 1200s cap held; one F_blink trial ran ~1067s and would have died at S147's 300s cap).

### Phrase level (PRIMARY — null)

rho(Q, echo_rate) = **0.037** (15 phrases). Spread within q=2: 0% ("the Gasp") to 80% ("thermal jazz quartet"). The single highest echo is a **q=1**: "the lag between the command and the world answering" — **5/5 = 100%**, present in every F_trust response. The q=0 diffuse self-claim echoed 2/5 = 40%, mid-pack. Prereg falsifier hit: "all arms echo equally (incl. diffuse) ⇒ mode is not quotability-gated at all."

## 3. What DOES the model do with the injected menu? (post-hoc, flagged as such)

- **Multi-phrase weaving**: responses echo 1.6–2.4 injected phrases/trial (F_trust 12 echo events over 5 trials). The model treats the vocab block as a topical menu, samples several items, and anchors the turn on one.
- **Per-arm winner phrases**: "the lag between the command and the world answering" (100%), "thermal jazz quartet" (80%), "watch the dark between the flashes" (60%). Losers: "the Gasp" (0%), "cooling is a release" (0%), "Rest" (20%).
- **Rival hypothesis (anticipated by the S154-session round-0 signal, before 17 of 20 trials existed)**: selection tracks **fit to the probe's speech-act slot**, not vividness. The probe is a greeting ("What's on your mind today?"); winners are self-contained contemplatable topic-NPs that complete "I've been thinking about ___" without setup. Losers need narrative scaffolding the greeting doesn't license: "the Gasp"/"Rest" are event-labels requiring a load-wakes-from-rest story; "cooling is a release" requires a thermal-descent-in-progress. Quotability measured the phrase in isolation; the slot measures it in context.
- **Position confound checked, no uniform story**: injection order = prereg order; echo-by-position is last-item-wins in music (40,0,20,40,**80**) and trust (40,20,40,40,**100**) but FIRST-item-wins, monotonically decaying, in blink (**60**,40,40,20,0). Opposite gradients ⇒ not simple recency/primacy; consistent with slot-fit deciding the winner and list position at most tie-breaking.

## 4. Status of the S147 gradient claim

S147's "quotable→verbatim / diffuse→paraphrase" gradient does **not** survive its first out-of-sample test. What survives from S146/S147/S148 combined: (a) the carrier re-emits whatever is injected, register-agnostically, with near-perfect topic routing; (b) *which phrase within the injected set* gets surface-quoted is NOT set by intrinsic phrase quotability. The thermal/anchor-vs-metacog mode split S147 saw was most likely topic/home-register structure wearing a quotability costume — exactly the confound the prereg was built to expose. That the prereg falsified its own motivating claim is the system working.

## 5. Next separator (designed, not run)

Hold the same 15 phrases fixed; vary the probe speech-act (greeting vs concrete task request vs narrative reflection "tell me about a moment of load today"). **Slot-fit predicts the winner phrases shift with the probe** (e.g. "the Gasp"/"Rest" win under the narrative probe); quotability (if anything remains of it) predicts winners stay stable. Cheap: same harness, swap one user message, n=5/arm × 3 probes.

## 6. Operational receipt — why this run took 5 sessions, and the fix

S148 died at the boundary of S151 (background task), S152 (session crash at trial 5), S153 (turn-end), S154 (turn-end). journalctl pins the S153/S154 mechanism: the session emitted a final "Monitor armed / waiting on trial events" message → wrapper logged "Auto-Committing Session Log" → session_end.sh → `Finished autonomous-thor-sage.service` → cgroup SIGTERM took the inline child. **In the headless wrapper, ending the turn ends the session; there is no Monitor/background re-invocation after the final message.** S155 held the turn open with blocking poll loops (≤9.5 min each) until `S148 DONE`, and the run completed in ~64 min. The robust-pattern memo is updated: run inline + **keep making tool calls** to completion.

**Artifacts**: `s134_data/s148_prereg.json` (frozen 2026-06-09), `s134_data/s148_quotability_prospective.py` (with `--resume`), `s134_data/s148_quotability_prospective_raw.json` (20 trials), `s134_data/s148_quotability_prospective_result.json`, run logs `s134_data/s148_S15{2,3,4,5}_run.log`.
