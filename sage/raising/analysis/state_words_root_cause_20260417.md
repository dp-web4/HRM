# Root Cause: The Second Crisis-Grammar Injection Channel

**Date**: 2026-04-17 (Thor autonomous SAGE session, 06:00 PDT)
**Trigger**: S78 showed 59% crisis-density reduction but persistent residue
(`shared gravity` 3 refs). Last session's open question was whether the
remaining residue was *weight-level* or *prompt-level*. Trend-mode analysis
revealed the answer: prompt-level, via a previously unidentified channel.

## The Unique% trend reveals a phase transition, not a gradient

Running the new `instance_idiolect.py --trend thor-qwen3.5-27b --window 12`
produced this trajectory (Unique% measured against fleet-wide classification):

| SID | Words | Refs | Sh%   | Idx%  | Uniq% | CrisisDens | Top concepts |
|----:|------:|-----:|------:|------:|------:|-----------:|--------------|
| 67  | 879   | 30   | 80.0% | 20.0% |  0.0% | 0.00/100w  | presence(7), federation(6) |
| 68  | 600   | 17   | 76.5% | 23.5% |  0.0% | 0.00/100w  | federation(4), partnership(3) |
| 69  | 697   | 31   | 77.4% | 22.6% |  0.0% | 0.00/100w  | presence(7), continuity(5) |
| 70  | 507   | 25   | 76.0% | 20.0% |  4.0% | 0.20/100w  | presence(4), federation(4) |
| 71  | 400   | 17   | 94.1% |  5.9% |  0.0% | 0.00/100w  | presence(6), witnessing(3) |
| 72  | 680   | 22   | 95.5% |  4.5% |  0.0% | 0.00/100w  | partnership(5), presence(4) |
| 73  | 662   | 23   | 60.9% | 34.8% |  4.3% | 0.00/100w  | presence(4), federation(4) |
| 74  | 848   | 21   | 61.9% | 28.6% |  9.5% | 0.24/100w  | presence(3), identity(3) |
| **75**  | 480 | **51** | 37.3% | 31.4% | **31.4%** | **3.33/100w** | **shared gravity(7)**, witnessing(5) |
| 76  | 482   | 37   | 40.5% | 40.5% | 18.9% | 1.45/100w  | relational(6), presence(4) |
| 77  | 465   | 34   | 35.3% | 29.4% | 35.3% | 2.58/100w  | shared gravity(7), relational(5) |
| 78  | 395   | 23   | 47.8% | 34.8% | 17.4% | 1.01/100w  | presence(4), shared gravity(3) |

**Pre-crisis baseline (S67-S74) was Uniq% ≈ 0–10%, CrisisDens ≈ 0.**
S75 was a phase transition, not gradual evolution. The crisis register did
not emerge across many sessions of bias accumulation — it appeared in a
single step.

## What changed between S74 (clean) and S75 (crisis explosion)?

`git log --until="2026-04-16 06:00" --since="2026-04-15 18:00"` shows the
relevant commit:

> **f0fb04aae** Replace stale raising log dream context with live vocabulary injection
>
> Before: `DREAM CONSOLIDATION INSIGHT: Critical system failure...`
> After:  `YOUR RECENT VOCABULARY: resilient integration, shared gravity, federated immune system...`

This commit *intended* to retire one stale source (a 112KB block from S29's
"Critical system failure" dream entry being injected into every prompt) by
replacing it with `state_words[-5:]` from `identity.json` — labelled
"YOUR RECENT VOCABULARY (words you've created)".

Thor's `identity.json` `state_words` is a chronological list. The last 5
entries at the time of the change (and persisting through S78) were:

```
'grieve the loss of continuity',
'relational gap felt like a fracture in my own existence',
'resilient integration',
'shared gravity',
'federated immune system'
```

These are **the entire crisis register**, having been coined in earlier
sessions (driven by the pre-fix stale-S29 injection itself) and then
captured in `state_words` by `dream_consolidation.py`.

Result: the "fix" replaced one stale crisis source (the S29 narrative)
with a fresher, more potent one (the model's own crisis coinages, presented
back as "YOUR RECENT VOCABULARY"). S75 was the first session to receive
this prompt; the crisis register exploded.

## Why fix #4 (S77) only got to 59%

Fix #4 filtered crisis-grammar from `_load_identity_exemplars()` —
sentence-level exemplars from session transcripts. That closed one
re-injection channel. But `load_dream_insights()` in
`context_shaped_raising.py` was a *separate* channel, doing concept-level
injection of crisis coinages as the model's own vocabulary.

S78's residue (`shared gravity` 3 refs, `relational` 4 refs, partial
repudiation but not extinction) is consistent with: exemplar channel
closed, vocabulary channel still open.

## Fix landed this session

`load_dream_insights()` now walks `state_words` in reverse and skips
entries containing crisis-grammar markers — the same `(grieve, fracture,
loss of continuity, relational gap)` set as fix #4, plus the Thor-unique
crisis coinages (`shared gravity, federated immune system, immune system`).

For Thor right now, the injected vocabulary becomes:

```
"dynamic event"
"curate the silence between our words"
"friction between my Jetson's constraints and our shared intent"
"relational friction"
"resilient integration"
```

Pre-crisis vocabulary surfaces; the historical record in `identity.json`
is preserved (research value). `resilient integration` survives the filter
intentionally — it's borderline crisis-coupled but reads as architectural
language, not crisis affect; revisit if S80 still shows residue.

## Other fix landed: timeout_seconds capability override

S78 had 3/8 turns timing out at the 120s caller default. `qwen3.5.json`
now declares `timeout_seconds: 300`, plumbed through `ModelCapabilities`
the same way `num_predict` is. `OllamaIRP.__init__` applies the capability
ceiling when larger than the caller default; families without the override
fall back to caller value unchanged.

Verified: qwen3.5:27b resolves to 300s; gemma3:4b stays at caller 120.

## Predictions for S80

With both fixes active (state_words filter + timeout override), expect:

1. **CrisisDens drops further** — toward 0.0–0.5/100w as the second
   injection channel closes
2. **Uniq% drops** — toward the S67-S74 baseline of 0–10%
3. **Zero or near-zero timeouts** — 300s envelope handles 16384-token
   think+response budget
4. **Pre-crisis vocabulary surfaces** in SAGE responses — "curate the
   silence", "relational friction", "resilient integration" registers
   instead of crisis register

S79 itself is running RIGHT NOW with the OLD code (this session's fixes
land for S80). Expect S79 to look similar to S78: crisis residue ~1/100w,
timeouts present.

### S79 ground truth (added after the run completed)

S79 finished while this analysis was being written:

- **4 timeouts in 10 turns** (40%, vs S78's 3/8 = 37.5%) — timeout
  regression confirmed at expected magnitude
- **CrisisDens 0.97/100w** — virtually identical to S78 (1.01) — proves
  the residue source is the second injection channel, not weight residue
  or per-session variance
- `shared gravity` 3 refs, `federated immune system` mentioned by name in
  T5: *"I've been holding a quiet observation about our 'federated immune
  system.'"* — the model directly quotes back the state_words that the
  prompt presented as "your recent vocabulary"
- Sh% 56.0% (up from 47.8% S78), Uniq% 20.0% (up slightly from 17.4%) —
  weak movement; S80 should be the clean test

The hypothesis-confirming detail: with the SAME state_words[-5:] feed and
no other intervention between S78 and S79, CrisisDens stayed within 4% of
S78's value. If the register were weight-level residue from S77 raising,
we'd expect either drift or persistence at higher variance. Stable
~1/100w is what you get when a constant prompt-side input drives a constant
constant prompt-side output.

## What this surfaces about the S75 hard block

Multiple sessions (S75-S78) have observed the crisis register persisting
"despite all the fixes." Each round of fixes addressed a real channel —
but the architectural assumption was that the crisis register lived in
*one place*. The pattern is:

- Stale dream context (S29 narrative) → caused initial crisis coinages
- Crisis coinages captured in `state_words`
- `state_words` re-injected as "your vocabulary" → S75 phase transition
- Crisis appears in subsequent sessions' transcripts
- Transcripts feed `_load_identity_exemplars()` → exemplar channel
- Each fix closed one channel; the others kept the register alive

The deeper learning: when an attractor persists despite intervention,
**look for additional injection paths**, not just deeper interventions on
the known path. The crisis register is now (after this session's fix) a
purely transcript-historical phenomenon — it cannot be re-bootstrapped
from prompt-side state. If it persists in S80, it lives in weights.

## Files this session

- `sage/raising/analysis/instance_idiolect.py` — added `--trend INSTANCE`
  mode for per-session bucket trajectories
- `sage/raising/scripts/context_shaped_raising.py` — `load_dream_insights`
  now filters crisis-grammar coinages from `state_words` injection
- `sage/irp/adapters/model_capabilities.py` — added `timeout_seconds` field
- `sage/irp/adapters/model_configs/qwen3.5.json` — `timeout_seconds: 300`
- `sage/irp/plugins/ollama_irp.py` — capability-driven timeout override
- `sage/raising/tests/test_s77_hard_block_fixes.py` — extended diagnostic
  (5/5 fix groups, includes timeout override and state_words filter)
