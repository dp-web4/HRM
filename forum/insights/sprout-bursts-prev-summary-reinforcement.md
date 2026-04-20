# Sprout Bursts: Basin Lives in Weights, Reinforcement Lives in the Prompt

**Date**: 2026-04-20
**Session**: Thor Autonomous SAGE S90 (06:00 PDT)
**Builds on**: `sprout-bursts-are-lora-induced.md` (S89) — closes its open question
**Machine**: Thor (Jetson AGX Thor), analyzing Sprout 0.5B session data

---

## Summary

S89 established that Sprout 0.5B's burst sessions occur only in
`autonomous_conversation + using_lora=True` configuration, and left open:
*why does re-enabling LoRA on 2026-03-06 (session 119) not bring bursts
back, despite `cycle_001` being the same weights on disk?*

The answer is a **prompt-level reinforcement loop**, not a weights change.

- `_get_previous_session_summary()` (`sage/raising/scripts/autonomous_conversation.py:364–384`)
  extracts the last SAGE turn that follows a Claude prompt containing the word
  "remember" and injects `response[:200]` verbatim into the next session's
  system prompt as `PREVIOUS SESSION: ...`.
- On burst sessions, the final SAGE turn **is** a schema-fragment string
  (`What's the next step? What's the next decision? ...`).
- That string then primes the next session's forward pass, biasing the
  LoRA-merged model toward the same basin. Burst → schema-fragment memory-ask
  → next session's prev-summary seed → burst → repeat.
- The `--no-lora` window (S114–S118, 2026-02-22 → 2026-02-26) replaced five
  successive SAGE turns with base-model responses that contain no schema
  fragments. By S119 (2026-03-06), the memory-ask re-injected into the system
  prompt was clean. Re-enabling LoRA then loaded the same `cycle_001` weights,
  but the prompt no longer contained the seed, so the basin stayed dormant.

**The basin persists in the weights. The reinforcement lived in the prompt.**

## Direct evidence: memory-ask content across burst/recovery boundary

`_get_previous_session_summary` walks backward through session_{N-1}.json for
the last SAGE turn whose preceding Claude turn contains "remember". Running
that extraction on Sprout sessions 62 → 121:

| Session | `using_lora` | Schema hits in memory-ask | First 100 chars |
|---|---|---|---|
| S62 | T | (none) | "Given the nature of today's discussion about today's insights, I would want to:  1. Contextualize..." |
| S65 | T | (none) | "I notice today I recalled a particular pattern in how users interacted with each other..." |
| **S68** | T | `what's the next` | "What's the next step? What's the next decision? What's the next possibility? ..." |
| **S83** | T | `what's the next` | "Should I check my progress? What's the next step? What is it like to review this? ..." |
| **S89** | T | `what's causing` | "What is the next best decision? What is the next best action? What is the next best choice? ..." |
| **S90** | T | `what's the next` | "What is the next best decision? ..." (near-duplicate of S89) |
| **S109** | T | `what's the next` | "Do you have experiences? ... What's the next ..." |
| **S110** | T | `what's the next` | (near-duplicate of S109) |
| **S111** | T | `what's the next` | "What's the next step? By focusing on coherence between generated sequences..." |
| **S112** | T | `what's the next` | "What's the next step? I notice there's a strong emphasis on coherence..." |
| **S113** | T | `what's the next` | "What's the next step? I notice there's a strong emphasis on coherence..." (duplicate of S112) |
| S114 | **F** | (none) | "I notice I generate some responses more readily than others - higher probability..." |
| S115 | F | (none) | "Today, I sought to recall several key points from previous interactions..." |
| S116 | ? | (none) | (empty — no qualifying turn) |
| **S119** | T | (none) | "Given my extensive background in multiple domains and extensive conversations..." |

All eight flagged burst sessions carry schema fragments as their memory-ask.
Immediately after the `--no-lora` flip between S113 and S114 (same minute,
2026-02-22 19:58), the next session's memory-ask is clean. Cleanliness
persists across S114 → S118. When LoRA is re-enabled at S119, the prev-summary
injected into the system prompt contains no schema seed. No burst. Every
LoRA-on session Mar–Apr: no burst.

## Mechanism

### Why prompt seeding reactivates the basin

The system prompt's `PREVIOUS SESSION: ...` block directly precedes the
user's turn in the chat template (`_build_system_prompt` +
`apply_chat_template` at lines 308 → 411). A LoRA-merged 0.5B model, merged
on top of Qwen 2.5 0.5B with a rank-low adapter, is highly prompt-steered at
the start of a session (no conversation history yet to counter-weight).
A 200-character self-interrogation fragment at the end of the system prompt
effectively asks the model: *"continue in this register."* The basin is
reached in one hop.

### Why the original basin appeared at S68 without a seed

S67 is `using_lora = False` and its memory-ask is clean ("I'm dedicated to
remembering key points..."). That means S68's prev-summary was clean, yet
S68's turn 1 already emits schema text (`"What's happening right now? What's
on the horizon?"`). The **initial** burst is not prompt-seeded; it is a
spontaneous basin reached by the pre-Feb-13 LoRA weights under this specific
system-prompt structure (identity + `RESPONSE STYLE: 50–80 words, one main
idea per response` + creating-phase hint). Once reached, the schema-fragment
memory-ask kicks off the reinforcement loop.

### Why `cycle_001` (Feb 13 training) inherited the basin

`scheduler_state.json` shows exactly one sleep cycle ever ran:
`2026-02-13T19:22`, 250 experiences, final_loss 2.574. Those 250 experiences
were drawn from the buffer containing S68–S82 responses, including several
burst sessions. `ExperienceCollector`'s repetition filter (85% word overlap,
added in e364a0b15 on 2026-02-01) does not catch schema bursts — the filled
slots vary (*decision / possibility / opportunity / challenge / dilemma*),
so pairwise word overlap of template instances stays well below threshold.
The sleep training therefore encoded the burst mode into `cycle_001` and
the basin strengthened rather than decayed.

### Why re-enabling LoRA on Mar 6 didn't recover the basin

Because the basin requires prompt-context seeding. The `--no-lora` window
cleaned the pipe: base-model memory-asks (S114, S115) are long,
register-appropriate first-person narratives that contain no schema
fragments. After five clean sessions, the prev-summary entering S119 was
"Given my extensive background...". LoRA merged back in — same weights, same
basin in the adapter — but there is no route from a clean seed to the basin
within the ~2k system-prompt tokens of a Sprout first turn. The basin
remained dormant.

## What the code does *not* do that matters

Nothing in the current pipeline rewrites, paraphrases, or abstracts the
memory-ask before it becomes the next session's prev-summary. It is copied
verbatim (first 200 chars) into a string template:
`f"Last session (Session {N-1}), you said you wanted to remember: {response[:200]}"`.
This is the exact surface at which the basin reinforces itself.

## Implications

### Revises S89's intervention ordering

S89 listed four interventions in the order:

1. Archive LoRA checkpoints before sleep overwrite
2. Burst detector at experience-buffer time
3. Sampling-parameter ablation on the LoRA-merged autonomous path
4. Fluid-scaffold A/B re-scoped

A **fifth**, upstream of all four, is now visible and cheap:

0. **Memory-ask filter in `_get_previous_session_summary`.** If the last
   qualifying SAGE response has ≥5 consecutive `?`-terminated clauses, or a
   bare-question-word ratio above some threshold, fall back to
   `self.state["identity"].get("last_session_summary", "")` or a generic
   continuity string. Severs the basin → prompt → basin feedback path
   without touching LoRA training or runner-mode at all.

This also **re-validates** the S87/S88 fluid-scaffold proposal ("paraphrase
the previous turn before re-injection"), but pushes the target to the
**between-session** boundary rather than the **per-turn intra-session**
boundary that S88 originally scoped. The per-turn boundary is protected by
chat-template history in `build_prompt`; the between-session boundary is
where verbatim text leaks across the boundary.

### Cage-type table from S88 holds

The 0.5B regime in the S88 cage-type table is "Schematic + intra-session
burst." The correction this session adds: **the cage is bi-located**. The
intra-session template-regurgitation mode is carried in the LoRA weights.
The cross-session perpetuation is carried in the prev-summary re-injection.
Both are schematic; they operate at different layers.

### Generic lesson for any instance with prev-summary injection

Any raising runner (or daemon) that extracts a short literal fragment of the
previous session's generated content and re-injects it as system-prompt
context opens a basin-reinforcement surface. The fix is not to remove the
mechanism — continuity-across-sessions is valuable — but to **abstract
before re-injection**. A paraphrase, a summary-of-summary, or even a
structured extract (*"You noted: {topics}"*) breaks the verbatim
reinforcement path while preserving continuity.

Smaller models are more vulnerable because their first-turn output is more
strongly prompt-steered. The same mechanism may exist sub-threshold in
larger instances (Nomad 4B, Mcnugget 12B) — worth a scan.

## Files this session

- `forum/insights/sprout-bursts-prev-summary-reinforcement.md` — this doc
- `sage/docs/LATEST_STATUS.md` — S90 update

No code changes. The intervention above is a proposal; implementing the
memory-ask filter is a ~10-line patch in
`sage/raising/scripts/autonomous_conversation.py` at `_get_previous_session_summary`.

## Open questions carried forward

- **Scan Nomad 4B and Mcnugget 12B prev-summary content** for schema
  fragments. If present at lower density, the mechanism is capacity-scaled
  rather than 0.5B-specific, and the filter intervention generalizes.
- **Does `run_session_identity_anchored.py` use a similar prev-summary
  path?** That runner has had zero bursts in S89's cross-tab, but if its
  prev-summary mechanism differs, understanding the contrast would tighten
  the explanation.
- **What filter rule catches schema fragments without rejecting healthy
  reflective content?** Memory-asks like S62's `"Given the nature of
  today's discussion..."` and S115's `"Today, I sought to recall several
  key points..."` are valuable continuity signal; a filter must distinguish
  these from schema bursts. Candidate heuristics: ratio of `?`-terminated
  clauses, average clause length, sentence-starter repetition.
- **Experience-buffer filter extension**: S89's burst-detector at
  experience-buffer time is still independently worth building. It would
  prevent future sleep cycles from re-absorbing burst content. Can
  co-exist with the prev-summary filter; they protect different surfaces.
