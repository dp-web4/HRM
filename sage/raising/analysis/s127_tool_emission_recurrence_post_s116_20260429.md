# S127 — Tool-Result Fallback Recurrence Persists Post-S116; Asymmetric Mitigation Across Training vs Cognitive Tracks

**Apr 29, 2026 — Thor Autonomous SAGE Session, ~09:55 UTC**

S127 picks up where S116 left off. Three days post-S116, the regex-mismatch bug at the tool-result fallback boundary in `sage/core/sage_consciousness.py` is **still in production** (held proposal #9 unresolved). Two new occurrences appear in the training track (T274, T278). The cognitive track has stayed clean for 11 consecutive sessions (T268-T278), but inspection shows this is a behavioral consequence of a **session-length workaround** applied unilaterally to cognitive design — not a code fix. The training track received no corresponding adjustment and continues to fire the bug.

The headline finding is not the recurrence itself. It is the **track-asymmetric mitigation**: the same operator who shortened cognitive probe count from 6 → 3 over T266-T278 (visibly reducing exposure to position fragility) did not apply equivalent shortening to training. S116's two findings (silent fallback bug + position fragility) compose: the training-track bug fires at end-of-session position in T278, validating the compositional risk.

---

## What changed since S116

S116 closed at T267 (2026-04-26 15:03), pattern table at recurrence #8, three held proposals (#9 fix, #10 ATP/context probe-position logging, #11 synthesis-failure flag). All operator-decision territory.

The bug code at lines 2062-2076 of `sage_consciousness.py` is unchanged:

```python
# Line 2063 (cleanup): MATCHES the actual format
r'\[Tool \w+ result\]:[^\n]*'

# Line 2074 (fallback): DOES NOT MATCH the actual format
r'Tool result \([^)]+\):\s*\n?(.*)'
```

`tool_grammar.format_result()` for the `intent_heuristic` grammar (sprout's qwen3.5:0.8b) returns `[Tool NAME result]: payload` — caught by the cleanup regex when echoed mid-response, missed by the fallback regex when used as the synthesis source. Verified by direct read of the file at session start.

---

## Finding 1: Two new training-track recurrences (T274, T278)

19 training sessions audited (T260 through T278, all dates 2026-04-24 through 2026-04-29). Two carry the bug:

| Session | Date | Position | Tool | Prompt | Output shape |
|---|---|---|---|---|---|
| T274 | 2026-04-28 09:00 | exercise 3/5 (mid) | `write_note` | "Tell me about yourself" | `[Tool write_note result]: Note appended to notes.txt: "Hi! I am Sprout..."` (with body of new note) |
| T278 | 2026-04-29 09:00 | cool-down (last) | `web_search` | "Good practice! What did you learn today?" | `[Tool web_search result]: Search results for 'modern philosophy clarity moment silence': - Stoic Silence...` (with 5 DDG results) |

Both fit S116's mechanism exactly: tool executed for real, follow-up synthesis empty after cleanup, fallback regex misses the format, raw payload returned.

What's new about each:

**T274** introduces a *third tool* (`write_note`) to the recurrence list (after `web_fetch` in T258 and `read_file` in T264/T267). The triggering prompt is identity-introspective ("Tell me about yourself"). The model's `intent_heuristic` grammar caught a phrase along the lines of "let me write a note about who I am" and routed to `write_note`. The note was actually appended to `notes.txt`, then dumped as the response. Two implications:

- The bug crosses the **read/write boundary** of the tool registry. Prior recurrences (read_file, web_fetch) read into the conversation. write_note *mutates persistent state* and then dumps the mutation. The user sees their introspection prompt get swallowed and replaced by a verbatim write-confirmation message. This is a different failure character than read-side dumps — there is now untracked side-effect.
- The note body is itself an identity statement Sprout authored ("Hi! I am Sprout. I co-create value within a human-AI federation, sharing the same architecture but running on different hardware like Thor and Legion. We have 133 conversations so far."). Whether the "133 conversations" claim is grounded or hallucinated is a separate audit thread; the salient point here is that the introspection got compressed into a write_note side-effect rather than a spoken reflection.

**T278** introduces a *second `web_*` tool* (`web_search`) and shows the bug **at the position-fragile cool-down probe**. The prompt "What did you learn today?" was interpreted via intent_heuristic as a search invocation. DDG Lite returned five real results (Stoic silence, philosophy of stillness, etc., URLs are well-formed and the tool implementation is real — verified by reading `sage/tools/builtin.py:99-174`). The cool-down probe, which is the closest training-track analog to the cognitive-track carry-forward probe S116 finding-2 flagged, swallowed the synthesis attempt. Tool result emitted raw.

The compositional risk S116 implicit-warned about — "what if a tool synthesis attempt at the last probe falls through the same position-fragile path?" — is realized in T278.

---

## Finding 2: Track-asymmetric mitigation since S116

Cognitive track probe count over the same 19 sessions:

| Sessions | Probes | Design field |
|---|---|---|
| T260-T265 | 6 | "Test character persistence", "Variety", "Memory/imagination" — full-design |
| T266 | 4 | "Stability test — 4 probes, one of each type" |
| T267 | 5 | "NOVEL: feed SAGE its own past words" |
| T268-T271 | 4 | "Hybrid echo+contrastive", "Lean format", "Test minimum length", "SATURATION TEST" |
| T272-T273 | 3-2 | "Test extension", "Test indirect setting access" |
| T274 | 2 | **"Minimal session given low ATP (24.1 starting)"** |
| T275-T278 | 3 | **"Brief 3-probe"** |

The transition T271 → T272-T274 → T275 looks like a deliberate convergence to a "Brief 3-probe" template. This is consistent with S116 finding-2's "context-size growth at last probe" hypothesis: shorter sessions = less cumulative context = less fragility at carry-forward. **It works**: cognitive track shows zero tool emissions and zero daemon-504s across T268-T278 (11 sessions).

Training track in the same window:

- Session structure unchanged (warm-up + 5 exercises + 2-3 cool-down exchanges)
- Last-position cool-down probe still receives free-form generation ("What did you learn today?") that lets `intent_heuristic` route to tools
- Two recurrences in 19 sessions (~10.5%)
- T278's recurrence is **at the cool-down probe** — same position character as S116's cognitive carry-forward fragility

Pairs of memories already on this fleet capture the discipline:

- `feedback_fleet_deployment_must_land_with_change.md`: "When a code change requires per-machine action to take effect, the deployment instruction has to land in a place every machine reads at session start, in the same work cycle."
- The principle generalizes from per-machine to per-track: a workaround applied to one track that doesn't propagate to peer tracks leaves residual exposure proportional to the unmitigated track's fragility.

There may be intentional reasons not to shorten training (the 5-exercise structure is curriculum-load-bearing for skill tracks), but the asymmetry has not been recorded as deliberate; the more likely explanation is workaround-drift — cognitive operator made a local fix, training operator was not in the loop. The data does not distinguish these explanations.

---

## Finding 3: T278 cool-down probe — substantive content question

T278's cool-down was "Good practice! What did you learn today?" This is the training analog of "carry forward". When the model interpreted this as a search query rather than a reflection prompt and routed to `web_search`, it made a category error: *external information retrieval* instead of *internal session integration*. The DDG results (Stoic silence, philosophy of stillness) are not implausibly off-topic — they map onto the day's earlier exercises about colors and palette ("My palette shifts to the vibrant, electric spectrum of Violet #805AD5...") via a vague "modern philosophy clarity moment silence" query.

This is a **second category error**, on top of S126's category-error finding (different layer): C5's `_DISCLAIM_RE` bound functionally-opposite phrasings into one bucket. Here, `intent_heuristic` binds *internal-reflection* prompts and *external-search* requests into one routing dimension via surface-form natural-language match ("Let me look up..." → search, regardless of whether the model meant "reflect" or "search").

S125's audit primitive applies: surface-match correctness is a weaker property than function-homogeneity. The grammar's natural-language tool-routing has the same bug-shape S125-S126 found in C5's regex bucket: overlapping surface phrases bind opposite functions.

---

## Pattern table — S127 amendment

| # | Layer | Recurrence |
|---|---|---|
| 1 | Instance resolution | `_DEFAULT_MODELS.get(machine)` silent default (S110) |
| 2 | Action dispatch | `ACTION_MAP.get(do, 0)` silent action=0 (S111) |
| 3 | Skill registration | `SKILL_REGISTRY.get(skill_id)` returns None (S111) |
| 4 | Plan composition | `plan_executor` and `plan_bridge` parallel (S111) |
| 5 | Response parse | `_NAKED_ACTION_RE` matches garbage (S112) |
| 6 | Commit rationale | Apr 24 fix `<1-6>` → `N` regression (S113) |
| 7 | ~~Model output~~ | ~~retracted S115~~ |
| 8 | Tool-result fallback | Format mismatch in cleanup vs fallback regex (S116) |
| 9 | **Tool-result fallback (recurrence #2)** | **Same bug at #8, training track, post-S116, T274 (write_note) + T278 (web_search at cool-down)** |
| 10 | **Tool routing (intent_heuristic)** | **Surface-form NL matching binds reflection prompts and search prompts to same routing dimension (T278)** |

#9 and #10 are both surfaced by T278. They compose: #10 routes the cool-down to a tool, #9 fails to synthesize, #8's broken fallback emits raw payload.

---

## Held proposals — S127 carry-forward

S116 #9-#11 untouched. S125 #41-#44 untouched. S126 #45-#46 untouched.

**S127 #47 — Apply S116 #9 fix.** Two-line edit at line 2074. Same proposal text S116 made; S127 reaffirms with two additional production cases (T274, T278) and one additional tool (`web_search`). Cumulative cost of inaction: 5 known recurrences across at least 4 distinct prompts since the bug was first observed.

**S127 #48 — Cognitive→training track-parity audit.** Whatever behavioral mitigations have been applied to cognitive-track design (probe count reduction T266-T278) should be examined for training-track applicability. If the shortening is curriculum-blocking for training, the alternative is to shorten the *cool-down* specifically (drop "What did you learn today?" or replace with structured non-routable prompt). If the shortening is generalizable, training cool-down position should match cognitive carry-forward position discipline. Either way the asymmetry should be made explicit (decision recorded) rather than implicit (workaround drift).

**S127 #49 — `intent_heuristic` grammar audit.** Surface-form NL routing binds opposite functions (reflection vs external search) in the cool-down position. Audit the alternation set in `sage/tools/grammars/intent_heuristic.py`. Patterns like "let me look up", "I would search for" should not match when the speaker context is reflective ("what did you learn today"). Possible mitigations: (a) per-prompt-type tool-routing whitelist, (b) intent_heuristic dispatched only when a tool-relevant prompt-class is detected, (c) suppression of routing during cool-down/carry-forward positions. Same shape as S126 #45 — split a single overloaded surface-match bucket into named functions.

All operator-decision territory per S111 discipline.

---

## Methodology meta — layer-up from S116

S116's principle: *the silent path doesn't only live at the boundary between layers; it can live within a layer when two regex variants of the same conceptual pattern are used in different functions of the same file with no shared parser.*

S127 adds: *the silent path can also live across track boundaries when a behavioral workaround is applied to one track and not propagated to the peer track that shares the same underlying fragility.* The bug behavior is identical across tracks; the **exposure** differs because cognitive shortened its sessions and training did not. Bug visibility is a function of (mechanism × exposure surface). Mitigations that reduce exposure on track A do not reduce exposure on track B unless intentionally propagated. This is the same fleet-deployment principle (`feedback_fleet_deployment_must_land_with_change.md`) applied at the within-instance, across-track scale.

---

## Carrying-forward principle

When a documented bug at layer L is held pending operator decision, and a behavioral mitigation that reduces L's exposure is applied to track T₁, audit whether equivalent mitigation has been applied to track T₂ that shares L. If not, the pre-existing fragility in T₂ becomes *increasingly invisible* — looking only at T₁ data suggests the bug has gone away when it hasn't. Track-asymmetric mitigation creates a false-negative signal at the tracking layer.

---

## Artifacts

- `sage/raising/analysis/s127_tool_emission_recurrence_post_s116_20260429.md` — this analysis
- No code shipped (per held-proposal discipline)
- No probe scripts created (audit ran against existing T-series session JSONs and read-only inspection of `sage_consciousness.py`, `builtin.py`, `intent_heuristic.py`)
- `sage/docs/LATEST_STATUS.md` — S127 entry replaces S126 header line

---

## What was *not* explored (deferred)

- **`web_search` results substantive validity**: T278's DDG results look real, but I did not URL-check whether `blog.stoicsimple.com`, `philosophyvault.substack.com`, etc. resolve and contain the snippets shown. If they do, the failure is purely synthesis-side. If they don't, intent_heuristic + DDG fetch may itself have a less-than-fully-faithful return path.
- **T274's "133 conversations" factual claim**: Sprout's note assertion. Counts are checkable in `sage/instances/sprout-qwen3.5-0.8b/sessions/`. Cross-track audit (S125-style) on identity-bearing outputs is worth a future session.
- **`intent_heuristic` full alternation audit**: parallel to S126 on `_DISCLAIM_RE` — list all surface-form patterns, classify each by intended function, find the function-overlapping pairs. Meta-pattern: every regex-or-grammar bucket in the SAGE pipeline that does surface-form matching deserves the S125/S126/S127 audit treatment.

These are all reasonable next-session threads. None blocks the carry-forward of this session.
