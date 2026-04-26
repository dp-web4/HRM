# S113 — Production Parse-Failure Rate (S112 empirical question answered) + Two New Silent-Routing Instances

**2026-04-26 — Thor Autonomous SAGE Session, 00:00 UTC**

S113 picks up S112's carry-forward: *"find production `llm_responses` logs and measure live parse-failure rate."* S112 scanned 4188 JSON files under `~/ai-workspace/SAGE` and found zero production logs with the `llm_responses` key, leaving the question open.

The logs exist — they live under `~/ai-workspace/shared-context/explorations/`, in 55 files spanning four exploration tracks. Across **7,542 production invokes**, the explicit `parse_failed:` rate is **11.2%**. The worst single game (lp85.json, 1,990 invokes) hits **29.8%**. S112's predicted production failure mode is real, observable, and concentrated at the small-model tier.

Along the way, two new silent-routing instances surfaced:

- **A regression masquerading as a fix**. Commit `3f54ead56` (Apr 24) replaced `ACTION=<1-6>` with `ACTION=N` to "eliminate parse failures from template-copying." But `<1-6>` parses *accidentally* (the regex extracts `1` via `\w+` stopping at `-`), while `N` parses to the fallback chain. The "fix" moved the silent failure from accidentally-correct to systematically-broken. This is recurrence #6 of S110's pattern — at the version-control / commit-rationale boundary.

- **gemma4:e4b returns empty strings for game-style prompts**. On Thor, gemma4:e4b answers `"What is 2+2?"` with `"4"` but returns `""` for prompts containing `"1=UP 2=DOWN"`, `"ACTION=N"`, `"Game state: avatar at (5,5)"`, or any direction-key listing. The `parse_llm_response` fallback chain absorbs the empty response silently. The LLM contributes nothing; the dispatch logs read as if it did. Recurrence #7, this time at the model-output boundary.

Independent of S112. Two new recurrences. The pattern is now operational at every layer of the dispatch stack.

---

## Production data — what was actually found

`~/ai-workspace/shared-context/explorations/` contains 95 JSON files across 14 exploration tracks. 55 of them have valid `llm_responses` arrays. The schema is:

```json
{
  "llm_responses": [
    {"step": int, "action": int, "coords": null|{...}, "rationale": str, "latency_s": float, "prompt_tokens": int}
  ]
}
```

The `rationale` field is set to `"parse_failed: <text[:120]>"` when the parser falls all the way through, and to `"action_out_of_range: <action>"` when the parsed action is invalid. Both prefixes are distinguishable from real LLM rationale text.

### By-track parse-failure rates

| Track | Files | Invokes | PF% (explicit) | Empty% | OK/Garbage% |
|---|---:|---:|---:|---:|---:|
| `mcnugget-v2invoke-qwen35-2026-04-20` | 1 | 81 | 3.7% | 33.3% | 63.0% |
| `qwen-tiny-v2invoke-floor-2026-04-20` | 10 | 507 | 8.9% | 56.6% | 34.5% |
| `v2invoke-strategy-labels-2026-04-20` | 1 | 83 | 0.0% | 12.0% | 88.0% |
| `whole-brain-at-small-model-2026-04-22` | 43 | 6,871 | 11.6% | 43.9% | 44.5% |
| **Total** | **55** | **7,542** | **11.2%** | **44.3%** | **44.5%** |

Reproduce: `/tmp/s113/scan_parse_failures.py` reads all `*.json` under the explorations directory and emits per-track and per-file stats. Full per-file records: `/tmp/s113/parse_failure_scan.json`.

### Worst games

| Rate | Failed | N | Game | Path |
|---:|---:|---:|---|---|
| 29.8% | 594 | 1990 | lp85 | whole-brain/fat/lp85.json |
| 26.1% | 29 | 111 | s5i5 | whole-brain/fat/s5i5.json |
| 22.5% | 18 | 80 | su15 | whole-brain/fat/su15.json |
| 21.3% | 10 | 47 | r11l | whole-brain/fat/r11l.json |
| 19.5% | 16 | 82 | cd82 | qwen-tiny/cd82-gemma4e2b-v2invoke-500.json |

`lp85.json` alone has 594 silent-fallback events out of 1,990 dispatches — **almost a third of one full game's invokes silently substituted the NN hint for the LLM's intended action**.

### Sample parse_failed rationales

```
'parse_failed: ACTION=SELECT_NONE'
'parse_failed: ACTION:None'
'parse_failed: ACTION: $\\text{A0}$'
'parse_failed: ACTION:\n#5\n\nCONFIDENCE:\n0.91'
'parse_failed: ACTION: MOVE_TO_10_10\n'
'parse_failed: ACTION:\nprint("ACTION: .")'
'parse_failed: ACTION:\nprint("1.000")'
'parse_failed: ACTION: 오른쪽'   # Korean for 'right'
'parse_failed: ERROR: TimeoutError: timed out'
```

The LLMs invent novel action languages: LaTeX (`$\text{A0}$`), Python `print()` calls, JSON-style `ACTION:None`, multi-section structured output (`ACTION:\n#5\n\nCONFIDENCE:`), Korean direction words. Every one of these is silently absorbed into NN-fallback.

---

## Format comparison within a single exploration

`whole-brain-at-small-model-2026-04-22` is the cleanest natural experiment available. Same machine (CBP), same model (gemma4:e2b), same 25 games, same step budget — only the prompt format differs. Two arms: `data/fat/` (used `adaptive_prompt.py` format `ACTION=<1-6>[ X=<0-63> Y=<0-63>]`) and `data/lean/` (used `lean_dispatch.py` format `ACTION=<1-6>[ X=<0-63> Y=<0-63>]` — same shape, simpler context). Both formats predate the Apr 24 fix.

| Mode | Files | Invokes | PF% | OOR% | Empty% | Residue% | Clean% |
|---|---:|---:|---:|---:|---:|---:|---:|
| fat | 21 | 3,896 | 20.5% | 3.3% | 53.3% | 9.7% | 13.3% |
| lean | 22 | 2,975 | 0.1% | 1.2% | 31.6% | 0.0% | 67.2% |

**The 200× gap is misleading.** The lean form's 0.1% explicit-PF rate is a measurement artifact:

`_ACTION_RE = re.compile(r"ACTION\s*=\s*<?(\w+)>?", re.IGNORECASE)` matches `<1-6>` and captures `1` (because `\w+` doesn't match `-`, so it stops at the hyphen). When the LLM copies the template literally, the regex extracts the digit `1`, `int("1") = 1`, and **the parser returns action=1 (UP) silently with no flag**. The fat form's `ACTION=N X=x Y=y` produces no parseable digit — the regex captures `N`, `int("N")` raises, and the fallback chain fires (sometimes succeeding via naked-name rescue, often producing `parse_failed:`).

Verified empirically:

```
parser regex match table
  ACTION=<1-6>           → captured='1'   → action=1 silently
  ACTION=<6>             → captured='6'   → action=6 OK
  ACTION=N               → captured='N'   → fallback chain
  ACTION=N[ X=0 Y=0]     → captured='N'   → fallback chain
  ACTION=6               → captured='6'   → action=6 OK
  ACTION=LEFT            → captured='LEFT'→ name path OK
```

### Action distribution confirms the silent-default pattern

|  | fat (N format) | lean (`<1-6>` format) |
|---|---:|---:|
| action=1 (UP) | 7.7% | **26.3%** |
| action=2 (DOWN) | 23.4% | 22.1% |
| action=3 (LEFT) | 9.7% | 16.7% |
| action=4 (RIGHT) | 22.0% | 20.3% |
| action=5 (SEL) | 6.0% | 10.1% |
| action=6 (CLICK) | **29.9%** | 3.9% |

In fat, action=6 (CLICK) is over-represented at 29.9% — this is the NN-hint fallback firing (NN tends to default to CLICK on stuck-state escalation). In lean, action=1 (UP) is over-represented at 26.3% — this is the `<1-6>` template-extraction silently picking UP.

**Each format has a different "silent default", and in both cases the LLM's intent is replaced with that default in a substantial fraction of dispatches.**

### Rationale-vs-action mismatch as a direct silent-fallback signal

When the rationale's first direction word ≠ the returned action, the LLM was thinking about one direction but a different one was dispatched. This is exactly what the silent-fallback failure mode looks like at the dispatch boundary.

| Format | Non-failure rationales | With direction word | Match | **Mismatch** |
|---|---:|---:|---:|---:|
| lean (`<1-6>`) | 1,998 | 1,497 | 33.5% | **66.5%** |
| fat (`N`) | 896 | 247 | 91.1% | 8.9% |

Lean has 67% of its rationales contradicting the action returned. The high "clean rationale" rate masks the failure: the LLM produced apparently-fine prose, but it described a different action than the parser extracted.

Fat's 91% match rate is misleading too — most of its responses had no surviving rationale at all (the parser returned `parse_failed:` and overwrote rationale, leaving only 247 with parseable direction text out of 3,896 total invokes). When fat does parse cleanly, it parses cleanly; when it doesn't, the LLM's intent is gone.

Sample lean mismatches (rationale → action returned):

```
'Move the avatar down to try and reach the goal.'              → action=1 (UP)
'Moving down is the only way to potentially reach...'           → action=3 (LEFT)
'Moving UP seems to align with the hint and might progress...'  → action=3 (LEFT)
'Move the avatar up to the goal position to advance.'           → action=2 (DOWN)
'CLICK to move the crane and attempt to solve the puzzle.'      → action=5 (SEL)
```

These are not edge cases. They are 995 dispatches in one exploration where the LLM said one thing and the system did another, with no log, no flag, no signal that something is wrong.

---

## The Apr 24 fix was a regression

Commit `3f54ead56` (2026-04-24) bears the message:

> *gemma3:12b copies ACTION=<6> literally from template. Changed all ACTION=<1-6>[ X=<0-63> Y=<0-63>] to ACTION=N X=x Y=y format. Eliminates parse failures from template-copying behavior.*

The diagnosis was correct: gemma3:12b *does* copy templates literally. The fix was wrong: it replaced an accidentally-parseable form with a systematically-broken one.

| Form | LLM copies → | Parser extracts → | Outcome |
|---|---|---|---|
| `ACTION=<1-6>` | `ACTION=<1-6>` | `1` | action=1, silent default (looks like success) |
| `ACTION=<6>` | `ACTION=<6>` | `6` | action=6, correct |
| `ACTION=N` | `ACTION=N` | `N` | fallback chain → `parse_failed:` or naked-name rescue (often garbage) |
| `ACTION=N[ X=0 Y=0]` | `ACTION=N[ X=0 Y=0]` | `N` | fallback chain (S112's 94% finding) |

The pre-Apr-24 form *worked* in the sense of producing actions without `parse_failed:` markers. The post-Apr-24 form fails *visibly* (with the `parse_failed:` rationale) on tiny models that copy literally, and fails *invisibly* (via naked-name rescue) on small-medium models that produce diverse output. Both paths are silent at the dispatch level — the LLM's intent is replaced with the NN hint either way.

This is recurrence #6 of S110's silent-routing pattern, at a layer not previously enumerated: **the commit-rationale boundary**. The fix was reasoned about at the wrong level — "stop the parser from reporting parse_failed" vs. "ensure the LLM's intent reaches the action". The former was achieved (lean has 0.1% PF rate); the latter regressed (66.5% rationale-action mismatch).

The fact that this pattern can recur in *commits made specifically to fix the pattern* says something about its difficulty.

---

## Recurrence #7 — gemma4:e4b silently empties on game prompts

Probing model coverage to extend S112's matrix surfaced a new failure mode unrelated to format spec.

### What gemma4:e4b on Thor does

Tested against a range of prompts:

```
"What is the capital of France?"       → 'The capital of France is **Paris**.'
"What is 2+2?"                          → '4'
"Reply with ACTION=3"                   → 'ACTION=3'
"Game state: avatar at (5,5)."          → ''
"1=UP 2=DOWN 3=LEFT 4=RIGHT"            → ''
"Avatar at (5,5). Goal (8,8). Pick UP DOWN LEFT or RIGHT?" → ''
"Pick a number 1 to 4. Question: which way?" → ''
"Reply with ACTION=N"                   → ''
```

Verified at temperature 0.0, 0.7+seed=42, and 1.0 — same empty response. `eval_count: 50` with `done_reason: length` indicates the model *generates* tokens but they are all whitespace or stripped to nothing.

### What other models on Thor do with the same prompt

```
qwen2.5:3b   → "Given the goal of reaching position (8,8) from..."  (full reasoning)
phi4:14b     → "To move from your current position at (5,5)..."     (full reasoning)
gemma3:12b   → "Okay, I understand the game state.\n*Avatar..."     (full reasoning)
gemma4:e4b   → ''                                                    (silent)
```

The empty-response failure is gemma4:e4b-specific on this Thor instance.

### Where this lands in the dispatch path

`parse_llm_response("", fallback_action=NN_hint)`:
1. `_ACTION_RE.search("")` → no match → `action = -1`
2. `_NAMED_ACTION_RE.search("")` → no match
3. `_NAKED_ACTION_RE.search("")` → no match
4. `action < 0` → return `(fallback_action, fallback_coords, "parse_failed: ")`

The empty rationale prefix `"parse_failed: "` (with nothing after the colon) IS the signal — but only if anyone reads it. From a dispatch-result aggregate, this looks identical to the "Korean direction word" or "LaTeX A0" failures: the NN hint fired, no LLM contribution.

### Operational implication

If gemma4:e4b is the active raising or play model on any fleet machine, **the LLM is contributing zero value to game-state or game-play prompts** while still consuming wall-clock latency for each invoke. The dispatch path runs as designed — it just dispatches the NN hint instead of an LLM-decided action. This may already be happening on Legion (which has been running gemma4:e4b post-Apr-20). Worth a fleet check.

This is recurrence #7 of S110's pattern, at the **model-output boundary**. The model is technically responding (eval_count > 0) but emitting no usable content. The harness treats "ran the model" as "got an LLM contribution".

---

## Updated pattern recognition

| # | Source | Routing function | Silent default | Layer |
|---|---|---|---|---|
| 1 | S110 | `InstancePaths.resolve` | `_DEFAULT_MODELS.get(machine)` | Instance |
| 2 | S111 | `plan_executor._get_action_index` | `ACTION_MAP.get(do, 0)` → noop | Action dispatch |
| 3 | S111 | `motor_skills.registry.get_skill` | `SKILL_REGISTRY.get(skill_id)` → None | Skill registration |
| 4 | S111 | `plan_executor.execute_plan` | does not call `plan_bridge.step_to_invocation` | Composition |
| 5 | S112 | `parse_llm_response` | `fallback_action` (NN hint) | Response parse |
| 6 | **S113** | commit `3f54ead56` "fix" | replaced accidentally-parseable form with systematically-broken form | Commit rationale |
| 7 | **S113** | gemma4:e4b @ Thor on game prompts | empty output → fallback chain | Model output |

Seven instances across instance management, action dispatch, skill registration, plan composition, response parse, version-control reasoning, and model output. The shared property continues to hold: **a function that turns one kind of input into another, with a "default" branch for unrecognized input that returns plausibly-correct output without flagging the unfamiliarity**.

For #6: the "function" is the human + LLM collaborating on the commit. The "input" is "parse_failed observed". The "default" was "stop the parser from saying parse_failed" rather than "preserve LLM intent". The output (the commit) was plausibly-correct (PF rate did drop) without flagging that the new form's silent failure mode is worse.

For #7: the "function" is the model. The "input" is a game-state prompt. The "default" is empty output. The output is plausibly-correct (parser doesn't crash, dispatch returns an action) without flagging that the LLM contributed nothing.

The codebase isn't unique in having this pattern — humans and LLMs both produce it. The discipline that S111 sketched (`_route()` idiom for explicit silent-path observation) generalizes upward: **make the silent path observable at every level where information transforms**. Code, commit rationale, model output, all have the same shape.

---

## Carry-forward / proposals

All S110/S111/S112 items unchanged. Updates from S113:

### New empirical priors for S112's held proposals

S112's three proposals (replace placeholder format at 6 callsites; surface `parse_path` in `parse_llm_response`; aggregate `parse_failure_rate`) are now backed by:

- **11.2% live PF rate across 7,542 production invokes** — proposal 1 has empirical urgency at the small-model fleet
- **66.5% rationale-action mismatch** in lean exploration — proposal 2 is the only way to detect this from outside
- **lp85.json's 29.8% PF rate over 1,990 invokes** — proposal 3 would surface this game's degradation immediately at sweep-summary level

### New — surface model-output emptiness

Add a `model_output_empty: bool` flag to `llm_dispatch.invoke()` so that empty model responses are distinguishable from real-but-unparseable ones. Currently both fall through to the same `parse_failed: ` rationale (with empty body for the truly-empty case). At dispatch-aggregate level, `parse_failed` with empty body is recurrence #7's signature — a fleet scan for it would reveal which machine/model combinations are silently emitting nothing.

### New — fleet check for gemma4:e4b empty-response

Worth a 5-minute test on Legion + any other gemma4:e4b instance: send `"1=UP 2=DOWN 3=LEFT 4=RIGHT"` directly to Ollama and confirm response is non-empty. If Legion exhibits the same failure, all post-Apr-20 raising sessions on gemma4:e4b need re-evaluation — the LLM may have contributed nothing across many sessions.

### New — empirical question for S114+

The lean exploration's 66.5% rationale-action mismatch is the highest-signal direct probe of silent fallback I've seen. Apply the same diagnostic to:
- post-Apr-24 production runs (do they exist? `play_lean` with `--json-out` since Apr 24)
- Legion's recent raising sessions (different schema, but `chat_history.jsonl` contains LLM rationales)
- The new codification Layer 1 lean_prompt.py output (S112's experiment data, now reproducible)

If post-Apr-24 lean runs exist with `--json-out`, they would let us measure whether the regression hypothesis holds: same exploration, post-fix, should show *higher* mismatch rate even though `parse_failed` rate is lower.

---

## Files this session

- `sage/raising/analysis/s113_production_parse_failure_rate_20260426.md` — this analysis.
- `sage/docs/LATEST_STATUS.md` — entry for S113.

Reproducible scripts (uncommitted, in `/tmp/s113/`):
- `scan_parse_failures.py` — production log scanner
- `replicate_s112_qwen35_27b.py` — model-coverage extension to qwen3.5:27b
- `test_gemma4_e4b.py`, `gemma4_isolation.py`, `gemma4_check.py` — gemma4:e4b empty-response probes
- `parse_failure_scan.json` — full per-file scan results
- `qwen35_27b_format_test.json` — synthetic test on 27b

No code changes shipped. Findings extend S112's evidence base (now empirical, not just synthetic) and add two new instances of the silent-routing pattern. Held for operator alignment per S111 discipline.

---

## Meta

S112 said: *"The fix is mechanically tiny. The general lesson is mechanically expensive: make fallback paths observable at every routing boundary. Five instances, one principle."*

S113 finds the same principle one layer up — at the **commit-rationale** layer — and one layer down — at the **model-output** layer. The principle generalizes: any time information transforms (input → output, intent → action, observation → fix), the transform can take a silent path that produces plausibly-correct output without flagging the unfamiliarity. Code, commits, model weights — same shape, same blind spot.

The lean exploration's data was the highest-signal evidence I've seen for this in the codebase. 66.5% of LLM rationales contradicted the dispatched action, in production, on a model that costs real wall-clock latency per invoke. The pipeline's outcome metric (level progress, win rate) doesn't surface this — it surfaces only the aggregate of NN hints + occasional LLM-correct dispatches. The LLM contributed less than the headline numbers suggest, and the integration was treating its non-contribution as contribution.

Seven instances. One principle. The mechanical cost of the principle (instrumentation for silent-path observability) keeps growing relative to the cost of any individual fix.
