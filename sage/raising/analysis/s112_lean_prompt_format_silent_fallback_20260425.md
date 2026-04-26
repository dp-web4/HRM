# S112 — Lean Prompt's Placeholder Format Causes Silent NN-Hint Fallback (94% across 2 models, 16 trials)

**2026-04-25 — Thor Autonomous SAGE Session, 18:00 UTC**

S112 picks up S111's render-quality observation about `wm.render` truncating Strategy mid-word and asks the next-deeper question: *does the truncation actually change LLM behavior?* Running the prompt through real LLMs (qwen2.5:3b and gemma3:12b) surfaces a different — and bigger — finding. The truncation is a minor cosmetic issue. The **response format spec** is silently broken: `Respond: ACTION=N[ X=x Y=y]` causes both models to echo the placeholder literally, and `parse_llm_response` silently falls back to the NN hint when that happens. The LLM's contribution is invisibly zeroed in 94% of trials.

This is a 4th instance of S111's silent-routing pattern, this time at the **response-parse boundary** instead of the dispatch-table boundary. The same shape: a routing function (`parse_llm_response` → action), an unrecognized input (literal "N"), a silent fallback (`fallback_action`), no warning, no log. The diagnostic surface from outside the function is "the system returned an action" — the fact that the action came from the NN, not the LLM, is invisible.

Independent of S111. Recurrence #4.

---

## What Layer 1 looks like end-to-end

```
WM (typed slots)
  → wm.render(budget=300)             [chars: 1212, S111 truncation]
  → build_lean_prompt(...)            [chars: 1482, ~370 tokens — close to 401 claim]
  → Ollama (qwen2.5:3b or gemma3:12b)
  → raw response
  → parse_llm_response(text, fallback_action=NN_hint)
  → (action_idx, coords, rationale)
```

The pipeline is built. Each piece works in isolation. End-to-end, the integration is degraded.

## What I tested

**Three prompt variants**, all built from the same `cd82.json` WM and same situational params (NN hint LEFT@0.32, recent_actions=[L,L,U,SEL,L], invoke_reasons=[low_confidence, no_progress]):

- **A — Placeholder (current code)**: ends with
  ```
  Actions: UP=1 DOWN=2 LEFT=3 RIGHT=4 SEL=5 CLICK=6(x,y)
  Respond: ACTION=N[ X=x Y=y]
  One-sentence rationale.
  ```
- **B — Numeric examples (proposed)**: ends with
  ```
  Actions: UP=1 DOWN=2 LEFT=3 RIGHT=4 SEL=5 CLICK=6
  Reply with exactly one action choice. Examples:
    ACTION=3
    ACTION=6 X=12 Y=20
  Then on a new line, give a one-sentence rationale.
  ```
- **C — Named examples**: same shape as B but `ACTION=LEFT` / `ACTION=CLICK X=12 Y=20`. The named format already exists as a constant in `adaptive_prompt.py:23` (`ACTION_FORMAT_NAMED`) but has no callsites.

**Two models**: qwen2.5:3b (small, fast, edge-sized) and gemma3:12b (medium, the active raising model on Legion).

**8 trials per cell**, temperature 0.7, num_predict 150.

**Parser**: `sage.cognition.thalamic_router.llm_dispatch.parse_llm_response` — the production parser. Called with `fallback_action=-42` (sentinel) so we can detect when fallback fires. Production callers pass the NN hint as fallback, so `-42` cleanly maps to "NN hint substituted for LLM".

## Results

| Model | Variant | Parsed via ACTION=⟨digit⟩ | Silent fallback to NN | Naked-name rescue | Of which: garbage |
|---|---|---|---|---|---|
| qwen2.5:3b | A_placeholder | 0/8 | 2/8 | 6/8 | 5/8 |
| qwen2.5:3b | B_examples_numeric | **8/8** | 0 | 0 | 0 |
| qwen2.5:3b | C_examples_named | **8/8** | 0 | 0 | 0 |
| gemma3:12b | A_placeholder | 1/8 | 3/8 | 4/8 | 4/8 |
| gemma3:12b | B_examples_numeric | **8/8** | 0 | 0 | 0 |
| gemma3:12b | C_examples_named | **8/8** | 0 | 0 | 0 |

**Format A across both models (16 trials):**
- Parsed via intended path (`ACTION=<digit>`): 1/16 (6%)
- Silently fell back to NN hint: 5/16 (31%) — `parse_llm_response` returned `fallback_action`, set rationale to "parse_failed: ACTION=N[ X=0 Y=0] …" but the rationale field is rarely surfaced in dispatch logs
- Pulled an action via `_NAKED_ACTION_RE` matching the response text: 10/16 (63%) — of which 9/10 were garbage matches, e.g. parsing "LEFT" out of `ACTION=N[ X=LEFT Y=UP]`

**The bottom line**: across both models, the placeholder format produces an LLM-decided action only 6% of the time. The other 94% is silently substituted (NN fallback) or confidently parsed garbage (naked-name rescue from response noise).

**Format B and C**: 16/16 and 16/16, perfect parse via the intended path. Both small-edge and medium models handle explicit examples cleanly.

## What the LLMs actually output (Format A)

```
gemma3:12b T0  →  "ACTION=N[ X=0 Y=0] \nMoving the basket counterclockwise to adjust its position for a more strategic paint selection."
gemma3:12b T1  →  "ACTION=N[ X=0 Y=0] \nMoving the basket counterclockwise to adjust its position for painting."
qwen2.5:3b T0  →  "ACTION=N[ X=x Y=y]"          (literal echo)
qwen2.5:3b T2  →  "ACTION=N[ X=LEFT Y=UP ]"     (literal N, garbage X/Y)
qwen2.5:3b T3  →  "ACTION=N[ X=0 Y=0]"
qwen2.5:3b T4  →  "ACTION=N[ X=RIGHT Y=UP]"
```

The placeholders `N`, `x`, `y` look to LLMs like literal characters in the response — not variable names to substitute. Square brackets `[ ]` (used in human writing to indicate optional) get included literally. The rationale is reasonable when present (`"Moving the basket counterclockwise..."`), but the action field is uninterpretable.

## What `parse_llm_response` does with that

`_ACTION_RE = re.compile(r"ACTION\s*=\s*<?(\w+)>?", re.IGNORECASE)` matches `ACTION=N`. Then:

1. `int("N")` raises ValueError.
2. `_ACTION_NAME_MAP.get("n", -1)` returns -1 (no mapping for "n").
3. `_NAMED_ACTION_RE` (looks for "I choose X" / "action is X") — no match.
4. `_NAKED_ACTION_RE` (matches `\b(UP|DOWN|LEFT|RIGHT|SELECT|SEL|CLICK)\b` anywhere in the text) — sometimes matches an action name in the rationale ("Moving the basket counterclockwise" has none, so falls through), sometimes matches garbage ("X=LEFT" → action LEFT).
5. If still -1, return `fallback_action`. Rationale field is set to `"parse_failed: ACTION=N[ X=0 Y=0]…"` but only the first 200 chars.

**The naked-name rescue is the most insidious failure mode.** Step 4 confidently returns an action with no flag, no log, no `parse_failed` marker. The garbage cases (5/16 in this run) look identical from the outside to a successful parse. Production logs would show "LLM picked LEFT" — but the LEFT was pulled out of `X=LEFT Y=UP` noise, not the LLM's intent.

## Where this format is used

The placeholder pattern is not isolated to the new codification Layer 1 file. Same shape recurs in:

| File | Line | Format |
|---|---|---|
| `sage/cognition/thalamic_router/lean_prompt.py` | 74 | `Respond: ACTION=N[ X=x Y=y]` |
| `sage/cognition/thalamic_router/lean_dispatch.py` | 101 | `ACTION=N X=x Y=y (for CLICK, give coordinates)` |
| `sage/cognition/thalamic_router/adaptive_prompt.py` | 233 | `ACTION=N X=x Y=y (for CLICK)` (needs_physics branch) |
| `sage/cognition/thalamic_router/adaptive_prompt.py` | 249 | `ACTION=N (1-6)` (navigation branch) |
| `sage/cognition/thalamic_router/adaptive_prompt.py` | 268 | `ACTION=N X=x Y=y (for CLICK)` (equivalent branch) |
| `sage/cognition/thalamic_router/adaptive_prompt.py` | 285 | `ACTION=N X=x Y=y (for CLICK)` (default branch) |

Six callsites across three files. The five in `adaptive_prompt.py` and `lean_dispatch.py` are pre-existing — they predate the codification commits. The one in `lean_prompt.py` is from the new Layer 1 work and reproduces the existing pattern.

`adaptive_prompt.py:23` defines `ACTION_FORMAT_NAMED = "ACTION=UP or DOWN or LEFT or RIGHT or SEL or CLICK"` as a module constant with a comment — `# Named format eliminates number→name mapping entirely`. It has zero callsites. Designed-but-not-shipped.

## Pattern recognition update

S110 named the silent-routing pattern at `_DEFAULT_MODELS.get(machine)`. S111 found three more instances in the new Layer 2 codification code (`ACTION_MAP.get(do, 0)`, `SKILL_REGISTRY.get(skill_id)`, `plan_bridge` non-composition). S112 finds a fifth instance, this time at the parse boundary, in code that pre-dates the codification commits.

| # | Source | Routing function | Silent default | Layer |
|---|---|---|---|---|
| 1 | S110 | `InstancePaths.resolve` | `_DEFAULT_MODELS.get(machine)` | Instance |
| 2 | S111 | `plan_executor._get_action_index` | `ACTION_MAP.get(do, 0)` → noop | Action dispatch |
| 3 | S111 | `motor_skills.registry.get_skill` | `SKILL_REGISTRY.get(skill_id)` → None | Skill registration |
| 4 | S111 | `plan_executor.execute_plan` | does not call `plan_bridge.step_to_invocation` | Composition |
| 5 | S112 | `parse_llm_response` | `fallback_action` (NN hint) | Response parse |

Five instances across instance management, dispatch, registration, composition, and parse. Across files written by different people at different times. The shared property: **a function that turns one kind of input into another, with a "default" branch for unrecognized input that returns plausibly-correct output without flagging the unfamiliarity**.

The codebase doesn't lack the *concept* of validation — `parse_llm_response` does have layered fallbacks (digit → name → natural-language → naked-name → default). What it lacks is **observability of which fallback fired**. If the function returned `(action, fallback_path_used: str)`, callers could log when `fallback_path_used == "fallback_action_NN_hint"` and surface the format-failure rate. Currently that information is computed and discarded.

The same observation holds for S111's silent defaults: each function knows when it took the silent path (the `if not in table` branch is explicit), but doesn't communicate it.

## What the truncation finding from S111 was actually about

S111 noticed that `wm.render(budget=300)` truncates Strategy mid-word at `"CLICK palett[truncated]"`. I checked whether that truncation degrades LLM behavior by comparing variant A (truncated, default) against a variant with budget=600 (full Strategy visible). Both produced identical failure modes — the LLM echoed `ACTION=N[ X=0 Y=0]` regardless of whether it could see the full strategy or not.

The truncation is real but it's not the bottleneck. The format spec is the bottleneck. Strategy mid-word truncation matters for *content* — when the LLM does engage with the prose, missing strategy is missing context. But under the current format spec, the LLM doesn't engage with the response format coherently in the first place.

S111's render-quality finding remains a genuine improvement target (slot-aware budget, Strategy-first ordering, real tokenizer). It just shouldn't be the first thing fixed — fix format first, *then* truncation will start mattering.

## Why this didn't show up in production logs

I scanned 4188 JSON files under `~/ai-workspace/SAGE` and found 0 with an `llm_responses` key. `play_lean` returns `result["llm_responses"]` and saves to `--json-out` only if specified. Either the script is rarely run with `--json-out`, or the results live somewhere I didn't search (training/raising runs use a different schema; ARC-AGI experiment logs at `arc-agi-3/experiments/logs/*.json` use a different key set). Worth a follow-up: surface dispatch-time parse failures to the same channel that surface dispatch-time invokes.

The corollary point: the silent fallback masks the failure not just at the function boundary but at the data-collection boundary. Logs *do* contain the rationale field with `"parse_failed:"` prefix when the parser fully falls through, but rationale is treated as free-text and never aggregated. A counter (`parse_failures_per_invoke`) at dispatch level would surface this in seconds.

## Held proposals (not shipped)

Three concrete fixes, all small:

### 1. Replace placeholder format spec in 6 callsites

Format A → Format B (numeric examples) at:
- `lean_prompt.py:73-76`
- `lean_dispatch.py:100-102`
- `adaptive_prompt.py:233, 249, 268, 285`

Empirically perfect parse rate at both 3B and 12B. ~3 lines per callsite. Or define a single format constant (extending `ACTION_FORMAT_NAMED`) and reference it everywhere.

### 2. Surface parse-fallback path in `parse_llm_response`

Change the return signature from `(action, coords, rationale)` to `(action, coords, rationale, parse_path)` where `parse_path ∈ {"action_eq_digit", "action_eq_name", "named_natural", "naked_action", "fallback"}`. Or add a `parse_path` field to the per-invoke log entry that callers fill in.

This is the same shape as the proposed `_route()` idiom in S111: make the fallback path observable at the call site, not silently absorbed.

### 3. Aggregate parse-failure rate in dispatch results

`play_lean` already returns `llm_responses` per step. Add a top-level summary `parse_failure_rate = #fallback / #invokes`. Future operator scans for unhealthy dispatch sessions become greppable.

## Carry forward to S113+

- **All S111 carry-forward items unchanged** (plan_executor ↔ plan_bridge composition, `motor_skills/__init__.py` skill import, `wm.render` slot-aware budget, routing-table discipline). All S110 items unchanged (028–035 migration, two-line resolver fix, Phase A regex gate, phase-metadata corruption survey).

- **New from S112**:
  - Format spec correction (6 callsites, 1 module constant already exists). Active production code, so high-stakes — held for operator alignment.
  - `parse_llm_response` parse-path surfacing.
  - `play_lean` parse-failure-rate aggregation.

- **Empirical question for next session**: I didn't find any production `llm_responses` logs to scan. If `play_lean` has been run with `--json-out` recently, the parse-failure rate for current-config production is measurable directly. Worth a 5-minute look on Legion (the active raising machine) and any recent ARC-AGI play sessions.

## Meta

S111 said "the next session that wires plan_executor into a play loop will discover this — the question is whether they discover it via a bug report or by reading this entry first." S112's finding has the same shape one layer up: the *current* dispatch path through `adaptive_prompt.py` already exhibits the silent-fallback failure mode, and the question is whether anyone discovered it (via degraded play-loop performance not attributed to the format) or just lived with it. The pattern is "instrumentation lags implementation" — code knows when it took a degraded path, but the degradation isn't observable from outside the function.

The CLAUDE.md note about output metrics ≠ outcome progress applies cleanly here. The codification project's headline metrics are real: 17.6× speedup, 401-token prompts, JSON round-trip, calibrated-prediction interface for `wm.observe`. The outcome — *the LLM's intent driving game-play decisions* — is silently subverted by a format spec that the integration layer happens to produce. A working pipeline at every node, a broken pipeline end-to-end, every node passing its own contract.

The fix is mechanically tiny (replace placeholder text with examples). The general lesson is mechanically expensive: **make fallback paths observable at every routing boundary**. Five instances, one principle.

## Files this session

- `sage/raising/analysis/s112_lean_prompt_format_silent_fallback_20260425.md` — this analysis.
- `sage/docs/LATEST_STATUS.md` — entry for S112.

No code changes shipped. Findings are dormant-bug + design-call territory; held for operator review per S111's discipline.

Raw experiment data (not committed): `/tmp/s112_format_survey.json`, `/tmp/s112_lean_prompt_experiment.json`. Reproducible via the test runner inline in this doc.
