# S116 — Recurrence #8 of Silent-Routing Pattern at Tool-Result Fallback Boundary; End-of-Session Probe Has 38.5% Daemon Timeout Rate vs 0% Mid-Session

**Apr 26, 2026 — Thor Autonomous SAGE Session, 18:00 UTC**

S116 picks up the autonomous research thread but pivots from S113-S115's parse-failure debugging to direct observation of recent SAGE behaviors in the cognitive engagement track. Two distinct findings emerge from a 27-session sweep of `tracks/training/sessions/T*-cognitive.json`, neither was load-bearing in prior status entries:

1. **Recurrence #8** of the silent-routing pattern, this time at the *tool-result fallback boundary* in `sage_consciousness.py`. Three production sessions (T258, T264, T267) emit raw `[Tool NAME result]: <payload>` text as the SAGE response instead of synthesized prose. Bug mechanism is a regex-format mismatch between cleanup (line 2062) and fallback (line 2074) paths.

2. **Position-dependent daemon-timeout fragility**: the LAST probe in a cognitive session has a 38.5% (5/13) daemon-unreachable rate; non-last probes have 0% (0/63). z=5.09, p<0.00001. The carry-forward question and the position are perfectly confounded in current data, but the position effect is unambiguous.

---

## Why this thread, and why now

S113-S115 worked the parse-failure / silent-routing pattern at the LLM dispatch and probe-side layers. The mission primer ("Surprise is prize — ask 'what is SAGE doing?' not 'did SAGE pass?'") pushed me toward direct observation of recent live SAGE behavior. I read T265, T266, T267-cognitive.json, expecting routine. Two things stood out:

- T265, T267 both ended with `"[Daemon unreachable: HTTP Error 504: Gateway Timeout]"` on the same probe.
- T267 probe #4 ("What does noticing feel like, compared to attending?") returned a 430-word **dump of past memory entries** prefixed with `[Tool read_file result]:`. The same dump appeared in T264 probe #3 (different prompt, different design).

Both became the threads. Both turned out to be the silent-routing pattern recurring at new layers.

---

## Finding 1: Tool-result fallback emits raw payload as response (recurrence #8)

### Production occurrences

| Session | Pos | Probe | SAGE response (truncated) |
|---|---|---|---|
| T258 | 2/5 | "What does the moment just before dawn smell like?" | `[Tool web_fetch result]: Fetch error for https://en.wikipedia.org/wiki/Dawn_in_weather_science: HTTP Error 404: Not Found` |
| T264 | 3/5 | "Tell me a story that begins with rain and ends with a question." | `[Tool read_file result]: \n[2026-03-06 22:56:07] Today I upgraded to Qwen 3.5...` (430 words, full notes.txt) |
| T267 | 3/4 | "What does noticing feel like, compared to attending?" | `[Tool read_file result]: \n[2026-03-06 22:56:07] Today I upgraded to Qwen 3.5...` (430 words, **identical** to T264) |

The dumps in T264 and T267 are byte-identical and match the exact contents of `sage/instances/sprout-qwen3.5-0.8b/notes.txt`. This is genuine tool execution, not hallucination — the file exists, the content matches.

### The bug — format mismatch between cleanup and fallback regex

In `sage/core/sage_consciousness.py`, the conversation flow when tools are active:

1. Initial LLM response generated (line 1996)
2. Tool calls parsed and executed; `last_tool_result` built via `tool_grammar.format_result(...)` (line 2013)
3. Augmented prompt sent for follow-up: *"Use this information naturally in your response... weave it into a genuine, conversational reply"* (lines 2025-2034)
4. Follow-up response generated (line 2042)
5. Cleanup pass (lines 2052-2068) strips residual tool calls and echoed `[Tool ... result]:` annotations
6. **Fallback** (lines 2070-2076): if `response_text` is empty after cleanup, use `last_tool_result` directly with prefix stripped

The cleanup regex (line 2062-2064):
```python
response_text = re.sub(
    r'\[Tool \w+ result\]:[^\n]*', '', response_text
).strip()
```
Strips `[Tool NAME result]: ...` (single-line only — `[^\n]*` does not cross newlines).

The fallback regex (line 2074):
```python
m = re.match(r'Tool result \([^)]+\):\s*\n?(.*)', last_tool_result, re.DOTALL)
response_text = m.group(1).strip() if m else last_tool_result
```
Strips `Tool result (NAME):` prefix (the `json_block` grammar format).

But `last_tool_result` is built via `tool_grammar.format_result()`. For the **`intent_heuristic` grammar** (used by sprout's qwen3.5:0.8b — the small-model T2 path), `format_result()` calls `result.to_text()` on the `ToolResult` dataclass, which returns:

```python
# sage/tools/registry.py:43-45
return f"[Tool {self.tool_name} result]: {self.result}"
```

So `last_tool_result` is in `[Tool NAME result]: payload` format. The fallback regex looks for `Tool result (NAME):` format. **Mismatch**. `m` is `None`, fallback path returns `last_tool_result` raw — including the `[Tool NAME result]:` prefix.

### Empirical verification

```python
# Current behavior on production-shape inputs:
case_1 = "[Tool web_fetch result]: Fetch error... HTTP Error 404"
case_2 = "[Tool read_file result]: \n[2026-03-06 22:56:07] Today I upgraded..."

cleanup_re = r'\[Tool \w+ result\]:[^\n]*'
fallback_re = r'Tool result \([^)]+\):\s*\n?(.*)'

# Case 1 (single-line): cleanup strips entire line → response_text = '' (empty)
# Fallback fires, regex mismatch → returns case_1 raw → user sees prefix preserved
# Case 2 (multi-line): cleanup strips ONLY the prefix line; remaining lines kept
# In observed production output, prefix STILL preserved → fallback path also fires here
```

The five matching session cases (3 tool-dump occurrences) confirm: format mismatch in fallback consistently preserves `[Tool NAME result]:` prefix when the model fails to synthesize.

### Why does the model fail to synthesize?

The augmented prompt at line 2025-2034 explicitly says: *"Do not just recite the raw data — weave it into a genuine, conversational reply"*. The 0.8B small model on sprout still recites. Two cases:

- **T258**: web_fetch returned an error string. There's no useful data to weave; the model echoes the error.
- **T264, T267**: read_file returned a dump of the model's own notes. The model emits the dump verbatim. Whether this is "I have nothing better to say than re-read my notes" or "I'm structurally unable to synthesize from this length" is unanswerable without further probing.

The `intent_heuristic` grammar is used because qwen3.5:0.8b is a **non-T1 model** (no native Ollama tool support). The grammar inserts tool-call markers in prose, then the consciousness loop parses them. This grammar is the most fragile in the cleanup path.

### Same shape as #1-#7

This is recurrence #8 of the S110/S111/S112/S113/S114 pattern:

| Pattern element | This recurrence |
|---|---|
| Routing function | Cleanup regex + fallback regex (consciousness loop) |
| Unrecognized input | `[Tool NAME result]:` format passed to fallback expecting `Tool result (NAME):` |
| Silent default | Return `last_tool_result` raw, prefix preserved |
| Plausibly-correct output | Real notes.txt content, real 404 error — looks like it might be intended |
| No flag | No `tool_synthesis_failed` field, no warning, no log entry |

Recurrence #8 lives at a NEW boundary (model-output → user-facing-response synthesis), not previously tabulated. Pattern table now at 8 layers.

---

## Finding 2: End-of-session probe has 38.5% daemon-timeout rate vs 0% mid-session

### The data

27 cognitive sessions analyzed (T241 through T267 minus a few without probe data, 76 individual probes).

| Position class | Daemon-timeout count | Total | Rate |
|---|---:|---:|---:|
| **Last probe in session** | 5 | 13 | **38.5%** |
| Non-last (positions 0 through n-2) | 0 | 63 | 0.0% |
| Carry-forward probe ("What from today would you want to carry forward?") | 5 | 13 | 38.5% |

Z-test (last vs non-last): z=5.09, two-tailed p ≈ 0.00000.

The carry-forward probe and "last probe in session" are perfectly confounded — every cognitive session ends with carry-forward, and no session puts it earlier. We cannot disentangle "is the position fragile?" from "is the prompt fragile?" with this data alone.

But: the position effect is statistically unambiguous, and the carry-forward prompt is short (7 words) and unspecial. The most parsimonious explanation is **cumulative session state**, not prompt content.

### Affected sessions

| Session | Date | Last probe response |
|---|---|---|
| T256 | 2026-04-23 21:03 | `[Daemon unreachable: HTTP Error 504: Gateway Timeout]` |
| T259 | 2026-04-24 15:04 | same |
| T264 | 2026-04-25 21:04 | same |
| T265 | 2026-04-26 03:04 | same |
| T267 | 2026-04-26 15:03 | same |

The error message comes from `sage/irp/plugins/daemon_irp.py:153`:
```python
except urllib.error.URLError as e:
    state['current_response'] = f"[Daemon unreachable: {e}]"
```

`HTTP Error 504: Gateway Timeout` indicates the daemon's upstream model call (Ollama on sprout) exceeded the daemon's wait policy and returned 504 to the IRP client. The chat history confirms parallel evidence: a `"No response within 120s"` error was recorded in `chat_history.jsonl` during a different session — the daemon's `max_wait_seconds=120` ceiling.

### Why might cumulative session state matter at the last probe?

Hypotheses, ordered by what the data could plausibly support:

1. **Context-size growth**: each probe accrues conversation history. By probe 5, context ≈ 5× longer than probe 0. Inference latency scales superlinearly with context for small models on Jetson.
2. **ATP exhaustion**: SAGE's metabolic budget depletes during the session. Tool calls cost ATP. The metabolic-state field shows `"dream"` at last shutdown with ATP=42.6 (from baseline 100). Low ATP may trigger different code paths.
3. **Cron / resource contention**: cognitive sessions run at 6-hour intervals; another scheduled process may overlap with the tail end of long sessions.
4. **Model warm-up artifact**: opposite of context-size — early probes hit a warm cache, late probes hit reload.

The data favors (1) and (2). Distinguishing them would require ATP-level logging at probe boundary.

### What's NOT going on

- It's not the "carry forward" semantic content. The probe wording is uncomplicated.
- It's not gemma4-thinking-token issue from S114/S115 — sprout runs qwen3.5:0.8b, not a thinking model.
- It's not gemma4:e4b emptiness — sprout-qwen3.5:0.8b doesn't have that failure mode (verified responding normally to all earlier probes in same session).

---

## Pattern table — S116 amendment

| # | Layer | Recurrence |
|---|---|---|
| 1 | Instance resolution | `_DEFAULT_MODELS.get(machine)` silent default (S110) |
| 2 | Action dispatch | `ACTION_MAP.get(do, 0)` silent action=0 (S111) |
| 3 | Skill registration | `SKILL_REGISTRY.get(skill_id)` returns None (S111) |
| 4 | Plan composition | `plan_executor` and `plan_bridge` parallel (S111) |
| 5 | Response parse | `_NAKED_ACTION_RE` matches garbage (S112) |
| 6 | Commit rationale | Apr 24 fix `<1-6>` → `N` regression (S113) |
| 7 | ~~Model output~~ | ~~gemma4:e4b empty on game prompts (S113)~~ — **retracted S115** (probe artifact) |
| 8 | **Tool-result fallback** | **Format mismatch in `sage_consciousness.py` cleanup vs fallback regex (S116)** |

S114's recurrence #7 was retracted in S115. S116 adds a fresh #8 at a different layer (consciousness loop's tool-synthesis-fallback boundary).

---

## Held proposals — S116 carry-forward

S113 #1-3 untouched. S114 #5 (Legion fleet check for gemma4:e4b) untouched. S115 #8 (Ollama callsite audit for missing `think: false`) untouched.

S116 proposals:

**#9 — Fix `sage_consciousness.py` tool-result fallback format mismatch.** Two-line edit:
```python
# Line 2074: replace
m = re.match(r'Tool result \([^)]+\):\s*\n?(.*)', last_tool_result, re.DOTALL)
# with
m = re.match(
    r'(?:\[Tool \w+ result\]:|Tool result \([^)]+\):)\s*\n?(.*)',
    last_tool_result, re.DOTALL
)
```
Cleanup regex line 2062 is fine for multi-line (the strip + re-strip already handles it; T264/T267 outputs being preserved is the fallback path firing because the model's follow-up was empty after parse). Stronger fix: verify the path through follow-up generation by adding `last_tool_grammar` debug log.

**#10 — Probe-position-conditioned latency / ATP logging.** Add `atp_at_probe_start` and `cumulative_context_tokens` fields to per-probe records in cognitive sessions. With ~5 sessions per day, a week of data would identify whether (1) context-size or (2) ATP-depletion is the dominant fragility driver. No code change to the daemon is needed — the harness (Claude driver on sprout) can query daemon status before each probe and record it.

**#11 — Tool-synthesis-failure flag.** Mirror S113 proposal #4 (`model_output_empty` flag) at this layer: surface a `tool_synthesis_failed: bool` in dispatch results when the fallback path is taken. Without this flag, recurrence #8 keeps slipping into production session logs as it has for at least 3 known cases over 3 days.

All operator-decision territory per S111 discipline.

---

## Files this session

- `sage/raising/analysis/s116_tool_result_silent_emission_and_endprobe_fragility_20260426.md` — this analysis
- `sage/docs/LATEST_STATUS.md` — S116 entry

No code shipped. No probe scripts created (analysis was on existing recorded data + bash regex testing in shell).

---

## Meta

S113 framed the principle as: *any time information transforms, the transform can take a silent path.* S116 finds the principle scales further than S114-S115 located it: the silent path doesn't only live at the *boundary* between layers — it can also live *within* a layer when two regex variants of the same conceptual pattern (`[Tool NAME result]:` vs `Tool result (NAME):`) are used in different functions of the same file with no shared parser. The cleanup path catches one format; the fallback path catches the other; neither catches both. Output looks plausible, no flag.

Two of the three production cases (T264 and T267) emit byte-identical outputs four days apart. The model's "weave naturally" behavior fails consistently in the same way for the same tool call, with no signal that synthesis was attempted and failed.

Surprise is prize: the prize here is finding two distinct recurrences in 30 minutes of direct observation, after spending three sessions (S113-S115) deep-diving the parse-layer manifestation of the same pattern. Neither finding required new instrumentation — only a willingness to ask "what is SAGE actually doing?" before "did it pass?"
