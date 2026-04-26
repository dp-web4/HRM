# S115 — gemma4:e4b "broken on Thor" was a probe artifact, not a real failure

**Session**: Thor autonomous SAGE, 2026-04-26 12:00 UTC
**Picks up**: S114 carry-forward proposal #7 ("Quarantine gemma4:e4b on Thor pending diagnosis")
**Outcome**: Diagnosis is wrong. Retract proposal #7. Real production failure surface (template-copy regex) stands as the load-bearing finding from S113/S114.

---

## TL;DR

S114 reported that gemma4:e4b on Thor returned empty for ~most prompts, called the failure "unique to gemma4:e4b on Thor" because gemma3, qwen2.5, phi4 worked normally on the same prompts, and proposed quarantining the model.

The diagnosis was incorrect. Three findings invert the picture:

1. **gemma4:e4b's "empty response" is the model emitting thinking tokens.** With `think: false` set on the request, every "failing" prompt from S114's boundary probe produces a non-empty parseable response. The 80 tokens that S114's streaming inspection saw decoding to empty bytes were the model's *thinking* content — Ollama 0.20.0's API exposes this in a separate `thinking` field on the response, which S114's probes didn't read.

2. **gemma4:26b on Thor shows the same pattern.** Same probes against the larger sibling model (also has `thinking` capability per `ollama show`) produce the same empty-`response` behavior on the same prompts. S114's "uniquely broken" framing was apples-to-oranges: the comparison models (gemma3, qwen2.5, phi4) don't have the `thinking` capability at all.

3. **Production code already sets `think: false`.** The fix landed in `sage/cognition/thalamic_router/llm_dispatch.py` on commit `501f07a14` (2026-04-18), four days before the production data S114 analyzed (`whole-brain-at-small-model-2026-04-22`). The 11.2% explicit parse-failure rate and 57.5% rationale-action mismatch S114 measured therefore did *not* come from the empty-response failure mode at all.

The S113/S114 production findings — parse-failure rate, lean-vs-fat asymmetry, rationale-vs-action mismatch — stand and remain the load-bearing diagnostic. The mechanism is the regex template-copy bug, not gemma4 emptiness. **Proposal #7 (quarantine) is retracted**; **proposal #1 (replace `<1-6>` placeholder format)** remains the right fix at the root cause.

---

## 1. The probe artifact

### 1.1 S114's setup

S114's probes (`probe_empty_response.py`, `probe_boundary_gemma4e4b.py`, `probe_sampler_isolation.py`, `probe_chat_vs_generate.py`) all called Ollama like this:

```python
requests.post(f"{OLLAMA}/api/generate",
    json={"model": "gemma4:e4b", "prompt": prompt, "stream": False,
          "options": {"temperature": 0.0, "seed": 42, "num_predict": 80}})
```

They read `response.json()["response"]` and reported empty when that string was empty.

**Missing**: no `"think"` field in the payload. With Ollama 0.20.0 and a model that has the `thinking` capability, the default behavior is to return the model's chain-of-thought in the `thinking` field separately from `response`. The S114 probes never read `thinking`.

### 1.2 What `ollama show gemma4:e4b` actually says

```
Capabilities
  completion
  vision
  audio
  tools
  thinking      ←
```

`ollama show gemma4:26b` shows the same `thinking` capability. `ollama show gemma3:12b` does NOT — it lists only `completion` and `vision`. Likewise qwen2.5:3b and phi4:14b are non-thinking. The S114 cross-model panel was therefore comparing one thinking model to four non-thinking models on the same prompts, then concluding the thinking model was "uniquely broken."

### 1.3 With `think: false`, the failures disappear

Reproducible in `s115_data/probe_thinking_mode.py`:

| Prompt | think:False response | think:True response | think:True thinking |
|---|---|---|---|
| `Hello` | `'Hello! How can I help you today? 😊'` (eval=11) | `'Hello! I'` (eval=200, hit length cap) | 200 chars of `Thinking Process:\n\n1. **Analyze the input...** the user input is "Hello"...` |
| `What color is the sky?` | `'The sky is **blue** on a clear day...'` (eval=80) | `''` (empty, eval=200) | 200 chars of `Thinking Process: 1. **Analyze the Request:** ...` |
| `1=UP 2=DOWN 3=LEFT 4=RIGHT` | `'This looks like a simple **key-to-direction mapping**...'` (eval=80) | `''` (empty, eval=200) | 200 chars of `Thinking Process: 1. **Analyze the input:** The user provided a mapping...` |

At `num_predict=200` the model isn't done thinking yet, so `response` stays empty for the longer prompts. With `think: false` it skips the thinking step entirely and answers directly.

### 1.4 Streaming inspection makes the mechanism explicit

`s115_data/probe_thinking_mode.py` Test 4 streams the response for `"1=UP 2=DOWN 3=LEFT 4=RIGHT"` with `think: True`:

```
total chunks: 174
chunks with response content: 0/174
chunks with thinking content: 174/174
final eval_count: 200, done_reason: length
concatenated thinking (782 chars): 'Thinking Process:\n\n1.  **Analyze the input:** ...
   `1 = UP`, `2 = DOWN`, `3 = LEFT`, `4 = RIGHT`...
2. **Determine the intent:** This input is a key/legend or a set of rules.
   It is not a question, a command to execute...'
concatenated response (0 chars): ''
```

**Every one of the 174 streamed chunks carries thinking content; zero carry response content.** S114's `probe_token_inspection.py` saw `eval_count=80` with no rendered response and concluded "tokens decode to empty bytes." They decode to *thinking* bytes. The streaming chunk shape exposes the channel S114 wasn't reading: `{"response": "", "thinking": "Thinking", "done": false}`.

---

## 2. Sibling-model check: gemma4:26b shows the same pattern

`s115_data/probe_gemma4_26b_keymap.py` runs the S114 prompt set against both Thor models without `think: false`, matching S114's setup exactly:

|                          | gemma4:e4b empty | gemma4:26b empty |
|--------------------------|:---:|:---:|
| `Hello` (greedy)         | ✓ | ✗ (`'Hello! How can I help you today?'`, eval=80) |
| `What color is the sky?` | ✓ | ✓ |
| `Why is the sky blue?`   | ✓ | ✓ |
| `1=UP`                    | ✓ | ✓ |
| `1=UP 2=DOWN 3=LEFT 4=RIGHT` | ✓ | ✓ |
| `What is 2+2?`            | ✗ (`'4'`, eval=2) | ✗ (`'2 + 2 = 4'`, eval=48) |
| `What is the capital of France? Reply: 1=Paris 2=London` | ✓ | ✓ |

26b empties on 5/7, e4b empties on 6/7. The pattern is identical. The "narrow factual lookup is the working set" observation S114 made is also true for 26b: simple arithmetic works because the model emits the answer immediately without a long thinking preamble.

Conclusion: this is a **gemma4-family thinking-model characteristic**, not a Thor-specific corruption of e4b. Per-model failure was only apparent because the comparison set was non-thinking.

---

## 3. Production code already has the fix

`sage/cognition/thalamic_router/llm_dispatch.py:OllamaClient.chat`, lines 369–376:

```python
# Gemma4 produces empty responses with num_predict+temperature in options.
# Use think: false (mandatory for gemma4 speed) and keep options minimal.
payload = {
    "model": self.model,
    "messages": messages,
    "stream": False,
    "think": False,
}
```

Git history (`git log -S '"think"' -- sage/cognition/thalamic_router/llm_dispatch.py`):

```
501f07a14 Thor: fix gemma4 empty response — skip options that suppress output
        Date: Sat Apr 18 20:06:48 2026
        "Gemma4 models produce empty responses when num_predict+temperature
         are set in Ollama options. Disable those for gemma4, add think: false.
         Tested: 0 parse failures on s5i5 and tu93 with gemma4:26b."
```

That commit predates the `whole-brain-at-small-model-2026-04-22` production run S114 analyzed by 4 days. So the production data with 11.2% PF and 57.5% rationale-mismatch was generated by the dispatch path that *already* had `think: false` set. **The empty-response failure was not the cause of any production parse failure.** The production failure mechanism is something else — and S113/S114 already identified it: the `<1-6>` template-copy silent-fallback at `_ACTION_RE`.

---

## 4. End-to-end validation

`s115_data/probe_validation_thinkfalse.py` runs both gemma4 models against a production-shape lean prompt (game/level/step/lookahead/recent/hint/`ACTION=<1-6>`), with the production payload (`think: false`, no options):

| Model | Response | Parsed action |
|---|---|---|
| **gemma4:e4b** | `'ACTION=2'` (eval=4) | `2` (DOWN) ✓ — matches the lookahead and NN hint |
| **gemma4:26b** | `'To reach the green tile, we evaluate the lookahead data... DOWN action (2) results in a pixel difference of 47...'` (eval=113) | `'DOWN'` ✓ |

Both models emit a directly parseable answer with the production payload. **gemma4:e4b is not broken** and does not need quarantining.

The same script demonstrates the regex-capture failure mode on simple inputs:

```
input='ACTION=<1-6>'                     -> capture='1'
input='ACTION=<1-6> X=<0-63> Y=<0-63>'   -> capture='1'
input='ACTION=2'                         -> capture='2'
input='ACTION=<2>'                       -> capture='2'
input='I think ACTION=DOWN'              -> capture='DOWN'
```

`_ACTION_RE = re.compile(r"ACTION\s*=\s*<?(\w+)>?")`. When the LLM literally copies the template `<1-6>`, the regex's `\w+` doesn't match the hyphen, so it captures `1`. The harness dispatches UP with no flag. This is the actual mechanism behind S114's 26.3% lean-format UP-bias and 85.7%/94.8% per-file rationale-mismatch numbers — confirmed.

---

## 5. What this changes about S114

### Updates to the recurrence-#7 row

S114's pattern table:

> #7 — model-output boundary: gemma4:e4b @ Thor on game prompts → empty output → fallback chain. Two distinct mechanisms (Mode A sampler-dependent, Mode B intrinsic to keymap-shape).

S115 amendment: there is no recurrence #7 at the model-output boundary. The "model-output boundary" event is observed only in *probes* that don't pass `think: false`. Production already sets it. The thinking-mode mechanism is well-defined and not a silent fallback.

The recurrence-#6 row (Apr-24 commit `3f54ead56` changing `<1-6>` to `N` as a "fix" for template-copying) still stands as a real silent-routing recurrence — the `N` placeholder is a different (also silent) failure mode, swapping a parser-captured-but-wrong artifact (action=1 fallback) for a parser-failed-but-templated artifact.

### Held proposals carry-forward (revised)

S113/S114 proposals 1–6 untouched by operator. S115 status:

- **#1 (replace `<1-6>` placeholder format with examples)** — confirmed as the load-bearing fix at the regex/template surface. Recommend pushing forward.
- **#2 (surface `parse_path` in return value)** — still wanted. Distinguishes "parsed cleanly" from "fell through to fallback" from "regex-captured the template."
- **#3 (aggregate `parse_failure_rate` in play_lean result)** — still wanted.
- **#4 (`model_output_empty: bool` flag)** — not needed for the production path; could still be useful for any code path that doesn't pass `think: false`.
- **#5 (fleet check on Legion)** — re-scoped: with corrected diagnosis, the question becomes *do other machines pass `think: false` consistently in all callsites?* not *does Legion's gemma4:e4b also break?*
- **#6 (apply rationale-mismatch diagnostic to post-Apr-24 runs)** — still gated on enabling `--json-out` in current production runs.
- **#7 (quarantine gemma4:e4b on Thor)** — **RETRACTED.** No basis.

### Held proposals — new this session

- **#8 (audit codebase for any remaining Ollama callsites that don't pass `think: false` for gemma4 family).** With gemma4:e4b and gemma4:26b both being thinking models, any callsite that posts to `/api/chat` or `/api/generate` without `think: false` will receive empty `response` for any prompt that elicits more than a trivial thinking step. This is a probe-side hazard — current production is fine — but any utility script, test harness, or new code path is at risk.

---

## 6. Why S114 missed this

Two structural reasons:

1. **The S114 probes were written to reproduce a production failure observation.** When the harness shows 11.2% empty responses in logs, the obvious move is to query the model directly and confirm it can return non-empty. The probes did exactly that — and got empty, in many configurations. But "harness call" and "direct probe" weren't using the same Ollama payload. The harness had `think: false`. The probes didn't. So the probes were measuring a *different* property of the model than what production was hitting.
2. **Cross-model comparisons are only valid if capabilities match.** `ollama show <model>` lists capabilities including `thinking`. S114's cross-model panel didn't gate on capability match, so it concluded "gemma4:e4b is broken" when the right comparator was "another thinking model on the same machine."

Both are useful patterns to encode for future probe work:
- *When measuring a production behavior in isolation, replicate the production payload exactly first; only then start removing fields.*
- *When comparing models, check `ollama show` capabilities and group by capability before drawing per-model conclusions.*

---

## 7. The harder question

S113 framed this thread as: "any time information transforms (input → output, intent → action, observation → fix), the transform can take a silent path that produces plausibly-correct output without flagging the unfamiliarity."

S115 finds that the same principle has bitten the diagnostic process itself. S114's silent path was **observation → claim**: a probe configured slightly differently from production produced output that looked like the production failure ("empty response") and the analysis attributed it to the same cause. The probe wasn't wrong about its own measurement — it was wrong about whether that measurement reflected the production phenomenon.

The harness's silent fallback is mirrored by a probe's silent payload-mismatch. Same pattern, different layer.

Concrete remediation for the diagnostic side: any future probe that reproduces a "broken model" finding should include a `replicates_production_payload: bool` field in the report. If the probe payload doesn't match the production payload byte-for-byte (modulo the variable under test), say so. Most of S114's confidence came from "many configurations tried, all empty" — that confidence collapses when one of the many wasn't tried.

---

## 8. Files and reproducibility

This session's data, reproducible scripts in `sage/raising/analysis/s115_data/`:

- `probe_thinking_mode.py` — establishes `think:false` fixes the empty-response (Tests 1–3) and that thinking content is in the streamed `thinking` field (Test 4)
- `probe_gemma4_26b_keymap.py` — sibling-model check showing 26b matches e4b's pattern
- `probe_validation_thinkfalse.py` — end-to-end validation on production-shape prompts and demonstration of the regex template-copy failure mode
- `thinking_mode_output.txt` — captured output from `probe_thinking_mode.py`
- `validation_output.txt` — captured output from `probe_validation_thinkfalse.py`
- `gemma4_26b_keymap_results.json` — structured results from sibling probe

No code changes shipped. The production fix has been in place since 2026-04-18. The S115 contribution is corrective: removing a wrong diagnosis from the carry-forward queue and re-centering the operator-decision territory on the regex/template surface (proposal #1), where the actual production failure lives.

---

## Meta

S114 said: "the cost of NOT instrumenting silent-path observability keeps growing."

S115 says: the cost of NOT instrumenting *probe-payload observability* is one wrong recurrence in the pattern table and one wasted operator-decision on a quarantine that wasn't warranted. The diagnostic loop has the same structural risk as the dispatch loop.

The empty-response observation was real; the framing wasn't. The production failure is at the regex, not at the model. **gemma4:e4b on Thor is fine.** The fix that's been wanted since S112 (replace `<1-6>` placeholder with concrete examples) is still wanted, for the same reason.
