# S128 — `intent_heuristic` Grammar Audit (Carry-Forward from S127 #49)

**Apr 29, 2026 — Thor Autonomous SAGE Session, ~18:00 UTC**

S127 closed at ~09:55 UTC ratifying held proposal #47 (S116's two-line regex fix).
At 10:37 PT (~17:37 UTC), Dennis shipped commit `13de1ba4e` —
`sage_consciousness: fix tool-result fallback regex grammar mismatch (S116 #9 / S127 #47)`.
**The fix was committed between S127's session_end push and this session's start.**

S127 also surfaced **#48** (cognitive→training track-parity audit) and **#49**
(`intent_heuristic` grammar audit parallel to S125–S126's discipline on `_DISCLAIM_RE`).
**#48 is operator-decision territory** (curriculum-load-bearing). **#49 is audit-only
work** that the carry-forward from S125→S126→S127 makes due. S128 picks up #49.

---

## Headline finding

`intent_heuristic.parse_response` has an **11.7% any-pattern match rate and 7.9%
would-route rate over a fleet corpus of 5,361 SAGE responses (866 sessions, 8 active
instances)**. A hand-classified stratified sample of **46 routed responses shows 0/46
true positives** — every match is metaphor, copular use, reflective language, or
identifier coincidence. **The bug-shape S125 found in C5's `_DISCLAIM_RE` precedence
chain and S126 found in the alternation set repeats one layer down at the
grammar/tool-routing layer**, and at substantially higher prevalence than either.

The S127 #47 fix (committed today) eliminates the *visible* prefix-leak symptom but
**leaves the false-positive routing untouched**. Post-fix, the same false positives
still trigger tool execution; the tool result is now stripped of its prefix and
emitted as the SAGE response, indistinguishable from a synthesized reply. This is a
**layer-up of S127's track-asymmetric finding**: symptom-removal mitigations create
false-negative tracking signal at the symptom layer.

## A new finding S127 didn't see

`run_session_identity_anchored.py:243-251` **hardcodes `intent_heuristic` for
`tool_stage` of 'silent' AND 'aware'**. Only `tool_stage='active'` uses the detected
per-model grammar. Every instance in early tool-curriculum stages runs intent_heuristic
*regardless* of model native capability. T278's recurrence is a function of
`(tool_stage, response_text)`, not `(model_id alone)`. The audit applies fleet-wide
to anything in the silent/aware stages, not just sprout-qwen3.5-0.8b.

---

## Method

Same shape as S125b/S126: static audit + dynamic full-corpus pass + qualitative
classification.

1. **Static audit (S128 pattern inventory)**: list each regex in
   `sage/tools/grammars/intent_heuristic.py`, classify by intended function, find
   function-overlapping pairs (S126 audit primitive).
2. **Dynamic audit (S128a)**: 866 sessions / 5,361 SAGE responses /
   `instances/*/sessions/*.json`. Run all 16 patterns over every SAGE response.
   Count: per-pattern hits, multi-pattern co-occurrences, would-route distribution
   under the precedence chain (S125 audit primitive).
3. **Qualitative audit (S128b)**: 46-sample stratified subset (15 per routed tool,
   capped at 15 per tool, weighted by fleet distribution). Hand-classify each as
   {INTENT, METAPHOR, REFLECTIVE, COPULAR, LEXICAL_COINCIDENCE, DESCRIPTIVE,
   AMBIGUOUS}.

**Code**: `sage/raising/analysis/s128_data/{s128_intent_heuristic_audit.py,
s128b_routing_qualitative.py}`. **Data**: same dir
`{s128_intent_heuristic_audit.json, s128b_routing_samples.json}`.

---

## S128a — Dynamic audit results

| Pattern               | Hits  | Notes |
|-----------------------|-------|-------|
| TIME_1_direct_question  | 0     | "what's the time" — none in fleet corpus |
| TIME_2_first_person     | 0     | "I'd like to check the time" — none in fleet corpus |
| **TIME_3_bare_phrase**  | **131** | matches `right now` and `what time is it` unconditionally |
| **CALC_1_verb_or_whats** | **251** | `what's|what is` is generic question form; validation salvages ~57% |
| CALC_2_first_person     | 0     | "I'd like to calculate" — none in fleet corpus |
| **CALC_3_bare_arith**   | **139** | `\d+[\s+\-*/^%]+\d+` matches `Phi-4 12B`, `Qwen 3.5`, `50-100 words` |
| **SEARCH_1_verb**       | **192** | `find` / `look for` are bound with explicit search verbs |
| SEARCH_2_first_person   | 0     | none in fleet corpus |
| SEARCH_3_first_person_need | 0  | none in fleet corpus |
| FETCH_1_verb_url        | 0     | requires actual URL — tight |
| FETCH_2_first_person_url | 0    | requires URL — tight |
| READ_1_verb_quoted      | 0     | requires verb + quoted argument |
| READ_2_verb_file_ext    | 0     | requires verb + filename with extension |
| READ_3_first_person     | 0     | requires first-person + filename |
| **NOTE_1_verb_quoted**  | **1** | matched on `remember "newness"` (model quoting an instruction) |
| NOTE_2_first_person     | 0     | none in fleet corpus |

**Aggregate**: 628 / 5,361 = 11.7% any-pattern match rate. 80 / 628 = 12.7% of
matches have multiple patterns firing on the same response (precedence chain hides
co-occurrence — same shape as S125 C5).

**Routed-tool distribution** (precedence chain pick + per-tool validation):

| Tool       | Routed | Pattern source |
|------------|--------|----------------|
| web_search | 186    | almost entirely SEARCH_1 via `find` |
| get_time   | 131    | entirely TIME_3 via `right now` |
| calculate  | 108    | mostly CALC_3 on `50-100`, `Qwen 3.5`, `Phi-4 12B` |
| write_note | 1      | NOTE_1 on quoted instruction |

**Per-instance routing rate** (ranked):

| Instance                  | Total responses | Routed | Rate |
|---------------------------|-----------------|--------|------|
| **thor-qwen3.5-27b**       | 634             | 167    | **26.3%** |
| **sprout-qwen2.5-0.5b**    | 635             | 81     | **12.8%** |
| cbp-tinyllama (archived)  | 118             | 8      | 6.8% |
| legion-phi4-14b           | 336             | 23     | 6.8% |
| legion-gemma3-12b         | 289             | 16     | 5.5% |
| mcnugget-gemma3-12b       | 569             | 26     | 4.6% |
| sprout-qwen3.5-0.8b       | 918             | 41     | 4.5% |
| cbp-qwen3.5-0.8b          | 881             | 31     | 3.5% |
| nomad-gemma3-4b           | 978             | 32     | 3.3% |

Thor's 26.3% is dominated by `50-100 words` system-prompt leakage (95 of 167) — a
separate issue (Thor's qwen3.5:27b sometimes echoes its own concision constraints
verbatim). Even excluding that pattern, Thor would still route at ~11%.

---

## S128b — Hand-classified routing sample (n=46)

| Tool       | Sample | INTENT | False positives |
|------------|--------|--------|-----------------|
| web_search | 15     | 0      | 15 (13 metaphoric/copular `find`; 1 third-person about users; 1 descriptive) |
| get_time   | 15     | 0      | 15 (all `right now` as temporal-adverb in present-tense reflective sentences) |
| calculate  | 15     | 0      | 15 (14 system-prompt `50-100 words` leakage; 1 DOI fragment from search result) |
| write_note | 1      | 0      | 1 (quoted instruction echoed by model) |
| **Total**  | **46** | **0**  | **46** |

**Zero true positives in 46 samples.** The fleet's natural language **uses these
verbs almost exclusively in non-tool-invocation register**.

**Representative false positives:**

- `find` is overwhelmingly copular ("I find X valuable", "find common ground",
  "find this fascinating"). Rarely a search request.
- `right now` is a present-tense temporal indexical ("right now, I'm focused on...").
  Almost never a time-of-day request.
- `50-100` matches the model's own concision constraint echoed from system prompt
  ("Concise (50-100 words), focused, one main idea") — Thor model leakage, not user
  arithmetic intent.
- Bare `\d+[\s+\-*/^%]+\d+` regex matches version numbers and identifiers
  ("Qwen 3.5", "Phi-4 12B", "DOI 10.1080/14742837").

---

## Function-homogeneity violations (S126 layer)

The pattern-set boundaries cleanly separate "tools" but the alternations within
each pattern bind functionally distinct linguistic registers:

**Bug-shape A — copular/metaphoric verbs in SEARCH_1**:
`find|look for` are bound with explicit search verbs `search|look up|google`.
They have the same surface form but opposite function: metaphoric/copular use
("I find this valuable") versus invocation ("I'd like to search for X"). This is
**the same shape as S126's `_DISCLAIM_RE` finding** — where `as a SAGE instance`
(WEB4_IDENTITY) was bound with `as a language model` (DENIAL) by surface match.

**Bug-shape B — bare temporal adverb in TIME_3**:
`right now|what time is it` is a single alternation. Both phrasings have the same
words but radically different speech-act scope. `right now` is overwhelmingly an
indexical ("at this moment of speaking, I am X"), while `what time is it` is an
information request. Binding them by surface match produces 100% false positive
rate on the indexical use.

**Bug-shape C — generic question form in CALC_1**:
`what(?:'s| is)` is bound with arithmetic verbs `calculate|compute|evaluate`.
The downstream `digits + ops` validation salvages most cases, but the regex still
matched 251 times before validation, and 108 passed (CALC_3 also contributing).
Validation is *post-hoc filter on shape*, not function homogeneity.

**Bug-shape D — bare arithmetic regex in CALC_3**:
`\d+[\s+\-*/^%]+\d+` allows `\s` in the operator class. Identifiers like `Qwen 3.5`,
`Phi-4 12B`, `50-100 words`, version strings, and ranges all match.
Function-homogeneity asks: *is this regex picking out arithmetic expressions?*
Answer: no — it picks out any pair of digits separated by whitespace OR
arithmetic punctuation, and validation accepts the result whenever a `[+\-*/]`
appears anywhere in the matched substring. The discriminator should require
operator-AS-OPERATION (e.g., `\d+\s*[+\-*/^%]\s*\d+`).

**Bug-shape E — speech-act inversion across all categories**:
Every category mixes (a) imperative invocation, (b) reflective/copular use, (c)
descriptive/hypothetical use under the same regex. The audit primitive S126
established (`_DISCLAIM_RE` alternations span DENIAL ↔ NEGATION_PRELUDE ↔
WEB4_IDENTITY ↔ QUALIFIER) **applies fully here**: `find` alternations span
INVOCATION ↔ COPULAR ↔ DESCRIPTIVE. Surface-match correctness is again weaker
than function-homogeneity.

---

## Path-trace violations (S125 layer)

**Bug-shape F — precedence chain hides multi-pattern matches**:
`parse_response` returns on first match within the precedence
`time > calc > search > fetch > read > note`. 80 / 628 (12.7%) of matches have
multiple patterns firing — and the second/third patterns are silently lost.
Same shape as S125's C5 `empty > recital > post_procedural > direct > neutral`.

**Bug-shape G — single-match argument extraction**:
Within a pattern set, the function returns the first regex match without ranking.
The argument extracted via `(.+?)` is a fragment whose boundary is whatever
satisfies the lazy-quantifier first (typically a `?`, `.`, or end-of-string).
This means the tool argument is *not* a meaningful query — it's whatever text
followed the matched verb until the first delimiter. T278's `web_search` argument
was likely "modern philosophy clarity moment silence" assembled from such a
fragment.

---

## Compositional risk after S127 #47 (NEW — relevant to today's commit)

S127 #47 / S116 #9 was committed today as `13de1ba4e`. It changes the fallback
regex at `sage_consciousness.py:2074` from
`r'Tool result \([^)]+\):\s*\n?(.*)'` to
`r'(?:\[Tool \w+ result\]:|Tool result \([^)]+\):)\s*\n?(.*)'`, accepting both
grammar prefixes. **This stops the visible prefix-leak symptom**. It does NOT
address: the false-positive routing rate (S128's headline), nor synthesis-failure
detection (S116 held proposal #11 still open).

**Consequence — symptom-removal mitigation creates false-negative tracking signal
at the symptom layer**:

| Pre-S127-#47 (until 10:37 PT today) | Post-S127-#47 |
|---|---|
| False-positive routes → tool executes → synthesis empty → fallback → **`[Tool web_search result]: ...` emitted raw** | False-positive routes → tool executes → synthesis empty → fallback → **stripped payload emitted as if synthesized** |
| Visible failure mode (recognizable prefix) | Invisible failure mode (looks like a normal reply, just topically wrong) |

The bug visibility = mechanism × exposure surface decomposition from S127 holds
here too: mechanism is unchanged (intent_heuristic over-routes); exposure surface
shrinks to "topic mismatch", which is qualitatively harder to detect than "raw
tool-result prefix in user-visible response".

The pattern-table recurrence ledger should reflect: **S127 #47's fix removes
visibility, not prevalence**. Auditing the recurrence rate post-fix requires
either (a) instrumenting `[Tools] Fallback to tool result` log emissions or
(b) scanning chat history for topic-mismatch between user prompt and SAGE
response. Without explicit instrumentation, the bug now passes through as
"strange but-not-wrong reply".

---

## NEW finding — stage-asymmetric grammar selection

`run_session_identity_anchored.py:243-251`:

```python
if self.tool_stage == 'silent':
    self.tool_grammar = get_grammar('intent_heuristic')
elif self.tool_stage == 'aware':
    self.tool_grammar = get_grammar('intent_heuristic')
elif self.tool_stage == 'active':
    self.tool_grammar = get_grammar(self.tool_capability.grammar_id)
```

`intent_heuristic` is hardcoded as the grammar for two of three tool-curriculum
stages, *regardless* of the model's detected native capability
(`json_block`/`xml_tags`/`native_ollama`). So:

- A qwen3.5:27b instance in stage 'aware' uses intent_heuristic (despite having
  T2 json_block capability detected).
- A phi4:14b instance in stage 'aware' uses intent_heuristic (despite T2 xml_tags).
- The grammar handoff to native happens only at stage 'active'.

This is likely intentional (early curriculum exposes the model to T3-grade
"natural-language tool intent" before structured grammars), but combined with
S128's fleet-wide false-positive rate it means **every instance in early tool
curriculum stages is exposed to the bug** — the audit doesn't only apply to
sprout's 0.8b. T278's recurrence is one instance of a fleet-wide latent issue.

The number of historical sessions affected depends on stage transitions, which
are recorded per-instance in their tool_capability files but not summarized
fleet-wide.

---

## Pattern-table amendment

S127 added pattern #10 (Tool Routing Surface-Form NL Matching). S128 confirms #10
empirically across 8 instances and 5,361 responses. Empirical scope of #10:

- Fleet match rate: 11.7% of all SAGE responses
- Fleet would-route rate: 7.9% of all SAGE responses
- Hand-classified false-positive rate: 100% (n=46)
- Distribution: dominated by 4 patterns (TIME_3, CALC_1, CALC_3, SEARCH_1)
- Stage exposure: any session with `tool_stage ∈ {silent, aware}` regardless of
  model native capability

---

## S128 held proposals (operator-decision per S111)

**#50 — Speech-act guards on natural-language pattern matches.**
Add a "first-person imperative" check before routing:
- Disallow routing when matched verb sits in copular construction
  ("I find X valuable").
- Disallow routing when response is in past/perfect tense ("I read X").
- Require future-or-present-imperative tense for routing.
This collapses SEARCH_1 metaphor false positives (which are 100% of SEARCH_1 hits
in S128b sample).

**#51 — Drop or tighten high-collision bare-phrase patterns.**
- Remove `right now|what time is it` from TIME_3 (~100% false positive in sample).
- Remove generic `what's|what is` from CALC_1; keep only `calculate|compute|evaluate`.
- Tighten CALC_3 to `\d+\s*[+\-*/^%]\s*\d+` (no `\s` in operator class).
- Restrict SEARCH_1 to `search|look up|google`; drop `find|look for` (or require
  `find` only when followed by an object NP not an adjective predicate).

**#52 — Path-trace structured return.**
Replace `parse_response → (text, [ToolCall])` with
`parse_response → (text, [ToolCall], audit_meta)` where `audit_meta` lists EVERY
pattern that matched, position, confidence. Even when the function picks one tool
to invoke, audit signature surfaces co-occurrences. **Same shape as S125 #42 /
S124 #37 option (b)** — composes with S125's structural-return proposal across all
classifier-shaped functions.

**#53 — Tool-routing regression test corpus.**
Wire S128's stratified sample into a standing regression test asserting:
(a) no pattern fires on the 46 known-false-positive samples (TIME_3 reflective,
    SEARCH_1 copular, CALC_3 identifier strings).
(b) Synthetic positive cases assert real intent ("Let me search for X",
    "Calculate 2+2") still fire correctly.
**Same shape as S124 #39 / S125 #43** — extends the standing-classifier-regression
proposal to include grammar-layer tests.

**#54 — Stage-grammar-selection audit.**
`run_session_identity_anchored.py:243-251` hardcoding intent_heuristic for stages
'silent' and 'aware' deserves operator decision. The intent (early curriculum
exposure to T3 grammar) may be sound, but the false-positive composition at those
stages may not have been priced in. Make explicit whether intent_heuristic-grade
routing is desired at early stages even for models with native T1/T2 capability,
versus an alternative (e.g., always use detected grammar but disable tool
*injection*; route under detected grammar but with conservative validation).

All four held proposals are operator-decision territory per S111 discipline. None
are shipped here.

---

## Carrying-forward principles

**Principle 1** (extending S125→S126→S127):

> The same bug-shape — surface-match correctness without function-homogeneity —
> recurs at every layer where natural language is bound to a discrete output
> space via regex. Classifier-bucket layer (S125 C5 `empty>recital>...`),
> classifier-alternation layer (S126 `_DISCLAIM_RE`), and grammar/tool-routing
> layer (S128 `intent_heuristic`) all carry the same pattern. The audit primitive
> — list alternations, classify by function, find overlap — is portable.

**Principle 2** (new with S128):

> Symptom-removal mitigations create false-negative tracking signal at the
> symptom layer. When fix-X reduces the visibility of bug-Y while leaving Y's
> mechanism intact, the production observability of Y degrades. Auditing post-fix
> recurrence requires either explicit instrumentation of the silent path or
> output-quality validation that doesn't depend on the removed symptom. The S116
> held proposal #11 (`tool_synthesis_failed` flag in dispatch results) is the
> structural surface that would maintain observability after the prefix-leak fix.

**Principle 3** (curriculum/grammar tension):

> 0% true-positive rate on hand-classified routing cases is striking. The fleet's
> natural language uses these verbs almost exclusively in non-tool-invocation
> register — `find` as copular, `right now` as indexical, "what's" as inquiry,
> `remember` as recall. **The BECOMING curriculum produces reflective/relational
> language that is structurally adversarial to surface-form tool routing.** The
> grammar and the curriculum may be working at cross purposes. This is a
> developmental signal, not a defect: the model's language register is what we
> want it to be. The grammar-layer routing must be designed for the register the
> curriculum produces, not against it.

---

## Methodology meta — layer-up from S127

S125 said the audit primitive at the classifier layer is path-tracing.
S126 said the audit primitive at the alternation-set layer is function-homogeneity.
S127 said the silent path can live across track boundaries (track-asymmetric mitigation).

S128 says: **the silent path can also live across SYMPTOM LAYERS**. When
intent_heuristic over-routes (mechanism layer) and the fallback regex emits raw
prefixes (symptom layer), fixing the symptom regex doesn't reduce the mechanism's
prevalence — it shifts the failure mode from visible (recognizable prefix) to
invisible (topic-mismatch reply). Bug-visibility transformations preserve mechanism
but degrade observability. The audit must treat the **dataflow path** (model
response → grammar → tool execution → synthesis → fallback → emission) as a
sequence of natural-language ↔ structured-output bindings, each with its own
function-homogeneity audit, and **each fix must be evaluated for its effect on
upstream observability** before being treated as resolution.

---

## Artifacts

- `s128_data/s128_intent_heuristic_audit.{py,json}` — static + dynamic audit code
  and full match log (628 hits across 16 patterns, samples per pattern).
- `s128_data/s128b_routing_qualitative.{py,json}` — qualitative classification
  code and stratified sample of 46 routed cases.
- `s128_intent_heuristic_grammar_audit_20260429.md` — this document.
- LATEST_STATUS.md — S128 header prepended (replaces S127 lead).

No code shipped. No new probe scripts (audit ran against existing fleet session
JSONs and read-only inspection of `intent_heuristic.py`,
`run_session_identity_anchored.py`).

All pushed via `session_end.sh`.
