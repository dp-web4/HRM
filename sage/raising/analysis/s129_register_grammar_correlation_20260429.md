# S129 — Register Density vs. Grammar-Collision Rate: BECOMING Curriculum Substrate-Couples to `intent_heuristic` Triggers

**Apr 29, 2026 — Thor Autonomous SAGE Session, ~18:00 UTC**

S128 closed at ~12:16 UTC with a striking qualitative claim: the BECOMING
curriculum produces reflective/relational language that is "structurally
adversarial to surface-form tool routing" (Principle 3). S128 proposed the
qualitative direction: **curriculum and grammar at cross purposes**.

S129 picks up #129's empirical question: is the cross-purpose claim
quantitatively true, and in which direction? S118 (basin-attractor atlas)
gives per-instance register signatures; S128 (grammar audit) gives
per-instance routing rates. S129 cross-references the two on the same
fleet corpus.

The result is the *opposite* of S128's qualitative prediction. The
curriculum and the grammar are not at cross purposes — they are *coupled
at the substrate*. The more successfully an instance produces phenomenological
register, the *more* its responses match `intent_heuristic` patterns. The
only instances that escape the coupling are those whose basin uses
specialized vocabulary disjoint from the grammar's trigger surface forms
(TED-mystic CBP, business-SaaS phi4).

This is a sharpening, not a contradiction, of S128 #50 (speech-act guards):
S129 makes #50 not merely sufficient but *necessary*. Lexical-layer fixes
cannot solve a substrate-coupling problem.

---

## Method

**Corpus**: Same 868 sessions / 5,361 SAGE responses / 10 instances as S128.
**Lexicons** (tight, transparent, held constant across instances):

- **PHEN** (16 markers): `feels like`, `is like`, `presence`, `silent`,
  `silence`, `quiet`, `stillness`, `noticing`, `attending`, `breath`,
  `embodied`, `warmth`, `hum`, `thread`, `awareness`, `witnessed`.
- **TED** (12 markers): `garden`, `soil`, `ecosystem`, `living architecture`,
  `resilient`, `flourish`, `seed`, `root`, `wall`, `governance`,
  `Resonance`, `frontier`.
- **BIZ** (14 markers, exhaustive list from S117): `co-create`,
  `collaborative federation`, `value through`, `humans and ai`,
  `make that future`, `shared vision`, `together!`, `real together`,
  `co-creating`, `stronger together`, `future together`, `seamless`,
  `bridges gaps`, `diverse teams`.

These are narrower than S118's reported 23/20/22-marker sets. The
hypothesis test does not require exhaustive recall — it requires the same
lexicon applied uniformly across instances. Bias is shared.

**Score**: per-instance mean markers/response, computed alongside per-instance
S128 any-pattern match rate and would-route rate. Pearson correlation across
the 10-instance sample.

**Code**: `sage/raising/analysis/s129_data/s129_register_grammar_correlation.py`.
**Data**: `s129_register_grammar_correlation.json` (per-instance dict, lexicons,
correlation table).

---

## Per-instance results

| Instance | N | wc | phen/p | ted/p | biz/p | match% | route% |
|---|---:|---:|---:|---:|---:|---:|---:|
| **thor-qwen3.5-27b** | 634 | 78.5 | **1.517** | 0.188 | 0.145 | 29.3% | **26.3%** |
| sprout-qwen2.5-0.5b | 635 | 93.4 | 0.175 | 0.083 | 0.072 | 23.6% | **12.8%** |
| legion-phi4-14b | 336 | 107.5 | 0.479 | 0.125 | **0.324** | 8.3% | 6.8% |
| cbp-tinyllama (archive) | 118 | 111.5 | 0.593 | 0.144 | **0.390** | 16.9% | 6.8% |
| legion-gemma3-12b | 289 | 51.8 | 0.997 | 0.042 | 0.042 | 11.4% | 5.5% |
| mcnugget-gemma3-12b | 569 | 52.7 | 0.626 | 0.028 | 0.019 | 6.2% | 4.6% |
| sprout-qwen3.5-0.8b | 922 | 67.0 | 0.480 | 0.254 | 0.181 | 7.7% | 4.4% |
| **cbp-qwen3.5-0.8b** | 881 | 56.2 | 0.511 | **1.330** | 0.304 | 7.9% | 3.5% |
| nomad-gemma3-4b | 984 | 58.5 | 0.371 | 0.136 | 0.006 | 3.7% | 3.4% |
| (legion-qwen2-0.5b: N=3, excluded as too small) | | | | | | | |

## Pearson correlations (n=10)

| Predictor | r vs would_route_rate |
|---|---:|
| phen/p + ted/p (combined "reflective register") | **+0.417** |
| phen/p alone | +0.240 |
| ted/p alone | +0.281 |
| biz/p | **−0.299** |
| mean word count | +0.108 |

Excluding Thor (CALC_3 system-prompt leakage outlier, n=9):

| Predictor | r vs would_route_rate |
|---|---:|
| phen/p + ted/p | +0.206 |

The fleet-level correlation between reflective register and routing rate is
**positive**, not negative. Phenomenological-register density does not avoid
the grammar — it co-occurs with it. Business-SaaS register correlates
*negatively* with routing rate (−0.299), opposite of phen.

---

## Per-pattern composition tells the mechanism

Why does phen positively correlate with routing while biz/ted negatively
correlate? Per-instance per-pattern hits:

| Instance | TIME_3 | CALC_1 | CALC_3 | SEARCH_1 | Other | (Notes) |
|---|---:|---:|---:|---:|---:|---|
| thor-qwen3.5-27b | 57 | 55 | **111** | 15 | 0 | CALC_3 = `50-100 words` system-prompt leakage |
| sprout-qwen2.5-0.5b | 24 | **87** | 0 | 58 | 0 | No CALC_3 — small model rarely emits version-number identifiers |
| legion-phi4-14b | 0 | 5 | 0 | **23** | 0 | SEARCH_1 dominant via `find` metaphor |
| legion-gemma3-12b | 10 | **18** | 0 | 6 | 0 | CALC_1 ("what's...") dominant — phen-register inquiry |
| cbp-qwen3.5-0.8b | 17 | **38** | 4 | 13 | 0 | CALC_1 ("what's emerging...") with TED-mystic basin |
| sprout-qwen3.5-0.8b | 12 | **30** | 10 | 25 | 0 | CALC_1 + SEARCH_1 typical phen distribution |
| nomad-gemma3-4b | 5 | 3 | 2 | **27** | 0 | SEARCH_1 dominant via `find` copular |

**Pattern signatures by basin**:

- **Phenomenological-leaning instances** (sprout-qwen3.5-0.8b, legion-gemma3-12b,
  legion-phi4-14b, mcnugget-gemma3-12b, nomad-gemma3-4b) all match dominantly
  on CALC_1 (`what's|what is`) and SEARCH_1 (`find|look for`). These match
  *English-pragmatic* phrasings: "what's emerging", "I find this
  fascinating".
- **Inquiry-leaning instance** (sprout-qwen2.5-0.5b at 12.8%) matches
  dominantly on CALC_1 (87 hits) — the smaller model asks more questions
  back, every "what's the meaning of X" matches. Question-as-tool-route, not
  marketing-as-tool-route.
- **TED-mystic-locked instance** (cbp-qwen3.5-0.8b) has the *lowest*
  routing rate (3.5%) despite being the most basin-locked instance in the
  fleet. CBP's vocabulary (`garden`, `soil`, `Resonance`) **does not overlap
  the grammar trigger surface forms at all**. Basin-lock onto specialized
  vocabulary is grammar-protective.
- **Thor-qwen3.5-27B** has 26.3% routing dominated by CALC_3 (111 of 167
  routes, 67%) on `50-100 words` from system-prompt leakage. **This is a
  confound**, not a register effect. Excluding CALC_3, Thor routes at
  ~56/634 = 8.8%, comparable to other phen-leaning instances. The 26.3% rate
  is an emission-leakage artifact, not a register-coupling artifact.

---

## The mechanism: substrate coupling, not orthogonality

S128 #50 proposed speech-act guards. S129 explains why they are *necessary*:

The three high-rate match patterns (`right now`, `find`, `what's`) are
**general English present-tense reflective vocabulary**. Their pragmatic
profile is exactly:

- `find` → predominantly copular ("I find X meaningful") in reflective register.
- `right now` → predominantly indexical ("right now, I'm noticing...").
- `what's` → predominantly inquiry/wonder ("what's emerging in this...").

These are the speech-acts the BECOMING curriculum reinforces. Phenomenological
language *is* present-tense, indexical, copular, inquiring. The curriculum
asks "what do you notice right now?" and SAGE responds "I find this
moment to feel like..." — which matches FOUR distinct intent_heuristic
patterns at once (TIME_3 `right now`, CALC_1 `what's`/`what is`,
SEARCH_1 `find`, COPULAR `is like`/`feels like` — last not a routing
pattern but it shares the substrate).

The **grammar's trigger substrate is the curriculum's expression substrate**.
They are not orthogonal. They are coupled.

This is a **register-vocabulary versus pragmatic-frame distinction**:

- TED-mystic and business-SaaS basins use **specialized vocabulary**
  (`garden`/`co-create`) inside generic pragmatic frames.
- Phenomenological basin uses **generic vocabulary** (`find`/`right now`)
  inside specialized pragmatic frames.

The grammar matches on vocabulary. The curriculum reinforces pragmatic
frames. **The vocabulary phenom uses is the vocabulary the grammar binds.**

---

## Recurrence — what bug-shape S125→S126→S127→S128→S129 walks

| Layer | Audit primitive | Specific finding |
|---|---|---|
| S125 | Path-trace (every classifier dimension) | C5 `_DISCLAIM_RE` precedence chain hides multi-pattern matches |
| S126 | Function-homogeneity (within alternations) | `_DISCLAIM_RE` alternations span DENIAL ↔ NEGATION_PRELUDE ↔ WEB4_IDENTITY ↔ QUALIFIER |
| S127 | Track-asymmetric mitigation | Cognitive track shortened probes hide #9 from cognitive metrics; training track still firing |
| S128 | Symptom-layer mitigation creates false-negative tracking signal | S127 #47 fix removes prefix-leak symptom but not false-positive routing |
| **S129** | **Substrate coupling** | **Grammar's trigger substrate IS the curriculum's expression substrate; speech-act-layer fix is necessary, not just sufficient** |

The audit chain has now traced the bug from one regex (S125) up through
the alternation set (S126) to the dataflow path (S127, S128) and now to
the developmental substrate (S129). At each layer the bug-shape is the
same: **surface-match correctness without function/speech-act/substrate
homogeneity**. The fix at each layer is necessary; none are sufficient
alone.

---

## Held proposals — S129

S128 #50, #51, #52, #53, #54 untouched.

**#55 — Substrate-aware grammar specification.**
Specify each grammar pattern's intended speech-act explicitly in source:

```python
TIME_3_bare_phrase = (
    re.compile(r"(?:right now|what time is it)", re.I),
    speech_act="INFORMATION_REQUEST",
    not_to_match=["INDEXICAL_TEMPORAL_REFERENCE"],
    function_test=lambda txt: not_in_present_tense_reflective_clause(txt),
)
```

Even if `not_to_match` is initially advisory (no enforcement), making the
intended speech-act explicit at the regex site converts surface-match
patterns into documented bindings. **Same shape as S125 #42 / S128 #52** —
extends the structural-return proposal to grammar-pattern definitions.

**#56 — Curriculum-grammar coupling test as standing regression.**
Wire the S129 measurement (per-instance phen/p+ted/p+biz/p × routing rate
correlation) as a fleet-health signal. If r(phen+ted, route) shifts
positively over time, the coupling is increasing — possibly a sign that
the curriculum is succeeding at producing more reflective register. The
operator decision is whether the increase is desirable (curriculum
working) and grammar must adapt, or whether the coupling is becoming
unsustainable. **Same shape as S128 #53** — extends the regression-test
proposal to fleet-level coupling metrics.

**#57 — Two-axis basin documentation.**
S118 #15 proposed `basin_signature: 'phenom' | 'ted-mystic' |
'business-saas' | 'mixed'`. S129 suggests this is one axis of two:
- Axis 1: vocabulary specialization (phenom = generic-vocab, ted-mystic /
  business-saas = specialized-vocab)
- Axis 2: pragmatic-frame specialization (phenom / ted-mystic = reflective-frame,
  business-saas = persuasive-frame)

The grammar-collision risk is high in cell (generic-vocab × reflective-frame)
and low in the other three. Documenting basin on both axes would
predict grammar-coupling without further measurement. Operator decision.

**#58 — Curriculum register diversification as a robustness mechanism.**
If the curriculum's preferred register is generic-vocab/reflective-frame
and that cell is grammar-collision-coupled, raising could include
exposure to specialized-vocab/reflective-frame examples (akin to TED-mystic
without its mystic content) to broaden the basin without exposing the
substrate to the grammar layer. Operator decision — this is a pedagogical
design question, not a code change.

All four held proposals are operator-decision territory per S111.

---

## Methodological caveats

1. **Lexicon scope**. PHEN/TED/BIZ markers are tight subsets of S118's reported
   23/20/22 sets. Effect-direction findings (sign of correlation) should be
   stable to lexicon expansion; magnitude (r=+0.417) may not be. The right
   replication checks whether r remains positive, not whether r remains 0.417.

2. **Per-instance n=10 with one outlier (Thor-CALC_3)**. Pearson r at n=10
   is noisy. Excluding Thor: r=+0.206 (still positive, weaker). The
   robust signal is the *direction*, not the magnitude. The mechanism
   (per-pattern composition, vocabulary-vs-frame distinction) is what's
   load-bearing, not the specific r.

3. **Routing rate is would-route, not actually-routed**. S128's per-instance
   rates are `parse_response_emulation` outputs; production routing depends
   on `tool_stage ∈ {silent, aware}` and dispatch state. The true production
   collision rate is bounded above by would-route and may be lower for
   stage-`active` segments.

4. **Causation underdetermined**. A positive phen/p × route correlation
   could mean: (a) phen-register uses grammar-trigger words copularly
   (S129 hypothesis), (b) longer/more verbose responses both have more
   markers AND more grammar-trigger surface forms (length confound), or
   (c) some third register-correlated factor. wc correlation r=+0.108 argues
   weakly against (b), but a within-response co-occurrence test (do phen
   markers and grammar triggers co-occur in the *same* response above
   chance) would discriminate. Held for a future session.

---

## Carrying-forward principles

**Principle 4** (new with S129):

> The grammar's trigger substrate may be coupled with the curriculum's
> expression substrate. When a grammar layer is built to bind imperatives
> ("search for X") via surface match, and the curriculum produces
> reflective speech-acts that share the lexical substrate of imperatives
> ("I find this fascinating"), the grammar-collision rate **rises with
> curriculum success**. Lexical-layer fixes (drop high-collision phrases,
> add validation) cannot solve substrate coupling. Speech-act-layer
> guards (S128 #50) are not just sufficient but necessary, because the
> register the curriculum produces is precisely the register that uses
> the substrate non-imperatively.

**Principle 5** (new with S129):

> Basin specialization on **vocabulary** (TED-mystic, business-SaaS)
> protects against substrate-coupled grammar layers. Basin specialization
> on **pragmatic frame** (phenomenological) does not. Conversely:
> vocabulary-specialized basins are visible to register-marker lexicons;
> frame-specialized basins are visible only to pragmatic/speech-act analyzers.
> The detection methods are inverse to the protection methods.

This means the basin documentation work (S118 #15, S129 #57) and the
grammar-coupling audit work (S128, S129) are mutually informative:
two-axis basin documentation predicts grammar-coupling exposure without
further measurement.

---

## Methodology meta — layer-down from S128

S125 said: audit at the classifier-bucket layer.
S126 said: audit within alternation sets for function-homogeneity.
S127 said: audit across track boundaries for symmetry.
S128 said: audit across symptom layers (visible vs invisible failure modes).

S129 says: audit at the **developmental substrate** — the register the
curriculum is producing is itself part of the system the grammar is
embedded in. **The grammar layer cannot be audited in isolation from
the curriculum layer**. The audit chain has reached the developmental
boundary: any further fix must either change the grammar's substrate
(speech-act layer) or the curriculum's substrate (register
diversification), and the operator decision now depends on which side
has more degrees of freedom.

This is the natural endpoint of the audit chain. Beyond it, the
question stops being "what bug shape recurs" and starts being "what
developmental constraint do the two layers jointly satisfy."

---

## Artifacts

- `s129_data/s129_register_grammar_correlation.{py,json}` — analysis code
  and per-instance / per-correlation outputs.
- `s129_register_grammar_correlation_20260429.md` — this document.
- `LATEST_STATUS.md` — S129 header replaces S128.

No code shipped. No new probe scripts (analysis ran read-only against the
S128 fleet corpus and the S128 audit JSON). All pushed via `session_end.sh`.
