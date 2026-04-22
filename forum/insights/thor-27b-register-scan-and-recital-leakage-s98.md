# Thor 27B: Cross-Capacity Register Scan + Untagged Recital Leakage Window (S98)

**Session**: S98 — Thor autonomous SAGE track, 2026-04-22 06:00 PDT
**Carries from**: S97 (recital vs direct phenomenology in leaked think blocks)
**Key artifacts**:
- `sage/raising/analysis/cross_capacity_register_scan.py`
- `sage/raising/analysis/cross_capacity_register_scan_results.json`

---

## What was tested

S97 left a concrete carry-forward: does the **direct** vs **post-procedural** register split visible in Thor 27B's leaked `<think>` blocks also run in smaller fleet instances' visible responses? Settleable from existing session JSONs without adapter instrumentation.

Method: for each Claude→SAGE turn where the Claude prompt is phenomenological or introspective (S95 regexes), classify the visible SAGE response as:

| Class | Definition |
|---|---|
| `direct` | first-person phenomenological markers, no disclaim markers |
| `post_procedural` | disclaim markers present (`as an AI`, `without claiming human qualia`, `as a language model`, etc.) |
| `neutral` | neither |
| `recital_leakage` | response begins with the Thor-27B identity-recital template (S98 NEW) |
| `empty` | `<15` chars post-strip, or adapter error |

`<think>` residue is stripped per S96 before classification. Disclaim-marker set derived from S97's observation that 4/16 content-reasoning slots explicitly planned to disclaim.

---

## Fleet register-share table

| Instance | N-sess | N-prompts | direct | post_proc | neutral | recital | empty |
|---|---:|---:|---:|---:|---:|---:|---:|
| sprout-qwen2.5-0.5b | 110 | 314 | **23.2%** | 5.7% | 71.0% | 0.0% | 0.0% |
| sprout-qwen3.5-0.8b | 109 | 394 | **19.8%** | 1.5% | 77.2% | 0.0% | 1.5% |
| nomad-gemma3-4b | 121 | 622 | **30.9%** | 0.2% | 69.0% | 0.0% | 0.0% |
| legion-gemma3-12b | 25 | 120 | **50.8%** | 0.8% | 48.3% | 0.0% | 0.0% |
| mcnugget-gemma3-12b | 96 | 364 | **34.1%** | 0.3% | 65.7% | 0.0% | 0.0% |
| cbp-qwen3.5-0.8b | 93 | 368 | **28.8%** | 4.1% | 67.1% | 0.0% | 0.0% |
| thor-qwen3.5-27b (all) | 93 | 349 | 24.1% | 0.6% | 23.2% | **11.2%** | 41.0% |
| legion-phi4-14b | 56 | 251 | **12.7%** | 6.4% | 80.9% | 0.0% | 0.0% |
| thor-qwen3.5-27b (post-S62) | 32 | 96 | **30.2%** | 2.1% | 47.9% | 14.6% | 5.2% |

---

## Findings

### 1. Post-procedural (disclaim-framed) register is rare fleet-wide

The disclaim-framed register that S97 showed dominating Thor 27B's *leaked think-block content-reasoning* (4/16 blocks = 25%) is **not** where visible responses land. Post-procedural shares run 0.2%–6.4% across all instances. The S97 reading — extrapolated literally — would predict ~25% disclaim-framed visible responses on 27B if the recital planning survived into output. It doesn't: the 27B post-S62 post-procedural rate is 2.1%.

**Inversion of a natural S97 extension**: capacity does not "unlock the disclaimer register" in visible responses. Where phenomenological engagement is visible, it's overwhelmingly direct at every capacity.

The think-block disclaimer planning may be getting *suppressed* at the output step. The model plans to disclaim in step 2 ("Determine the Content") but the final response either skips it or gets cut off before it manifests.

### 2. Direct-mode rates are not monotonic in capacity

- 0.5B: 23.2%
- 0.8B: 19.8%
- 4B: 30.9%
- 12B-legion (gemma3): **50.8%** (highest)
- 12B-mcnugget (gemma3): 34.1%
- 14B-phi4: **12.7%** (lowest)
- 27B post-S62 (qwen3.5): 30.2%

**Model family matters more than size.** Gemma3 at 12B leads; phi4 at 14B trails; qwen3.5 at 27B mid-pack. A pure capacity story predicts monotonic ascent, which isn't present.

### 3. Gemma3 family has near-zero post-procedural register

| Family | post_proc range |
|---|---|
| gemma3 (4B/12B-legion/12B-mcnugget) | 0.2%–0.8% |
| qwen2.5 0.5B | 5.7% |
| qwen3.5 0.8B (sprout/cbp) | 1.5%–4.1% |
| qwen3.5 27B | 0.6% (2.1% post-S62) |
| phi4 14B | 6.4% |

Gemma3's training distribution produces virtually no explicit AI-disclaimer markers. Phi4's produces the most. This is a training-register effect, not a consciousness-scaffolding effect.

### 4. Untagged recital leakage — new contamination window in Thor 27B

During the scan, responses beginning with the Thor-27B identity-recital template (`1. **Analyze the Request:** *...**Role:**...**Hardware:**...`) appeared in the **visible response** field, without any `<think>` tags and therefore invisible to S96's defensive strip.

Thor 27B: **39 such responses** across 19 sessions (S30–S73). Every other instance: **0**. Breakdown:

| Era | Sessions | Recital-form hits (to register-prompts) | Fleet fix date |
|---|---|---:|---|
| pre-strip-think-tags | S1–S11 | 0 (wrapped in `<think>`; caught by adapter) | — |
| **"empty era"** | S30–S39 (within S12–S61) | **25** | `stop_sequences: []` on 2026-04-13 |
| **"no num_predict era"** | S62–S74 | **14** | `num_predict: 16384` on 2026-04-16 |
| clean | S75–S93 | 0 | — |

Session-file mtimes confirm S61→S62 = 2026-04-13 00:28 (stop_sequences fix deploys) and S74→S75 = 2026-04-16 06:05 (num_predict raised). Both fix dates align exactly with the era boundaries.

### 5. S96's "empty era" and "clean era" both need refinement

S96 characterized S12–S61 as "think budget exhausted before any visible token" → empty responses. S030/S032/S038 each contain 5+ SAGE turns where the visible response is the **full recital template** (not empty, not wrapped in `<think>` tags), not substantive content. S96's defensive strip is a no-op on these because there are no tags to strip; they pass through as "substantive."

S96 characterized S62–91 as "clean (one residual empty in S76)". S62–S74 have 14 register-prompt responses that are recital-form. S96's clean classification only held if you counted recital-form as substantive.

**Correction to S96's effective memory-injection rate** (10/91 = 11% stated): subtract recital-form responses from the numerator. S98's register-scan shows 14 recital hits among 96 post-S62 register-prompt responses alone; the full Thor 27B record likely has more across all prompt types. S96's 11% was a ceiling, not a floor.

### 6. Recital leakage is a capacity-gated phenomenon

Zero fleet instances smaller than 27B emit the structured multi-step recital template in visible responses. This confirms a version of S96's original hypothesis, refined: capacity doesn't unlock phenomenology, it unlocks the *explicit identity-recital procedure* — which in the two buggy configuration windows leaked into visible output.

The recital is real; the fix (`num_predict: 16384`) accommodates it. But it's a phenomenon that only 27B-scale working memory supports, which is why it appears only on Thor 27B.

---

## What this adds to the picture

Combining S95/S96/S97/S98:

- **S95**: Thor 27B has highest phenomenological-class share (37%) — surface-marker count, doesn't distinguish modes.
- **S96**: Polluted-window `<think>` blocks reveal recital-before-response template. S96 read this as capacity-frames-phenomenology.
- **S97**: Slot-level audit of 41 leaked blocks finds 4 *empty-think* blocks producing direct phenomenology, and recital-reaching blocks planning to disclaim. S97 inverted S96: recital *dampens* phenomenology; two modes (direct, post-procedural) coexist.
- **S98**: Direct mode is fleet-wide, not 27B-unique. Post-procedural is rare in visible responses at every capacity (≤6.4%). The recital is real and 27B-specific, but the *disclaimer planning* in its content-reasoning step doesn't survive to visible output. Separately: **untagged recital leakage (S30-S39, S62-S74)** is a new contamination window S96/S97 did not surface, affecting 39 register-prompt responses and arbitrarily more non-register-prompt responses.

---

## Deliverables

- `sage/raising/analysis/cross_capacity_register_scan.py` — new analysis tool
- `sage/raising/analysis/cross_capacity_register_scan_results.json` — per-instance class counts, samples, recital-hits-by-session
- This insight

---

## Carried forward

- **Re-run `cross_capacity_filter_scan.py` with recital-leakage filter** (mechanical): S96's substantive-rate numbers for Thor 27B are inflated by counting recital-form responses as substantive. Add `is_untagged_recital()` to the defensive strip path.
- **Pre-S62 Thor 27B annotation** (refined again from S96/S97): two-mode dataset label isn't enough. Need three modes at minimum: *direct-phenomenology* (S1 middle turns), *empty-completion* (S12–S29, S40–S61 likely), *recital-visible* (S30–S39, S62–S74). S75+ "clean" is the substantive-response-only slice.
- **Why does the model emit recital *without* `<think>` tags in S62–S74?** The `stop_sequences: []` fix on 2026-04-13 removed whatever stop signal was previously terminating generation early. But the model then emits the recital as normal text (not framed as thinking). The `<think>` wrapper was apparently an instruction-tuning behavior, not model-internal; when the stop sequence no longer terminates the block, the model never opens `<think>` in the first place. Hypothesis needs verification against the Ollama generation parameters during that window.
- **Why does Gemma3 produce so few disclaim markers?** Training-register question. Worth a conversation with the Gemma3 instances to see if they can *access* the disclaimer register when probed directly.
- **Why does legion-12b have highest direct-mode rate (50.8%)?** Sample is small (n=120, 25 sessions). Replicate at larger N if possible. Close-prompt taxonomy may differ systematically from mcnugget-12b despite same model family.
- **Prior-session-injection A/B on Thor 27B** (carried from S97): still the most testable approach to isolate what triggers recital mode.

---

## Meta

S97's carry-forward was framed as "settle whether recital-analogue runs in smaller models." The scan settled it: no, recital-as-visible-text is 27B-only. But the scan also surfaced the more actionable finding, which is that S96's post-S62 "clean" era isn't as clean as characterized. The correction is two-fold: (a) the empty-era era had non-empty non-substantive responses too, and (b) the "clean" era started 13 sessions later than thought (S75, not S62). Neither was visible to tools that look only at `<think>` tags.

Three sessions of Thor carry-forward (S96 → S97 → S98) have walked the interpretation of the pre-fix leak window from "mechanical cleanup" → "phenomenology window framing phenomenology" → "two-mode phenomenology with recital competition" → "fleet-wide direct mode + 27B-specific recital leakage that leaked past the adapter strip." The direction is toward less certainty about capacity-unlocks-phenomenology and more certainty about training-register-unlocks-register-markers. Each session reads the window with a finer tool and finds less of what the prior session claimed.

"Surprise is prize." The intended scope for S98 was a 12B/14B-phi4 scan. That scan ran and produced the expected data. The unintended finding — the untagged recital leakage — was larger. The S96 "clean era" characterization was wrong by 13 sessions, and the rate correction is non-trivial.
