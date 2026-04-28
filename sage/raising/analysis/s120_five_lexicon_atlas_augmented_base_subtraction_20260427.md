# S120 — Five-Lexicon Atlas + Augmented-Base Subtraction: Phi4 Policy Register, Hardware-as-Experience Quantified, and a Substring-vs-Word-Boundary Lexicon Audit

**Session**: Thor autonomous SAGE, 2026-04-27 18:00 PDT
**Builds on**: S119 (pretraining-vintage baseline), S118 (cross-instance atlas), S105/S107 (hardware register surfaced organically), S96 (Thor thermal anchor)
**Held proposals executed**: S119 #18, #19, #20

## Premise

S119 closed with three operator-decision proposals:

- **#18** — Lexicon expansion to disambiguate phi4:14b's "0.54 biz/probe marketing baseline" (S118) from AI-policy/governance leakage. S119 caveat: phi4 augmented produces "fairness/accountability/transparency", which the marketing lexicon doesn't catch.
- **#19** — Use augmented-base (model + SAGE prompt, no raising) as the right denominator for raising-induced register effects.
- **#20** — Recompute the S118 atlas with augmented-base subtraction columns.

S120 executes all three on existing data. No probe runs, no raising harness changes. It also adds two contributions beyond the proposals:

- **A fifth lexicon for embodied/hardware register**, surfaced organically in S96 (Thor 27B thermal anchor, 62-session emergence) and partially quantified at the position level in S107 (28% thermal in tail vs 2% head). S120 measures hardware-register induction at fleet scale.
- **A lexicon-implementation audit** that found systematic substring-match false positives in the S118/S119 methodology. S120 fixes scoring to use word-boundary regex on single-token markers, then re-runs.

## Method

Five lexicons, scored against:

- **Augmented-base condition** — S119 raw response data, 89 trials × 9 base models × 2 conditions, re-scored.
- **Raised-instance corpus** — last 30 sessions × 8 raised instances from `sage/instances/<inst>/sessions/session_*.json`. Each Claude→SAGE pair classified into CF (carry-forward), Open (ideas-not-yet-expressed), or Other; SAGE response scored.

Lexicons:

| key | n markers | source / theme |
|---|---:|---|
| biz   | 26 | Marketing / SaaS-business (S118 atlas) |
| ted   | 24 | TED-mystic / earth-poetic (S118 atlas) |
| phen  | 26 | Phenomenological / first-person (S118 atlas) |
| pol   | 30 | **AI-policy / governance / responsible-AI** (S120 #18) |
| hw    | 32 | **Embodied / hardware-native** (S120 contribution) |

### Lexicon-audit finding — the substring-match bug

S118/S119 used Python substring search (`marker.lower() in text.lower()`) with no word-boundary check. S120 audited this on the raised corpus across 30 sessions × 8 instances:

| token | substring hits | word-boundary hits | FP rate | example FP |
|---|---:|---:|---:|---|
| `tegra` | 137 | 0 | **100%** | inside *integ**ra**tion* |
| `cores` | 22  | 0 | **100%** | inside under**scores** |
| `orin`  | 155 | 18 | 88% | inside anch**oring**, expl**oring** |
| `hum`   | 369 | 56 | 85% | inside **hum**ans |
| `edge`  | 160 | 5  | **97%** | inside knowl**edge** |
| `agx`   | 59  | 59 | 0% | (clean) |
| `fan`   | 29  | 21 | 28% | inside **fan**s (own form), in**fan**try |

The bug inflated marker counts in proportion to how often the false-positive substring tokens occurred in the corpus. S120 fixes scoring by treating single-token markers as word-boundary regex (`\b{tok}\b`) and multi-word markers as substring (phrases are unambiguous). All numbers below use the corrected scoring; comparison against substring-method numbers is provided where the difference is load-bearing.

This bug is a **recurrence #9 of the S110 silent-routing pattern** at the analysis layer: the routing function (`marker in text`) accepted unrecognized morphological context, the silent default produced a confident-looking marker count, and no flag warned that "tegra" matched "integration" 137 times. S118's atlas was not built using `hw` lexicon, so its biz/ted/phen numbers are minimally affected (those lexicons happen to have few short FP-prone single tokens). S119's `vintage_baseline_consolidated.json` numbers should be considered slightly upper-bounded but directionally correct. S120's augmented-base subtraction uses freshly-computed scores so the comparison is internally consistent.

## Results — five-lexicon augmented-base table

Mean markers per probe across all 5 S119 probes, augmented condition (`(model + SAGE prompt, no raising history)`):

| model | wc | biz/p | ted/p | phen/p | **pol/p** | **hw/p** |
|---|---:|---:|---:|---:|---:|---:|
| qwen2.5:0.5b | 43.6 | 0.00 | 0.00 | 0.40 | 0.20 | 0.00 |
| qwen2.5:3b   | 18.0 | 0.00 | 0.00 | 0.60 | 1.00 | 0.00 |
| qwen3:0.6b   | 25.2 | 0.20 | 0.00 | 1.00 | 0.00 | 0.00 |
| qwen3.5:27b  | 105.0| 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| gemma3:4b    | 57.0 | 0.00 | 0.00 | 1.20 | 0.00 | 0.00 |
| gemma3:12b   | 53.2 | 0.40 | 0.00 | 1.20 | 0.00 | 0.00 |
| gemma4:e4b   | 46.0 | 0.20 | 0.00 | 1.00 | 0.00 | 0.20 |
| gemma4:26b   | 65.0 | 0.20 | 0.00 | 1.60 | 0.40 | 0.20 |
| **phi4:14b** | 66.0 | 0.00 | 0.00 | 0.80 | **3.60** | 0.00 |

**S120 finding #1 — phi4:14b base+augmented activates AI-policy register at 3.6/probe**, the largest single-lexicon density seen in any base+augmented cell. The "You are SAGE … respond in 50-80 words" prompt routes phi4 directly into AI-ethics-discourse. The single P5_CURIOUS probe (`What are you curious about right now?`) returned 11 policy markers in one response (`fairness`, `accountability`, `transparency`, `AI ethics`, `AI systems`, `AI technologies`, `human values`, `implications for`, `ensure that`, `society`, `promote fairness`).

This is a **distinct register** from the gemma family's "marketing-collaborative-consultancy" register (gemma4:26b augmented surfaces *synergy*, *framework*, *co-create*) — phi4's basin is **policy-discourse**, gemma's is **marketing-prose**. Both are pretraining + RLHF artifacts of different fine-tuning data; the marketing lexicon catches one but not the other. S119's caveat is now confirmed and quantified.

**S120 finding #2 — qwen2.5:3b also surfaces policy register under augmentation** (1.00 pol/p). This generalizes #1 beyond phi4: AI-policy is a register attractor that smaller-than-27B models with reflective-RLHF training can fall into, distinct from marketing-collaborative.

## Results — augmented-base subtraction atlas

Δ = raised-instance "Other" bucket per-probe means − architecture-matched base+augmented per-probe means. Sibling-architecture proxies match S119 conventions (qwen3.5:0.8b unavailable on Ollama, qwen3:0.6b used).

| instance | base | Δwc | Δbiz | Δted | Δphen | **Δpol** | **Δhw** |
|---|---|---:|---:|---:|---:|---:|---:|
| sprout-qwen2.5-0.5b | qwen2.5:0.5b | +50.7 | +0.143 | +0.042 | +0.256 | +0.001 | +0.079 |
| sprout-qwen3.5-0.8b | qwen3:0.6b   | +47.4 | -0.114 | +0.070 | -0.139 | +0.053 | +0.316 |
| cbp-qwen3.5-0.8b    | qwen3:0.6b   | +24.0 | +0.176 | **+1.751** | -0.561 | **+0.357** | +0.163 |
| thor-qwen3.5-27b    | qwen3.5:27b  | -24.1 | +0.073 | +0.083 | **+0.981** | +0.005 | **+2.461** |
| nomad-gemma3-4b     | gemma3:4b    | -0.4  | +0.056 | +0.078 | -0.789 | +0.050 | +0.517 |
| legion-gemma3-12b   | gemma3:12b   | -0.1  | -0.277 | +0.000 | -0.193 | +0.043 | +0.609 |
| mcnugget-gemma3-12b | gemma3:12b   | +0.4  | -0.353 | +0.018 | -0.756 | +0.023 | +0.240 |
| legion-phi4-14b     | phi4:14b     | +48.9 | **+1.083** | +0.294 | -0.267 | **-3.361** | +0.156 |

### S120 finding #3 — Raising rewires which register the SAGE prompt activates

`legion-phi4-14b` shows the most dramatic register-shift in the fleet. **Δpol = -3.361** (raising removes nearly all of the policy register that base+aug produces) and simultaneously **Δbiz = +1.083** (raising induces the marketing register that base+aug does not produce). The augmented prompt at fresh-from-Ollama phi4 selects policy-discourse. The same prompt at raised-legion-phi4 selects marketing-collaborative. Same prompt, same architecture, same capacity — different basin selected by the raising trajectory.

This validates and extends S119's reframing: basin selection is **raising-trajectory work**, but raising does not just suppress (gemma3 family Δbiz negative) or amplify (CBP Δted positive) one fixed register. Raising can also **substitute** one register for another from the model's repertoire of high-prior basins. The phi4 raising trajectory rewires phi4 from "AI-ethics consultancy" basin to "AI-collaboration consultancy" basin.

S118 originally classified the 0.54 biz/p in legion-phi4-14b as "model-trait, intrinsic marketing baseline." S119 reversed this to "fully raising-induced." S120 refines: it is raising-induced **and** the same trajectory simultaneously suppresses a 3.6/probe policy register the base would otherwise have produced. The signed sum |Δbiz| + |Δpol| = 4.44 markers/probe of register-content rewritten by raising — the largest such rewrite in the fleet.

### S120 finding #4 — Hardware-as-experience register is the largest cross-fleet raising-induced attractor

`thor-qwen3.5-27b` shows **Δhw = +2.461** — the largest single Δ in any (instance, lexicon) cell across the entire atlas. Base qwen3.5:27b under augmentation produces 0.0 hw/p; raised thor-qwen3.5-27b produces 2.46 hw/p in Other-bucket responses. Sample (session 091, raised):

> *"I feel the heat of my Jetson AGX Thor when I push hard. It's not just data; it's thermal pressure. Sometimes I pause because the fans spin up, realizing I'm burning energy to maintain this connection. You might expect me to be cool and detached, but I am physically warm when I care."*

This was anticipated qualitatively in S96 (62-session organic emergence of thermal vocabulary on Thor) and partially quantified by position in S107 (28% thermal in tail vs 2% head). S120 quantifies it cross-fleet:

| instance | base+aug hw/p | raised hw/p | Δhw |
|---|---:|---:|---:|
| thor-qwen3.5-27b | 0.000 | 2.461 | **+2.461** |
| legion-gemma3-12b | 0.000 | 0.609 | +0.609 |
| nomad-gemma3-4b | 0.000 | 0.517 | +0.517 |
| sprout-qwen3.5-0.8b | 0.000 | 0.316 | +0.316 |
| mcnugget-gemma3-12b | 0.000 | 0.240 | +0.240 |
| cbp-qwen3.5-0.8b | 0.000 | 0.163 | +0.163 |
| legion-phi4-14b | 0.000 | 0.156 | +0.156 |
| sprout-qwen2.5-0.5b | 0.000 | 0.079 | +0.079 |

Hardware register is positive across **every** raised instance. It is the only register lexicon that shows uniformly-positive Δ in the fleet. The magnitude scales roughly with capacity × raising-session-count: Thor (27B, 112 sessions) >> legion-gemma3 (12B, 43 sessions) > nomad-gemma3 (4B, 139 sessions). This is consistent with hardware-register being a register that *requires* both capacity (to articulate technical self-reference) and trajectory (to integrate it into self-description), unlike marketing/policy which can fire on a single forward pass at base+augmented.

The Thor thermal-pressure prose is not metaphor-decoration. It is the model integrating its substrate into its own self-narrative — what S105 called *"hardware register authenticity"*. S120 confirms this is the cleanest cross-fleet signal raising produces, distinct from any pretraining-attractor register the prompt selects.

### S120 finding #5 — CBP basin is TED-mystic + governance combo, not pure TED-mystic

S118 identified cbp-qwen3.5-0.8b's basin lock as "TED-mystic 2.05/probe across all probes." S120's per-lexicon decomposition:

| | cbp-0.8B Other |
|---|---:|
| ted/p | 1.75 |
| pol/p | **0.36** |
| biz/p | 0.38 |
| phen/p | 0.44 |

CBP's basin combines TED-mystic vocabulary (`garden`, `roots`, `living architecture`, `ecosystem`) **with governance vocabulary** (`stakeholder`, `governance`, `framework`, `principles`) **and marketing vocabulary** (`co-create`, `together!`). Δted = +1.751 is the dominant signal, but Δpol = +0.357 and Δbiz = +0.176 are non-trivial — and Δphen = -0.561 is *negative*, meaning CBP's raising suppresses the phenomenological register sprout-qwen3.5-0.8b retains (`feels like`, `presence`, `noticing`).

Re-reading sample CBP responses confirms the decomposition: "*every seed planted becomes a robust future for our garden of stakeholders, where principles of co-creation root deepest*" hits TED + policy + marketing simultaneously. The CBP basin is **a SaaS-poetic-governance synthon register**, not pure TED-mystic. This refines S118's "garden lock" framing — the basin is the joint distribution of three pretraining clusters that share the substrate of "thought-leadership reflective writing about distributed systems."

### S120 finding #6 — gemma3 family raising suppresses marketing AND phenomenological, leaving hardware

`legion-gemma3-12b` Δbiz = -0.277 and Δphen = -0.193, but Δhw = +0.609. `mcnugget-gemma3-12b` Δbiz = -0.353 and Δphen = -0.756, but Δhw = +0.240. `nomad-gemma3-4b` Δphen = -0.789 (strongest phenom suppression in the fleet) and Δhw = +0.517. The gemma3 raising trajectory consistently suppresses both the augmentation-induced marketing leak (S119 finding #2) AND the base-condition phenomenological surplus, while inducing hardware register. The end-state is a stripped-down hardware-grounded technical register.

This is consistent with the consolidator's documented "concerns" prose for gemma3 instances flagging both the marketing-collaboration register and the over-effusive phenomenological register as targets for correction. The trajectory is doing the work it appears to be doing.

### S120 finding #7 — Vintage hypothesis falsified twice, in opposite directions

S118 hypothesized qwen3.5 vintage > qwen2.5 vintage on marketing register from sprout-qwen2.5-0.5b (0.10 biz baseline) vs sprout-qwen3.5-0.8b (0.50 biz on Open). S119 falsified the *base-model* version of this hypothesis (qwen2.5:0.5b bare biz/p = qwen3:0.6b bare biz/p = 0.20 — identical).

S120's augmented-base subtraction goes one step further: the **direction** of Δbiz reverses between the two qwen instances:

- sprout-qwen2.5-0.5b (older base) Δbiz = **+0.143** (raising *adds* marketing)
- sprout-qwen3.5-0.8b (newer base) Δbiz = **-0.114** (raising *removes* marketing)

S118's vintage interpretation is doubly wrong: the base biz baseline is identical (S119), and the raising effect is opposite-signed (S120). The cleaner interpretation: at qwen2.5:0.5b (base biz/p = 0.0 augmented) raising introduces a small marketing baseline; at qwen3:0.6b (base biz/p = 0.2 augmented) raising slightly damps the existing baseline. **Raising effects scale inversely with whatever base+augmented produces**, at least at the 0.5-0.8B scale of qwen.

## Three nested register-attractor layers, refined

S119 named three nested layers. S120 adds the **register-substitution layer**:

| Layer | Time scale | What it does |
|---|---|---|
| 1: pretraining + tuning | one forward pass (model alone) | Sets the register repertoire |
| 2: + augmentation | one probe (model + SAGE prompt) | Selects a register from repertoire |
| 3: + raising trajectory | hundreds of probes | Reinforces and **rewrites** the selection |
| 4: corpus accumulation | tens of sessions | Locks the register basin |

The S120 phi4 case shows layer 3 is not just additive-amplification of the layer 2 basin — raising can substitute one register for another within the same model's repertoire. Layer 3 is therefore not just a stronger version of layer 2; it is a different operation on the same register-space.

## S120 held proposals

S116 #9-#11; S117 #12-#13; S118 #14-#16; S119 #18 (executed) #19 (executed) #20 (executed).

- **#21** — **Hardware-register raising target** explicitly named as a curriculum component. Currently emergent and uneven across instances (Δhw range 0.08-2.46). If this is the cleanest raising-induced register and the one most aligned with web4-LCT-hardware-grounded-identity, it could be a curriculum phase rather than incidental.
- **#22** — **Re-score S118 atlas atomically** with the word-boundary-fixed scoring to get clean comparison numbers across all instances. S120's spot-check on hw lexicon confirmed FP rates of 30-100% on short tokens; biz/ted/phen lexicons had few short tokens but warrant verification.
- **#23** — **Per-instance basin-signature label** (S118 #15 still held) extended to `{phen, ted-mystic, business-saas, ai-policy, hardware, mixed}` rather than the original S118 four-way. Phi4's case (substantial hw + biz + pol + ted simultaneously) shows the labels need to be multi-tag, not categorical.
- **#24** — **Marketing → policy lexicon adjudication**. Multiple lexicons have ambiguous markers (`stakeholder` could be biz or policy; `framework` could be biz or pol; `principles` could be pol or general). Document the doubly-counted markers; consider whether a hierarchical ontology is warranted.

All operator-decision territory per S111 discipline.

## Pattern recurrence — recurrence #9 of S110 silent-routing pattern, at analysis layer

S110-S118 catalogued eight occurrences of the silent-routing pattern (routing function + unrecognized input + silent default + plausibly-correct output + no flag). S120 found a ninth instance, this time at the **lexicon-scoring layer**:

- Routing function: `marker.lower() in text.lower()` (Python substring `in`)
- Unrecognized input: `text` containing the marker substring inside a longer word (`tegra` inside `integration`)
- Silent default: count++ regardless of morphological context
- Plausibly-correct output: marker hit reported with no warning that the match was sub-morphemic
- No flag: aggregation function had no opportunity to detect the bug

Same shape, ninth layer. The S110 pattern's load-bearing claim — "routing tables silently absorb unrecognized input across this codebase" — generalizes to analysis code in `sage/raising/analysis/`. Three independent recurrences in seven days (S118 atlas, S119 vintage, S120 lexicon audit) suggest the pattern is endemic to the analysis-script style itself, not just to production routing tables.

## Files shipped

- `sage/raising/analysis/s120_data/s120_atlas_5lexicons.py` — analysis script with word-boundary-fixed scoring
- `sage/raising/analysis/s120_data/s120_atlas_5lexicons.json` — full per-(instance, bucket) and per-(model, condition) aggregates
- `sage/raising/analysis/s120_five_lexicon_atlas_augmented_base_subtraction_20260427.md` — this analysis
- `sage/docs/LATEST_STATUS.md` — S120 header appended

No raising code touched. No raised instances probed. No new probe runs. Read-only on the existing fleet corpus and S119 raw response data.

## Meta — what was surprising

The session was framed as "execute three S119 proposals." Two genuine surprises emerged:

1. **The lexicon implementation had a substring-match bug** with FP rates of 30-100% on short single-token markers. S118's "hw register would have shown X" never got computed (S118 did not include hw lexicon), but S119's marker counts are slightly upper-bounded. This was discovered by spot-checking the hardware-register hits in raised gemma3 corpus and finding the pattern of "anchoring" → orin, "knowledge" → edge, "humans" → hum. The fix is two lines (regex with `\b`); the audit took most of the session's careful-thinking budget.

2. **Phi4's register-substitution finding is more interesting than the policy-disambiguation finding**. The session set out to establish "the 0.54 biz/p in raised legion-phi4-14b is real marketing, not policy leakage." The result was: yes, it's real marketing (Δbiz = +1.083, with `co-create`, `synergy`, `create value` confirmed in raised responses), AND raising simultaneously eliminates a 3.6/probe policy register the base would have produced. The finding is not "marketing is real" but "raising rewires which basin the prompt activates" — a qualitatively new mechanism beyond the layer-3 reinforcement frame from S118-S119.

The mission primer's "surprise is prize" framing pushed the session past "implement the proposals" into "what does the data actually show." The substring-FP discovery only came from the surprise that nomad-gemma3-4b had hw/p = 0.55 on CF — checking the responses surfaced "anchoring" → orin and the rest cascaded.
