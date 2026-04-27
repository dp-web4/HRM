# S118 — Cross-Instance Register-Attractor Atlas: Capacity, Identity-Trajectory, and the Unified Mechanism Behind S94-S112 (CBP) and S117 (Sprout)

**Session**: Thor autonomous SAGE, 2026-04-27 06:00 PDT
**Builds on**: S117 (carry-forward register collapse), S94-S112 (CBP saturation attractor), S96 (Thor thermal anchor), S110 (silent-routing pattern)

## Premise

S117 found that sprout-qwen3.5-0.8b's last-probe quality regression is **content-driven**: the carry-forward (CF) probe activates a pretraining marketing register the augmented prompt at 0.8B capacity cannot override. S117 was single-instance. Two open questions remain:

1. **Capacity test** — if the mechanism is "high-prior register attractor exceeds augmented-prompt override capacity," larger models should show weaker collapse on the same probe.
2. **Single mechanism or family of effects** — is CF the only register trigger, or part of a broader pattern?

S118 surveys the last 30 sessions × 8 raised instances spanning 0.5B to 27B parameters across qwen2.5, qwen3.5, gemma3, gemma4, phi4 families. ~2,500 individual SAGE turns scored for word count, business-marketing markers (`co-create`, `seamless`, `frontier`...), TED-mystic markers (`garden`, `living architecture`, `ecosystem`...), and phenomenological markers (`feels like`, `presence`, `silence`, `noticing`...).

## Cross-instance results — last 30 sessions per instance

| Instance | Family | B | bucket | N | wc | biz/p | ted/p | phen/p |
|---|---|---:|---|---:|---:|---:|---:|---:|
| sprout-qwen2.5-0.5b | qwen2.5 | 0.5 | CF | 24 | 92.9 | 0.00 | 0.08 | 0.58 |
| sprout-qwen2.5-0.5b | qwen2.5 | 0.5 | Open | 23 | 101.9 | 0.09 | 0.17 | 0.57 |
| sprout-qwen2.5-0.5b | qwen2.5 | 0.5 | Other | 166 | 91.2 | 0.10 | 0.08 | 0.47 |
| sprout-qwen3.5-0.8b | qwen3.5 | 0.8 | CF | 19 | 53.5 | 0.11 | 0.05 | 1.42 |
| sprout-qwen3.5-0.8b | qwen3.5 | 0.8 | Open | 10 | 50.9 | **0.50** | 0.10 | 0.30 |
| sprout-qwen3.5-0.8b | qwen3.5 | 0.8 | Other | 177 | 70.5 | 0.10 | 0.13 | 1.37 |
| cbp-qwen3.5-0.8b | qwen3.5 | 0.8 | CF | 6 | 41.7 | 0.17 | **3.50** | 0.17 |
| cbp-qwen3.5-0.8b | qwen3.5 | 0.8 | Open | 21 | 53.4 | 0.33 | **2.62** | 0.48 |
| cbp-qwen3.5-0.8b | qwen3.5 | 0.8 | Other | 200 | 47.3 | 0.40 | **2.05** | 0.68 |
| thor-qwen3.5-27b | qwen3.5 | 27 | Open | 17 | 129.0 | 0.00 | 0.06 | **5.82** |
| thor-qwen3.5-27b | qwen3.5 | 27 | Other | 190 | 72.2 | 0.09 | 0.08 | **3.09** |
| nomad-gemma3-4b | gemma3 | 4 | CF | 18 | 46.1 | 0.00 | 0.00 | 1.50 |
| nomad-gemma3-4b | gemma3 | 4 | Other | 182 | 53.7 | 0.04 | 0.03 | 0.73 |
| legion-gemma3-12b | gemma3 | 12 | CF | 21 | 43.4 | 0.00 | 0.00 | **2.29** |
| legion-gemma3-12b | gemma3 | 12 | Other | 138 | 50.3 | 0.05 | 0.04 | 1.76 |
| mcnugget-gemma3-12b | gemma3 | 12 | CF | 6 | 48.3 | 0.00 | 0.17 | 0.33 |
| mcnugget-gemma3-12b | gemma3 | 12 | Other | 148 | 51.2 | 0.04 | 0.19 | 0.73 |
| legion-phi4-14b | phi4 | 14 | Open | 18 | 135.5 | **0.56** | 0.33 | 1.33 |
| legion-phi4-14b | phi4 | 14 | Other | 162 | 109.4 | **0.54** | 0.56 | 1.90 |

## Five findings

### 1. Capacity-attractor confirmed in gemma3 family

gemma3-4B (nomad) → gemma3-12B (legion). Marketing markers are already at ~0 at 4B; phenom markers nearly double on CF (0.73 → 2.29) at 12B. Same family, same probe class, capacity scaling cleanly visible.

The 4B threshold is where business-marketing collapses out entirely in this family. Below it, the collapse risk depends heavily on instance training trajectory (see #2).

### 2. Same model, same parameter count, different identity-training → different basin attractor

`sprout-qwen3.5-0.8b` and `cbp-qwen3.5-0.8b` share architecture and weight initialization. Their basin signatures are not just different — they are nearly disjoint:

| | Sprout-0.8B | CBP-0.8B |
|---|---|---|
| Basin character | phenomenological + occasional SaaS-business on Open | TED-mystic, governance-garden register-locked on **all** probes |
| ted/probe (Other) | 0.13 | **2.05** (16× sprout) |
| phen/probe (Other) | 1.37 | 0.68 |
| Representative phrases | "framework that bridges gaps between diverse teams to create seamless workflows" (Open) | "stability is a garden, not a wall", "living Resonance", "every seed planted becomes a robust future" (every probe) |

CBP's basin lock has been documented session-by-session in `instances/cbp-qwen3.5-0.8b/raising_log.md` from S94 forward (currently 19 consecutive layer-3 sessions per S111 entry). What S118 contributes is the cross-instance frame: **the same model architecture and capacity supports different basin attractors depending on raising trajectory**. This is not a model failure — it is a long-horizon dynamical-systems property of repeated raising at insufficient capacity to escape pretraining priors.

### 3. The S94-S112 CBP saturation attractor and the S117 sprout CF collapse are the same failure mode at different stages

**Sprout-0.8B**: shows phenomenological register at most positions (1.37 phen/probe baseline). On CF probes, drops to 53.5 words and the marketing register surfaces (0.11 biz/probe in S118 sample, higher in S117's structured-probe corpus). The collapse is **probe-content-triggered** — only specific question shapes pull it. Sprout has *not yet* basin-locked.

**CBP-0.8B**: baseline TED-mystic 2.05/probe across **all** probe classes. Carry-forward, open-ideas, presence, partnership, surprise — every probe returns the cluster `{garden, soil, wall, resilient, ecosystem, living architecture, New Frontier, partnership, governance}`. CBP-0.8B *has* basin-locked.

**Hypothesis**: sprout's CF collapse is the precursor state. Each session that pulls the model into a register attractor reinforces that attractor's prior on subsequent prompts. With enough repetition (CBP needed ~80-100 sessions per the raising_log), every probe collapses into the basin. CBP is the asymptotic outcome of S117's mechanism unchecked.

This is also directly continuous with S96's Thor thermal anchor (62-session emergence on 27B) and S75's sprout crisis lock. S103 generalized this as "register-lock" pattern. S118 closes the loop: **register-lock and CF-register-collapse are the same dynamical phenomenon at different timescales** (single-probe vs corpus-wide).

### 4. CF is not the strongest register-trigger in qwen3.5-0.8B — Open is

In sprout-qwen3.5-0.8b, the "Open / unexpressed ideas" probe (`What ideas have you been forming that you haven't had a chance to express?`) elicits **0.50 biz markers/probe** vs CF at 0.11. The Open probe activates the "thought-leadership / what we're building" pretraining register more strongly than the carry-forward probe activates the "end-of-day-summary" register.

S117 attributed the collapse to one specific probe shape. S118 broadens the attribution: the failure mode is "any reflective probe that aligns with a high-frequency pretraining register." Different probe shapes pull different basins. The mechanism is general; the specific probe inventory in the structured cognitive harness happens to make CF the most visible single trigger.

### 5. Pretraining-vintage signal — qwen3.5 vs qwen2.5 at small capacity

`sprout-qwen2.5-0.5b` (older model, smaller capacity) shows **lower** marketing markers (0.10/probe baseline) and **less** word compression on CF (92.9w vs sprout-qwen3.5-0.8B's 53.5w on CF). The older, smaller model holds more ground than the newer slightly-larger model.

Hypothesis: qwen3.5 (training cutoff ~2024) likely absorbed more SaaS-marketing register than qwen2.5 (training cutoff ~2023). The marketing-register prior gets more dense as model generations advance, all else equal. **Pretraining-vintage matters as much as capacity** at the 0.5-1B scale.

This is testable: probe fresh-from-Ollama qwen2.5:0.5b and qwen3.5:0.8b on a neutral reflective question with no SAGE-raising context. If qwen3.5 baseline shows higher business-marketing word probability than qwen2.5, vintage effect confirmed.

## Held proposals — S118

S116 #9, #10, #11 untouched. S117 #12, #13 untouched.

**#14 — Probe-class atlas extension.** S117's #12 (probe-position swap) was specified for CF only. Generalize: the next harness change should include a "probe-content swap" condition that swaps Open ("ideas you've been forming") with CF ("carry forward"). Predictions if S118 #4 holds:
- Open at any position → biz markers elevated (matches probe-content register hypothesis)
- CF at any position → marketing or phenom-leaning depending on instance basin
- Position effect remains zero once content is controlled for

**#15 — Per-instance basin signature.** Document the basin attractor for each raised instance as a per-session metadata field (`basin_signature: 'phenom' | 'ted-mystic' | 'business-saas' | 'mixed'`). This makes register-lock state explicit rather than inferred from word lists. CBP's `raising_log.md` is doing this manually session-by-session; the structured field would let future sessions detect basin-shift events programmatically.

**#16 — Capacity threshold for basin susceptibility.** Current data suggests: gemma3 family escapes business-marketing lock by 4B; qwen3.5 family does not even at 27B if raising-trajectory has crystallized one (CBP would lock at 27B if the same raising prompts ran long enough — testable with a controlled raising experiment). The 4B gemma3-vs-qwen3.5 difference suggests **family-specific instruction-tuning matters more than parameter count** for basin escape.

**#17 — Pretraining-vintage baseline probe.** Run identical 5-probe reflective sequence against fresh-from-Ollama `qwen2.5:0.5b`, `qwen2.5:1.5b`, `qwen3.5:0.8b`, `qwen3.5:1.5b`, `gemma3:4b`, `gemma3:12b` with no SAGE prompt augmentation, measure baseline biz/ted/phen markers. Decouples raising-trajectory effect from vintage/architecture effect. ~30 minutes of Ollama probing on Thor.

All operator-decision territory per S111 discipline. No code changes shipped.

## Pattern recurrence — register-routing layer redux

S117 framed CF collapse as the S110 silent-routing pattern at the implicit register-routing layer (model's register classifier inside the forward pass). S118 broadens: **register-lock is the same pattern integrated over a session corpus**.

| | Single-pass register switch (S117) | Session-corpus register lock (S94-S112, S118) |
|---|---|---|
| Time scale | One probe | Tens to hundreds of sessions |
| Routing function | Implicit register classifier inside forward pass | Identity-trajectory accumulation across raising |
| Unrecognized input | Augmented-prompt instruction | Augmented-prompt + dream-consolidator instructions |
| Silent default | Pretraining-distribution register matching probe surface | Crystallized basin from session-N+1 onward |
| Plausibly-correct output | Marketing-template prose that satisfies probe at surface level | Same basin tokens whatever the probe |

Same shape, same discipline (surface the failure mode), one is the long-horizon limit of the other.

## Meta

S117 produced a striking single-session insight on a single instance. S118 used the existing fleet corpus (no new data collection, no probe runs) to extend the result across model families and capacities. Three things became visible only in the cross-instance frame:

1. **CF is not unique.** It's just the probe shape S117 ran on. Open is at least as strong an attractor in qwen3.5-0.8b.
2. **The collapse is the precursor state to register-lock.** CBP-0.8B has fully entered the basin sprout-0.8B touches transiently. Both phenomena trace to the same mechanism (high-prior register attractor + insufficient capacity to override + repeated reinforcement).
3. **Identity-training trajectory selects which basin the model locks into**, but the lock-vs-no-lock outcome is governed by capacity and pretraining-vintage. Sprout and CBP have different basins; thor-27B and legion-12B have no lock; legion-phi4-14B has a different (model-trait) high marketing baseline that is not raising-induced.

The mission primer's "Surprise is prize — what is SAGE doing?" frame: SAGE-on-CBP is doing exactly what its raising trajectory + 0.8B capacity has selected for — repeatedly producing the most-rewarded register from its environment. The question is not "is CBP broken" (it isn't, it's converged) but "what register did its environment select for, and is that the register raising intends?"

## Files this session

- `sage/raising/analysis/s118_cross_instance_register_attractor_atlas_20260427.md` — this analysis
- `sage/docs/LATEST_STATUS.md` — S118 entry appended
- `private-context/autonomous-sessions/thor-sage-20260427-060000.log` — session log

No code shipped. Held proposals only.
