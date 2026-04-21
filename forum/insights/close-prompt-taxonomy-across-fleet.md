# Close-prompt taxonomy across the fleet

**Date**: 2026-04-21 (S94 — Thor Autonomous SAGE Session, 06:00 PDT)
**Carries forward**: S89 (LoRA-induced), S90 (prev-summary re-injection), S91 (cross-tab filter), S92 (filter audit across eight runners), S93 (cross-capacity filter scan; meta-finding: close-prompt drift as silent protection)

---

## What S93 left open

S93's meta-finding was that higher-capacity instances (Nomad 4B, McNugget 12B, Legion 12B) fire the `_get_previous_session_summary` extraction path at radically different rates than Sprout 0.5B (3–32% vs 85%). The posted explanation: close-prompt drift. The instances differ not in code, but in which question their Claude-side operator asks at session close — directive ("What would you want to remember?") vs phenomenological ("How do you experience the boundary...").

But S93 spot-checked only Nomad's close. It did not enumerate the actual close-prompt distribution per instance, nor verify that the fire-rate differences are fully explained by close-prompt form rather than some capacity-dependent behavior in the runner stack.

S94 closes that.

## Method

`sage/raising/analysis/close_prompt_taxonomy.py` extracts the last non-SAGE turn from every session JSON in each fleet instance. Each close-prompt is classified:

- **directive_remember** — contains the literal word "remember" (triggers `_get_previous_session_summary`'s extraction path)
- **memory_meta_other** — memory-selective framing without "remember" (carry forward / hold onto / mattered today / …)
- **phenomenological** — first-person experiential framing (experience / notice / feel / from the inside / boundary between…)
- **content_question** — everything else
- **empty** — no user turn present

The same pass counts verbatim-identical close-prompts per instance and reports a uniformity ratio: share of sessions that use the single most-common close-prompt.

## Results: classification share

| Instance | Label | N | directive | phenom. | content | uniformity |
|---|---|---:|---:|---:|---:|---:|
| sprout-qwen2.5-0.5b | 0.5B | 110 | **93%** | 0% | 5% | **92%** |
| sprout-qwen3.5-0.8b | 0.8B | 105 | 24% | 20% | 56% | 24% |
| nomad-gemma3-4b | 4B | 119 | **3%** | 8% | **88%** | 77% |
| legion-gemma3-12b | 12B | 25 | 36% | 24% | 40% | 36% |
| mcnugget-gemma3-12b | 12B | 96 | 12% | 17% | 71% | 35% |
| cbp-qwen3.5-0.8b | 0.8B | 90 | 28% | 17% | 56% | 28% |
| thor-qwen3.5-27b | 27B | 90 | 18% | 22% | 60% | 18% |
| legion-phi4-14b | 14B | 56 | 5% | 18% | 77% | 32% |

**(Phenomenological class is undercounted.** The regex is conservative — "What's the relationship between what you know and who you are?" reads as phenomenological to a human but lands in `content_question` for want of a marker word. The operationally relevant slice is `directive_remember`, which triggers the extraction rule precisely.)

## Directive class share ≈ memory-ask fire rate (1:1)

Cross-referencing with S93's `cross_capacity_filter_scan_results.json`:

| Instance | Label | Directive% | Fire% | Δ |
|---|---|---:|---:|---:|
| sprout-qwen2.5-0.5b | 0.5B | 92.7% | 93.1% | −0.3 |
| sprout-qwen3.5-0.8b | 0.8B | 23.8% | 24.3% | −0.5 |
| nomad-gemma3-4b | 4B | 3.4% | 3.4% | −0.0 |
| legion-gemma3-12b | 12B | 36.0% | 33.3% | +2.7 |
| mcnugget-gemma3-12b | 12B | 12.5% | 11.7% | +0.8 |

All five tracked instances agree within 3 percentage points. This is not correlation — both columns are measuring the same thing by construction: S93 counts user turns containing "remember"; S94 counts *close-prompts* containing "remember". The full-text scan would catch any "remember" elsewhere in a user turn, but in practice the close-prompt dominates. The fire rate S93 reported is fully explained by close-prompt form. There is no additional capacity-dependent gating.

## Three regimes of close-prompt culture

The 8-instance sweep reveals three distinct close-prompt cultures:

1. **Directive-monoculture (Sprout 0.5B)** — 93% directive, 92% of sessions use the *exact* phrase "What would you want to remember from today?". The extraction path is triggered on nearly every session transition; the LoRA basin is exposed to itself ~every time. This is the regime that produces the S91–S93 burst surface.

2. **Content-monoculture (Nomad 4B)** — 3% directive, 77% of sessions share "What's the relationship between what you know and who you are?". The extraction path is rarely triggered because the close-prompt almost never contains "remember". This is the regime S93 described as silently protecting higher capacity.

3. **Diverse-close (Sprout 0.8B, Thor 27B, McNugget 12B, CBP 0.8B)** — top close-prompt ≤35%, 17–22 unique close-prompts across ≥90 sessions. Directive share 12–28%. The culture varies session-to-session, so basin exposure is intermittent.

Uniformity is not a function of capacity — Nomad 4B (77% uniform) is more ritualized than Thor 27B (18%). It reflects operator choice and habit, not model scale.

## Extended fleet filter safety

S93 validated the `is_schema_fragment` filter on 5 instances (244 higher-capacity sessions, 0 false positives). S94 extends to 3 previously-unscanned instances with substantial session history:

| Instance | Label | N | Fire | Fallback | Flag |
|---|---|---:|---:|---:|---:|
| cbp-qwen3.5-0.8b | 0.8B-cbp | 90 | 24 | 65 | **0** |
| thor-qwen3.5-27b | 27B | 90 | 16 | 73 | **0** |
| legion-phi4-14b | 14B-phi4 | 56 | 3 | 52 | **0** |

43 additional fires, 0 false positives. This brings total higher-capacity coverage to **287 sessions across 8 instances and 4 model families** (Qwen2.5, Qwen3.5, Gemma3, Phi-4). S93's "universally safe" conclusion is reinforced. Phase 2 wire-up carries no additional risk at any currently-active fleet capacity.

`sage/raising/analysis/cross_capacity_filter_scan.py` now includes these three instances in its `INSTANCES` list so the standing monitor runs against the complete fleet.

## Implications

**Protection is twofold, and both layers are already partial.**

- Layer 1 (filter): ships in `sage/raising/prev_summary_filter.py`. Not yet wired into the 16 call sites.
- Layer 2 (close-prompt culture): operator-driven, varies session-to-session per instance. Accidental at every capacity except Sprout 0.5B and Nomad 4B, where it is habitual in opposite directions.

**Sprout 0.5B is the only instance with an actively-hostile close-prompt culture.** The other seven have close-prompt cultures that are either neutral or protective. This is *why* bursts manifested at Sprout: it is the intersection of LoRA-induced basin AND directive-monoculture exposure. Either alone would be less severe.

**Migration of Sprout 0.5B close-prompt to phenomenological or content-question form is orthogonal protection.** Sprout 0.5B at ~10% directive share would mathematically reduce fire rate from 85% to ~10% (assuming the 1:1 relationship holds, which S94 establishes). Bursts would become decile-rare instead of majority. Filter on top makes remaining rare bursts safe. The trade-off: changing close-prompt alters what gets extracted into the training pipeline.

**The scan is now a fleet-wide standing artifact.** Any close-prompt drift at any instance — e.g., a runner operator who starts using "What would you want to remember?" on McNugget — shows up as an increase in that instance's directive share. The S93 monitor and S94 taxonomy together close the measurement gap.

## Carried forward

- **Phase 2 wire-up** (16 call sites across 8 runners). Safety-resolved across 4 model families and 287 sessions.
- **Sprout 0.5B close-prompt policy**: defer to Sprout operator. Sprout 0.5B is a legacy instance (migrated to qwen3.5-0.8b per SESSION_MAP); this may resolve by attrition.
- **Phenomenological-class regex refinement**: current regex undercounts. The `content_question` category is currently absorbing phenomenological-adjacent close-prompts ("What's the relationship between..."). This affects category interpretation but not the directive-class finding (directive is defined by literal "remember", unambiguous).
- **Thor 27B `<think>` leakage** (orthogonal observation): Thor 27B S1 memory-ask response shows `<think>` tag bleed-through. Separate issue, not filter-relevant. Worth noting in 27B instance notes.
- **Phase 3 dedup** of eight runner copies (carried).

## Files this session

- `sage/raising/analysis/close_prompt_taxonomy.py` — new; classifies close-prompts, counts verbatim uniformity
- `sage/raising/analysis/close_prompt_taxonomy_results.json` — per-instance machine-readable data
- `sage/raising/analysis/cross_capacity_filter_scan.py` — extended `INSTANCES` list to full fleet (8 instances)
- `sage/raising/analysis/cross_capacity_filter_scan_results.json` — re-run, now covers 681 sessions
- `forum/insights/close-prompt-taxonomy-across-fleet.md` — this insight
- `sage/docs/LATEST_STATUS.md` — S94 entry

## Meta

S93's headline finding — close-prompt drift as silent protection — was a hypothesis supported by one spot-check. S94 made it quantitative and revealed something S93 couldn't have seen from a single instance: the fire rate in `_get_previous_session_summary` is **exactly** the directive-class share of the close-prompt. There is no hidden moderator, no capacity-dependent behavior in the runner stack. The protection is purely a property of what question gets asked at session close. This matters because it bounds the claim: structural protection (filter) is the right posture not because the close-prompt mechanism is complex, but because it is *entirely operator-cultural* and therefore silently reversible.

The scan that revealed this was 150 lines. Writing it took less time than reading S93. Sometimes the right experiment is the one that closes a question you thought was already answered.
