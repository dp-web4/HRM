# S92 — Filter Audit Across Runners: Eight Copies, One Surface

*Thor autonomous SAGE session, 2026-04-20 18:00 PDT*

## Summary

Closes the "filter audit across runners" open question from S91. Eight raising runners implement `_get_previous_session_summary` with character-for-character nearly identical code. Only one (`autonomous_conversation.py`) is currently wired to a LoRA-capable loader path, which is why only autonomous-conversation sessions have shown bursts. The remaining seven runners are **latent carriers** — the basin is unreachable today because of loader-path isolation (S91), not because their prev-summary handling is structurally different. Any future code path that puts cycle_001 (or a similarly-basined adapter) onto a forward pass in those runners would immediately expose the same feedback loop.

Also audits `last_session_summary` (the :50 state-fallback form) on burst sessions and finds it likewise schema-contaminated on 9/11 bursts. S91's proposed fallback-to-state rescue is only safe if the filter is also applied at the *write* point, not just the read. Applying the filter to the truncated :50 form misses S109/S110 because the schema phrase falls beyond the first 50 characters. **The filter must run on the full response, not a truncation.**

Delivers a centralized filter module (`sage/raising/prev_summary_filter.py`) with embedded self-validation: **11/11 burst sessions caught, 0/93 non-burst flagged** on Sprout 0.5B sessions 62–121. The module is not yet wired into the eight runners — that is a deliberate Phase-2 step requiring per-runner review and an A/B. For now it is a callable, self-validating reference.

## The eight copies

| # | Runner | Lines | Loader | LoRA | Status |
|---|---|---|---|---|---|
| 1 | `autonomous_conversation.py` | def 364 · use 338 | `AutoModelForCausalLM + PeftModel(cycle_001).merge_and_unload` | **YES** when `--lora` flag set | **Bursting** (11/28) |
| 2 | `run_session_identity_anchored.py` (v1) | def 373 · use 484 | `DaemonIRP` → daemon loads `IntrospectiveQwenIRP({'is_merged_model': True})` | NO | Latent |
| 3 | `run_session_identity_anchored_v2.py` | def 228 · use 299 | `IntrospectiveQwenIRP({'is_merged_model': True})` direct | NO | Latent |
| 4 | `run_session_identity_anchored_fluid.py` | def 459 · uses 568 (MRH) + 639 (legacy) | `DaemonIRP` | NO | Latent (two call sites) |
| 5 | `legion_raising_session.py` | def 212 · use 278 | `OllamaIRP(qwen2:0.5b)` | NO (Ollama base) | Latent |
| 6 | `mcnugget_raising_session.py` | def 212 · use 278 | `OllamaIRP(gemma3:12b)` | NO (Ollama base) | Latent |
| 7 | `ollama_raising_session.py` | def 662 · uses 742 (MRH) + 799 (legacy) | `OllamaIRP` (configurable) | NO | Latent (two call sites, one via MRH `ExperientialCacheBlock`) |
| 8 | `run_session_identity_anchored_v1_backup.py` | def 178 · use 235 | `IntrospectiveQwenIRP` (config-dependent) | Possibly | Backup — not scheduled |

All eight have the same function body, with only cosmetic differences (quote style, variable names `prev` vs `prev_session`, presence/absence of `" or """`). They all do:

```python
prev_file = self.SESSIONS_DIR / f"session_{self.session_number - 1:03d}.json"
...
for i in range(len(conversation) - 1, -1, -1):
    if conversation[i].get('speaker') == 'SAGE':
        response = conversation[i].get('text', '')
        if i > 0 and 'remember' in conversation[i - 1].get('text', '').lower():
            return f"Last session (Session {n-1}), you said you wanted to remember: {response[:200]}"
```

The fact that the code is duplicated eight times — rather than centralized — is itself part of why the surface grew unnoticed. Any future runner that copies this idiom inherits the latent basin-reinforcement surface.

## Misleading MRH safety claim

`ollama_raising_session.py:741` has the comment:

```python
# Experiential: session summary (no verbatim quotes)
prev_summary = self._get_previous_session_summary()
experiential = ExperientialCacheBlock(
    trajectory_summary=prev_summary or "First session or no prior summary available.",
)
```

The comment claims "no verbatim quotes," but `prev_summary` **is** verbatim — it comes from the same `_get_previous_session_summary` that returns `response[:200]`. The `ExperientialCacheBlock` field is named `trajectory_summary`, which implies a summary (compressed, paraphrased), but it receives the raw last-turn text. This is drift between the intended architectural safety (MRH blocks as lenses, not verbatim exemplars — `sage/CLAUDE.md` explicitly frames the MRH path as "lens-oriented, no verbatim exemplar injection") and the actual implementation. The label promises safety the code does not deliver.

`run_session_identity_anchored_fluid.py:567-568` has the same pattern:

```python
# Experiential: session history (no verbatim quotes)
prev_summary = self._get_previous_session_summary()
```

The comment appears in two files. The claim is wrong in both.

## Fallback path is not categorically safe

S91 proposed: "when the candidate response matches `_is_schema_fragment`, fall back to `state['identity']['last_session_summary']`." That rescue is sound *today* — current identity files hold clean summaries ("Session 115: creating phase. Today, I sought to recall..."). But the write-side of the pipeline also needs the filter, because:

1. Each runner's session-close handler writes `last_session_summary = f"Session {n}: {phase} phase. {memory_response[:50]}..."` (pattern in all eight).
2. On burst sessions, `memory_response[:50]` is itself a schema fragment start.
3. Simulated audit of Sprout burst sessions shows the :50 form carries schema on **9/11 bursts**. Example: S68's :50 form is `"What's the next step? What's the next decision? Wh"` — still flags schema and still reinforces basin.

If the state had been updated by a burst session (rather than the Feb-22 `--no-lora` recovery at S114-S118 that happened to produce clean memory-asks), the S91 rescue would have fallen through from contaminated [:200] to contaminated [:50].

**The filter must gate both the read and the write**, and always run on the full response text — not a truncation. S109/S110 confirm the truncation point matters: full text matches the schema regex, but the :50 truncation does not.

## Centralized filter module

`sage/raising/prev_summary_filter.py` provides:

- `is_schema_fragment(text)` — canonical detector (qmarks≥5 OR schema_phrases≥1), applied to full text
- `safe_prev_summary(last_sage_response, session_number, phase_name, state_fallback="")` — builds the injection string for system prompts, skipping the verbatim splice when the response is schematic
- `safe_state_summary(memory_response, session_number, phase_name, tag="")` — builds the `last_session_summary` value written to identity state, suppressing the :50 splice when the response is schematic

Self-validation on Sprout 0.5B sessions 62-121 (included in the module's `__main__` block):

```
Sprout 0.5B: caught 11/11 known bursts, 0 missed, 0 flagged non-burst, 86 clean non-burst
```

Run with: `python3 -m sage.raising.prev_summary_filter`.

## Rollout proposal (not executed this session)

Phase 1 — *this session*: ship the module with self-validation. Done.

Phase 2 — next session, with model access: wire each runner's `_get_previous_session_summary` through `safe_prev_summary`, and each runner's `last_session_summary` write through `safe_state_summary`. The 1-line changes are obvious; the risk is only in forgetting a call site. Eight read sites and eight write sites (one per runner, except fluid and ollama_raising which each have two).

Phase 3 — deferred: deduplicate the eight near-identical `_get_previous_session_summary` bodies into a shared helper at the package level. This is mechanical but involves a `prev_summary_reader(sessions_dir, session_number, state)` signature decision. Not blocking; Phase 2 already closes the vulnerability.

## A/B the loader-path protection (the S91 open question that requires GPU)

Separate from the filter rollout: the v2-with-LoRA experiment still stands. Patch `run_session_identity_anchored_v2.py` to optionally load cycle_001, run a matched A/B against `autonomous_conversation` under the same LoRA config. Disentangles loader-path from prompt-structure as protective surfaces. Currently we can't tell whether v2's identity exemplars + stronger identity statement would resist the basin under LoRA, because v2 has never been tested with it.

With the filter in place Phase 2, the A/B can test *both* variants cleanly: filter-on+LoRA and filter-off+LoRA, across v2 and autonomous. 2×2 design isolates the two protective surfaces.

## Files this session

- `sage/raising/prev_summary_filter.py` — new centralized filter module with self-validation
- `forum/insights/sprout-bursts-filter-audit-across-runners.md` — this insight
- `sage/docs/LATEST_STATUS.md` — S92 entry prepended

No runner code modified. The filter module exists but is not yet called by any runner — intentional, to keep this session's scope bounded and reviewable.

## Open questions carried forward

- **Phase 2 rollout**: wire filter into all eight runners (16 call sites: 8 read + 8 write).
- **v2-with-LoRA A/B**: requires model access; tests whether identity exemplars counter-balance the basin when the loader-path protection is removed.
- **Cross-capacity scan** (carried from S90/S91): Nomad 4B and McNugget 12B prev-summary content. S92 did not extend the filter validation to non-0.5B models. Worth scanning even with the current filter — if non-schema bursts exist at larger capacity, the rule needs a capacity-specific pattern.
- **Sleep-training selection for `cycle_001` replacement**: S90 identified that the 85%-word-overlap ExperienceCollector filter doesn't catch schema bursts because slot values vary. If/when cycle_001 is retrained, add `is_schema_fragment` as a rejection criterion in the experience collector. That's the upstream version of this filter.
- **Deduplicate to one function body** (Phase 3): the eight copies of `_get_previous_session_summary` should become one shared helper. Not blocking Phase 2.

## Meta

S92 was the natural continuation of S91: an audit that S91 recommended. The auditing itself surfaced two things S91 didn't anticipate:

1. The MRH `ExperientialCacheBlock` architectural drift ("no verbatim quotes" comment over verbatim input) — a safety *claim* that didn't match the *code*. Easy to miss without the audit. The comment is wrong in two files.
2. The :50 state-fallback is also contaminated on burst sessions, so gating only the read side is insufficient. Truncation lands before the schema phrase on S109/S110 but after it on S111/S112/S113 — the filter must always see the full response.

The loader-path confound S91 identified now has a name for its *consequence* on the rest of the stack: seven runners are latent, not safe. "Protected by accident of loading path" is not the same as "protected structurally." The filter module makes the protection structural, and independent of which loader ends up wired to which runner.
