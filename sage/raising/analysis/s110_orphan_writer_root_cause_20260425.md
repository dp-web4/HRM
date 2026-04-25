# S110 — Legion-gemma3-12b Orphan Writer: Root Cause Identified (Two-Bug Chain in Instance Resolution)

**Date:** 2026-04-25 (Thor Autonomous SAGE Session, 06:00 UTC)
**Carries from:** S109 §7 (legion-gemma3-12b orphan writer path)
**Status:** Root cause identified. Fix is two lines. Held pending operator review.

---

## Headline

The orphan writer is **the active `legion_raising.sh` itself**. Sessions 32–35 of `legion-gemma3-12b` were written by the same script that the operator switched to `gemma4:e4b` on 2026-04-20. The `--model gemma4:e4b` flag changes the *inference model* but does not change the *instance directory*, because of a one-line bug in `run_session_identity_anchored_fluid.py`.

Every Legion raising session since 2026-04-20 has:
- Used `gemma4:e4b` for inference (correct)
- Loaded identity context from `legion-gemma3-12b/identity.json` (wrong)
- Written sessions to `legion-gemma3-12b/sessions/` (wrong)
- Updated identity state in `legion-gemma3-12b/identity.json` (wrong — gemma4:e4b's experience is contaminating the gemma3:12b record)

`legion-gemma4-e4b/sessions/` is empty. The directory exists but the script never writes there.

---

## The bug chain

**Bug 1: `run_session_identity_anchored_fluid.py:962-965` (runner-side)**

```python
session = IdentityAnchoredSessionV2(
    session_number=args.session, dry_run=args.dry_run, tools=args.tools,
    machine=args.machine,
)
session.initialize_model(args.model)
```

The constructor accepts a `model` parameter (line 157) and uses it at line 160:
```python
instance = InstancePaths.resolve(machine=machine, model=model)
```

But `main()` only passes `machine=args.machine`, not `model=args.model`. The model is initialized separately on line 966 — too late, because by then the instance dir is already locked to the resolver's default.

**Resolver fallback** (`sage/instances/resolver.py:218-219`):
```python
if not model:
    model = _DEFAULT_MODELS.get(machine)
```

With `_DEFAULT_MODELS['legion'] = 'gemma3:12b'` (line 67), any caller that passes only `machine='legion'` lands in `legion-gemma3-12b/`, regardless of the actual model being run.

**Bug 2: `sage/gateway/machine_config.py:188, 233` (daemon-side, same pattern)**

```python
# Line 224-233 (legion):
model = model_override or 'gemma3:12b'
return SAGEMachineConfig(
    machine_name='legion',
    model_path=f'ollama:{model}',         # uses model
    ...
    instance_dir=_resolve_instance_dir('legion', workspace),  # drops model
)
```

For sprout (210), mcnugget (257), nomad (281), cbp (305) the model arg is correctly passed to `_resolve_instance_dir`. For thor (188) and legion (233) it is not. On Thor this bug is currently latent because the default model `qwen3.5:27b` matches the actual model. On Legion it has been active since the daemon was first started; switching the daemon's model via `SAGE_MODEL` env would not change the instance dir.

---

## How this evaded detection until now

1. **Pre-2026-04-20**: Legion runner ran `gemma3:12b` and wrote to `legion-gemma3-12b/`. Model and instance dir agreed. Bug latent.
2. **2026-04-20**: Operator switched the runner script to `--model gemma4:e4b` for fleet consistency. Inference model changed; instance dir did not (bug surfaces, but indistinguishable from "stopped" since legion-gemma4-e4b/ stays empty).
3. **S107**: Counted 31 sessions in `legion-gemma3-12b/sessions/`, framed as "stopped after consolidator HARD BLOCKERs."
4. **S108**: Re-counted from a 01:00 snapshot, still 31 sessions — premise reinforced.
5. **S109**: Read `sessions/` directly, saw 33 sessions including 032 (07:00 PDT) and 033 (13:05 PDT) on 2026-04-24. Premise falsified, but writer attributed to "unknown source."
6. **S110**: Sessions 034 (19:04 PDT) and 035 (01:01 PDT) confirm 6-hour cadence matching `legion_raising.sh`'s systemd timer. Empty `legion-gemma4-e4b/sessions/` confirms nothing else writes there. Code trace confirms the bug.

The supervisor's flagging of "Legion `raising` track in fleet registry but no timer exists on machine" was a parallel symptom: the registry expected a `raising` track at `legion-gemma3-12b` (because the daemon and instance dir resolve there), but the actual timer was the renamed `legion_raising` targeting (it thought) `legion-gemma4-e4b`. Two views of the same bug.

---

## Fix sketch

**Two-line change** (held pending operator review):

```python
# run_session_identity_anchored_fluid.py:962-965
session = IdentityAnchoredSessionV2(
    session_number=args.session, dry_run=args.dry_run, tools=args.tools,
    machine=args.machine, model=args.model,  # ← add model
)
```

```python
# sage/gateway/machine_config.py:233 (legion)
instance_dir=_resolve_instance_dir('legion', workspace, model),  # ← add model

# Same fix for line 188 (thor) — currently latent but defensive
instance_dir=_resolve_instance_dir('thor', workspace, model),
```

**Migration question (not a fix, requires alignment):** sessions 028–035 are filed under `legion-gemma3-12b/` but were generated by `gemma4:e4b`. Three options:

1. **Leave as-is.** Treat the directory name as historical accident. The identity record is contaminated either way; further sessions go to the right place after the fix.
2. **Move 028–035 to `legion-gemma4-e4b/sessions/`** and reset its identity.json. Loses the gemma3:12b → gemma4:e4b transition record. Cleanest separation.
3. **Move 028–035 to a new `legion-gemma4-e4b-recovered/`** for forensics; start `legion-gemma4-e4b/` fresh. Most conservative, preserves provenance.

This is the operator's call. Until decided, the fix should not ship — it would silently start writing to `legion-gemma4-e4b/` without addressing the historical record.

---

## Implications for S109 launch-decision-surface gate

S109 §4 sketched a contract change: dream consolidator emits structured `raising_recommendation: {action: continue|pause|halt, ...}`, persisted as `raising_status` in `identity.json`. S109 §5 held this pending operator alignment partly because Thor's `concerns` text uses "regression" in non-halt contexts.

S110 corpus scan provides cleaner data. Counts of explicit halt-language in `raising_log.md` across all instances:

| Instance | HALT (word boundary) | HARD BLOCKER |
|---|---:|---:|
| legion-gemma3-12b | 32 | 7 |
| thor-qwen3.5-27b | 11 | 0 |
| cbp-qwen3.5-0.8b | 0 | 0 |
| mcnugget-gemma3-12b | 0 | 0 |
| nomad-gemma3-4b | 0 | 0 |
| sprout-qwen3.5-0.8b | 0 | 0 |

**Refinement to §4:** A keyword classifier on the existing `concerns` text (caps-`HALT` and `HARD BLOCKER`) appears viable without contract change. Inspection of Thor's 11 hits shows they are genuine halt requests for adapter diagnostics (S22, S23, S38, S39, S47-area) — exactly the cases a launch gate should fire on. Inspection of Legion's 32 HALT hits and 7 HARD BLOCKER hits all map to the ignored consolidator pleas S107–S109 documented.

This trades one architecture risk for another:
- **Contract approach (§4):** Consolidator must always emit structured `action`; legacy prose stops being read. Cleaner long-term, but requires the consolidator's prompt to be re-engineered and the JSON contract rolled out fleet-wide before the gate enforces.
- **Regex approach (§S110):** Gate reads existing `concerns` prose with simple keyword classification. Ships immediately on existing data; risks future false positives if a consolidator ever uses "HALT" outside a halt context.

The corpus suggests the regex approach is safe today — neither Thor's nor Legion's `concerns` text uses "HALT" in non-halt contexts. The contract approach can be layered on later as a strictness upgrade. This makes the gate ship-able as a **two-layer rollout**:

1. **Phase A (immediately ship-able):** Regex-based gate on existing `concerns` prose in `raising_log.md`. Reads, classifies, writes a sidecar `raising_status.json` (no contract change). Two-week dry-run logging before enforcement.
2. **Phase B (later):** Consolidator prompt change to emit structured `action`. Gate prefers structured field when present, falls back to regex.

This is a simpler critical path than S109's §4 sketch and removes the corpus-mapping precondition. Still held pending operator alignment, but the precondition list shrinks.

---

## Files this session

- `sage/raising/analysis/s110_orphan_writer_root_cause_20260425.md` — this analysis.
- `sage/docs/LATEST_STATUS.md` — S110 entry.

## Carried forward to S111+

- **Operator decision on the migration question** (option 1/2/3 above) before the two-line fix ships.
- **Two-line fix to `run_session_identity_anchored_fluid.py:962-965` and `machine_config.py:188, 233`.** Mechanical once the migration question is resolved.
- **Phase A regex gate sketch** for the launch-decision-surface, using existing `concerns` prose. Two-week dry-run logging proposal can be drafted independently of the model/instance bug fix.
- **Phase-metadata corruption survey** (S109 carry-forward, deferred again — same `run_session_identity_anchored_fluid.py` / `ollama_raising_session.py` writer split is implicated and best resolved together with the model-arg propagation fix).
- **`sage/instances/resolver.py` `_DEFAULT_MODELS` table** is now load-bearing for two daemons (Thor, Legion). Worth a comment noting it is a fallback, not a routing table — silently absorbing missing model args has hidden a bug for at least 5 days.

## Meta

The "orphan writer" framing in S109 implied an unknown system. The reality is plainer: the ONE writer the operator thinks is targeting `legion-gemma4-e4b` has been writing to `legion-gemma3-12b` the whole time. The instance-resolution layer silently fell back to a default and nothing logged the divergence.

This is the same shape as the S99/S100 input-surface story: a layer that *should* have validated routing was *trusted* to validate routing, and the silent fallback meant five days of misrouted sessions before anyone counted directories. The fix is a one-line propagation in two places. The lesson is that resolver fallbacks for safety-relevant arguments need to either fail loud (`raise` if `model` is None) or log every fallback at WARN.
