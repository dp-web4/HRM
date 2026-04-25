# S111 — Codification Project Layer 2 Recurs the S110 Silent-Routing Pattern

**Date:** 2026-04-25 (Thor Autonomous SAGE Session, 12:00 UTC)
**Carries from:** S110 (resolver fallback fail-loud) and the codification commits 502839d10/762137a8f/80f829bea
**Status:** Two latent bugs documented; one render quality issue. No fixes shipped — held for operator alignment, since the right fix changes architecture (plan_executor ↔ plan_bridge composition).

---

## Headline

The codification commits that landed since S110 (Layer 1 WM schema + Layer 2 plan executor + plan_bridge) work end-to-end at the schema level — `cd82.json` round-trips JSON, `lean_prompt.build_lean_prompt` produces a 401-token invoke prompt (target was 300–400). The 17.6× speedup is real, and the WM-as-prompt premise is sound.

**But the *same shape* of silent-routing-at-a-fallback that S110 documented in `InstancePaths.resolve` recurs in three new places at the cognition layer:**

1. `plan_executor._get_action_index` maps unknown `do` values to action_idx `0`, which the engine action set `GA={1..6}` does not contain. A plan step `{"do": "navigate_to", "x": 5, "y": 7}` therefore silently no-ops — `env.step` is never called, but the step is logged with `px_diff=0` and the plan pointer still advances.
2. `motor_skills/__init__.py` does *not* import `motor_skills/skills/*`. Skills auto-register via `register_skill()` only when their module is explicitly imported. So `get_skill("navigate_to")` returns `None` from a fresh process unless some upstream did the import. Verified: `python3 -c "from sage.cognition.motor_skills.registry import list_skills; print(list_skills())"` returns `[]`. Only after `import sage.cognition.motor_skills.skills.navigate_to` does it return `["navigate_to"]`.
3. `plan_executor.execute_plan` does not call `plan_bridge.step_to_invocation` at all. The bridge module exists and converts plan steps to `SkillInvocation`s correctly when the registry is populated — but no caller routes through it. Layer 2's executor and the skill bridge are two parallel implementations of the same conceptual responsibility (plan step → action) that don't compose.

The silent-noop in (1) is dormant *today* because the only populated WM (`cd82.json`) names actions `UP/DOWN/LEFT/RIGHT/SEL/CLICK` — all in `ACTION_MAP`. It will fire the moment a future game's WM, or an LLM-emitted plan, references skill names like `navigate_to`. (2) and (3) are why the bug is dormant rather than detected: even if a plan emitted `navigate_to`, the registry is empty and the bridge isn't called.

---

## What I verified

### Layer 1 works

```
$ python3 -c "from sage.cognition.thalamic_router.lean_prompt import load_wm_from_json
              wm = load_wm_from_json('sage/cognition/thalamic_router/wm_instances/cd82.json')
              print(wm.to_json() == GameWorldModel.from_json(wm.to_json()).to_json())"
True
```

`build_lean_prompt(wm, level=0, step=12, nn_hint='SEL', nn_confidence=0.72,
recent_actions=[3,3,1,5,3], action_ranking=[(5,0.42),(1,0.18),(3,0.14)],
frame_state='canvas: 22% painted | basket: position 0 (N)',
level_hint='paint top half blue', invoke_reasons=['nn_low_conf'])`
produces 1607 chars / ~401 tokens. Decision-relevant content fills the budget.

### Render budget clips Strategy mid-word

`lean_prompt.build_lean_prompt` calls `wm.render(budget_tokens=300)` at line 42. `wm_schema.render` enforces budget as `len(text) > budget_tokens * 4` (1200 chars). For cd82.json this trips at 1293 chars and truncates with `[truncated]` mid-sentence inside the Strategy slot:

```
Strategy: 1. Read target pattern colors. 2. CLICK palett[truncated]
```

Strategy is the *last* section appended in `render()` — Objects, Actions, Win, Physics, Failed, **Strategy** — and is the first thing dropped when budget is tight. It is also the most-actionable slot for the LLM. The render quality issue: char-budget enforcement is structurally biased against the most decision-relevant content.

This is not a hard bug (the prompt still works, the LLM has Physics + Win), but it is the opposite of the codification project's design intent: typed slots so each can be sized appropriately. A slot-aware budget — drop the strategy entirely before truncating mid-word, or render Strategy *before* Physics — would compose better with the schema.

### Layer 2 silent-noop verified

```
$ python3 -c "from sage.cognition.thalamic_router.plan_executor import _get_action_index
              print(_get_action_index({'do': 'navigate_to', 'x': 5, 'y': 7}))"
0
```

In `execute_plan` (line 192): `if action_idx in GA: ... fd = env.step(GA[action_idx])`. With `GA = {1: ACTION1, ..., 6: ACTION6}`, `0 not in GA`, so `env.step` is skipped. The fallthrough still executes:
- `curr_frame = np.array(fd.frame)[-1]` — same frame as before
- `px_diff = int(np.sum(curr_frame != prev_frame))` → 0
- `entry = {"action": "navigate_to", "px_diff": 0, ...}` is logged
- `expect_passed` is checked against `px_diff=0` (e.g. `frame_change > 100px` → False → recorded as failure)
- `plan_idx += 1` advances unconditionally after `repeat_count` reaches `repeat_target`

Net effect: the plan completes, no game-state changes, and the log shows zero-effect actions for every skill-named step. If an `expect` clause is present, it fails — but the failure is attributed to the *action* being ineffective, not to the *executor* not knowing the action. Same diagnostic confusion S109 had with the orphan writer ("the script stopped" vs "the script wrote elsewhere"); same root cause shape ("the layer absorbed underspecified input silently").

### plan_bridge / motor_skills auto-registration gap

`grep "from sage.cognition.motor_skills" SAGE/` shows three importers:
- `sage.cognition.profile_edge:209` — explicitly does `import sage.cognition.motor_skills.skills` to trigger registration. Correct pattern.
- `sage.cognition.motor_skills.tests.test_motor_skills:8` — same explicit import. Correct pattern.
- `sage.cognition.motor_skills.skills/__init__.py:8` — imports `navigate_to`. Auto-registration works *if* this `skills` package is imported.

`motor_skills/__init__.py` re-exports `register_skill`, `get_skill`, `execute_skill`, but does *not* `from . import skills`. So callers that do `from sage.cognition.motor_skills import get_skill; get_skill("navigate_to")` get `None` — the registry exists but is empty.

The shape: a registration mechanism that requires every importer to remember to also `import package.skills`. Same as `_DEFAULT_MODELS` — a fallback that is *load-bearing* but undocumented as such, so callers don't realize they're trusting it.

---

## Pattern recognition: routing layers and silent input absorption

S110's lesson was: "resolver fallbacks for safety-relevant arguments should fail loud (raise on missing) or log every fallback at WARN. Defaults that route data based on missing arguments are the same shape as the consolidator concerns-prose problem at the launch gate — silent acceptance of underspecified input."

S111 finds the same shape recurring in the *new* code, written *after* S110. The reason isn't that S110's lesson was missed — the codification commits don't reference instance resolution at all. It's that the codebase has no shared discipline for "validate input at routing boundaries." Each routing layer (instance resolver, action mapper, skill registry) makes the same local choice independently: silent default, no log, no raise.

### The three layers at risk

| Layer | Routing function | Silent-default behavior | When it fires |
|---|---|---|---|
| Instance | `InstancePaths.resolve` | `_DEFAULT_MODELS.get(machine)` | Caller passes machine but no model (S110: ran for 5 days) |
| Action | `plan_executor._get_action_index` | `ACTION_MAP.get(do, 0)` → noop | Plan step names a skill not in ACTION_MAP |
| Skill | `motor_skills.registry.get_skill` | returns `None` | Caller didn't import `skills/*` |

Each is a small choice on its own. Together they compose: a plan that names a skill, dispatched through a path that didn't import the skills package, executed by a plan_executor that doesn't consult the bridge — three silent fallbacks chain into "plan ran, nothing happened, log shows zero-effect actions." The diagnostic burden falls on whoever notices the px_diff=0 trail, not on the layer that absorbed the routing intent.

### Why it keeps happening

I think the root is that "routing" isn't an explicit concept in the codebase — it's a property emergent across many small dispatch tables. No one writes a "router contract" because each table looks like just-a-dict. But each table is exactly the place where intent meets implementation, and where unrecognized intent should fail loud.

The S110 fix proposed adding a comment to `_DEFAULT_MODELS` clarifying that it is a fallback. That helps the next reader of *that* table. The deeper fix is a shared idiom for routing tables, e.g.:

```python
def _route(table, key, *, fallback=None, fallback_warns=True):
    if key not in table:
        if fallback is None:
            raise KeyError(f"{table.__name__} has no entry for {key!r}")
        if fallback_warns:
            log.warning(f"{table.__name__} fell back from {key!r} to {fallback!r}")
        return table[fallback]
    return table[key]
```

— with a callsite policy that routing-table accesses go through `_route`, not direct `.get(k, default)`. Three callsites; same idiom; loud or logged everywhere.

This is sketched, not proposed for adoption. The first instance of "route" in the SAGE codebase already exists at multiple layers (`thalamic_router/frame_router.py`, `gateway/`, `instances/resolver.py`) — adding another routing primitive risks naming collision. But the operator-decision question for S111 is: do we treat these three silent-default sites as instances of a missing pattern, or as three independent local choices to fix separately?

---

## Carry-forward (S112+)

### From S110, still pending

- **Operator decision on Legion sessions 028–035 migration**. No movement. Two-line fix to `run_session_identity_anchored_fluid.py:962-965` and `machine_config.py:188, 233` is held until decided.
- **Phase A regex gate** for launch-decision-surface (S110 §2). Independent of the model-arg fix; not drafted this session.
- **Phase-metadata corruption survey** (S109 carry-forward). Not addressed.

### From S111

- **plan_executor ↔ plan_bridge composition**. The two modules duplicate the responsibility "plan step → action." Either plan_executor should call `plan_bridge.step_to_invocation` and dispatch to `motor_skills.execute_skill` for non-`None` invocations, or plan_bridge should be retired. Architectural choice — held for operator review.
- **`motor_skills/__init__.py` should import `skills/*`** (or define an explicit `register_all_skills()` and document that it must be called). Currently the registration mechanism is silently incomplete from the package interface.
- **`wm.render()` slot-aware budget**. Char-count truncation drops Strategy mid-word. Either render Strategy before Physics, or compute budget per-slot, or use a real tokenizer instead of `len*4`.
- **Routing-table discipline**. Whether the three silent-default callsites are best treated as independent fixes or as a shared idiom is a design call worth the operator's input. Three callsites is below the typical threshold for extraction, but the shape-recurrence within one week (S110 → S111) is an indicator.

### What I learned

The codification project is a real architectural step — moving from prose to typed schemas is exactly the kind of structuring work that lets a system *verify* itself. WM schema's `observe()` method is a calibrated-prediction interface in miniature: a rule predicts, reality reports, the rule's confidence updates. That's the right shape for a young mind learning physics.

But "structured at the data layer" doesn't help if the *dispatch* layer between data and effectors silently absorbs unrecognized input. The codification work is one layer; routing discipline is another. Both need the same care.

S110 said: "the same shape as the S99/S100 input-surface story — a layer that should have validated routing was trusted to validate routing." S111 says: this shape is now visible *three* times in *one week of new code*. It is the load-bearing pattern, not an isolated bug.

---

## Files this session

- `sage/raising/analysis/s111_codification_routing_silence_20260425.md` — this analysis.
- `sage/docs/LATEST_STATUS.md` — S111 entry prepended.

No code changes shipped. The findings are dormant-bug + design-call territory; both warrant operator review before patching.
