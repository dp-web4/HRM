"""SNARC-driven cognitive-mode router (#52 MVP, 2026-05-09).

Design: object-space LLM mechanistic translation (2026-05-09), from a cross-model
design discussion. The router replaces the LLM's
single monolithic prompt with a switched ensemble of cognitive modes,
each anchored to one or two SNARC axes (Surprise, Novelty, Arousal,
Reward, Conflict).

The router is a SNARC SINK — reads axis values from the WM state +
recent trace, weights cognitive families by axis alignment, activates
the dominant one (with hysteresis to prevent thrashing). The affordance
graph (#50) and trace history are SNARC SOURCES.

Active modes: `perseveration_break`, `affordance_discovery`, `exploration`,
`macro_discovery`, and `commit_faith` (2026-05-23 — persist through a feedback
desert toward a plausible goal; the complement of perseveration_break, gated on
goal_plausibility; see `forum/nomad-faith-horizon-2026-05-23.md`). `anomaly` is
stubbed (needs single-event signal extraction). Each landed as a separate sprint
as the architecture matured.

Why perseveration-break first: Nomad's local sweep (2026-05-09) showed
gemma3:4b clicking obj_008 at (4, 32) 31 consecutive times on toy_a L=1
with no progress. Same failure surface dp called out in the
"reluctance-to-succeed" comment. This is the highest-frequency observed
failure mode — break-out from action loops.

Per `FLAGS.md`: gated by SAGE_COGNITIVE_ROUTER=1 (new flag).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Cognitive modes ────────────────────────────────────────────────────────

DEFAULT_MODE = "active"  # standard play loop; no special prompting


# ── SNARC signals (extracted from WM + trace) ──────────────────────────────

@dataclass
class SNARCSignals:
    """Per-step SNARC axis values extracted from current state.

    Each axis is float in [0, 1]. The router weights cognitive modes by
    axis alignment. Sources:

    - Surprise:   prediction-error magnitude. Currently approximated by
                  expectation_met=False rate over recent steps. Future:
                  use delta taxonomy (#49) to quantify "this was unexpected."
    - Novelty:    new objects appearing in graph (visibility transitions
                  to in_frame from never-seen). Currently approximated by
                  graph.next_id growth rate.
    - Arousal:    urgency / loop-detection. Approximated by consecutive
                  zero-progress steps (px_diff < threshold for N steps).
    - Reward:     recent productive actions. Approximated by px_diff > 10
                  rate over recent steps.
    - Conflict:   action-outcome mismatch. Approximated by engine_error
                  events + perseveration (same action, no change).
    """
    surprise: float = 0.0
    novelty: float = 0.0
    arousal: float = 0.0
    reward: float = 0.0
    conflict: float = 0.0
    # goal_plausibility — structural-plausibility prior on "there's a reachable
    # goal here worth committing to." Drives the commit_faith mode. NOT a SNARC
    # axis proper; it's the world-model's trust in the current goal hypothesis.
    # See `_goal_plausibility` — the swap point for membot's per-entry V3
    # (Andy/Waving-cat Q3 activity-log→T3/V3) once that glue lands.
    goal_plausibility: float = 0.0


def extract_snarc(wm: Any, recent_log: List[Dict[str, Any]],
                  prev_result: Optional[Dict[str, Any]] = None) -> SNARCSignals:
    """Extract SNARC axis values from current WM + recent execution log.

    `recent_log`: list of step records from the most recent plan execution
    (typically prev_result['execution_log']).

    Heuristic implementation; future versions tighten via #49 delta
    taxonomy + #50 affordance graph signals.
    """
    s = SNARCSignals()
    if not recent_log:
        return s

    n_steps = len(recent_log)
    px_diffs = [int(r.get("px_diff", 0) or 0) for r in recent_log]

    # Reward — productive-action rate (px_diff > 10)
    productive = sum(1 for p in px_diffs if p > 10)
    s.reward = productive / max(n_steps, 1)

    # Arousal — consecutive zero-progress steps at the END of the log
    # (the more we've been spinning, the more arousal). Caps at 1.0
    # after 10 zeros.
    trailing_zeros = 0
    for p in reversed(px_diffs):
        if p < 1:
            trailing_zeros += 1
        else:
            break
    s.arousal = min(trailing_zeros / 10.0, 1.0)

    # Conflict — engine_error rate + perseveration (same action repeated
    # with no progress).
    n_eng_err = sum(1 for r in recent_log if r.get("engine_error"))
    repeat_no_progress = 0
    for i in range(1, len(recent_log)):
        prev_action = recent_log[i - 1].get("action")
        curr_action = recent_log[i].get("action")
        if prev_action == curr_action and px_diffs[i] < 1:
            repeat_no_progress += 1
    s.conflict = min((n_eng_err + repeat_no_progress) / max(n_steps, 1), 1.0)

    # Surprise — expectation-failure rate
    exp_failures = sum(1 for r in recent_log
                       if r.get("expect") and not r.get("expect_passed", True))
    n_with_exp = sum(1 for r in recent_log if r.get("expect"))
    if n_with_exp > 0:
        s.surprise = exp_failures / n_with_exp

    # Novelty — graph next_id growth (proxy for new objects appearing).
    # Currently coarse; tighter version could track new-objects-this-frame.
    graph = getattr(wm, "level_objects", None)
    if graph is not None:
        # Newer objects (high id numbers) imply recent novelty. Compare
        # next_id to a "stable" baseline (assume baseline = first frame's count).
        # Without history tracking, just use frame_count as inverse proxy:
        # early frames = high novelty potential; later = lower
        s.novelty = min(1.0 / max(graph.frame_count + 1, 1), 1.0)

    s.goal_plausibility = _goal_plausibility(wm, recent_log)
    return s


def _goal_plausibility(wm: Any, recent_log: List[Dict[str, Any]]) -> float:
    """Structural-plausibility prior: how much should the agent TRUST that
    there's a reachable goal here worth committing unrewarded effort toward.

    ── SWAP POINT (Andy/Waving-cat Q3) ──
    Replace this heuristic with the membot WM-entry's accumulated **V3
    (Veracity/Validity)** once the activity-log→T3/V3 glue lands. The faith
    update rule is confirm/disconfirm-only: V3 rises on a terminal win, falls on
    positive disconfirmation, and is NEVER decayed during the silent unrewarded
    steps of a feedback desert (absence of confirmation ≠ disconfirmation). See
    `forum/nomad-faith-membot-q3-connection-2026-05-23.md`.

    Until then, a CONSERVATIVE coherence proxy (defaults to 0.0 so commit_faith
    never fires without positive evidence of a coherent goal-directed stretch):
    plausibility ≈ has_goal × prediction-tracking × (1 − engine-error rate).
    """
    # A faith-candidate portfolio (faith_portfolio.py), if attached to the WM, is
    # the authoritative source: best candidate's trust×plausibility IS the prior.
    pf = getattr(wm, "faith_portfolio", None)
    if pf is not None:
        try:
            raw = max(0.0, min(1.0, float(pf.best_score())))
            # P4 Path B (SAGE_GOAL_PLAUS_SMOOTH=1, default OFF) — EMA-smooth
            # instantaneous best_score so commit_faith mode selection isn't
            # silently lost to its structural co-correlation with conflict/surprise
            # spikes. Per `forum/nomad-zero-commit-faith-CORRECTION-2026-05-29.md`:
            # commit_faith's formula `+1.0*goal_plaus -0.6*conflict -0.4*surprise`
            # gets disqualified at peak goal_plaus because the same cycle that
            # raised goal_plaus (a confirm) also has high surprise. Sustained
            # high goal_plaus over a window decouples it from per-cycle spikes —
            # which aligns with the faith concept itself (sustained commitment,
            # not spike commitment). A/B-able vs raw via the flag. EMA alpha
            # tunable via SAGE_GOAL_PLAUS_EMA_ALPHA (default 0.3 → ~3-cycle effective window).
            import os as _os_p4
            if _os_p4.environ.get("SAGE_GOAL_PLAUS_SMOOTH", "") == "1":
                _alpha = 0.3
                try:
                    _alpha = float(_os_p4.environ.get("SAGE_GOAL_PLAUS_EMA_ALPHA", "0.3"))
                except (ValueError, TypeError):
                    pass
                _prev = getattr(wm, "_goal_plaus_ema", None)
                _ema = raw if _prev is None else (_alpha * raw + (1.0 - _alpha) * _prev)
                wm._goal_plaus_ema = _ema
                return _ema
            return raw
        except Exception:
            pass

    # explicit world-model confidence wins if exposed
    for attr in ("goal_plausibility", "goal_confidence"):
        v = getattr(wm, attr, None)
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
    gm = getattr(wm, "goal_manager", None)
    if gm is not None:
        v = getattr(gm, "confidence", None)
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))

    if not recent_log:
        return 0.0
    # has_goal: a win hypothesis, or at least objects to act toward
    has_goal = bool(getattr(wm, "win_hypothesis", None)
                    or getattr(wm, "win_condition", None))
    if not has_goal:
        graph = getattr(wm, "level_objects", None)
        if graph is not None:
            try:
                has_goal = len(graph.visible_nodes()) > 0
            except Exception:
                has_goal = False
    if not has_goal:
        return 0.0

    exp = [r for r in recent_log if r.get("expect")]
    track = (sum(1 for r in exp if r.get("expect_passed", True)) / len(exp)
             if exp else 0.5)            # no predictions logged → neutral 0.5
    n_eng_err = sum(1 for r in recent_log if r.get("engine_error"))
    coherent = max(0.0, 1.0 - n_eng_err / max(len(recent_log), 1))
    return round(track * coherent, 3)


# ── Mode definitions ───────────────────────────────────────────────────────

@dataclass
class CognitiveMode:
    """One cognitive mode in the switched ensemble.

    Each mode is anchored to one or two SNARC axes. The router activates
    the mode with the highest score = sum of (snarc_value × axis_weight).

    `prompt_override(base_prompt, ctx) -> str | None` returns a modified
    or replacement prompt when the mode is active; returns None to use
    the base prompt unchanged.
    """
    name: str
    description: str
    primary_axes: List[str]              # e.g., ["conflict", "arousal"]
    weights: Dict[str, float] = field(default_factory=dict)
    activation_threshold: float = 0.5    # sum of weighted axes must exceed
    prompt_override: Optional[Any] = None  # callable; None = no override


def _perseveration_break_prompt_override(base_prompt: str, ctx: Dict[str, Any]) -> str:
    """Inject perseveration-break framing into the prompt.

    Looks at recent_log to find the action(s) that have been tried with
    no progress, then prepends an explicit "you've been doing X — try
    something different" preamble.

    The base_prompt is the output from generate_plan's normal prompt
    construction; this prepends + appends to it.
    """
    recent_log = ctx.get("recent_log") or []
    if not recent_log:
        return base_prompt

    # Find recent actions with px_diff = 0 (the perseveration pattern)
    stuck_actions: Dict[str, int] = {}
    for r in recent_log[-15:]:  # last 15 steps
        px = int(r.get("px_diff", 0) or 0)
        if px < 1:
            key = r.get("action", "?")
            coords = r.get("coords")
            if coords:
                key = f"{key} at ({coords.get('x','?')},{coords.get('y','?')})"
            stuck_actions[key] = stuck_actions.get(key, 0) + 1

    if not stuck_actions:
        return base_prompt

    most_repeated = sorted(stuck_actions.items(), key=lambda kv: -kv[1])[:3]
    repeated_str = "; ".join(f'"{a}" ×{n}' for a, n in most_repeated)

    # Available action surfaces the LLM should consider instead
    avail = ctx.get("available_actions") or []
    avail_names_map = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                       5: "SEL", 6: "CLICK"}
    avail_names = [avail_names_map.get(a, f"ACTION{a}") for a in avail
                   if a in avail_names_map]

    preamble = f"""## ⚠ PERSEVERATION DETECTED — break the loop

Your recent attempts have produced ZERO state change. The same action(s) keep getting repeated without progress: {repeated_str}.

**This means**: those actions are dead-ends right now. Repeating them won't help.

**Do NOT repeat the actions listed above.** Pick a fundamentally different approach:
- Try a different ACTION class entirely ({", ".join(avail_names)})
- If you've been clicking one target, try a different target (or stop clicking entirely)
- If you've been moving in one direction, try the opposite or a perpendicular direction
- Consider that the current state might require SEL or another action you haven't tried

The most informative action right now is one that produces ANY state change, even an unexpected one. Bias toward exploration.

---

"""

    return preamble + base_prompt


_PERSEVERATION_BREAK = CognitiveMode(
    name="perseveration_break",
    description="Action loop detected — no progress for N steps. Force exploration of "
                "different action class.",
    primary_axes=["conflict", "arousal"],
    weights={"conflict": 0.5, "arousal": 0.5},
    activation_threshold=0.6,
    prompt_override=_perseveration_break_prompt_override,
)


# ── exploration mode ──────────────────────────────────────────────────────

def _exploration_prompt_override(base_prompt: str, ctx: Dict[str, Any]) -> str:
    """Inject exploration framing — fresh state, sample broadly.

    Activated when Novelty is high and Reward is low: the agent is in a
    state with little prior information. Don't commit to a hypothesis;
    gather diagnostic signal first.
    """
    avail = ctx.get("available_actions") or []
    avail_names_map = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                       5: "SEL", 6: "CLICK"}
    avail_names = [avail_names_map.get(a, f"ACTION{a}") for a in avail
                   if a in avail_names_map]

    wm = ctx.get("wm")
    n_visible = 0
    n_clickable = 0
    if wm is not None:
        graph = getattr(wm, "level_objects", None)
        if graph is not None:
            visible = graph.visible_nodes()
            n_visible = len(visible)
            n_clickable = sum(1 for n in visible if n.is_clickable)

    inv_summary = ""
    if n_visible > 0:
        inv_summary = (f"\n- {n_visible} objects visible "
                       f"({n_clickable} clickable). Try DIFFERENT objects each step "
                       f"to map what each does.")

    preamble = f"""## ↗ EXPLORATION mode — fresh state, sample broadly

You're early in this state. You don't yet know what each action does or which target rewards. **Resist the urge to commit to a single hypothesis.** Your job right now is to MAP THE TERRAIN, not solve the puzzle.

Sampling strategy:
- Cycle through the available action classes ({", ".join(avail_names)}) — at least one of each before repeating.
- Vary targets when CLICKing — DON'T click the same object twice in a row.{inv_summary}
- Each step's outcome (Δpx, level change, semantic_outcome) is a data point. Use them on the next plan.
- A productive 0px result is fine — it tells you that action did nothing here. Move on.

Once you observe a CONSISTENT cause-effect pattern, switch from sampling to exploitation.

---

"""
    return preamble + base_prompt


_EXPLORATION = CognitiveMode(
    name="exploration",
    description="Fresh state with low priors — broad action sampling instead of premature "
                "commitment to a single hypothesis.",
    primary_axes=["novelty"],
    weights={"novelty": 1.0, "reward": -0.5},  # high novelty, low reward
    activation_threshold=0.5,
    prompt_override=_exploration_prompt_override,
)


# ── affordance_discovery mode ──────────────────────────────────────────────

def _affordance_discovery_prompt_override(base_prompt: str, ctx: Dict[str, Any]) -> str:
    """Inject probing framing — predictions are off, probe systematically.

    Activated on high Surprise without high Conflict (so distinct from
    perseveration_break). The agent's mental model is wrong somewhere;
    rather than continue executing a broken plan, take diagnostic actions.
    """
    recent_log = ctx.get("recent_log") or []

    # Find specific predictions that failed
    failed_preds = []
    for r in recent_log[-10:]:
        if r.get("expect") and not r.get("expect_passed", True):
            actual = r.get("expect_actual", "?")
            failed_preds.append(f'"{r.get("expect")}" but observed: {actual[:80]}')
    failed_str = "\n  - ".join(failed_preds[:3]) if failed_preds else "(prior expectations diverged from observation)"

    wm = ctx.get("wm")
    visible_targets = []
    if wm is not None:
        graph = getattr(wm, "level_objects", None)
        if graph is not None:
            for n in graph.visible_nodes()[:8]:
                if n.is_clickable:
                    visible_targets.append(n.id)
            for n in graph.visible_nodes()[:8]:
                if not n.is_clickable and n.id not in visible_targets:
                    visible_targets.append(n.id)
                    if len(visible_targets) >= 8:
                        break

    targets_str = ""
    if visible_targets:
        targets_str = (f"\n\nSystematic probe — one click each on the next plan, "
                       f"observe each delta:\n  {', '.join(visible_targets)}")

    preamble = f"""## 🔍 AFFORDANCE DISCOVERY mode — predictions diverged, probe systematically

Your recent expectations didn't match observation:
  - {failed_str}

This means your causal model is incomplete. **Stop trying to win — start probing.** The next plan should ELICIT data, not advance the goal.

Probe principles:
- ONE distinct action per probe step. Don't repeat the same action+target — that just confirms what you already saw.
- After each step, READ the actual delta. The real signal is what changed, not what you expected.
- Probe diverse object classes — test UP/DOWN/LEFT/RIGHT/SEL once each before repeating.{targets_str}

Once you've gathered ≥3 cause-effect data points, the next plan revision will see them in the execution log and can build a reliable model.

---

"""
    return preamble + base_prompt


_AFFORDANCE_DISCOVERY = CognitiveMode(
    name="affordance_discovery",
    description="Predictions diverged from observation (Surprise high, Conflict not yet "
                "perseverative) — probe systematically to fix the causal model.",
    primary_axes=["surprise"],
    weights={"surprise": 1.0, "conflict": -0.3},  # surprise without conflict
    activation_threshold=0.55,
    prompt_override=_affordance_discovery_prompt_override,
)


# ── macro_discovery mode ───────────────────────────────────────────────────

def _macro_discovery_prompt_override(base_prompt: str, ctx: Dict[str, Any]) -> str:
    """Inject "you're on a streak" framing — name what works, keep doing it.

    Activated on high Reward + low Novelty: the agent has found a
    productive pattern. Don't break it experimenting; double down.
    Future: write the pattern back to the macro cart for cross-session
    retrieval.
    """
    recent_log = ctx.get("recent_log") or []

    # Find the most-productive recent action(s)
    by_action: Dict[str, List[int]] = {}
    for r in recent_log[-10:]:
        a = r.get("action", "?")
        coords = r.get("coords")
        key = a
        if coords:
            key = f"{a} target={coords.get('x','?')},{coords.get('y','?')}"
        by_action.setdefault(key, []).append(int(r.get("px_diff", 0) or 0))

    # Sort by total productive px_diff
    ranked = sorted(by_action.items(),
                    key=lambda kv: -sum(p for p in kv[1] if p > 10))[:3]
    streak_str = "\n  - ".join(
        f'{a} → {len([p for p in pxs if p > 10])} productive hits, total {sum(pxs)}px'
        for a, pxs in ranked
    )

    preamble = f"""## ✓ MACRO DISCOVERY mode — productive streak, name what works

Your recent plan produced consistent progress:
  - {streak_str}

**Don't break a working pattern by experimenting prematurely.** When something is working, do MORE of it.

Strategy for the next plan:
- Repeat the productive action(s) above — extend the streak. Use "repeat":N for clear repetition.
- Only deviate when the productive action stops producing change (px_diff drops to 0).
- If the streak has a clear name (e.g., "click the same target N times" → CLICK_TARGET_N), use the macro form so future plans can recognize and reuse it.

If progress continues for N more steps, the pattern is a candidate for the macro cart — the substrate that informs future games.

---

"""
    return preamble + base_prompt


_MACRO_DISCOVERY = CognitiveMode(
    name="macro_discovery",
    description="Sequence is producing consistent progress (Reward high, Novelty low) — "
                "lock in the pattern, document for cart promotion.",
    primary_axes=["reward"],
    weights={"reward": 1.0, "novelty": -0.3},  # high reward, stable
    activation_threshold=0.6,
    prompt_override=_macro_discovery_prompt_override,
)

# ── commit_faith mode ──────────────────────────────────────────────────────

def _commit_faith_prompt_override(base_prompt: str, ctx: Dict[str, Any]) -> str:
    """Inject FAITH framing — persist toward a plausible goal across a feedback
    desert. The complement of perseveration_break: there, the loop is
    DISCONFIRMED (no change / engine errors) so you must redirect; here, the plan
    is coherent and a goal is plausible but no REWARD has arrived yet, so the
    correct move is to HOLD COURSE, not redirect.

    Encodes the faith-horizon result (`forum/nomad-faith-horizon-2026-05-23.md`):
    abandon on positive disconfirmation, NOT on absence of confirmation; a silent
    unrewarded stretch is not evidence the goal is unreachable.
    """
    preamble = """## ⏳ COMMIT / FAITH mode — plausible goal, no reward yet: HOLD COURSE

Your actions are coherent (no contradictions, no error loops) and a goal here is plausible — but you haven't been rewarded yet. **That is expected.** Many tasks pay off only at completion (place ALL the pieces, then the win fires); the unrewarded steps in between are the cost of admission, not a sign you're wrong.

**Do NOT abandon or redirect just because the last steps produced no reward.** Absence of reward is not the same as disconfirmation.

- **Stay on the current plan.** Keep executing the sequence toward the goal.
- **Only change course on POSITIVE disconfirmation** — an action that does the *opposite* of what your model predicts, an engine error, or a contradiction. A silent (reward-less) step is not disconfirmation.
- If a step *does* contradict your model, that's real signal — then revise. Until then, commit.

Faith here means: invest the unrewarded steps on the hypothesis that the goal pays off at the end. Cross the desert.
"""
    # If a faith-candidate portfolio is attached, frame the current best bet +
    # the highest-value open question to resolve (directed exploration).
    _wm = ctx.get("wm")
    pf = getattr(_wm, "faith_portfolio", None)
    if pf is not None:
        try:
            best = pf.best()
            if best is not None:
                preamble += (f"\n**Current best candidate** (committing toward): "
                             f"{best.hypothesis} (trust {best.trust:.2f}, "
                             f"could-win {best.plausibility:.2f}).")
            # SAGE_WI_GOAL: the WinImaginer's concrete imagined terminal — the
            # COORDINATES the body must reach to win. Turns "hold course toward a
            # plausible goal" into "aim the body at (x,y)". The actionable goal
            # that Finding B showed was missing. Present only when the WI organ
            # is live for this game (wm._wi_terminal set by the play hook).
            _wi_term = getattr(_wm, "_wi_terminal", None)
            if _wi_term is not None:
                preamble += (f"\n**Imagined win-position** (aim the body here — the "
                             f"WinImaginer predicts the level clears when the body "
                             f"reaches this cell): x={_wi_term[0]:.0f}, y={_wi_term[1]:.0f}. "
                             f"Choose actions that move the body toward (x={_wi_term[0]:.0f}, "
                             f"y={_wi_term[1]:.0f}).")
            voi = pf.next_open_question()
            if voi is not None:
                preamble += (f"\n**Open question to resolve** (pick actions that answer it): "
                             f"{voi[1].text}")
            others = [c for c in pf.candidates.values() if best is None or c.id != best.id]
            if others:
                preamble += ("\n**Alternative candidates still in play** (don't discard — "
                             "a disconfirmation here just shifts weight to these): "
                             + "; ".join(f"{c.hypothesis}" for c in others[:3]))
        except Exception:
            pass
    preamble += "\n\n---\n\n"
    return preamble + base_prompt


# P4 Path A — commit_faith mode-weight tuning (Nomad 2026-05-30). Per
# `forum/nomad-zero-commit-faith-CORRECTION-2026-05-29.md`: commit_faith's
# weights `+1.0*goal_plaus -0.6*conflict -0.4*surprise` structurally lose at
# peak goal_plaus because the same confirm event that raises goal_plaus also
# fires the conflict/surprise axes weighted negatively. Path B (EMA-smoothed
# goal_plaus) was non-confirmed in the v34+ A/B; Path A retunes the weights
# themselves. Per-component env override lets us A/B different tunings without
# code edits. All four flags default to current values → no behavior change
# under default env. Register all four in `_BEHAVIOR_FLAGS` + `FLAGS.md`.
def _f(env: str, default: float) -> float:
    try:
        return float(os.environ.get(env, default))
    except (ValueError, TypeError):
        return default

_COMMIT_FAITH = CognitiveMode(
    name="commit_faith",
    description="Coherent plan toward a plausible goal but no reward yet (feedback "
                "desert). Persist — abandon only on disconfirmation (conflict/surprise), "
                "not on absence of reward. Plausibility-gated (the threshold-faith result).",
    primary_axes=["goal_plausibility"],
    # high plausibility drives it; conflict/surprise (disconfirmation) suppress it.
    # No reward term: faith is precisely for the unrewarded-but-undisconfirmed stretch.
    # Each weight tunable via env (Path A); defaults preserve original behavior.
    weights={
        "goal_plausibility": _f("SAGE_COMMIT_FAITH_GP_WEIGHT", 1.0),
        "conflict": _f("SAGE_COMMIT_FAITH_CONFLICT_WEIGHT", -0.6),
        "surprise": _f("SAGE_COMMIT_FAITH_SURPRISE_WEIGHT", -0.4),
    },
    activation_threshold=_f("SAGE_COMMIT_FAITH_ACTIVATION", 0.55),
    prompt_override=_commit_faith_prompt_override,
)


# ── anomaly mode (deferred to v2 — needs single-event signal extraction) ───

_ANOMALY = CognitiveMode(
    name="anomaly",
    description="(STUB) Acute single-event Surprise — large unexpected delta. Currently "
                "deferred: extract_snarc averages Surprise over recent steps, which "
                "smooths over single anomalies. Future version: track max-step delta "
                "vs running mean as a separate axis.",
    primary_axes=["surprise"],
    weights={"surprise": 1.0},
    activation_threshold=0.85,
)


# Modes registered in priority order (first-exceeds-threshold preference,
# but select_mode picks max score regardless — priority matters when scores tie).
# perseveration_break first because it addresses concrete failure;
# affordance_discovery before exploration because it's more targeted;
# macro_discovery last because it's "celebrate" not "redirect."
_MODES: List[CognitiveMode] = [
    _PERSEVERATION_BREAK,
    _AFFORDANCE_DISCOVERY,
    _COMMIT_FAITH,        # persist-through-desert; complement of perseveration_break
    _EXPLORATION,
    _MACRO_DISCOVERY,
    # _ANOMALY,  # deferred — needs single-event signal extraction
]


# ── Router ─────────────────────────────────────────────────────────────────

@dataclass
class RouterState:
    """Persistent per-game router state (lives across plan revisions).

    Hysteresis: once a mode activates, it stays active for at least
    MIN_DWELL_STEPS plan revisions even if SNARC signals would route
    elsewhere. Prevents thrashing between modes.
    """
    current_mode: str = DEFAULT_MODE
    steps_in_mode: int = 0
    last_activation_score: float = 0.0
    history: List[str] = field(default_factory=list)  # mode names

    MIN_DWELL_STEPS: int = 1                # at least N revisions in a mode before switching
    HYSTERESIS_THRESHOLD: float = 0.15      # new mode score must exceed current by this much


def select_mode(snarc: SNARCSignals,
                state: Optional[RouterState] = None) -> tuple:
    """Select the active cognitive mode given SNARC signals.

    Returns (mode_name, score, reason). When SAGE_COGNITIVE_ROUTER is
    not enabled, always returns (DEFAULT_MODE, 0.0, "router-disabled").

    Hysteresis logic (revised 2026-05-09):
    1. Score each registered mode.
    2. Compute the CURRENT-step score for the previously-active mode (if any).
       If it's now BELOW that mode's activation_threshold, the mode is
       "naturally expiring" — we let it go without margin requirement.
    3. Otherwise, MIN_DWELL_STEPS prevents mid-dwell switching, AND
       the new mode must beat current's CURRENT score by HYSTERESIS_THRESHOLD.
    """
    if os.environ.get("SAGE_COGNITIVE_ROUTER", "") != "1":
        return (DEFAULT_MODE, 0.0, "router-disabled")

    state = state or RouterState()

    # Score each enabled mode (record all, not just best)
    scores: Dict[str, float] = {}
    best_mode = DEFAULT_MODE
    best_score = 0.0
    for mode in _MODES:
        score = sum(getattr(snarc, axis, 0.0) * w
                    for axis, w in mode.weights.items())
        scores[mode.name] = score
        if score > mode.activation_threshold and score > best_score:
            best_mode = mode.name
            best_score = score

    # Compute current mode's current-step score + threshold
    current_now_score = scores.get(state.current_mode, 0.0)
    current_mode_def = get_mode(state.current_mode) if state.current_mode != DEFAULT_MODE else None
    current_threshold = current_mode_def.activation_threshold if current_mode_def else 0.0

    # Case 1: current mode "naturally expires" (its score dropped below its
    # own threshold). Allow the switch with no margin requirement.
    current_expired = (state.current_mode != DEFAULT_MODE
                       and current_now_score < current_threshold)
    if current_expired:
        # Don't apply hysteresis when current mode has expired
        return (best_mode, best_score, f"current expired (now {current_now_score:.2f} < {current_threshold})")

    # Case 2: still in MIN_DWELL_STEPS dwell window for non-default mode
    if (state.current_mode != DEFAULT_MODE
            and state.steps_in_mode < state.MIN_DWELL_STEPS):
        return (state.current_mode, current_now_score,
                f"hysteresis-dwell (in {state.current_mode} for {state.steps_in_mode} steps)")

    # Case 3: new mode must beat current's CURRENT score by margin
    if (best_mode != state.current_mode
            and state.current_mode != DEFAULT_MODE
            and best_score < current_now_score + state.HYSTERESIS_THRESHOLD):
        return (state.current_mode, current_now_score,
                f"hysteresis-margin ({best_mode}@{best_score:.2f} vs {state.current_mode}@{current_now_score:.2f})")

    return (best_mode, best_score, "score")


def get_mode(name: str) -> Optional[CognitiveMode]:
    """Look up a mode definition by name (case-insensitive)."""
    n = name.lower().strip()
    for m in _MODES:
        if m.name.lower() == n:
            return m
    return None


def update_state(state: RouterState, new_mode: str, score: float) -> RouterState:
    """Mutate state to reflect mode transition. Returns the same state for chaining."""
    if new_mode == state.current_mode:
        state.steps_in_mode += 1
    else:
        state.current_mode = new_mode
        state.steps_in_mode = 1
        state.last_activation_score = score
        state.history.append(new_mode)
    return state
