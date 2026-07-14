#!/usr/bin/env python3
"""Goal signal — tell the model HOW CLOSE it is, not just WHAT CHANGED.

The high-productivity never-solvers (re86, sb26, wa30, ls20, tr87, sk48)
have 80-97% productive actions but never solve. They have motor skills
but no compass. This module provides the compass.

Three signal types, each addressing a different gap:

1. **Repetition detector**: "You've done L/R/L/R 8 times — that's an
   oscillation, not progress. Try a new action or direction."

2. **Plateau detector**: "Your pixel signal has been flat at ~50px/step
   for 20 steps — you're churning, not converging. Change strategy."

3. **Win-condition framing (HONEST)**: the win_condition is the cortex's
   own unconfirmed HYPOTHESIS, not ground truth — on held-out games we
   don't know the goal. So we report it as a hypothesis and report the
   frame-change % as ACTIVITY (how much was altered), explicitly NOT as
   goal-progress (a winning and a losing move both change pixels). We do
   NOT fabricate goal-progress or emit canned goal-structure advice — a
   false signal that reads as guidance is worse than silence. dp directive
   2026-05-20: "honest goal is a major requirement."

These inject into the revision prompt alongside the existing spatial
progress block. They're context engineering — no architecture change.
Signals 1 (oscillation) and 2 (plateau) derive from real execution data
and are honest by construction.
"""
from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple


def detect_oscillation(execution_log: List[Dict], window: int = 12) -> Optional[str]:
    """Detect if the model is oscillating between the same actions.

    Returns a warning string if oscillation detected, None otherwise.
    """
    if len(execution_log) < window:
        return None

    recent = execution_log[-window:]
    actions = [e.get("action", "?") for e in recent]

    # Check for A-B-A-B pattern
    if len(set(actions)) <= 2:
        counts = Counter(actions)
        top2 = counts.most_common(2)
        if len(top2) == 2 and min(top2[0][1], top2[1][1]) >= window // 3:
            return (
                f"WARNING: oscillating between {top2[0][0]} and {top2[1][0]} "
                f"for the last {window} steps. This is not making progress — "
                f"try a DIFFERENT action entirely (e.g., SEL, CLICK, or "
                f"move in a direction you haven't tried)."
            )

    # Check for single-action repetition
    if len(set(actions)) == 1:
        return (
            f"WARNING: repeating {actions[0]} for {window} consecutive steps. "
            f"Try something different — this action alone won't solve the level."
        )

    return None


def detect_plateau(execution_log: List[Dict], window: int = 15,
                   threshold_px: int = 5) -> Optional[str]:
    """Detect if pixel signal has plateaued (no meaningful progress).

    Returns a warning if the last N steps all produced similar small diffs.
    """
    if len(execution_log) < window:
        return None

    recent = execution_log[-window:]
    diffs = [e.get("px_diff", 0) for e in recent]

    # All near-zero
    if max(diffs) <= threshold_px:
        return (
            f"PLATEAU: last {window} steps all produced ≤{threshold_px}px change. "
            f"Your current approach isn't working. The game state is barely changing. "
            f"Try: (1) different coordinates for CLICK, (2) a sequence you haven't "
            f"tried, or (3) an action from the WM rules you haven't used yet."
        )

    # High but flat — churning
    avg = sum(diffs) / len(diffs)
    variance = sum((d - avg) ** 2 for d in diffs) / len(diffs)
    if avg > 20 and variance < avg * 2:
        return (
            f"CHURNING: last {window} steps average {avg:.0f}px each but the "
            f"pattern is repetitive (low variance). You're moving things around "
            f"but not converging on the goal. What specific condition are you "
            f"trying to achieve? Change your sequence order, not just the actions."
        )

    return None


def ground_win_condition(win_condition: str, frame: np.ndarray,
                         initial_frame: np.ndarray) -> Optional[str]:
    """Translate win_condition into a concrete progress hint.

    Uses frame comparison to estimate progress toward the goal.
    """
    if not win_condition or win_condition == "unknown":
        return None

    # Compute what fraction of the frame has changed
    if frame.shape != initial_frame.shape:
        return None

    diff_mask = frame != initial_frame
    total_px = frame.size
    changed_px = int(diff_mask.sum())
    pct = 100.0 * changed_px / total_px

    # Check if changes are concentrated or spread
    if len(frame.shape) >= 2:
        h, w = frame.shape[:2]
        # Find the bounding box of changes
        ys, xs = np.where(diff_mask)
        if len(ys) > 0:
            y_span = int(ys.max()) - int(ys.min())
            x_span = int(xs.max()) - int(xs.min())
            concentrated = (y_span < h // 2 and x_span < w // 2)
        else:
            concentrated = False
    else:
        concentrated = False

    # HONEST framing (dp 2026-05-20, "honest goal is a major requirement"):
    #
    # On held-out games we do NOT know the win condition — `win_condition` is
    # the cortex's own unconfirmed HYPOTHESIS, and the frame-change % measures
    # how much has been ALTERED, not progress toward the goal (a winning move
    # and a losing move both change pixels). The previous version fabricated
    # goal-progress: it announced "Goal: match a target pattern" as fact and
    # emitted canned keyword advice ("are you acting on the right objects?")
    # that pretended to know the goal's structure. That is a false signal that
    # reads as guidance — worse than silence. We report only what is true.
    #
    # Honest structural feedback (oscillation, plateau) comes from
    # detect_oscillation / detect_plateau, which use real execution data and
    # are unaffected by this change.
    hint = (
        f"WIN-CONDITION HYPOTHESIS (yours — unconfirmed, you inferred it): "
        f"{win_condition[:120]}. Frame is {pct:.0f}% altered from start — this "
        f"is how much you've CHANGED, not confirmed progress toward the goal."
    )
    if concentrated:
        hint += " Your changes are spatially localized."
    # Genuinely useful meta-guidance (the embodiment-loop prior): if you can't
    # tell whether an action moved you toward the goal, you don't yet know the
    # goal — favor a cheap, reversible action that would DISAMBIGUATE it.
    hint += (
        " If you cannot tell whether your last actions moved you toward the "
        "goal, you do not yet know the goal — take a cheap action that would "
        "disambiguate it (test one variable) rather than committing to a "
        "plan built on the guess."
    )
    return hint


def generate_goal_signal(
    execution_log: List[Dict],
    win_condition: str,
    frame: Optional[np.ndarray] = None,
    initial_frame: Optional[np.ndarray] = None,
) -> str:
    """Generate a composite goal signal for the revision prompt.

    Returns a string to inject into the progress block, or empty string
    if no signal worth reporting.
    """
    signals = []

    # 1. Oscillation check
    osc = detect_oscillation(execution_log)
    if osc:
        signals.append(osc)

    # 2. Plateau check
    plat = detect_plateau(execution_log)
    if plat:
        signals.append(plat)

    # 3. Win-condition grounding
    if frame is not None and initial_frame is not None:
        wc = ground_win_condition(win_condition, frame, initial_frame)
        if wc:
            signals.append(wc)

    return "\n".join(signals)
