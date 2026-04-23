#!/usr/bin/env python3
"""
Adaptive prompt — context depth shaped by what the lookahead reveals.

The lookahead output IS the signal for how much world model to include:
- All actions blocked → click game → include click targets, skip movement strategy
- LEVEL ADVANCE in lookahead → trivial decision → minimal prompt
- Multiple equivalent actions → need world model to differentiate
- Clear winner → lean prompt, just confirm

The prompt is a question shaped by what kind of question it is.
"""

import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ACTION_NAMES = ["A0", "UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK"]

# Cache world models per game
_wm_cache: Dict[str, str] = {}
_wm_full_cache: Dict[str, str] = {}


def _load_wm(game_family: str, full: bool = False) -> str:
    """Load world model — brief or full depending on need."""
    cache = _wm_full_cache if full else _wm_cache
    if game_family in cache:
        return cache[game_family]

    for root in [
        Path(os.environ.get("SHARED_CONTEXT_DIR", "")),
        Path.home() / "ai-workspace" / "shared-context",
    ]:
        wm_path = root / "arc-agi-3" / "world-models" / f"{game_family}.md"
        if wm_path.exists():
            text = wm_path.read_text()
            if full:
                # Full: Objects + Rules + Win Condition + Strategy
                wanted = ["## Objects", "## Rules", "## Win Condition", "## Strategy"]
                sections = {s: [] for s in wanted}
                current = None
                for line in text.splitlines():
                    if line.startswith("## "):
                        current = line if line in wanted else None
                    elif current:
                        sections[current].append(line)
                result = "\n".join(
                    f"{s}\n{chr(10).join(sections[s]).strip()}"
                    for s in wanted if sections[s]
                )[:1000]
            else:
                # Brief: Rules + Win Condition + Strategy, prioritizing action descriptions
                sections = {}
                current = None
                for line in text.splitlines():
                    if line.startswith("## "):
                        current = line
                    elif current in ("## Rules", "## Win Condition", "## Strategy"):
                        sections.setdefault(current, []).append(line)
                # Keep lines that describe what actions DO (contain ACTION, SEL, CLICK, LAUNCH, etc)
                action_lines = []
                other_lines = []
                for s in ["## Rules", "## Win Condition", "## Strategy"]:
                    for line in sections.get(s, []):
                        stripped = line.strip()
                        if any(kw in stripped.upper() for kw in ["ACTION", "SEL", "CLICK", "LAUNCH", "MOVE", "SELECT", "PRESS"]):
                            action_lines.append(stripped)
                        elif stripped:
                            other_lines.append(stripped)
                # Action descriptions first, then other rules
                result = " ".join(action_lines[:5] + other_lines[:3])[:400]

            cache[game_family] = result
            return result

    cache[game_family] = ""
    return ""


def classify_situation(lookahead_text: str) -> str:
    """Classify the current situation from lookahead output.

    Returns one of:
    - 'trivial': LEVEL ADVANCE available — just pick it
    - 'click_game': all directional actions blocked, need click targets
    - 'clear_winner': one action is clearly best (much bigger change)
    - 'equivalent': multiple actions produce similar results — need world model
    - 'mixed': some actions work, some don't — normal play
    """
    lines = [l.strip() for l in lookahead_text.split('\n')
             if l.strip() and not l.strip().startswith('===') and not l.strip().startswith('(')]

    if "LEVEL ADVANCE" in lookahead_text:
        return "trivial"

    blocked = 0
    active = []
    for line in lines:
        lower = line.lower()
        if "no change" in lower or "trivial" in lower or "blocked" in lower:
            blocked += 1
        elif any(name in line for name in ["UP:", "DOWN:", "LEFT:", "RIGHT:", "SEL:"]):
            # Extract pixel count
            import re
            px_match = re.search(r'(\d+)px', line)
            if px_match:
                active.append(int(px_match.group(1)))

    if blocked >= 4:  # 4+ of 5 directional actions blocked
        return "click_game"

    if len(active) >= 2:
        max_px = max(active)
        min_px = min(active)
        if max_px > min_px * 3:
            return "clear_winner"
        if max_px < min_px * 1.5:
            return "equivalent"

    return "mixed"


def build_adaptive_prompt(
    game: str,
    level: int,
    step: int,
    lookahead_text: str,
    click_targets: str,
    nn_hint: str,
    nn_confidence: float,
    recent_actions: List[str],
    special_signals: List[str],
) -> str:
    """Build prompt with depth adapted to the situation."""

    situation = classify_situation(lookahead_text)
    recent_str = " → ".join(recent_actions[-5:]) if recent_actions else "(none)"

    signals = ""
    if special_signals:
        signals = "\n" + "\n".join(f"  ! {s}" for s in special_signals)

    # TRIVIAL: LEVEL ADVANCE available — minimal prompt
    if situation == "trivial":
        # Find which action produces level advance
        advance_action = "SEL"
        for line in lookahead_text.split('\n'):
            if "LEVEL ADVANCE" in line:
                for name in ["UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK"]:
                    if line.strip().startswith(name):
                        advance_action = name
                        break
        return f"""{game} L{level} step {step}. LEVEL ADVANCE available via {advance_action}!

{lookahead_text}

ACTION={ACTION_NAMES.index(advance_action) if advance_action in ACTION_NAMES else 5}
{advance_action} wins the level."""

    # CLICK GAME: directional actions blocked, need click targets
    if situation == "click_game":
        wm = _load_wm(game, full=False)
        return f"""{game} L{level} step {step}. Click-based puzzle.
{wm}

Directional actions blocked. Available click targets:
{click_targets if click_targets else "  Probe needed — try CLICK at grid intersections."}
{signals}
Recent: {recent_str}

Which click target advances the goal?
ACTION=6 X=<0-63> Y=<0-63>
<one sentence>"""

    # EQUIVALENT: multiple similar actions — need world model to differentiate
    if situation == "equivalent":
        wm = _load_wm(game, full=True)  # Full world model
        return f"""{game} L{level} step {step}.
{wm}

Actions NOW:
{lookahead_text}
{click_targets}
{signals}
Recent: {recent_str}
NN hint: {nn_hint} ({nn_confidence:.0%})

Multiple actions produce similar changes. Use the game mechanics above
to determine which one advances toward the win condition.
ACTION=<1-6>[ X=<0-63> Y=<0-63>]
<one sentence>"""

    # CLEAR WINNER or MIXED: lean prompt with brief world model
    wm = _load_wm(game, full=False)
    return f"""{game} L{level} step {step}.
{wm}

Actions NOW:
{lookahead_text}
{click_targets if click_targets else ""}
{signals}
Recent: {recent_str}
NN hint: {nn_hint} ({nn_confidence:.0%})

Which action makes the most progress?
ACTION=<1-6>[ X=<0-63> Y=<0-63>]
<one sentence>"""
