"""
Lean prompt builder — codification project Layer 1 integration.

Builds a ~300-400 token invoke prompt from GameWorldModel slots
instead of the 4K token prose prompt. The WM schema IS the prompt.

Usage:
    from sage.cognition.thalamic_router.lean_prompt import build_lean_prompt
    prompt = build_lean_prompt(wm, level=0, step=5, nn_hint="LEFT",
                               nn_confidence=0.8, recent_actions=[3,3,1,5,3])
"""

from typing import Dict, List, Optional, Tuple

from sage.cognition.thalamic_router.wm_schema import GameWorldModel

ACTION_NAMES = ["A0", "UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK"]


def build_lean_prompt(
    wm: GameWorldModel,
    level: int,
    step: int,
    nn_hint: str,
    nn_confidence: float,
    recent_actions: List[int],
    action_ranking: Optional[List[Tuple[int, float]]] = None,
    frame_state: Optional[str] = None,
    level_hint: Optional[str] = None,
    invoke_reasons: Optional[List[str]] = None,
) -> str:
    """Build a lean invoke prompt from WM schema slots.

    Target: ~300-400 tokens. Every token is decision-relevant.
    """
    sections = []

    # Header: game, level, step
    sections.append(f"{wm.game} level {level}, step {step}.")

    # World model (from typed slots — ~200 tokens)
    sections.append(wm.render(budget_tokens=300))

    # Level hint (if available, ~30 tokens)
    if level_hint:
        sections.append(f"Level hint: {level_hint[:150]}")

    # Frame state (computed facts, ~50 tokens)
    if frame_state:
        # Truncate to essentials
        lines = frame_state.strip().split('\n')[:5]
        sections.append("Frame: " + " | ".join(l.strip() for l in lines if l.strip()))

    # NN hint + ranking (~30 tokens)
    recent_str = " → ".join(
        ACTION_NAMES[a] if 0 <= a < len(ACTION_NAMES) else str(a)
        for a in recent_actions[-5:]
    ) or "(none)"

    hint_line = f"NN: {nn_hint} ({nn_confidence:.2f})"
    if action_ranking:
        top3 = ", ".join(f"{ACTION_NAMES[a]}({p:.2f})" for a, p in action_ranking[:3])
        hint_line += f" | Top 3: {top3}"

    sections.append(f"{hint_line}. Recent: {recent_str}")

    # Invoke reason (~10 tokens)
    if invoke_reasons:
        sections.append(f"Why invoked: {', '.join(invoke_reasons)}")

    # Response format (~20 tokens)
    sections.append(
        "Actions: UP=1 DOWN=2 LEFT=3 RIGHT=4 SEL=5 CLICK=6(x,y)\n"
        "Respond: ACTION=N[ X=x Y=y]\n"
        "One-sentence rationale."
    )

    return "\n\n".join(sections)


def load_wm_from_json(path: str) -> GameWorldModel:
    """Load a GameWorldModel from a JSON instance file."""
    import json
    from sage.cognition.thalamic_router.wm_schema import CausalRule

    with open(path) as f:
        d = json.load(f)
    rules = [CausalRule(**r) for r in d.pop("causal_rules", [])]
    return GameWorldModel(**d, causal_rules=rules)
