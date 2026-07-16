"""MechanicsBlock — how this world functions.

DATA, NOT STRATEGY (Sprout's constraint extended from IdentityBlock):
    Mechanics should describe what happens, not prescribe what to do.
    "Clicking colored tiles destroys them. Gravity pulls up."
    NOT "Try clicking tiles to create paths. Use gravity to navigate."
    The LLM derives strategy from data; prescriptive phrasing
    crystallizes.

Per Q7 consensus: per-game variations live here. Identity stays
mode-level.

Sprint: Phase 5 B5
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from sage.context.mrh.base import MRHBlock


# Preset mechanics profiles for runtime swap (e.g., stuck-state swap
# per Nomad's swap_recommendations mechanism).
_MECHANICS_PROFILES: dict[str, str] = {
    "stuck_escape": (
        "## Stuck-escape heuristics\n"
        "When recent actions have produced no state change:\n"
        "- If you were clicking, try movement. If moving, try clicking.\n"
        "- If a direction failed, try perpendicular (not its opposite).\n"
        "- Check whether there's an off-frame target you haven't tried.\n"
        "- Don't re-emit the exact action+coords that just failed."
    ),
    "level_transition": (
        "## Level transition\n"
        "Mechanics may shift at new levels. Re-read this block's "
        "world-model content before applying prior strategies. Colors "
        "that meant X on the previous level may mean Y now."
    ),
}


@dataclass
class MechanicsBlock(MRHBlock):
    """How the world functions for this invocation.

    Attributes:
        world_model_text: authoritative per-game mechanics markdown.
            Loaded from world-models/{game}.md
            by the dispatcher. Pure data, not strategy.
        mechanics_cluster: optional neighbor-game hint from the
            mechanics encoder (e.g., "dynamics near toy_f, toy_g").
        profile: optional runtime profile name (e.g., "stuck_escape")
            that adds a secondary section. Used for metacog-triggered
            swaps per Nomad's swap_recommendations.
        game_family: game family name, used in minimal() output.
    """
    priority: int = 75
    world_model_text: str = ""
    mechanics_cluster: str = ""
    profile: Optional[str] = None
    game_family: str = ""

    def __post_init__(self) -> None:
        self.kind = "mechanics"

    def render(self, budget_tokens: int) -> str:
        parts: List[str] = []

        if self.world_model_text:
            # Trim if over budget — but prefer to cut whole sections
            # rather than mid-paragraph truncation
            text = self.world_model_text
            budget_chars = budget_tokens * 4
            if len(text) > budget_chars:
                # Cut to last paragraph boundary within budget
                cutoff = text.rfind("\n\n", 0, budget_chars)
                if cutoff == -1:
                    cutoff = budget_chars
                text = text[:cutoff] + "\n\n(world model truncated under budget pressure)"
            parts.append(f"## Game mechanics (authoritative)\n\n{text}")

        if self.mechanics_cluster:
            parts.append(
                f"## Mechanics cluster\n\n"
                f"This game's learned dynamics signature is closest to: "
                f"{self.mechanics_cluster}.\n"
                f"Similar action-consequence patterns may apply."
            )

        if self.profile:
            profile_text = _MECHANICS_PROFILES.get(self.profile)
            if profile_text:
                parts.append(profile_text)

        if not parts:
            parts.append(f"## Game mechanics\n(no mechanics loaded for {self.game_family or 'this game'})")

        return "\n\n".join(parts)

    def summarize(self, budget_tokens: int) -> str:
        parts: List[str] = []
        if self.mechanics_cluster:
            parts.append(f"Mechanics cluster: {self.mechanics_cluster}")
        if self.world_model_text:
            # First ~150 chars of world model
            snippet = self.world_model_text.strip().split("\n\n", 1)[0][:200]
            parts.append(f"World model (abbrev): {snippet}")
        if self.profile:
            parts.append(f"Profile: {self.profile}")
        return "\n".join(parts) if parts else f"Mechanics: (none for {self.game_family})"

    def minimal(self) -> str:
        if self.game_family:
            return f"Mechanics: {self.game_family}" + (
                f" ({self.profile})" if self.profile else ""
            )
        return "Mechanics: (loaded)"

    def estimate_cost(self) -> int:
        base = (len(self.world_model_text) + len(self.mechanics_cluster)) // 4 + 16
        if self.profile and self.profile in _MECHANICS_PROFILES:
            base += len(_MECHANICS_PROFILES[self.profile]) // 4 + 8
        return base


def available_profiles() -> list[str]:
    """Return registered mechanics profile names."""
    return list(_MECHANICS_PROFILES.keys())
