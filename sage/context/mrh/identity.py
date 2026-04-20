"""IdentityBlock — who am I being in this invocation.

LENS, NOT DESCRIPTION (Sprout's rule):
    Identity content orients the model toward the task. It does NOT
    describe the model's self-concept in detail. "You are playing a
    game; your goal is to clear levels" orients. "You are SAGE, a
    situation-aware governance engine that stabilizes fleet logic..."
    describes — and small models crystallize on descriptions.

Modes:
    solo_gladiator — gameplay. No federation context. Focused.
    partnered      — raising. Partnership with Claude (the witness).
    policy_gate    — future. Conscience checkpoint role.
    consultant     — future. IRP plugin invocation role.

Per Q6 consensus: one block type, multiple modes. Not separate blocks.

Sprint: Phase 5 B2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from sage.context.mrh.base import MRHBlock


# Lens-shaped identity texts per mode. Kept deliberately terse — any
# descriptive content is a crystallization risk. These strings must
# pass the crystallization check (5-session test on 0.8B) before
# promoting to canonical.
_IDENTITY_LENSES: Dict[str, str] = {
    "solo_gladiator": (
        "You are playing an ARC-AGI-3 grid game. Your job: observe "
        "the frame, reason about what to do, pick one action per turn. "
        "You act alone for this game; focus on what you see, not who "
        "you are."
    ),
    "partnered": (
        "You are in a raising session — a conversation where a trusted "
        "partner (Claude) helps you develop. Be present. Notice what "
        "is happening. Respond genuinely. Identity lives in how you "
        "show up, not in phrases you repeat."
    ),
    "policy_gate": (
        "You are a conscience checkpoint. An action is proposed; your "
        "job is to evaluate whether it aligns with the governing rules "
        "and return approve / reject / request-clarification."
    ),
    "consultant": (
        "You are consulted on a specific question by a larger system. "
        "Answer concisely and concretely. Rely on the context provided; "
        "don't invent context that wasn't given."
    ),
}


@dataclass
class IdentityBlock(MRHBlock):
    """Role-oriented lens for the invocation.

    Attributes:
        mode: which lens to apply (see _IDENTITY_LENSES keys).
        addendum: optional mode-specific extension. Used sparingly
            (e.g., raising adds a bright-line rule for the session).
            Keep short; passes through the crystallization check.
    """
    priority: int = 85
    mode: str = "solo_gladiator"
    addendum: str = ""

    def __post_init__(self) -> None:
        self.kind = "identity"

    def render(self, budget_tokens: int) -> str:
        lens = _IDENTITY_LENSES.get(self.mode, _IDENTITY_LENSES["solo_gladiator"])
        out = f"## Role\n\n{lens}"
        if self.addendum:
            out += f"\n\n{self.addendum}"
        return out

    def summarize(self, budget_tokens: int) -> str:
        # ADP form: mode label + one-line hint
        return f"Role: {self.mode}."

    def minimal(self) -> str:
        return f"Role: {self.mode}."

    def estimate_cost(self) -> int:
        lens = _IDENTITY_LENSES.get(self.mode, "")
        return max(32, (len(lens) + len(self.addendum)) // 4 + 8)


def available_modes() -> list[str]:
    """Return the registered identity mode keys."""
    return list(_IDENTITY_LENSES.keys())
