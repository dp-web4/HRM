"""MRHBlock base class + MRHContext composer.

Sprint: Phase 5 B1 + B9
Design consensus: forum/phase-5-mrh-context-architecture-design.md
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from sage.context.mrh.identity import IdentityBlock
    from sage.context.mrh.sensors import SensorsBlock
    from sage.context.mrh.effectors import EffectorsBlock
    from sage.context.mrh.mechanics import MechanicsBlock
    from sage.context.mrh.experiential import ExperientialCacheBlock
    from sage.context.mrh.metabolic import MetabolicBlock
    from sage.context.mrh.task import TaskBlock


# Conservative token-counting heuristic: ~4 chars/token on English prose.
# Real tokenizer counts differ; this is for budget planning only. Blocks
# that care about exact counts can override estimate_cost.
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


class BlockRenderMode(Enum):
    """How a block is rendered this composition."""
    FULL = "full"          # ATP form: render() at full budget
    SUMMARY = "summary"    # ADP form: summarize() under pressure
    MINIMAL = "minimal"    # Never-omit floor: one-line indicator


@dataclass
class MRHBlock(ABC):
    """One slot in the model's epistemic horizon for a forward pass.

    LENS, NOT DESCRIPTION (Sprout's rule):
        Block content shapes what the model sees, not what the model IS.
        "You are playing a game. Goal: reach the gem." (lens: orients)
        "You are SAGE, a SAGE..." (description: crystallizes on small models)

    POPULATE, THEN RENDER (Thor's constraint):
        Block population happens in the dispatcher (I/O, API calls,
        database queries). render() and summarize() are pure string
        formatting with no side effects. This keeps render latency
        predictable and testable.

    ATP/ADP forms (Web4):
        render() = ATP: full fidelity, takes full budget
        summarize() = ADP: compressed lossy summary
        minimal() = never-omit floor line (cheap, always renders)

    Priority semantics:
        100 = irreducible (Sensors, Effectors, Task by default)
         50 = default
          0 = always drops first
        should_drop(pressure) returns True when priority < pressure*100.
    """
    priority: int = 50
    kind: str = ""

    @abstractmethod
    def render(self, budget_tokens: int) -> str:
        """ATP form: full-fidelity render within the given token budget.

        May truncate lossily if content exceeds budget. Returns a
        renderable string; empty string is allowed for blocks with
        no meaningful content this turn (rare; prefer minimal()).
        """

    @abstractmethod
    def summarize(self, budget_tokens: int) -> str:
        """ADP form: compressed summary for when render doesn't fit.

        Must be substantively smaller than render() but still carry
        the block's essential information. Typical: 1–3 lines.
        """

    def minimal(self) -> str:
        """Never-omit floor: one-line status indicator.

        Used when priority-based pressure would otherwise drop the block
        but consensus says "never omit, always summarize." Default
        implementation returns the block kind; subclasses override with
        something more informative.
        """
        return f"[{self.kind}]: present"

    def estimate_cost(self) -> int:
        """Expected ATP (full-render) token cost.

        Used by the composer for budget allocation. Default uses a
        cheap render+count heuristic; subclasses override when they
        can estimate more cheaply or more accurately.
        """
        try:
            return _estimate_tokens(self.render(budget_tokens=10_000))
        except Exception:
            return 0

    def should_drop(self, pressure: float) -> bool:
        """Return True if this block should drop below priority threshold.

        NOTE: 'drop' here means "fall through to minimal() line" — the
        block never disappears entirely (never-omit rule). The term is
        retained for symmetry with Web4 MRH framings where the horizon
        has a priority boundary.

        Semantics: pressure is [0, 1]. A block drops when its priority
        falls below pressure * 100. Priority 100 never drops (irreducible).
        Priority 0 always drops (but still renders minimal()).

        pressure: value in [0, 1]. 0 = no pressure, 1 = extreme.
        """
        return self.priority < pressure * 100.0


# ───────────────────────────────────────────────────────────────────
# Context
# ───────────────────────────────────────────────────────────────────


@dataclass
class MRHContext:
    """The complete MRH for one forward pass.

    System prompt vs user turn split:
      - System: stable identity + world model (Identity, Mechanics, Effectors)
      - User: current situation + narration (Task, Sensors, Metabolic, Experiential)

    This mirrors the three-party conversation pattern (Phase 4): system
    prompt is loaded once per session, user turn varies per invocation.
    The composer handles the split via the `is_system_block` field on
    each block type.

    See `compose()` for the assembly semantics.
    """
    identity: "IdentityBlock"
    sensors: "SensorsBlock"
    effectors: "EffectorsBlock"
    mechanics: "MechanicsBlock"
    experiential: "ExperientialCacheBlock"
    metabolic: "MetabolicBlock"
    task: "TaskBlock"

    # Budget hints (can be overridden per-call)
    system_budget_tokens: int = 8000
    user_budget_tokens: int = 4000

    def blocks(self) -> List[MRHBlock]:
        """All blocks in the context, in priority order (highest first)."""
        return sorted(
            [self.identity, self.sensors, self.effectors, self.mechanics,
             self.experiential, self.metabolic, self.task],
            key=lambda b: -b.priority,
        )

    def _render_block(
        self, block: MRHBlock, budget_tokens: int, pressure: float,
    ) -> str:
        """Pick ATP / ADP / minimal form based on budget and pressure.

        Strategy:
          - If block's estimated cost fits budget: render() (ATP).
          - Else if summary fits budget: summarize() (ADP).
          - Else: minimal() (never-omit floor).
        """
        cost = block.estimate_cost()
        if cost <= budget_tokens and not block.should_drop(pressure):
            return block.render(budget_tokens)
        # Under pressure: try summary
        summary = block.summarize(budget_tokens)
        if _estimate_tokens(summary) <= budget_tokens:
            return summary
        # Last resort: one-line floor
        return block.minimal()

    def _assemble(
        self, which_blocks: List[MRHBlock], budget_tokens: int,
    ) -> str:
        """Assemble a set of blocks into one text buffer within budget.

        Allocates tokens proportional to priority, then renders each.
        If total cost exceeds budget, lowest-priority blocks fall
        through to summary / minimal first.
        """
        if not which_blocks:
            return ""

        total_priority = sum(b.priority for b in which_blocks)
        if total_priority <= 0:
            return ""
        total_cost = sum(b.estimate_cost() for b in which_blocks)
        pressure = max(0.0, min(1.0, (total_cost / max(budget_tokens, 1)) - 1.0))

        parts: List[str] = []
        remaining = budget_tokens
        for block in which_blocks:
            share = max(
                128,   # every block gets at least this many tokens
                int(budget_tokens * block.priority / total_priority),
            )
            share = min(share, remaining)
            text = self._render_block(block, share, pressure)
            cost = _estimate_tokens(text)
            remaining -= cost
            parts.append(text)
            if remaining <= 0:
                break

        return "\n\n".join(p for p in parts if p)

    def compose(
        self,
        system_budget_tokens: Optional[int] = None,
        user_budget_tokens: Optional[int] = None,
    ) -> Tuple[str, str]:
        """Render the MRH into (system_prompt, user_turn).

        Blocks are routed based on a convention:
          - Identity, Mechanics, Effectors → system prompt (stable per session)
          - Task, Sensors, Metabolic, Experiential → user turn (varies per invocation)

        Budget is split per the context's system/user budgets (defaults
        or arguments). Priority ordering within each half.
        """
        sys_budget = system_budget_tokens if system_budget_tokens is not None else self.system_budget_tokens
        usr_budget = user_budget_tokens if user_budget_tokens is not None else self.user_budget_tokens

        system_blocks: List[MRHBlock] = sorted(
            [self.identity, self.mechanics, self.effectors],
            key=lambda b: -b.priority,
        )
        user_blocks: List[MRHBlock] = sorted(
            [self.task, self.sensors, self.metabolic, self.experiential],
            key=lambda b: -b.priority,
        )

        system_prompt = self._assemble(system_blocks, sys_budget)
        user_turn = self._assemble(user_blocks, usr_budget)

        return system_prompt, user_turn

    def swap_recommendations(self) -> List[str]:
        """Collect swap recommendations from all blocks.

        Currently only MetabolicBlock emits recommendations (Nomad B1),
        but the interface is extensible — any block can implement
        get_swap_recommendations() to suggest state-triggered swaps
        to the dispatcher.
        """
        recs: List[str] = []
        for block in self.blocks():
            fn = getattr(block, "get_swap_recommendations", None)
            if callable(fn):
                recs.extend(fn())
        return recs


# ───────────────────────────────────────────────────────────────────
# Image attachment helper (used by SensorsBlock + MRHContext consumers)
# ───────────────────────────────────────────────────────────────────


@dataclass
class ImageAttachment:
    """One image attachment to be sent alongside the text prompt."""
    png_bytes: bytes
    label: str = ""                 # optional: "prev_frame", "current_frame"
    estimated_tokens: int = 800     # rough vision-LLM estimate


def collect_image_attachments(ctx: MRHContext) -> List[ImageAttachment]:
    """Pull images from whichever block carries them (typically Sensors).

    Convention: any block may expose `.image_attachments: List[ImageAttachment]`.
    Currently only SensorsBlock does so, but the collector allows future
    blocks (e.g., a MediaBlock) to participate.
    """
    images: List[ImageAttachment] = []
    for block in ctx.blocks():
        attachments = getattr(block, "image_attachments", None)
        if attachments:
            images.extend(attachments)
    return images
