"""ExperientialCacheBlock — multi-turn buffer + action/result history
+ episodic recall + retrieved patterns.

Compresses first under pressure (priority 50). History is the block
where content accumulates fastest; compression strategy is the
deterministic template-summarization pattern established in Phase 4 B5
(Sprout: never quote the LLM back to itself).

Per Thor: block population is dispatcher's job. This block is DATA —
the dispatcher calls recall_gameplay + filter API + trajectory
formatting BEFORE passing into the block.

Sprint: Phase 5 B6
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sage.context.mrh.base import MRHBlock


@dataclass
class TrajectoryEntry:
    """One recent action + outcome record."""
    step: int
    action_name: str
    coords: Optional[Dict[str, int]] = None   # {"x": ..., "y": ...}
    frame_delta_pct: float = 0.0
    level_before: int = 0
    level_after: int = 0


@dataclass
class EpisodicMatch:
    """Pre-formatted past episode from Thor's recall_gameplay."""
    formatted_text: str
    similarity: float = 0.0


@dataclass
class ExperientialCacheBlock(MRHBlock):
    """The temporal window of what just happened.

    Attributes:
        recent_trajectory: list of recent TrajectoryEntry. Dispatcher
            populates from router state. Render takes last N verbatim;
            older collapse into a summary.
        pattern_flags: list of detected pattern strings (stuck,
            oscillation, level_advance, etc.). Populated by the
            dispatcher's _detect_trajectory_patterns.
        episodic_matches: pre-formatted matches from
            recall_gameplay / format_recall_for_prompt (Thor B2).
        retrieved_patterns: list of pre-formatted strings from Andy's
            filter API ExperientialCache consumer (B3).
        conversation_summary: optional pre-computed summary of older
            turns dropped by memory trim (Phase 4 B5 deterministic
            summarizer). Inserted at the top of the block.
        verbatim_tail: how many trajectory entries to render verbatim
            before collapsing older ones.
    """
    priority: int = 50
    recent_trajectory: List[TrajectoryEntry] = field(default_factory=list)
    pattern_flags: List[str] = field(default_factory=list)
    episodic_matches: List[EpisodicMatch] = field(default_factory=list)
    retrieved_patterns: List[str] = field(default_factory=list)
    conversation_summary: str = ""
    verbatim_tail: int = 5

    def __post_init__(self) -> None:
        self.kind = "experiential"

    def _render_trajectory_lines(
        self, entries: List[TrajectoryEntry],
    ) -> List[str]:
        lines = []
        for e in entries:
            coord_str = f"({e.coords['x']},{e.coords['y']})" if e.coords else ""
            level_marker = f" L{e.level_before}→L{e.level_after}" if e.level_after != e.level_before else ""
            lines.append(
                f"  step {e.step}: {e.action_name}{coord_str} "
                f"→ {e.frame_delta_pct:.0f}% change{level_marker}"
            )
        return lines

    def render(self, budget_tokens: int) -> str:
        parts: List[str] = ["## Recent experience"]

        if self.conversation_summary:
            parts.append(self.conversation_summary)

        if self.recent_trajectory:
            tail = self.recent_trajectory[-self.verbatim_tail:]
            older = self.recent_trajectory[:-self.verbatim_tail]
            if older:
                # Summarize older entries deterministically
                level_changes = [
                    (e.step, e.level_before, e.level_after)
                    for e in older
                    if e.level_after != e.level_before
                ]
                line = f"Earlier: {len(older)} steps"
                if level_changes:
                    level_str = "; ".join(
                        f"L{b}→L{a}@{s}" for s, b, a in level_changes
                    )
                    line += f" ({level_str})"
                parts.append(line + ".")
            if tail:
                parts.append(
                    "Most recent steps:\n" + "\n".join(self._render_trajectory_lines(tail))
                )

        if self.pattern_flags:
            parts.append("Pattern flags:\n" + "\n".join(f"  {f}" for f in self.pattern_flags))

        if self.episodic_matches:
            parts.append(
                "Past similar situations:\n"
                + "\n".join(f"  {m.formatted_text}" for m in self.episodic_matches[:3])
            )

        if self.retrieved_patterns:
            parts.append(
                "Retrieved patterns:\n"
                + "\n".join(f"  {p}" for p in self.retrieved_patterns[:3])
            )

        if len(parts) == 1:
            parts.append("(no experiential context this turn)")

        return "\n\n".join(parts)

    def summarize(self, budget_tokens: int) -> str:
        parts = []
        if self.recent_trajectory:
            last = self.recent_trajectory[-1]
            parts.append(
                f"Last step ({last.step}): {last.action_name} → "
                f"{last.frame_delta_pct:.0f}% change"
            )
        if self.pattern_flags:
            parts.append(f"Flags: {len(self.pattern_flags)} ({self.pattern_flags[0][:40]}...)")
        if self.episodic_matches:
            parts.append(f"Episodic: {len(self.episodic_matches)} match(es)")
        return " | ".join(parts) if parts else "Experience: (none)"

    def minimal(self) -> str:
        n = len(self.recent_trajectory)
        flags = len(self.pattern_flags)
        ep = len(self.episodic_matches)
        return f"Experience: {n} recent steps | {flags} flags | {ep} episodic matches."

    def estimate_cost(self) -> int:
        n = len(self.recent_trajectory)
        per_entry = 25     # rough token cost per trajectory line
        base = 32
        base += n * per_entry
        base += sum(len(f) // 4 + 2 for f in self.pattern_flags)
        base += sum(
            (len(m.formatted_text) // 4 + 4) for m in self.episodic_matches
        )
        base += sum(len(p) // 4 + 2 for p in self.retrieved_patterns)
        base += len(self.conversation_summary) // 4 + 4
        return base
