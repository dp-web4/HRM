"""SensorsBlock — what the model perceives this invocation.

Irreducible (priority 95). Without sensors there's nothing to reason
about. Images and text coexist in the same slot (Q4 consensus).

Sprint: Phase 5 B3
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from sage.context.mrh.base import MRHBlock, ImageAttachment


@dataclass
class SensorsBlock(MRHBlock):
    """What the model perceives this turn.

    Attributes:
        description: Short prose describing what's being perceived
            (e.g., "Two game frames side by side: LEFT=previous,
            RIGHT=current"). The lens for the attached media.
        image_attachments: List of ImageAttachment for vision-capable
            LLMs. The composer/dispatcher is responsible for routing
            these to the LLM client via its image-passing API.
        text_snippets: Non-image perceptions (e.g., for raising sessions
            or text-only game variants). Each snippet is a (label, text)
            pair.
    """
    priority: int = 95
    description: str = ""
    image_attachments: List[ImageAttachment] = field(default_factory=list)
    text_snippets: List[tuple] = field(default_factory=list)   # list of (label, text)

    def __post_init__(self) -> None:
        self.kind = "sensors"

    def render(self, budget_tokens: int) -> str:
        parts: List[str] = ["## Perception"]
        if self.description:
            parts.append(self.description)
        if self.image_attachments:
            count = len(self.image_attachments)
            labels = ", ".join(a.label or f"image_{i}" for i, a in enumerate(self.image_attachments))
            parts.append(f"{count} image(s) attached: {labels}.")
        for label, text in self.text_snippets:
            if label:
                parts.append(f"{label}: {text}")
            else:
                parts.append(text)
        if not (self.description or self.image_attachments or self.text_snippets):
            parts.append("(no perception provided this turn)")
        return "\n\n".join(parts)

    def summarize(self, budget_tokens: int) -> str:
        # ADP: terse one-liner describing what's present
        parts = []
        if self.description:
            parts.append(self.description[:120])
        if self.image_attachments:
            parts.append(f"{len(self.image_attachments)} image(s)")
        if self.text_snippets:
            parts.append(f"{len(self.text_snippets)} text")
        return "Perception: " + "; ".join(parts) if parts else "Perception: (none)"

    def minimal(self) -> str:
        n_img = len(self.image_attachments)
        n_txt = len(self.text_snippets)
        return f"Perception: {n_img} image / {n_txt} text."

    def estimate_cost(self) -> int:
        text_cost = (len(self.description) // 4 + 8
                     + sum((len(s[1]) if len(s) > 1 else 0) // 4 + 4 for s in self.text_snippets))
        image_cost = sum(a.estimated_tokens for a in self.image_attachments)
        return text_cost + image_cost
