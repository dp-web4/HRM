#!/usr/bin/env python3
"""
Router event stream — observable by metacog (Nomad) and training pipelines.
===========================================================================

Mirrors the pattern used in ``sage.cognition.working_memory.Event``:
lightweight dataclass, ``to_dict()`` for JSON serialization, ``kind``
drawn from a closed vocabulary.

The router pipeline is a heavy event source (one decision per tick on every
machine) so the schema stays tiny on purpose — no heavy payloads, just a
kind, ids, a timestamp, and an optional human-readable reason.

Spec: phase2/brain-arch/thalamic-router-prd.md §2.7
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Set


# ──────────────────────────────────────────────────────────────────────
# Event kind vocabulary
# ──────────────────────────────────────────────────────────────────────
# The first seven kinds mirror working_memory.Event for observer-side
# compatibility (a shared on_event callback can accept either).
# The remaining kinds are router-specific.

WM_COMPATIBLE_KINDS: Set[str] = {
    "write",
    "update",
    "evict",
    "clear",
    "tick",
    "consolidate",
    "warn",
}

ROUTER_KINDS: Set[str] = {
    "route",           # a decision was produced
    "dispatch",        # a decision was actually executed (shadow mode: not emitted)
    "veto",            # a decision was blocked (metacog / budget / unknown plugin)
    "coerce_noop",     # invalid output coerced to noop per PRD §3.3 rule 5
    "record_write",    # a RouterRecord was persisted
    "record_backfill", # outcome backfilled onto an existing record
}

VALID_EVENT_KINDS: Set[str] = WM_COMPATIBLE_KINDS | ROUTER_KINDS


@dataclass
class Event:
    """Mutation / decision event — observable by metacog.

    Fields mirror ``sage.cognition.working_memory.Event`` so a single
    ``on_event`` callback can subscribe to both streams.

    Attributes:
        kind: One of ``VALID_EVENT_KINDS``. Unknown kinds are rejected in
            ``__post_init__`` to keep the vocabulary closed.
        slot_id: Optional opaque id — a WM slot id, a record id, a plugin
            name; whatever makes sense for the kind.
        slot_type: Optional type tag — WM slot type, router action, plugin
            tier; whatever makes sense for the kind.
        timestamp: Wall-clock seconds (``time.time()``).
        reason: Optional free-text explanation for the event.
    """

    kind: str
    slot_id: Optional[str]
    slot_type: Optional[str]
    timestamp: float
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in VALID_EVENT_KINDS:
            raise ValueError(
                f"unknown event kind {self.kind!r}; "
                f"expected one of {sorted(VALID_EVENT_KINDS)}"
            )
        if not isinstance(self.timestamp, (int, float)):
            raise TypeError(
                f"timestamp must be numeric, got {type(self.timestamp).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        return cls(
            kind=data["kind"],
            slot_id=data.get("slot_id"),
            slot_type=data.get("slot_type"),
            timestamp=float(data["timestamp"]),
            reason=data.get("reason"),
        )
