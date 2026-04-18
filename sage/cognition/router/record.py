#!/usr/bin/env python3
"""
RouterRecord — the training-data unit written by the Phase 0 pipeline.
======================================================================

Every consciousness-loop decision is persisted as a ``RouterRecord``:
``(RouterInput → RouterOutput + outcome)``. Outcome is backfilled later
by Track 6 (SNARC trajectory + RPE signal over subsequent ticks), so the
field is optional at write time.

Schema versioning: every record carries ``schema_version``. The reader
(Track 4) is schema-version-aware; additions to ``RouterInput`` /
``RouterOutput`` require a bump and a migration strategy.

Spec: shared-context/arc-agi-3/phase2/brain-arch/router-sprint-1-phase-0.md
      (Track 1 deliverables)
"""

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from sage.cognition.router.inputs import RouterInput
from sage.cognition.router.outputs import RouterOutput


# Bumped when the record schema (or any of its nested schemas) changes
# in a non-backward-compatible way. Readers must reject unknown versions
# loudly rather than silently coerce.
ROUTER_SCHEMA_VERSION = "v0.1.0"


def _default_record_id() -> str:
    """uuid4 hex — canonical record identity."""
    return uuid.uuid4().hex


@dataclass
class RouterRecord:
    """One captured routing decision, suitable for training.

    Construction defaults fill ``record_id``, ``schema_version``,
    ``timestamp``, and ``machine`` (detected from env) so callers can
    usually construct with just ``router_input`` + ``router_output``.
    """

    router_input: RouterInput
    router_output: RouterOutput

    # Identity + provenance
    record_id: str = field(default_factory=_default_record_id)
    schema_version: str = ROUTER_SCHEMA_VERSION
    timestamp: float = field(default_factory=time.time)
    machine: str = ""

    # Backfilled by Track 6 — None at write time, populated later.
    # Shape is intentionally open (Dict[str, Any]) so we can iterate on
    # the outcome schema without another record-schema bump; a nested
    # ``outcome_schema_version`` inside the dict will pin it when we
    # freeze the shape.
    outcome: Optional[Dict[str, Any]] = None

    # ──────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError("record_id must be a non-empty string")
        if not isinstance(self.schema_version, str) or not self.schema_version:
            raise ValueError("schema_version must be a non-empty string")
        if not isinstance(self.timestamp, (int, float)):
            raise TypeError("timestamp must be numeric")
        if not isinstance(self.machine, str):
            raise TypeError("machine must be str")
        if not isinstance(self.router_input, RouterInput):
            raise TypeError("router_input must be a RouterInput instance")
        if not isinstance(self.router_output, RouterOutput):
            raise TypeError("router_output must be a RouterOutput instance")
        if self.outcome is not None and not isinstance(self.outcome, dict):
            raise TypeError("outcome must be dict or None")

    # ──────────────────────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict representation.

        Nested dataclasses are expanded via their own ``to_dict`` rather
        than relying on ``dataclasses.asdict`` — this preserves the
        explicit forward-compatible read path in each nested class.
        """
        return {
            "record_id": self.record_id,
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "machine": self.machine,
            "router_input": self.router_input.to_dict(),
            "router_output": self.router_output.to_dict(),
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouterRecord":
        """Reconstruct from a dict produced by ``to_dict``.

        Schema versioning: we DO NOT reject unknown versions here; the
        caller (Track 4 dataset reader) owns that policy because it may
        want to route old records through a migration function.
        """
        return cls(
            record_id=data["record_id"],
            schema_version=data.get("schema_version", ROUTER_SCHEMA_VERSION),
            timestamp=float(data["timestamp"]),
            machine=data.get("machine", ""),
            router_input=RouterInput.from_dict(data["router_input"]),
            router_output=RouterOutput.from_dict(data["router_output"]),
            outcome=data.get("outcome"),
        )

    # ──────────────────────────────────────────────────────────────
    # Outcome backfill helper (Track 6 seam)
    # ──────────────────────────────────────────────────────────────

    def with_outcome(self, outcome: Dict[str, Any]) -> "RouterRecord":
        """Return a copy of self with ``outcome`` set.

        Non-mutating so Track 6 can re-emit a ``record_backfill`` event
        without worrying about in-place aliasing with already-flushed
        JSONL buffers.
        """
        if not isinstance(outcome, dict):
            raise TypeError("outcome must be dict")
        return RouterRecord(
            record_id=self.record_id,
            schema_version=self.schema_version,
            timestamp=self.timestamp,
            machine=self.machine,
            router_input=self.router_input,
            router_output=self.router_output,
            outcome=dict(outcome),
        )
