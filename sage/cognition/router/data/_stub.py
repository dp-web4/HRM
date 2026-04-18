"""
Interim record-shape stub for Track 4.

Track 1 (schemas) ships `RouterInput`, `RouterOutput`, `RouterRecord`
separately. Track 4 doesn't block on it — we use this stub until the real
dataclasses land, then delete this file.

Every use site in writer/reader/tests carries a TODO marker:
    # TODO: replace _RouterRecordStub with sage.cognition.router.RouterRecord
    #       once Track 1 merges.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class _RouterRecordStub:
    """Minimal on-disk record shape. Matches Track 1's expected envelope.

    `payload` carries the full RouterInput + RouterOutput + outcome triple
    as plain dicts; this keeps the writer agnostic to the actual dataclass
    definitions while Track 1 iterates.
    """

    record_id: str
    schema_version: str
    timestamp: float
    machine: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "machine": self.machine,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_RouterRecordStub":
        return cls(
            record_id=d.get("record_id", ""),
            schema_version=d.get("schema_version", ""),
            timestamp=float(d.get("timestamp", 0.0)),
            machine=d.get("machine", "unknown"),
            payload=d.get("payload", {}) or {},
        )
