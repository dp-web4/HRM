"""
Metacog → cartridge writer.

Converts MetacogSignals into JSONL entries tagged with the canonical
schema for membot cartridge consumption. Each signal becomes a
retrievable pattern: "when this state signature occurred, metacog
fired X, the outcome was Y."

The writer produces JSONL that the membot cart builder converts to
embeddings. At dispatch time, the MRH ExperientialCacheBlock can
query: "have we seen this metacog pattern before? what happened?"

Entry format:
    {
        "text": "[role:metacog] [signal:perseveration] [game:toy_b] ...",
        "source": "metacog_observation",
        "game": "toy_b",
        "signal": "perseveration",
        "severity": 0.8,
        "machine": "nomad",
        "per_pattern_meta": {
            "signal": "perseveration",
            "severity": 0.8,
            "suggestion": "reframe or abandon current approach",
            ...
        }
    }

Tags follow the canonical schema from forum/canonical-tags.md:
    role:metacog, signal:{name}, game:{family}, machine:{slug},
    source:gameplay, scope:moment
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core import MetacogSignal


@dataclass
class CartridgeEntry:
    """One JSONL entry ready for membot cart building."""

    text: str
    source: str = "metacog_observation"
    game: str = ""
    signal: str = ""
    severity: float = 0.0
    machine: str = ""
    per_pattern_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "text": self.text,
            "source": self.source,
        }
        if self.game:
            d["game"] = self.game
        if self.signal:
            d["signal"] = self.signal
        if self.severity:
            d["severity"] = self.severity
        if self.machine:
            d["machine"] = self.machine
        if self.per_pattern_meta:
            d["per_pattern_meta"] = self.per_pattern_meta
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=False)


def _build_tag_prefix(
    signal: str,
    game: str = "",
    machine: str = "",
    scope: str = "moment",
) -> str:
    """Build the bracketed tag prefix for the entry text."""
    tags = [f"role:metacog", f"signal:{signal}"]
    if game:
        tags.append(f"game:{game}")
    if machine:
        tags.append(f"machine:{machine}")
    tags.append(f"scope:{scope}")
    tags.append("source:gameplay")
    return " ".join(f"[{t}]" for t in tags)


def _build_observation_text(sig: MetacogSignal, context: str = "") -> str:
    """Build the human+LLM-readable observation text."""
    parts = [f"Metacog {sig.signal} (severity {sig.severity:.1f})."]

    if sig.suggestion:
        parts.append(f"Suggestion: {sig.suggestion}.")

    # Include key evidence fields
    evidence_summary = []
    for k, v in list(sig.evidence.items())[:5]:
        if isinstance(v, float):
            evidence_summary.append(f"{k}={v:.3f}")
        elif isinstance(v, (int, str, bool)):
            evidence_summary.append(f"{k}={v}")
    if evidence_summary:
        parts.append(f"Evidence: {', '.join(evidence_summary)}.")

    if context:
        parts.append(f"Context: {context}.")

    return " ".join(parts)


def signal_to_entry(
    sig: MetacogSignal,
    *,
    game: str = "",
    machine: str = "",
    context: str = "",
    outcome: str = "",
) -> CartridgeEntry:
    """Convert a single MetacogSignal into a CartridgeEntry.

    Args:
        sig: the metacog signal
        game: game family if in gameplay context
        machine: machine slug
        context: optional description of what was happening
        outcome: optional description of what happened after the signal
    """
    tag_prefix = _build_tag_prefix(sig.signal, game=game, machine=machine)
    obs_text = _build_observation_text(sig, context=context)

    if outcome:
        obs_text += f" Outcome: {outcome}."

    text = f"{tag_prefix} {obs_text}"

    meta: Dict[str, Any] = {
        "signal": sig.signal,
        "severity": sig.severity,
        "suggestion": sig.suggestion,
        "tick": sig.tick,
    }
    if sig.evidence:
        meta["evidence"] = dict(sig.evidence)
    if outcome:
        meta["outcome"] = outcome

    return CartridgeEntry(
        text=text,
        game=game,
        signal=sig.signal,
        severity=sig.severity,
        machine=machine,
        per_pattern_meta=meta,
    )


class MetacogCartridgeWriter:
    """Accumulates metacog observations and writes them as JSONL.

    Usage:
        writer = MetacogCartridgeWriter(
            output_dir="fleet-learning/nomad",
            machine="nomad",
        )
        # During gameplay:
        for sig in metacog.active_signals():
            writer.observe(sig, game="toy_b", context="step 42, stuck on L2")

        # At session end:
        writer.flush()

    The JSONL file can then be fed to the membot cart builder:
        python -m membot.build_cart --input metacog_observations.jsonl
    """

    def __init__(
        self,
        output_dir: str | Path,
        machine: str = "",
        filename: str = "metacog_observations.jsonl",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.machine = machine
        self.filename = filename
        self._buffer: List[CartridgeEntry] = []

    def observe(
        self,
        sig: MetacogSignal,
        *,
        game: str = "",
        context: str = "",
        outcome: str = "",
    ) -> CartridgeEntry:
        """Record a metacog signal as a cartridge entry."""
        entry = signal_to_entry(
            sig, game=game, machine=self.machine,
            context=context, outcome=outcome,
        )
        self._buffer.append(entry)
        return entry

    @property
    def pending(self) -> int:
        return len(self._buffer)

    def flush(self) -> Path:
        """Write buffered entries to JSONL. Returns the output path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / self.filename
        with open(out_path, "a") as f:
            for entry in self._buffer:
                f.write(entry.to_json() + "\n")
        count = len(self._buffer)
        self._buffer.clear()
        return out_path

    def entries(self) -> List[CartridgeEntry]:
        """Return buffered entries without flushing."""
        return list(self._buffer)
