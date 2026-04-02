"""
MemoryHub — Unified gathering infrastructure for RLLF.

Registry of memory backends with fan-out writes and merged queries.
Same pattern as LLMPool and SensorHub — each backend implements a
common interface, the hub routes writes and queries.

Phase 1 of 3: Gathering layer (store + basic query).
"""

import hashlib
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    """Universal memory entry — common across all backends."""

    id: str                         # sha256[:16] or auto-generated
    timestamp: float
    modality: str                   # 'message', 'observation', 'tool_result', 'insight'
    content: str                    # The actual content (prompt, response, observation)
    content_type: str               # 'exchange', 'event', 'pattern', 'consolidated'

    # SNARC scores
    salience: float = 0.0
    surprise: float = 0.0
    novelty: float = 0.0
    arousal: float = 0.0
    reward: float = 0.0
    conflict: float = 0.0

    # Context
    model_name: str = ''            # Which LLM produced/processed this
    session: int = 0
    cycle: int = 0
    metabolic_state: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryBackend(ABC):
    """Interface for memory storage backends."""

    backend_id: str

    @abstractmethod
    def store(self, entry: MemoryEntry) -> str:
        """Store an entry. Returns entry ID."""

    @abstractmethod
    def query(self, filters: Dict, limit: int = 10) -> List[MemoryEntry]:
        """Query entries by filters (modality, min_salience, time_range, etc.)."""

    @abstractmethod
    def count(self) -> int:
        """Total entries stored."""

    @abstractmethod
    def health_check(self) -> bool:
        """Is this backend operational?"""


class MemoryHub:
    """Unified registry of memory backends."""

    def __init__(self):
        self._backends: Dict[str, MemoryBackend] = {}

    def register(self, backend: MemoryBackend) -> None:
        """Register a memory backend."""
        self._backends[backend.backend_id] = backend

    def unregister(self, backend_id: str) -> None:
        """Remove a memory backend."""
        self._backends.pop(backend_id, None)

    def store(self, entry: MemoryEntry,
              backends: Optional[List[str]] = None) -> Dict[str, str]:
        """Store to specified backends (or all). Returns {backend_id: entry_id}."""
        targets = self._resolve_backends(backends)
        results = {}
        for bid, backend in targets.items():
            try:
                entry_id = backend.store(entry)
                results[bid] = entry_id
            except Exception as e:
                results[bid] = f'error:{e}'
        return results

    def query(self, filters: Dict,
              backends: Optional[List[str]] = None,
              limit: int = 10,
              recency_half_life: float = 3600.0) -> List[MemoryEntry]:
        """Query across backends, merge results, sort by recency-weighted salience.

        Scoring: final_score = salience * (0.5 + 0.5 * exp(-ln2 * age / half_life))
        - At age=0: full weight (1.0)
        - At age=half_life: 0.75
        - At age=infinity: 0.5 floor (old high-salience memories never disappear)

        Args:
            recency_half_life: Half-life in seconds for recency weighting (default: 1h).
                              Set to 0 to disable recency weighting.
        """
        targets = self._resolve_backends(backends)
        seen_ids = set()
        merged = []

        for backend in targets.values():
            try:
                entries = backend.query(filters, limit=limit)
                for entry in entries:
                    if entry.id not in seen_ids:
                        seen_ids.add(entry.id)
                        merged.append(entry)
            except Exception:
                continue

        now = time.time()
        _ln2 = math.log(2)

        def _score(e: MemoryEntry) -> float:
            if recency_half_life <= 0:
                return e.salience
            age = max(0.0, now - e.timestamp)
            recency = 0.5 + 0.5 * math.exp(-_ln2 * age / recency_half_life)
            return e.salience * recency

        merged.sort(key=_score, reverse=True)
        return merged[:limit]

    def stats(self) -> Dict[str, Any]:
        """Per-backend entry counts + health."""
        result = {}
        for bid, backend in self._backends.items():
            try:
                result[bid] = {
                    'count': backend.count(),
                    'healthy': backend.health_check(),
                }
            except Exception as e:
                result[bid] = {'count': -1, 'healthy': False, 'error': str(e)}
        return result

    def list_backends(self) -> List[str]:
        """List registered backend IDs."""
        return list(self._backends.keys())

    def _resolve_backends(self,
                          backend_ids: Optional[List[str]]) -> Dict[str, MemoryBackend]:
        """Resolve backend list — all if None, else filtered."""
        if backend_ids is None:
            return dict(self._backends)
        return {bid: self._backends[bid]
                for bid in backend_ids if bid in self._backends}
