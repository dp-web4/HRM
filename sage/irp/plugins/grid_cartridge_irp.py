"""
GridCartridgeIRP — ARC-AGI-3 game memory as an IRP plugin.

Wraps cross-session game memory (per-game cartridges) into SAGE's
consciousness loop. Three query modes:

  similar_state   — embedding cosine similarity (every step, fast path)
                    "Have I seen this grid state before?"

  action_outcome  — text search over stored step records (before action selection)
                    "What happened last time I did action 3 near color 5?"

  cross_level     — embedding search across ALL levels for this game (level start)
                    "Does this level remind me of a previous one?"

Write side:
  write_step_record(obs, step_record) — stores SAGE's reasoning as odd-pattern
  entries alongside the frame embedding.

Backends (transport-agnostic):
  - Local JSON files (arc-agi-3/experiments/cartridges/*.grid.json)
  - Andy's .cart.npz paired lattice format (membot/vision/grid_cartridge.py)
  - Andy's GridCartridge HTTP endpoint (plug in via andy_url config)

On load, tries .cart.npz first (Andy's format), then falls back to local JSON.
On write, always writes local JSON + optionally pushes to Andy's endpoint.

Key design: embedding search uses numpy cosine similarity. For the sizes we
operate at (512-dim, hundreds of entries per session), this is instant.

2026-04-01 — Phase 1 (local file backend, cosine similarity search)
2026-04-03 — Phase 2 wiring: load from Andy's .cart.npz, parse paired passages
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sage.irp.base import IRPPlugin, IRPState
from sage.irp.plugins.grid_vision_irp import GridObservation, StepRecord

log = logging.getLogger("sage.grid_cartridge_irp")

CARTRIDGE_DIR = Path("arc-agi-3/experiments/cartridges")

# Query modes
QUERY_SIMILAR_STATE = "similar_state"
QUERY_ACTION_OUTCOME = "action_outcome"
QUERY_CROSS_LEVEL = "cross_level"

# Cognitive type priority weights for search reranking
COGNITIVE_PRIORITY = {
    "reflection": 1.5,   # hypothesis+strategy+key_insight — highest value for recall
    "discovery":  1.3,   # novel pattern first seen
    "goal_update": 1.2,  # objective revision
    None: 1.0,           # routine step
}


@dataclass
class CartridgeEntry:
    """A single stored step: SAGE's reasoning + embedding for that frame."""
    step_record: Dict[str, Any]         # StepRecord.to_dict()
    frame_objects: List[Dict]           # GridObservation.objects (compact summary)
    timestamp: float
    level_id: str
    game_id: str
    cognitive_type: Optional[str] = None
    embedding: Optional[np.ndarray] = None  # 512-dim float32 (CLIP ViT-B-32)
    session_id: int = 0                      # which session created this entry
    perishability: str = "volatile"          # volatile|replaceable|archival-important

    def to_dict(self) -> Dict[str, Any]:
        import base64
        d: Dict[str, Any] = {
            "step_record": self.step_record,
            "frame_objects": self.frame_objects,
            "timestamp": self.timestamp,
            "level_id": self.level_id,
            "game_id": self.game_id,
            "cognitive_type": self.cognitive_type,
            "session_id": self.session_id,
            "perishability": self.perishability,
        }
        if self.embedding is not None:
            d["embedding"] = base64.b64encode(
                self.embedding.astype(np.float32).tobytes()
            ).decode("ascii")
        else:
            d["embedding"] = None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CartridgeEntry":
        import base64
        embedding = None
        if d.get("embedding") is not None:
            raw = d["embedding"]
            embedding = np.frombuffer(base64.b64decode(raw), dtype=np.float32).copy()
        return cls(
            step_record=d.get("step_record", {}),
            frame_objects=d.get("frame_objects", []),
            timestamp=d.get("timestamp", 0.0),
            level_id=d.get("level_id", ""),
            game_id=d.get("game_id", ""),
            cognitive_type=d.get("cognitive_type"),
            embedding=embedding,
            session_id=d.get("session_id", 0),
            perishability=d.get("perishability", "volatile"),
        )


@dataclass
class SearchResult:
    """A single hit returned by cartridge search."""
    entry: CartridgeEntry
    score: float               # cosine similarity (0–1) or text match score
    query_mode: str
    prior_reasoning: str = ""  # extracted reasoning_text for quick display

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "query_mode": self.query_mode,
            "prior_reasoning": self.prior_reasoning,
            "step_record": self.entry.step_record,
            "frame_objects": self.entry.frame_objects,
            "level_id": self.entry.level_id,
            "cognitive_type": self.entry.cognitive_type,
            "timestamp": self.entry.timestamp,
        }


class GridCartridgeIRP(IRPPlugin):
    """ARC-AGI-3 game memory search as an IRP plugin.

    Config keys:
        entity_id: str — plugin identity (default: 'grid_cartridge_irp')
        game_id: str — current game ID (e.g., 'sc25-f9b21a2f')
        top_k: int — max search results (default: 3)
        min_score: float — minimum similarity to include (default: 0.25)
        cartridge_dir: str — path for local JSON files (default: 'arc-agi-3/experiments/cartridges')
        andy_url: str — Andy's GridCartridge HTTP endpoint (Phase 2, optional)
        reflection_boost: bool — boost cognitive_type='reflection' in ranking (default: True)
    """

    def __init__(self, config: Dict[str, Any]):
        config.setdefault('entity_id', 'grid_cartridge_irp')
        config.setdefault('max_iterations', 1)
        super().__init__(config)

        self.game_id = config.get('game_id', 'unknown')
        self.game_family = self.game_id.split('-')[0] if '-' in self.game_id else self.game_id
        self.top_k = config.get('top_k', 3)
        self.min_score = config.get('min_score', 0.25)
        self.cartridge_dir = Path(config.get('cartridge_dir', str(CARTRIDGE_DIR)))
        self.andy_url = config.get('andy_url', None)  # Phase 2 hook
        self.reflection_boost = config.get('reflection_boost', True)
        self.recency_half_life = config.get('recency_half_life', 3600.0)  # 1h default
        self.session_id = config.get('session_id', 0)

        # In-memory entry store: keyed by level_id
        self._entries: Dict[str, List[CartridgeEntry]] = {}  # level_id → entries
        self._all_entries: List[CartridgeEntry] = []         # flat list for cross-level search
        self._write_count = 0
        self._last_results: List[SearchResult] = []

        self._load_from_disk()
        print(f"✓ GridCartridgeIRP initialized (game={self.game_id}, "
              f"entries={len(self._all_entries)})")

    # -------------------------------------------------------------------------
    # IRP contract
    # -------------------------------------------------------------------------

    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        """Mount cartridge for a new level.

        x0: {"query_embedding": np.ndarray, "query_text": str, "level_id": str}
            or a GridObservation (uses its embedding + level_id)
        task_ctx: {"level_id": str, "query_mode": str, "game_id": str}
        """
        if isinstance(x0, GridObservation):
            query_embedding = x0.embedding
            query_text = x0.perception_notes or ""
            level_id = x0.level_id
        elif isinstance(x0, dict):
            query_embedding = x0.get("query_embedding")
            query_text = x0.get("query_text", "")
            level_id = x0.get("level_id", task_ctx.get("level_id", ""))
        else:
            query_embedding = None
            query_text = ""
            level_id = task_ctx.get("level_id", "")

        if task_ctx.get("game_id"):
            self.game_id = task_ctx["game_id"]
            self.game_family = self.game_id.split('-')[0] if '-' in self.game_id else self.game_id

        mode = task_ctx.get("query_mode", QUERY_SIMILAR_STATE)
        n_level = len(self._entries.get(level_id, []))
        n_total = len(self._all_entries)

        return IRPState(
            x={
                "query_embedding": query_embedding,
                "query_text": query_text,
                "level_id": level_id,
                "query_mode": mode,
                "results": [],
            },
            step_idx=0,
            energy_val=1.0,
            meta={
                "level_id": level_id,
                "query_mode": mode,
                "n_entries_level": n_level,
                "n_entries_total": n_total,
            },
        )

    def step(self, state: IRPState, noise_schedule: Any = None) -> IRPState:
        """Execute the requested query against the cartridge."""
        t0 = time.time()

        mode = state.x.get("query_mode", QUERY_SIMILAR_STATE)
        level_id = state.x.get("level_id", "")
        query_embedding = state.x.get("query_embedding")
        query_text = state.x.get("query_text", "")

        if mode == QUERY_SIMILAR_STATE:
            results = self._search_by_embedding(query_embedding, level_id, cross_level=False)
        elif mode == QUERY_ACTION_OUTCOME:
            results = self._search_by_text(query_text, level_id, cross_level=False)
        elif mode == QUERY_CROSS_LEVEL:
            results = self._search_by_embedding(query_embedding, level_id, cross_level=True)
        else:
            results = []

        self._last_results = results
        elapsed_ms = (time.time() - t0) * 1000
        best_score = results[0].score if results else 0.0

        return IRPState(
            x={
                **state.x,
                "results": [r.to_dict() for r in results],
            },
            step_idx=state.step_idx + 1,
            energy_val=1.0 - best_score,
            meta={
                **state.meta,
                "n_results": len(results),
                "best_score": best_score,
                "elapsed_ms": round(elapsed_ms, 2),
                "query_mode": mode,
            },
        )

    def energy(self, state: IRPState) -> float:
        """Energy = 1 - best_score. Perfect recall = 0, no match = 1."""
        return state.energy_val if state.energy_val is not None else 1.0

    def halt(self, history: List[IRPState]) -> bool:
        """One search per consciousness cycle."""
        return len(history) >= 1

    # -------------------------------------------------------------------------
    # Write side — odd-pattern entries
    # -------------------------------------------------------------------------

    def write_step_record(self, obs: GridObservation, step_record: StepRecord) -> None:
        """Store SAGE's reasoning for this frame as an odd-pattern entry.

        Called after action selection, before executing the action.
        Persisted to disk every 10 writes.
        """
        # Perishability: reflections and discoveries are archival, routine steps volatile
        ctype = step_record.cognitive_type
        perish = "archival-important" if ctype in ("reflection", "discovery") else "volatile"

        entry = CartridgeEntry(
            step_record=step_record.to_dict(),
            frame_objects=obs.objects,
            timestamp=time.time(),
            level_id=obs.level_id,
            game_id=self.game_id,
            cognitive_type=ctype,
            embedding=obs.embedding.copy() if obs.embedding is not None else None,
            session_id=self.session_id,
            perishability=perish,
        )

        if obs.level_id not in self._entries:
            self._entries[obs.level_id] = []
        self._entries[obs.level_id].append(entry)
        self._all_entries.append(entry)
        self._write_count += 1

        if self._write_count % 10 == 0:
            self._save_to_disk()

        # Phase 2 hook: send to Andy's endpoint
        if self.andy_url:
            self._push_to_andy(entry)

    def flush(self) -> None:
        """Force-save cartridge to disk (call at level/session end)."""
        self._save_to_disk()
        log.info(f"GridCartridgeIRP flushed: {len(self._all_entries)} entries, "
                 f"game={self.game_id}")

    # -------------------------------------------------------------------------
    # Search backends
    # -------------------------------------------------------------------------

    def _recency_weight(self, entry: CartridgeEntry) -> float:
        """Recency multiplier: 0.5 floor, 1.0 at now, 0.75 at half-life."""
        if self.recency_half_life <= 0:
            return 1.0
        age = max(0.0, time.time() - entry.timestamp)
        import math
        return 0.5 + 0.5 * math.exp(-math.log(2) * age / self.recency_half_life)

    def _search_by_embedding(
        self,
        query_embedding: Optional[np.ndarray],
        current_level_id: str,
        cross_level: bool,
    ) -> List[SearchResult]:
        """Cosine similarity search with recency weighting. Fast — all in numpy."""
        if query_embedding is None:
            return []

        pool = self._all_entries if cross_level else self._entries.get(current_level_id, [])
        if not pool:
            return []

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        scored: List[Tuple[float, CartridgeEntry]] = []

        for entry in pool:
            if entry.embedding is None:
                continue
            e_norm = entry.embedding / (np.linalg.norm(entry.embedding) + 1e-9)
            cosine = float(np.dot(q_norm, e_norm))
            # Apply cognitive type priority boost
            if self.reflection_boost:
                boost = COGNITIVE_PRIORITY.get(entry.cognitive_type, 1.0)
                cosine = min(cosine * boost, 1.0)
            # Apply recency weighting
            cosine *= self._recency_weight(entry)
            if cosine >= self.min_score:
                scored.append((cosine, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(
                entry=entry,
                score=score,
                query_mode=QUERY_CROSS_LEVEL if cross_level else QUERY_SIMILAR_STATE,
                prior_reasoning=entry.step_record.get("reasoning_text") or
                                entry.step_record.get("hypothesis") or "",
            )
            for score, entry in scored[: self.top_k]
        ]

    def _search_by_text(
        self,
        query_text: str,
        current_level_id: str,
        cross_level: bool,
    ) -> List[SearchResult]:
        """Keyword search over step record text fields.

        Scores = number of query terms found in reasoning fields.
        For Phase 2: replace with Andy's embedding text search.
        """
        if not query_text:
            return []

        pool = self._all_entries if cross_level else self._entries.get(current_level_id, [])
        if not pool:
            return []

        terms = [t.lower() for t in query_text.split() if len(t) > 2]
        if not terms:
            return []

        scored: List[Tuple[float, CartridgeEntry]] = []

        for entry in pool:
            sr = entry.step_record
            haystack = " ".join(filter(None, [
                sr.get("reasoning_text", ""),
                sr.get("hypothesis", ""),
                sr.get("strategy", ""),
                sr.get("key_insight", ""),
                sr.get("action_rationale", ""),
            ])).lower()

            hits = sum(1 for t in terms if t in haystack)
            if hits == 0:
                continue

            score = hits / len(terms)
            if self.reflection_boost:
                boost = COGNITIVE_PRIORITY.get(entry.cognitive_type, 1.0)
                score = min(score * boost, 1.0)
            # Apply recency weighting
            score *= self._recency_weight(entry)
            if score >= self.min_score:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchResult(
                entry=entry,
                score=score,
                query_mode=QUERY_ACTION_OUTCOME,
                prior_reasoning=entry.step_record.get("reasoning_text") or
                                entry.step_record.get("hypothesis") or "",
            )
            for score, entry in scored[: self.top_k]
        ]

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _cartridge_path(self) -> Path:
        self.cartridge_dir.mkdir(parents=True, exist_ok=True)
        return self.cartridge_dir / f"{self.game_family}.grid.json"

    def _npz_cartridge_path(self) -> Path:
        """Andy's .cart.npz paired lattice format."""
        return self.cartridge_dir / f"{self.game_family}.cart.npz"

    def _load_from_disk(self) -> None:
        """Load entries from disk. Tries .cart.npz first, then .grid.json."""
        # Try Andy's paired lattice format first
        npz_path = self._npz_cartridge_path()
        if npz_path.exists():
            n_loaded = self._load_from_npz(npz_path)
            if n_loaded > 0:
                log.info(f"GridCartridgeIRP loaded {n_loaded} entries from {npz_path}")
                return

        # Fall back to local JSON
        json_path = self._cartridge_path()
        if json_path.exists():
            self._load_from_json(json_path)

    def _load_from_npz(self, path: Path) -> int:
        """Load from Andy's .cart.npz paired lattice format.

        Even patterns: CLIP embedding + grid + objects (searchable)
        Odd patterns: SAGE reasoning (or placeholder)

        Uses Andy's passage parsers: load_grid_from_passage(),
        load_objects_from_passage(), load_sage_record_from_passage().
        """
        try:
            # Lazy import — membot may not be on PYTHONPATH everywhere
            import sys
            membot_vision = Path(__file__).parent.parent.parent.parent / "membot" / "vision"
            if str(membot_vision) not in sys.path:
                sys.path.insert(0, str(membot_vision))
            # Also check sibling repo path
            alt_path = Path("membot") / "vision"
            if alt_path.exists() and str(alt_path) not in sys.path:
                sys.path.insert(0, str(alt_path))

            from grid_cartridge import (
                load_grid_from_passage,
                load_objects_from_passage,
                load_sage_record_from_passage,
            )
        except ImportError:
            log.debug("grid_cartridge parsers not available — skipping .cart.npz load")
            return 0

        try:
            cart = np.load(str(path), allow_pickle=True)
            embeddings = cart["embeddings"]
            passages = cart["passages"]
            n_entries = len(passages)
            loaded = 0

            # Process paired entries (even = frame, odd = SAGE reasoning)
            for i in range(0, n_entries - 1, 2):
                even_passage = str(passages[i])
                odd_passage = str(passages[i + 1])
                even_embedding = embeddings[i]

                # Parse even pattern
                objects = load_objects_from_passage(even_passage)

                # Parse header for level_id and step
                level_id = ""
                step_num = 0
                for line in even_passage.split("\n"):
                    if line.startswith("[GRID]"):
                        parts = line.split("|")
                        if parts:
                            level_id = parts[0].replace("[GRID]", "").strip()
                        if len(parts) >= 2 and "Step" in parts[1]:
                            try:
                                step_num = int(parts[1].strip().split("Step")[1].strip())
                            except (ValueError, IndexError):
                                pass
                        break

                # Parse odd pattern for SAGE reasoning
                sage_record = load_sage_record_from_passage(odd_passage) or {}

                # Build step_record from sage_record or minimal stub
                step_record = sage_record if sage_record else {
                    "step": step_num,
                    "source": "andy_cart_npz",
                    "level_id": level_id,
                }

                entry = CartridgeEntry(
                    step_record=step_record,
                    frame_objects=objects,
                    timestamp=time.time(),  # No timestamp in npz; use load time
                    level_id=level_id or self.game_id,
                    game_id=self.game_id,
                    cognitive_type=sage_record.get("cognitive_type"),
                    embedding=even_embedding.copy() if even_embedding is not None else None,
                    session_id=0,
                    perishability="archival-important",  # Andy's data is curated
                )

                if entry.level_id not in self._entries:
                    self._entries[entry.level_id] = []
                self._entries[entry.level_id].append(entry)
                self._all_entries.append(entry)
                loaded += 1

            return loaded

        except Exception as e:
            log.warning(f"GridCartridgeIRP .cart.npz load failed ({path}): {e}")
            return 0

    def _load_from_json(self, path: Path) -> None:
        """Load from local JSON format (Phase 1)."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for raw in data.get("entries", []):
                entry = CartridgeEntry.from_dict(raw)
                self._all_entries.append(entry)
                if entry.level_id not in self._entries:
                    self._entries[entry.level_id] = []
                self._entries[entry.level_id].append(entry)
            log.info(f"GridCartridgeIRP loaded {len(self._all_entries)} entries "
                     f"from {path}")
        except Exception as e:
            log.warning(f"GridCartridgeIRP load failed ({path}): {e}")

    def _save_to_disk(self) -> None:
        path = self._cartridge_path()
        try:
            data = {
                "game_id": self.game_id,
                "game_family": self.game_family,
                "n_entries": len(self._all_entries),
                "saved_at": time.time(),
                "entries": [e.to_dict() for e in self._all_entries],
            }
            with open(path, "w") as f:
                json.dump(data, f)
            log.debug(f"GridCartridgeIRP saved {len(self._all_entries)} entries to {path}")
        except Exception as e:
            log.warning(f"GridCartridgeIRP save failed ({path}): {e}")

    # -------------------------------------------------------------------------
    # Phase 2 hook — Andy's endpoint
    # -------------------------------------------------------------------------

    def _push_to_andy(self, entry: CartridgeEntry) -> None:
        """Push odd-pattern entry to Andy's GridCartridge endpoint.

        Sends the SAGE step record as a JSON payload. Andy's server
        writes it to the odd pattern of the corresponding cart entry.
        Non-blocking: failures are logged but don't interrupt the game loop.
        """
        try:
            import requests
            payload = {
                "game_id": entry.game_id,
                "level_id": entry.level_id,
                "step_record": entry.step_record,
                "cognitive_type": entry.cognitive_type,
                "timestamp": entry.timestamp,
            }
            resp = requests.post(
                f"{self.andy_url.rstrip('/')}/api/grid/step_record",
                json=payload,
                timeout=5,
            )
            if resp.status_code == 200:
                log.debug(f"Pushed step_record to Andy ({entry.cognitive_type})")
            else:
                log.debug(f"Andy endpoint returned {resp.status_code}")
        except Exception as e:
            log.debug(f"Push to Andy failed (non-blocking): {e}")

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def get_reflections(self, limit: int = 5) -> List[CartridgeEntry]:
        """Return the most recent reflection entries across all levels.

        Used at session start to seed hypothesis/strategy without relying
        on text search matching. Direct accessor — always works if data exists.
        """
        reflections = [e for e in self._all_entries if e.cognitive_type == "reflection"]
        return reflections[-limit:]

    def get_discoveries(self, limit: int = 5) -> List[CartridgeEntry]:
        """Return the most recent discovery entries (level-up moments)."""
        discoveries = [e for e in self._all_entries if e.cognitive_type == "discovery"]
        return discoveries[-limit:]

    @property
    def stats(self) -> Dict[str, Any]:
        """Current cartridge stats."""
        n_reflections = sum(
            1 for e in self._all_entries if e.cognitive_type == "reflection"
        )
        return {
            "game_id": self.game_id,
            "n_entries_total": len(self._all_entries),
            "n_levels": len(self._entries),
            "n_reflections": n_reflections,
            "write_count": self._write_count,
            "last_results": len(self._last_results),
        }

    def level_summary(self, level_id: str) -> str:
        """Human-readable summary of stored memory for a level."""
        entries = self._entries.get(level_id, [])
        if not entries:
            return f"Level {level_id}: no entries"
        n_reflect = sum(1 for e in entries if e.cognitive_type == "reflection")
        last = entries[-1].step_record
        return (
            f"Level {level_id}: {len(entries)} entries, "
            f"{n_reflect} reflections | "
            f"last action={last.get('action_taken')}, "
            f"rationale='{last.get('action_rationale', '')[:60]}'"
        )
