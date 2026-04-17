"""
EpisodicIndex — the hippocampal pattern-completion engine.

Core operations:
- bind(): store an episode with its embedding
- recall(): find similar past episodes from a partial cue
- consolidate(): dream-state compression
- forget(): graceful decay of low-utility episodes
"""

import json
import logging
import math
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np

from sage.cognition.episodic.data import (
    EMBEDDING_DIM,
    ConsolidationReport,
    Episode,
    EpisodicCue,
    RecallResult,
)

logger = logging.getLogger(__name__)


class EpisodicIndex:
    """Hippocampal episodic index with pattern-completion retrieval."""

    def __init__(self, db_path: Optional[str] = None, recency_half_life: float = 86400.0):
        """
        Args:
            db_path: SQLite path for persistence. None = in-memory only.
            recency_half_life: seconds for recency decay (default 1 day).
        """
        self._episodes: dict[str, Episode] = {}  # episode_id → Episode
        self._embeddings: Optional[np.ndarray] = None  # (N, EMBEDDING_DIM) matrix
        self._id_order: list[str] = []  # maps matrix row → episode_id
        self._dirty = False  # whether in-memory differs from SQLite
        self._recency_half_life = recency_half_life

        # SQLite persistence
        self._db_path = db_path
        self._db: Optional[sqlite3.Connection] = None
        if db_path:
            self._init_db(db_path)
            self._load_from_db()

    # ─── Public interface ──────────────────────────────────────────

    def bind(self, episode: Episode) -> str:
        """Store an episode. Computes embedding if not present. Returns episode_id."""
        if episode.embedding is None:
            episode.compute_embedding()

        self._episodes[episode.episode_id] = episode
        self._id_order.append(episode.episode_id)

        # Append to embedding matrix
        emb = episode.embedding.reshape(1, -1)
        if self._embeddings is None:
            self._embeddings = emb
        else:
            self._embeddings = np.vstack([self._embeddings, emb])

        # Persist
        if self._db:
            self._store_to_db(episode)

        self._dirty = True
        logger.debug(f"Bound episode {episode.episode_id} (tags={episode.tags})")
        return episode.episode_id

    def recall(self, cue: EpisodicCue, k: int = 5) -> list[RecallResult]:
        """
        Pattern-completion retrieval.

        Given a partial cue, find the k most similar past episodes.
        Uses masked cosine similarity — only compares dimensions
        where the cue provides data.
        """
        if not self._episodes or self._embeddings is None:
            return []

        cue_emb, cue_mask = cue.to_embedding_and_mask()

        # Check if any dimensions are active
        if cue_mask.sum() == 0:
            # No embedding dimensions — fall back to time/tag filtering
            return self._filter_only_recall(cue, k)

        # Masked cosine similarity
        masked_db = self._embeddings * cue_mask  # (N, D)
        masked_cue = cue_emb * cue_mask  # (D,)

        # Norms
        db_norms = np.linalg.norm(masked_db, axis=1)  # (N,)
        cue_norm = np.linalg.norm(masked_cue)

        if cue_norm == 0:
            return self._filter_only_recall(cue, k)

        # Similarities
        dots = masked_db @ masked_cue  # (N,)
        norms = db_norms * cue_norm
        norms = np.maximum(norms, 1e-8)
        similarities = dots / norms  # (N,)

        # Apply filters
        now = time.time()
        scores = np.zeros(len(similarities), dtype=np.float32)

        for i, eid in enumerate(self._id_order):
            ep = self._episodes[eid]

            # Time window filter
            if cue.time_window:
                if ep.timestamp < cue.time_window[0] or ep.timestamp > cue.time_window[1]:
                    scores[i] = -1.0
                    continue

            # Min salience filter
            if cue.min_salience > 0:
                ep_salience = max(ep.snarc_scores.values()) if ep.snarc_scores else 0
                if ep_salience < cue.min_salience:
                    scores[i] = -1.0
                    continue

            # Recency weight: exponential decay
            age = now - ep.timestamp
            recency = math.exp(-math.log(2) * age / self._recency_half_life)

            # Access boost: frequently-recalled episodes are more retrievable
            access_boost = 1.0 + 0.1 * min(ep.access_count, 10)

            scores[i] = similarities[i] * (0.7 + 0.3 * recency) * access_boost

        # Top-k
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            eid = self._id_order[idx]
            ep = self._episodes[eid]
            ep.access_count += 1
            ep.last_access = now

            age = now - ep.timestamp
            recency = math.exp(-math.log(2) * age / self._recency_half_life)

            results.append(RecallResult(
                episode=ep,
                similarity=float(similarities[idx]),
                recency_weight=recency,
                access_boost=1.0 + 0.1 * min(ep.access_count, 10),
            ))

        return results

    def consolidate(self, similarity_threshold: float = 0.85, min_cluster_size: int = 3) -> ConsolidationReport:
        """
        Dream-state consolidation.

        Merges similar episodes into prototypes. Removes low-utility episodes.
        """
        t0 = time.time()
        report = ConsolidationReport(episodes_before=len(self._episodes))

        if len(self._episodes) < min_cluster_size * 2:
            report.episodes_after = len(self._episodes)
            report.duration_seconds = time.time() - t0
            return report

        # Simple greedy clustering by similarity
        remaining = set(self._id_order)
        clusters = []

        for eid in list(remaining):
            if eid not in remaining:
                continue
            ep = self._episodes[eid]
            cluster = [eid]
            remaining.discard(eid)

            for other_eid in list(remaining):
                other_ep = self._episodes[other_eid]
                sim = self._pairwise_similarity(ep, other_ep)
                if sim >= similarity_threshold:
                    cluster.append(other_eid)
                    remaining.discard(other_eid)

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
            else:
                # Not enough to cluster — keep as individual episodes
                pass

        # Create prototypes from clusters
        for cluster in clusters:
            prototype = self._create_prototype(cluster)
            report.prototypes_created += 1
            report.episodes_merged += len(cluster)

            # Remove individual episodes, keep prototype
            for eid in cluster:
                if eid != prototype.episode_id:
                    del self._episodes[eid]

        # Rebuild index
        self._rebuild_index()

        report.episodes_after = len(self._episodes)
        report.duration_seconds = time.time() - t0

        logger.info(f"Consolidation: {report.episodes_before} → {report.episodes_after} "
                     f"({report.prototypes_created} prototypes, {report.episodes_merged} merged)")
        return report

    def forget(self, max_age_days: float = 30, min_accesses: int = 0) -> int:
        """Remove old, low-utility episodes."""
        cutoff = time.time() - max_age_days * 86400
        to_forget = [
            eid for eid, ep in self._episodes.items()
            if ep.timestamp < cutoff and ep.access_count <= min_accesses
        ]

        for eid in to_forget:
            del self._episodes[eid]
            if self._db:
                self._db.execute("DELETE FROM episodes WHERE episode_id = ?", (eid,))

        if to_forget:
            self._rebuild_index()
            if self._db:
                self._db.commit()
            logger.info(f"Forgot {len(to_forget)} episodes")

        return len(to_forget)

    def stats(self) -> dict:
        """Index statistics."""
        if not self._episodes:
            return {"count": 0}

        timestamps = [ep.timestamp for ep in self._episodes.values()]
        return {
            "count": len(self._episodes),
            "oldest": min(timestamps),
            "newest": max(timestamps),
            "age_range_hours": (max(timestamps) - min(timestamps)) / 3600,
            "total_accesses": sum(ep.access_count for ep in self._episodes.values()),
            "embedding_dim": EMBEDDING_DIM,
            "memory_mb": round(self._embeddings.nbytes / 1024 / 1024, 2) if self._embeddings is not None else 0,
        }

    # ─── Private methods ───────────────────────────────────────────

    def _pairwise_similarity(self, ep1: Episode, ep2: Episode) -> float:
        """Cosine similarity between two episode embeddings."""
        if ep1.embedding is None or ep2.embedding is None:
            return 0.0
        dot = np.dot(ep1.embedding, ep2.embedding)
        n1 = np.linalg.norm(ep1.embedding)
        n2 = np.linalg.norm(ep2.embedding)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(dot / (n1 * n2))

    def _create_prototype(self, cluster: list[str]) -> Episode:
        """Create a prototype episode from a cluster of similar episodes."""
        episodes = [self._episodes[eid] for eid in cluster]

        # Average embedding
        avg_emb = np.mean([ep.embedding for ep in episodes], axis=0)

        # Merge tags
        all_tags = set()
        for ep in episodes:
            all_tags.update(ep.tags)

        # Aggregate stats
        avg_reward = np.mean([ep.reward for ep in episodes])
        success_rate = np.mean([1.0 if ep.success else 0.0 for ep in episodes if ep.success is not None])

        # Most common action
        actions = [ep.action_taken for ep in episodes if ep.action_taken]
        most_common_action = max(set(actions), key=actions.count) if actions else None

        prototype = Episode(
            session_id=episodes[0].session_id,
            cycle_id=0,  # Prototype, not from a specific cycle
            state_signature=episodes[0].state_signature,  # Use first as template
            snarc_scores={k: float(np.mean([ep.snarc_scores.get(k, 0) for ep in episodes]))
                          for k in ['surprise', 'novelty', 'arousal', 'reward', 'conflict']},
            action_taken=most_common_action,
            reward=float(avg_reward),
            success=success_rate > 0.5 if success_rate > 0 else None,
            embedding=avg_emb.astype(np.float32),
            tags=list(all_tags) + ['prototype', f'merged_{len(cluster)}'],
            access_count=sum(ep.access_count for ep in episodes),
        )

        self._episodes[prototype.episode_id] = prototype
        return prototype

    def _rebuild_index(self):
        """Rebuild the embedding matrix and ID order from current episodes."""
        self._id_order = sorted(self._episodes.keys(), key=lambda eid: self._episodes[eid].timestamp)
        if self._id_order:
            self._embeddings = np.vstack([self._episodes[eid].embedding for eid in self._id_order])
        else:
            self._embeddings = None

    def _filter_only_recall(self, cue: EpisodicCue, k: int) -> list[RecallResult]:
        """Fallback recall when no embedding dimensions are active."""
        candidates = list(self._episodes.values())

        # Apply filters
        if cue.time_window:
            candidates = [ep for ep in candidates
                          if cue.time_window[0] <= ep.timestamp <= cue.time_window[1]]

        if cue.tags:
            tag_set = set(cue.tags)
            candidates = [ep for ep in candidates if tag_set & set(ep.tags)]

        # Sort by recency
        candidates.sort(key=lambda ep: ep.timestamp, reverse=True)
        now = time.time()

        return [
            RecallResult(
                episode=ep,
                similarity=0.5,  # Unknown similarity
                recency_weight=math.exp(-math.log(2) * (now - ep.timestamp) / self._recency_half_life),
            )
            for ep in candidates[:k]
        ]

    # ─── SQLite persistence ────────────────────────────────────────

    def _init_db(self, db_path: str):
        """Initialize SQLite database."""
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(db_path)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                timestamp REAL,
                session_id TEXT,
                cycle_id INTEGER,
                data TEXT,
                embedding BLOB
            )
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_session ON episodes(session_id)")
        self._db.commit()

    def _store_to_db(self, episode: Episode):
        """Persist a single episode to SQLite."""
        if not self._db:
            return
        self._db.execute(
            "INSERT OR REPLACE INTO episodes (episode_id, timestamp, session_id, cycle_id, data, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                episode.episode_id,
                episode.timestamp,
                episode.session_id,
                episode.cycle_id,
                json.dumps(episode.to_dict(), default=str),
                episode.embedding.tobytes() if episode.embedding is not None else None,
            )
        )
        self._db.commit()

    def _load_from_db(self):
        """Load all episodes from SQLite into memory."""
        if not self._db:
            return
        t0 = time.time()
        cursor = self._db.execute("SELECT data, embedding FROM episodes ORDER BY timestamp")
        count = 0
        for row in cursor:
            data = json.loads(row[0])
            ep = Episode.from_dict(data)
            if row[1] is not None:
                ep.embedding = np.frombuffer(row[1], dtype=np.float32).copy()
            elif ep.embedding is None:
                ep.compute_embedding()
            self._episodes[ep.episode_id] = ep
            count += 1

        self._rebuild_index()
        elapsed = time.time() - t0
        logger.info(f"Loaded {count} episodes from {self._db_path} in {elapsed:.2f}s")
