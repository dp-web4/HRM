"""
SQLiteBackend — Persistent memory backend using SQLite.

Promotes the HierarchicalMemoryStore schema patterns from experiments/
into a proper MemoryBackend for the live consciousness loop.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sage.memory.hub import MemoryBackend, MemoryEntry


class SQLiteBackend(MemoryBackend):
    """SQLite-backed memory storage with indexed queries."""

    def __init__(self, db_path, backend_id: str = 'sqlite'):
        self.db_path = Path(db_path)
        self.backend_id = backend_id
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """Create database and schema if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                modality TEXT NOT NULL,
                content TEXT NOT NULL,
                content_type TEXT NOT NULL,
                salience REAL NOT NULL,
                surprise REAL,
                novelty REAL,
                arousal REAL,
                reward REAL,
                conflict REAL,
                model_name TEXT,
                session INTEGER,
                cycle INTEGER,
                metabolic_state TEXT,
                metadata TEXT,
                retrieval_count INTEGER DEFAULT 0,
                last_retrieved REAL,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );

            CREATE INDEX IF NOT EXISTS idx_memories_salience
                ON memories(salience DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_modality
                ON memories(modality);
            CREATE INDEX IF NOT EXISTS idx_memories_timestamp
                ON memories(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_model
                ON memories(model_name);
            CREATE INDEX IF NOT EXISTS idx_memories_content_type
                ON memories(content_type);
        """)
        self._conn.commit()

    def store(self, entry: MemoryEntry) -> str:
        """Store a memory entry. Returns entry ID."""
        metadata_json = json.dumps(entry.metadata) if entry.metadata else '{}'
        self._conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, timestamp, modality, content, content_type,
                salience, surprise, novelty, arousal, reward, conflict,
                model_name, session, cycle, metabolic_state, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (entry.id, entry.timestamp, entry.modality, entry.content,
             entry.content_type, entry.salience, entry.surprise, entry.novelty,
             entry.arousal, entry.reward, entry.conflict, entry.model_name,
             entry.session, entry.cycle, entry.metabolic_state, metadata_json)
        )
        self._conn.commit()
        return entry.id

    def query(self, filters: Dict, limit: int = 10) -> List[MemoryEntry]:
        """Query entries by filters. Dynamically builds WHERE clauses."""
        clauses = []
        params = []

        if 'modality' in filters:
            clauses.append("modality = ?")
            params.append(filters['modality'])

        if 'content_type' in filters:
            clauses.append("content_type = ?")
            params.append(filters['content_type'])

        if 'min_salience' in filters:
            clauses.append("salience >= ?")
            params.append(filters['min_salience'])

        if 'model_name' in filters:
            clauses.append("model_name = ?")
            params.append(filters['model_name'])

        if 'time_after' in filters:
            clauses.append("timestamp >= ?")
            params.append(filters['time_after'])

        if 'time_before' in filters:
            clauses.append("timestamp <= ?")
            params.append(filters['time_before'])

        if 'metabolic_state' in filters:
            clauses.append("metabolic_state = ?")
            params.append(filters['metabolic_state'])

        if 'search' in filters:
            clauses.append("content LIKE ?")
            params.append(f"%{filters['search']}%")

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM memories WHERE {where} ORDER BY salience DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def count(self) -> int:
        """Total entries stored."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM memories")
        return cursor.fetchone()[0]

    def health_check(self) -> bool:
        """Is this backend operational?"""
        try:
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def query_by_salience(self, min_salience: float,
                          limit: int = 10) -> List[MemoryEntry]:
        """Query entries above a salience threshold."""
        return self.query({'min_salience': min_salience}, limit=limit)

    def query_by_time_range(self, start: float, end: float,
                            limit: int = 100) -> List[MemoryEntry]:
        """Query entries within a time range."""
        return self.query({'time_after': start, 'time_before': end}, limit=limit)

    def query_by_model(self, model_name: str,
                       limit: int = 10) -> List[MemoryEntry]:
        """Query entries from a specific model."""
        return self.query({'model_name': model_name}, limit=limit)

    def get_stats(self) -> Dict[str, Any]:
        """Counts by modality, content_type, model. Avg salience. Date range."""
        stats: Dict[str, Any] = {'total': self.count()}

        # By modality
        cursor = self._conn.execute(
            "SELECT modality, COUNT(*) FROM memories GROUP BY modality")
        stats['by_modality'] = dict(cursor.fetchall())

        # By content_type
        cursor = self._conn.execute(
            "SELECT content_type, COUNT(*) FROM memories GROUP BY content_type")
        stats['by_content_type'] = dict(cursor.fetchall())

        # By model
        cursor = self._conn.execute(
            "SELECT model_name, COUNT(*) FROM memories GROUP BY model_name")
        stats['by_model'] = dict(cursor.fetchall())

        # Average salience
        cursor = self._conn.execute(
            "SELECT AVG(salience) FROM memories")
        row = cursor.fetchone()
        stats['avg_salience'] = round(row[0], 4) if row[0] is not None else 0.0

        # Date range
        cursor = self._conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM memories")
        row = cursor.fetchone()
        stats['time_range'] = {
            'earliest': row[0],
            'latest': row[1],
        } if row[0] is not None else None

        return stats

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _row_to_entry(self, row) -> MemoryEntry:
        """Convert a database row to a MemoryEntry."""
        # Column order matches CREATE TABLE:
        # id, timestamp, modality, content, content_type,
        # salience, surprise, novelty, arousal, reward, conflict,
        # model_name, session, cycle, metabolic_state, metadata,
        # retrieval_count, last_retrieved, created_at
        metadata = {}
        if row[15]:
            try:
                metadata = json.loads(row[15])
            except (json.JSONDecodeError, TypeError):
                pass

        return MemoryEntry(
            id=row[0],
            timestamp=row[1],
            modality=row[2],
            content=row[3],
            content_type=row[4],
            salience=row[5],
            surprise=row[6] or 0.0,
            novelty=row[7] or 0.0,
            arousal=row[8] or 0.0,
            reward=row[9] or 0.0,
            conflict=row[10] or 0.0,
            model_name=row[11] or '',
            session=row[12] or 0,
            cycle=row[13] or 0,
            metabolic_state=row[14] or '',
            metadata=metadata,
        )
