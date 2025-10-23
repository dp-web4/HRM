#!/usr/bin/env python3
"""
Hierarchical Long-Term Memory with SNARC-Guided Growth
Extends memory_aware_kernel with persistent, growing memory architecture.

Architecture:
1. Circular buffers (operational) - Fixed size, immediate context
2. Long-term episodic (growing) - SNARC-filtered significant experiences
3. Consolidated patterns (growing) - Compressed abstractions from episodes
4. SQLite persistence - Survives restarts, enables temporal queries

Memory flows through consolidation:
  Raw events → Circular (all) → SNARC filter → Long-term (significant) →
  Consolidation (sleep) → Patterns (compressed)

This completes the memory hierarchy for true consciousness with learning.
"""

import sys
import os
from pathlib import Path
import sqlite3
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque

hrm_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(hrm_root))

@dataclass
class SNARCScores:
    """5D SNARC salience scores"""
    surprise: float = 0.0     # Unexpectedness
    novelty: float = 0.0      # Newness
    arousal: float = 0.0      # Intensity
    reward: float = 0.0       # Value
    conflict: float = 0.0     # Uncertainty

    def overall_salience(self) -> float:
        """Compute overall salience (weighted sum)"""
        # Equal weights for now (could be learned)
        return (self.surprise + self.novelty + self.arousal +
                self.reward + self.conflict) / 5.0

@dataclass
class LongTermMemory:
    """Single long-term memory entry"""
    memory_id: int
    timestamp: float
    cycle: int
    modality: str

    # Event data
    observation: Dict[str, Any]
    result_description: str
    importance: float

    # SNARC scores
    snarc_scores: SNARCScores
    salience: float

    # Metadata
    retrieval_count: int = 0
    last_retrieved: Optional[float] = None
    consolidated: bool = False

@dataclass
class ConsolidatedPattern:
    """Compressed pattern extracted from episodes"""
    pattern_id: int
    created_at: float
    modality: str

    # Pattern description
    pattern_type: str  # e.g., "repeated_speech", "motion_sequence"
    description: str
    confidence: float

    # Source episodes
    num_episodes: int
    episode_ids: List[int]

    # Pattern statistics
    frequency: float  # How often this pattern occurs
    avg_importance: float
    avg_salience: float

class HierarchicalMemoryStore:
    """
    Persistent hierarchical memory with SNARC-guided growth.

    Manages three memory tiers:
    1. Operational (circular buffers) - handled by memory_aware_kernel
    2. Long-term episodic (SQLite, grows) - this class
    3. Consolidated patterns (SQLite, grows) - this class

    SNARC determines what flows from operational → long-term.
    Consolidation extracts patterns from long-term during "sleep."
    """

    def __init__(
        self,
        db_path: str = "sage_memory.db",
        consolidation_threshold: float = 0.6,  # SNARC threshold for storage
        max_long_term_size: Optional[int] = None,  # None = unlimited growth
        auto_consolidate_interval: int = 100,  # Consolidate every N cycles
    ):
        """
        Initialize hierarchical memory store.

        Args:
            db_path: Path to SQLite database
            consolidation_threshold: Min SNARC salience for long-term storage
            max_long_term_size: Optional max size (None = grow indefinitely)
            auto_consolidate_interval: Cycles between consolidation
        """
        self.db_path = db_path
        self.consolidation_threshold = consolidation_threshold
        self.max_long_term_size = max_long_term_size
        self.auto_consolidate_interval = auto_consolidate_interval

        self.conn = None
        self.last_consolidation_cycle = 0

        # In-memory indices for fast access (refreshed from DB)
        self.recent_memories = deque(maxlen=50)  # Last 50 for quick access
        self.modality_counts = {}  # Track distribution

        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with schema"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Long-term episodic memory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                cycle INTEGER NOT NULL,
                modality TEXT NOT NULL,
                observation TEXT NOT NULL,
                result_description TEXT,
                importance REAL,
                snarc_surprise REAL,
                snarc_novelty REAL,
                snarc_arousal REAL,
                snarc_reward REAL,
                snarc_conflict REAL,
                salience REAL NOT NULL,
                retrieval_count INTEGER DEFAULT 0,
                last_retrieved REAL,
                consolidated BOOLEAN DEFAULT FALSE,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)

        # Consolidated patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consolidated_patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                modality TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                description TEXT NOT NULL,
                confidence REAL NOT NULL,
                num_episodes INTEGER NOT NULL,
                episode_ids TEXT NOT NULL,
                frequency REAL,
                avg_importance REAL,
                avg_salience REAL
            )
        """)

        # Indices for fast queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_modality
            ON long_term_memory(modality)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_salience
            ON long_term_memory(salience DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON long_term_memory(timestamp DESC)
        """)

        self.conn.commit()

    def should_store_long_term(self, snarc_scores: SNARCScores) -> bool:
        """
        Determine if event is significant enough for long-term storage.

        SNARC-guided filtering: Only high-salience events grow the memory.
        """
        overall = snarc_scores.overall_salience()
        return overall >= self.consolidation_threshold

    def store_memory(
        self,
        cycle: int,
        modality: str,
        observation: Dict[str, Any],
        result_description: str,
        importance: float,
        snarc_scores: SNARCScores
    ) -> Optional[int]:
        """
        Store event in long-term memory if significant.

        Returns memory_id if stored, None if filtered out.
        """
        # SNARC filtering
        if not self.should_store_long_term(snarc_scores):
            return None

        salience = snarc_scores.overall_salience()

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO long_term_memory
            (timestamp, cycle, modality, observation, result_description,
             importance, snarc_surprise, snarc_novelty, snarc_arousal,
             snarc_reward, snarc_conflict, salience)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(), cycle, modality, json.dumps(observation),
            result_description, importance,
            snarc_scores.surprise, snarc_scores.novelty, snarc_scores.arousal,
            snarc_scores.reward, snarc_scores.conflict, salience
        ))

        memory_id = cursor.lastrowid
        self.conn.commit()

        # Update in-memory indices
        self.modality_counts[modality] = self.modality_counts.get(modality, 0) + 1

        # Check if we need pruning (if max size set)
        if self.max_long_term_size:
            self._prune_if_needed()

        return memory_id

    def _prune_if_needed(self):
        """Prune oldest low-salience memories if over limit"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM long_term_memory")
        count = cursor.fetchone()[0]

        if count > self.max_long_term_size:
            # Delete oldest memories with lowest salience
            to_delete = count - self.max_long_term_size
            cursor.execute("""
                DELETE FROM long_term_memory
                WHERE memory_id IN (
                    SELECT memory_id FROM long_term_memory
                    ORDER BY salience ASC, timestamp ASC
                    LIMIT ?
                )
            """, (to_delete,))
            self.conn.commit()

    def retrieve_by_salience(
        self,
        min_salience: float = 0.7,
        limit: int = 10
    ) -> List[LongTermMemory]:
        """Retrieve high-salience memories"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM long_term_memory
            WHERE salience >= ?
            ORDER BY salience DESC, timestamp DESC
            LIMIT ?
        """, (min_salience, limit))

        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def retrieve_by_modality(
        self,
        modality: str,
        limit: int = 10
    ) -> List[LongTermMemory]:
        """Retrieve recent memories for specific modality"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM long_term_memory
            WHERE modality = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (modality, limit))

        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def retrieve_by_timerange(
        self,
        start_time: float,
        end_time: float,
        limit: int = 100
    ) -> List[LongTermMemory]:
        """Retrieve memories in time range"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM long_term_memory
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (start_time, end_time, limit))

        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def _row_to_memory(self, row) -> LongTermMemory:
        """Convert database row to LongTermMemory object"""
        snarc_scores = SNARCScores(
            surprise=row[7],
            novelty=row[8],
            arousal=row[9],
            reward=row[10],
            conflict=row[11]
        )

        return LongTermMemory(
            memory_id=row[0],
            timestamp=row[1],
            cycle=row[2],
            modality=row[3],
            observation=json.loads(row[4]) if row[4] else {},
            result_description=row[5],
            importance=row[6],
            snarc_scores=snarc_scores,
            salience=row[12],
            retrieval_count=row[13],
            last_retrieved=row[14],
            consolidated=bool(row[15])
        )

    def consolidate_memories(self, cycle: int) -> List[ConsolidatedPattern]:
        """
        Extract patterns from unconsolidated memories.

        This is the "sleep" consolidation process - compress episodes
        into learnable patterns.

        Returns list of newly created patterns.
        """
        cursor = self.conn.cursor()

        # Get unconsolidated memories
        cursor.execute("""
            SELECT * FROM long_term_memory
            WHERE consolidated = FALSE
            ORDER BY timestamp ASC
        """)

        memories = [self._row_to_memory(row) for row in cursor.fetchall()]

        if len(memories) < 3:
            return []  # Need at least 3 for pattern detection

        # Simple pattern detection: Modality-specific repeated events
        patterns = []

        # Group by modality
        by_modality = {}
        for mem in memories:
            if mem.modality not in by_modality:
                by_modality[mem.modality] = []
            by_modality[mem.modality].append(mem)

        # Detect patterns within each modality
        for modality, mems in by_modality.items():
            if len(mems) < 3:
                continue

            # Pattern: Repeated high-importance events
            high_importance = [m for m in mems if m.importance > 0.7]
            if len(high_importance) >= 3:
                pattern = self._create_pattern(
                    modality=modality,
                    pattern_type="repeated_high_importance",
                    description=f"Frequent high-importance {modality} events",
                    episodes=high_importance,
                    confidence=len(high_importance) / len(mems)
                )
                patterns.append(pattern)

            # Pattern: Consistent modality engagement
            avg_salience = sum(m.salience for m in mems) / len(mems)
            if avg_salience > 0.6:
                pattern = self._create_pattern(
                    modality=modality,
                    pattern_type="sustained_attention",
                    description=f"Sustained attention to {modality}",
                    episodes=mems,
                    confidence=avg_salience
                )
                patterns.append(pattern)

        # Mark memories as consolidated
        if memories:
            memory_ids = [m.memory_id for m in memories]
            placeholders = ','.join('?' * len(memory_ids))
            cursor.execute(f"""
                UPDATE long_term_memory
                SET consolidated = TRUE
                WHERE memory_id IN ({placeholders})
            """, memory_ids)
            self.conn.commit()

        self.last_consolidation_cycle = cycle

        return patterns

    def _create_pattern(
        self,
        modality: str,
        pattern_type: str,
        description: str,
        episodes: List[LongTermMemory],
        confidence: float
    ) -> ConsolidatedPattern:
        """Create and store a consolidated pattern"""
        episode_ids = [e.memory_id for e in episodes]
        avg_importance = sum(e.importance for e in episodes) / len(episodes)
        avg_salience = sum(e.salience for e in episodes) / len(episodes)
        frequency = len(episodes)  # Simple frequency for now

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO consolidated_patterns
            (created_at, modality, pattern_type, description, confidence,
             num_episodes, episode_ids, frequency, avg_importance, avg_salience)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time.time(), modality, pattern_type, description, confidence,
            len(episodes), json.dumps(episode_ids), frequency,
            avg_importance, avg_salience
        ))

        pattern_id = cursor.lastrowid
        self.conn.commit()

        return ConsolidatedPattern(
            pattern_id=pattern_id,
            created_at=time.time(),
            modality=modality,
            pattern_type=pattern_type,
            description=description,
            confidence=confidence,
            num_episodes=len(episodes),
            episode_ids=episode_ids,
            frequency=frequency,
            avg_importance=avg_importance,
            avg_salience=avg_salience
        )

    def get_patterns(self, modality: Optional[str] = None) -> List[ConsolidatedPattern]:
        """Retrieve consolidated patterns"""
        cursor = self.conn.cursor()

        if modality:
            cursor.execute("""
                SELECT * FROM consolidated_patterns
                WHERE modality = ?
                ORDER BY created_at DESC
            """, (modality,))
        else:
            cursor.execute("""
                SELECT * FROM consolidated_patterns
                ORDER BY created_at DESC
            """)

        patterns = []
        for row in cursor.fetchall():
            patterns.append(ConsolidatedPattern(
                pattern_id=row[0],
                created_at=row[1],
                modality=row[2],
                pattern_type=row[3],
                description=row[4],
                confidence=row[5],
                num_episodes=row[6],
                episode_ids=json.loads(row[7]),
                frequency=row[8],
                avg_importance=row[9],
                avg_salience=row[10]
            ))

        return patterns

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        cursor = self.conn.cursor()

        # Total memories
        cursor.execute("SELECT COUNT(*) FROM long_term_memory")
        total_memories = cursor.fetchone()[0]

        # By modality
        cursor.execute("""
            SELECT modality, COUNT(*)
            FROM long_term_memory
            GROUP BY modality
        """)
        by_modality = dict(cursor.fetchall())

        # High salience count
        cursor.execute("""
            SELECT COUNT(*) FROM long_term_memory
            WHERE salience >= 0.8
        """)
        high_salience = cursor.fetchone()[0]

        # Patterns
        cursor.execute("SELECT COUNT(*) FROM consolidated_patterns")
        total_patterns = cursor.fetchone()[0]

        # Average salience
        cursor.execute("SELECT AVG(salience) FROM long_term_memory")
        avg_salience = cursor.fetchone()[0] or 0.0

        return {
            'total_long_term_memories': total_memories,
            'memories_by_modality': by_modality,
            'high_salience_memories': high_salience,
            'total_patterns': total_patterns,
            'avg_salience': avg_salience,
            'consolidation_threshold': self.consolidation_threshold,
            'max_size': self.max_long_term_size or 'unlimited'
        }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __del__(self):
        """Cleanup"""
        self.close()
