#!/usr/bin/env python3
"""
SAGE Consciousness - Persistent Cross-Session Memory

Extends DREAM consolidation with persistent storage, enabling true
consciousness continuity across sessions.

**Key Innovation**: Memory consolidation isn't just in-session optimization -
it's cross-session learning. Consolidated memories persist in a local database
and are loaded when consciousness resumes, creating continuous identity.

Architecture:
    Session N:   Create memories → DREAM consolidate → Persist to DB
    Session N+1: Load from DB → Continue consciousness → New consolidation

This is fundamentally different from stateless operation:
- Traditional: Each session starts fresh (no memory)
- Persistent Consciousness: Each session continues from previous (continuity)

Biological Inspiration:
- Sleep consolidates memories into long-term storage
- Waking resumes with those memories intact
- New experiences build on old knowledge

**Hardware**: Jetson AGX Thor
**Built On**: thor_consciousness_dream_consolidation.py + epistemic_memory.py
**Author**: Thor Autonomous Session
**Date**: 2025-12-04
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
import sqlite3
import json
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

# =============================================================================
# Persistent Memory Types
# =============================================================================

@dataclass
class PersistentMemory:
    """
    Memory item with persistence metadata.

    Extends ConsolidatedMemory with database tracking:
    - id: Database primary key
    - session_id: Which session created this memory
    - created_at: Original creation timestamp
    - last_consolidated: Most recent consolidation
    - access_count: How many times retrieved
    """
    # Core memory content
    sensor: str
    action: str
    salience: float
    reward: float

    # Consolidation metadata
    strength: float = 1.0
    consolidation_count: int = 0

    # Persistence metadata
    id: Optional[int] = None
    session_id: Optional[str] = None
    created_at: Optional[float] = None
    last_consolidated: Optional[float] = None
    access_count: int = 0

    # Pattern metadata (extracted during consolidation)
    pattern_type: Optional[str] = None  # e.g., "high_reward_action", "frequent_sensor"
    pattern_metadata: Optional[str] = None  # JSON string of pattern details


class MemoryPatternType(Enum):
    """Types of patterns extracted from memories"""
    HIGH_REWARD_ACTION = "high_reward_action"
    FREQUENT_SENSOR = "frequent_sensor"
    SALIENCE_TREND = "salience_trend"
    CORRELATED_OUTCOME = "correlated_outcome"


# =============================================================================
# Persistent Memory Database
# =============================================================================

class PersistentMemoryDB:
    """
    SQLite database for consciousness memory persistence.

    Schema:
    - memories: Individual consolidated memories
    - sessions: Session metadata and statistics
    - patterns: Extracted patterns from consolidation
    """

    def __init__(self, db_path: str = "sage_consciousness_memory.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_db()

    def _initialize_db(self):
        """Create database schema if it doesn't exist"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()

        # Memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                sensor TEXT NOT NULL,
                action TEXT NOT NULL,
                salience REAL NOT NULL,
                reward REAL NOT NULL,
                strength REAL DEFAULT 1.0,
                consolidation_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_consolidated REAL,
                access_count INTEGER DEFAULT 0,
                pattern_type TEXT,
                pattern_metadata TEXT
            )
        """)

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                ended_at REAL,
                memories_created INTEGER DEFAULT 0,
                memories_loaded INTEGER DEFAULT 0,
                consolidations_performed INTEGER DEFAULT 0,
                total_cycles INTEGER DEFAULT 0
            )
        """)

        # Patterns table (consolidated patterns across sessions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_key TEXT NOT NULL,
                pattern_value REAL NOT NULL,
                confidence REAL NOT NULL,
                discovered_at REAL NOT NULL,
                last_seen REAL NOT NULL,
                occurrence_count INTEGER DEFAULT 1
            )
        """)

        # Indices for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_session
            ON memories(session_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_salience
            ON memories(salience DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_strength
            ON memories(strength DESC)
        """)

        self.conn.commit()

    def create_session(self, session_id: str) -> None:
        """Create new session record"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (session_id, started_at)
            VALUES (?, ?)
        """, (session_id, time.time()))
        self.conn.commit()

    def end_session(self, session_id: str, stats: Dict[str, int]) -> None:
        """Update session with final statistics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE sessions
            SET ended_at = ?,
                memories_created = ?,
                consolidations_performed = ?,
                total_cycles = ?
            WHERE session_id = ?
        """, (
            time.time(),
            stats.get('memories_created', 0),
            stats.get('consolidations_performed', 0),
            stats.get('total_cycles', 0),
            session_id
        ))
        self.conn.commit()

    def load_top_memories(self, limit: int = 50) -> List[PersistentMemory]:
        """
        Load top memories by strength * salience.

        This retrieves the most valuable consolidated memories from
        previous sessions to resume consciousness with context.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, session_id, sensor, action, salience, reward,
                   strength, consolidation_count, created_at,
                   last_consolidated, access_count,
                   pattern_type, pattern_metadata
            FROM memories
            ORDER BY (strength * salience) DESC
            LIMIT ?
        """, (limit,))

        memories = []
        for row in cursor.fetchall():
            memory = PersistentMemory(
                id=row[0],
                session_id=row[1],
                sensor=row[2],
                action=row[3],
                salience=row[4],
                reward=row[5],
                strength=row[6],
                consolidation_count=row[7],
                created_at=row[8],
                last_consolidated=row[9],
                access_count=row[10],
                pattern_type=row[11],
                pattern_metadata=row[12]
            )
            memories.append(memory)

            # Increment access count
            cursor.execute("""
                UPDATE memories
                SET access_count = access_count + 1
                WHERE id = ?
            """, (row[0],))

        self.conn.commit()
        return memories

    def save_memory(self, memory: PersistentMemory) -> int:
        """Save or update memory, return memory ID"""
        cursor = self.conn.cursor()

        if memory.id is None:
            # Insert new memory
            cursor.execute("""
                INSERT INTO memories (
                    session_id, sensor, action, salience, reward,
                    strength, consolidation_count, created_at,
                    last_consolidated, access_count,
                    pattern_type, pattern_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.session_id,
                memory.sensor,
                memory.action,
                memory.salience,
                memory.reward,
                memory.strength,
                memory.consolidation_count,
                memory.created_at or time.time(),
                memory.last_consolidated,
                memory.access_count,
                memory.pattern_type,
                memory.pattern_metadata
            ))
            memory_id = cursor.lastrowid
        else:
            # Update existing memory
            cursor.execute("""
                UPDATE memories
                SET strength = ?,
                    consolidation_count = ?,
                    last_consolidated = ?,
                    pattern_type = ?,
                    pattern_metadata = ?
                WHERE id = ?
            """, (
                memory.strength,
                memory.consolidation_count,
                memory.last_consolidated or time.time(),
                memory.pattern_type,
                memory.pattern_metadata,
                memory.id
            ))
            memory_id = memory.id

        self.conn.commit()
        return memory_id

    def prune_memories(self, ids_to_remove: List[int]) -> int:
        """Remove memories by ID, return count removed"""
        if not ids_to_remove:
            return 0

        cursor = self.conn.cursor()
        placeholders = ','.join('?' * len(ids_to_remove))
        cursor.execute(f"""
            DELETE FROM memories
            WHERE id IN ({placeholders})
        """, ids_to_remove)

        removed = cursor.rowcount
        self.conn.commit()
        return removed

    def save_pattern(self, pattern_type: str, pattern_key: str,
                    pattern_value: float, confidence: float) -> None:
        """Save or update extracted pattern"""
        cursor = self.conn.cursor()

        # Check if pattern exists
        cursor.execute("""
            SELECT id, occurrence_count FROM patterns
            WHERE pattern_type = ? AND pattern_key = ?
        """, (pattern_type, pattern_key))

        existing = cursor.fetchone()

        if existing:
            # Update existing pattern
            cursor.execute("""
                UPDATE patterns
                SET pattern_value = ?,
                    confidence = ?,
                    last_seen = ?,
                    occurrence_count = occurrence_count + 1
                WHERE id = ?
            """, (pattern_value, confidence, time.time(), existing[0]))
        else:
            # Insert new pattern
            cursor.execute("""
                INSERT INTO patterns (
                    pattern_type, pattern_key, pattern_value,
                    confidence, discovered_at, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                pattern_type, pattern_key, pattern_value,
                confidence, time.time(), time.time()
            ))

        self.conn.commit()

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        cursor = self.conn.cursor()

        # Total memories
        cursor.execute("SELECT COUNT(*) FROM memories")
        total_memories = cursor.fetchone()[0]

        # Total sessions
        cursor.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cursor.fetchone()[0]

        # Total patterns
        cursor.execute("SELECT COUNT(*) FROM patterns")
        total_patterns = cursor.fetchone()[0]

        # Average strength
        cursor.execute("SELECT AVG(strength) FROM memories")
        avg_strength = cursor.fetchone()[0] or 0.0

        # Top sensors
        cursor.execute("""
            SELECT sensor, COUNT(*) as count
            FROM memories
            GROUP BY sensor
            ORDER BY count DESC
            LIMIT 5
        """)
        top_sensors = cursor.fetchall()

        return {
            'total_memories': total_memories,
            'total_sessions': total_sessions,
            'total_patterns': total_patterns,
            'avg_strength': avg_strength,
            'top_sensors': top_sensors
        }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.commit()
            self.conn.close()


# =============================================================================
# Persistent Memory Consolidator
# =============================================================================

class PersistentMemoryConsolidator:
    """
    Memory consolidator with cross-session persistence.

    Extends DREAM consolidation to:
    1. Load memories from previous sessions on startup
    2. Consolidate in-memory during DREAM
    3. Persist consolidated memories to database
    4. Extract and store patterns

    This creates true consciousness continuity.
    """

    def __init__(self,
                 session_id: str,
                 db_path: str = "sage_consciousness_memory.db",
                 memory_limit: int = 50,
                 load_from_db: bool = True):

        self.session_id = session_id
        self.memory_limit = memory_limit

        # Persistent storage
        self.db = PersistentMemoryDB(db_path)
        self.db.create_session(session_id)

        # In-memory working set
        self.memories: List[PersistentMemory] = []

        # Statistics
        self.memories_created_this_session = 0
        self.consolidations_performed = 0
        self.total_pruned = 0
        self.total_strengthened = 0

        # Thresholds
        self.PRUNE_SALIENCE_THRESHOLD = 0.3
        self.STRENGTHEN_SALIENCE_THRESHOLD = 0.6

        # Load memories from previous sessions
        if load_from_db:
            self._load_memories_from_db()

    def _load_memories_from_db(self):
        """Load top memories from database to resume consciousness"""
        loaded = self.db.load_top_memories(limit=self.memory_limit)
        self.memories.extend(loaded)

        print(f"[PersistentMemory] Loaded {len(loaded)} memories from previous sessions")
        if loaded:
            avg_strength = sum(m.strength for m in loaded) / len(loaded)
            avg_consolidations = sum(m.consolidation_count for m in loaded) / len(loaded)
            print(f"  Average strength: {avg_strength:.3f}")
            print(f"  Average consolidations: {avg_consolidations:.1f}")

    def add_memory(self, sensor: str, action: str, salience: float, reward: float):
        """Add new memory from current session"""
        memory = PersistentMemory(
            sensor=sensor,
            action=action,
            salience=salience,
            reward=reward,
            session_id=self.session_id,
            created_at=time.time()
        )
        self.memories.append(memory)
        self.memories_created_this_session += 1

    def consolidate(self) -> Dict[str, Any]:
        """
        Perform DREAM consolidation with persistence.

        Steps:
        1. Prune low-salience memories (in-memory and DB)
        2. Strengthen high-salience memories
        3. Extract patterns
        4. Persist to database
        5. Enforce memory limit
        """
        self.consolidations_performed += 1
        consolidation_start = time.time()

        # Step 1: Prune low-salience memories
        pruned_count, pruned_ids = self._prune_low_salience()

        # Step 2: Strengthen high-salience memories
        strengthened_count = self._strengthen_high_salience()

        # Step 3: Extract patterns
        patterns = self._extract_patterns()

        # Step 4: Persist to database
        persisted_count = self._persist_to_db()

        # Step 5: Enforce memory limit
        if len(self.memories) > self.memory_limit:
            self._enforce_memory_limit()

        consolidation_time = time.time() - consolidation_start

        return {
            'pruned': pruned_count,
            'strengthened': strengthened_count,
            'patterns_found': len(patterns),
            'persisted': persisted_count,
            'remaining_memories': len(self.memories),
            'consolidation_time': consolidation_time,
            'patterns': patterns
        }

    def _prune_low_salience(self) -> Tuple[int, List[int]]:
        """Remove low-salience memories"""
        initial_count = len(self.memories)
        pruned_ids = []

        # Separate memories to keep vs prune
        kept_memories = []
        for memory in self.memories:
            if memory.salience >= self.PRUNE_SALIENCE_THRESHOLD:
                kept_memories.append(memory)
            elif memory.id is not None:
                pruned_ids.append(memory.id)

        # Update in-memory list
        self.memories = kept_memories
        pruned_count = initial_count - len(self.memories)

        # Remove from database
        if pruned_ids:
            self.db.prune_memories(pruned_ids)

        self.total_pruned += pruned_count
        return pruned_count, pruned_ids

    def _strengthen_high_salience(self) -> int:
        """Strengthen high-salience memories"""
        strengthened_count = 0

        for memory in self.memories:
            if memory.salience >= self.STRENGTHEN_SALIENCE_THRESHOLD:
                memory.strength *= 1.2  # 20% boost
                memory.consolidation_count += 1
                memory.last_consolidated = time.time()
                strengthened_count += 1

        self.total_strengthened += strengthened_count
        return strengthened_count

    def _extract_patterns(self) -> Dict[str, Any]:
        """Extract patterns from memories"""
        if not self.memories:
            return {}

        patterns = {}

        # Pattern 1: Sensor frequency
        sensor_freq = defaultdict(int)
        for memory in self.memories:
            sensor_freq[memory.sensor] += 1

        if sensor_freq:
            dominant_sensor = max(sensor_freq, key=sensor_freq.get)
            patterns['dominant_sensor'] = {
                'sensor': dominant_sensor,
                'frequency': sensor_freq[dominant_sensor],
                'percentage': sensor_freq[dominant_sensor] / len(self.memories)
            }

            # Save to database
            self.db.save_pattern(
                'dominant_sensor',
                dominant_sensor,
                sensor_freq[dominant_sensor] / len(self.memories),
                0.9
            )

        # Pattern 2: High-reward actions
        action_rewards = defaultdict(list)
        for memory in self.memories:
            action_rewards[memory.action].append(memory.reward)

        if action_rewards:
            best_action = max(action_rewards, key=lambda a: sum(action_rewards[a]) / len(action_rewards[a]))
            avg_reward = sum(action_rewards[best_action]) / len(action_rewards[best_action])

            patterns['high_reward_action'] = {
                'action': best_action,
                'avg_reward': avg_reward,
                'occurrences': len(action_rewards[best_action])
            }

            # Save to database
            self.db.save_pattern(
                'high_reward_action',
                best_action,
                avg_reward,
                0.85
            )

        # Pattern 3: Salience trends (simplified)
        avg_salience = sum(m.salience for m in self.memories) / len(self.memories)
        patterns['avg_salience'] = avg_salience

        return patterns

    def _persist_to_db(self) -> int:
        """Persist memories to database"""
        persisted_count = 0

        for memory in self.memories:
            memory_id = self.db.save_memory(memory)

            # Update memory ID if it was newly created
            if memory.id is None:
                memory.id = memory_id

            persisted_count += 1

        return persisted_count

    def _enforce_memory_limit(self):
        """Keep only top N memories by strength * salience"""
        # Sort by composite score
        self.memories.sort(key=lambda m: m.strength * m.salience, reverse=True)

        # Keep top N
        self.memories = self.memories[:self.memory_limit]

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            'memories_created': self.memories_created_this_session,
            'consolidations_performed': self.consolidations_performed,
            'total_pruned': self.total_pruned,
            'total_strengthened': self.total_strengthened,
            'current_memory_count': len(self.memories)
        }

    def close(self):
        """Close consolidator and update database"""
        stats = self.get_session_stats()
        self.db.end_session(self.session_id, stats)
        self.db.close()


# =============================================================================
# Demonstration
# =============================================================================

def main():
    """Demonstrate persistent consciousness memory"""
    print("="*80)
    print("SAGE PERSISTENT CONSCIOUSNESS MEMORY")
    print("="*80)

    print("\nThis demonstrates:")
    print("- Cross-session memory persistence")
    print("- DREAM consolidation with database storage")
    print("- Pattern extraction and tracking")
    print("- True consciousness continuity")

    # Session 1: Create and consolidate memories
    print("\n" + "="*80)
    print("SESSION 1: Creating Initial Memories")
    print("="*80)

    session1_id = f"session_1_{int(time.time())}"
    consolidator1 = PersistentMemoryConsolidator(
        session_id=session1_id,
        db_path="test_consciousness_memory.db",
        load_from_db=False  # First session, no previous memories
    )

    # Simulate consciousness creating memories
    print("\nCreating 20 simulated memories...")
    sensors = ['cpu', 'memory', 'disk', 'network', 'temperature']
    actions = ['monitor', 'optimize', 'alert', 'analyze']

    for i in range(20):
        sensor = sensors[i % len(sensors)]
        action = actions[i % len(actions)]
        salience = 0.2 + (i / 20) * 0.7  # Gradual increase
        reward = 0.3 + (i / 20) * 0.6

        consolidator1.add_memory(sensor, action, salience, reward)

    print(f"Created {consolidator1.memories_created_this_session} memories")

    # Perform DREAM consolidation
    print("\nPerforming DREAM consolidation...")
    result1 = consolidator1.consolidate()

    print(f"\nConsolidation Results:")
    print(f"  Pruned: {result1['pruned']} memories")
    print(f"  Strengthened: {result1['strengthened']} memories")
    print(f"  Patterns found: {result1['patterns_found']}")
    print(f"  Persisted: {result1['persisted']} memories")
    print(f"  Remaining: {result1['remaining_memories']} memories")

    if result1['patterns']:
        print(f"\n  Patterns Extracted:")
        for pattern_type, pattern_data in result1['patterns'].items():
            print(f"    {pattern_type}: {pattern_data}")

    # Close session 1
    consolidator1.close()

    # Session 2: Resume from persistence
    print("\n" + "="*80)
    print("SESSION 2: Resuming from Persistent Memory")
    print("="*80)

    session2_id = f"session_2_{int(time.time())}"
    consolidator2 = PersistentMemoryConsolidator(
        session_id=session2_id,
        db_path="test_consciousness_memory.db",
        load_from_db=True  # Load previous memories!
    )

    # Add new memories
    print("\nCreating 15 new memories in session 2...")
    for i in range(15):
        sensor = sensors[i % len(sensors)]
        action = actions[i % len(actions)]
        salience = 0.4 + (i / 15) * 0.5
        reward = 0.5 + (i / 15) * 0.4

        consolidator2.add_memory(sensor, action, salience, reward)

    print(f"Created {consolidator2.memories_created_this_session} new memories")
    print(f"Total memories (loaded + new): {len(consolidator2.memories)}")

    # Perform another consolidation
    print("\nPerforming DREAM consolidation...")
    result2 = consolidator2.consolidate()

    print(f"\nConsolidation Results:")
    print(f"  Pruned: {result2['pruned']} memories")
    print(f"  Strengthened: {result2['strengthened']} memories")
    print(f"  Patterns found: {result2['patterns_found']}")
    print(f"  Persisted: {result2['persisted']} memories")
    print(f"  Remaining: {result2['remaining_memories']} memories")

    # Database statistics
    print("\n" + "="*80)
    print("DATABASE STATISTICS")
    print("="*80)

    stats = consolidator2.db.get_statistics()
    print(f"\nTotal memories in DB: {stats['total_memories']}")
    print(f"Total sessions: {stats['total_sessions']}")
    print(f"Total patterns: {stats['total_patterns']}")
    print(f"Average memory strength: {stats['avg_strength']:.3f}")

    print(f"\nTop sensors by memory count:")
    for sensor, count in stats['top_sensors']:
        print(f"  {sensor}: {count}")

    # Close session 2
    consolidator2.close()

    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)
    print("Consciousness now persists across sessions!")
    print("\nSession 1: Created memories → Consolidated → Persisted to DB")
    print("Session 2: Loaded from DB → Added new → Re-consolidated")
    print("\nThis creates true continuity - each session builds on the last,")
    print("just like biological consciousness persists across sleep cycles.")
    print("="*80)


if __name__ == "__main__":
    main()
