#!/usr/bin/env python3
"""
Long-Term Memory (LTM) for SAGE SNARC System
=============================================

Persistent storage for high-salience episodic memories.
Consolidates important experiences from STM for long-term retention.

Track 2: SNARC Memory - Component 2/3
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import time
import json
import pickle
from pathlib import Path
from collections import defaultdict

# Import SNARC data structures and STM
try:
    from sage.services.snarc.data_structures import (
        SalienceReport,
        SalienceBreakdown,
        CognitiveStance
    )
    from sage.memory.stm import STMEntry
except ModuleNotFoundError:
    # Fallback for standalone testing
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).parent.parent.parent))
    from sage.services.snarc.data_structures import (
        SalienceReport,
        SalienceBreakdown,
        CognitiveStance
    )
    from sage.memory.stm import STMEntry


@dataclass
class EpisodicMemory:
    """
    Single episodic memory in LTM

    Compressed representation of a significant experience:
    - When it happened
    - What was salient about it
    - What sensors were involved
    - What action was taken
    - What the outcome was
    - Why it was important

    Semantic compression: Store meaning, not raw details.
    """
    memory_id: str  # Unique identifier
    timestamp: float  # When this occurred
    cycle_id: int  # Cycle number

    # Salience information
    salience_score: float  # Why this was memorable
    salience_breakdown: Dict[str, float]  # 5D breakdown
    focus_target: str  # Which sensor/region
    cognitive_stance: str  # CognitiveStance value

    # Compressed sensor state (not full tensors!)
    sensor_summary: Dict[str, Any]  # High-level summary per sensor

    # Action and outcome
    action_taken: Optional[str] = None
    reward: Optional[float] = None
    outcome_success: Optional[bool] = None

    # Context tags for retrieval
    tags: List[str] = field(default_factory=list)

    # Metadata
    consolidation_timestamp: float = field(default_factory=time.time)
    access_count: int = 0  # How many times retrieved
    last_access: Optional[float] = None

    def __post_init__(self):
        """Validate memory"""
        if not 0.0 <= self.salience_score <= 1.0:
            raise ValueError(f"salience_score must be 0-1, got {self.salience_score}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'memory_id': self.memory_id,
            'timestamp': self.timestamp,
            'cycle_id': self.cycle_id,
            'salience_score': self.salience_score,
            'salience_breakdown': self.salience_breakdown,
            'focus_target': self.focus_target,
            'cognitive_stance': self.cognitive_stance,
            'sensor_summary': self.sensor_summary,
            'action_taken': self.action_taken,
            'reward': self.reward,
            'outcome_success': self.outcome_success,
            'tags': self.tags,
            'consolidation_timestamp': self.consolidation_timestamp,
            'access_count': self.access_count,
            'last_access': self.last_access
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodicMemory':
        """Create from dictionary"""
        return cls(**data)

    def mark_accessed(self):
        """Mark this memory as accessed (for usage tracking)"""
        self.access_count += 1
        self.last_access = time.time()


class LongTermMemory:
    """
    Persistent storage for high-salience episodic memories

    Design:
    - Store only high-salience events (threshold-based)
    - Compress sensor data (summary, not raw tensors)
    - Disk-backed for persistence
    - Indexed for fast retrieval
    - Decay/forgetting for low-importance memories

    Integration:
    - Populated from STM consolidation
    - Queried by SNARC for context
    - Queried by retrieval for similarity search
    - Persistent across sessions
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        consolidation_threshold: float = 0.7,
        max_memories: int = 10000,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Long-Term Memory

        Args:
            storage_path: Directory to store persistent memories
            consolidation_threshold: Minimum salience to consolidate
            max_memories: Maximum memories to retain
            device: Device for tensor operations
        """
        self.storage_path = storage_path or Path.home() / '.sage' / 'ltm'
        self.consolidation_threshold = consolidation_threshold
        self.max_memories = max_memories
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory index
        self.memories: Dict[str, EpisodicMemory] = {}

        # Indices for fast retrieval
        self._salience_index: List[Tuple[float, str]] = []  # (salience, memory_id)
        self._time_index: List[Tuple[float, str]] = []  # (timestamp, memory_id)
        self._sensor_index: Dict[str, List[str]] = defaultdict(list)  # sensor -> [memory_ids]
        self._stance_index: Dict[str, List[str]] = defaultdict(list)  # stance -> [memory_ids]
        self._tag_index: Dict[str, List[str]] = defaultdict(list)  # tag -> [memory_ids]

        # Statistics
        self.total_consolidations = 0
        self.total_retrievals = 0
        self.forgetting_events = 0

        # Load existing memories
        self._load_from_disk()

    def consolidate_from_stm(self, stm_entry: STMEntry) -> Optional[EpisodicMemory]:
        """
        Consolidate STM entry into LTM (if salient enough)

        Args:
            stm_entry: Entry from short-term memory

        Returns:
            EpisodicMemory if consolidated, None if not salient enough
        """
        # Check salience threshold
        if stm_entry.salience_report.salience_score < self.consolidation_threshold:
            return None

        # Compress sensor data (don't store full tensors!)
        sensor_summary = self._compress_sensor_data(stm_entry.sensor_snapshots)

        # Create episodic memory
        memory_id = f"mem_{int(time.time() * 1000)}_{stm_entry.cycle_id}"

        memory = EpisodicMemory(
            memory_id=memory_id,
            timestamp=stm_entry.timestamp,
            cycle_id=stm_entry.cycle_id,
            salience_score=stm_entry.salience_report.salience_score,
            salience_breakdown={
                'surprise': stm_entry.salience_report.salience_breakdown.surprise,
                'novelty': stm_entry.salience_report.salience_breakdown.novelty,
                'arousal': stm_entry.salience_report.salience_breakdown.arousal,
                'reward': stm_entry.salience_report.salience_breakdown.reward,
                'conflict': stm_entry.salience_report.salience_breakdown.conflict
            },
            focus_target=stm_entry.salience_report.focus_target,
            cognitive_stance=stm_entry.salience_report.suggested_stance.value,
            sensor_summary=sensor_summary,
            action_taken=str(stm_entry.action_taken) if stm_entry.action_taken is not None else None,
            reward=stm_entry.reward,
            outcome_success=stm_entry.outcome_success,
            tags=self._generate_tags(stm_entry)
        )

        # Store
        self._add_memory(memory)

        return memory

    def _compress_sensor_data(self, sensor_snapshots: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress sensor data for long-term storage

        Instead of storing full tensors, store:
        - Shape and dtype
        - Mean and std
        - Min and max
        - Norm (magnitude)

        Args:
            sensor_snapshots: Raw sensor data

        Returns:
            Compressed sensor summary
        """
        summary = {}

        for sensor_id, data in sensor_snapshots.items():
            if isinstance(data, torch.Tensor):
                summary[sensor_id] = {
                    'type': 'tensor',
                    'shape': list(data.shape),
                    'dtype': str(data.dtype),
                    'mean': float(data.mean().item()),
                    'std': float(data.std().item()),
                    'min': float(data.min().item()),
                    'max': float(data.max().item()),
                    'norm': float(torch.norm(data).item())
                }
            elif isinstance(data, np.ndarray):
                summary[sensor_id] = {
                    'type': 'numpy',
                    'shape': list(data.shape),
                    'dtype': str(data.dtype),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'norm': float(np.linalg.norm(data))
                }
            else:
                summary[sensor_id] = {
                    'type': 'other',
                    'value': str(data)[:100]  # Truncate long strings
                }

        return summary

    def _generate_tags(self, stm_entry: STMEntry) -> List[str]:
        """
        Generate retrieval tags for memory

        Args:
            stm_entry: STM entry to tag

        Returns:
            List of tag strings
        """
        tags = []

        # Sensor tag
        tags.append(f"sensor:{stm_entry.salience_report.focus_target}")

        # Stance tag
        tags.append(f"stance:{stm_entry.salience_report.suggested_stance.value}")

        # Salience level tags
        sal = stm_entry.salience_report.salience_score
        if sal >= 0.9:
            tags.append("salience:critical")
        elif sal >= 0.7:
            tags.append("salience:high")
        elif sal >= 0.5:
            tags.append("salience:medium")

        # Outcome tags
        if stm_entry.outcome_success is not None:
            tags.append(f"outcome:{'success' if stm_entry.outcome_success else 'failure'}")

        # Dimension-specific tags (for dominant dimensions)
        bd = stm_entry.salience_report.salience_breakdown
        dims = {
            'surprise': bd.surprise,
            'novelty': bd.novelty,
            'arousal': bd.arousal,
            'reward': bd.reward,
            'conflict': bd.conflict
        }

        # Tag dimensions above 0.7
        for dim, value in dims.items():
            if value >= 0.7:
                tags.append(f"dim:{dim}")

        return tags

    def _add_memory(self, memory: EpisodicMemory):
        """
        Add memory to LTM and update indices

        Args:
            memory: Episodic memory to add
        """
        # Add to main storage
        self.memories[memory.memory_id] = memory

        # Update indices
        self._salience_index.append((memory.salience_score, memory.memory_id))
        self._salience_index.sort(reverse=True, key=lambda x: x[0])

        self._time_index.append((memory.timestamp, memory.memory_id))
        self._time_index.sort(reverse=True, key=lambda x: x[0])

        self._sensor_index[memory.focus_target].append(memory.memory_id)
        self._stance_index[memory.cognitive_stance].append(memory.memory_id)

        for tag in memory.tags:
            self._tag_index[tag].append(memory.memory_id)

        # Update stats
        self.total_consolidations += 1

        # Check capacity and prune if needed
        if len(self.memories) > self.max_memories:
            self._prune_memories()

        # Persist to disk
        self._save_memory(memory)

    def get_by_id(self, memory_id: str) -> Optional[EpisodicMemory]:
        """
        Retrieve memory by ID

        Args:
            memory_id: Memory identifier

        Returns:
            EpisodicMemory if found, None otherwise
        """
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.mark_accessed()
            self.total_retrievals += 1
            return memory
        return None

    def get_most_salient(self, n: int = 10) -> List[EpisodicMemory]:
        """
        Get N most salient memories

        Args:
            n: Number of memories to retrieve

        Returns:
            List of memories (sorted by salience, descending)
        """
        memory_ids = [mid for _, mid in self._salience_index[:n]]
        memories = [self.memories[mid] for mid in memory_ids if mid in self.memories]

        for mem in memories:
            mem.mark_accessed()

        self.total_retrievals += len(memories)

        return memories

    def get_recent(self, n: int = 10) -> List[EpisodicMemory]:
        """
        Get N most recent memories

        Args:
            n: Number of memories to retrieve

        Returns:
            List of memories (sorted by time, newest first)
        """
        memory_ids = [mid for _, mid in self._time_index[:n]]
        memories = [self.memories[mid] for mid in memory_ids if mid in self.memories]

        for mem in memories:
            mem.mark_accessed()

        self.total_retrievals += len(memories)

        return memories

    def query_by_sensor(self, sensor_id: str, n: int = 10) -> List[EpisodicMemory]:
        """
        Query memories focused on specific sensor

        Args:
            sensor_id: Sensor to filter by
            n: Maximum number of memories

        Returns:
            List of memories (sorted by salience)
        """
        memory_ids = self._sensor_index.get(sensor_id, [])

        # Sort by salience
        memories = [(self.memories[mid].salience_score, mid) for mid in memory_ids if mid in self.memories]
        memories.sort(reverse=True, key=lambda x: x[0])

        result = [self.memories[mid] for _, mid in memories[:n]]

        for mem in result:
            mem.mark_accessed()

        self.total_retrievals += len(result)

        return result

    def query_by_tags(self, tags: List[str], match_all: bool = False, n: int = 10) -> List[EpisodicMemory]:
        """
        Query memories by tags

        Args:
            tags: Tags to search for
            match_all: If True, memory must have all tags. If False, any tag matches.
            n: Maximum number of memories

        Returns:
            List of memories (sorted by salience)
        """
        if not tags:
            return []

        # Get candidate memory IDs
        if match_all:
            # Intersection of all tag indices
            candidates = set(self._tag_index.get(tags[0], []))
            for tag in tags[1:]:
                candidates &= set(self._tag_index.get(tag, []))
        else:
            # Union of all tag indices
            candidates = set()
            for tag in tags:
                candidates |= set(self._tag_index.get(tag, []))

        # Sort by salience
        memories = [(self.memories[mid].salience_score, mid) for mid in candidates if mid in self.memories]
        memories.sort(reverse=True, key=lambda x: x[0])

        result = [self.memories[mid] for _, mid in memories[:n]]

        for mem in result:
            mem.mark_accessed()

        self.total_retrievals += len(result)

        return result

    def query_time_window(self, start_time: float, end_time: float, n: Optional[int] = None) -> List[EpisodicMemory]:
        """
        Query memories within time window

        Args:
            start_time: Start timestamp
            end_time: End timestamp
            n: Maximum number (None = all)

        Returns:
            List of memories in time range
        """
        memories = [
            (timestamp, mid) for timestamp, mid in self._time_index
            if start_time <= timestamp <= end_time
        ]

        if n is not None:
            memories = memories[:n]

        result = [self.memories[mid] for _, mid in memories if mid in self.memories]

        for mem in result:
            mem.mark_accessed()

        self.total_retrievals += len(result)

        return result

    def _prune_memories(self, target_size: Optional[int] = None):
        """
        Prune low-importance memories when capacity exceeded

        Strategy: Remove memories with:
        - Low salience
        - Low access count
        - Old consolidation time

        Args:
            target_size: Target number of memories to retain (default: 90% of max)
        """
        if target_size is None:
            target_size = int(self.max_memories * 0.9)

        num_to_remove = len(self.memories) - target_size

        if num_to_remove <= 0:
            return

        # Score each memory for retention
        retention_scores = []
        for memory_id, memory in self.memories.items():
            # Score based on: salience (50%), access count (30%), age (20%)
            age_score = 1.0 / (1.0 + (time.time() - memory.consolidation_timestamp) / 86400)  # Decay over days
            access_score = min(memory.access_count / 10.0, 1.0)  # Normalize to 10 accesses

            retention_score = (
                0.5 * memory.salience_score +
                0.3 * access_score +
                0.2 * age_score
            )

            retention_scores.append((retention_score, memory_id))

        # Sort by retention score (ascending = worst first)
        retention_scores.sort(key=lambda x: x[0])

        # Remove lowest-scoring memories
        for i in range(num_to_remove):
            _, memory_id = retention_scores[i]
            self._remove_memory(memory_id)
            self.forgetting_events += 1

    def _remove_memory(self, memory_id: str):
        """
        Remove memory from LTM and all indices

        Args:
            memory_id: ID of memory to remove
        """
        if memory_id not in self.memories:
            return

        memory = self.memories[memory_id]

        # Remove from indices
        self._salience_index = [(sal, mid) for sal, mid in self._salience_index if mid != memory_id]
        self._time_index = [(ts, mid) for ts, mid in self._time_index if mid != memory_id]

        self._sensor_index[memory.focus_target] = [
            mid for mid in self._sensor_index[memory.focus_target] if mid != memory_id
        ]

        self._stance_index[memory.cognitive_stance] = [
            mid for mid in self._stance_index[memory.cognitive_stance] if mid != memory_id
        ]

        for tag in memory.tags:
            self._tag_index[tag] = [mid for mid in self._tag_index[tag] if mid != memory_id]

        # Remove from storage
        del self.memories[memory_id]

        # Remove from disk
        memory_file = self.storage_path / f"{memory_id}.json"
        if memory_file.exists():
            memory_file.unlink()

    def _save_memory(self, memory: EpisodicMemory):
        """Save memory to disk"""
        memory_file = self.storage_path / f"{memory.memory_id}.json"
        with open(memory_file, 'w') as f:
            json.dump(memory.to_dict(), f, indent=2)

    def _load_from_disk(self):
        """Load all memories from disk on initialization"""
        if not self.storage_path.exists():
            return

        memory_files = list(self.storage_path.glob("mem_*.json"))

        for memory_file in memory_files:
            try:
                with open(memory_file, 'r') as f:
                    data = json.load(f)
                    memory = EpisodicMemory.from_dict(data)
                    self.memories[memory.memory_id] = memory

                    # Rebuild indices
                    self._salience_index.append((memory.salience_score, memory.memory_id))
                    self._time_index.append((memory.timestamp, memory.memory_id))
                    self._sensor_index[memory.focus_target].append(memory.memory_id)
                    self._stance_index[memory.cognitive_stance].append(memory.memory_id)
                    for tag in memory.tags:
                        self._tag_index[tag].append(memory.memory_id)

            except Exception as e:
                print(f"Warning: Failed to load {memory_file}: {e}")

        # Sort indices
        self._salience_index.sort(reverse=True, key=lambda x: x[0])
        self._time_index.sort(reverse=True, key=lambda x: x[0])

    def get_stats(self) -> Dict[str, Any]:
        """Get LTM statistics"""
        return {
            'total_memories': len(self.memories),
            'capacity': self.max_memories,
            'utilization': len(self.memories) / self.max_memories,
            'consolidation_threshold': self.consolidation_threshold,
            'total_consolidations': self.total_consolidations,
            'total_retrievals': self.total_retrievals,
            'forgetting_events': self.forgetting_events,
            'unique_sensors': len(self._sensor_index),
            'unique_stances': len(self._stance_index),
            'unique_tags': len(self._tag_index),
            'storage_path': str(self.storage_path)
        }

    def clear(self):
        """Clear all memories (reset to empty state)"""
        # Clear in-memory
        self.memories.clear()
        self._salience_index.clear()
        self._time_index.clear()
        self._sensor_index.clear()
        self._stance_index.clear()
        self._tag_index.clear()

        # Clear disk
        for memory_file in self.storage_path.glob("mem_*.json"):
            memory_file.unlink()

        # Reset stats
        self.total_consolidations = 0
        self.total_retrievals = 0
        self.forgetting_events = 0


def test_ltm():
    """Test Long-Term Memory implementation"""

    print("\n" + "="*60)
    print("TESTING LONG-TERM MEMORY (LTM)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create temporary storage
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Storage: {temp_dir}")

    # Create LTM
    ltm = LongTermMemory(
        storage_path=temp_dir,
        consolidation_threshold=0.7,
        max_memories=50,
        device=device
    )

    # Test 1: Create mock STM entries and consolidate
    print("\n1. Consolidating high-salience STM entries...")

    from sage.memory.stm import STMEntry

    consolidated_count = 0
    for i in range(30):
        # Mix of high and low salience
        salience_score = 0.9 if i % 3 == 0 else 0.5

        salience = SalienceReport(
            focus_target='vision' if i % 2 == 0 else 'proprioception',
            salience_score=salience_score,
            salience_breakdown=SalienceBreakdown(
                surprise=np.random.uniform(0, 1),
                novelty=np.random.uniform(0, 1),
                arousal=np.random.uniform(0, 1),
                reward=np.random.uniform(0, 1),
                conflict=np.random.uniform(0, 1)
            ),
            suggested_stance=CognitiveStance.EXPLORATORY
        )

        entry = STMEntry(
            timestamp=time.time() + i * 10,
            cycle_id=i,
            salience_report=salience,
            sensor_snapshots={
                'vision': torch.randn(10, device=device),
                'proprioception': torch.randn(14, device=device)
            },
            outcome_success=i % 5 == 0
        )

        result = ltm.consolidate_from_stm(entry)
        if result:
            consolidated_count += 1

    print(f"   Consolidated {consolidated_count} / 30 entries (threshold: 0.7)")

    # Test 2: Retrieval by salience
    print("\n2. Retrieving most salient memories...")
    top_memories = ltm.get_most_salient(n=5)
    print(f"   Retrieved {len(top_memories)} memories")
    if top_memories:
        print(f"   Top salience: {top_memories[0].salience_score:.3f}")

    # Test 3: Retrieval by sensor
    print("\n3. Querying memories focused on 'vision'...")
    vision_memories = ltm.query_by_sensor('vision', n=5)
    print(f"   Found {len(vision_memories)} vision-focused memories")

    # Test 4: Tag-based query
    print("\n4. Querying memories by tags...")
    high_sal_memories = ltm.query_by_tags(['salience:critical'], n=5)
    print(f"   Found {len(high_sal_memories)} critical-salience memories")

    # Test 5: Time window query
    print("\n5. Querying recent time window...")
    recent_window = ltm.query_time_window(
        start_time=time.time() - 100,
        end_time=time.time() + 500,
        n=10
    )
    print(f"   Found {len(recent_window)} memories in time window")

    # Test 6: Statistics
    print("\n6. LTM Statistics...")
    stats = ltm.get_stats()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Capacity: {stats['capacity']}")
    print(f"   Utilization: {stats['utilization']:.1%}")
    print(f"   Consolidations: {stats['total_consolidations']}")
    print(f"   Retrievals: {stats['total_retrievals']}")

    # Test 7: Persistence (save and reload)
    print("\n7. Testing persistence...")
    first_memory_id = list(ltm.memories.keys())[0] if ltm.memories else None

    # Create new LTM instance (should load from disk)
    ltm2 = LongTermMemory(storage_path=temp_dir, device=device)
    print(f"   Loaded {len(ltm2.memories)} memories from disk")

    if first_memory_id:
        reloaded = ltm2.get_by_id(first_memory_id)
        print(f"   Verified memory retrieval: {reloaded is not None}")

    # Test 8: Pruning
    print("\n8. Testing memory pruning...")
    # Add more memories to trigger pruning
    for i in range(30, 60):
        salience = SalienceReport(
            focus_target='vision',
            salience_score=0.75,
            salience_breakdown=SalienceBreakdown(
                surprise=0.8, novelty=0.7, arousal=0.6, reward=0.5, conflict=0.4
            ),
            suggested_stance=CognitiveStance.CONFIDENT_EXECUTION
        )

        entry = STMEntry(
            timestamp=time.time() + i * 10,
            cycle_id=i,
            salience_report=salience,
            sensor_snapshots={'vision': torch.randn(10, device=device)}
        )

        ltm2.consolidate_from_stm(entry)

    final_stats = ltm2.get_stats()
    print(f"   Memories after adding 30 more: {final_stats['total_memories']}")
    print(f"   Forgetting events: {final_stats['forgetting_events']}")
    print(f"   (Should be <= capacity: {final_stats['capacity']})")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

    print("\n" + "="*60)
    print("✅ LTM TESTS PASSED")
    print("="*60)

    return ltm2


if __name__ == "__main__":
    test_ltm()
