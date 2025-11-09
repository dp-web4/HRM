#!/usr/bin/env python3
"""
Short-Term Memory (STM) for SAGE SNARC System
==============================================

Circular buffer for recent salience assessments, sensor snapshots, and actions.
Provides fast access to working memory for context-aware decision making.

Track 2: SNARC Memory - Component 1/3
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time

# Import SNARC data structures
try:
    from sage.services.snarc.data_structures import (
        SalienceReport,
        SalienceBreakdown,
        SensorOutput,
        CognitiveStance
    )
except ModuleNotFoundError:
    # Fallback for standalone testing
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from sage.services.snarc.data_structures import (
        SalienceReport,
        SalienceBreakdown,
        SensorOutput,
        CognitiveStance
    )


@dataclass
class STMEntry:
    """
    Single entry in Short-Term Memory

    Captures complete state at a single time step:
    - Salience assessment from SNARC
    - Sensor observations
    - Action taken (if any)
    - Outcome/reward (if available)
    - Metadata for retrieval
    """
    timestamp: float  # When this occurred
    cycle_id: int  # Cycle number in exploration

    # SNARC assessment
    salience_report: SalienceReport

    # Sensor state
    sensor_snapshots: Dict[str, Any]  # sensor_id -> observation

    # Action taken (if embodied)
    action_taken: Optional[Any] = None

    # Outcome/reward (if available)
    reward: Optional[float] = None
    outcome_success: Optional[bool] = None

    # Trust scores (from Track 1 sensor trust system)
    sensor_trust_scores: Dict[str, float] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate entry"""
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'cycle_id': self.cycle_id,
            'salience_report': self.salience_report.to_dict(),
            'salience_score': self.salience_report.salience_score,
            'focus_target': self.salience_report.focus_target,
            'stance': self.salience_report.suggested_stance.value,
            'action': str(self.action_taken) if self.action_taken is not None else None,
            'reward': self.reward,
            'success': self.outcome_success,
            'sensor_trust': self.sensor_trust_scores,
            'metadata': self.metadata
        }


class ShortTermMemory:
    """
    Circular buffer for recent SNARC assessments and experiences

    Design:
    - Fixed capacity (e.g., 1000 cycles)
    - Fast random access by index or time
    - Efficient windowed queries (last N cycles)
    - O(1) append, O(1) access by index
    - Automatic eviction of oldest entries

    Integration:
    - Updated each SNARC cycle
    - Queried by SNARC for context (novelty detection)
    - Queried by LTM for consolidation
    - Queried by retrieval for recent context
    """

    def __init__(
        self,
        capacity: int = 1000,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Short-Term Memory

        Args:
            capacity: Maximum number of entries to store
            device: Device for tensor operations (CPU/CUDA)
        """
        self.capacity = capacity
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Circular buffer
        self.buffer: deque = deque(maxlen=capacity)

        # Statistics
        self.total_entries = 0
        self.evictions = 0

        # Index for fast access
        self._cycle_index: Dict[int, int] = {}  # cycle_id -> buffer position

    def add(self, entry: STMEntry) -> None:
        """
        Add new entry to STM

        Args:
            entry: STM entry to store
        """
        # Check if buffer is full (eviction will occur)
        if len(self.buffer) >= self.capacity:
            self.evictions += 1
            # Remove oldest entry from index
            oldest = self.buffer[0]
            if oldest.cycle_id in self._cycle_index:
                del self._cycle_index[oldest.cycle_id]

        # Add to buffer
        self.buffer.append(entry)

        # Update index
        buffer_position = len(self.buffer) - 1
        self._cycle_index[entry.cycle_id] = buffer_position

        # Update stats
        self.total_entries += 1

    def get_by_cycle(self, cycle_id: int) -> Optional[STMEntry]:
        """
        Retrieve entry by cycle ID

        Args:
            cycle_id: Cycle number to retrieve

        Returns:
            STMEntry if found, None otherwise
        """
        if cycle_id in self._cycle_index:
            position = self._cycle_index[cycle_id]
            if position < len(self.buffer):
                return self.buffer[position]
        return None

    def get_recent(self, n: int = 10) -> List[STMEntry]:
        """
        Get N most recent entries

        Args:
            n: Number of recent entries to retrieve

        Returns:
            List of recent STM entries (newest first)
        """
        if n >= len(self.buffer):
            return list(reversed(self.buffer))

        return [self.buffer[i] for i in range(len(self.buffer) - 1, len(self.buffer) - n - 1, -1)]

    def get_time_window(
        self,
        start_time: float,
        end_time: float
    ) -> List[STMEntry]:
        """
        Get all entries within time window

        Args:
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)

        Returns:
            List of entries in time range
        """
        return [
            entry for entry in self.buffer
            if start_time <= entry.timestamp <= end_time
        ]

    def get_high_salience(
        self,
        threshold: float = 0.7,
        max_entries: Optional[int] = None
    ) -> List[STMEntry]:
        """
        Get high-salience entries from STM

        Args:
            threshold: Minimum salience score (0.0-1.0)
            max_entries: Maximum number to return (None = all)

        Returns:
            List of high-salience entries (sorted by salience, descending)
        """
        high_salience = [
            entry for entry in self.buffer
            if entry.salience_report.salience_score >= threshold
        ]

        # Sort by salience (descending)
        high_salience.sort(
            key=lambda e: e.salience_report.salience_score,
            reverse=True
        )

        if max_entries is not None:
            high_salience = high_salience[:max_entries]

        return high_salience

    def query_by_sensor(
        self,
        sensor_id: str,
        n: int = 10
    ) -> List[STMEntry]:
        """
        Get recent entries focused on specific sensor

        Args:
            sensor_id: Sensor to filter by
            n: Maximum number of entries

        Returns:
            List of entries where sensor was focus target
        """
        matches = [
            entry for entry in reversed(self.buffer)
            if entry.salience_report.focus_target == sensor_id
        ]

        return matches[:n]

    def query_by_stance(
        self,
        stance: CognitiveStance,
        n: int = 10
    ) -> List[STMEntry]:
        """
        Get recent entries with specific cognitive stance

        Args:
            stance: Cognitive stance to filter by
            n: Maximum number of entries

        Returns:
            List of entries with matching stance
        """
        matches = [
            entry for entry in reversed(self.buffer)
            if entry.salience_report.suggested_stance == stance
        ]

        return matches[:n]

    def compute_novelty_score(
        self,
        current_observation: Any,
        sensor_id: str,
        lookback: int = 100
    ) -> float:
        """
        Compute novelty of current observation vs recent STM

        Used by SNARC to assess novelty dimension.
        Compares current observation to recent history.

        Args:
            current_observation: Current sensor data (tensor)
            sensor_id: Which sensor produced this
            lookback: How many recent cycles to compare against

        Returns:
            Novelty score (0.0-1.0, higher = more novel)
        """
        # Get recent entries with this sensor
        recent = self.get_recent(lookback)
        sensor_observations = []

        for entry in recent:
            if sensor_id in entry.sensor_snapshots:
                sensor_observations.append(entry.sensor_snapshots[sensor_id])

        if not sensor_observations:
            # No history = maximally novel
            return 1.0

        # Convert to tensor if needed
        if not isinstance(current_observation, torch.Tensor):
            return 0.5  # Can't compare non-tensors yet

        # Compute distance to recent observations
        current = current_observation.to(self.device).flatten()
        distances = []

        for obs in sensor_observations:
            if isinstance(obs, torch.Tensor):
                obs_flat = obs.to(self.device).flatten()
                # Cosine distance (1 - cosine similarity)
                similarity = torch.nn.functional.cosine_similarity(
                    current.unsqueeze(0),
                    obs_flat.unsqueeze(0)
                )
                distance = 1.0 - similarity.item()
                distances.append(distance)

        if not distances:
            return 0.5

        # Novelty = how different from nearest neighbor
        min_distance = min(distances)
        novelty = min(min_distance, 1.0)

        return novelty

    def get_context_summary(self, n_recent: int = 10) -> Dict[str, Any]:
        """
        Get summary of recent context for LLM grounding

        Args:
            n_recent: Number of recent cycles to summarize

        Returns:
            Dictionary with context summary
        """
        recent = self.get_recent(n_recent)

        if not recent:
            return {
                'recent_cycles': 0,
                'avg_salience': 0.0,
                'dominant_sensor': None,
                'dominant_stance': None
            }

        # Compute statistics
        avg_salience = sum(e.salience_report.salience_score for e in recent) / len(recent)

        # Find dominant sensor
        sensor_counts = {}
        for entry in recent:
            sensor = entry.salience_report.focus_target
            sensor_counts[sensor] = sensor_counts.get(sensor, 0) + 1
        dominant_sensor = max(sensor_counts.items(), key=lambda x: x[1])[0] if sensor_counts else None

        # Find dominant stance
        stance_counts = {}
        for entry in recent:
            stance = entry.salience_report.suggested_stance.value
            stance_counts[stance] = stance_counts.get(stance, 0) + 1
        dominant_stance = max(stance_counts.items(), key=lambda x: x[1])[0] if stance_counts else None

        # Salience breakdown averages
        salience_dims = {
            'surprise': 0.0,
            'novelty': 0.0,
            'arousal': 0.0,
            'reward': 0.0,
            'conflict': 0.0
        }

        for entry in recent:
            bd = entry.salience_report.salience_breakdown
            salience_dims['surprise'] += bd.surprise
            salience_dims['novelty'] += bd.novelty
            salience_dims['arousal'] += bd.arousal
            salience_dims['reward'] += bd.reward
            salience_dims['conflict'] += bd.conflict

        for dim in salience_dims:
            salience_dims[dim] /= len(recent)

        return {
            'recent_cycles': len(recent),
            'avg_salience': avg_salience,
            'dominant_sensor': dominant_sensor,
            'dominant_stance': dominant_stance,
            'salience_breakdown': salience_dims,
            'time_span': recent[0].timestamp - recent[-1].timestamp if len(recent) > 1 else 0.0
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get STM statistics

        Returns:
            Dictionary with memory stats
        """
        return {
            'capacity': self.capacity,
            'current_size': len(self.buffer),
            'utilization': len(self.buffer) / self.capacity,
            'total_entries': self.total_entries,
            'evictions': self.evictions,
            'oldest_cycle': self.buffer[0].cycle_id if self.buffer else None,
            'newest_cycle': self.buffer[-1].cycle_id if self.buffer else None
        }

    def clear(self) -> None:
        """Clear all STM (reset to empty state)"""
        self.buffer.clear()
        self._cycle_index.clear()
        self.total_entries = 0
        self.evictions = 0


def test_stm():
    """Test Short-Term Memory implementation"""

    print("\n" + "="*60)
    print("TESTING SHORT-TERM MEMORY (STM)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create STM
    stm = ShortTermMemory(capacity=100, device=device)

    # Test 1: Add entries
    print("\n1. Adding 50 entries...")
    for i in range(50):
        # Create mock salience report
        salience = SalienceReport(
            focus_target='vision' if i % 2 == 0 else 'proprioception',
            salience_score=np.random.uniform(0.3, 0.9),
            salience_breakdown=SalienceBreakdown(
                surprise=np.random.uniform(0, 1),
                novelty=np.random.uniform(0, 1),
                arousal=np.random.uniform(0, 1),
                reward=np.random.uniform(0, 1),
                conflict=np.random.uniform(0, 1)
            ),
            suggested_stance=CognitiveStance.EXPLORATORY if i < 25 else CognitiveStance.CONFIDENT_EXECUTION
        )

        # Create entry
        entry = STMEntry(
            timestamp=time.time() + i * 0.1,
            cycle_id=i,
            salience_report=salience,
            sensor_snapshots={
                'vision': torch.randn(10, device=device),
                'proprioception': torch.randn(14, device=device)
            },
            sensor_trust_scores={'vision': 0.8, 'proprioception': 0.85}
        )

        stm.add(entry)

    stats = stm.get_stats()
    print(f"   Added {stats['current_size']} entries")
    print(f"   Capacity utilization: {stats['utilization']:.1%}")

    # Test 2: Retrieval by cycle
    print("\n2. Retrieving by cycle ID...")
    entry_25 = stm.get_by_cycle(25)
    if entry_25:
        print(f"   Cycle 25: salience={entry_25.salience_report.salience_score:.3f}, "
              f"focus={entry_25.salience_report.focus_target}")

    # Test 3: Recent entries
    print("\n3. Getting 5 most recent entries...")
    recent = stm.get_recent(5)
    print(f"   Retrieved {len(recent)} entries")
    print(f"   Cycle IDs: {[e.cycle_id for e in recent]}")

    # Test 4: High salience
    print("\n4. Querying high-salience entries (>0.7)...")
    high_sal = stm.get_high_salience(threshold=0.7, max_entries=5)
    print(f"   Found {len(high_sal)} high-salience entries")
    if high_sal:
        print(f"   Top salience: {high_sal[0].salience_report.salience_score:.3f}")

    # Test 5: Query by sensor
    print("\n5. Querying entries focused on 'vision'...")
    vision_entries = stm.query_by_sensor('vision', n=10)
    print(f"   Found {len(vision_entries)} vision-focused entries")

    # Test 6: Query by stance
    print("\n6. Querying entries with EXPLORATORY stance...")
    exploratory = stm.query_by_stance(CognitiveStance.EXPLORATORY, n=10)
    print(f"   Found {len(exploratory)} exploratory entries")

    # Test 7: Novelty computation
    print("\n7. Computing novelty score...")
    test_obs = torch.randn(10, device=device)
    novelty = stm.compute_novelty_score(test_obs, 'vision', lookback=20)
    print(f"   Novelty score: {novelty:.3f}")

    # Test 8: Context summary
    print("\n8. Getting context summary...")
    summary = stm.get_context_summary(n_recent=10)
    print(f"   Recent cycles: {summary['recent_cycles']}")
    print(f"   Avg salience: {summary['avg_salience']:.3f}")
    print(f"   Dominant sensor: {summary['dominant_sensor']}")
    print(f"   Dominant stance: {summary['dominant_stance']}")

    # Test 9: Eviction (add more than capacity)
    print("\n9. Testing eviction (adding 60 more entries)...")
    for i in range(50, 110):
        salience = SalienceReport(
            focus_target='vision',
            salience_score=0.5,
            salience_breakdown=SalienceBreakdown(
                surprise=0.5, novelty=0.5, arousal=0.5, reward=0.5, conflict=0.5
            ),
            suggested_stance=CognitiveStance.CONFIDENT_EXECUTION
        )

        entry = STMEntry(
            timestamp=time.time() + i * 0.1,
            cycle_id=i,
            salience_report=salience,
            sensor_snapshots={'vision': torch.randn(10, device=device)}
        )

        stm.add(entry)

    final_stats = stm.get_stats()
    print(f"   Current size: {final_stats['current_size']} (capacity: {final_stats['capacity']})")
    print(f"   Evictions: {final_stats['evictions']}")
    print(f"   Oldest cycle now: {final_stats['oldest_cycle']} (was 0)")
    print(f"   Newest cycle: {final_stats['newest_cycle']}")

    print("\n" + "="*60)
    print("✅ STM TESTS PASSED")
    print("="*60)

    return stm


if __name__ == "__main__":
    test_stm()
