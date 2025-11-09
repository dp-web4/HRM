#!/usr/bin/env python3
"""
Memory Retrieval System for SAGE SNARC
=======================================

Unified interface for querying both Short-Term Memory (STM) and Long-Term Memory (LTM).
Provides context-aware retrieval for SNARC decision-making.

Track 2: SNARC Memory - Component 3/3
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time

# Import memory components
try:
    from sage.memory.stm import ShortTermMemory, STMEntry
    from sage.memory.ltm import LongTermMemory, EpisodicMemory
    from sage.services.snarc.data_structures import CognitiveStance
except ModuleNotFoundError:
    # Fallback for standalone testing
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from sage.memory.stm import ShortTermMemory, STMEntry
    from sage.memory.ltm import LongTermMemory, EpisodicMemory
    from sage.services.snarc.data_structures import CognitiveStance


class MemorySource(Enum):
    """Where a memory came from"""
    STM = "short_term"
    LTM = "long_term"
    BOTH = "both"


@dataclass
class RetrievalResult:
    """
    Result of a memory query

    Contains memories from STM and/or LTM with metadata about retrieval.
    """
    query_type: str  # Type of query performed
    stm_results: List[STMEntry] = field(default_factory=list)
    ltm_results: List[EpisodicMemory] = field(default_factory=list)
    retrieval_time: float = 0.0  # Milliseconds
    total_results: int = 0

    def __post_init__(self):
        """Compute total results"""
        self.total_results = len(self.stm_results) + len(self.ltm_results)

    def all_memories(self) -> List[Union[STMEntry, EpisodicMemory]]:
        """Get all memories (STM + LTM) in single list"""
        return list(self.stm_results) + list(self.ltm_results)

    def get_salience_scores(self) -> List[float]:
        """Get salience scores from all memories"""
        scores = []

        for entry in self.stm_results:
            scores.append(entry.salience_report.salience_score)

        for memory in self.ltm_results:
            scores.append(memory.salience_score)

        return scores

    def summary(self) -> Dict[str, Any]:
        """Get summary of retrieval results"""
        return {
            'query_type': self.query_type,
            'total_results': self.total_results,
            'stm_count': len(self.stm_results),
            'ltm_count': len(self.ltm_results),
            'retrieval_time_ms': self.retrieval_time,
            'avg_salience': sum(self.get_salience_scores()) / self.total_results if self.total_results > 0 else 0.0
        }


class MemoryRetrieval:
    """
    Unified memory retrieval interface for SNARC

    Provides intelligent queries across both STM and LTM:
    - Recent context (STM-focused)
    - Similar experiences (LTM-focused)
    - High-salience events (both)
    - Sensor-specific history (both)
    - Time-windowed queries (both)

    Handles:
    - STM → LTM consolidation scheduling
    - Intelligent query routing
    - Result aggregation and ranking
    """

    def __init__(
        self,
        stm: ShortTermMemory,
        ltm: LongTermMemory,
        consolidation_interval: int = 100,  # Consolidate every N cycles
        device: Optional[torch.device] = None
    ):
        """
        Initialize Memory Retrieval

        Args:
            stm: Short-term memory instance
            ltm: Long-term memory instance
            consolidation_interval: How often to consolidate STM→LTM (cycles)
            device: Device for tensor operations
        """
        self.stm = stm
        self.ltm = ltm
        self.consolidation_interval = consolidation_interval
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Track consolidation
        self.cycles_since_consolidation = 0
        self.total_consolidations = 0

    def add_experience(self, stm_entry: STMEntry):
        """
        Add new experience to memory system

        Automatically handles:
        - Adding to STM
        - Triggering consolidation when needed

        Args:
            stm_entry: New STM entry to store
        """
        # Add to STM
        self.stm.add(stm_entry)

        # Check if consolidation needed
        self.cycles_since_consolidation += 1

        if self.cycles_since_consolidation >= self.consolidation_interval:
            self.consolidate()

    def consolidate(self):
        """
        Consolidate STM → LTM

        Transfers high-salience STM entries to LTM for long-term retention.
        """
        # Get high-salience entries from STM
        high_salience = self.stm.get_high_salience(
            threshold=self.ltm.consolidation_threshold
        )

        # Consolidate each to LTM
        consolidated_count = 0
        for entry in high_salience:
            result = self.ltm.consolidate_from_stm(entry)
            if result:
                consolidated_count += 1

        # Reset consolidation timer
        self.cycles_since_consolidation = 0
        self.total_consolidations += 1

        return consolidated_count

    def get_recent_context(self, n: int = 10) -> RetrievalResult:
        """
        Get recent context for SNARC decision-making

        Queries STM for most recent experiences.

        Args:
            n: Number of recent entries

        Returns:
            RetrievalResult with recent STM entries
        """
        start_time = time.time()

        stm_results = self.stm.get_recent(n)

        elapsed = (time.time() - start_time) * 1000  # Convert to ms

        return RetrievalResult(
            query_type="recent_context",
            stm_results=stm_results,
            retrieval_time=elapsed
        )

    def get_high_salience(
        self,
        threshold: float = 0.7,
        max_results: int = 20,
        source: MemorySource = MemorySource.BOTH
    ) -> RetrievalResult:
        """
        Get high-salience memories

        Args:
            threshold: Minimum salience score
            max_results: Maximum total results
            source: Query STM, LTM, or both

        Returns:
            RetrievalResult with high-salience memories
        """
        start_time = time.time()

        stm_results = []
        ltm_results = []

        if source in (MemorySource.STM, MemorySource.BOTH):
            stm_results = self.stm.get_high_salience(threshold, max_results)

        if source in (MemorySource.LTM, MemorySource.BOTH):
            ltm_results = self.ltm.get_most_salient(max_results)
            # Filter by threshold
            ltm_results = [m for m in ltm_results if m.salience_score >= threshold]

        # Limit total results
        if source == MemorySource.BOTH:
            # Interleave STM and LTM results by salience
            all_with_salience = (
                [(e.salience_report.salience_score, 'stm', e) for e in stm_results] +
                [(m.salience_score, 'ltm', m) for m in ltm_results]
            )
            all_with_salience.sort(reverse=True, key=lambda x: x[0])

            stm_results = [item for _, src, item in all_with_salience if src == 'stm'][:max_results]
            ltm_results = [item for _, src, item in all_with_salience if src == 'ltm'][:max_results]

        elapsed = (time.time() - start_time) * 1000

        return RetrievalResult(
            query_type="high_salience",
            stm_results=stm_results,
            ltm_results=ltm_results,
            retrieval_time=elapsed
        )

    def query_by_sensor(
        self,
        sensor_id: str,
        n: int = 10,
        source: MemorySource = MemorySource.BOTH
    ) -> RetrievalResult:
        """
        Query memories focused on specific sensor

        Args:
            sensor_id: Sensor to query
            n: Maximum results per source
            source: Query STM, LTM, or both

        Returns:
            RetrievalResult with sensor-specific memories
        """
        start_time = time.time()

        stm_results = []
        ltm_results = []

        if source in (MemorySource.STM, MemorySource.BOTH):
            stm_results = self.stm.query_by_sensor(sensor_id, n)

        if source in (MemorySource.LTM, MemorySource.BOTH):
            ltm_results = self.ltm.query_by_sensor(sensor_id, n)

        elapsed = (time.time() - start_time) * 1000

        return RetrievalResult(
            query_type=f"sensor:{sensor_id}",
            stm_results=stm_results,
            ltm_results=ltm_results,
            retrieval_time=elapsed
        )

    def query_by_stance(
        self,
        stance: CognitiveStance,
        n: int = 10,
        source: MemorySource = MemorySource.BOTH
    ) -> RetrievalResult:
        """
        Query memories with specific cognitive stance

        Args:
            stance: Cognitive stance to query
            n: Maximum results per source
            source: Query STM, LTM, or both

        Returns:
            RetrievalResult with stance-specific memories
        """
        start_time = time.time()

        stm_results = []
        ltm_results = []

        if source in (MemorySource.STM, MemorySource.BOTH):
            stm_results = self.stm.query_by_stance(stance, n)

        if source in (MemorySource.LTM, MemorySource.BOTH):
            # Query LTM by tags
            tag = f"stance:{stance.value}"
            ltm_results = self.ltm.query_by_tags([tag], n=n)

        elapsed = (time.time() - start_time) * 1000

        return RetrievalResult(
            query_type=f"stance:{stance.value}",
            stm_results=stm_results,
            ltm_results=ltm_results,
            retrieval_time=elapsed
        )

    def query_time_window(
        self,
        start_time: float,
        end_time: float,
        max_results: int = 50,
        source: MemorySource = MemorySource.BOTH
    ) -> RetrievalResult:
        """
        Query memories within time window

        Args:
            start_time: Start timestamp
            end_time: End timestamp
            max_results: Maximum results
            source: Query STM, LTM, or both

        Returns:
            RetrievalResult with time-windowed memories
        """
        start = time.time()

        stm_results = []
        ltm_results = []

        if source in (MemorySource.STM, MemorySource.BOTH):
            stm_results = self.stm.get_time_window(start_time, end_time)

        if source in (MemorySource.LTM, MemorySource.BOTH):
            ltm_results = self.ltm.query_time_window(start_time, end_time)

        # Limit results
        if len(stm_results) + len(ltm_results) > max_results:
            # Sort all by timestamp, take most recent
            all_with_time = (
                [(e.timestamp, 'stm', e) for e in stm_results] +
                [(m.timestamp, 'ltm', m) for m in ltm_results]
            )
            all_with_time.sort(reverse=True, key=lambda x: x[0])

            stm_results = [item for _, src, item in all_with_time if src == 'stm'][:max_results]
            ltm_results = [item for _, src, item in all_with_time if src == 'ltm'][:max_results]

        elapsed = (time.time() - start) * 1000

        return RetrievalResult(
            query_type="time_window",
            stm_results=stm_results,
            ltm_results=ltm_results,
            retrieval_time=elapsed
        )

    def compute_novelty(
        self,
        current_observation: Any,
        sensor_id: str,
        lookback_stm: int = 100,
        use_ltm: bool = True
    ) -> float:
        """
        Compute novelty of current observation

        Compares against recent STM and (optionally) relevant LTM.

        Args:
            current_observation: Current sensor data
            sensor_id: Which sensor
            lookback_stm: How many STM cycles to compare
            use_ltm: Whether to include LTM in novelty computation

        Returns:
            Novelty score (0.0-1.0, higher = more novel)
        """
        # Get STM novelty
        stm_novelty = self.stm.compute_novelty_score(
            current_observation,
            sensor_id,
            lookback_stm
        )

        if not use_ltm:
            return stm_novelty

        # Get relevant LTM memories
        ltm_memories = self.ltm.query_by_sensor(sensor_id, n=20)

        if not ltm_memories:
            return stm_novelty

        # Compute novelty vs LTM (using compressed sensor summaries)
        # This is approximate since LTM doesn't store full tensors

        if not isinstance(current_observation, torch.Tensor):
            return stm_novelty

        current_stats = {
            'mean': float(current_observation.mean().item()),
            'std': float(current_observation.std().item()),
            'norm': float(torch.norm(current_observation).item())
        }

        # Compare to LTM sensor summaries
        ltm_distances = []
        for memory in ltm_memories:
            if sensor_id in memory.sensor_summary:
                summary = memory.sensor_summary[sensor_id]
                if summary.get('type') == 'tensor':
                    # Compute distance in summary space
                    distance = abs(current_stats['mean'] - summary['mean']) / (summary['std'] + 1e-6)
                    distance += abs(current_stats['norm'] - summary['norm']) / (summary['norm'] + 1e-6)
                    ltm_distances.append(distance / 2.0)  # Normalize

        if ltm_distances:
            ltm_novelty = min(min(ltm_distances), 1.0)
        else:
            ltm_novelty = stm_novelty

        # Combine STM and LTM novelty (weighted average)
        # STM is more reliable (full data), LTM provides long-term context
        combined_novelty = 0.7 * stm_novelty + 0.3 * ltm_novelty

        return combined_novelty

    def get_context_for_snarc(
        self,
        current_sensor_id: str,
        n_recent: int = 10,
        n_similar: int = 5
    ) -> Dict[str, Any]:
        """
        Get comprehensive context for SNARC assessment

        Provides:
        - Recent context (last N cycles)
        - Similar past experiences (same sensor)
        - Summary statistics

        Args:
            current_sensor_id: Current focus sensor
            n_recent: Number of recent cycles
            n_similar: Number of similar past experiences

        Returns:
            Dictionary with context for SNARC
        """
        # Recent context
        recent = self.get_recent_context(n_recent)

        # Similar experiences from LTM
        similar_ltm = self.ltm.query_by_sensor(current_sensor_id, n_similar)

        # Summary statistics
        recent_summary = self.stm.get_context_summary(n_recent)

        return {
            'recent_cycles': recent.stm_results,
            'recent_summary': recent_summary,
            'similar_experiences': similar_ltm,
            'ltm_available': len(similar_ltm) > 0,
            'stm_size': len(self.stm.buffer),
            'ltm_size': len(self.ltm.memories)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            'stm': self.stm.get_stats(),
            'ltm': self.ltm.get_stats(),
            'retrieval': {
                'consolidation_interval': self.consolidation_interval,
                'cycles_since_consolidation': self.cycles_since_consolidation,
                'total_consolidations': self.total_consolidations
            }
        }


def test_retrieval():
    """Test Memory Retrieval system"""

    print("\n" + "="*60)
    print("TESTING MEMORY RETRIEVAL")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create memory system
    from sage.services.snarc.data_structures import SalienceReport, SalienceBreakdown
    import tempfile
    from pathlib import Path

    temp_dir = Path(tempfile.mkdtemp())
    print(f"Storage: {temp_dir}")

    stm = ShortTermMemory(capacity=100, device=device)
    ltm = LongTermMemory(storage_path=temp_dir, consolidation_threshold=0.7, device=device)
    retrieval = MemoryRetrieval(stm, ltm, consolidation_interval=20)

    # Test 1: Add experiences
    print("\n1. Adding 50 experiences...")
    for i in range(50):
        salience_score = 0.9 if i % 5 == 0 else np.random.uniform(0.4, 0.7)

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
            suggested_stance=CognitiveStance.EXPLORATORY if i < 25 else CognitiveStance.CONFIDENT_EXECUTION
        )

        entry = STMEntry(
            timestamp=time.time() + i * 0.1,
            cycle_id=i,
            salience_report=salience,
            sensor_snapshots={
                'vision': torch.randn(10, device=device),
                'proprioception': torch.randn(14, device=device)
            }
        )

        retrieval.add_experience(entry)

    stats = retrieval.get_stats()
    print(f"   STM: {stats['stm']['current_size']} entries")
    print(f"   LTM: {stats['ltm']['total_memories']} memories")
    print(f"   Consolidations: {stats['retrieval']['total_consolidations']}")

    # Test 2: Recent context
    print("\n2. Retrieving recent context...")
    recent = retrieval.get_recent_context(n=10)
    print(f"   Retrieved: {recent.total_results} entries")
    print(f"   Time: {recent.retrieval_time:.2f}ms")

    # Test 3: High salience
    print("\n3. Querying high-salience memories...")
    high_sal = retrieval.get_high_salience(threshold=0.7, max_results=10, source=MemorySource.BOTH)
    summary = high_sal.summary()
    print(f"   Total: {summary['total_results']} (STM: {summary['stm_count']}, LTM: {summary['ltm_count']})")
    print(f"   Avg salience: {summary['avg_salience']:.3f}")
    print(f"   Time: {summary['retrieval_time_ms']:.2f}ms")

    # Test 4: Sensor query
    print("\n4. Querying vision-focused memories...")
    vision = retrieval.query_by_sensor('vision', n=10, source=MemorySource.BOTH)
    print(f"   Total: {vision.total_results} (STM: {len(vision.stm_results)}, LTM: {len(vision.ltm_results)})")

    # Test 5: Stance query
    print("\n5. Querying EXPLORATORY stance...")
    exploratory = retrieval.query_by_stance(CognitiveStance.EXPLORATORY, n=10, source=MemorySource.BOTH)
    print(f"   Total: {exploratory.total_results}")

    # Test 6: Novelty computation
    print("\n6. Computing novelty score...")
    test_obs = torch.randn(10, device=device)
    novelty = retrieval.compute_novelty(test_obs, 'vision', lookback_stm=20, use_ltm=True)
    print(f"   Novelty (with LTM): {novelty:.3f}")

    novelty_stm_only = retrieval.compute_novelty(test_obs, 'vision', lookback_stm=20, use_ltm=False)
    print(f"   Novelty (STM only): {novelty_stm_only:.3f}")

    # Test 7: SNARC context
    print("\n7. Getting context for SNARC...")
    context = retrieval.get_context_for_snarc('vision', n_recent=5, n_similar=3)
    print(f"   Recent cycles: {len(context['recent_cycles'])}")
    print(f"   Similar experiences: {len(context['similar_experiences'])}")
    print(f"   Recent summary: {context['recent_summary']['avg_salience']:.3f} avg salience")

    # Test 8: Time window
    print("\n8. Querying time window...")
    now = time.time()
    window = retrieval.query_time_window(now - 5, now + 10, max_results=20, source=MemorySource.BOTH)
    print(f"   Found {window.total_results} memories in time window")

    # Test 9: Manual consolidation
    print("\n9. Manual consolidation trigger...")
    before_ltm = len(ltm.memories)
    consolidated = retrieval.consolidate()
    after_ltm = len(ltm.memories)
    print(f"   Consolidated {consolidated} entries")
    print(f"   LTM before: {before_ltm}, after: {after_ltm}")

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

    print("\n" + "="*60)
    print("✅ RETRIEVAL TESTS PASSED")
    print("="*60)

    return retrieval


if __name__ == "__main__":
    test_retrieval()
