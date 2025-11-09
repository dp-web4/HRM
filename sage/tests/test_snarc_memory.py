#!/usr/bin/env python3
"""
Comprehensive Test Suite for SNARC Memory System
================================================

Tests Track 2 implementation:
- Short-Term Memory (STM)
- Long-Term Memory (LTM)
- Memory Retrieval
- STM→LTM consolidation
- Integration with SNARC

Test Scenarios:
1. Normal operation (STM + LTM)
2. Automatic consolidation
3. Memory capacity and eviction
4. Retrieval performance
5. Novelty computation
6. Persistence (disk-backed LTM)
7. Integration with SNARC salience
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import time
import tempfile
import shutil
from typing import Dict, List

# Import memory system
from sage.memory.stm import ShortTermMemory, STMEntry
from sage.memory.ltm import LongTermMemory, EpisodicMemory
from sage.memory.retrieval import MemoryRetrieval, MemorySource

# Import SNARC data structures
from sage.services.snarc.data_structures import (
    SalienceReport,
    SalienceBreakdown,
    CognitiveStance
)


def create_mock_stm_entry(
    cycle_id: int,
    salience_score: float,
    focus_target: str = 'vision',
    stance: CognitiveStance = CognitiveStance.EXPLORATORY,
    device='cpu'
) -> STMEntry:
    """Helper to create mock STM entry"""

    salience = SalienceReport(
        focus_target=focus_target,
        salience_score=salience_score,
        salience_breakdown=SalienceBreakdown(
            surprise=np.random.uniform(0, 1),
            novelty=np.random.uniform(0, 1),
            arousal=np.random.uniform(0, 1),
            reward=np.random.uniform(0, 1),
            conflict=np.random.uniform(0, 1)
        ),
        suggested_stance=stance
    )

    entry = STMEntry(
        timestamp=time.time() + cycle_id * 0.1,
        cycle_id=cycle_id,
        salience_report=salience,
        sensor_snapshots={
            'vision': torch.randn(10, device=device),
            'proprioception': torch.randn(14, device=device)
        },
        sensor_trust_scores={'vision': 0.8, 'proprioception': 0.85},
        action_taken=None,
        reward=np.random.uniform(0, 1) if cycle_id % 10 == 0 else None,
        outcome_success=cycle_id % 10 == 0
    )

    return entry


def test_stm_basic():
    """Test 1: STM basic operations"""
    print("\n" + "="*60)
    print("TEST 1: STM Basic Operations")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stm = ShortTermMemory(capacity=100, device=device)

    # Add 50 entries
    print("\nAdding 50 entries to STM...")
    for i in range(50):
        entry = create_mock_stm_entry(i, salience_score=np.random.uniform(0.3, 0.9), device=device)
        stm.add(entry)

    stats = stm.get_stats()
    assert stats['current_size'] == 50, "Should have 50 entries"
    assert stats['evictions'] == 0, "No evictions yet"
    print(f"✓ STM size: {stats['current_size']}, evictions: {stats['evictions']}")

    # Test retrieval
    recent = stm.get_recent(10)
    assert len(recent) == 10, "Should get 10 recent entries"
    assert recent[0].cycle_id > recent[-1].cycle_id, "Should be newest first"
    print(f"✓ Recent retrieval: {len(recent)} entries, newest cycle: {recent[0].cycle_id}")

    # Test high salience
    high_sal = stm.get_high_salience(threshold=0.7, max_entries=10)
    assert all(e.salience_report.salience_score >= 0.7 for e in high_sal), "All should be high salience"
    print(f"✓ High-salience retrieval: {len(high_sal)} entries (threshold: 0.7)")

    # Test novelty computation
    test_obs = torch.randn(10, device=device)
    novelty = stm.compute_novelty_score(test_obs, 'vision', lookback=20)
    assert 0.0 <= novelty <= 1.0, "Novelty should be in [0,1]"
    print(f"✓ Novelty computation: {novelty:.3f}")

    print("\n✅ TEST 1 PASSED: STM basic operations working")


def test_stm_eviction():
    """Test 2: STM eviction when capacity exceeded"""
    print("\n" + "="*60)
    print("TEST 2: STM Eviction")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stm = ShortTermMemory(capacity=50, device=device)

    # Add 100 entries (exceeds capacity)
    print("\nAdding 100 entries (capacity: 50)...")
    for i in range(100):
        entry = create_mock_stm_entry(i, salience_score=0.5, device=device)
        stm.add(entry)

    stats = stm.get_stats()
    assert stats['current_size'] == 50, f"Should be at capacity (50), got {stats['current_size']}"
    assert stats['evictions'] == 50, f"Should have 50 evictions, got {stats['evictions']}"
    assert stats['oldest_cycle'] == 50, f"Oldest should be cycle 50, got {stats['oldest_cycle']}"
    assert stats['newest_cycle'] == 99, f"Newest should be cycle 99, got {stats['newest_cycle']}"

    print(f"✓ Size: {stats['current_size']}/{stats['capacity']}")
    print(f"✓ Evictions: {stats['evictions']}")
    print(f"✓ Cycle range: {stats['oldest_cycle']} → {stats['newest_cycle']}")

    print("\n✅ TEST 2 PASSED: STM eviction working correctly")


def test_ltm_consolidation():
    """Test 3: LTM consolidation from STM"""
    print("\n" + "="*60)
    print("TEST 3: LTM Consolidation")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_dir = Path(tempfile.mkdtemp())

    ltm = LongTermMemory(
        storage_path=temp_dir,
        consolidation_threshold=0.7,
        max_memories=100,
        device=device
    )

    # Create 30 entries: 10 high-salience, 20 low-salience
    print("\nConsolidating 30 STM entries (10 high-salience, 20 low)...")
    consolidated_count = 0

    for i in range(30):
        # High salience every 3rd entry
        salience_score = 0.85 if i % 3 == 0 else 0.5

        entry = create_mock_stm_entry(i, salience_score=salience_score, device=device)

        result = ltm.consolidate_from_stm(entry)
        if result:
            consolidated_count += 1

    assert consolidated_count == 10, f"Should consolidate 10 high-salience, got {consolidated_count}"
    print(f"✓ Consolidated: {consolidated_count} / 30 entries")

    # Test retrieval
    most_salient = ltm.get_most_salient(n=5)
    assert len(most_salient) == 5, "Should get 5 memories"
    assert all(m.salience_score >= 0.7 for m in most_salient), "All should be high salience"
    print(f"✓ Most salient: {len(most_salient)} memories, top: {most_salient[0].salience_score:.3f}")

    # Test sensor query
    vision_memories = ltm.query_by_sensor('vision', n=10)
    print(f"✓ Vision-focused: {len(vision_memories)} memories")

    # Test tag query
    high_sal_memories = ltm.query_by_tags(['salience:high'], n=10)
    print(f"✓ Tag query (salience:high): {len(high_sal_memories)} memories")

    stats = ltm.get_stats()
    print(f"✓ LTM stats: {stats['total_memories']} memories, {stats['unique_tags']} tags")

    # Cleanup
    shutil.rmtree(temp_dir)

    print("\n✅ TEST 3 PASSED: LTM consolidation working")


def test_ltm_persistence():
    """Test 4: LTM persistence (save/load from disk)"""
    print("\n" + "="*60)
    print("TEST 4: LTM Persistence")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_dir = Path(tempfile.mkdtemp())

    # Create LTM and add memories
    print("\nCreating LTM with 10 memories...")
    ltm1 = LongTermMemory(storage_path=temp_dir, consolidation_threshold=0.7, device=device)

    memory_ids = []
    for i in range(10):
        entry = create_mock_stm_entry(i, salience_score=0.8, device=device)
        result = ltm1.consolidate_from_stm(entry)
        if result:
            memory_ids.append(result.memory_id)

    assert len(memory_ids) == 10, "Should create 10 memories"
    print(f"✓ Created {len(memory_ids)} memories")

    # Create new LTM instance (should load from disk)
    print("\nReloading LTM from disk...")
    ltm2 = LongTermMemory(storage_path=temp_dir, consolidation_threshold=0.7, device=device)

    assert len(ltm2.memories) == 10, f"Should load 10 memories, got {len(ltm2.memories)}"
    print(f"✓ Loaded {len(ltm2.memories)} memories from disk")

    # Verify memory retrieval
    first_memory = ltm2.get_by_id(memory_ids[0])
    assert first_memory is not None, "Should retrieve first memory"
    assert first_memory.memory_id == memory_ids[0], "Memory ID should match"
    print(f"✓ Verified memory retrieval: {first_memory.memory_id}")

    # Cleanup
    shutil.rmtree(temp_dir)

    print("\n✅ TEST 4 PASSED: LTM persistence working")


def test_retrieval_integration():
    """Test 5: Memory retrieval integration (STM + LTM)"""
    print("\n" + "="*60)
    print("TEST 5: Memory Retrieval Integration")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_dir = Path(tempfile.mkdtemp())

    # Create memory system
    stm = ShortTermMemory(capacity=100, device=device)
    ltm = LongTermMemory(storage_path=temp_dir, consolidation_threshold=0.7, device=device)
    retrieval = MemoryRetrieval(stm, ltm, consolidation_interval=20)

    # Add 50 experiences
    print("\nAdding 50 experiences...")
    for i in range(50):
        # High salience every 5th entry
        salience_score = 0.85 if i % 5 == 0 else np.random.uniform(0.4, 0.65)

        entry = create_mock_stm_entry(
            i,
            salience_score=salience_score,
            focus_target='vision' if i % 2 == 0 else 'proprioception',
            stance=CognitiveStance.EXPLORATORY if i < 25 else CognitiveStance.CONFIDENT_EXECUTION,
            device=device
        )

        retrieval.add_experience(entry)

    stats = retrieval.get_stats()
    print(f"✓ STM: {stats['stm']['current_size']} entries")
    print(f"✓ LTM: {stats['ltm']['total_memories']} memories")
    print(f"✓ Consolidations: {stats['retrieval']['total_consolidations']}")

    # Test recent context
    recent = retrieval.get_recent_context(n=10)
    assert len(recent.stm_results) == 10, "Should get 10 recent entries"
    print(f"✓ Recent context: {recent.total_results} entries ({recent.retrieval_time:.2f}ms)")

    # Test high salience (both STM and LTM)
    high_sal = retrieval.get_high_salience(threshold=0.7, max_results=20, source=MemorySource.BOTH)
    summary = high_sal.summary()
    assert summary['total_results'] > 0, "Should find high-salience memories"
    print(f"✓ High salience: {summary['total_results']} total (STM: {summary['stm_count']}, LTM: {summary['ltm_count']})")

    # Test sensor query
    vision = retrieval.query_by_sensor('vision', n=10, source=MemorySource.BOTH)
    assert vision.total_results > 0, "Should find vision-focused memories"
    print(f"✓ Vision query: {vision.total_results} results")

    # Test stance query
    exploratory = retrieval.query_by_stance(CognitiveStance.EXPLORATORY, n=10, source=MemorySource.BOTH)
    print(f"✓ Exploratory stance: {exploratory.total_results} results")

    # Test novelty computation
    test_obs = torch.randn(10, device=device)
    novelty = retrieval.compute_novelty(test_obs, 'vision', lookback_stm=20, use_ltm=True)
    assert 0.0 <= novelty <= 1.0, "Novelty should be in [0,1]"
    print(f"✓ Novelty (with LTM): {novelty:.3f}")

    # Test SNARC context
    context = retrieval.get_context_for_snarc('vision', n_recent=5, n_similar=3)
    assert len(context['recent_cycles']) > 0, "Should have recent cycles"
    print(f"✓ SNARC context: {len(context['recent_cycles'])} recent, {len(context['similar_experiences'])} similar")

    # Cleanup
    shutil.rmtree(temp_dir)

    print("\n✅ TEST 5 PASSED: Retrieval integration working")


def test_automatic_consolidation():
    """Test 6: Automatic STM→LTM consolidation"""
    print("\n" + "="*60)
    print("TEST 6: Automatic Consolidation")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_dir = Path(tempfile.mkdtemp())

    stm = ShortTermMemory(capacity=100, device=device)
    ltm = LongTermMemory(storage_path=temp_dir, consolidation_threshold=0.7, device=device)
    retrieval = MemoryRetrieval(stm, ltm, consolidation_interval=10)  # Consolidate every 10 cycles

    # Add 35 experiences (should trigger 3 consolidations at cycles 10, 20, 30)
    print("\nAdding 35 experiences (consolidation interval: 10)...")

    for i in range(35):
        # 30% high salience
        salience_score = 0.85 if i % 3 == 0 else 0.5

        entry = create_mock_stm_entry(i, salience_score=salience_score, device=device)
        retrieval.add_experience(entry)

    stats = retrieval.get_stats()
    consolidations = stats['retrieval']['total_consolidations']
    cycles_since = stats['retrieval']['cycles_since_consolidation']

    assert consolidations == 3, f"Should have 3 consolidations, got {consolidations}"
    assert cycles_since == 5, f"Should be 5 cycles since last consolidation, got {cycles_since}"

    print(f"✓ Consolidations: {consolidations} (expected 3)")
    print(f"✓ Cycles since last: {cycles_since} (expected 5)")
    print(f"✓ LTM memories: {stats['ltm']['total_memories']}")

    # Cleanup
    shutil.rmtree(temp_dir)

    print("\n✅ TEST 6 PASSED: Automatic consolidation working")


def test_performance():
    """Test 7: Performance benchmarks"""
    print("\n" + "="*60)
    print("TEST 7: Performance Benchmarks")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_dir = Path(tempfile.mkdtemp())

    stm = ShortTermMemory(capacity=1000, device=device)
    ltm = LongTermMemory(storage_path=temp_dir, consolidation_threshold=0.7, device=device)
    retrieval = MemoryRetrieval(stm, ltm, consolidation_interval=50)

    # Fill memory system
    print("\nFilling memory system (500 cycles)...")
    start = time.time()

    for i in range(500):
        salience_score = 0.85 if i % 4 == 0 else np.random.uniform(0.4, 0.65)
        entry = create_mock_stm_entry(i, salience_score=salience_score, device=device)
        retrieval.add_experience(entry)

    fill_time = time.time() - start
    print(f"✓ Fill time: {fill_time:.3f}s ({500/fill_time:.1f} cycles/sec)")

    # Benchmark retrieval operations
    print("\nBenchmarking retrieval operations (100 queries each)...")

    # Recent context
    start = time.time()
    for _ in range(100):
        retrieval.get_recent_context(n=10)
    recent_time = (time.time() - start) * 1000 / 100  # ms per query
    print(f"✓ Recent context: {recent_time:.2f}ms/query")

    # High salience
    start = time.time()
    for _ in range(100):
        retrieval.get_high_salience(threshold=0.7, max_results=20, source=MemorySource.BOTH)
    high_sal_time = (time.time() - start) * 1000 / 100
    print(f"✓ High salience: {high_sal_time:.2f}ms/query")

    # Sensor query
    start = time.time()
    for _ in range(100):
        retrieval.query_by_sensor('vision', n=10, source=MemorySource.BOTH)
    sensor_time = (time.time() - start) * 1000 / 100
    print(f"✓ Sensor query: {sensor_time:.2f}ms/query")

    # Novelty computation
    test_obs = torch.randn(10, device=device)
    start = time.time()
    for _ in range(100):
        retrieval.compute_novelty(test_obs, 'vision', lookback_stm=50, use_ltm=True)
    novelty_time = (time.time() - start) * 1000 / 100
    print(f"✓ Novelty compute: {novelty_time:.2f}ms/query")

    # Verify performance targets for Nano
    assert recent_time < 10.0, f"Recent context should be <10ms, got {recent_time:.2f}ms"
    assert high_sal_time < 20.0, f"High salience should be <20ms, got {high_sal_time:.2f}ms"
    assert novelty_time < 50.0, f"Novelty should be <50ms, got {novelty_time:.2f}ms"

    print(f"\n✓ All queries meet real-time targets (<100ms)")

    # Cleanup
    shutil.rmtree(temp_dir)

    print("\n✅ TEST 7 PASSED: Performance benchmarks met")


def run_all_tests():
    """Run all test scenarios"""
    print("\n" + "#"*60)
    print("# SNARC MEMORY SYSTEM - COMPREHENSIVE TEST SUITE")
    print("# Track 2: Jetson Nano Deployment Roadmap")
    print("#"*60)

    start_time = time.time()

    try:
        test_stm_basic()
        test_stm_eviction()
        test_ltm_consolidation()
        test_ltm_persistence()
        test_retrieval_integration()
        test_automatic_consolidation()
        test_performance()

        elapsed = time.time() - start_time

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✅")
        print("="*60)
        print(f"\nElapsed time: {elapsed:.2f}s")
        print("\nTrack 2 (SNARC Memory) implementation validated:")
        print("  ✓ Short-term memory (STM) working")
        print("  ✓ Long-term memory (LTM) working")
        print("  ✓ Memory retrieval working")
        print("  ✓ Automatic consolidation working")
        print("  ✓ Persistence (disk-backed) working")
        print("  ✓ Performance targets met (<100ms queries)")
        print("  ✓ Novelty computation working")
        print("  ✓ SNARC integration ready")
        print("\nSystem ready for integration with SNARC and autonomous exploration.")
        print("="*60)

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
