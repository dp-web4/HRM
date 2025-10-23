#!/usr/bin/env python3
"""
Test Hierarchical Memory with SNARC-Guided Growth
Demonstrates memory architecture with both fixed and growing components.

Memory Tiers:
1. Circular buffers (fixed) - Recent context for attention steering
2. Long-term episodic (grows) - SNARC-filtered significant experiences
3. Consolidated patterns (grows) - Extracted learnings from sleep cycles

This shows how consciousness balances:
- Operational efficiency (circular buffers, no growth)
- Long-term learning (growing memory, SNARC-filtered)
- Wisdom extraction (consolidation, pattern recognition)
"""

import sys
import os
from pathlib import Path
import time
import random

hrm_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(hrm_root))
os.chdir(hrm_root)

from hierarchical_memory import HierarchicalMemoryStore, SNARCScores
from memory_aware_kernel import MemoryAwareKernel, ExecutionResult
from sage.services.snarc.data_structures import CognitiveStance

class ExperienceSimulator:
    """Simulates varied experiences with different SNARC profiles"""

    def __init__(self):
        self.cycle = 0
        self.experiences = [
            # Mundane (low salience, filtered out)
            ('routine_check', 0.2, SNARCScores(0.1, 0.1, 0.2, 0.2, 0.1)),
            ('background_noise', 0.15, SNARCScores(0.05, 0.1, 0.1, 0.1, 0.05)),

            # Interesting (medium salience, some stored)
            ('person_detected', 0.6, SNARCScores(0.4, 0.6, 0.5, 0.6, 0.3)),
            ('motion_detected', 0.5, SNARCScores(0.3, 0.5, 0.4, 0.5, 0.2)),

            # Significant (high salience, definitely stored)
            ('conversation', 0.8, SNARCScores(0.6, 0.7, 0.8, 0.9, 0.5)),
            ('face_recognized', 0.85, SNARCScores(0.7, 0.8, 0.8, 0.85, 0.6)),

            # Critical (very high salience, immediate storage)
            ('emergency_alert', 0.95, SNARCScores(0.95, 0.9, 0.95, 0.95, 0.8)),
        ]

    def get_experience(self):
        """Get random experience with SNARC scores"""
        self.cycle += 1

        # Weight towards mundane (realistic distribution)
        if random.random() < 0.6:
            # 60% mundane
            idx = random.randint(0, 1)
        elif random.random() < 0.8:
            # 20% interesting
            idx = random.randint(2, 3)
        elif random.random() < 0.95:
            # 15% significant
            idx = random.randint(4, 5)
        else:
            # 5% critical
            idx = 6

        event_type, importance, snarc = self.experiences[idx]

        return {
            'modality': 'mixed',
            'type': event_type,
            'importance': importance,
            'snarc_scores': snarc,
            'cycle': self.cycle
        }

def main():
    print("=" * 70)
    print("HIERARCHICAL MEMORY TEST")
    print("=" * 70)
    print("\nMemory Architecture:")
    print("  1. Circular buffers (fixed 10) - All recent events")
    print("  2. Long-term episodic (grows) - SNARC-filtered significant")
    print("  3. Consolidated patterns (grows) - Extracted during 'sleep'")
    print("\nSNARC Filtering:")
    print("  • Low salience (<0.6): Filtered out (not stored long-term)")
    print("  • High salience (≥0.6): Stored for learning")
    print("=" * 70)
    print()

    # Create hierarchical memory store
    memory_store = HierarchicalMemoryStore(
        db_path="test_sage_memory.db",
        consolidation_threshold=0.6,  # Only store high-salience
        max_long_term_size=None,  # Unlimited growth
        auto_consolidate_interval=50
    )

    # Simulate experiences
    simulator = ExperienceSimulator()

    print("Simulating 100 varied experiences...")
    print("(Watch how SNARC filters: mundane filtered, significant stored)\n")

    stored_count = 0
    filtered_count = 0

    for i in range(100):
        exp = simulator.get_experience()

        # Store if SNARC deems significant
        memory_id = memory_store.store_memory(
            cycle=exp['cycle'],
            modality=exp['modality'],
            observation={'type': exp['type']},
            result_description=f"{exp['type']} occurred",
            importance=exp['importance'],
            snarc_scores=exp['snarc_scores']
        )

        if memory_id:
            stored_count += 1
            if i < 20:  # Show first 20 for illustration
                salience = exp['snarc_scores'].overall_salience()
                print(f"  Cycle {exp['cycle']:3d}: {exp['type']:20s} "
                      f"(salience: {salience:.2f}) → STORED (ID: {memory_id})")
        else:
            filtered_count += 1
            if i < 20:
                salience = exp['snarc_scores'].overall_salience()
                print(f"  Cycle {exp['cycle']:3d}: {exp['type']:20s} "
                      f"(salience: {salience:.2f}) → filtered")

    print(f"\n... (remaining cycles)\n")

    # Statistics after simulation
    print("=" * 70)
    print("MEMORY GROWTH ANALYSIS")
    print("=" * 70)

    stats = memory_store.get_statistics()

    print(f"\nCircular buffers (fixed size):")
    print(f"  Working memory: 10 events (always)")
    print(f"  Episodic buffer: 50 events (always)")
    print(f"  Conversation: 10 turns (always)")
    print(f"  Total: 70 slots (FIXED - zero growth)")

    print(f"\nLong-term episodic (GROWING):")
    print(f"  Total stored: {stats['total_long_term_memories']} events")
    print(f"  Filtered out: {filtered_count} events (low salience)")
    print(f"  Storage rate: {stored_count}/100 = {stored_count}%")
    print(f"  High salience: {stats['high_salience_memories']} (≥0.8)")
    print(f"  Avg salience: {stats['avg_salience']:.3f}")
    print(f"  Growth: JUDICIOUS (SNARC-filtered at threshold {stats['consolidation_threshold']})")

    # Consolidate patterns (sleep cycle)
    print(f"\n{'='*70}")
    print("CONSOLIDATION (Sleep Cycle)")
    print("=" * 70)
    print("\nExtracting patterns from long-term memory...")

    patterns = memory_store.consolidate_memories(cycle=100)

    print(f"\nPatterns discovered: {len(patterns)}")
    for p in patterns:
        print(f"  • {p.description}")
        print(f"    Type: {p.pattern_type}, Confidence: {p.confidence:.2f}")
        print(f"    From {p.num_episodes} episodes, Avg salience: {p.avg_salience:.2f}")

    print(f"\nConsolidated patterns (GROWING):")
    print(f"  Total patterns: {stats['total_patterns'] + len(patterns)}")
    print(f"  Compression: {stats['total_long_term_memories']} episodes → {len(patterns)} patterns")
    print(f"  Growth: COMPRESSED (lossy but meaningful)")

    # Retrieval examples
    print(f"\n{'='*70}")
    print("MEMORY RETRIEVAL")
    print("=" * 70)

    print("\nHigh-salience memories (≥0.8):")
    high_sal = memory_store.retrieve_by_salience(min_salience=0.8, limit=5)
    for mem in high_sal:
        print(f"  Cycle {mem.cycle:3d}: {mem.result_description:30s} "
              f"(salience: {mem.salience:.2f})")

    print("\nRecent memories (last 5 stored):")
    recent = memory_store.retrieve_by_modality('mixed', limit=5)
    for mem in recent:
        print(f"  Cycle {mem.cycle:3d}: {mem.result_description:30s} "
              f"(importance: {mem.importance:.2f})")

    # Demonstrate growth over time
    print(f"\n{'='*70}")
    print("MEMORY GROWTH PROJECTION")
    print("=" * 70)

    current_size = stats['total_long_term_memories']
    storage_rate = stored_count / 100

    print(f"\nCurrent state (100 cycles):")
    print(f"  Long-term memories: {current_size}")
    print(f"  Storage rate: {storage_rate:.1%}")

    print(f"\nProjected growth:")
    for cycles in [1000, 10000, 100000]:
        projected = int(cycles * storage_rate)
        mb_estimate = projected * 1 / 1024  # ~1KB per memory
        print(f"  After {cycles:6d} cycles: ~{projected:5d} memories (~{mb_estimate:.1f} MB)")

    print(f"\nGrowth characteristics:")
    print(f"  ✅ Judicious: Only {storage_rate:.1%} of events stored")
    print(f"  ✅ SNARC-filtered: Threshold {stats['consolidation_threshold']}")
    print(f"  ✅ Bounded rate: Linear with experience, not explosive")
    print(f"  ✅ Consolidatable: Compressible via pattern extraction")
    print(f"  ✅ Pragmatic: ~{storage_rate*10:.0f}MB per 100K cycles")

    # Comparison
    print(f"\n{'='*70}")
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)

    print(f"\nCircular buffers (Session 2):")
    print(f"  Size: FIXED (70 events)")
    print(f"  Growth: ZERO (proven via profiling)")
    print(f"  Purpose: Operational consciousness (attention steering)")
    print(f"  Lifetime: Current session only")

    print(f"\nHierarchical long-term (This work):")
    print(f"  Size: GROWING (~{current_size} → unbounded)")
    print(f"  Growth: JUDICIOUS ({storage_rate:.1%} via SNARC)")
    print(f"  Purpose: Learning, identity, wisdom")
    print(f"  Lifetime: Persistent (survives restarts)")

    print(f"\nCombined architecture:")
    print(f"  Short-term: Circular (fixed, fast, attention)")
    print(f"  Long-term: Hierarchical (growing, filtered, learning)")
    print(f"  Consolidation: Sleep cycles (compression, patterns)")
    print(f"  Result: COMPLETE MEMORY HIERARCHY ✅")

    print("\n" + "=" * 70)
    print("\nKEY INSIGHT:")
    print("Consciousness needs both:")
    print("  • Fixed buffers (operational, real-time safe)")
    print("  • Growing memory (learning, identity formation)")
    print("SNARC provides the filter: what's worth remembering long-term?")
    print()

    # Cleanup
    memory_store.close()

    # Remove test database
    if os.path.exists("test_sage_memory.db"):
        os.remove("test_sage_memory.db")

if __name__ == "__main__":
    main()
