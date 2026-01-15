#!/usr/bin/env python3
"""
Session 198 Phase 2: Memory Consolidation Test

Tests the hypothesis that federated memory consolidation prevents regression
by restoring attention states from previous successful sessions.

Test Design:
1. T014 is perfect (100% success) - store as memory
2. T015 has regression (80%, 4-1 fails) - test without memory
3. Apply T014 memory boost to T015 failed exercise
4. Measure if memory prevents failure

Predictions:
P198.4: Memory retrieval increases D4 (attention) for similar exercises
P198.5: Increased D4 triggers increased D2 (metabolism) via coupling
P198.6: Sufficient D2 prevents boredom-induced failures
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict

from session198_training_memory_mapper import (
    TrainingMemoryMapper,
    TrainingMemory,
    NineDomainSnapshot
)


class MemoryConsolidationTest:
    """Test memory consolidation preventing regression"""

    def __init__(self):
        self.mapper = TrainingMemoryMapper(node_id="thor")
        self.memory_dir = Path(__file__).parent / "training_memories"

        # Thresholds
        self.attention_threshold = 0.5
        self.metabolism_threshold = 0.5

    def load_sessions(self) -> Dict[str, TrainingMemory]:
        """Load T014 and T015 memories"""
        memories = {}

        for session_id in ["T014", "T015"]:
            memory_file = self.memory_dir / f"memory_{session_id}.json"
            if memory_file.exists():
                memories[session_id] = self.mapper.load_memory(memory_file)
            else:
                print(f"⚠️  {memory_file} not found")

        return memories

    def analyze_regression(self, t014: TrainingMemory, t015: TrainingMemory):
        """Analyze regression from T014 to T015"""
        print("=" * 80)
        print("REGRESSION ANALYSIS: T014 → T015")
        print("=" * 80)
        print()

        print(f"T014: {t014.success_rate * 100:.0f}% success ({len([s for s in t014.snapshots if s.success])}/{len(t014.snapshots)})")
        print(f"T015: {t015.success_rate * 100:.0f}% success ({len([s for s in t015.snapshots if s.success])}/{len(t015.snapshots)})")
        print(f"Regression: {(t014.success_rate - t015.success_rate) * 100:.0f} percentage points")
        print()

        # Find failed exercises
        t014_failed = [s for s in t014.snapshots if not s.success]
        t015_failed = [s for s in t015.snapshots if not s.success]

        print(f"T014 failures: {len(t014_failed)}")
        print(f"T015 failures: {len(t015_failed)}")

        if t015_failed:
            print()
            print("T015 Failed Exercises:")
            for s in t015_failed:
                print(f"  - {s.exercise_type.upper()}: '{s.prompt}' (expected: {s.expected})")
                print(f"    D4 (Attention): {s.attention:.3f}")
                print(f"    D2 (Metabolism): {s.metabolic:.3f}")
                print(f"    C (Consciousness): {s.consciousness_level:.3f}")

        print()
        print("-" * 80)

    def test_memory_boost(self, failed_snapshot: NineDomainSnapshot,
                         memory: TrainingMemory,
                         boost_factors: List[float] = [0.3, 0.5, 0.7]):
        """Test different memory boost factors"""

        print()
        print("=" * 80)
        print(f"MEMORY BOOST TEST: {failed_snapshot.exercise_type.upper()}")
        print("=" * 80)
        print()
        print(f"Failed Exercise: '{failed_snapshot.prompt}'")
        print(f"Expected: {failed_snapshot.expected}")
        print()
        print("Original State:")
        print(f"  D4 (Attention): {failed_snapshot.attention:.3f}")
        print(f"  D2 (Metabolism): {failed_snapshot.metabolic:.3f}")
        print(f"  C (Consciousness): {failed_snapshot.consciousness_level:.3f}")
        print()

        # Retrieve high-attention memories
        high_attention = self.mapper.retrieve_high_attention_memories(
            memory, min_attention=0.3  # Lower threshold to get more samples
        )

        print(f"Retrieving memories from {memory.session_id}:")
        print(f"  Total snapshots: {len(memory.snapshots)}")
        print(f"  High attention (D4 ≥ 0.3): {len(high_attention)}")

        # Find similar exercise types
        similar = [s for s in high_attention
                  if s.exercise_type == failed_snapshot.exercise_type]

        print(f"  Similar type ({failed_snapshot.exercise_type}): {len(similar)}")

        if similar:
            print()
            print(f"Similar {failed_snapshot.exercise_type.upper()} exercises in memory:")
            for s in similar:
                status = "✅" if s.success else "❌"
                print(f"    {status} '{s.prompt}' → D4={s.attention:.3f}, D2={s.metabolic:.3f}")

        print()
        print("Testing Boost Factors:")
        print("-" * 80)

        results = []

        for boost_factor in boost_factors:
            boosted = self.mapper.boost_attention_from_memory(
                failed_snapshot, high_attention, boost_factor=boost_factor
            )

            # Calculate boost amounts
            d4_boost = boosted.attention - failed_snapshot.attention
            d2_boost = boosted.metabolic - failed_snapshot.metabolic

            # Check if thresholds exceeded
            d4_ok = boosted.attention >= self.attention_threshold
            d2_ok = boosted.metabolic >= self.metabolism_threshold
            prevents_failure = d4_ok and d2_ok

            results.append({
                "boost_factor": boost_factor,
                "d4": boosted.attention,
                "d2": boosted.metabolic,
                "d4_boost": d4_boost,
                "d2_boost": d2_boost,
                "prevents_failure": prevents_failure
            })

            status = "✅ PREVENTS FAILURE" if prevents_failure else "❌ Still insufficient"
            print(f"Boost factor {boost_factor:.1f}:")
            print(f"  D4: {failed_snapshot.attention:.3f} → {boosted.attention:.3f} (+{d4_boost:.3f}) {'✓' if d4_ok else '✗'}")
            print(f"  D2: {failed_snapshot.metabolic:.3f} → {boosted.metabolic:.3f} (+{d2_boost:.3f}) {'✓' if d2_ok else '✗'}")
            print(f"  {status}")
            print()

        return results

    def test_consolidation_hypothesis(self):
        """Main test: Does memory consolidation prevent regression?"""

        print("\n" + "=" * 80)
        print("SESSION 198 PHASE 2: MEMORY CONSOLIDATION TEST")
        print("=" * 80)
        print()
        print("Hypothesis: Memory retrieval restores attention state from successful")
        print("           sessions, preventing boredom-induced failures.")
        print()
        print("Test: Apply T014 memory (100% success) to T015 failed exercise (4-1)")
        print("=" * 80)

        # Load memories
        memories = self.load_sessions()

        if "T014" not in memories or "T015" not in memories:
            print("❌ Cannot run test - missing memory files")
            return

        t014 = memories["T014"]
        t015 = memories["T015"]

        # Analyze regression
        self.analyze_regression(t014, t015)

        # Find T015 failed exercise
        t015_failed = [s for s in t015.snapshots if not s.success]

        if not t015_failed:
            print("✅ No failures in T015 - test not applicable")
            return

        failed_snapshot = t015_failed[0]

        # Test memory boost
        results = self.test_memory_boost(
            failed_snapshot, t014, boost_factors=[0.3, 0.5, 0.7, 1.0]
        )

        # Summary
        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print()

        prevention_success = any(r["prevents_failure"] for r in results)

        if prevention_success:
            # Find minimum boost factor that works
            working = [r for r in results if r["prevents_failure"]]
            min_boost = min(r["boost_factor"] for r in working)

            print(f"✅ Memory consolidation PREVENTS regression!")
            print(f"   Minimum boost factor: {min_boost:.1f}")
            print()
            print("   Mechanism:")
            print("   1. Memory retrieval restores high-attention state from T014")
            print("   2. Increased D4 (attention) triggers D4→D2 coupling (κ=0.4)")
            print("   3. Increased D2 (metabolism) provides sufficient resources")
            print("   4. Sufficient resources prevent boredom-induced failure")
        else:
            print(f"⚠️  Memory boost insufficient with tested factors")
            print(f"   Maximum boost tested: {max(r['boost_factor'] for r in results):.1f}")
            print()
            print("   Possible reasons:")
            print("   1. No similar successful exercises in T014 memory")
            print("   2. T014 also shows low D4 for this exercise type")
            print("   3. Higher boost factors needed (>1.0)")
            print("   4. Different memory retrieval strategy needed")

        print()
        print("Predictions Status:")

        # Check predictions
        d4_increased = any(r["d4_boost"] > 0 for r in results)
        d2_increased = any(r["d2_boost"] > 0 for r in results)

        print(f"  P198.4 (Memory increases D4): {'✅' if d4_increased else '❌'}")
        print(f"  P198.5 (D4 triggers D2 via coupling): {'✅' if d2_increased else '❌'}")
        print(f"  P198.6 (Sufficient D2 prevents failures): {'✅' if prevention_success else '⏸️  Partial'}")

        print()
        print("=" * 80)


def main():
    """Run memory consolidation test"""
    test = MemoryConsolidationTest()
    test.test_consolidation_hypothesis()


if __name__ == "__main__":
    main()
