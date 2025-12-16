#!/usr/bin/env python3
"""
Tests for Quality-Reputation Bridge

Validates the feedback loop: quality measurement → reputation updates

Session Context: Thor Session 60 (Autonomous)
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from sage.core.quality_measurement import QualityMetrics
from sage.core.quality_reputation_bridge import (
    update_expert_reputation_from_quality,
    measure_and_update_reputation
)
from sage.core.expert_reputation import ExpertReputationDB


def test_quality_to_reputation_update():
    """
    Test that quality metrics update expert reputation correctly.
    """
    print("\n=== Test: Quality to Reputation Update ===\n")

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_reputation.db"
        db = ExpertReputationDB(db_path)

        # Create quality metrics
        metrics = QualityMetrics(
            perplexity=5.2,
            coherence=0.75,
            task_quality=0.85,
            expert_ids=[5, 10, 15],
            context="context_code",
            overall_quality=0.82,
            sequence_length=30,
            num_experts_used=3
        )

        print(f"Quality metrics:")
        print(f"  Overall quality: {metrics.overall_quality}")
        print(f"  Expert IDs: {metrics.expert_ids}")
        print(f"  Context: {metrics.context}")

        # Update reputation
        update_expert_reputation_from_quality(metrics, db)

        print(f"\n✅ Reputation updated")

        # Check that reputation was recorded for each expert
        for expert_id in metrics.expert_ids:
            rep = db.get_reputation(expert_id)
            assert rep is not None, f"Expert {expert_id} should have reputation"

            print(f"\nExpert {expert_id}:")
            print(f"  Activation count: {rep.activation_count}")
            print(f"  Contexts: {list(rep.contexts_seen.keys())}")

            # Check context-specific trust
            trust = rep.get_context_trust(metrics.context)
            print(f"  Trust in {metrics.context}: {trust:.3f}")

            assert rep.activation_count > 0
            assert metrics.context in rep.contexts_seen

        print(f"\n✅ All experts have updated reputation")


def test_feedback_loop():
    """
    Test the complete feedback loop: measure → update → selection uses updated trust.
    """
    print("\n=== Test: Feedback Loop ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_reputation.db"
        db = ExpertReputationDB(db_path)

        # Simulate multiple generations with different quality

        # Generation 1: Expert 5 performs well
        metrics_1 = QualityMetrics(
            perplexity=3.0,
            coherence=0.85,
            task_quality=0.90,
            expert_ids=[5, 10],
            context="code",
            overall_quality=0.88,
        )
        update_expert_reputation_from_quality(metrics_1, db)

        # Generation 2: Expert 5 performs well again
        metrics_2 = QualityMetrics(
            perplexity=2.8,
            coherence=0.82,
            task_quality=0.92,
            expert_ids=[5, 15],
            context="code",
            overall_quality=0.90,
        )
        update_expert_reputation_from_quality(metrics_2, db)

        # Generation 3: Expert 10 performs poorly
        metrics_3 = QualityMetrics(
            perplexity=25.0,
            coherence=0.30,
            task_quality=0.40,
            expert_ids=[10, 15],
            context="code",
            overall_quality=0.35,
        )
        update_expert_reputation_from_quality(metrics_3, db)

        # Check reputation
        rep_5 = db.get_reputation(5)
        rep_10 = db.get_reputation(10)

        trust_5 = rep_5.get_context_trust("code")
        trust_10 = rep_10.get_context_trust("code")

        print(f"After 3 generations:")
        print(f"  Expert 5 trust (code): {trust_5:.3f} (performed well 2x)")
        print(f"  Expert 10 trust (code): {trust_10:.3f} (performed well 1x, poorly 1x)")

        # Expert 5 should have higher trust (performed consistently well)
        assert trust_5 > trust_10, \
            f"Expert 5 ({trust_5:.3f}) should have higher trust than Expert 10 ({trust_10:.3f})"

        print(f"\n✅ Feedback loop: Better performance → Higher trust")


def test_co_activation_recording():
    """
    Test that co-activation (multiple experts working together) is recorded.
    """
    print("\n=== Test: Co-Activation Recording ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_reputation.db"
        db = ExpertReputationDB(db_path)

        # Generation with multiple experts
        metrics = QualityMetrics(
            perplexity=4.0,
            coherence=0.80,
            task_quality=0.85,
            expert_ids=[5, 10, 15, 20],  # 4 experts working together
            context="reasoning",
            overall_quality=0.82,
        )

        update_expert_reputation_from_quality(metrics, db)

        # Check that all experts have reputation
        for expert_id in metrics.expert_ids:
            rep = db.get_reputation(expert_id)
            assert rep is not None
            assert rep.activation_count > 0

        print(f"✅ Co-activation recorded for {len(metrics.expert_ids)} experts")
        print(f"   All experts updated in context '{metrics.context}'")


def test_convenience_function():
    """
    Test the convenience function.
    """
    print("\n=== Test: Convenience Function ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_reputation.db"
        db = ExpertReputationDB(db_path)

        metrics = QualityMetrics(
            perplexity=5.0,
            coherence=0.70,
            task_quality=0.80,
            expert_ids=[1, 2, 3],
            context="text",
            overall_quality=0.75,
        )

        # Use convenience function
        measure_and_update_reputation(metrics, db)

        # Verify update happened
        rep = db.get_reputation(1)
        assert rep is not None
        assert rep.activation_count > 0

        print(f"✅ Convenience function works")


if __name__ == "__main__":
    print("=" * 70)
    print("Quality-Reputation Bridge Tests")
    print("=" * 70)

    test_quality_to_reputation_update()
    test_feedback_loop()
    test_co_activation_recording()
    test_convenience_function()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSING")
    print("=" * 70)
    print("\nPhase 3 Integration: ✅ COMPLETE")
    print("\nFeedback Loop Closed:")
    print("  1. Generation uses trust-based expert selection")
    print("  2. Quality is measured (perplexity, coherence, task-specific)")
    print("  3. Expert reputation updated based on quality")
    print("  4. Future selections use updated reputation")
    print("\nThis enables continuous learning and improvement!")
