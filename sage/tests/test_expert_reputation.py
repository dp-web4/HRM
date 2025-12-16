#!/usr/bin/env python3
"""
Test Expert Reputation System

Validates Web4 trust pattern application to SAGE expert management.

Test Coverage:
1. ExpertReputation creation and updates
2. Contextual trust evolution
3. Co-activation tracking
4. Substitution recording
5. Database persistence (round-trip)
6. Trust-based decision making

Author: Claude (Legion Web4 Session 55)
Date: 2025-12-15
"""

import tempfile
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.core.expert_reputation import (
    ExpertReputation,
    ExpertReputationDB,
    record_expert_activation,
    record_expert_co_activation
)


def test_expert_reputation_creation():
    """Test creating expert reputation with defaults."""
    rep = ExpertReputation(expert_id=42, component="thinker")

    assert rep.expert_id == 42
    assert rep.component == "thinker"
    assert rep.activation_count == 0
    assert rep.convergence_rate == 0.5  # Default neutral
    assert rep.stability == 0.5
    assert rep.first_seen is None
    assert len(rep.context_trust) == 0
    assert len(rep.co_activated_with) == 0


def test_context_trust_update():
    """Test Bayesian-style context trust updates."""
    rep = ExpertReputation(expert_id=15, component="thinker")

    # Initial update: unknown context starts at 0.5 (neutral prior)
    rep.update_context_trust("code", evidence_quality=0.9, learning_rate=0.1)

    # Trust should move toward evidence
    # Expected: 0.5 * 0.9 + 0.1 * 0.9 = 0.54
    assert "code" in rep.context_trust
    assert 0.53 < rep.context_trust["code"] < 0.55

    # Second update: trust should continue evolving
    rep.update_context_trust("code", evidence_quality=0.9, learning_rate=0.1)
    assert rep.context_trust["code"] > 0.55  # Moving higher

    # Different context: independent trust
    rep.update_context_trust("prose", evidence_quality=0.3, learning_rate=0.1)
    assert "prose" in rep.context_trust
    assert rep.context_trust["prose"] < 0.5  # Moving lower from neutral


def test_activation_recording():
    """Test recording expert activation updates all metrics."""
    rep = ExpertReputation(expert_id=7, component="thinker")

    performance = {
        'convergence': 0.85,
        'stability': 0.92,
        'confidence': 0.78,
        'quality': 0.88
    }

    # Record activation
    rep.record_activation("technical", performance)

    # Check updates
    assert rep.activation_count == 1
    assert rep.contexts_seen["technical"] == 1
    assert rep.first_seen is not None
    assert rep.last_used is not None

    # Performance metrics updated (EMA from 0.5 default)
    # α=0.1: new = 0.9 * 0.5 + 0.1 * 0.85 = 0.535
    assert 0.53 < rep.convergence_rate < 0.54
    assert 0.53 < rep.context_trust["technical"] < 0.55

    # Second activation in same context
    rep.record_activation("technical", performance)
    assert rep.activation_count == 2
    assert rep.contexts_seen["technical"] == 2


def test_co_activation_tracking():
    """Test tracking which experts work well together."""
    rep = ExpertReputation(expert_id=3, component="thinker")

    # Expert 3 co-activated with expert 15 and 28
    rep.record_co_activation(15, combined_quality=0.85)
    rep.record_co_activation(28, combined_quality=0.92)

    assert rep.co_activated_with[15] == 1
    assert rep.co_activated_with[28] == 1
    assert rep.successful_pairs[15] == 0.85
    assert rep.successful_pairs[28] == 0.92

    # Second co-activation with expert 15 (quality varies)
    rep.record_co_activation(15, combined_quality=0.95)
    assert rep.co_activated_with[15] == 2
    # EMA update: 0.9 * 0.85 + 0.1 * 0.95 = 0.86
    assert 0.855 < rep.successful_pairs[15] < 0.865


def test_substitution_recording():
    """Test recording substitution effectiveness."""
    rep = ExpertReputation(expert_id=23, component="thinker")

    # Expert 23 substituted for expert 47 with good results
    rep.record_substitution(
        requested_expert_id=47,
        quality_delta=0.15,  # Positive: substitution helped!
        context="creative"
    )

    assert len(rep.substituted_for) == 1
    requested, delta, ctx = rep.substituted_for[0]
    assert requested == 47
    assert delta == 0.15
    assert ctx == "creative"

    # Successful substitution should boost trust in that context
    # Starting from 0.5, boost by min(0.1, 0.15) = 0.1 → 0.6
    assert "creative" in rep.context_trust
    assert rep.context_trust["creative"] == 0.6


def test_database_persistence():
    """Test saving and loading reputation from database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_reputation.db"
        db = ExpertReputationDB(db_path)

        # Create and populate reputation
        rep = ExpertReputation(expert_id=42, component="thinker")
        rep.record_activation("code", {
            'convergence': 0.9,
            'stability': 0.85,
            'confidence': 0.8,
            'quality': 0.87
        })
        rep.record_co_activation(15, 0.92)
        rep.record_substitution(47, 0.12, "code")

        # Save to database
        db.save(rep)

        # Load from database
        loaded = db.get_reputation(42)

        assert loaded is not None
        assert loaded.expert_id == 42
        assert loaded.activation_count == 1
        assert "code" in loaded.context_trust
        assert 15 in loaded.co_activated_with
        assert len(loaded.substituted_for) == 1

        # Verify context trust preserved
        assert loaded.context_trust["code"] == rep.context_trust["code"]

        db.close()


def test_database_get_or_create():
    """Test get_or_create creates new reputation if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_reputation.db"
        db = ExpertReputationDB(db_path)

        # Get non-existent expert (should create)
        rep = db.get_or_create(99)
        assert rep.expert_id == 99
        assert rep.activation_count == 0

        # Modify and save
        rep.activation_count = 5
        db.save(rep)

        # Get again (should load existing)
        rep2 = db.get_or_create(99)
        assert rep2.activation_count == 5

        db.close()


def test_database_multiple_experts():
    """Test storing multiple expert reputations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_reputation.db"
        db = ExpertReputationDB(db_path)

        # Create multiple experts
        for expert_id in [3, 15, 28, 47]:
            rep = db.get_or_create(expert_id)
            rep.record_activation("test_context", {
                'quality': 0.7 + (expert_id * 0.01)
            })
            db.save(rep)

        # Load all reputations
        all_reps = db.get_all_reputations("thinker")

        assert len(all_reps) >= 4
        expert_ids = {r.expert_id for r in all_reps}
        assert 3 in expert_ids
        assert 15 in expert_ids
        assert 28 in expert_ids
        assert 47 in expert_ids

        db.close()


def test_database_statistics():
    """Test database statistics calculation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_reputation.db"
        db = ExpertReputationDB(db_path)

        # Create some data
        rep1 = db.get_or_create(1)
        rep1.record_activation("code", {'quality': 0.8})
        rep1.record_activation("prose", {'quality': 0.6})
        rep1.record_co_activation(2, 0.85)
        db.save(rep1)

        rep2 = db.get_or_create(2)
        rep2.record_activation("code", {'quality': 0.75})
        rep2.record_substitution(3, 0.1, "code")
        db.save(rep2)

        # Get statistics
        stats = db.get_statistics()

        assert stats['total_experts_tracked'] >= 2
        assert stats['unique_contexts'] >= 2  # "code" and "prose"
        assert stats['expert_pairs_tracked'] >= 1  # (1, 2) pair
        assert stats['total_substitutions'] >= 1

        db.close()


def test_convenience_functions():
    """Test convenience functions for common operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_reputation.db"
        db = ExpertReputationDB(db_path)

        # Record activation via convenience function
        record_expert_activation(
            expert_id=42,
            context="test",
            performance={'quality': 0.9},
            db=db
        )

        # Check it was recorded
        rep = db.get_reputation(42)
        assert rep is not None
        assert rep.activation_count == 1

        # Record co-activation via convenience function
        record_expert_co_activation(
            expert_ids=[42, 47, 53],
            combined_quality=0.88,
            db=db
        )

        # Check co-activations recorded
        rep = db.get_reputation(42)
        assert 47 in rep.co_activated_with
        assert 53 in rep.co_activated_with

        db.close()


def test_context_trust_evolution():
    """Test how context trust evolves over multiple observations."""
    rep = ExpertReputation(expert_id=7, component="thinker")

    # Simulate 10 high-quality activations in "code" context
    for _ in range(10):
        rep.record_activation("code", {
            'quality': 0.9,
            'convergence': 0.85,
            'stability': 0.88,
            'confidence': 0.82
        })

    # Trust should converge toward high quality
    assert rep.context_trust["code"] > 0.7

    # Simulate 5 low-quality activations in "prose" context
    for _ in range(5):
        rep.record_activation("prose", {
            'quality': 0.3,
            'convergence': 0.4,
            'stability': 0.35,
            'confidence': 0.45
        })

    # Trust should converge toward low quality
    assert rep.context_trust["prose"] < 0.45

    # Original context trust should remain high
    assert rep.context_trust["code"] > 0.7


def test_serialization_round_trip():
    """Test to_dict and from_dict serialization."""
    rep = ExpertReputation(expert_id=42, component="thinker")
    rep.record_activation("code", {'quality': 0.85})
    rep.record_co_activation(15, 0.9)

    # Serialize
    data = rep.to_dict()

    # Deserialize
    rep2 = ExpertReputation.from_dict(data)

    assert rep2.expert_id == rep.expert_id
    assert rep2.activation_count == rep.activation_count
    assert rep2.context_trust == rep.context_trust
    assert rep2.co_activated_with == rep.co_activated_with


if __name__ == "__main__":
    # Run tests
    print("Testing Expert Reputation System...")
    print()

    test_expert_reputation_creation()
    print("✓ Expert reputation creation")

    test_context_trust_update()
    print("✓ Context trust updates")

    test_activation_recording()
    print("✓ Activation recording")

    test_co_activation_tracking()
    print("✓ Co-activation tracking")

    test_substitution_recording()
    print("✓ Substitution recording")

    test_database_persistence()
    print("✓ Database persistence")

    test_database_get_or_create()
    print("✓ Database get_or_create")

    test_database_multiple_experts()
    print("✓ Multiple experts storage")

    test_database_statistics()
    print("✓ Database statistics")

    test_convenience_functions()
    print("✓ Convenience functions")

    test_context_trust_evolution()
    print("✓ Context trust evolution")

    test_serialization_round_trip()
    print("✓ Serialization round-trip")

    print()
    print("✅ All tests passed!")
    print()
    print("Expert Reputation System validated:")
    print("- Web4 trust patterns successfully applied to SAGE")
    print("- Contextual trust evolution working")
    print("- Database persistence functional")
    print("- Ready for integration with expert selection")
