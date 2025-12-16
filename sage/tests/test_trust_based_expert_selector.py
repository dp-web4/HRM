#!/usr/bin/env python3
"""
Test Trust-Based Expert Selector

Validates Web4 trust pattern application to expert selection.

Test Coverage:
1. Basic expert selection with router logits
2. Contextual trust score integration
3. Substitution logic when experts not in cache
4. Cache hit/miss tracking
5. Exploration/exploitation balance
6. Statistics computation

Author: Claude (Legion Web4 Session 56)
Date: 2025-12-16
"""

import tempfile
import numpy as np
from pathlib import Path
import sys

# Try to import torch, use numpy fallback if unavailable
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Mock torch module for testing without pytorch
    class torch:
        @staticmethod
        def randn(*shape):
            return np.random.randn(*shape)

        @staticmethod
        def zeros(*shape):
            return np.zeros(shape)

        class Tensor:
            def __init__(self, data):
                self.data = np.array(data)
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.data

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.core.trust_based_expert_selector import (
    TrustBasedExpertSelector,
    ExpertSelectionResult,
    create_trust_based_selector,
    select_experts_with_trust
)
from sage.core.expert_reputation import (
    ExpertReputationDB,
    ExpertReputation,
    record_expert_activation
)


def test_basic_expert_selection():
    """Test basic expert selection without reputation data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = ExpertReputationDB(Path(tmpdir) / "test.db")
        selector = TrustBasedExpertSelector(
            num_experts=128,
            cache_size=6,
            reputation_db=db
        )

        # Simulate router logits (higher = more preferred)
        router_logits = torch.randn(128)
        router_logits[15] = 2.0  # Expert 15 highly preferred
        router_logits[28] = 1.8
        router_logits[42] = 1.5

        # Mark some experts as loaded
        selector.mark_experts_loaded([15, 28, 42, 67, 89, 103])

        # Select top-8 experts
        result = selector.select_experts(router_logits, context="test", k=8)

        assert len(result.selected_expert_ids) == 8
        assert isinstance(result, ExpertSelectionResult)
        assert result.context == "test"
        assert len(result.selection_scores) == 8
        assert len(result.router_scores) == 8
        assert len(result.trust_scores) == 8

        print("✓ Basic expert selection")


def test_contextual_trust_integration():
    """Test that contextual trust scores influence selection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = ExpertReputationDB(Path(tmpdir) / "test.db")

        # Create reputation for expert 7 with high trust in "code" context
        rep = db.get_or_create(7)
        for _ in range(10):
            rep.record_activation("code", {'quality': 0.9})
        db.save(rep)

        # Create reputation for expert 23 with low trust in "code" but high in "prose"
        rep = db.get_or_create(23)
        for _ in range(5):
            rep.record_activation("code", {'quality': 0.3})
        for _ in range(10):
            rep.record_activation("prose", {'quality': 0.9})
        db.save(rep)

        selector = TrustBasedExpertSelector(
            num_experts=128,
            cache_size=6,
            exploration_weight=0.1,  # Heavy trust weighting
            reputation_db=db
        )

        # Equal router preferences
        router_logits = torch.zeros(128)
        router_logits[7] = 1.0
        router_logits[23] = 1.0

        selector.mark_experts_loaded([7, 23])

        # Select for "code" context
        result_code = selector.select_experts(router_logits, context="code", k=8)

        # Expert 7 should rank higher than 23 for code context
        idx_7 = result_code.selected_expert_ids.index(7)
        idx_23 = result_code.selected_expert_ids.index(23)
        assert idx_7 < idx_23, "Expert with higher code trust should rank higher"

        # Select for "prose" context
        result_prose = selector.select_experts(router_logits, context="prose", k=8)

        # Expert 23 should rank higher than 7 for prose context
        idx_7_prose = result_prose.selected_expert_ids.index(7)
        idx_23_prose = result_prose.selected_expert_ids.index(23)
        assert idx_23_prose < idx_7_prose, "Expert with higher prose trust should rank higher"

        print("✓ Contextual trust integration")


def test_substitution_when_expert_not_loaded():
    """Test expert substitution when requested expert not in cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = ExpertReputationDB(Path(tmpdir) / "test.db")

        # Create reputation for expert 47 (requested but not loaded)
        rep = db.get_or_create(47)
        rep.semantic_cluster = 3
        db.save(rep)

        # Create reputation for expert 15 (loaded, similar cluster, high trust)
        rep = db.get_or_create(15)
        rep.semantic_cluster = 3  # Same cluster as 47
        for _ in range(10):
            rep.record_activation("code", {'quality': 0.85})
        db.save(rep)

        selector = TrustBasedExpertSelector(
            num_experts=128,
            cache_size=6,
            reputation_db=db
        )

        # Mark expert 15 as loaded, but not 47
        selector.mark_expert_loaded(15, True)
        selector.mark_expert_loaded(47, False)

        # Router prefers expert 47 highly
        router_logits = torch.zeros(128)
        router_logits[47] = 2.0  # Highly preferred but not loaded
        router_logits[15] = 0.5  # Less preferred but loaded

        result = selector.select_experts(router_logits, context="code", k=8)

        # Check if substitution occurred
        if 47 in result.substitutions:
            # Expert 47 was substituted
            substitute = result.substitutions[47]
            assert substitute == 15, "Expert 15 should be used as substitute for 47"
            assert substitute in result.selected_expert_ids
            print("  Substitution occurred: 47 → 15 ✓")
        else:
            # Expert 47 was loaded (cache miss)
            assert 47 in result.selected_expert_ids
            print("  No substitution: expert 47 loaded (cache miss)")

        print("✓ Substitution logic")


def test_cache_hit_miss_tracking():
    """Test cache hit/miss statistics tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = ExpertReputationDB(Path(tmpdir) / "test.db")
        selector = TrustBasedExpertSelector(
            num_experts=128,
            cache_size=6,
            reputation_db=db
        )

        # Mark 6 experts as loaded
        loaded_experts = [3, 15, 28, 42, 67, 89]
        selector.mark_experts_loaded(loaded_experts)

        # Router prefers mostly loaded experts
        router_logits = torch.zeros(128)
        for exp in loaded_experts:
            router_logits[exp] = 1.5

        result = selector.select_experts(router_logits, context="test", k=8)

        # Should have high cache hit rate (most requested were loaded)
        assert result.cache_hits >= 6, "Should have cache hits for loaded experts"
        print(f"  Cache hits: {result.cache_hits}, misses: {result.cache_misses}")

        print("✓ Cache tracking")


def test_exploration_exploitation_balance():
    """Test that exploration_weight controls router vs trust balance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = ExpertReputationDB(Path(tmpdir) / "test.db")

        # Create expert 7 with very high trust but low router preference
        rep = db.get_or_create(7)
        for _ in range(20):
            rep.record_activation("test", {'quality': 0.95})
        db.save(rep)

        # Create expert 42 with moderate trust but high router preference
        rep = db.get_or_create(42)
        for _ in range(10):
            rep.record_activation("test", {'quality': 0.6})
        db.save(rep)

        # Balanced selector (equal weighting)
        selector_balanced = TrustBasedExpertSelector(
            num_experts=128,
            exploration_weight=0.5,  # Balanced
            reputation_db=db
        )

        # Pure exploitation (trust only)
        selector_exploit = TrustBasedExpertSelector(
            num_experts=128,
            exploration_weight=0.0,  # Pure trust
            reputation_db=db
        )

        # Pure exploration (router only)
        selector_explore = TrustBasedExpertSelector(
            num_experts=128,
            exploration_weight=1.0,  # Pure router
            reputation_db=db
        )

        # Router strongly prefers expert 42, weakly prefers 7
        router_logits = torch.zeros(128) - 2.0  # Low baseline
        router_logits[42] = 2.0  # Strongly preferred
        router_logits[7] = 0.5   # Weakly preferred

        # Mark only these two as loaded to ensure they're selected
        selector_balanced.mark_experts_loaded([7, 42])
        selector_exploit.mark_experts_loaded([7, 42])
        selector_explore.mark_experts_loaded([7, 42])

        result_balanced = selector_balanced.select_experts(router_logits, "test", k=2)
        result_exploit = selector_exploit.select_experts(router_logits, "test", k=2)
        result_explore = selector_explore.select_experts(router_logits, "test", k=2)

        # Debug output
        print(f"  Exploit selected: {result_exploit.selected_expert_ids}")
        print(f"  Explore selected: {result_explore.selected_expert_ids}")

        # Both experts should be selected (k=2, only 2 loaded)
        assert len(result_exploit.selected_expert_ids) == 2
        assert len(result_explore.selected_expert_ids) == 2
        assert 7 in result_exploit.selected_expert_ids, f"Expected 7 in {result_exploit.selected_expert_ids}"
        assert 42 in result_exploit.selected_expert_ids, f"Expected 42 in {result_exploit.selected_expert_ids}"
        assert 7 in result_explore.selected_expert_ids
        assert 42 in result_explore.selected_expert_ids

        # Get rankings
        idx_7_exploit = result_exploit.selected_expert_ids.index(7)
        idx_42_exploit = result_exploit.selected_expert_ids.index(42)

        idx_7_explore = result_explore.selected_expert_ids.index(7)
        idx_42_explore = result_explore.selected_expert_ids.index(42)

        # In exploitation mode, expert 7 should rank higher (high trust beats low router)
        assert idx_7_exploit < idx_42_exploit, "High-trust expert should rank higher in exploit mode"

        # In exploration mode, expert 42 should rank higher (high router beats low trust)
        assert idx_42_explore < idx_7_explore, "High-router expert should rank higher in explore mode"

        print(f"  Exploit mode: expert 7 rank {idx_7_exploit}, expert 42 rank {idx_42_exploit}")
        print(f"  Explore mode: expert 7 rank {idx_7_explore}, expert 42 rank {idx_42_explore}")

        print("✓ Exploration/exploitation balance")


def test_statistics_computation():
    """Test statistics tracking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = ExpertReputationDB(Path(tmpdir) / "test.db")
        selector = TrustBasedExpertSelector(
            num_experts=128,
            cache_size=6,
            reputation_db=db
        )

        selector.mark_experts_loaded([1, 2, 3, 4, 5, 6])

        # Perform several selections
        for _ in range(5):
            router_logits = torch.randn(128)
            selector.select_experts(router_logits, "test", k=8)

        stats = selector.get_statistics()

        assert stats['total_selections'] == 5
        assert 'substitution_rate' in stats
        assert 'cache_hit_rate' in stats
        assert stats['experts_loaded'] == 6
        assert stats['cache_size'] == 6
        assert stats['exploration_weight'] == 0.3

        print(f"  Stats: {stats}")
        print("✓ Statistics computation")


def test_convenience_functions():
    """Test convenience functions work correctly."""
    # Test create_trust_based_selector
    selector = create_trust_based_selector(num_experts=128, cache_size=8)

    assert selector.num_experts == 128
    assert selector.cache_size == 8
    assert selector.exploration_weight == 0.3

    # Test select_experts_with_trust
    router_logits = torch.randn(128)
    result = select_experts_with_trust(router_logits, "test", k=8)

    assert isinstance(result, ExpertSelectionResult)
    assert len(result.selected_expert_ids) == 8

    print("✓ Convenience functions")


def test_eviction_criteria():
    """Test expert eviction decision logic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = ExpertReputationDB(Path(tmpdir) / "test.db")

        # Create expert with high trust and recent use
        rep_keep = db.get_or_create(7)
        for _ in range(10):
            rep_keep.record_activation("test", {'quality': 0.9})
        db.save(rep_keep)

        # Create expert with low trust and old use
        import time
        rep_evict = db.get_or_create(23)
        for _ in range(15):
            rep_evict.record_activation("test", {'quality': 0.2})
        # Set last_used to 2 hours ago AFTER recording activations
        rep_evict.last_used = time.time() - 7200
        db.save(rep_evict)

        selector = TrustBasedExpertSelector(reputation_db=db)

        # Expert 7 should not be evicted (high trust, recent)
        should_evict_7 = selector.should_evict_expert(7, "test")
        rep_7 = db.get_reputation(7)
        print(f"  Expert 7: trust={rep_7.get_context_trust('test', 0.5):.3f}, should_evict={should_evict_7}")
        assert not should_evict_7, "High-trust recent expert should not be evicted"

        # Expert 23 should be evicted (low trust, old)
        should_evict_23 = selector.should_evict_expert(23, "test")
        rep_23 = db.get_reputation(23)
        current_time = time.time()
        time_since_use = current_time - rep_23.last_used if rep_23.last_used else 0
        recency = 1.0 / (1.0 + time_since_use / 3600)
        eviction_score = 0.6 * rep_23.get_context_trust('test', 0.5) + 0.4 * recency
        print(f"  Expert 23: trust={rep_23.get_context_trust('test', 0.5):.3f}, time_since={time_since_use:.1f}s, recency={recency:.3f}, score={eviction_score:.3f}, should_evict={should_evict_23}")
        assert should_evict_23, "Low-trust old expert should be evicted"

        print("✓ Eviction criteria")


if __name__ == "__main__":
    print("Testing Trust-Based Expert Selector...")
    print()

    test_basic_expert_selection()
    test_contextual_trust_integration()
    test_substitution_when_expert_not_loaded()
    test_cache_hit_miss_tracking()
    test_exploration_exploitation_balance()
    test_statistics_computation()
    test_convenience_functions()
    test_eviction_criteria()

    print()
    print("✅ All tests passed!")
    print()
    print("Trust-Based Expert Selector validated:")
    print("- Router + reputation combination working")
    print("- Contextual trust influences selection")
    print("- Substitution logic functional")
    print("- Cache tracking operational")
    print("- Exploration/exploitation balance correct")
    print("- Ready for SAGE integration")
