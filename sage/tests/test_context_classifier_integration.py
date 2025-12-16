#!/usr/bin/env python3
"""
Test ContextClassifier integration with TrustBasedExpertSelector

Validates that automatic context classification works with trust-based
expert selection for contextual adaptation.

Session Context: Thor Session 58 (Autonomous)
Building on:
  - Session 56 (Legion): TrustBasedExpertSelector
  - Session 57 (Legion): ContextClassifier
  - Session 57 (Thor): Integration demonstration
"""

import numpy as np
import tempfile
from pathlib import Path

from sage.core.context_classifier import ContextClassifier
from sage.core.trust_based_expert_selector import (
    TrustBasedExpertSelector,
    create_trust_based_selector
)
from sage.core.expert_reputation import (
    ExpertReputationDB,
    record_expert_activation
)


def test_context_classifier_integration_basic():
    """
    Test basic integration: ContextClassifier automatically classifies
    input embeddings for trust-based expert selection.
    """
    print("\n=== Test: Basic ContextClassifier Integration ===\n")

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_reputation.db"
        db = ExpertReputationDB(db_path)

        # Create context classifier
        classifier = ContextClassifier(
            num_contexts=3,  # Small for testing
            embedding_dim=128
        )

        # Create synthetic embeddings for 3 distinct contexts
        np.random.seed(42)

        # Context 0: Cluster around [1, 0, 0, ...]
        context_0_embeddings = np.random.randn(20, 128) * 0.3
        context_0_embeddings[:, 0] += 2.0

        # Context 1: Cluster around [0, 1, 0, ...]
        context_1_embeddings = np.random.randn(20, 128) * 0.3
        context_1_embeddings[:, 1] += 2.0

        # Context 2: Cluster around [0, 0, 1, ...]
        context_2_embeddings = np.random.randn(20, 128) * 0.3
        context_2_embeddings[:, 2] += 2.0

        # Combine and fit classifier
        all_embeddings = np.vstack([
            context_0_embeddings,
            context_1_embeddings,
            context_2_embeddings
        ])
        classifier.fit(all_embeddings)

        print(f"Classifier fitted with {len(all_embeddings)} embeddings")
        print(f"Number of contexts: {classifier.num_contexts}")

        # Create trust-based selector with context classifier
        # Use lower exploration_weight (more trust) for clearer demonstration
        selector = TrustBasedExpertSelector(
            num_experts=32,
            cache_size=8,
            exploration_weight=0.2,  # Lower = more trust influence
            context_classifier=classifier,
            reputation_db=db
        )

        print(f"\nTrustBasedExpertSelector created")
        print(f"  Context classifier: {'Yes' if selector.context_classifier else 'No'}")
        print(f"  Num experts: {selector.num_experts}")
        print(f"  Cache size: {selector.cache_size}")

        # Build reputation for different experts in different contexts
        # First, classify sample embeddings to get context IDs
        sample_0 = context_0_embeddings[0]
        sample_1 = context_1_embeddings[0]
        sample_2 = context_2_embeddings[0]

        ctx_0_info = classifier.classify(sample_0)
        ctx_1_info = classifier.classify(sample_1)
        ctx_2_info = classifier.classify(sample_2)

        ctx_0_id = ctx_0_info.context_id
        ctx_1_id = ctx_1_info.context_id
        ctx_2_id = ctx_2_info.context_id

        print(f"\nClassified contexts:")
        print(f"  Sample 0 → {ctx_0_id} (confidence: {ctx_0_info.confidence:.3f})")
        print(f"  Sample 1 → {ctx_1_id} (confidence: {ctx_1_info.confidence:.3f})")
        print(f"  Sample 2 → {ctx_2_id} (confidence: {ctx_2_info.confidence:.3f})")

        # Build reputation: Expert 5 excels in context 0
        for _ in range(15):
            record_expert_activation(5, ctx_0_id, {'quality': 0.95}, db=db)
            record_expert_activation(10, ctx_0_id, {'quality': 0.6}, db=db)

        # Expert 10 excels in context 1
        for _ in range(15):
            record_expert_activation(5, ctx_1_id, {'quality': 0.5}, db=db)
            record_expert_activation(10, ctx_1_id, {'quality': 0.92}, db=db)

        # Expert 15 excels in context 2
        for _ in range(15):
            record_expert_activation(5, ctx_2_id, {'quality': 0.55}, db=db)
            record_expert_activation(15, ctx_2_id, {'quality': 0.90}, db=db)

        print(f"\nReputation built:")
        print(f"  Expert 5: Excels in {ctx_0_id}")
        print(f"  Expert 10: Excels in {ctx_1_id}")
        print(f"  Expert 15: Excels in {ctx_2_id}")

        # Test automatic context classification during selection
        # Router prefers expert 5 uniformly
        router_logits = np.zeros(32)
        router_logits[5] = 3.0  # High preference
        router_logits[10] = 2.0
        router_logits[15] = 2.0
        router_logits[20] = 1.5

        # Mark experts as loaded
        selector.mark_experts_loaded([5, 10, 15, 20])

        print("\n--- Selection with automatic context classification ---")

        # Selection in context 0 (should prefer expert 5)
        result_0 = selector.select_experts(
            router_logits=router_logits,
            context=None,  # Will be auto-classified
            k=4,
            input_embedding=sample_0  # Context 0 embedding
        )

        print(f"\nContext 0 selection:")
        print(f"  Classified context: {result_0.context}")
        print(f"  Top 3 experts: {result_0.selected_expert_ids[:3]}")
        print(f"  Selection scores: {[f'{s:.3f}' for s in result_0.selection_scores[:3]]}")

        # Selection in context 1 (should prefer expert 10)
        result_1 = selector.select_experts(
            router_logits=router_logits,
            context=None,  # Will be auto-classified
            k=4,
            input_embedding=sample_1  # Context 1 embedding
        )

        print(f"\nContext 1 selection:")
        print(f"  Classified context: {result_1.context}")
        print(f"  Top 3 experts: {result_1.selected_expert_ids[:3]}")
        print(f"  Selection scores: {[f'{s:.3f}' for s in result_1.selection_scores[:3]]}")

        # Selection in context 2 (should prefer expert 15)
        result_2 = selector.select_experts(
            router_logits=router_logits,
            context=None,  # Will be auto-classified
            k=4,
            input_embedding=sample_2  # Context 2 embedding
        )

        print(f"\nContext 2 selection:")
        print(f"  Classified context: {result_2.context}")
        print(f"  Top 3 experts: {result_2.selected_expert_ids[:3]}")
        print(f"  Selection scores: {[f'{s:.3f}' for s in result_2.selection_scores[:3]]}")

        # Validate context adaptation
        # The key insight: contexts were classified, and reputation built for those contexts
        # So the expert that has high reputation in THAT context should rank high

        # Create mapping: which expert has high reputation in which classified context?
        expert_for_context = {}
        expert_for_context[ctx_0_id] = 5   # Expert 5 excels in ctx_0_id
        expert_for_context[ctx_1_id] = 10  # Expert 10 excels in ctx_1_id
        expert_for_context[ctx_2_id] = 15  # Expert 15 excels in ctx_2_id

        # Validate: the expert with high reputation in each context ranks high
        expected_expert_0 = expert_for_context[result_0.context]
        expected_expert_1 = expert_for_context[result_1.context]
        expected_expert_2 = expert_for_context[result_2.context]

        assert expected_expert_0 in result_0.selected_expert_ids[:2], \
            f"Expert {expected_expert_0} should rank high in {result_0.context}, got: {result_0.selected_expert_ids[:3]}"

        assert expected_expert_1 in result_1.selected_expert_ids[:2], \
            f"Expert {expected_expert_1} should rank high in {result_1.context}, got: {result_1.selected_expert_ids[:3]}"

        assert expected_expert_2 in result_2.selected_expert_ids[:2], \
            f"Expert {expected_expert_2} should rank high in {result_2.context}, got: {result_2.selected_expert_ids[:3]}"

        print(f"\n✅ Context adaptation validated!")
        print(f"   Context {result_0.context} → Expert {expected_expert_0} ranks high")
        print(f"   Context {result_1.context} → Expert {expected_expert_1} ranks high")
        print(f"   Context {result_2.context} → Expert {expected_expert_2} ranks high")
        print("   Same router preferences → different selections by context")
        print("   Automatic classification working correctly")


def test_fallback_to_manual_context():
    """
    Test that manual context specification still works when
    context_classifier is not provided.
    """
    print("\n=== Test: Fallback to Manual Context ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_reputation.db"
        db = ExpertReputationDB(db_path)

        # Create selector WITHOUT context classifier
        selector = create_trust_based_selector(
            num_experts=32,
            cache_size=8,
            context_classifier=None  # No classifier
        )

        print(f"TrustBasedExpertSelector created WITHOUT context classifier")

        # Build reputation for manual contexts
        for _ in range(10):
            record_expert_activation(5, "code", {'quality': 0.9}, db=db)
            record_expert_activation(10, "text", {'quality': 0.88}, db=db)

        # Select with manual context strings
        router_logits = np.random.randn(32)
        selector.mark_experts_loaded([5, 10, 15, 20])

        result_code = selector.select_experts(
            router_logits=router_logits,
            context="code",  # Manual context
            k=4
        )

        result_text = selector.select_experts(
            router_logits=router_logits,
            context="text",  # Manual context
            k=4
        )

        print(f"\nManual context 'code': {result_code.context}")
        print(f"Manual context 'text': {result_text.context}")

        assert result_code.context == "code"
        assert result_text.context == "text"

        print("\n✅ Manual context specification works!")


def test_default_context_fallback():
    """
    Test that selector falls back to "general" context when no
    context is provided and no classifier available.
    """
    print("\n=== Test: Default Context Fallback ===\n")

    selector = create_trust_based_selector(
        num_experts=32,
        cache_size=8,
        context_classifier=None
    )

    router_logits = np.random.randn(32)

    # No context provided, no classifier, no embedding
    result = selector.select_experts(
        router_logits=router_logits,
        context=None,  # No context
        k=4
        # No input_embedding either
    )

    print(f"Fallback context: {result.context}")
    assert result.context == "general"

    print("\n✅ Default context fallback works!")


if __name__ == "__main__":
    print("=" * 70)
    print("ContextClassifier Integration Tests")
    print("=" * 70)

    test_context_classifier_integration_basic()
    test_fallback_to_manual_context()
    test_default_context_fallback()

    print("\n" + "=" * 70)
    print("✅ ALL INTEGRATION TESTS PASSING")
    print("=" * 70)
    print("\nIntegration Complete:")
    print("  - ContextClassifier automatically classifies embeddings")
    print("  - TrustBasedExpertSelector uses classified contexts")
    print("  - Contextual trust enables adaptive expert selection")
    print("  - Manual context specification still supported")
    print("  - Fallback to 'general' context when needed")
    print("\nPhase 2 of integration pathway: ✅ COMPLETE")
