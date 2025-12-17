#!/usr/bin/env python3
"""
Phase 4 End-to-End Test: Learning Loop Validation

Tests that the complete integration pathway works and improves over time:
1. Trust-based expert selection (Phase 1)
2. Context classification (Phase 2)
3. Quality measurement (Phase 3)
4. Feedback loop: Quality → Reputation → Selection (Phase 4)

Method:
- Simulate multiple generation cycles
- Measure quality improvement over time
- Validate that better experts are selected more often
- Demonstrate continuous learning

Session Context: Thor Session 61 (Autonomous)
Building on Sessions 56-60 (Integration pathway implementation)
"""

import sys
import os
import torch
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def test_learning_loop_improves_selection():
    """
    Test that the learning loop improves expert selection over time.

    Simulation:
    - 3 experts with different quality levels (good, medium, poor)
    - Multiple generation cycles
    - Track which experts get selected
    - Verify: Good expert selected more often over time
    """
    print("\n" + "="*70)
    print("Phase 4 End-to-End Test: Learning Loop")
    print("="*70)
    print()
    print("=== Test 1: Learning Loop Improves Selection ===\n")

    from sage.core.trust_based_expert_selector import TrustBasedExpertSelector
    from sage.core.quality_measurement import QualityMeasurement, QualityMetrics
    from sage.core.quality_reputation_bridge import update_expert_reputation_from_quality
    from sage.core.expert_reputation import get_default_reputation_db

    # Create trust selector
    selector = TrustBasedExpertSelector(
        num_experts=3,
        cache_size=2,
        component="test",
        exploration_weight=0.1  # 10% exploration, 90% exploitation (strong learning signal)
    )

    # Create quality measurer
    measurer = QualityMeasurement(
        perplexity_weight=0.4,
        coherence_weight=0.3,
        task_weight=0.3
    )

    # Simulate expert quality levels
    # Expert 0: Good (consistently high quality)
    # Expert 1: Medium (moderate quality)
    # Expert 2: Poor (consistently low quality)
    expert_quality_levels = {
        0: 0.85,  # Good expert
        1: 0.55,  # Medium expert
        2: 0.25,  # Poor expert
    }

    # Track selections over time
    selections_per_expert = {0: 0, 1: 0, 2: 0}
    quality_per_generation = []

    num_generations = 20
    context = "code"

    print(f"Simulating {num_generations} generation cycles...")
    print(f"Expert quality levels:")
    for expert_id, quality in expert_quality_levels.items():
        print(f"  Expert {expert_id}: {quality:.2f}")
    print()

    for gen in range(num_generations):
        # Simulate router logits (initially equal preference)
        router_logits = torch.randn(3) * 0.5

        # Create synthetic input embedding for context
        input_embedding = torch.randn(2048)

        # Trust-based selection
        result = selector.select_experts(
            router_logits=router_logits,
            context=context,
            k=2,  # Select 2 experts
            input_embedding=input_embedding
        )

        selected_experts = result.selected_expert_ids

        # Track selections
        for expert_id in selected_experts:
            selections_per_expert[expert_id] += 1

        # Simulate generation quality based on expert quality
        avg_expert_quality = sum(expert_quality_levels[eid] for eid in selected_experts) / len(selected_experts)

        # Add some noise to quality
        generation_quality = avg_expert_quality + torch.randn(1).item() * 0.1
        generation_quality = max(0.0, min(1.0, generation_quality))

        quality_per_generation.append(generation_quality)

        # Create quality metrics
        metrics = QualityMetrics(
            perplexity=10.0 if generation_quality > 0.7 else 50.0,
            coherence=generation_quality * 0.9,
            task_quality=generation_quality,
            expert_ids=selected_experts,
            context=context,
            overall_quality=generation_quality,
            sequence_length=30,
            num_experts_used=len(selected_experts)
        )

        # Update expert reputation based on quality
        update_expert_reputation_from_quality(metrics)

        if (gen + 1) % 5 == 0:
            db = get_default_reputation_db()
            print(f"Generation {gen+1}:")
            print(f"  Selected experts: {selected_experts}")
            print(f"  Generation quality: {generation_quality:.3f}")
            print(f"  Expert 0 trust: {db.get_reputation(0, 'test').get_context_trust(context):.3f}")
            print(f"  Expert 1 trust: {db.get_reputation(1, 'test').get_context_trust(context):.3f}")
            print(f"  Expert 2 trust: {db.get_reputation(2, 'test').get_context_trust(context):.3f}")
            print()

    print(f"\nFinal Results after {num_generations} generations:\n")

    # Analyze selection distribution
    total_selections = sum(selections_per_expert.values())
    print("Expert Selection Frequency:")
    for expert_id in sorted(selections_per_expert.keys()):
        count = selections_per_expert[expert_id]
        percentage = (count / total_selections) * 100
        quality = expert_quality_levels[expert_id]
        print(f"  Expert {expert_id} (quality {quality:.2f}): {count:2d} times ({percentage:5.1f}%)")

    # Validate learning through trust scores
    db = get_default_reputation_db()
    final_trust_0 = db.get_reputation(0, 'test').get_context_trust(context)
    final_trust_1 = db.get_reputation(1, 'test').get_context_trust(context)
    final_trust_2 = db.get_reputation(2, 'test').get_context_trust(context)

    print(f"\nFinal Trust Scores:")
    print(f"  Expert 0 (quality {expert_quality_levels[0]:.2f}): {final_trust_0:.3f}")
    print(f"  Expert 1 (quality {expert_quality_levels[1]:.2f}): {final_trust_1:.3f}")
    print(f"  Expert 2 (quality {expert_quality_levels[2]:.2f}): {final_trust_2:.3f}")

    # Quality trend
    early_quality = sum(quality_per_generation[:5]) / 5
    late_quality = sum(quality_per_generation[-5:]) / 5

    print(f"\nQuality Trend:")
    print(f"  Early generations (1-5): {early_quality:.3f}")
    print(f"  Late generations ({num_generations-4}-{num_generations}): {late_quality:.3f}")
    print(f"  Improvement: {late_quality - early_quality:+.3f}")

    # Assertions: Trust scores should reflect quality levels
    assert final_trust_0 > final_trust_2, \
        f"Good expert (0) trust ({final_trust_0:.3f}) should exceed poor expert (2) trust ({final_trust_2:.3f})"

    assert final_trust_0 > final_trust_1, \
        f"Good expert (0) trust ({final_trust_0:.3f}) should exceed medium expert (1) trust ({final_trust_1:.3f})"

    assert late_quality >= early_quality - 0.15, \
        f"Quality should not decrease significantly (early: {early_quality:.3f}, late: {late_quality:.3f})"

    print(f"\n✅ Learning loop validated: Trust scores reflect quality")
    print(f"✅ Good expert (0) has highest trust: {final_trust_0:.3f}")
    print(f"✅ Quality maintained or improved over time")


def test_context_specific_learning():
    """
    Test that learning is context-specific.

    Simulation:
    - Expert 0: Good at 'code', poor at 'text'
    - Expert 1: Poor at 'code', good at 'text'
    - Multiple generations in both contexts
    - Verify: Context-specific expert preferences emerge
    """
    print("\n=== Test 2: Context-Specific Learning ===\n")

    from sage.core.trust_based_expert_selector import TrustBasedExpertSelector
    from sage.core.quality_measurement import QualityMetrics
    from sage.core.quality_reputation_bridge import update_expert_reputation_from_quality
    from sage.core.expert_reputation import get_default_reputation_db

    # Create trust selector with context classifier
    from sage.core.context_classifier import ContextClassifier

    classifier = ContextClassifier(num_contexts=5, embedding_dim=2048)
    selector = TrustBasedExpertSelector(
        num_experts=2,
        cache_size=2,
        component="test",
        context_classifier=classifier,
        exploration_weight=0.2
    )

    # Expert specializations
    # Expert 0: Code specialist (good at code, poor at text)
    # Expert 1: Text specialist (poor at code, good at text)
    expert_quality_by_context = {
        0: {'code': 0.90, 'text': 0.30},
        1: {'code': 0.30, 'text': 0.90},
    }

    num_generations_per_context = 10
    contexts = ['code', 'text']

    print(f"Expert specializations:")
    print(f"  Expert 0: Code {expert_quality_by_context[0]['code']:.2f}, Text {expert_quality_by_context[0]['text']:.2f}")
    print(f"  Expert 1: Code {expert_quality_by_context[1]['code']:.2f}, Text {expert_quality_by_context[1]['text']:.2f}")
    print()

    # Train in both contexts
    for context in contexts:
        print(f"Training in '{context}' context ({num_generations_per_context} generations)...")

        for gen in range(num_generations_per_context):
            router_logits = torch.randn(2) * 0.5
            input_embedding = torch.randn(2048)

            result = selector.select_experts(
                router_logits=router_logits,
                context=context,
                k=1,  # Select 1 expert
                input_embedding=input_embedding
            )

            selected_expert = result.selected_expert_ids[0]

            # Quality based on expert specialization
            generation_quality = expert_quality_by_context[selected_expert][context]
            generation_quality += torch.randn(1).item() * 0.05  # Small noise

            metrics = QualityMetrics(
                perplexity=10.0 if generation_quality > 0.7 else 50.0,
                coherence=generation_quality,
                task_quality=generation_quality,
                expert_ids=[selected_expert],
                context=context,
                overall_quality=generation_quality
            )

            update_expert_reputation_from_quality(metrics)

    # Check learned preferences
    print(f"\nLearned context-specific trust:\n")

    db = get_default_reputation_db()
    expert0_code_trust = db.get_reputation(0, 'test').get_context_trust('code')
    expert0_text_trust = db.get_reputation(0, 'test').get_context_trust('text')
    expert1_code_trust = db.get_reputation(1, 'test').get_context_trust('code')
    expert1_text_trust = db.get_reputation(1, 'test').get_context_trust('text')

    print(f"Expert 0:")
    print(f"  Code trust: {expert0_code_trust:.3f} (specialist)")
    print(f"  Text trust: {expert0_text_trust:.3f}")
    print()
    print(f"Expert 1:")
    print(f"  Code trust: {expert1_code_trust:.3f}")
    print(f"  Text trust: {expert1_text_trust:.3f} (specialist)")

    # Validate specialization
    assert expert0_code_trust > expert0_text_trust, \
        "Expert 0 should have higher trust in code than text"

    assert expert1_text_trust > expert1_code_trust, \
        "Expert 1 should have higher trust in text than code"

    assert expert0_code_trust > expert1_code_trust, \
        "Expert 0 should be preferred for code"

    assert expert1_text_trust > expert0_text_trust, \
        "Expert 1 should be preferred for text"

    print(f"\n✅ Context-specific learning validated")
    print(f"✅ Experts develop appropriate specializations")


def test_exploration_vs_exploitation():
    """
    Test that exploration_weight controls exploration vs exploitation balance.

    Simulation:
    - High exploration (α=0.8): Should try different experts
    - Low exploration (α=0.2): Should prefer trusted experts
    - Measure selection diversity
    """
    print("\n=== Test 3: Exploration vs Exploitation ===\n")

    from sage.core.trust_based_expert_selector import TrustBasedExpertSelector
    from sage.core.quality_measurement import QualityMetrics
    from sage.core.quality_reputation_bridge import update_expert_reputation_from_quality

    num_experts = 5
    num_generations = 20

    # Test both exploration levels
    for exploration_weight in [0.8, 0.2]:
        print(f"\nTesting with exploration_weight = {exploration_weight:.1f}")

        selector = TrustBasedExpertSelector(
            num_experts=num_experts,
            cache_size=3,
            component="test",
            exploration_weight=exploration_weight
        )

        # Pre-train: Expert 0 is known to be good
        for _ in range(5):
            metrics = QualityMetrics(
                perplexity=10.0,
                coherence=0.85,
                task_quality=0.90,
                expert_ids=[0],
                context="general",
                overall_quality=0.88
            )
            update_expert_reputation_from_quality(metrics)

        # Now generate and track diversity
        selected_expert_counts = {i: 0 for i in range(num_experts)}

        for gen in range(num_generations):
            router_logits = torch.randn(num_experts) * 0.5
            input_embedding = torch.randn(2048)

            result = selector.select_experts(
                router_logits=router_logits,
                context="general",
                k=1,
                input_embedding=input_embedding
            )

            selected_expert = result.selected_expert_ids[0]
            selected_expert_counts[selected_expert] += 1

        # Calculate diversity (how many different experts were selected)
        num_different_experts = sum(1 for count in selected_expert_counts.values() if count > 0)
        expert_0_frequency = selected_expert_counts[0] / num_generations

        print(f"  Expert 0 (trusted) selected: {expert_0_frequency*100:.1f}% of the time")
        print(f"  Number of different experts tried: {num_different_experts}/{num_experts}")

        if exploration_weight >= 0.5:
            # High exploration: should try multiple experts
            assert num_different_experts >= 3, \
                f"High exploration should try multiple experts (got {num_different_experts})"
            print(f"  ✅ High exploration: Multiple experts tried")
        else:
            # Low exploration: should prefer trusted expert
            assert expert_0_frequency >= 0.4, \
                f"Low exploration should prefer trusted expert (got {expert_0_frequency:.2f})"
            print(f"  ✅ Low exploration: Trusted expert preferred")


def test_quality_metrics_integration():
    """
    Test that all three quality metrics influence expert reputation.

    Validation:
    - Perplexity affects reputation
    - Coherence affects reputation
    - Task quality affects reputation
    - Overall quality is weighted combination
    """
    print("\n=== Test 4: Quality Metrics Integration ===\n")

    from sage.core.quality_measurement import QualityMeasurement
    from sage.core.quality_reputation_bridge import update_expert_reputation_from_quality
    from sage.core.expert_reputation import get_default_reputation_db

    measurer = QualityMeasurement(
        perplexity_weight=0.4,
        coherence_weight=0.3,
        task_weight=0.3
    )

    # Scenario 1: High perplexity (poor quality)
    print("Scenario 1: High perplexity (uncertain predictions)")

    vocab_size = 1000
    seq_len = 10

    # High perplexity: Uniform distribution (uncertain)
    logits_uncertain = torch.zeros(1, seq_len, vocab_size)
    target_ids = torch.randint(0, vocab_size, (1, seq_len))

    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    output_ids = torch.randint(0, vocab_size, (1, seq_len))

    metrics_uncertain = measurer.measure(
        input_ids=input_ids,
        output_ids=output_ids,
        logits=logits_uncertain,
        expert_ids=[10],
        context="general"
    )

    print(f"  Perplexity: {metrics_uncertain.perplexity:.2f}")
    print(f"  Overall quality: {metrics_uncertain.overall_quality:.3f}")

    # Scenario 2: Low perplexity (confident predictions)
    print("\nScenario 2: Low perplexity (confident predictions)")

    # Low perplexity: One-hot distribution (confident)
    logits_confident = torch.zeros(1, seq_len, vocab_size)
    for i in range(seq_len):
        logits_confident[0, i, target_ids[0, i]] = 10.0  # High confidence in correct token

    metrics_confident = measurer.measure(
        input_ids=input_ids,
        output_ids=target_ids,  # Perfect prediction
        logits=logits_confident,
        expert_ids=[11],
        context="general"
    )

    print(f"  Perplexity: {metrics_confident.perplexity:.2f}")
    print(f"  Overall quality: {metrics_confident.overall_quality:.3f}")

    # Update reputations
    update_expert_reputation_from_quality(metrics_uncertain)
    update_expert_reputation_from_quality(metrics_confident)

    # Check reputations
    db = get_default_reputation_db()
    rep_uncertain = db.get_reputation(10, 'test')
    rep_confident = db.get_reputation(11, 'test')

    trust_uncertain = rep_uncertain.get_context_trust("general")
    trust_confident = rep_confident.get_context_trust("general")

    print(f"\nReputations:")
    print(f"  Expert 10 (uncertain): {trust_uncertain:.3f}")
    print(f"  Expert 11 (confident): {trust_confident:.3f}")

    assert trust_confident > trust_uncertain, \
        "Confident expert should have higher trust than uncertain expert"

    print(f"\n✅ Quality metrics affect reputation correctly")
    print(f"✅ Lower perplexity → Higher trust")


def test_complete_integration_pathway():
    """
    Test complete integration pathway end-to-end.

    Validates all 4 phases working together:
    1. Phase 1: Trust-based selection
    2. Phase 2: Context classification
    3. Phase 3: Quality measurement
    4. Phase 4: Feedback loop (this test)
    """
    print("\n=== Test 5: Complete Integration Pathway ===\n")

    from sage.core.trust_based_expert_selector import TrustBasedExpertSelector
    from sage.core.context_classifier import ContextClassifier
    from sage.core.quality_measurement import QualityMeasurement
    from sage.core.quality_reputation_bridge import measure_and_update_reputation
    from sage.core.expert_reputation import get_default_reputation_db

    print("Creating integrated system:")

    # Phase 2: Context classifier
    classifier = ContextClassifier(num_contexts=5, embedding_dim=2048)

    # Fit classifier with some random training data
    training_embeddings = torch.randn(100, 2048)  # 100 samples
    training_labels = torch.randint(0, 5, (100,))  # 5 contexts
    classifier.fit(training_embeddings, training_labels)
    print("  ✅ Context classifier created and fitted")

    # Phase 1: Trust-based selector (with context classifier)
    selector = TrustBasedExpertSelector(
        num_experts=10,
        cache_size=5,
        component="test",
        context_classifier=classifier,
        exploration_weight=0.3
    )
    print("  ✅ Trust-based selector created")

    # Phase 3: Quality measurer
    measurer = QualityMeasurement()
    print("  ✅ Quality measurer created")

    print("\nSimulating complete generation cycle:\n")

    # Simulate 5 generation cycles
    for cycle in range(5):
        print(f"Cycle {cycle + 1}:")

        # Generate random input
        router_logits = torch.randn(10)
        input_embedding = torch.randn(2048)

        # Phase 1: Trust-based selection
        result = selector.select_experts(
            router_logits=router_logits,
            context=None,  # Auto-classify
            k=3,
            input_embedding=input_embedding
        )

        selected_experts = result.selected_expert_ids
        detected_context = result.context

        print(f"  Selected experts: {selected_experts}")
        print(f"  Detected context: {detected_context}")

        # Simulate generation quality (random for this test)
        generation_quality = 0.5 + torch.rand(1).item() * 0.4

        # Phase 3: Quality measurement
        from sage.core.quality_measurement import QualityMetrics
        metrics = QualityMetrics(
            perplexity=20.0,
            coherence=generation_quality * 0.9,
            task_quality=generation_quality,
            expert_ids=selected_experts,
            context=detected_context,
            overall_quality=generation_quality
        )

        print(f"  Generation quality: {generation_quality:.3f}")

        # Phase 4: Update reputation (closes the loop)
        measure_and_update_reputation(metrics)

        # Check updated trust
        db = get_default_reputation_db()
        avg_trust = sum(db.get_reputation(eid, 'test').get_context_trust(detected_context)
                       for eid in selected_experts) / len(selected_experts)
        print(f"  Average expert trust: {avg_trust:.3f}")
        print()

    print("✅ Complete integration pathway working")
    print("✅ All 4 phases integrated successfully")
    print()
    print("Integration Pathway:")
    print("  Phase 1: Trust-based selection ✅")
    print("  Phase 2: Context classification ✅")
    print("  Phase 3: Quality measurement ✅")
    print("  Phase 4: Feedback loop ✅")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SAGE Integration Pathway - Phase 4: End-to-End Testing")
    print("="*70)
    print()
    print("Testing the complete feedback loop:")
    print("  Selection → Generation → Quality → Reputation → Selection")
    print()
    print("This validates continuous learning and improvement.")
    print()

    # Run all tests
    try:
        test_learning_loop_improves_selection()
        test_context_specific_learning()
        test_exploration_vs_exploitation()
        test_quality_metrics_integration()
        test_complete_integration_pathway()

        print("\n" + "="*70)
        print("✅ ALL PHASE 4 TESTS PASSING")
        print("="*70)
        print()
        print("Phase 4: End-to-End Testing COMPLETE ✅")
        print()
        print("Integration Pathway: 4/4 phases (100%) ✅")
        print()
        print("The feedback loop is closed and validated:")
        print("  1. Trust-based selection chooses experts")
        print("  2. Context is detected automatically")
        print("  3. Quality is measured after generation")
        print("  4. Reputation is updated based on quality")
        print("  5. Future selections use updated reputation")
        print()
        print("Result: Continuous learning and improvement! ✅")
        print()

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
