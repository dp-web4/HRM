#!/usr/bin/env python3
"""
Tests for Quality Measurement System

Validates Phase 3 implementation: measuring generation quality for expert
reputation updates.

Session Context: Thor Session 60 (Autonomous)
"""

import sys
import os
import torch
import numpy as np

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from sage.core.quality_measurement import (
    QualityMeasurement,
    QualityMetrics,
    measure_generation_quality
)


def test_quality_measurement_basic():
    """
    Test basic quality measurement functionality.
    """
    print("\n=== Test: Quality Measurement Basic ===\n")

    # Create measurer
    measurer = QualityMeasurement()

    print(f"✅ QualityMeasurement created")
    print(f"   Perplexity weight: {measurer.perplexity_weight}")
    print(f"   Coherence weight: {measurer.coherence_weight}")
    print(f"   Task weight: {measurer.task_weight}")

    # Create synthetic data
    batch_size = 1
    seq_len_in = 10
    seq_len_out = 20
    vocab_size = 100

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len_in))
    output_ids = torch.randint(0, vocab_size, (batch_size, seq_len_out))
    logits = torch.randn(batch_size, seq_len_out, vocab_size)

    expert_ids = [5, 10, 15, 20]
    context = "general"

    # Measure quality
    metrics = measurer.measure(
        input_ids=input_ids,
        output_ids=output_ids,
        logits=logits,
        expert_ids=expert_ids,
        context=context
    )

    print(f"\n✅ Quality measured successfully")
    print(f"   Perplexity: {metrics.perplexity:.2f}")
    print(f"   Coherence: {metrics.coherence:.3f}")
    print(f"   Task quality: {metrics.task_quality:.3f}")
    print(f"   Overall quality: {metrics.overall_quality:.3f}")
    print(f"   Expert IDs: {metrics.expert_ids}")
    print(f"   Context: {metrics.context}")

    # Validate metrics are in expected ranges
    assert metrics.perplexity > 0, "Perplexity should be positive"
    assert 0 <= metrics.coherence <= 1, "Coherence should be 0-1"
    assert 0 <= metrics.task_quality <= 1, "Task quality should be 0-1"
    assert 0 <= metrics.overall_quality <= 1, "Overall quality should be 0-1"
    assert metrics.expert_ids == expert_ids
    assert metrics.context == context

    print(f"\n✅ All metrics in valid ranges")


def test_perplexity_measurement():
    """
    Test perplexity measurement specifically.
    """
    print("\n=== Test: Perplexity Measurement ===\n")

    measurer = QualityMeasurement()

    # Create data where model is confident (low perplexity)
    batch_size = 1
    seq_len = 10
    vocab_size = 100

    # Create logits with high confidence for correct tokens
    logits = torch.randn(batch_size, seq_len, vocab_size) * 0.1  # Low variance
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Set correct token logits very high
    for i in range(seq_len):
        logits[0, i, target_ids[0, i]] = 10.0  # High confidence

    perplexity = measurer.measure_perplexity(logits, target_ids)

    print(f"High confidence perplexity: {perplexity:.2f}")
    assert perplexity < 10, f"Expected low perplexity for confident model, got {perplexity}"

    # Create data where model is uncertain (high perplexity)
    logits_uncertain = torch.randn(batch_size, seq_len, vocab_size)  # High variance
    perplexity_uncertain = measurer.measure_perplexity(logits_uncertain, target_ids)

    print(f"Low confidence perplexity: {perplexity_uncertain:.2f}")
    assert perplexity_uncertain > perplexity, "Uncertain model should have higher perplexity"

    print(f"\n✅ Perplexity measurement working correctly")


def test_coherence_measurement():
    """
    Test coherence measurement (n-gram overlap).
    """
    print("\n=== Test: Coherence Measurement ===\n")

    measurer = QualityMeasurement()

    vocab_size = 100

    # High coherence: output continues input pattern
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output_ids = torch.tensor([[5, 6, 7, 8, 9, 10]])  # Overlapping bigrams

    coherence_high = measurer.measure_coherence(input_ids, output_ids)
    print(f"High coherence (overlapping pattern): {coherence_high:.3f}")

    # Low coherence: completely different tokens
    output_ids_low = torch.tensor([[50, 51, 52, 53, 54, 55]])  # No overlap

    coherence_low = measurer.measure_coherence(input_ids, output_ids_low)
    print(f"Low coherence (no overlap): {coherence_low:.3f}")

    assert coherence_high > coherence_low, "High overlap should have higher coherence"

    print(f"\n✅ Coherence measurement distinguishes patterns")


def test_task_specific_quality():
    """
    Test task-specific quality measurement for different contexts.
    """
    print("\n=== Test: Task-Specific Quality ===\n")

    measurer = QualityMeasurement()

    input_ids = torch.randint(0, 100, (1, 10))

    # Test code context (prefers moderate length)
    output_code = torch.randint(0, 100, (1, 30))  # Good length for code
    quality_code = measurer.measure_task_quality(
        input_ids, output_code, "context_code", None
    )
    print(f"Code quality (30 tokens): {quality_code:.3f}")

    # Test text context (prefers diversity and good length)
    output_text = torch.randint(0, 100, (1, 40))  # Good length for text
    quality_text = measurer.measure_task_quality(
        input_ids, output_text, "context_text", None
    )
    print(f"Text quality (40 tokens): {quality_text:.3f}")

    # Test reasoning context (prefers moderate length)
    output_reasoning = torch.randint(0, 100, (1, 25))  # Good length for reasoning
    quality_reasoning = measurer.measure_task_quality(
        input_ids, output_reasoning, "context_reasoning", None
    )
    print(f"Reasoning quality (25 tokens): {quality_reasoning:.3f}")

    # All should be > 0
    assert quality_code > 0
    assert quality_text > 0
    assert quality_reasoning > 0

    print(f"\n✅ Task-specific quality measurement working")


def test_overall_quality_combination():
    """
    Test that overall quality combines metrics correctly.
    """
    print("\n=== Test: Overall Quality Combination ===\n")

    # Create measurer with known weights
    measurer = QualityMeasurement(
        perplexity_weight=0.5,
        coherence_weight=0.3,
        task_weight=0.2,
    )

    # Create test data
    batch_size = 1
    seq_len = 20
    vocab_size = 100

    input_ids = torch.randint(0, vocab_size, (batch_size, 10))
    output_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Create logits with moderate confidence
    logits = torch.randn(batch_size, seq_len, vocab_size) * 2.0
    for i in range(seq_len):
        logits[0, i, output_ids[0, i]] = 5.0  # Moderate confidence

    metrics = measurer.measure(
        input_ids=input_ids,
        output_ids=output_ids,
        logits=logits,
        expert_ids=[1, 2, 3],
        context="general"
    )

    print(f"Individual metrics:")
    print(f"  Perplexity: {metrics.perplexity:.2f}")
    print(f"  Coherence: {metrics.coherence:.3f}")
    print(f"  Task quality: {metrics.task_quality:.3f}")
    print(f"\nCombined:")
    print(f"  Overall quality: {metrics.overall_quality:.3f}")

    # Overall should be weighted combination
    assert 0 <= metrics.overall_quality <= 1

    # Verify it's a reasonable combination
    perplexity_normalized = measurer._normalize_perplexity(metrics.perplexity)
    expected = (
        0.5 * perplexity_normalized +
        0.3 * metrics.coherence +
        0.2 * metrics.task_quality
    )

    assert abs(metrics.overall_quality - expected) < 0.01, \
        f"Expected {expected:.3f}, got {metrics.overall_quality:.3f}"

    print(f"\n✅ Overall quality combines metrics correctly")


def test_convenience_function():
    """
    Test the convenience function for measuring quality.
    """
    print("\n=== Test: Convenience Function ===\n")

    # Create test data
    batch_size = 1
    input_ids = torch.randint(0, 100, (batch_size, 10))
    output_ids = torch.randint(0, 100, (batch_size, 20))
    logits = torch.randn(batch_size, 20, 100)

    # Use convenience function
    metrics = measure_generation_quality(
        input_ids=input_ids,
        output_ids=output_ids,
        logits=logits,
        expert_ids=[5, 10, 15],
        context="code"
    )

    print(f"✅ Convenience function works")
    print(f"   Overall quality: {metrics.overall_quality:.3f}")
    print(f"   Expert IDs: {metrics.expert_ids}")
    print(f"   Context: {metrics.context}")

    assert isinstance(metrics, QualityMetrics)
    assert metrics.expert_ids == [5, 10, 15]
    assert metrics.context == "code"


def test_with_ground_truth():
    """
    Test quality measurement with ground truth (supervised).
    """
    print("\n=== Test: With Ground Truth ===\n")

    measurer = QualityMeasurement()

    batch_size = 1
    seq_len = 20
    vocab_size = 100

    input_ids = torch.randint(0, vocab_size, (batch_size, 10))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Perfect match
    output_perfect = target_ids.clone()
    logits = torch.randn(batch_size, seq_len, vocab_size)

    metrics_perfect = measurer.measure(
        input_ids=input_ids,
        output_ids=output_perfect,
        logits=logits,
        expert_ids=[1],
        context="general",
        target_ids=target_ids
    )

    print(f"Perfect match task quality: {metrics_perfect.task_quality:.3f}")
    assert metrics_perfect.task_quality == 1.0, "Perfect match should have quality 1.0"

    # Partial match (50% correct)
    output_partial = target_ids.clone()
    output_partial[0, :seq_len//2] = torch.randint(0, vocab_size, (seq_len//2,))

    metrics_partial = measurer.measure(
        input_ids=input_ids,
        output_ids=output_partial,
        logits=logits,
        expert_ids=[1],
        context="general",
        target_ids=target_ids
    )

    print(f"50% match task quality: {metrics_partial.task_quality:.3f}")
    assert 0.4 < metrics_partial.task_quality < 0.6, "50% match should be ~0.5"

    print(f"\n✅ Ground truth evaluation working")


if __name__ == "__main__":
    print("=" * 70)
    print("Quality Measurement Tests")
    print("=" * 70)

    test_quality_measurement_basic()
    test_perplexity_measurement()
    test_coherence_measurement()
    test_task_specific_quality()
    test_overall_quality_combination()
    test_convenience_function()
    test_with_ground_truth()

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSING")
    print("=" * 70)
    print("\nPhase 3 Quality Measurement: ✅ IMPLEMENTED")
    print("\nNext Steps:")
    print("  1. Integrate with expert reputation updates")
    print("  2. Test with actual Q3-Omni generation")
    print("  3. Implement Phase 4 (end-to-end testing)")
