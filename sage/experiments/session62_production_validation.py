#!/usr/bin/env python3
"""
Session 62: Production Validation - Trust-Based Selection with Q3-Omni

Goal: Empirically validate that trust-based expert selection improves
      generation quality with actual Q3-Omni weights.

Method:
1. Baseline: Standard router-only selection
2. Trust-augmented: Router + trust-based selection (α=0.3)
3. Measure: Perplexity, coherence, quality metrics
4. Compare: Quality improvement over time
5. Visualize: Trust evolution, expert specialization

Expected Outcome:
- Trust-augmented shows quality improvement over time
- Certain experts emerge as specialists for different contexts
- Optimal exploration_weight identified
"""

import sys
import os
import torch
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.compression.selective_language_model import SelectiveLanguageModel
from sage.core.trust_based_expert_selector import TrustBasedExpertSelector
from sage.core.context_classifier import ContextClassifier
from sage.core.quality_measurement import QualityMeasurement
from sage.core.quality_reputation_bridge import update_expert_reputation_from_quality, QualityMetrics
from sage.core.expert_reputation import get_default_reputation_db


def setup_extraction_dir():
    """Get Q3-Omni extraction directory."""
    extraction_dir = Path.home() / "ai-workspace/HRM/model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"

    if not extraction_dir.exists():
        raise RuntimeError(f"Extraction directory not found: {extraction_dir}")

    # Verify key components exist
    required = ["experts", "routers", "embeddings", "lm_head", "attention"]
    for comp in required:
        comp_dir = extraction_dir / comp
        if not comp_dir.exists():
            raise RuntimeError(f"Required component missing: {comp}")

    return str(extraction_dir)


def create_test_prompts():
    """Create diverse test prompts for different contexts."""
    return [
        # Code context
        {
            "text": "def fibonacci(n):",
            "context": "code",
            "expected": "implementation of fibonacci sequence"
        },
        {
            "text": "class DataProcessor:",
            "context": "code",
            "expected": "class definition with methods"
        },

        # Text/reasoning context
        {
            "text": "The key insight of quantum mechanics is",
            "context": "reasoning",
            "expected": "explanation of quantum principles"
        },
        {
            "text": "In summary, the main argument suggests that",
            "context": "reasoning",
            "expected": "logical conclusion"
        },

        # General context
        {
            "text": "Once upon a time in",
            "context": "text",
            "expected": "narrative continuation"
        },
        {
            "text": "The weather today is",
            "context": "text",
            "expected": "descriptive text"
        },
    ]


def run_baseline_validation(extraction_dir, num_generations=10):
    """
    Run baseline validation with standard router-only selection.

    This establishes the quality baseline without trust-based selection.
    """
    print("\n" + "="*70)
    print("BASELINE: Router-Only Selection")
    print("="*70)

    # Create model without trust selector
    print("\nInitializing model (1 layer for speed)...")
    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=1,  # Start with 1 layer for faster iteration
        max_loaded_experts=16,
        device="cpu",  # Use CPU for now (GPU would be faster)
        trust_selector=None  # NO trust-based selection
    )

    # Create quality measurer
    measurer = QualityMeasurement()

    # Test prompts
    prompts = create_test_prompts()

    baseline_qualities = []

    print(f"\nRunning {num_generations} generations (baseline)...\n")

    for gen in range(num_generations):
        # Select random prompt
        prompt_data = prompts[gen % len(prompts)]
        prompt_text = prompt_data["text"]

        print(f"Generation {gen+1}/{num_generations}: '{prompt_text[:30]}...'")

        # Tokenize (simple: just use first N tokens for now)
        # In production, we'd use proper tokenizer
        input_ids = torch.randint(0, 152064, (1, 10))  # Simulate tokenized input

        try:
            # Forward pass
            with torch.no_grad():
                logits = model(input_ids, debug=(gen==0))

            # Generate continuation (greedy for now)
            output_ids = torch.argmax(logits, dim=-1)

            # Measure quality
            # Note: Without proper tokenizer, metrics will be rough estimates
            # But they'll still show relative differences
            quality = measurer.measure_perplexity(logits, output_ids)

            baseline_qualities.append(quality)

            print(f"  Perplexity: {quality:.2f}")

        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            baseline_qualities.append(float('inf'))

    avg_baseline = sum(q for q in baseline_qualities if q != float('inf')) / max(len([q for q in baseline_qualities if q != float('inf')]), 1)

    print(f"\nBaseline Results:")
    print(f"  Average perplexity: {avg_baseline:.2f}")
    print(f"  Generations completed: {len([q for q in baseline_qualities if q != float('inf')])}/{num_generations}")

    return baseline_qualities


def run_trust_augmented_validation(extraction_dir, num_generations=10, exploration_weight=0.3):
    """
    Run trust-augmented validation with trust-based expert selection.

    This tests whether trust-based selection improves quality over time.
    """
    print("\n" + "="*70)
    print(f"TRUST-AUGMENTED: Router + Trust (α={exploration_weight})")
    print("="*70)

    # Create context classifier
    print("\nCreating context classifier...")
    classifier = ContextClassifier(num_contexts=5, embedding_dim=2048)

    # Fit with synthetic training data for now
    # In production, we'd use real embeddings
    training_embeddings = torch.randn(100, 2048)
    training_labels = torch.randint(0, 5, (100,))
    classifier.fit(training_embeddings, training_labels)

    # Create trust selector
    print("Creating trust-based expert selector...")
    trust_selector = TrustBasedExpertSelector(
        num_experts=128,  # Q3-Omni has 128 experts per layer
        cache_size=16,    # Match max_loaded_experts
        component="thinker",
        context_classifier=classifier,
        exploration_weight=exploration_weight
    )

    # Create model WITH trust selector
    print("Initializing model with trust-based selection...")
    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=1,
        max_loaded_experts=16,
        device="cpu",
        trust_selector=trust_selector  # ENABLE trust-based selection
    )

    # Create quality measurer
    measurer = QualityMeasurement()

    # Test prompts
    prompts = create_test_prompts()

    trust_qualities = []
    trust_evolution = {0: [], 1: [], 2: []}  # Track trust for first 3 experts

    print(f"\nRunning {num_generations} generations (trust-augmented)...\n")

    for gen in range(num_generations):
        # Select random prompt
        prompt_data = prompts[gen % len(prompts)]
        prompt_text = prompt_data["text"]
        context = prompt_data["context"]

        print(f"Generation {gen+1}/{num_generations}: '{prompt_text[:30]}...' (context: {context})")

        # Tokenize
        input_ids = torch.randint(0, 152064, (1, 10))

        try:
            # Forward pass (with trust-based selection!)
            with torch.no_grad():
                logits = model(input_ids, debug=(gen==0))

            # Generate continuation
            output_ids = torch.argmax(logits, dim=-1)

            # Measure quality
            quality = measurer.measure_perplexity(logits, output_ids)

            # Get expert IDs used (from trust selector's last selection)
            # For now, simulate expert selection result
            expert_ids = [0, 1, 2, 3]  # These would come from actual selection

            # Create quality metrics
            metrics = QualityMetrics(
                perplexity=quality,
                coherence=0.5,  # Simplified for this test
                task_quality=0.6 if quality < 100 else 0.4,
                expert_ids=expert_ids,
                context=context,
                overall_quality=0.7 if quality < 100 else 0.5
            )

            # Update expert reputation
            update_expert_reputation_from_quality(metrics)

            trust_qualities.append(quality)

            # Track trust evolution for first 3 experts
            db = get_default_reputation_db()
            for expert_id in [0, 1, 2]:
                rep = db.get_reputation(expert_id, "thinker")
                if rep:
                    trust = rep.get_context_trust(context)
                    trust_evolution[expert_id].append(trust)

            print(f"  Perplexity: {quality:.2f}")
            print(f"  Expert 0 trust: {trust_evolution[0][-1]:.3f}" if trust_evolution[0] else "")

        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            trust_qualities.append(float('inf'))

    avg_trust = sum(q for q in trust_qualities if q != float('inf')) / max(len([q for q in trust_qualities if q != float('inf')]), 1)

    print(f"\nTrust-Augmented Results:")
    print(f"  Average perplexity: {avg_trust:.2f}")
    print(f"  Generations completed: {len([q for q in trust_qualities if q != float('inf')])}/{num_generations}")

    # Show trust evolution
    print(f"\nTrust Evolution:")
    for expert_id in [0, 1, 2]:
        if trust_evolution[expert_id]:
            print(f"  Expert {expert_id}: {trust_evolution[expert_id][0]:.3f} → {trust_evolution[expert_id][-1]:.3f}")

    return trust_qualities, trust_evolution


def compare_results(baseline_qualities, trust_qualities):
    """Compare baseline vs trust-augmented results."""
    print("\n" + "="*70)
    print("COMPARISON: Baseline vs Trust-Augmented")
    print("="*70)

    # Filter out errors
    baseline_valid = [q for q in baseline_qualities if q != float('inf')]
    trust_valid = [q for q in trust_qualities if q != float('inf')]

    if not baseline_valid or not trust_valid:
        print("\n⚠️  Insufficient valid generations for comparison")
        return

    baseline_avg = sum(baseline_valid) / len(baseline_valid)
    trust_avg = sum(trust_valid) / len(trust_valid)

    improvement = ((baseline_avg - trust_avg) / baseline_avg) * 100

    print(f"\nResults:")
    print(f"  Baseline average perplexity: {baseline_avg:.2f}")
    print(f"  Trust-augmented average perplexity: {trust_avg:.2f}")
    print(f"  Improvement: {improvement:+.1f}%")

    if improvement > 0:
        print(f"\n✅ Trust-based selection IMPROVED quality by {improvement:.1f}%")
    elif improvement > -5:
        print(f"\n➡️  Trust-based selection MAINTAINED quality (within 5%)")
    else:
        print(f"\n❌ Trust-based selection DECREASED quality by {abs(improvement):.1f}%")

    # Quality over time
    print(f"\nQuality Trend (Trust-Augmented):")
    if len(trust_valid) >= 5:
        early = sum(trust_valid[:3]) / 3
        late = sum(trust_valid[-3:]) / 3
        trend = ((early - late) / early) * 100
        print(f"  Early (1-3): {early:.2f}")
        print(f"  Late ({len(trust_valid)-2}-{len(trust_valid)}): {late:.2f}")
        print(f"  Improvement: {trend:+.1f}%")

        if trend > 0:
            print(f"  ✅ Quality IMPROVED over time (learning effect!)")
        else:
            print(f"  ➡️  Quality stable (may need more generations)")


def main():
    """Run complete production validation experiment."""
    print("\n" + "="*70)
    print("Session 62: Production Validation with Q3-Omni")
    print("="*70)
    print("\nGoal: Validate trust-based selection improves generation quality")
    print("Method: Compare baseline (router-only) vs trust-augmented")
    print()

    # Setup
    try:
        extraction_dir = setup_extraction_dir()
        print(f"✅ Q3-Omni extraction found: {extraction_dir}")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return

    # Configuration
    num_generations = 10  # Start small for testing
    exploration_weight = 0.3  # 30% exploration, 70% exploitation

    print(f"\nConfiguration:")
    print(f"  Generations per test: {num_generations}")
    print(f"  Exploration weight (α): {exploration_weight}")
    print(f"  Model layers: 1 (for speed)")
    print(f"  Device: CPU (GPU would be faster)")

    # Run experiments
    try:
        # Baseline
        baseline_qualities = run_baseline_validation(
            extraction_dir,
            num_generations=num_generations
        )

        # Trust-augmented
        trust_qualities, trust_evolution = run_trust_augmented_validation(
            extraction_dir,
            num_generations=num_generations,
            exploration_weight=exploration_weight
        )

        # Compare
        compare_results(baseline_qualities, trust_qualities)

        print("\n" + "="*70)
        print("✅ Production validation complete!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
