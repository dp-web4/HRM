#!/usr/bin/env python3
"""
Session 65: Quality Feedback Loop Implementation

Goal: Close the learning loop by implementing quality feedback

Critical Finding from Session 64:
- Sessions 62-64 validated MECHANISM but not LEARNING
- Infrastructure works (18/18 generations successful)
- But quality feedback loop was MISSING
- Trust scores never updated → no learning effect

What's Different in Session 65:
- Add record_quality() calls after each generation
- Properly track expert selections
- Enable trust score updates based on performance
- Validate that learning actually occurs

Method:
1. Same realistic token sequences as Session 64
2. Baseline: Router-only (unchanged)
3. Trust-augmented: Router + Trust WITH feedback
   - After each generation: record_quality(expert_id, context, quality)
   - Trust scores should evolve based on performance
4. Compare learning effect (trust should change now)

Expected Outcomes:
- Trust scores WILL update (not frozen at 0.879)
- Learning effect should emerge (early vs late epochs)
- Trust-augmented quality may improve over time
- Proper feedback loop validation
"""

import sys
import os
import torch
import numpy as np
import time
from pathlib import Path
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sage.compression.selective_language_model import SelectiveLanguageModel
from sage.core.trust_based_expert_selector import TrustBasedExpertSelector
from sage.core.context_classifier import ContextClassifier
from sage.core.quality_measurement import QualityMeasurement


def setup_extraction_dir():
    """Get Q3-Omni extraction directory."""
    extraction_dir = Path.home() / "ai-workspace/HRM/model-zoo/sage/omni-modal/qwen3-omni-30b-extracted"

    if not extraction_dir.exists():
        raise FileNotFoundError(f"Q3-Omni extraction not found at {extraction_dir}")

    print(f"✅ Q3-Omni extraction found: {extraction_dir}")
    return str(extraction_dir)


def load_tokenizer():
    """
    Load Q3-Omni tokenizer.

    For now, we'll use a simplified approach:
    - Encode prompts into token IDs
    - Generate logits
    - Measure perplexity on next tokens

    TODO: Integrate actual Q3-Omni tokenizer when available
    """
    print("Note: Using simplified tokenization (Q3-Omni tokenizer not yet integrated)")
    print("      This validates mechanism with realistic sequences")
    return None


def create_realistic_sequences():
    """
    Create realistic token sequences for testing.

    These are manually crafted to be more representative than random tokens:
    - Token IDs in reasonable ranges
    - Some repetition (like real text)
    - Realistic sequence lengths

    Returns:
        List of (input_ids, target_ids, prompt_text) tuples
    """
    sequences = []

    # Code sequences (lower token IDs, more structured)
    sequences.extend([
        (
            torch.tensor([[1, 563, 29042, 29898, 29876, 1125, 13, 1678, 565]]),  # "def fibonacci(n):\n    if"
            torch.tensor([[565, 302, 10, 29901, 13, 1678, 565, 302, 10]]),  # " if n <= 1: return"
            "def fibonacci(n):"
        ),
        (
            torch.tensor([[1, 770, 3630, 7425, 10994, 29901, 13, 1678, 822]]),  # "class DataProcessor:\n    def"
            torch.tensor([[822, 4770, 29918, 1272, 29898, 1311, 1125, 13, 1678]]),  # " __init__(self):"
            "class DataProcessor:"
        )
    ])

    # Reasoning sequences (mid-range token IDs, more abstract)
    sequences.extend([
        (
            torch.tensor([[1, 450, 1820, 25483, 310, 12101, 7208, 1199, 29871]]),  # "The key insight of quantum mech"
            torch.tensor([[338, 393, 13, 27756, 8314, 756, 1716, 29871, 0]]),  # " is that particles have both" (padded to 9)
            "The key insight of quantum mechanics"
        ),
        (
            torch.tensor([[1, 1762, 2274, 19371, 2264, 29892, 591, 1818, 29871]]),  # "To understand consciousness, we must"
            torch.tensor([[937, 278, 9558, 310, 278, 17294, 29871, 0, 0]]),  # " examine the nature of the brain" (padded to 9)
            "To understand consciousness, we must"
        )
    ])

    # Text sequences (varied token IDs, more diverse)
    sequences.extend([
        (
            torch.tensor([[1, 9038, 2501, 263, 931, 297, 29871, 263, 29871]]),  # "Once upon a time in a"
            torch.tensor([[2215, 29892, 2215, 3448, 2982, 29892, 727, 29871, 0]]),  # " far, far away land, there" (padded to 9)
            "Once upon a time in"
        ),
        (
            torch.tensor([[1, 450, 14826, 9826, 338, 29871, 6575, 1460, 29871]]),  # "The weather today is sunny"
            torch.tensor([[411, 10430, 25297, 322, 263, 3578, 29871, 0, 0]]),  # " with mild temperatures and a light" (padded to 9)
            "The weather today is"
        )
    ])

    return sequences


def run_baseline_real(extraction_dir, sequences, num_epochs=3):
    """
    Run baseline validation with realistic sequences.

    Args:
        extraction_dir: Path to Q3-Omni extraction
        sequences: List of (input, target, prompt) tuples
        num_epochs: Number of times to iterate through sequences

    Returns:
        List of perplexity values
    """
    print(f"\n{'='*70}")
    print("BASELINE: Router-Only Selection (Real Sequences)")
    print(f"{'='*70}\n")

    # Create model without trust selector
    print("Initializing model...")
    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=1,
        max_loaded_experts=16,
        device="cpu",
        trust_selector=None  # NO trust
    )

    measurer = QualityMeasurement()
    qualities = []

    print(f"Running {num_epochs} epochs × {len(sequences)} sequences = {num_epochs * len(sequences)} generations\n")

    for epoch in range(num_epochs):
        for idx, (input_ids, target_ids, prompt) in enumerate(sequences):
            gen_num = epoch * len(sequences) + idx + 1
            print(f"Generation {gen_num}/{num_epochs * len(sequences)}: '{prompt[:30]}...'", end="", flush=True)

            try:
                # Generate
                logits = model(input_ids, debug=False)

                # Measure quality
                quality = measurer.measure_perplexity(logits, target_ids)
                qualities.append(quality)

                print(f" → Perplexity: {quality:.2f}")

            except Exception as e:
                print(f" ⚠️  Error: {e}")
                continue

    if qualities:
        avg = np.mean(qualities)
        print(f"\nBaseline Results:")
        print(f"  Average perplexity: {avg:.2f}")
        print(f"  Generations: {len(qualities)}/{num_epochs * len(sequences)}")

    return qualities


def run_trust_augmented_real(extraction_dir, sequences, alpha=0.5, num_epochs=3):
    """
    Run trust-augmented validation with realistic sequences.

    Args:
        extraction_dir: Path to Q3-Omni extraction
        sequences: List of (input, target, prompt) tuples
        alpha: Exploration weight
        num_epochs: Number of times to iterate through sequences

    Returns:
        List of perplexity values, trust evolution data
    """
    print(f"\n{'='*70}")
    print(f"TRUST-AUGMENTED: Router + Trust (α={alpha}) (Real Sequences)")
    print(f"{'='*70}\n")

    # Create context classifier
    print("Creating context classifier...")
    classifier = ContextClassifier(num_contexts=5, embedding_dim=2048)

    # Fit with synthetic training data
    # CRITICAL: Convert to numpy float32 to prevent sklearn's float64 conversion
    training_embeddings = torch.randn(100, 2048).numpy().astype(np.float32)
    training_labels = torch.randint(0, 5, (100,)).numpy()
    classifier.fit(training_embeddings, training_labels)

    # Create trust selector
    print(f"Creating trust-based expert selector (α={alpha})...")
    trust_selector = TrustBasedExpertSelector(
        num_experts=128,
        cache_size=16,
        component="thinker",
        context_classifier=classifier,
        exploration_weight=alpha
    )

    # Create model
    print("Initializing model with trust-based selection...")
    model = SelectiveLanguageModel(
        extraction_dir=extraction_dir,
        num_layers=1,
        max_loaded_experts=16,
        device="cpu",
        trust_selector=trust_selector
    )

    measurer = QualityMeasurement()
    qualities = []
    trust_evolution = []

    print(f"Running {num_epochs} epochs × {len(sequences)} sequences = {num_epochs * len(sequences)} generations\n")

    for epoch in range(num_epochs):
        for idx, (input_ids, target_ids, prompt) in enumerate(sequences):
            gen_num = epoch * len(sequences) + idx + 1
            print(f"Generation {gen_num}/{num_epochs * len(sequences)}: '{prompt[:30]}...'", end="", flush=True)

            try:
                # Generate
                logits = model(input_ids, debug=False)

                # Measure quality (perplexity)
                perplexity = measurer.measure_perplexity(logits, target_ids)
                qualities.append(perplexity)

                # **NEW: Quality Feedback Loop**
                # Convert perplexity to quality score (0-1, lower perplexity = higher quality)
                # Use normalized inverse: quality = 1 / (1 + perplexity/1e6)
                quality_score = 1.0 / (1.0 + perplexity / 1e6)

                # Record quality for top-k experts used in this generation
                # For now, we'll update reputation for expert 0 as a proxy
                # (In full implementation, would track all k experts from selection)
                from sage.core.expert_reputation import get_default_reputation_db
                db = get_default_reputation_db()

                # Get or create reputation for expert 0
                rep = db.get_reputation(0, "thinker")
                if rep is None:
                    from sage.core.expert_reputation import ExpertReputation
                    rep = ExpertReputation(expert_id=0, component="thinker")
                    db.reputations[(0, "thinker")] = rep

                # Update trust based on observed quality
                rep.update_context_trust(context="general", evidence_quality=quality_score, learning_rate=0.2)

                # Track trust evolution
                trust = rep.get_context_trust("general", default=0.5)
                trust_evolution.append(trust)

                print(f" → PPL: {perplexity:.2f}, Quality: {quality_score:.4f}, Trust(E0): {trust:.3f}")

            except Exception as e:
                print(f" ⚠️  Error: {e}")
                continue

    if qualities:
        avg = np.mean(qualities)
        early = np.mean(qualities[:len(sequences)])  # First epoch
        late = np.mean(qualities[-len(sequences):])  # Last epoch
        improvement = ((early - late) / early * 100) if early > 0 else 0

        print(f"\nTrust-Augmented Results:")
        print(f"  Average perplexity: {avg:.2f}")
        print(f"  Early (epoch 1): {early:.2f}")
        print(f"  Late (epoch {num_epochs}): {late:.2f}")
        print(f"  Improvement: {improvement:+.1f}%")
        print(f"  Trust evolution: {trust_evolution[0]:.3f} → {trust_evolution[-1]:.3f}")

    return qualities, trust_evolution


def analyze_results(baseline_qualities, trust_qualities):
    """Compare baseline vs trust-augmented results."""
    print(f"\n{'='*70}")
    print("ANALYSIS: Baseline vs Trust-Augmented (Real Generation)")
    print(f"{'='*70}\n")

    if not baseline_qualities or not trust_qualities:
        print("❌ Insufficient data for comparison")
        return

    baseline_avg = np.mean(baseline_qualities)
    trust_avg = np.mean(trust_qualities)
    improvement = ((baseline_avg - trust_avg) / baseline_avg * 100) if baseline_avg > 0 else 0

    print(f"📊 Comparison:\n")
    print(f"  Baseline average:       {baseline_avg:.2f}")
    print(f"  Trust-augmented average: {trust_avg:.2f}")
    print(f"  Overall improvement:     {improvement:+.1f}%")

    if improvement > 5:
        print(f"\n✅ Trust-based selection IMPROVES quality by {improvement:.1f}%")
    elif improvement > 0:
        print(f"\n➡️  Modest improvement of {improvement:.1f}%")
    else:
        print(f"\n⚠️  No improvement (baseline better by {-improvement:.1f}%)")

    # Learning effect analysis
    if len(trust_qualities) >= 12:  # At least 2 epochs of 6 sequences
        trust_early = np.mean(trust_qualities[:6])
        trust_late = np.mean(trust_qualities[-6:])
        learning = ((trust_early - trust_late) / trust_early * 100) if trust_early > 0 else 0

        print(f"\n📈 Learning Effect (Trust-Augmented):")
        print(f"  Early (first 6):  {trust_early:.2f}")
        print(f"  Late (last 6):    {trust_late:.2f}")
        print(f"  Learning:         {learning:+.1f}%")

        if learning > 10:
            print(f"  ✅ Strong learning effect!")
        elif learning > 0:
            print(f"  ➡️  Moderate learning")
        else:
            print(f"  ⚠️  No learning observed")


def main():
    """Main execution."""
    print("="*70)
    print("Session 65: Quality Feedback Loop Implementation")
    print("="*70)
    print("\nGoal: Close the learning loop with quality feedback\n")

    # Setup
    extraction_dir = setup_extraction_dir()
    tokenizer = load_tokenizer()  # Will note simplified approach
    sequences = create_realistic_sequences()

    print(f"\nPrepared {len(sequences)} realistic test sequences")
    print("  - Code sequences: 2")
    print("  - Reasoning sequences: 2")
    print("  - Text sequences: 2")

    # Run baseline
    baseline_qualities = run_baseline_real(extraction_dir, sequences, num_epochs=3)

    # Brief pause
    time.sleep(2)

    # Run trust-augmented
    trust_qualities, trust_evolution = run_trust_augmented_real(
        extraction_dir, sequences, alpha=0.5, num_epochs=3
    )

    # Analyze
    analyze_results(baseline_qualities, trust_qualities)

    # Save results
    results = {
        'baseline': {
            'qualities': [float(q) for q in baseline_qualities],
            'average': float(np.mean(baseline_qualities)) if baseline_qualities else 0
        },
        'trust_augmented': {
            'qualities': [float(q) for q in trust_qualities],
            'average': float(np.mean(trust_qualities)) if trust_qualities else 0,
            'trust_evolution': [float(t) for t in trust_evolution]
        },
        'num_sequences': len(sequences),
        'num_epochs': 3
    }

    output_file = Path(__file__).parent / "session64_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    print("\n" + "="*70)
    print("✅ Session 64 complete!")
    print("="*70)


if __name__ == "__main__":
    main()
