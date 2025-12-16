#!/usr/bin/env python3
"""
Quality-Reputation Bridge: Connects quality measurement to expert reputation

Implements the feedback loop for Phase 3:
1. Measure generation quality
2. Update expert reputation based on quality
3. Trust-based selection uses updated reputation

This closes the learning loop for contextual expert selection.

Session Context: Thor Session 60 (Autonomous)
"""

from typing import List, Optional
from sage.core.quality_measurement import QualityMetrics
from sage.core.expert_reputation import (
    record_expert_activation,
    record_expert_co_activation,
    ExpertReputationDB,
    get_default_reputation_db
)


def update_expert_reputation_from_quality(
    metrics: QualityMetrics,
    db: Optional[ExpertReputationDB] = None
):
    """
    Update expert reputation based on quality measurement.

    This is the feedback loop that enables learning:
    1. Generation happens with certain experts
    2. Quality is measured
    3. Expert reputation updated based on quality
    4. Future selections use updated reputation

    Args:
        metrics: QualityMetrics from quality measurement
        db: Expert reputation database (uses default if None)
    """
    if db is None:
        db = get_default_reputation_db()

    # Record activation for each expert used
    performance = {
        'quality': metrics.overall_quality,
        'perplexity': metrics.perplexity,
        'coherence': metrics.coherence,
        'task_quality': metrics.task_quality,
    }

    for expert_id in metrics.expert_ids:
        record_expert_activation(
            expert_id=expert_id,
            context=metrics.context,
            performance=performance,
            db=db
        )

    # Record co-activation (experts working together)
    if len(metrics.expert_ids) > 1:
        record_expert_co_activation(
            expert_ids=metrics.expert_ids,
            combined_quality=metrics.overall_quality,
            db=db
        )


def measure_and_update_reputation(
    quality_metrics: QualityMetrics,
    db: Optional[ExpertReputationDB] = None
) -> None:
    """
    Convenience function: measure quality and update reputation in one call.

    Args:
        quality_metrics: Measured quality metrics
        db: Reputation database
    """
    update_expert_reputation_from_quality(quality_metrics, db)


# Example usage in generation loop:
"""
# In SelectiveLanguageModel.generate() or similar:

from sage.core.quality_measurement import measure_generation_quality
from sage.core.quality_reputation_bridge import update_expert_reputation_from_quality

# Generate
logits = model(input_ids, debug=False)
output_ids = torch.argmax(logits, dim=-1)

# Measure quality
metrics = measure_generation_quality(
    input_ids=input_ids,
    output_ids=output_ids,
    logits=logits,
    expert_ids=expert_ids_used,  # From generation
    context=context  # From context classification
)

# Update expert reputation
update_expert_reputation_from_quality(metrics, db=reputation_db)

# Next generation will use updated reputation for trust-based selection!
"""
