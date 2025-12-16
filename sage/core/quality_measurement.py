#!/usr/bin/env python3
"""
Quality Measurement for Expert Reputation Updates

Measures generation quality to update expert reputation in the feedback loop.
Implements Phase 3 of the integration pathway.

Metrics:
- Perplexity: Model confidence in generation
- Coherence: N-gram overlap and semantic consistency
- Task-specific: Depends on generation type (code, text, reasoning)

Session Context: Thor Session 60 (Autonomous)
Building on:
  - Session 56 (Legion): TrustBasedExpertSelector
  - Session 57 (Legion): ContextClassifier
  - Session 58 (Thor): ContextClassifier integration
  - Session 59 (Thor): Phase 1 integration
  - Session 60 (Thor): Phase 3 quality measurement ← This session
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import math


@dataclass
class QualityMetrics:
    """
    Quality metrics for a generation.

    Contains all measured quality scores that can be used to update
    expert reputation.
    """
    perplexity: float              # Model confidence (lower is better)
    coherence: float               # Semantic consistency (0-1, higher is better)
    task_quality: float            # Task-specific quality (0-1, higher is better)
    expert_ids: List[int]          # Experts used in generation
    context: str                   # Context classification
    overall_quality: float         # Combined quality score (0-1)

    # Optional metadata
    sequence_length: int = 0
    generation_time: float = 0.0
    num_experts_used: int = 0


class QualityMeasurement:
    """
    Measures generation quality for expert reputation updates.

    Implements Phase 3 of integration pathway:
    - Measures perplexity, coherence, task-specific quality
    - Computes overall quality score
    - Returns metrics suitable for reputation updates

    Usage:
        measurer = QualityMeasurement()
        metrics = measurer.measure(
            input_ids=input_ids,
            output_ids=output_ids,
            logits=logits,
            expert_ids=expert_ids,
            context="code"
        )
        # Use metrics to update expert reputation
    """

    def __init__(
        self,
        perplexity_weight: float = 0.4,
        coherence_weight: float = 0.3,
        task_weight: float = 0.3,
    ):
        """
        Initialize quality measurement.

        Args:
            perplexity_weight: Weight for perplexity in overall score
            coherence_weight: Weight for coherence in overall score
            task_weight: Weight for task-specific quality
        """
        self.perplexity_weight = perplexity_weight
        self.coherence_weight = coherence_weight
        self.task_weight = task_weight

    def measure(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor,
        logits: torch.Tensor,
        expert_ids: List[int],
        context: str,
        target_ids: Optional[torch.Tensor] = None,
    ) -> QualityMetrics:
        """
        Measure generation quality.

        Args:
            input_ids: Input token IDs [batch, seq_in]
            output_ids: Generated token IDs [batch, seq_out]
            logits: Model output logits [batch, seq_out, vocab]
            expert_ids: Expert IDs used in generation
            context: Context classification
            target_ids: Optional ground truth for supervised quality

        Returns:
            QualityMetrics with all measurements
        """
        # Measure perplexity
        perplexity = self.measure_perplexity(logits, output_ids)

        # Measure coherence
        coherence = self.measure_coherence(input_ids, output_ids)

        # Measure task-specific quality
        task_quality = self.measure_task_quality(
            input_ids, output_ids, context, target_ids
        )

        # Compute overall quality (0-1, higher is better)
        # Note: perplexity is inverted (lower is better)
        perplexity_score = self._normalize_perplexity(perplexity)

        overall_quality = (
            self.perplexity_weight * perplexity_score +
            self.coherence_weight * coherence +
            self.task_weight * task_quality
        )

        return QualityMetrics(
            perplexity=perplexity,
            coherence=coherence,
            task_quality=task_quality,
            expert_ids=expert_ids,
            context=context,
            overall_quality=overall_quality,
            sequence_length=output_ids.shape[1],
            num_experts_used=len(set(expert_ids)),
        )

    def measure_perplexity(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:
        """
        Measure perplexity (model confidence).

        Lower perplexity = higher confidence = better quality

        Args:
            logits: Model output [batch, seq, vocab]
            target_ids: Target token IDs [batch, seq]

        Returns:
            Perplexity score
        """
        # Compute cross-entropy loss
        batch_size, seq_length, vocab_size = logits.shape

        # Reshape for cross entropy
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = target_ids.view(-1)

        # Cross entropy (mean over tokens)
        loss = torch.nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            reduction='mean'
        )

        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()

        return perplexity

    def measure_coherence(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> float:
        """
        Measure coherence (semantic consistency).

        Measures n-gram overlap between input and output.
        Higher overlap = better coherence (for continuation tasks).

        Args:
            input_ids: Input tokens [batch, seq_in]
            output_ids: Output tokens [batch, seq_out]

        Returns:
            Coherence score (0-1, higher is better)
        """
        # Simple coherence: bigram overlap
        # More sophisticated: could use sentence embeddings

        # Flatten to 1D
        input_tokens = input_ids.flatten().tolist()
        output_tokens = output_ids.flatten().tolist()

        # Extract bigrams
        input_bigrams = set(
            tuple(input_tokens[i:i+2])
            for i in range(len(input_tokens) - 1)
        )

        output_bigrams = set(
            tuple(output_tokens[i:i+2])
            for i in range(len(output_tokens) - 1)
        )

        # Compute overlap
        if len(output_bigrams) == 0:
            return 0.0

        overlap = len(input_bigrams & output_bigrams)
        coherence = overlap / max(len(output_bigrams), 1)

        # Normalize to 0-1 range (empirical: 0.1-0.5 is typical)
        coherence = min(coherence * 2.0, 1.0)

        return coherence

    def measure_task_quality(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor,
        context: str,
        target_ids: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Measure task-specific quality.

        Different quality measures for different contexts:
        - code: Syntax validity (simplified)
        - text: Length and diversity
        - reasoning: Logical structure (simplified)

        Args:
            input_ids: Input tokens
            output_ids: Output tokens
            context: Context classification
            target_ids: Optional ground truth

        Returns:
            Task quality score (0-1, higher is better)
        """
        # If we have ground truth, use exact match
        if target_ids is not None:
            matches = (output_ids == target_ids).float().mean().item()
            return matches

        # Otherwise, use heuristic based on context
        seq_length = output_ids.shape[1]

        if "code" in context.lower():
            # Code quality: check for basic structure
            # (This is simplified - real code quality needs AST parsing)
            return self._measure_code_quality(output_ids)

        elif "text" in context.lower() or "general" in context.lower():
            # Text quality: length and diversity
            return self._measure_text_quality(output_ids)

        elif "reasoning" in context.lower():
            # Reasoning quality: logical structure
            return self._measure_reasoning_quality(output_ids)

        else:
            # Default: based on length (longer = better, up to a point)
            ideal_length = 50  # tokens
            length_score = 1.0 - abs(seq_length - ideal_length) / ideal_length
            return max(0.0, min(1.0, length_score))

    def _measure_code_quality(self, output_ids: torch.Tensor) -> float:
        """
        Simplified code quality measurement.

        In production: Would use AST parsing, syntax checking, etc.
        Here: Basic heuristics.
        """
        # Heuristic: reasonable length (10-100 tokens)
        seq_length = output_ids.shape[1]

        if seq_length < 10:
            return 0.3  # Too short
        elif seq_length > 100:
            return 0.5  # Too long
        else:
            return 0.7  # Reasonable length

    def _measure_text_quality(self, output_ids: torch.Tensor) -> float:
        """
        Simplified text quality measurement.

        Measures length and token diversity.
        """
        tokens = output_ids.flatten().tolist()
        seq_length = len(tokens)

        # Diversity: unique tokens / total tokens
        unique_tokens = len(set(tokens))
        diversity = unique_tokens / max(seq_length, 1)

        # Length score (prefer 20-80 tokens)
        if seq_length < 20:
            length_score = seq_length / 20
        elif seq_length > 80:
            length_score = 1.0 - (seq_length - 80) / 100
        else:
            length_score = 1.0

        # Combined
        quality = 0.6 * length_score + 0.4 * diversity
        return min(1.0, max(0.0, quality))

    def _measure_reasoning_quality(self, output_ids: torch.Tensor) -> float:
        """
        Simplified reasoning quality measurement.

        In production: Would check logical structure, premise-conclusion, etc.
        Here: Basic heuristics.
        """
        seq_length = output_ids.shape[1]

        # Heuristic: reasoning needs moderate length (15-60 tokens)
        if seq_length < 15:
            return 0.4  # Too short for reasoning
        elif seq_length > 60:
            return 0.6  # Might be rambling
        else:
            return 0.8  # Good length for reasoning

    def _normalize_perplexity(self, perplexity: float) -> float:
        """
        Normalize perplexity to 0-1 score (higher is better).

        Typical perplexity ranges:
        - Excellent: 1-10
        - Good: 10-50
        - Fair: 50-100
        - Poor: 100+

        Args:
            perplexity: Perplexity value (lower is better)

        Returns:
            Normalized score 0-1 (higher is better)
        """
        # Use sigmoid-like transformation
        # score = 1 / (1 + perplexity/10)
        # This maps:
        #   perplexity=10 → score=0.5
        #   perplexity=1 → score≈0.9
        #   perplexity=100 → score≈0.09

        score = 1.0 / (1.0 + perplexity / 10.0)
        return score


# Convenience function
def measure_generation_quality(
    input_ids: torch.Tensor,
    output_ids: torch.Tensor,
    logits: torch.Tensor,
    expert_ids: List[int],
    context: str,
    target_ids: Optional[torch.Tensor] = None,
) -> QualityMetrics:
    """
    Convenience function to measure generation quality.

    Args:
        input_ids: Input token IDs
        output_ids: Generated token IDs
        logits: Model output logits
        expert_ids: Expert IDs used
        context: Context classification
        target_ids: Optional ground truth

    Returns:
        QualityMetrics object
    """
    measurer = QualityMeasurement()
    return measurer.measure(
        input_ids=input_ids,
        output_ids=output_ids,
        logits=logits,
        expert_ids=expert_ids,
        context=context,
        target_ids=target_ids,
    )
