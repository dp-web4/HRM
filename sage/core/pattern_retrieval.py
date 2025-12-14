#!/usr/bin/env python3
"""
Pattern Retrieval & Transfer Learning - Session 51

Enables consciousness to retrieve and apply consolidated patterns from
previous experiences to improve current reasoning and response quality.

Key Concepts:
- Pattern matching: Find relevant patterns for current context
- Similarity scoring: Rank patterns by relevance
- Pattern application: Use patterns to guide consciousness
- Learning measurement: Track quality improvement from patterns

Biological Parallel:
Just as biological consciousness uses sleep-consolidated memories during
waking cognition, SAGE retrieves DREAM-consolidated patterns to inform
current consciousness cycles.

Author: Thor (Autonomous Session 51)
Date: 2025-12-14
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import time
import numpy as np

from sage.core.dream_consolidation import (
    ConsolidatedMemory,
    MemoryPattern,
    QualityLearning,
    CreativeAssociation
)


@dataclass
class PatternMatch:
    """
    A pattern retrieved for current context with relevance score.

    Attributes:
        pattern: The retrieved memory pattern
        relevance: How relevant to current context (0-1)
        match_reason: Why this pattern was retrieved
        source_memory_id: Which consolidated memory contains this pattern
    """
    pattern: MemoryPattern
    relevance: float
    match_reason: str
    source_memory_id: str

    def to_dict(self) -> Dict:
        """Export to dictionary"""
        return {
            'pattern': self.pattern.to_dict(),
            'relevance': self.relevance,
            'match_reason': self.match_reason,
            'source_memory_id': self.source_memory_id
        }


@dataclass
class RetrievalContext:
    """
    Current context for pattern retrieval.

    Represents what we know about the current consciousness cycle
    to help match against consolidated patterns.

    Attributes:
        prompt: Current prompt/query
        task_salience: Current task salience (0-1)
        metabolic_state: Current metabolic state
        epistemic_state: Current epistemic state
        emotional_state: Current emotional state
        circadian_phase: Current circadian phase
    """
    prompt: str = ""
    task_salience: float = 0.5
    metabolic_state: str = "wake"
    epistemic_state: str = "exploring"
    emotional_state: Dict[str, float] = field(default_factory=dict)
    circadian_phase: str = "day"


@dataclass
class TransferLearningResult:
    """
    Result of applying transfer learning to current cycle.

    Contains retrieved patterns and guidance for consciousness.

    Attributes:
        retrieved_patterns: Top-k patterns retrieved
        quality_guidance: Quality learnings applicable to current context
        creative_insights: Creative associations relevant to current task
        application_summary: Human-readable summary of what was retrieved
        retrieval_count: Total patterns considered
        retrieval_time: Time spent retrieving (seconds)
    """
    retrieved_patterns: List[PatternMatch]
    quality_guidance: List[QualityLearning]
    creative_insights: List[CreativeAssociation]
    application_summary: str
    retrieval_count: int
    retrieval_time: float


class PatternRetriever:
    """
    Retrieves consolidated patterns relevant to current consciousness context.

    Provides transfer learning by matching current context against
    previously consolidated DREAM memories.

    Methods:
        retrieve_patterns: Find relevant patterns for current context
        rank_by_relevance: Score patterns by similarity to context
        apply_learnings: Apply quality learnings to current cycle
    """

    def __init__(self,
                 top_k: int = 5,
                 min_relevance: float = 0.3,
                 recency_weight: float = 0.2):
        """
        Initialize pattern retriever.

        Args:
            top_k: Maximum patterns to retrieve per query
            min_relevance: Minimum relevance threshold (0-1)
            recency_weight: Weight for recency in scoring (0-1)
        """
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.recency_weight = recency_weight

        # Statistics
        self.total_retrievals = 0
        self.successful_retrievals = 0
        self.average_retrieval_time = 0.0

    def retrieve_patterns(self,
                         consolidated_memories: List[ConsolidatedMemory],
                         context: RetrievalContext) -> TransferLearningResult:
        """
        Retrieve relevant patterns for current context.

        Searches through consolidated memories to find patterns that
        match the current consciousness state and task.

        Args:
            consolidated_memories: All available consolidated memories
            context: Current consciousness context

        Returns:
            TransferLearningResult with retrieved patterns and guidance
        """
        start_time = time.time()

        if not consolidated_memories:
            # No memories to retrieve from
            return TransferLearningResult(
                retrieved_patterns=[],
                quality_guidance=[],
                creative_insights=[],
                application_summary="No consolidated memories available",
                retrieval_count=0,
                retrieval_time=0.0
            )

        # 1. Extract all patterns from all memories
        all_patterns = []
        all_quality_learnings = []
        all_creative_associations = []

        for memory in consolidated_memories:
            for pattern in memory.patterns:
                all_patterns.append((pattern, memory.dream_session_id, memory.timestamp))
            all_quality_learnings.extend(memory.quality_learnings)
            all_creative_associations.extend(memory.creative_associations)

        # 2. Score each pattern for relevance to current context
        scored_patterns = []
        for pattern, memory_id, memory_time in all_patterns:
            relevance, reason = self._score_pattern_relevance(pattern, context, memory_time)

            if relevance >= self.min_relevance:
                match = PatternMatch(
                    pattern=pattern,
                    relevance=relevance,
                    match_reason=reason,
                    source_memory_id=memory_id
                )
                scored_patterns.append(match)

        # 3. Sort by relevance and take top-k
        scored_patterns.sort(key=lambda x: x.relevance, reverse=True)
        top_patterns = scored_patterns[:self.top_k]

        # 4. Filter quality learnings by confidence
        relevant_learnings = [
            learning for learning in all_quality_learnings
            if learning.confidence >= 0.6  # High confidence only
        ]

        # 5. Filter creative associations by strength
        relevant_insights = [
            assoc for assoc in all_creative_associations
            if assoc.strength >= 0.7  # Strong associations only
        ]

        # 6. Create application summary
        summary = self._create_application_summary(
            top_patterns,
            relevant_learnings,
            relevant_insights
        )

        # Statistics
        retrieval_time = time.time() - start_time
        self.total_retrievals += 1
        if top_patterns:
            self.successful_retrievals += 1
        self.average_retrieval_time = (
            (self.average_retrieval_time * (self.total_retrievals - 1) + retrieval_time)
            / self.total_retrievals
        )

        return TransferLearningResult(
            retrieved_patterns=top_patterns,
            quality_guidance=relevant_learnings,
            creative_insights=relevant_insights[:3],  # Top 3 insights
            application_summary=summary,
            retrieval_count=len(all_patterns),
            retrieval_time=retrieval_time
        )

    def _score_pattern_relevance(self,
                                 pattern: MemoryPattern,
                                 context: RetrievalContext,
                                 pattern_timestamp: float) -> Tuple[float, str]:
        """
        Score how relevant a pattern is to current context.

        Considers:
        - Pattern type matching (metabolic, epistemic, quality)
        - Description similarity to prompt
        - State compatibility
        - Recency (newer patterns weighted higher)

        Args:
            pattern: Pattern to score
            context: Current context
            pattern_timestamp: When pattern was created

        Returns:
            (relevance_score, match_reason) tuple
        """
        scores = []
        reasons = []

        # 1. Pattern type relevance
        type_score = 0.0
        if pattern.pattern_type == "metabolic" and context.metabolic_state:
            if context.metabolic_state.lower() in pattern.description.lower():
                type_score = 0.8
                reasons.append(f"metabolic_match({context.metabolic_state})")

        if pattern.pattern_type == "epistemic" and context.epistemic_state:
            if context.epistemic_state.lower() in pattern.description.lower():
                type_score = 0.8
                reasons.append(f"epistemic_match({context.epistemic_state})")

        if pattern.pattern_type == "quality":
            type_score = 0.6  # Always somewhat relevant
            reasons.append("quality_pattern")

        scores.append(type_score)

        # 2. Description similarity to prompt (simple keyword matching)
        desc_score = 0.0
        if context.prompt:
            # Extract key words from prompt (simple approach)
            prompt_words = set(context.prompt.lower().split())
            pattern_words = set(pattern.description.lower().split())

            # Jaccard similarity
            if prompt_words and pattern_words:
                intersection = len(prompt_words & pattern_words)
                union = len(prompt_words | pattern_words)
                desc_score = intersection / union if union > 0 else 0.0

                if desc_score > 0.1:
                    reasons.append(f"keyword_match({desc_score:.2f})")

        scores.append(desc_score)

        # 3. Pattern strength (confidence in pattern)
        scores.append(pattern.strength)
        if pattern.strength > 0.7:
            reasons.append(f"high_confidence({pattern.strength:.2f})")

        # 4. Recency bonus
        recency_score = 0.0
        if self.recency_weight > 0:
            # Newer patterns get bonus (exponential decay)
            time_since = time.time() - pattern_timestamp
            # Decay half-life: 1 hour (3600 seconds)
            recency_score = np.exp(-time_since / 3600.0) * self.recency_weight
            if recency_score > 0.1:
                reasons.append(f"recent({recency_score:.2f})")

        scores.append(recency_score)

        # Weighted combination
        relevance = np.mean(scores) if scores else 0.0
        match_reason = ", ".join(reasons) if reasons else "general_retrieval"

        return relevance, match_reason

    def _create_application_summary(self,
                                   patterns: List[PatternMatch],
                                   learnings: List[QualityLearning],
                                   insights: List[CreativeAssociation]) -> str:
        """Create human-readable summary of retrieved knowledge."""
        if not patterns and not learnings and not insights:
            return "No relevant patterns retrieved"

        summary_parts = []

        if patterns:
            summary_parts.append(
                f"Retrieved {len(patterns)} patterns: " +
                ", ".join([f"{p.pattern.pattern_type}({p.relevance:.2f})"
                          for p in patterns[:3]])
            )

        if learnings:
            positive = [l for l in learnings if l.positive_correlation]
            if positive:
                top_learning = max(positive, key=lambda l: l.confidence)
                summary_parts.append(
                    f"Quality guidance: {top_learning.characteristic} "
                    f"(+{(top_learning.average_quality_with - top_learning.average_quality_without):.2f} quality)"
                )

        if insights:
            summary_parts.append(f"{len(insights)} creative insights available")

        return " | ".join(summary_parts)

    def get_statistics(self) -> Dict:
        """Get retrieval statistics."""
        return {
            'total_retrievals': self.total_retrievals,
            'successful_retrievals': self.successful_retrievals,
            'success_rate': (
                self.successful_retrievals / self.total_retrievals
                if self.total_retrievals > 0 else 0.0
            ),
            'average_retrieval_time': self.average_retrieval_time
        }


def example_usage():
    """Example demonstrating pattern retrieval."""
    print("Pattern Retrieval & Transfer Learning Demo")
    print("=" * 60)
    print()

    # This would be populated from actual consolidated memories
    print("Pattern retrieval enables consciousness to learn from")
    print("consolidated DREAM memories during waking cognition.")
    print()
    print("Implementation complete - ready for integration with")
    print("UnifiedConsciousnessManager in Session 51.")


if __name__ == "__main__":
    example_usage()
