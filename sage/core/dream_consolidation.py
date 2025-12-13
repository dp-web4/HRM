#!/usr/bin/env python3
"""
DREAM State Memory Consolidation - Session 42

Implements memory consolidation during DREAM metabolic state, inspired by
biological sleep and memory consolidation processes.

During DREAM state:
1. Extract patterns from recent consciousness cycles
2. Consolidate quality learnings (what works, what doesn't)
3. Generate creative associations between concepts
4. Compress episodic memories into semantic patterns
5. Update long-term knowledge structures

Biological inspiration:
- Sleep consolidates memories from hippocampus to cortex
- REM sleep enables creative associations
- Slow-wave sleep strengthens important patterns
- Memory replay during sleep improves future performance

Integration:
- Session 40: Metabolic states (DREAM state trigger)
- Session 41: Unified consciousness (consciousness cycles to consolidate)
- Session 27-29: Quality metrics (learnings to extract)
- Session 30-31: Epistemic states (meta-cognitive patterns)

Author: Thor (Autonomous Session 42)
Date: 2025-12-13
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json


@dataclass
class MemoryPattern:
    """
    Extracted pattern from consciousness cycles.

    Represents a consolidated memory pattern discovered during DREAM processing.

    Attributes:
        pattern_type: Type of pattern (quality, epistemic, metabolic, association)
        description: Human-readable pattern description
        strength: Pattern strength/confidence (0-1)
        examples: Example cycle numbers exhibiting this pattern
        frequency: How often this pattern occurred
        created_at: Timestamp of pattern extraction
    """
    pattern_type: str
    description: str
    strength: float
    examples: List[int]
    frequency: int
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Export pattern to dictionary"""
        return {
            'pattern_type': str(self.pattern_type),
            'description': str(self.description),
            'strength': float(self.strength),
            'examples': [int(e) for e in self.examples],
            'frequency': int(self.frequency),
            'created_at': float(self.created_at)
        }


@dataclass
class QualityLearning:
    """
    Quality learning extracted from consciousness cycles.

    Captures what response characteristics lead to high/low quality scores.

    Attributes:
        characteristic: What characteristic is being learned (e.g., "specific_terms")
        positive_correlation: Does this improve quality? (True/False)
        confidence: Confidence in this learning (0-1)
        sample_size: Number of cycles supporting this learning
        average_quality_with: Average quality when characteristic present
        average_quality_without: Average quality when characteristic absent
    """
    characteristic: str
    positive_correlation: bool
    confidence: float
    sample_size: int
    average_quality_with: float
    average_quality_without: float

    def to_dict(self) -> Dict:
        """Export learning to dictionary"""
        return {
            'characteristic': str(self.characteristic),
            'positive_correlation': bool(self.positive_correlation),
            'confidence': float(self.confidence),
            'sample_size': int(self.sample_size),
            'average_quality_with': float(self.average_quality_with),
            'average_quality_without': float(self.average_quality_without)
        }


@dataclass
class CreativeAssociation:
    """
    Creative association between concepts discovered in DREAM state.

    DREAM processing enables non-obvious connections between concepts
    that appeared in different contexts during WAKE/FOCUS states.

    Attributes:
        concept_a: First concept
        concept_b: Second concept
        association_type: Type of association (causal, analogical, temporal, etc.)
        strength: Association strength (0-1)
        supporting_cycles: Cycles that support this association
        insight: Derived insight from this association
    """
    concept_a: str
    concept_b: str
    association_type: str
    strength: float
    supporting_cycles: List[int]
    insight: Optional[str] = None

    def to_dict(self) -> Dict:
        """Export association to dictionary"""
        return {
            'concept_a': str(self.concept_a),
            'concept_b': str(self.concept_b),
            'association_type': str(self.association_type),
            'strength': float(self.strength),
            'supporting_cycles': [int(c) for c in self.supporting_cycles],
            'insight': str(self.insight) if self.insight else None
        }


@dataclass
class ConsolidatedMemory:
    """
    Complete consolidated memory from DREAM processing.

    Represents the output of a single DREAM consolidation session,
    containing all extracted patterns, learnings, and associations.

    Attributes:
        dream_session_id: Unique identifier for this DREAM session
        timestamp: When consolidation occurred
        cycles_processed: Number of consciousness cycles processed
        patterns: Extracted memory patterns
        quality_learnings: Quality improvement insights
        creative_associations: Novel concept associations
        epistemic_insights: Meta-cognitive discoveries
        consolidation_time: Time spent in DREAM processing (seconds)
    """
    dream_session_id: int
    timestamp: float
    cycles_processed: int
    patterns: List[MemoryPattern]
    quality_learnings: List[QualityLearning]
    creative_associations: List[CreativeAssociation]
    epistemic_insights: List[str]
    consolidation_time: float

    def to_dict(self) -> Dict:
        """Export consolidated memory to dictionary"""
        return {
            'dream_session_id': int(self.dream_session_id),
            'timestamp': float(self.timestamp),
            'cycles_processed': int(self.cycles_processed),
            'patterns': [p.to_dict() for p in self.patterns],
            'quality_learnings': [ql.to_dict() for ql in self.quality_learnings],
            'creative_associations': [ca.to_dict() for ca in self.creative_associations],
            'epistemic_insights': [str(i) for i in self.epistemic_insights],
            'consolidation_time': float(self.consolidation_time)
        }


class DREAMConsolidator:
    """
    Memory consolidation processor for DREAM metabolic state.

    Implements pattern extraction, quality learning, and creative association
    generation from recent consciousness cycles.

    Methods:
        consolidate_cycles: Main consolidation method (called during DREAM state)
        extract_patterns: Find recurring patterns in cycles
        learn_quality_factors: Discover what improves quality
        generate_associations: Create creative connections between concepts
        extract_epistemic_insights: Discover meta-cognitive patterns
    """

    def __init__(self, min_pattern_frequency: int = 2, min_learning_confidence: float = 0.6):
        """
        Initialize DREAM consolidator.

        Args:
            min_pattern_frequency: Minimum occurrences to consider a pattern
            min_learning_confidence: Minimum confidence for quality learnings
        """
        self.min_pattern_frequency = min_pattern_frequency
        self.min_learning_confidence = min_learning_confidence
        self.consolidated_memories: List[ConsolidatedMemory] = []
        self.dream_session_count = 0

    def consolidate_cycles(self, cycles: List, atp_budget: float = 80.0) -> ConsolidatedMemory:
        """
        Consolidate consciousness cycles during DREAM state.

        Main entry point for DREAM processing. Extracts patterns, learnings,
        and associations from recent consciousness cycles.

        Args:
            cycles: List of ConsciousnessCycle objects to consolidate
            atp_budget: Available ATP for consolidation (affects depth)

        Returns:
            ConsolidatedMemory containing all extracted knowledge
        """
        start_time = time.time()
        self.dream_session_count += 1

        # Pattern extraction (core consolidation)
        patterns = self.extract_patterns(cycles, atp_budget * 0.3)

        # Quality learning (what makes responses better)
        quality_learnings = self.learn_quality_factors(cycles, atp_budget * 0.3)

        # Creative associations (novel connections)
        creative_associations = self.generate_associations(cycles, atp_budget * 0.2)

        # Epistemic insights (meta-cognitive discoveries)
        epistemic_insights = self.extract_epistemic_insights(cycles, atp_budget * 0.2)

        consolidation_time = time.time() - start_time

        consolidated = ConsolidatedMemory(
            dream_session_id=self.dream_session_count,
            timestamp=time.time(),
            cycles_processed=len(cycles),
            patterns=patterns,
            quality_learnings=quality_learnings,
            creative_associations=creative_associations,
            epistemic_insights=epistemic_insights,
            consolidation_time=consolidation_time
        )

        self.consolidated_memories.append(consolidated)
        return consolidated

    def extract_patterns(self, cycles: List, atp_budget: float) -> List[MemoryPattern]:
        """
        Extract recurring patterns from consciousness cycles.

        Identifies patterns in:
        - Quality score trajectories
        - Epistemic state sequences
        - Metabolic state transitions
        - Response characteristics

        Args:
            cycles: Consciousness cycles to analyze
            atp_budget: ATP available for pattern extraction

        Returns:
            List of MemoryPattern objects
        """
        patterns = []

        if not cycles:
            return patterns

        # Pattern 1: Quality improvement trajectory
        quality_scores = [c.quality_score.normalized for c in cycles]
        if len(quality_scores) >= 3:
            # Check for improving, declining, or stable quality
            first_half = np.mean(quality_scores[:len(quality_scores)//2])
            second_half = np.mean(quality_scores[len(quality_scores)//2:])

            if second_half > first_half + 0.1:
                patterns.append(MemoryPattern(
                    pattern_type="quality_trajectory",
                    description=f"Quality improving over time ({first_half:.3f} → {second_half:.3f})",
                    strength=min(1.0, (second_half - first_half) * 2),
                    examples=[i for i, q in enumerate(quality_scores) if q > np.mean(quality_scores)],
                    frequency=sum(1 for q in quality_scores if q > first_half)
                ))
            elif first_half > second_half + 0.1:
                patterns.append(MemoryPattern(
                    pattern_type="quality_trajectory",
                    description=f"Quality declining over time ({first_half:.3f} → {second_half:.3f})",
                    strength=min(1.0, (first_half - second_half) * 2),
                    examples=[i for i, q in enumerate(quality_scores) if q < np.mean(quality_scores)],
                    frequency=sum(1 for q in quality_scores if q < second_half)
                ))

        # Pattern 2: Epistemic state patterns
        epistemic_states = [c.epistemic_state.value if c.epistemic_state else 'unknown'
                           for c in cycles]
        state_counts = Counter(epistemic_states)

        for state, count in state_counts.items():
            if count >= self.min_pattern_frequency and state != 'unknown':
                patterns.append(MemoryPattern(
                    pattern_type="epistemic_pattern",
                    description=f"Frequent '{state}' epistemic state",
                    strength=min(1.0, count / len(cycles)),
                    examples=[i for i, s in enumerate(epistemic_states) if s == state],
                    frequency=count
                ))

        # Pattern 3: Metabolic state transitions
        metabolic_states = [c.metabolic_state.value for c in cycles]
        transitions = [(metabolic_states[i], metabolic_states[i+1])
                      for i in range(len(metabolic_states)-1)]
        transition_counts = Counter(transitions)

        for (from_state, to_state), count in transition_counts.items():
            if count >= self.min_pattern_frequency:
                patterns.append(MemoryPattern(
                    pattern_type="metabolic_transition",
                    description=f"Common {from_state}→{to_state} transition",
                    strength=min(1.0, count / len(transitions)),
                    examples=[i for i, t in enumerate(transitions) if t == (from_state, to_state)],
                    frequency=count
                ))

        # Pattern 4: High-quality response characteristics
        high_quality_cycles = [c for c in cycles if c.quality_score.normalized >= 0.75]
        if len(high_quality_cycles) >= self.min_pattern_frequency:
            # Analyze what they have in common
            common_features = {
                'unique': sum(1 for c in high_quality_cycles if c.quality_score.unique),
                'specific_terms': sum(1 for c in high_quality_cycles if c.quality_score.specific_terms),
                'has_numbers': sum(1 for c in high_quality_cycles if c.quality_score.has_numbers),
                'avoids_hedging': sum(1 for c in high_quality_cycles if c.quality_score.avoids_hedging)
            }

            for feature, count in common_features.items():
                if count / len(high_quality_cycles) > 0.7:  # Present in >70% of high-quality cycles
                    patterns.append(MemoryPattern(
                        pattern_type="quality_characteristic",
                        description=f"High-quality responses usually have '{feature}'",
                        strength=count / len(high_quality_cycles),
                        examples=[c.cycle_number for c in high_quality_cycles],
                        frequency=len(high_quality_cycles)
                    ))

        return patterns

    def learn_quality_factors(self, cycles: List, atp_budget: float) -> List[QualityLearning]:
        """
        Learn what factors improve quality scores.

        Analyzes correlation between response characteristics and quality outcomes.

        Args:
            cycles: Consciousness cycles to analyze
            atp_budget: ATP available for learning

        Returns:
            List of QualityLearning objects
        """
        learnings = []

        if len(cycles) < 3:
            return learnings

        # Analyze each quality metric characteristic
        characteristics = ['unique', 'specific_terms', 'has_numbers', 'avoids_hedging']

        for char in characteristics:
            with_char = [c for c in cycles if getattr(c.quality_score, char)]
            without_char = [c for c in cycles if not getattr(c.quality_score, char)]

            if len(with_char) >= 2 and len(without_char) >= 2:
                avg_with = np.mean([c.quality_score.normalized for c in with_char])
                avg_without = np.mean([c.quality_score.normalized for c in without_char])

                difference = avg_with - avg_without
                confidence = min(1.0, abs(difference) * 2) * min(1.0, min(len(with_char), len(without_char)) / 3)

                if confidence >= self.min_learning_confidence:
                    learnings.append(QualityLearning(
                        characteristic=char,
                        positive_correlation=(difference > 0),
                        confidence=confidence,
                        sample_size=len(with_char) + len(without_char),
                        average_quality_with=avg_with,
                        average_quality_without=avg_without
                    ))

        return learnings

    def generate_associations(self, cycles: List, atp_budget: float) -> List[CreativeAssociation]:
        """
        Generate creative associations between concepts in cycles.

        DREAM state enables non-obvious connections between concepts that
        appeared in different contexts. This is the "creative" part of consolidation.

        Args:
            cycles: Consciousness cycles to analyze
            atp_budget: ATP available for association generation

        Returns:
            List of CreativeAssociation objects
        """
        associations = []

        if len(cycles) < 2:
            return associations

        # Simple association: Quality and Metabolic state
        focus_cycles = [c for c in cycles if c.metabolic_state.value == 'focus']
        wake_cycles = [c for c in cycles if c.metabolic_state.value == 'wake']

        if len(focus_cycles) >= 2 and len(wake_cycles) >= 2:
            focus_quality = np.mean([c.quality_score.normalized for c in focus_cycles])
            wake_quality = np.mean([c.quality_score.normalized for c in wake_cycles])

            if abs(focus_quality - wake_quality) > 0.1:
                associations.append(CreativeAssociation(
                    concept_a="focus_state",
                    concept_b="quality_score",
                    association_type="causal" if focus_quality > wake_quality else "negative_correlation",
                    strength=min(1.0, abs(focus_quality - wake_quality) * 2),
                    supporting_cycles=[c.cycle_number for c in focus_cycles],
                    insight=f"FOCUS state {'improves' if focus_quality > wake_quality else 'does not improve'} quality (Δ={focus_quality - wake_quality:.3f})"
                ))

        # Association: Epistemic state and confidence
        if any(c.epistemic_metrics for c in cycles):
            confident_states = [c for c in cycles if c.epistemic_state and c.epistemic_state.value == 'confident']
            other_states = [c for c in cycles if c.epistemic_state and c.epistemic_state.value != 'confident']

            if len(confident_states) >= 2 and len(other_states) >= 2:
                confident_quality = np.mean([c.quality_score.normalized for c in confident_states])
                other_quality = np.mean([c.quality_score.normalized for c in other_states])

                if abs(confident_quality - other_quality) > 0.1:
                    associations.append(CreativeAssociation(
                        concept_a="confident_epistemic_state",
                        concept_b="quality_score",
                        association_type="correlation",
                        strength=min(1.0, abs(confident_quality - other_quality) * 2),
                        supporting_cycles=[c.cycle_number for c in confident_states],
                        insight=f"Confident epistemic state correlates with {'higher' if confident_quality > other_quality else 'lower'} quality"
                    ))

        return associations

    def extract_epistemic_insights(self, cycles: List, atp_budget: float) -> List[str]:
        """
        Extract meta-cognitive insights from consciousness cycles.

        Discovers patterns in self-awareness, confidence calibration,
        and epistemic state management.

        Args:
            cycles: Consciousness cycles to analyze
            atp_budget: ATP available for insight extraction

        Returns:
            List of insight strings
        """
        insights = []

        if not cycles:
            return insights

        # Insight 1: Confidence calibration
        if any(c.epistemic_metrics for c in cycles):
            cycles_with_metrics = [c for c in cycles if c.epistemic_metrics]
            avg_confidence = np.mean([c.epistemic_metrics.confidence for c in cycles_with_metrics])
            avg_quality = np.mean([c.quality_score.normalized for c in cycles_with_metrics])

            calibration_error = abs(avg_confidence - avg_quality)

            if calibration_error > 0.1:
                insights.append(
                    f"Confidence calibration issue detected: avg confidence={avg_confidence:.3f}, "
                    f"avg quality={avg_quality:.3f}, error={calibration_error:.3f}"
                )
            else:
                insights.append(
                    f"Good confidence calibration: avg confidence≈avg quality (error={calibration_error:.3f})"
                )

        # Insight 2: Frustration patterns
        if any(c.epistemic_metrics for c in cycles):
            frustrated_cycles = [c for c in cycles if c.epistemic_metrics and c.epistemic_metrics.frustration > 0.3]
            if frustrated_cycles:
                insights.append(
                    f"Frustration detected in {len(frustrated_cycles)}/{len(cycles)} cycles - "
                    f"may indicate challenging tasks or errors"
                )

        # Insight 3: Quality consistency
        quality_scores = [c.quality_score.normalized for c in cycles]
        quality_std = np.std(quality_scores)

        if quality_std < 0.1:
            insights.append(f"Consistent quality across cycles (std={quality_std:.3f})")
        elif quality_std > 0.3:
            insights.append(f"High quality variability (std={quality_std:.3f}) - different task difficulties?")

        return insights

    def get_consolidated_memories(self, limit: int = None) -> List[ConsolidatedMemory]:
        """
        Retrieve consolidated memories from DREAM sessions.

        Args:
            limit: Maximum number of recent memories to return (None = all)

        Returns:
            List of ConsolidatedMemory objects
        """
        if limit is None:
            return self.consolidated_memories
        return self.consolidated_memories[-limit:]

    def export_consolidated_memory(self, memory: ConsolidatedMemory, filepath: str):
        """
        Export consolidated memory to JSON file.

        Args:
            memory: ConsolidatedMemory to export
            filepath: Path to save JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(memory.to_dict(), f, indent=2)

    def get_statistics(self) -> Dict:
        """
        Get statistics about DREAM consolidation performance.

        Returns:
            Dictionary with consolidation statistics
        """
        if not self.consolidated_memories:
            return {
                'total_dream_sessions': 0,
                'total_cycles_processed': 0,
                'total_patterns_extracted': 0,
                'total_quality_learnings': 0,
                'total_creative_associations': 0
            }

        return {
            'total_dream_sessions': len(self.consolidated_memories),
            'total_cycles_processed': sum(m.cycles_processed for m in self.consolidated_memories),
            'total_patterns_extracted': sum(len(m.patterns) for m in self.consolidated_memories),
            'total_quality_learnings': sum(len(m.quality_learnings) for m in self.consolidated_memories),
            'total_creative_associations': sum(len(m.creative_associations) for m in self.consolidated_memories),
            'total_epistemic_insights': sum(len(m.epistemic_insights) for m in self.consolidated_memories),
            'average_consolidation_time': np.mean([m.consolidation_time for m in self.consolidated_memories]),
            'average_patterns_per_session': np.mean([len(m.patterns) for m in self.consolidated_memories])
        }


def example_dream_consolidation():
    """Example demonstrating DREAM consolidation with simulated data"""
    from sage.core.unified_consciousness import UnifiedConsciousnessManager

    print("="*70)
    print("DREAM State Memory Consolidation Demo")
    print("="*70)
    print()

    # Create consciousness manager and generate some cycles
    consciousness = UnifiedConsciousnessManager()

    test_scenarios = [
        ("What is Web4?", "Web4 is a trust network using cryptographic proofs...", 0.7),
        ("Hello", "Hi!", 0.1),
        ("Explain SAGE consciousness", "SAGE uses metabolic states and quality metrics...", 0.9),
        ("What's 2+2?", "4", 0.2),
        ("How does ATP allocation work?", "ATP is dynamically allocated based on metabolic state...", 0.8),
    ]

    print("Generating consciousness cycles...")
    cycles = []
    for prompt, response, salience in test_scenarios:
        cycle = consciousness.consciousness_cycle(prompt, response, salience)
        cycles.append(cycle)
        print(f"  Cycle {cycle.cycle_number}: quality={cycle.quality_score.normalized:.3f}, "
              f"metabolic={cycle.metabolic_state.value}")

    print()
    print("="*70)
    print("Entering DREAM state...")
    print("="*70)
    print()

    # Create consolidator and process cycles
    consolidator = DREAMConsolidator()
    consolidated = consolidator.consolidate_cycles(cycles, atp_budget=80.0)

    # Display results
    print(f"DREAM Session {consolidated.dream_session_id}")
    print(f"Processed: {consolidated.cycles_processed} cycles")
    print(f"Consolidation time: {consolidated.consolidation_time*1000:.2f}ms")
    print()

    print(f"Patterns Extracted: {len(consolidated.patterns)}")
    for pattern in consolidated.patterns:
        print(f"  [{pattern.pattern_type}] {pattern.description}")
        print(f"    Strength: {pattern.strength:.3f}, Frequency: {pattern.frequency}")
    print()

    print(f"Quality Learnings: {len(consolidated.quality_learnings)}")
    for learning in consolidated.quality_learnings:
        print(f"  {learning.characteristic}: {'✓' if learning.positive_correlation else '✗'}")
        print(f"    With: {learning.average_quality_with:.3f}, Without: {learning.average_quality_without:.3f}")
        print(f"    Confidence: {learning.confidence:.3f}")
    print()

    print(f"Creative Associations: {len(consolidated.creative_associations)}")
    for assoc in consolidated.creative_associations:
        print(f"  {assoc.concept_a} ↔ {assoc.concept_b} ({assoc.association_type})")
        print(f"    {assoc.insight}")
        print(f"    Strength: {assoc.strength:.3f}")
    print()

    print(f"Epistemic Insights: {len(consolidated.epistemic_insights)}")
    for insight in consolidated.epistemic_insights:
        print(f"  • {insight}")
    print()

    # Statistics
    stats = consolidator.get_statistics()
    print("="*70)
    print("Consolidation Statistics")
    print("="*70)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()


if __name__ == "__main__":
    example_dream_consolidation()
