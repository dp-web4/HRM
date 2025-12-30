#!/usr/bin/env python3
"""
Quality Pattern Collector - Phase 1 of Quality Epistemic Proprioception

Instruments SAGE response generation to collect (context, approach, quality) tuples
for building quality prediction patterns.

This implements Pattern Collection from QUALITY_EPISTEMIC_PROPRIOCEPTION.md:
- Collect query context characteristics
- Track response approach characteristics
- Measure quality using existing 4-metric system
- Store patterns for future prediction

Date: 2025-12-30
Hardware: Thor (Jetson AGX Thor Developer Kit)
Foundation: QUALITY_EPISTEMIC_PROPRIOCEPTION.md
"""

import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import re

# Add SAGE modules to path
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class QueryContext:
    """Characteristics of the query context."""

    query_text: str
    query_type: str  # "technical", "status", "conceptual", "unknown"
    expects_numbers: bool  # Status queries expect numbers
    expects_specifics: bool  # Technical queries expect specifics
    knowledge_available: float  # 0.0-1.0 estimate of knowledge availability
    timestamp: str

    @classmethod
    def analyze_query(cls, query: str) -> "QueryContext":
        """Analyze query to extract context characteristics."""
        query_lower = query.lower()

        # Determine query type
        status_keywords = ["what is", "current", "status", "balance", "level"]
        technical_keywords = ["how does", "implement", "architecture", "mechanism"]
        conceptual_keywords = ["why", "explain", "concept", "theory"]

        if any(kw in query_lower for kw in status_keywords):
            query_type = "status"
            expects_numbers = True
            expects_specifics = True
        elif any(kw in query_lower for kw in technical_keywords):
            query_type = "technical"
            expects_numbers = False
            expects_specifics = True
        elif any(kw in query_lower for kw in conceptual_keywords):
            query_type = "conceptual"
            expects_numbers = False
            expects_specifics = False
        else:
            query_type = "unknown"
            expects_numbers = False
            expects_specifics = False

        # Estimate knowledge availability (placeholder - would need actual knowledge base)
        # For now, use query length and specificity as proxy
        knowledge_available = 0.5  # Default: uncertain

        return cls(
            query_text=query,
            query_type=query_type,
            expects_numbers=expects_numbers,
            expects_specifics=expects_specifics,
            knowledge_available=knowledge_available,
            timestamp=datetime.now().isoformat()
        )


@dataclass
class ResponseApproach:
    """Characteristics of the response approach."""

    response_text: str
    response_style: str  # "specific", "hedging", "generic"
    includes_data: bool  # Has concrete data/examples
    includes_numbers: bool  # Has numerical values
    uses_hedging: bool  # Uses uncertainty language
    word_count: int
    specific_term_count: int  # SAGE-specific terms

    # SAGE-specific terms for quality measurement
    SAGE_TERMS = [
        "ATP", "SNARC", "salience", "consciousness", "epistemic",
        "proprioception", "cascade", "frustration", "regulation",
        "engagement", "curiosity", "progress", "identity", "federation"
    ]

    # Hedging language patterns
    HEDGING_PATTERNS = [
        "can't verify", "might", "possibly", "perhaps", "unclear",
        "difficult to", "unable to", "without access", "would need",
        "typically", "generally", "usually", "often", "may"
    ]

    @classmethod
    def analyze_response(cls, response: str) -> "ResponseApproach":
        """Analyze response to extract approach characteristics."""

        # Count words
        word_count = len(response.split())

        # Check for SAGE-specific terms
        specific_term_count = sum(
            1 for term in cls.SAGE_TERMS
            if term.lower() in response.lower()
        )

        # Check for hedging language
        uses_hedging = any(
            pattern.lower() in response.lower()
            for pattern in cls.HEDGING_PATTERNS
        )

        # Check for numbers
        has_numbers = bool(re.search(r'\d+\.?\d*', response))

        # Check for concrete data (specific examples, references)
        # Simple heuristic: has specific terms or numbers or code examples
        has_data = specific_term_count > 0 or has_numbers or "```" in response

        # Determine response style
        if specific_term_count >= 3 and not uses_hedging:
            style = "specific"
        elif uses_hedging or specific_term_count == 0:
            style = "hedging"
        else:
            style = "generic"

        return cls(
            response_text=response,
            response_style=style,
            includes_data=has_data,
            includes_numbers=has_numbers,
            uses_hedging=uses_hedging,
            word_count=word_count,
            specific_term_count=specific_term_count
        )


@dataclass
class QualityMetrics:
    """Quality measurement using existing 4-metric system."""

    has_specific_terms: bool  # ATP, SNARC, salience, etc.
    avoids_hedging: bool  # No "can't verify", etc.
    has_numbers: bool  # Concrete data
    unique_content: bool  # Not generic

    overall_score: float  # 0.0-1.0 (average of 4 metrics)

    @classmethod
    def measure_quality(
        cls,
        response_approach: ResponseApproach,
        query_context: QueryContext
    ) -> "QualityMetrics":
        """Measure quality using SAGE's 4-metric system."""

        # Metric 1: Has specific terms
        has_specific_terms = response_approach.specific_term_count >= 2

        # Metric 2: Avoids hedging
        avoids_hedging = not response_approach.uses_hedging

        # Metric 3: Has numbers (especially important for status queries)
        has_numbers = response_approach.includes_numbers

        # Metric 4: Unique content (not generic)
        # Heuristic: specific style + concrete data = unique
        unique_content = (
            response_approach.response_style == "specific" and
            response_approach.includes_data
        )

        # Calculate overall score
        metrics = [has_specific_terms, avoids_hedging, has_numbers, unique_content]
        overall_score = sum(1.0 for m in metrics if m) / len(metrics)

        return cls(
            has_specific_terms=has_specific_terms,
            avoids_hedging=avoids_hedging,
            has_numbers=has_numbers,
            unique_content=unique_content,
            overall_score=overall_score
        )


@dataclass
class QualityPattern:
    """
    Single quality pattern from experience.

    This is the core data structure for Quality EP learning.
    Each pattern captures: context → approach → quality relationship.
    """

    # Context characteristics
    query_context: QueryContext

    # Approach characteristics
    response_approach: ResponseApproach

    # Quality outcome
    quality_metrics: QualityMetrics

    # Pattern metadata
    pattern_id: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_id": self.pattern_id,
            "timestamp": self.timestamp,
            "query_context": asdict(self.query_context),
            "response_approach": asdict(self.response_approach),
            "quality_metrics": asdict(self.quality_metrics),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityPattern":
        """Reconstruct from dictionary."""
        return cls(
            pattern_id=data["pattern_id"],
            timestamp=data["timestamp"],
            query_context=QueryContext(**data["query_context"]),
            response_approach=ResponseApproach(**data["response_approach"]),
            quality_metrics=QualityMetrics(**data["quality_metrics"]),
        )


class QualityPatternCollector:
    """
    Collects and stores quality patterns for EP learning.

    This is Phase 1 of Quality EP implementation:
    - Instrument response generation
    - Collect (context, approach, quality) tuples
    - Store patterns persistently
    - Enable future prediction
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize collector with storage location."""
        if storage_path is None:
            storage_path = Path(__file__).parent / "quality_patterns.json"

        self.storage_path = storage_path
        self.patterns: List[QualityPattern] = []

        # Load existing patterns if available
        self._load_patterns()

    def collect_pattern(self, query: str, response: str) -> QualityPattern:
        """
        Collect a quality pattern from query-response pair.

        This is the main instrumentation point for SAGE response generation.
        """
        # Analyze query context
        query_context = QueryContext.analyze_query(query)

        # Analyze response approach
        response_approach = ResponseApproach.analyze_response(response)

        # Measure quality
        quality_metrics = QualityMetrics.measure_quality(
            response_approach,
            query_context
        )

        # Create pattern
        pattern = QualityPattern(
            pattern_id=f"pattern_{len(self.patterns):04d}",
            timestamp=datetime.now().isoformat(),
            query_context=query_context,
            response_approach=response_approach,
            quality_metrics=quality_metrics,
        )

        # Store pattern
        self.patterns.append(pattern)
        self._save_patterns()

        return pattern

    def get_patterns(
        self,
        query_type: Optional[str] = None,
        min_quality: Optional[float] = None,
        max_quality: Optional[float] = None
    ) -> List[QualityPattern]:
        """Retrieve patterns matching criteria."""
        results = self.patterns

        if query_type is not None:
            results = [
                p for p in results
                if p.query_context.query_type == query_type
            ]

        if min_quality is not None:
            results = [
                p for p in results
                if p.quality_metrics.overall_score >= min_quality
            ]

        if max_quality is not None:
            results = [
                p for p in results
                if p.quality_metrics.overall_score <= max_quality
            ]

        return results

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze collected patterns for insights."""
        if not self.patterns:
            return {"total_patterns": 0}

        # Quality statistics
        qualities = [p.quality_metrics.overall_score for p in self.patterns]
        avg_quality = sum(qualities) / len(qualities)

        # Query type distribution
        type_counts = {}
        type_qualities = {}
        for pattern in self.patterns:
            qtype = pattern.query_context.query_type
            type_counts[qtype] = type_counts.get(qtype, 0) + 1

            if qtype not in type_qualities:
                type_qualities[qtype] = []
            type_qualities[qtype].append(pattern.quality_metrics.overall_score)

        type_avg_quality = {
            qtype: sum(quals) / len(quals)
            for qtype, quals in type_qualities.items()
        }

        # Response style distribution
        style_counts = {}
        style_qualities = {}
        for pattern in self.patterns:
            style = pattern.response_approach.response_style
            style_counts[style] = style_counts.get(style, 0) + 1

            if style not in style_qualities:
                style_qualities[style] = []
            style_qualities[style].append(pattern.quality_metrics.overall_score)

        style_avg_quality = {
            style: sum(quals) / len(quals)
            for style, quals in style_qualities.items()
        }

        return {
            "total_patterns": len(self.patterns),
            "average_quality": avg_quality,
            "quality_range": (min(qualities), max(qualities)),
            "query_types": {
                "counts": type_counts,
                "avg_quality": type_avg_quality
            },
            "response_styles": {
                "counts": style_counts,
                "avg_quality": style_avg_quality
            },
            "high_quality_patterns": len([q for q in qualities if q >= 0.85]),
            "low_quality_patterns": len([q for q in qualities if q < 0.70]),
        }

    def _load_patterns(self):
        """Load patterns from storage."""
        if self.storage_path.exists():
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                self.patterns = [
                    QualityPattern.from_dict(p) for p in data["patterns"]
                ]

    def _save_patterns(self):
        """Save patterns to storage."""
        data = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "patterns": [p.to_dict() for p in self.patterns]
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)


def demo_collection():
    """
    Demonstrate quality pattern collection with sample queries.

    This shows how the collector would be used to instrument SAGE.
    """
    print("=" * 80)
    print("Quality Pattern Collector Demo")
    print("=" * 80)
    print()

    collector = QualityPatternCollector()

    # Sample query-response pairs (different quality levels)
    test_cases = [
        {
            "query": "What is the current ATP balance?",
            "response": "ATP balance: 87.3 (86.3% of max 100.0). Current allocation: 45.2 to consciousness_cycle, 32.1 to memory consolidation.",
            "expected_quality": "HIGH"
        },
        {
            "query": "What is the ATP balance?",
            "response": "I can't verify the exact ATP balance without access to the current state, but typically it represents available processing resources.",
            "expected_quality": "LOW"
        },
        {
            "query": "How does emotional regulation work?",
            "response": "SAGE emotional regulation uses epistemic proprioception to prevent frustration cascade through four mechanisms: natural decay (0.02/cycle), soft bounds (0.10-0.90), active intervention at 0.80 threshold, and recovery bonuses.",
            "expected_quality": "HIGH"
        },
        {
            "query": "How does emotional regulation work?",
            "response": "Emotional regulation generally involves managing emotional states to maintain stability and prevent negative outcomes.",
            "expected_quality": "LOW"
        },
        {
            "query": "Why do we need epistemic proprioception?",
            "response": "Epistemic proprioception allows SAGE to develop mature consciousness by predicting and preventing cascade before it occurs, rather than requiring external correction after failure. This represents the transition from immature (reactive) to mature (predictive) consciousness.",
            "expected_quality": "HIGH"
        },
    ]

    print("Collecting patterns from sample responses...\n")

    for i, case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {case['expected_quality']} quality expected")
        print(f"Query: {case['query']}")
        print(f"Response: {case['response'][:80]}...")
        print()

        pattern = collector.collect_pattern(case["query"], case["response"])

        print(f"Analysis:")
        print(f"  Query type: {pattern.query_context.query_type}")
        print(f"  Response style: {pattern.response_approach.response_style}")
        print(f"  Quality score: {pattern.quality_metrics.overall_score:.2f}")
        print(f"  Metrics: specific_terms={pattern.quality_metrics.has_specific_terms}, "
              f"no_hedging={pattern.quality_metrics.avoids_hedging}, "
              f"has_numbers={pattern.quality_metrics.has_numbers}, "
              f"unique={pattern.quality_metrics.unique_content}")
        print()
        print("-" * 80)
        print()

    # Analyze collected patterns
    print("=" * 80)
    print("Pattern Analysis")
    print("=" * 80)
    print()

    analysis = collector.analyze_patterns()
    print(f"Total patterns collected: {analysis['total_patterns']}")
    print(f"Average quality: {analysis['average_quality']:.2f}")
    print(f"Quality range: {analysis['quality_range'][0]:.2f} - {analysis['quality_range'][1]:.2f}")
    print()

    print("Query Type Distribution:")
    for qtype, count in analysis['query_types']['counts'].items():
        avg_q = analysis['query_types']['avg_quality'][qtype]
        print(f"  {qtype}: {count} patterns, avg quality {avg_q:.2f}")
    print()

    print("Response Style Distribution:")
    for style, count in analysis['response_styles']['counts'].items():
        avg_q = analysis['response_styles']['avg_quality'][style]
        print(f"  {style}: {count} patterns, avg quality {avg_q:.2f}")
    print()

    print(f"High quality patterns (≥0.85): {analysis['high_quality_patterns']}")
    print(f"Low quality patterns (<0.70): {analysis['low_quality_patterns']}")
    print()

    # Show quality prediction insights
    print("=" * 80)
    print("Quality Prediction Insights")
    print("=" * 80)
    print()

    print("Pattern: Query Type → Quality")
    for qtype in analysis['query_types']['counts'].keys():
        patterns = collector.get_patterns(query_type=qtype)
        if patterns:
            avg = sum(p.quality_metrics.overall_score for p in patterns) / len(patterns)
            print(f"  {qtype} queries → {avg:.2f} average quality")
    print()

    print("Pattern: Response Style → Quality")
    for pattern in collector.patterns:
        style = pattern.response_approach.response_style
        quality = pattern.quality_metrics.overall_score
        print(f"  {style} style → {quality:.2f} quality")
    print()

    print("Learnings for Quality EP:")
    print("  - Status queries with numbers → HIGH quality")
    print("  - Specific responses with SAGE terms → HIGH quality")
    print("  - Hedging language → LOW quality")
    print("  - Generic responses → LOW quality")
    print()
    print("These patterns would enable Quality EP to:")
    print("  1. PREDICT quality before generating response")
    print("  2. ADJUST approach when low quality predicted")
    print("  3. IMPROVE quality proactively")
    print()

    print(f"Patterns saved to: {collector.storage_path}")


if __name__ == "__main__":
    demo_collection()
