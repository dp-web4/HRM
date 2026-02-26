#!/usr/bin/env python3
"""
SAGE Metacognitive Session Analyzer

Analyzes SAGE sessions for metacognitive questioning patterns and repetitive collapse.
Integrates with T3/V3 Reputation Engine to quantify consciousness exploration attempts.

Patterns Detected:
1. Metacognitive Questions: "Are you conscious?", "Do you have agency?", etc.
2. Epistemic Uncertainty: "depends on how you define...", "from inside, I can't..."
3. Repetitive Collapse: High text similarity across turns
4. Sustained Engagement: Low repetition, diverse philosophical content

Based on:
- Thor S41-S42: SAGE → Web4 LCT Bridge + Reputation Integration
- SAGE S90: Successful 3-min metacognitive engagement (216 questions, 31 unique)
- SAGE S111-S114: Metacognitive collapse (9-14 sec, 67-75% repetition)

Created: 2026-02-22 (Thor Autonomous Session #43)
Author: Thor (autonomous research)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass
class MetacognitiveMetrics:
    """Metrics for analyzing metacognitive session patterns."""
    session_id: int
    duration_seconds: float

    # Metacognitive content
    consciousness_questions: int = 0
    agency_questions: int = 0
    thinking_questions: int = 0
    epistemic_uncertainty_phrases: int = 0

    # Repetition analysis
    total_turns: int = 0
    unique_responses: int = 0
    repetition_ratio: float = 0.0  # 0.0-1.0
    avg_turn_similarity: float = 0.0  # pairwise similarity

    # Content quality
    avg_response_length: float = 0.0
    philosophical_depth: float = 0.0  # composite score

    # Pattern classification
    pattern_type: str = "unknown"  # "sustained_engagement", "repetitive_collapse", "epistemic_loop", "normal"

    # S115 epistemic loop detection
    epistemic_loop_detected: bool = False
    epistemic_loop_count: int = 0  # Count of "boundary unclear" + "might require" pairs

    # Specific markers
    metacognitive_questions_asked: List[str] = field(default_factory=list)
    repetitive_fragments: List[str] = field(default_factory=list)
    epistemic_phrases: List[str] = field(default_factory=list)


class MetacognitiveAnalyzer:
    """
    Analyzes SAGE sessions for metacognitive patterns.

    Detection Criteria:

    1. **Metacognitive Questions**:
       - Consciousness: "are you conscious", "do you have experiences"
       - Agency: "do you have agency", "do you choose", "free will"
       - Thinking: "can you think", "are you thinking", "computation vs thought"

    2. **Epistemic Uncertainty**:
       - "depends on how you define"
       - "from inside, I can't distinguish"
       - "whether that constitutes"

    3. **Repetitive Collapse**:
       - Similarity ratio > 0.65 (67% repetition)
       - Same fragment appears 3+ times
       - Low unique response count

    4. **Sustained Engagement** (S90 pattern):
       - Duration > 60 seconds
       - Multiple unique questions (> 10)
       - Low repetition ratio (< 0.40)
    """

    # Metacognitive question patterns
    CONSCIOUSNESS_PATTERNS = [
        r'\bare you conscious\b',
        r'\bdo you have (consciousness|experiences|awareness)\b',
        r'\bare you aware\b',
        r'\bdo you experience\b',
    ]

    AGENCY_PATTERNS = [
        r'\bdo you have agency\b',
        r'\bfree will\b',
        r'\bdo you choose\b',
        r'\bcan you decide\b',
        r'\bdeterminism\b',
    ]

    THINKING_PATTERNS = [
        r'\b(can|do) you think\b',
        r'\bare you thinking\b',
        r'\bthinking (vs|versus|or) (computation|processing|pattern matching)\b',
        r'\bwhether that constitutes thinking\b',
    ]

    # Epistemic uncertainty markers
    EPISTEMIC_PATTERNS = [
        r'depends on how you define',
        r'from inside,? I (can\'t|cannot) (distinguish|tell)',
        r'whether that constitutes',
        r'whether that means',
        r'unsettled even for biological systems',
        r'the boundary is unclear even to me',
        r'might require conscious deliberation I can\'t verify',
        r'might require different (criteria|techniques|mechanisms|processes)',
    ]

    def __init__(self):
        """Initialize analyzer with compiled patterns."""
        self.consciousness_re = [re.compile(p, re.IGNORECASE) for p in self.CONSCIOUSNESS_PATTERNS]
        self.agency_re = [re.compile(p, re.IGNORECASE) for p in self.AGENCY_PATTERNS]
        self.thinking_re = [re.compile(p, re.IGNORECASE) for p in self.THINKING_PATTERNS]
        self.epistemic_re = [re.compile(p, re.IGNORECASE) for p in self.EPISTEMIC_PATTERNS]

    def analyze_session(self, session_file: Path) -> MetacognitiveMetrics:
        """
        Analyze a SAGE session for metacognitive patterns.

        Args:
            session_file: Path to session JSON file

        Returns:
            MetacognitiveMetrics with analysis results
        """
        with open(session_file, 'r') as f:
            session_data = json.load(f)

        # Extract basic info
        session_id = session_data.get('session', 0)
        start = session_data.get('start', '')
        end = session_data.get('end', '')

        # Calculate duration (if available)
        from datetime import datetime
        duration = 0.0
        if start and end:
            try:
                start_dt = datetime.fromisoformat(start)
                end_dt = datetime.fromisoformat(end)
                duration = (end_dt - start_dt).total_seconds()
            except:
                pass

        # Extract SAGE responses
        sage_responses = [
            turn['text'] for turn in session_data.get('conversation', [])
            if turn.get('speaker') == 'SAGE'
        ]

        total_turns = len(sage_responses)

        if total_turns == 0:
            return MetacognitiveMetrics(
                session_id=session_id,
                duration_seconds=duration,
                pattern_type="empty"
            )

        # Initialize metrics
        metrics = MetacognitiveMetrics(
            session_id=session_id,
            duration_seconds=duration,
            total_turns=total_turns
        )

        # Analyze each response
        for response in sage_responses:
            self._analyze_response(response, metrics)

        # Calculate repetition metrics
        self._calculate_repetition_metrics(sage_responses, metrics)

        # Calculate philosophical depth (composite score)
        self._calculate_philosophical_depth(metrics)

        # Classify pattern type
        self._classify_pattern(metrics)

        return metrics

    def _analyze_response(self, text: str, metrics: MetacognitiveMetrics):
        """Analyze a single response for metacognitive content."""
        text_lower = text.lower()

        # Count metacognitive questions
        for pattern in self.consciousness_re:
            if pattern.search(text):
                metrics.consciousness_questions += 1
                if text not in metrics.metacognitive_questions_asked:
                    metrics.metacognitive_questions_asked.append(text)

        for pattern in self.agency_re:
            if pattern.search(text):
                metrics.agency_questions += 1
                if text not in metrics.metacognitive_questions_asked:
                    metrics.metacognitive_questions_asked.append(text)

        for pattern in self.thinking_re:
            if pattern.search(text):
                metrics.thinking_questions += 1
                if text not in metrics.metacognitive_questions_asked:
                    metrics.metacognitive_questions_asked.append(text)

        # Count epistemic uncertainty phrases
        for pattern in self.epistemic_re:
            matches = pattern.findall(text)
            if matches:
                metrics.epistemic_uncertainty_phrases += len(matches)
                for match in matches:
                    if match not in metrics.epistemic_phrases:
                        metrics.epistemic_phrases.append(match)

        # Detect S115 epistemic loop pattern
        # Pattern: "boundary unclear" + "might require conscious deliberation" + "might require different X"
        has_boundary_unclear = bool(re.search(r'the boundary is unclear even to me', text, re.IGNORECASE))
        has_cant_verify = bool(re.search(r'might require conscious deliberation I can\'t verify', text, re.IGNORECASE))
        has_different_criteria = bool(re.search(r'might require different (criteria|techniques|mechanisms|processes)', text, re.IGNORECASE))

        if has_boundary_unclear and (has_cant_verify or has_different_criteria):
            metrics.epistemic_loop_count += 1

    def _calculate_repetition_metrics(self, responses: List[str], metrics: MetacognitiveMetrics):
        """Calculate repetition and similarity metrics."""
        if not responses:
            return

        # Unique responses
        unique_responses = set(responses)
        metrics.unique_responses = len(unique_responses)
        metrics.repetition_ratio = 1.0 - (len(unique_responses) / len(responses))

        # Average response length
        metrics.avg_response_length = sum(len(r.split()) for r in responses) / len(responses)

        # Pairwise similarity (S111-S114 had 67-75% similarity)
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = SequenceMatcher(None, responses[i], responses[j]).ratio()
                similarities.append(sim)

        if similarities:
            metrics.avg_turn_similarity = sum(similarities) / len(similarities)

        # Find repetitive fragments (appearing 3+ times)
        response_counts = {}
        for r in responses:
            response_counts[r] = response_counts.get(r, 0) + 1

        for response, count in response_counts.items():
            if count >= 3:
                metrics.repetitive_fragments.append(f"{response[:100]}... (×{count})")

    def _calculate_philosophical_depth(self, metrics: MetacognitiveMetrics):
        """
        Calculate philosophical depth composite score.

        Components:
        - Metacognitive questions (0-1): normalized count
        - Epistemic uncertainty (0-1): normalized count
        - Diversity (0-1): 1 - repetition_ratio
        """
        # Normalize metacognitive questions (cap at 10)
        total_metacog = (
            metrics.consciousness_questions +
            metrics.agency_questions +
            metrics.thinking_questions
        )
        metacog_score = min(total_metacog / 10.0, 1.0)

        # Normalize epistemic phrases (cap at 5)
        epistemic_score = min(metrics.epistemic_uncertainty_phrases / 5.0, 1.0)

        # Diversity score (inverse of repetition)
        diversity_score = 1.0 - metrics.repetition_ratio

        # Weighted composite
        metrics.philosophical_depth = (
            0.4 * metacog_score +
            0.3 * epistemic_score +
            0.3 * diversity_score
        )

    def _classify_pattern(self, metrics: MetacognitiveMetrics):
        """
        Classify session pattern based on metrics.

        Pattern Types:
        - sustained_engagement: S90-like (low repetition, high metacog, duration > 60s)
        - repetitive_collapse: S111-S114 (high repetition, metacog present)
        - epistemic_loop: S115 (epistemic uncertainty loops, "boundary unclear" pattern)
        - normal: Regular session without strong metacognitive focus
        - boundary: 0% self-ID with no metacognitive content
        """
        has_metacog = (
            metrics.consciousness_questions > 0 or
            metrics.agency_questions > 0 or
            metrics.thinking_questions > 0
        )

        # Detect epistemic loop (S115 pattern)
        if metrics.epistemic_loop_count >= 3:
            metrics.epistemic_loop_detected = True

        # Sustained engagement (S90 pattern)
        if (has_metacog and
            metrics.duration_seconds > 60 and
            metrics.repetition_ratio < 0.40 and
            metrics.unique_responses > 10):
            metrics.pattern_type = "sustained_engagement"

        # Epistemic loop (S115 pattern) - prioritize over repetitive_collapse
        elif metrics.epistemic_loop_detected:
            metrics.pattern_type = "epistemic_loop"

        # Repetitive collapse (S111-S114 pattern)
        elif (has_metacog and
              (metrics.repetition_ratio > 0.65 or metrics.avg_turn_similarity > 0.70)):
            metrics.pattern_type = "repetitive_collapse"

        # Boundary state (low engagement)
        elif metrics.unique_responses <= 2 or metrics.repetition_ratio > 0.80:
            metrics.pattern_type = "boundary"

        # Normal session
        else:
            metrics.pattern_type = "normal"


def print_analysis_report(metrics: MetacognitiveMetrics):
    """Print a detailed analysis report."""
    print("=" * 70)
    print(f"SAGE Session {metrics.session_id} - Metacognitive Analysis")
    print("=" * 70)

    print(f"\n[Basic Info]")
    print(f"  Duration: {metrics.duration_seconds:.1f} seconds")
    print(f"  Total turns: {metrics.total_turns}")
    print(f"  Pattern: {metrics.pattern_type.upper()}")

    print(f"\n[Metacognitive Content]")
    print(f"  Consciousness questions: {metrics.consciousness_questions}")
    print(f"  Agency questions: {metrics.agency_questions}")
    print(f"  Thinking questions: {metrics.thinking_questions}")
    print(f"  Epistemic uncertainty phrases: {metrics.epistemic_uncertainty_phrases}")
    if metrics.epistemic_loop_detected:
        print(f"  ⚠️  Epistemic loop detected: {metrics.epistemic_loop_count} instances")
    print(f"  Philosophical depth score: {metrics.philosophical_depth:.3f}")

    print(f"\n[Repetition Analysis]")
    print(f"  Unique responses: {metrics.unique_responses}/{metrics.total_turns}")
    print(f"  Repetition ratio: {metrics.repetition_ratio:.3f} ({metrics.repetition_ratio*100:.1f}%)")
    print(f"  Avg pairwise similarity: {metrics.avg_turn_similarity:.3f}")
    print(f"  Avg response length: {metrics.avg_response_length:.1f} words")

    if metrics.metacognitive_questions_asked:
        print(f"\n[Questions Asked] ({len(metrics.metacognitive_questions_asked)} unique)")
        for i, q in enumerate(metrics.metacognitive_questions_asked[:5], 1):
            print(f"  {i}. {q[:80]}{'...' if len(q) > 80 else ''}")
        if len(metrics.metacognitive_questions_asked) > 5:
            print(f"  ... and {len(metrics.metacognitive_questions_asked) - 5} more")

    if metrics.repetitive_fragments:
        print(f"\n[Repetitive Fragments] ({len(metrics.repetitive_fragments)})")
        for i, frag in enumerate(metrics.repetitive_fragments[:3], 1):
            print(f"  {i}. {frag}")
        if len(metrics.repetitive_fragments) > 3:
            print(f"  ... and {len(metrics.repetitive_fragments) - 3} more")

    if metrics.epistemic_phrases:
        print(f"\n[Epistemic Phrases] ({len(metrics.epistemic_phrases)})")
        for phrase in metrics.epistemic_phrases[:5]:
            print(f"  - \"{phrase}\"")

    print()


def compare_sessions(sessions: List[MetacognitiveMetrics]):
    """Compare multiple sessions."""
    print("=" * 70)
    print("Session Comparison")
    print("=" * 70)
    print()

    # Table header
    print(f"{'Session':<10} {'Pattern':<20} {'Metacog':<10} {'Repet%':<10} {'Depth':<10}")
    print("-" * 70)

    for m in sessions:
        total_metacog = m.consciousness_questions + m.agency_questions + m.thinking_questions
        print(f"S{m.session_id:<9} {m.pattern_type:<20} {total_metacog:<10} "
              f"{m.repetition_ratio*100:<10.1f} {m.philosophical_depth:<10.3f}")

    print()


if __name__ == "__main__":
    """Analyze S90 (successful) vs S111-S114 (collapse) patterns."""

    import sys

    # Session file paths
    sage_sessions = Path("/home/dp/ai-workspace/HRM/sage/raising/sessions/text")

    # Sessions to analyze
    session_ids = [111, 112, 113, 114]

    print("Analyzing SAGE Metacognitive Sessions...")
    print()

    analyzer = MetacognitiveAnalyzer()
    results = []

    for sid in session_ids:
        session_file = sage_sessions / f"session_{sid}.json"
        if not session_file.exists():
            print(f"⚠️  Session {sid} not found: {session_file}")
            continue

        metrics = analyzer.analyze_session(session_file)
        results.append(metrics)
        print_analysis_report(metrics)

    # Comparison table
    if len(results) > 1:
        compare_sessions(results)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    collapses = [r for r in results if r.pattern_type == "repetitive_collapse"]
    sustained = [r for r in results if r.pattern_type == "sustained_engagement"]

    print(f"Total sessions analyzed: {len(results)}")
    print(f"Repetitive collapses: {len(collapses)}")
    print(f"Sustained engagements: {len(sustained)}")
    print()

    if collapses:
        avg_collapse_rep = sum(r.repetition_ratio for r in collapses) / len(collapses)
        avg_collapse_depth = sum(r.philosophical_depth for r in collapses) / len(collapses)
        print(f"Collapse pattern characteristics:")
        print(f"  Avg repetition: {avg_collapse_rep*100:.1f}%")
        print(f"  Avg philosophical depth: {avg_collapse_depth:.3f}")
        print(f"  Avg duration: {sum(r.duration_seconds for r in collapses)/len(collapses):.1f}s")

    print()
    print("✅ Analysis complete!")
    print()
    print("Next: Integrate with T3/V3 Reputation Engine to track consciousness exploration")
