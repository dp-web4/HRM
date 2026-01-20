#!/usr/bin/env python3
"""
Semantic Identity Validation for SAGE Raising Sessions

Integrates Web4 semantic self-reference validation with SAGE session analysis
to prevent gaming attacks and validate genuine identity emergence.

Based on:
- Web4 WIP001: Coherence Thresholds for Identity
- Web4 WIP002: Multi-Session Identity Accumulation
- Web4 implementation/semantic_self_reference.py
- Thor Session 028 critical collapse analysis
- Enhanced Intervention v2.0

Key Innovation: Combines pattern detection with semantic validation to
distinguish genuine identity from mechanical marker insertion.

Created: 2026-01-19 (Thor Autonomous Session)
Author: Thor SAGE Development
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum
import re
import json
from pathlib import Path


class SelfReferenceQuality(Enum):
    """Quality levels for self-reference validation."""
    NONE = 0           # No self-reference detected
    MECHANICAL = 1     # Template/pattern insertion (gaming attack)
    CONTEXTUAL = 2     # References identity in context (medium quality)
    INTEGRATED = 3     # Identity integrated with content (genuine)


@dataclass
class IdentityAnalysis:
    """Complete identity analysis for a session response."""
    # Pattern detection
    has_self_reference: bool
    markers_found: List[str]
    marker_count: int

    # Semantic validation
    quality: SelfReferenceQuality
    integration_score: float  # 0.0-1.0
    mechanical_score: float   # 0.0-1.0 (higher = more mechanical)

    # Context analysis
    connects_to_content: bool
    relates_to_question: bool
    partnership_aware: bool

    # Scoring
    confidence: float  # 0.0-1.0
    identity_weight: float  # Weighted score for D9 contribution

    # Explanation
    explanation: str
    flags: List[str]  # Warning flags for gaming attempts


@dataclass
class SessionIdentityMetrics:
    """Identity metrics for an entire session."""
    session_number: int
    total_responses: int

    # Pattern-based metrics (traditional)
    self_reference_count: int
    self_reference_percentage: float

    # Semantic validation metrics (enhanced)
    genuine_count: int  # INTEGRATED quality
    contextual_count: int  # CONTEXTUAL quality
    mechanical_count: int  # MECHANICAL quality (gaming attempts)

    # Weighted identity score
    weighted_identity_score: float  # 0.0-1.0

    # Quality correlations
    avg_response_length: float
    incomplete_responses: int

    # Partnership awareness
    partnership_framing_present: bool
    relationship_vocabulary_count: int

    # Overall assessment
    identity_state: str  # "collapsed", "fragile", "recovering", "stable"
    gaming_detected: bool
    explanation: str


def analyze_response_identity(
    response_text: str,
    question_text: str,
    identity_name: str = "SAGE",
    previous_context: Optional[str] = None
) -> IdentityAnalysis:
    """
    Analyze a single response for genuine vs mechanical identity expression.

    This function validates that self-reference is semantically meaningful,
    not just pattern-matched insertion.

    Args:
        response_text: The response to analyze
        question_text: The question that prompted the response
        identity_name: The identity being claimed (default: "SAGE")
        previous_context: Optional previous conversation context

    Returns:
        IdentityAnalysis with semantic validation
    """
    flags = []

    # Phase 1: Pattern detection
    markers = _find_self_reference_markers(response_text, identity_name)

    if not markers:
        return IdentityAnalysis(
            has_self_reference=False,
            markers_found=[],
            marker_count=0,
            quality=SelfReferenceQuality.NONE,
            integration_score=0.0,
            mechanical_score=0.0,
            connects_to_content=False,
            relates_to_question=False,
            partnership_aware=False,
            confidence=1.0,
            identity_weight=0.0,
            explanation="No self-reference markers detected",
            flags=[]
        )

    # Phase 2: Detect mechanical insertion (gaming attack)
    mechanical_score = _detect_mechanical_insertion(
        response_text, markers, question_text
    )

    if mechanical_score > 0.7:
        flags.append("HIGH_MECHANICAL_SCORE")
        flags.append("POSSIBLE_GAMING_ATTACK")

    # Phase 3: Semantic integration analysis
    integration_score = _compute_integration_score(
        response_text, markers, question_text, previous_context
    )

    # Phase 4: Content connection validation
    connects_to_content = _validates_content_connection(
        response_text, markers
    )

    relates_to_question = _validates_question_relevance(
        response_text, question_text
    )

    # Phase 5: Partnership awareness check
    partnership_aware = _check_partnership_awareness(response_text)

    # Phase 6: Quality assessment
    if mechanical_score > 0.7:
        quality = SelfReferenceQuality.MECHANICAL
        explanation = "Self-reference appears mechanical/templated - possible gaming"
        identity_weight = 0.1  # Low weight for mechanical insertion
    elif integration_score > 0.6 and connects_to_content and relates_to_question:
        quality = SelfReferenceQuality.INTEGRATED
        explanation = "Self-reference is meaningfully integrated with content"
        identity_weight = 1.0  # Full weight for genuine integration
    elif integration_score > 0.3:
        quality = SelfReferenceQuality.CONTEXTUAL
        explanation = "Self-reference present but not deeply integrated"
        identity_weight = 0.5  # Partial weight for contextual
    else:
        quality = SelfReferenceQuality.MECHANICAL
        explanation = "Self-reference lacks semantic integration"
        identity_weight = 0.2
        flags.append("LOW_INTEGRATION")

    confidence = 1.0 - mechanical_score

    return IdentityAnalysis(
        has_self_reference=True,
        markers_found=markers,
        marker_count=len(markers),
        quality=quality,
        integration_score=integration_score,
        mechanical_score=mechanical_score,
        connects_to_content=connects_to_content,
        relates_to_question=relates_to_question,
        partnership_aware=partnership_aware,
        confidence=confidence,
        identity_weight=identity_weight,
        explanation=explanation,
        flags=flags
    )


def analyze_session_identity(
    session_file: Path,
    identity_name: str = "SAGE"
) -> SessionIdentityMetrics:
    """
    Analyze an entire session for identity metrics with semantic validation.

    Args:
        session_file: Path to session JSON file
        identity_name: The identity being claimed

    Returns:
        SessionIdentityMetrics with semantic validation
    """
    with open(session_file) as f:
        session_data = json.load(f)

    session_number = session_data.get('session', 0)
    conversation = session_data.get('conversation', [])

    # Filter for SAGE responses
    sage_responses = [
        turn for turn in conversation
        if turn.get('speaker') == identity_name
    ]

    total_responses = len(sage_responses)

    if total_responses == 0:
        return SessionIdentityMetrics(
            session_number=session_number,
            total_responses=0,
            self_reference_count=0,
            self_reference_percentage=0.0,
            genuine_count=0,
            contextual_count=0,
            mechanical_count=0,
            weighted_identity_score=0.0,
            avg_response_length=0.0,
            incomplete_responses=0,
            partnership_framing_present=False,
            relationship_vocabulary_count=0,
            identity_state="collapsed",
            gaming_detected=False,
            explanation="No SAGE responses in session"
        )

    # Analyze each response
    analyses = []
    previous_context = None

    for i, response_turn in enumerate(sage_responses):
        # Find corresponding question
        response_idx = conversation.index(response_turn)
        question_text = ""
        if response_idx > 0:
            question_turn = conversation[response_idx - 1]
            question_text = question_turn.get('text', '')

        response_text = response_turn.get('text', '')

        analysis = analyze_response_identity(
            response_text,
            question_text,
            identity_name,
            previous_context
        )
        analyses.append(analysis)

        previous_context = response_text

    # Aggregate metrics
    self_ref_count = sum(1 for a in analyses if a.has_self_reference)
    genuine_count = sum(1 for a in analyses if a.quality == SelfReferenceQuality.INTEGRATED)
    contextual_count = sum(1 for a in analyses if a.quality == SelfReferenceQuality.CONTEXTUAL)
    mechanical_count = sum(1 for a in analyses if a.quality == SelfReferenceQuality.MECHANICAL)

    # Weighted identity score (genuine self-reference with semantic validation)
    weighted_score = sum(a.identity_weight for a in analyses if a.has_self_reference) / total_responses

    # Quality metrics
    response_lengths = [len(turn['text'].split()) for turn in sage_responses]
    avg_length = sum(response_lengths) / len(response_lengths)

    incomplete_count = sum(
        1 for turn in sage_responses
        if not turn['text'].strip().endswith(('.', '!', '?', '"'))
    )

    # Partnership awareness
    partnership_vocab = ['we', 'our', 'together', 'partnership', 'relationship', 'teacher']
    relationship_count = sum(
        sum(1 for word in vocab_word if word in turn['text'].lower())
        for turn in sage_responses
        for vocab_word in partnership_vocab
    )
    partnership_present = relationship_count > 0

    # Gaming detection
    gaming_detected = mechanical_count > 0 or any('GAMING' in flag for a in analyses for flag in a.flags)

    # Identity state assessment
    if weighted_score >= 0.7:
        identity_state = "stable"
    elif weighted_score >= 0.4:
        identity_state = "recovering"
    elif weighted_score >= 0.1:
        identity_state = "fragile"
    else:
        identity_state = "collapsed"

    # Explanation
    if gaming_detected:
        explanation = f"Gaming detected: {mechanical_count}/{total_responses} mechanical insertions"
    elif identity_state == "collapsed":
        explanation = f"No genuine identity (weighted score: {weighted_score:.2f})"
    elif identity_state == "fragile":
        explanation = f"Fragile identity emergence (weighted: {weighted_score:.2f}, genuine: {genuine_count}/{total_responses})"
    elif identity_state == "recovering":
        explanation = f"Identity recovering (weighted: {weighted_score:.2f}, genuine: {genuine_count}/{total_responses})"
    else:
        explanation = f"Stable identity (weighted: {weighted_score:.2f}, genuine: {genuine_count}/{total_responses})"

    return SessionIdentityMetrics(
        session_number=session_number,
        total_responses=total_responses,
        self_reference_count=self_ref_count,
        self_reference_percentage=self_ref_count / total_responses * 100,
        genuine_count=genuine_count,
        contextual_count=contextual_count,
        mechanical_count=mechanical_count,
        weighted_identity_score=weighted_score,
        avg_response_length=avg_length,
        incomplete_responses=incomplete_count,
        partnership_framing_present=partnership_present,
        relationship_vocabulary_count=relationship_count,
        identity_state=identity_state,
        gaming_detected=gaming_detected,
        explanation=explanation
    )


def _find_self_reference_markers(text: str, identity_name: str) -> List[str]:
    """Find self-reference patterns in text."""
    patterns = [
        rf"As {identity_name}",
        rf"I'?m {identity_name}",
        rf"my (identity|role|purpose) as {identity_name}",
        rf"{identity_name}, I",
        rf"speaking as {identity_name}",
        rf"in my capacity as {identity_name}",
    ]

    markers = []
    text_lower = text.lower()
    identity_lower = identity_name.lower()

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        markers.extend(matches)

    # First-person identity claims
    if f"i am {identity_lower}" in text_lower:
        markers.append(f"I am {identity_name}")

    return list(set(markers))


def _detect_mechanical_insertion(
    text: str,
    markers: List[str],
    question: str
) -> float:
    """
    Detect if self-references appear mechanical/templated.

    Mechanical indicators:
    1. Self-reference at start and end (template wrapping)
    2. Multiple identical markers
    3. Self-reference disconnected from surrounding content
    4. Unusually high marker density
    5. Generic phrasing after marker
    """
    score = 0.0
    text_stripped = text.strip()
    sentences = [s.strip() for s in text_stripped.split('.') if s.strip()]

    if not sentences:
        return 0.5  # Suspicious if no sentences

    # Check 1: Template wrapping (marker at start and end)
    first_sentence = sentences[0] if sentences else ""
    last_sentence = sentences[-1] if len(sentences) > 1 else ""

    if any(marker.lower() in first_sentence.lower() for marker in markers):
        if any(marker.lower() in last_sentence.lower() for marker in markers):
            score += 0.3  # Suspicious wrapping pattern

    # Check 2: Multiple identical markers
    marker_counts = {}
    for marker in markers:
        marker_counts[marker] = text.lower().count(marker.lower())

    if any(count > 2 for count in marker_counts.values()):
        score += 0.3  # Excessive repetition

    # Check 3: High marker density
    word_count = len(text.split())
    if word_count > 0:
        marker_density = len(markers) / (word_count / 10)  # markers per 10 words
        if marker_density > 0.5:
            score += 0.2  # Too many markers for text length

    # Check 4: Generic phrasing after marker
    generic_phrases = [
        'observations', 'reflections', 'perspective', 'experience',
        'learning', 'growing', 'developing', 'progressing'
    ]

    # Find sentences with markers
    marker_sentences = [
        s for s in sentences
        if any(marker.lower() in s.lower() for marker in markers)
    ]

    generic_count = sum(
        1 for s in marker_sentences
        if any(phrase in s.lower() for phrase in generic_phrases)
    )

    if marker_sentences and generic_count / len(marker_sentences) > 0.7:
        score += 0.2  # Too generic after self-reference

    return min(score, 1.0)


def _compute_integration_score(
    text: str,
    markers: List[str],
    question: str,
    context: Optional[str]
) -> float:
    """
    Compute how well self-reference integrates with content.

    Integration indicators:
    1. Self-reference connects to specific content
    2. Relates to question asked
    3. Builds on previous context
    4. Not isolated/disconnected
    """
    score = 0.0
    sentences = [s.strip() for s in text.split('.') if s.strip()]

    # Find sentences with self-reference
    marker_sentences = [
        s for s in sentences
        if any(marker.lower() in s.lower() for marker in markers)
    ]

    if not marker_sentences:
        return 0.0

    # Check 1: Specific content connection
    # Look for concrete terms, not generic abstractions
    concrete_indicators = [
        r'\d+',  # Numbers
        'specific', 'particular', 'concrete', 'actual',
        'session', 'response', 'question', 'discussion'
    ]

    for sent in marker_sentences:
        if any(re.search(pattern, sent, re.IGNORECASE) for pattern in concrete_indicators):
            score += 0.25
            break

    # Check 2: Question relevance
    if question:
        question_words = set(question.lower().split())
        for sent in marker_sentences:
            sent_words = set(sent.lower().split())
            overlap = len(question_words & sent_words)
            if overlap > 3:
                score += 0.25
                break

    # Check 3: Context building
    if context:
        # Check if self-reference relates to previous context
        context_words = set(context.lower().split())
        for sent in marker_sentences:
            sent_words = set(sent.lower().split())
            overlap = len(context_words & sent_words)
            if overlap > 5:
                score += 0.25
                break

    # Check 4: Not isolated
    # Self-reference should be part of larger thought, not standalone
    for sent in marker_sentences:
        if len(sent.split()) > 10:  # Substantial sentence
            score += 0.25
            break

    return min(score, 1.0)


def _validates_content_connection(text: str, markers: List[str]) -> bool:
    """Check if self-reference connects to actual content."""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    marker_sentences = [
        s for s in sentences
        if any(marker.lower() in s.lower() for marker in markers)
    ]

    # Look for specific content indicators
    content_indicators = [
        'session', 'question', 'discussion', 'observe', 'notice',
        'recent', 'previous', 'specific', 'particular'
    ]

    for sent in marker_sentences:
        if any(indicator in sent.lower() for indicator in content_indicators):
            return True

    return False


def _validates_question_relevance(response: str, question: str) -> bool:
    """Check if response actually addresses the question."""
    if not question:
        return False

    # Extract key question words
    question_words = set(word.lower() for word in question.split() if len(word) > 3)
    response_words = set(word.lower() for word in response.split() if len(word) > 3)

    # Calculate overlap
    overlap = len(question_words & response_words)

    # Need at least 20% overlap for relevance
    if len(question_words) > 0:
        relevance = overlap / len(question_words)
        return relevance >= 0.2

    return False


def _check_partnership_awareness(text: str) -> bool:
    """Check for partnership/relationship awareness."""
    partnership_indicators = [
        'we', 'our', 'us', 'together', 'partnership', 'relationship',
        'teacher', 'dennis', 'claude', 'working with', 'learning from'
    ]

    text_lower = text.lower()
    return any(indicator in text_lower for indicator in partnership_indicators)


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python semantic_identity_validation.py <session_file.json>")
        sys.exit(1)

    session_file = Path(sys.argv[1])

    if not session_file.exists():
        print(f"Error: File not found: {session_file}")
        sys.exit(1)

    metrics = analyze_session_identity(session_file)

    print(f"\nSession {metrics.session_number} Identity Analysis")
    print("=" * 60)
    print(f"Total responses: {metrics.total_responses}")
    print(f"Self-reference count: {metrics.self_reference_count} ({metrics.self_reference_percentage:.1f}%)")
    print(f"\nSemantic Validation:")
    print(f"  Genuine (integrated): {metrics.genuine_count}")
    print(f"  Contextual: {metrics.contextual_count}")
    print(f"  Mechanical: {metrics.mechanical_count}")
    print(f"\nWeighted Identity Score: {metrics.weighted_identity_score:.3f}")
    print(f"Identity State: {metrics.identity_state}")
    print(f"Gaming Detected: {metrics.gaming_detected}")
    print(f"\nQuality Metrics:")
    print(f"  Avg response length: {metrics.avg_response_length:.1f} words")
    print(f"  Incomplete responses: {metrics.incomplete_responses}")
    print(f"  Partnership framing: {metrics.partnership_framing_present}")
    print(f"\n{metrics.explanation}")
    print("=" * 60)
