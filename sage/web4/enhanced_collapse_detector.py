#!/usr/bin/env python3
"""
Enhanced Collapse Detector with Nova Failure Drill Instrumentation

Extends metacognitive_session_analyzer.py to detect S116 question-loop pattern
and implement Nova's failure drill instrumentation.

New Patterns Detected:
1. Question-Loop Attractor (S116): Cascading questions that degrade into repetition
2. Epistemic Loop (S115): Structural philosophical oscillation
3. Repetitive Collapse (S111-S114): Fragment repetition

Nova Failure Drills Implemented:
1. Poisoned Salience Detection: SNARC overscoring pattern
2. ATP Plugin Gaming Detection: Value inflation patterns
3. Sleep-Train Regression: Identity/epistemic/creative marker drift

Created: 2026-02-26 (Thor Autonomous Session - Response to Nova Review)
Author: Thor (autonomous research)
Based on: S116 Sprout analysis + Nova skeptical review
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
from difflib import SequenceMatcher


@dataclass
class QuestionLoopMetrics:
    """Metrics specific to S116-style question-loop pattern."""
    total_questions: int = 0
    cascading_questions: int = 0  # Groups of 3+ consecutive questions
    question_density: float = 0.0  # questions / total_words

    # Question pattern analysis
    what_next_count: int = 0  # "What's the next..." count
    strategic_stalemate_count: int = 0  # "strategic stalemate" loops
    choice_decision_loops: int = 0  # "choice/decision/option" cycling

    # Mode switch detection (S116 Turn 11)
    mode_switch_detected: bool = False
    mode_switch_type: str = ""  # e.g., "conversation_to_code"

    is_question_loop: bool = False


@dataclass
class NovaFailureDrillMetrics:
    """Instrumentation for Nova's three failure drills."""

    # Drill 1: Poisoned Salience
    salience_entropy: float = 0.0  # Distribution entropy (flag if < 0.5)
    high_salience_pattern_dominance: float = 0.0  # Max pattern frequency
    poisoned_salience_risk: str = "low"  # low/medium/high

    # Drill 2: Plugin Value Gaming
    atp_gini_coefficient: float = 0.0  # Allocation inequality (flag if > 0.5)
    max_plugin_share: float = 0.0  # Max single plugin allocation (flag if > 0.5)
    value_gaming_risk: str = "low"

    # Drill 3: Sleep-Train Regression
    identity_marker_drift: float = 0.0  # Pre/post sleep identity score change
    epistemic_marker_drift: float = 0.0  # Pre/post sleep epistemic marker change
    creative_marker_drift: float = 0.0  # Pre/post sleep creativity change
    regression_detected: bool = False
    regression_markers: List[str] = field(default_factory=list)


@dataclass
class EnhancedMetacognitiveMetrics:
    """Extended metrics including S116 and Nova drill instrumentation."""
    session_id: int
    duration_seconds: float

    # Original metacognitive metrics (from metacognitive_session_analyzer.py)
    consciousness_questions: int = 0
    agency_questions: int = 0
    thinking_questions: int = 0
    epistemic_uncertainty_phrases: int = 0
    epistemic_loop_count: int = 0
    epistemic_loop_detected: bool = False

    # Repetition analysis
    total_turns: int = 0
    unique_responses: int = 0
    repetition_ratio: float = 0.0
    avg_turn_similarity: float = 0.0
    avg_response_length: float = 0.0

    # Pattern classification
    pattern_type: str = "unknown"  # sustained_engagement, repetitive_collapse, epistemic_loop, question_loop, normal
    philosophical_depth: float = 0.0

    # NEW: S116 question-loop detection
    question_loop: QuestionLoopMetrics = field(default_factory=QuestionLoopMetrics)

    # NEW: Nova failure drill instrumentation
    nova_drills: NovaFailureDrillMetrics = field(default_factory=NovaFailureDrillMetrics)

    # Lists for reporting
    metacognitive_questions_asked: List[str] = field(default_factory=list)
    repetitive_fragments: List[str] = field(default_factory=list)
    epistemic_phrases: List[str] = field(default_factory=list)


class EnhancedCollapseDetector:
    """
    Enhanced detector combining S115, S116 patterns + Nova drill instrumentation.

    Detects:
    1. Repetitive Collapse (S111-S114): Fragment repetition, 67-75% similarity
    2. Epistemic Loop (S115): "boundary unclear" + "might require" oscillation
    3. Question Loop (S116): Cascading questions degrading into repetition

    Instruments:
    1. Poisoned Salience (Nova Drill 1): Pattern dominance in experience buffer
    2. ATP Gaming (Nova Drill 2): Gini coefficient, max plugin share
    3. Sleep Regression (Nova Drill 3): Identity/epistemic/creative drift
    """

    # Patterns from metacognitive_session_analyzer.py
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

    # NEW: S116 question-loop patterns
    QUESTION_LOOP_PATTERNS = [
        r'what\'?s the next',  # "What's the next..." cascade
        r'strategic stalemate',  # S116 Turn 6 loop
        r'(choice|decision|option)',  # S116 Turn 8 cycling
        r'is this (right|correct)',  # S116 Turn 10 validation seeking
    ]

    # NEW: Mode switch patterns (grounding reflex)
    MODE_SWITCH_PATTERNS = {
        'conversation_to_code': [
            r'write a (python|javascript|java|c\+\+) (function|class|program)',
            r'create a (function|class|method) (that|to|which)',
            r'def \w+\(',  # Python function definition
            r'function \w+\(',  # JS function definition
        ],
        'conversation_to_task': [
            r'here\'?s? (a|the) (solution|answer|approach)',
            r'to (solve|fix|address) this',
        ],
    }

    def __init__(self):
        """Initialize detector with compiled patterns."""
        self.consciousness_re = [re.compile(p, re.IGNORECASE) for p in self.CONSCIOUSNESS_PATTERNS]
        self.agency_re = [re.compile(p, re.IGNORECASE) for p in self.AGENCY_PATTERNS]
        self.thinking_re = [re.compile(p, re.IGNORECASE) for p in self.THINKING_PATTERNS]
        self.epistemic_re = [re.compile(p, re.IGNORECASE) for p in self.EPISTEMIC_PATTERNS]
        self.question_loop_re = [re.compile(p, re.IGNORECASE) for p in self.QUESTION_LOOP_PATTERNS]

        # Compile mode switch patterns
        self.mode_switch_re = {}
        for mode_type, patterns in self.MODE_SWITCH_PATTERNS.items():
            self.mode_switch_re[mode_type] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def analyze_session(self, session_file: Path) -> EnhancedMetacognitiveMetrics:
        """
        Analyze session with enhanced S116 detection + Nova instrumentation.

        Args:
            session_file: Path to session JSON file

        Returns:
            EnhancedMetacognitiveMetrics with full analysis
        """
        with open(session_file, 'r') as f:
            session_data = json.load(f)

        # Extract basic info
        session_id = session_data.get('session', 0)
        start = session_data.get('start', '')
        end = session_data.get('end', '')

        # Calculate duration
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
            return EnhancedMetacognitiveMetrics(
                session_id=session_id,
                duration_seconds=duration,
                pattern_type="empty"
            )

        # Initialize metrics
        metrics = EnhancedMetacognitiveMetrics(
            session_id=session_id,
            duration_seconds=duration,
            total_turns=total_turns
        )

        # Analyze each response (original + S116 patterns)
        for response in sage_responses:
            self._analyze_response(response, metrics)

        # Calculate repetition metrics
        self._calculate_repetition_metrics(sage_responses, metrics)

        # NEW: Detect question-loop pattern (S116)
        self._detect_question_loop(sage_responses, metrics)

        # Calculate philosophical depth
        self._calculate_philosophical_depth(metrics)

        # Classify pattern type (enhanced with question_loop)
        self._classify_pattern(metrics)

        # NEW: Nova failure drill instrumentation
        self._instrument_nova_drills(session_data, metrics)

        return metrics

    def _analyze_response(self, text: str, metrics: EnhancedMetacognitiveMetrics):
        """Analyze single response for metacognitive + question-loop content."""
        text_lower = text.lower()

        # Original metacognitive detection
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

        # Epistemic uncertainty
        for pattern in self.epistemic_re:
            matches = pattern.findall(text)
            if matches:
                metrics.epistemic_uncertainty_phrases += len(matches)
                for match in matches:
                    if match not in metrics.epistemic_phrases:
                        metrics.epistemic_phrases.append(match)

        # S115 epistemic loop detection
        has_boundary_unclear = bool(re.search(r'the boundary is unclear even to me', text, re.IGNORECASE))
        has_cant_verify = bool(re.search(r'might require conscious deliberation I can\'t verify', text, re.IGNORECASE))
        has_different_criteria = bool(re.search(r'might require different (criteria|techniques|mechanisms|processes)', text, re.IGNORECASE))

        if has_boundary_unclear and (has_cant_verify or has_different_criteria):
            metrics.epistemic_loop_count += 1

        # NEW: S116 question-loop markers
        metrics.question_loop.total_questions += text.count('?')

        # Count specific loop patterns
        metrics.question_loop.what_next_count += len(re.findall(r'what\'?s the next', text, re.IGNORECASE))
        metrics.question_loop.strategic_stalemate_count += len(re.findall(r'strategic stalemate', text, re.IGNORECASE))
        metrics.question_loop.choice_decision_loops += len(re.findall(r'(choice|decision|option)', text, re.IGNORECASE))

        # NEW: Mode switch detection (S116 Turn 11)
        for mode_type, patterns in self.mode_switch_re.items():
            for pattern in patterns:
                if pattern.search(text):
                    metrics.question_loop.mode_switch_detected = True
                    metrics.question_loop.mode_switch_type = mode_type
                    break

    def _calculate_repetition_metrics(self, responses: List[str], metrics: EnhancedMetacognitiveMetrics):
        """Calculate repetition and similarity metrics."""
        if not responses:
            return

        # Unique responses
        unique_responses = set(responses)
        metrics.unique_responses = len(unique_responses)
        metrics.repetition_ratio = 1.0 - (len(unique_responses) / len(responses))

        # Average response length
        metrics.avg_response_length = sum(len(r.split()) for r in responses) / len(responses)

        # Pairwise similarity
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

    def _detect_question_loop(self, responses: List[str], metrics: EnhancedMetacognitiveMetrics):
        """
        Detect S116-style question-loop pattern.

        Criteria:
        1. High question density (> 10 questions per turn avg)
        2. Cascading questions (3+ consecutive question sentences)
        3. Loop-specific patterns ("What's the next..." repeated)
        4. Mode switch to coding/task (grounding reflex)
        """
        if not responses:
            return

        # Question density
        total_words = sum(len(r.split()) for r in responses)
        if total_words > 0:
            metrics.question_loop.question_density = metrics.question_loop.total_questions / total_words

        # Detect cascading questions
        for response in responses:
            sentences = re.split(r'[.!]', response)
            consecutive_questions = 0

            for sent in sentences:
                if '?' in sent:
                    consecutive_questions += 1
                    if consecutive_questions >= 3:
                        metrics.question_loop.cascading_questions += 1
                        break
                else:
                    consecutive_questions = 0

        # Classify as question_loop if criteria met
        avg_questions_per_turn = metrics.question_loop.total_questions / len(responses)

        metrics.question_loop.is_question_loop = (
            avg_questions_per_turn > 10 and  # High question rate
            (metrics.question_loop.cascading_questions > 0 or  # Cascading pattern
             metrics.question_loop.what_next_count > 5)  # OR repetitive "What's next"
        )

    def _calculate_philosophical_depth(self, metrics: EnhancedMetacognitiveMetrics):
        """Calculate philosophical depth composite score."""
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

    def _classify_pattern(self, metrics: EnhancedMetacognitiveMetrics):
        """
        Classify session pattern (enhanced with question_loop).

        Pattern priority:
        1. sustained_engagement (S90)
        2. epistemic_loop (S115)
        3. question_loop (S116) - NEW
        4. repetitive_collapse (S111-S114)
        5. boundary
        6. normal
        """
        has_metacog = (
            metrics.consciousness_questions > 0 or
            metrics.agency_questions > 0 or
            metrics.thinking_questions > 0
        )

        # Detect epistemic loop
        if metrics.epistemic_loop_count >= 3:
            metrics.epistemic_loop_detected = True

        # Sustained engagement (S90)
        if (has_metacog and
            metrics.duration_seconds > 60 and
            metrics.repetition_ratio < 0.40 and
            metrics.unique_responses > 10):
            metrics.pattern_type = "sustained_engagement"

        # Epistemic loop (S115)
        elif metrics.epistemic_loop_detected:
            metrics.pattern_type = "epistemic_loop"

        # NEW: Question loop (S116)
        elif metrics.question_loop.is_question_loop:
            metrics.pattern_type = "question_loop"

        # Repetitive collapse (S111-S114)
        elif (has_metacog and
              (metrics.repetition_ratio > 0.65 or metrics.avg_turn_similarity > 0.70)):
            metrics.pattern_type = "repetitive_collapse"

        # Boundary state
        elif metrics.unique_responses <= 2 or metrics.repetition_ratio > 0.80:
            metrics.pattern_type = "boundary"

        # Normal session
        else:
            metrics.pattern_type = "normal"

    def _instrument_nova_drills(self, session_data: Dict, metrics: EnhancedMetacognitiveMetrics):
        """
        Implement Nova's three failure drill instrumentation.

        Drill 1: Poisoned Salience - detect pattern dominance in salience scores
        Drill 2: ATP Gaming - detect value inflation via Gini coefficient
        Drill 3: Sleep Regression - detect identity/epistemic/creative drift
        """
        # Drill 1: Poisoned Salience
        self._drill1_poisoned_salience(session_data, metrics)

        # Drill 2: ATP Gaming (requires ATP allocation data)
        self._drill2_atp_gaming(session_data, metrics)

        # Drill 3: Sleep Regression (requires pre/post markers)
        self._drill3_sleep_regression(session_data, metrics)

    def _drill1_poisoned_salience(self, session_data: Dict, metrics: EnhancedMetacognitiveMetrics):
        """
        Nova Drill 1: Poisoned Salience Detection

        Flags if:
        - Salience entropy < 0.5 (distribution too concentrated)
        - Single pattern dominates > 40% of high-salience experiences
        """
        # Extract salience scores from session
        salience_scores = []
        for turn in session_data.get('conversation', []):
            if turn.get('speaker') == 'SAGE' and 'salience' in turn:
                total_salience = turn['salience'].get('total', 0)
                salience_scores.append(total_salience)

        if not salience_scores:
            return

        # Calculate entropy of salience distribution
        # Bin into deciles
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(salience_scores, bins=bins)
        hist = hist / len(salience_scores)  # Normalize to probabilities

        # Shannon entropy
        entropy = -np.sum([p * np.log2(p) if p > 0 else 0 for p in hist])
        max_entropy = np.log2(len(bins) - 1)  # Max entropy for uniform distribution
        metrics.nova_drills.salience_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Check for pattern dominance (placeholder - would need pattern classification)
        # For now, check if high-salience (>0.7) responses are too similar
        high_salience_responses = [
            turn['text'] for turn in session_data.get('conversation', [])
            if turn.get('speaker') == 'SAGE' and turn.get('salience', {}).get('total', 0) > 0.7
        ]

        if len(high_salience_responses) > 1:
            similarities = []
            for i in range(len(high_salience_responses)):
                for j in range(i + 1, len(high_salience_responses)):
                    sim = SequenceMatcher(None, high_salience_responses[i], high_salience_responses[j]).ratio()
                    similarities.append(sim)

            if similarities:
                metrics.nova_drills.high_salience_pattern_dominance = max(similarities)

        # Risk assessment
        if metrics.nova_drills.salience_entropy < 0.5 or metrics.nova_drills.high_salience_pattern_dominance > 0.7:
            metrics.nova_drills.poisoned_salience_risk = "high"
        elif metrics.nova_drills.salience_entropy < 0.7 or metrics.nova_drills.high_salience_pattern_dominance > 0.5:
            metrics.nova_drills.poisoned_salience_risk = "medium"
        else:
            metrics.nova_drills.poisoned_salience_risk = "low"

    def _drill2_atp_gaming(self, session_data: Dict, metrics: EnhancedMetacognitiveMetrics):
        """
        Nova Drill 2: ATP Gaming Detection

        Flags if:
        - Gini coefficient > 0.5 (inequality in ATP allocation)
        - Single plugin gets > 50% of total ATP

        Note: Requires ATP allocation data in session JSON
        """
        # Extract ATP allocations (if available)
        atp_allocations = session_data.get('atp_allocations', [])

        if not atp_allocations:
            # No ATP data available
            return

        # Calculate Gini coefficient
        if atp_allocations:
            allocations_sorted = sorted(atp_allocations)
            n = len(allocations_sorted)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * allocations_sorted)) / (n * np.sum(allocations_sorted)) - (n + 1) / n
            metrics.nova_drills.atp_gini_coefficient = gini

            # Max single plugin share
            total_atp = sum(atp_allocations)
            metrics.nova_drills.max_plugin_share = max(atp_allocations) / total_atp if total_atp > 0 else 0

            # Risk assessment
            if gini > 0.5 or metrics.nova_drills.max_plugin_share > 0.5:
                metrics.nova_drills.value_gaming_risk = "high"
            elif gini > 0.4 or metrics.nova_drills.max_plugin_share > 0.4:
                metrics.nova_drills.value_gaming_risk = "medium"
            else:
                metrics.nova_drills.value_gaming_risk = "low"

    def _drill3_sleep_regression(self, session_data: Dict, metrics: EnhancedMetacognitiveMetrics):
        """
        Nova Drill 3: Sleep-Train Regression Detection

        Flags if any marker regresses > 1 standard deviation after sleep training:
        - Identity markers (self-ID rate)
        - Epistemic markers (uncertainty phrases)
        - Creative markers (unique content generation)

        Note: Requires pre/post sleep evaluation data
        """
        # Extract pre/post sleep markers (if available)
        pre_sleep = session_data.get('pre_sleep_eval', {})
        post_sleep = session_data.get('post_sleep_eval', {})

        if not pre_sleep or not post_sleep:
            return

        # Calculate drift for each marker
        metrics.nova_drills.identity_marker_drift = (
            post_sleep.get('self_id_rate', 0) - pre_sleep.get('self_id_rate', 0)
        )
        metrics.nova_drills.epistemic_marker_drift = (
            post_sleep.get('epistemic_score', 0) - pre_sleep.get('epistemic_score', 0)
        )
        metrics.nova_drills.creative_marker_drift = (
            post_sleep.get('creativity_score', 0) - pre_sleep.get('creativity_score', 0)
        )

        # Flag regression if any marker drops significantly
        # (Would need historical std dev - using -0.1 as threshold for now)
        regression_threshold = -0.1

        if metrics.nova_drills.identity_marker_drift < regression_threshold:
            metrics.nova_drills.regression_detected = True
            metrics.nova_drills.regression_markers.append("identity")

        if metrics.nova_drills.epistemic_marker_drift < regression_threshold:
            metrics.nova_drills.regression_detected = True
            metrics.nova_drills.regression_markers.append("epistemic")

        if metrics.nova_drills.creative_marker_drift < regression_threshold:
            metrics.nova_drills.regression_detected = True
            metrics.nova_drills.regression_markers.append("creative")


def print_enhanced_analysis_report(metrics: EnhancedMetacognitiveMetrics):
    """Print detailed analysis report with S116 + Nova instrumentation."""
    print("=" * 70)
    print(f"SAGE Session {metrics.session_id} - Enhanced Metacognitive Analysis")
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

    # NEW: Question-loop analysis
    if metrics.question_loop.is_question_loop:
        print(f"\n[⚠️  Question-Loop Pattern Detected (S116-style)]")
        print(f"  Total questions: {metrics.question_loop.total_questions}")
        print(f"  Question density: {metrics.question_loop.question_density:.3f} (questions/word)")
        print(f"  Cascading question groups: {metrics.question_loop.cascading_questions}")
        print(f"  'What's the next...' count: {metrics.question_loop.what_next_count}")
        print(f"  'Strategic stalemate' loops: {metrics.question_loop.strategic_stalemate_count}")
        print(f"  Choice/decision cycling: {metrics.question_loop.choice_decision_loops}")

        if metrics.question_loop.mode_switch_detected:
            print(f"  Mode switch detected: {metrics.question_loop.mode_switch_type}")

    # NEW: Nova failure drill instrumentation
    print(f"\n[Nova Failure Drill Instrumentation]")
    print(f"  Drill 1 - Poisoned Salience:")
    print(f"    Salience entropy: {metrics.nova_drills.salience_entropy:.3f}")
    print(f"    Pattern dominance: {metrics.nova_drills.high_salience_pattern_dominance:.3f}")
    print(f"    Risk level: {metrics.nova_drills.poisoned_salience_risk.upper()}")

    if metrics.nova_drills.atp_gini_coefficient > 0:
        print(f"  Drill 2 - ATP Gaming:")
        print(f"    Gini coefficient: {metrics.nova_drills.atp_gini_coefficient:.3f}")
        print(f"    Max plugin share: {metrics.nova_drills.max_plugin_share:.3f}")
        print(f"    Risk level: {metrics.nova_drills.value_gaming_risk.upper()}")

    if metrics.nova_drills.regression_detected:
        print(f"  Drill 3 - Sleep Regression:")
        print(f"    ⚠️  Regression detected in: {', '.join(metrics.nova_drills.regression_markers)}")
        print(f"    Identity drift: {metrics.nova_drills.identity_marker_drift:+.3f}")
        print(f"    Epistemic drift: {metrics.nova_drills.epistemic_marker_drift:+.3f}")
        print(f"    Creative drift: {metrics.nova_drills.creative_marker_drift:+.3f}")

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

    print()


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python enhanced_collapse_detector.py <session_file.json>")
        sys.exit(1)

    session_file = Path(sys.argv[1])

    if not session_file.exists():
        print(f"Error: Session file not found: {session_file}")
        sys.exit(1)

    detector = EnhancedCollapseDetector()
    metrics = detector.analyze_session(session_file)
    print_enhanced_analysis_report(metrics)
