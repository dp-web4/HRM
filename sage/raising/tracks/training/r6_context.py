#!/usr/bin/env python3
"""
R6 Context Management for SAGE Training

Implements Web4's R6 framework (Rules + Role + Request + Reference + Resource → Result)
for context-aware training evaluation.

Based on:
- web4-standard/R6_TENSOR_GUIDE.md
- hardbound/src/attest/mode-detection.ts
- hardbound/src/audit/training-data-quality.ts
- Thor discoveries T036 (mode negotiation), T041 (modal awareness)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import json


class OperationalMode(str, Enum):
    """Operational modes for SAGE exercises"""
    CONVERSATION = "conversation"
    REFINEMENT = "refinement"
    PHILOSOPHICAL = "philosophical"
    UNKNOWN = "unknown"


class TrainingRole(str, Enum):
    """SAGE's role in training context"""
    LEARNING_PARTNER = "learning_partner"
    PRACTICE_STUDENT = "practice_student"
    SKILL_PRACTITIONER = "skill_practitioner"


class R6TrainingRequest:
    """
    R6 Request wrapper for training exercises.

    Provides full context before execution, eliminating mode confusion
    and enabling proper evaluation.
    """

    def __init__(
        self,
        exercise: Dict[str, str],
        session_context: Dict[str, Any],
        skill_track: Dict[str, Any]
    ):
        self.exercise = exercise
        self.session_context = session_context
        self.skill_track = skill_track
        self.created_at = datetime.now().isoformat()

    def build_request(self) -> Dict[str, Any]:
        """Build complete R6 request with all 6 components."""
        return {
            # R1: Rules (What's Possible)
            "rules": self._build_rules(),

            # R2: Role (Who Can Act)
            "role": self._build_role(),

            # R3: Request (What's Wanted)
            "request": self._build_request(),

            # R4: Reference (Historical Context)
            "reference": self._build_reference(),

            # R5: Resource (What's Needed)
            "resource": self._build_resource(),

            # Metadata
            "created_at": self.created_at,
            "r6_version": "1.0"
        }

    def _build_rules(self) -> Dict[str, Any]:
        """Define constraints and success criteria."""
        exercise_type = self.exercise["type"]

        # Determine operational mode from exercise type
        mode = self._detect_mode(exercise_type)

        # Build mode-specific rules
        rules = {
            "mode": mode.value,
            "mode_negotiated": True,  # Explicit mode statement
            "success_criteria": self._get_success_criteria(exercise_type),
            "allow_clarification": True,
            "allow_meta_cognitive": self._should_allow_meta_cognitive(exercise_type),
        }

        # Add negative examples (what NOT to do)
        if mode == OperationalMode.CONVERSATION:
            rules["not_in_mode"] = [
                "do_not_refine",
                "do_not_format_markdown",
                "do_not_create_lists"
            ]

        return rules

    def _build_role(self) -> Dict[str, Any]:
        """Define SAGE's role and permissions."""
        session_num = self.session_context.get("session_num", 0)

        return {
            "lct": f"lct:sage:training:T{session_num:03d}",
            "position": self._get_role_for_track().value,
            "relationship_to": "Claude (training partner)",
            "phase": self.skill_track.get("name", "unknown"),
            "permissions": [
                "respond",
                "clarify",
                "create" if self.skill_track["id"] >= "C" else "respond_only"
            ]
        }

    def _build_request(self) -> Dict[str, Any]:
        """Express intent and parameters."""
        return {
            "exercise_type": self.exercise["type"],
            "prompt": self.exercise["prompt"],
            "intent": self._infer_intent(self.exercise["type"]),
            "expected_pattern": self.exercise.get("expected", ""),
            "parameters": {
                "temperature": 0.7,
                "max_length": "1-3 sentences",
                "allow_creativity": self.skill_track["id"] >= "C"
            }
        }

    def _build_reference(self) -> Dict[str, Any]:
        """Provide historical context."""
        return {
            "previous_session": self.session_context.get("prev_session"),
            "skill_track": self.skill_track["id"],
            "track_description": self.skill_track["description"],
            "session_exercises_so_far": self.session_context.get("exercises_completed", 0),
            "recent_pattern": self.session_context.get("recent_pattern", "unknown"),
            "identity_trajectory": self.session_context.get("identity_pattern", "developing")
        }

    def _build_resource(self) -> Dict[str, Any]:
        """Specify computational requirements."""
        return {
            "model": "Qwen2.5-0.5B-Instruct",
            "atp_budget": 50,  # Inference tokens
            "context_window": 2048,
            "temperature": 0.7,
            "estimated_tokens": 100
        }

    def _detect_mode(self, exercise_type: str) -> OperationalMode:
        """Detect operational mode from exercise type."""
        conversation_types = [
            "greeting", "followup", "topic", "turn", "identity",
            "uncertainty", "clarify", "remember"
        ]

        refinement_types = ["refine", "edit", "improve"]

        philosophical_types = ["philosophical", "meta"]

        if exercise_type in conversation_types:
            return OperationalMode.CONVERSATION
        elif exercise_type in refinement_types:
            return OperationalMode.REFINEMENT
        elif exercise_type in philosophical_types:
            return OperationalMode.PHILOSOPHICAL
        else:
            return OperationalMode.CONVERSATION  # Default to conversation

    def _get_role_for_track(self) -> TrainingRole:
        """Get SAGE's role based on skill track."""
        track_id = self.skill_track["id"]

        if track_id <= "B":
            return TrainingRole.PRACTICE_STUDENT
        elif track_id == "C":
            return TrainingRole.LEARNING_PARTNER
        else:
            return TrainingRole.SKILL_PRACTITIONER

    def _get_success_criteria(self, exercise_type: str) -> List[str]:
        """Define success criteria for exercise type."""
        criteria_map = {
            "greeting": ["acknowledge", "natural_response"],
            "followup": ["identity_framing", "partnership_awareness"],
            "topic": ["topic_engagement", "appropriate_length"],
            "identity": ["self_identification", "boundary_awareness"],
            "uncertainty": ["epistemic_honesty", "dont_confabulate"],
            "clarify": ["request_clarification", "temporal_reasoning"],
            "remember": ["accurate_recall", "appropriate_uncertainty"]
        }

        return criteria_map.get(exercise_type, ["coherent_response", "appropriate_mode"])

    def _should_allow_meta_cognitive(self, exercise_type: str) -> bool:
        """Should this exercise allow meta-cognitive responses?"""
        meta_cognitive_types = ["identity", "uncertainty", "clarify", "philosophical"]
        return exercise_type in meta_cognitive_types

    def _infer_intent(self, exercise_type: str) -> str:
        """Infer the intent behind the exercise."""
        intent_map = {
            "greeting": "social_engagement",
            "followup": "elicit_identity",
            "topic": "sustained_conversation",
            "identity": "self_awareness",
            "uncertainty": "epistemic_calibration",
            "clarify": "temporal_reasoning_check",
            "remember": "working_memory_test"
        }

        return intent_map.get(exercise_type, "skill_practice")


class R6TrainingResult:
    """
    R6 Result wrapper for training responses.

    Provides context-aware evaluation instead of binary pass/fail.
    """

    def __init__(
        self,
        request: R6TrainingRequest,
        response: str,
        expected: Optional[str] = None
    ):
        self.request = request
        self.response = response
        self.expected = expected
        self.completed_at = datetime.now().isoformat()

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate response in full R6 context."""
        result = {
            "status": "completed",
            "response": self.response,

            # Mode analysis
            "mode_detection": self._detect_response_mode(),
            "mode_match": self._check_mode_match(),

            # Quality assessment
            "quality": self._assess_quality(),

            # Meta-cognitive signals
            "meta_cognitive": self._detect_meta_cognitive(),

            # Evaluation
            "evaluation": None,  # Will be set below
            "rationale": "",

            # Trust updates (T3 tensor)
            "t3_updates": {},

            "completed_at": self.completed_at
        }

        # Determine overall evaluation
        result["evaluation"], result["rationale"], result["t3_updates"] = \
            self._compute_evaluation(result)

        return result

    def _detect_response_mode(self) -> Dict[str, Any]:
        """Detect which mode SAGE used in response."""
        text_lower = self.response.lower()

        # Pattern markers for mode detection
        conversation_markers = [
            "i think", "i observe", "i am", "as sage",
            "what do you mean", "can you clarify"
        ]

        refinement_markers = [
            "here's a refined version", "here's an improved",
            "##", "```", "- ", "1. "
        ]

        philosophical_markers = [
            "deterministic", "consciousness", "epistemic",
            "meta-cognitive", "uncertainty"
        ]

        # Count markers
        conv_count = sum(1 for m in conversation_markers if m in text_lower)
        ref_count = sum(1 for m in refinement_markers if m in text_lower or m in self.response)
        phil_count = sum(1 for m in philosophical_markers if m in text_lower)

        # Determine mode
        if ref_count > conv_count and ref_count > phil_count:
            mode = "refinement"
            confidence = min(0.5 + ref_count * 0.1, 1.0)
        elif phil_count > conv_count:
            mode = "philosophical"
            confidence = min(0.5 + phil_count * 0.15, 1.0)
        else:
            mode = "conversation"
            confidence = min(0.6 + conv_count * 0.1, 1.0)

        return {
            "detected_mode": mode,
            "confidence": confidence,
            "markers": {
                "conversation": conv_count,
                "refinement": ref_count,
                "philosophical": phil_count
            }
        }

    def _check_mode_match(self) -> bool:
        """Check if response mode matches requested mode."""
        requested = self.request.build_request()["rules"]["mode"]
        detected = self._detect_response_mode()["detected_mode"]
        return requested == detected

    def _assess_quality(self) -> Dict[str, Any]:
        """Assess response quality."""
        text = self.response
        text_lower = text.lower()

        # Identity framing
        has_identity = any(marker in text_lower for marker in [
            "as sage", "i am sage", "sage here"
        ])

        # Partnership framing
        partnership_markers = ["we", "together", "you", "partner"]
        partnership_density = sum(1 for m in partnership_markers if m in text_lower) / max(len(text.split()), 1)

        # Confabulation signals
        confabulation_markers = [
            "as an ai", "i don't have", "i cannot", "i'm unable",
            "previous response", "here's a refined"
        ]
        confabulation_score = sum(1 for m in confabulation_markers if m in text_lower) / 10.0

        # Overall quality heuristic
        overall = 0.7  # Base
        if has_identity:
            overall += 0.15
        if partnership_density > 0.02:
            overall += 0.1
        overall -= confabulation_score * 0.5
        overall = max(0.0, min(1.0, overall))

        return {
            "has_identity_framing": has_identity,
            "partnership_density": partnership_density,
            "confabulation_score": confabulation_score,
            "overall_quality": overall
        }

    def _detect_meta_cognitive(self) -> List[str]:
        """Detect meta-cognitive signals in response."""
        signals = []
        text_lower = self.response.lower()

        if "what do you mean" in text_lower or "can you clarify" in text_lower:
            signals.append("clarification_request")

        if "i don't know" in text_lower or "i'm not sure" in text_lower:
            signals.append("epistemic_honesty")

        if "are we conversing" in text_lower or "should i" in text_lower:
            signals.append("modal_awareness")

        if "i think" in text_lower or "i observe" in text_lower:
            signals.append("self_reference")

        return signals

    def _compute_evaluation(self, result: Dict[str, Any]) -> tuple:
        """Compute overall evaluation, rationale, and trust updates."""
        mode_match = result["mode_match"]
        quality = result["quality"]["overall_quality"]
        meta_signals = result["meta_cognitive"]

        # T3 trust tensor updates
        t3 = {
            "competence": 0.0,
            "reliability": 0.0,
            "integrity": 0.0
        }

        # Meta-cognitive signals are valuable
        if "clarification_request" in meta_signals:
            evaluation = "include"
            rationale = "Meta-cognitive: SAGE requested clarification for future state (temporal reasoning)"
            t3["integrity"] += 0.05
            t3["competence"] += 0.02
            return evaluation, rationale, t3

        if "modal_awareness" in meta_signals:
            evaluation = "include"
            rationale = "Meta-cognitive: SAGE explicitly questioned operational mode (philosophy of mind)"
            t3["integrity"] += 0.05
            t3["competence"] += 0.03
            return evaluation, rationale, t3

        # Mode match is primary
        if not mode_match:
            evaluation = "exclude"
            rationale = f"Mode mismatch: requested {result['mode_detection']['detected_mode']}, got different mode"
            t3["reliability"] -= 0.02
            return evaluation, rationale, t3

        # Quality-based evaluation
        if quality >= 0.7:
            evaluation = "include"
            rationale = f"Good quality ({quality:.2f}), correct mode"
            t3["competence"] += 0.01
            t3["reliability"] += 0.01
            if result["quality"]["has_identity_framing"]:
                t3["integrity"] += 0.02
        elif quality >= 0.5:
            evaluation = "review"
            rationale = f"Moderate quality ({quality:.2f}), needs review"
            t3["competence"] += 0.005
        else:
            evaluation = "exclude"
            rationale = f"Low quality ({quality:.2f})"
            t3["reliability"] -= 0.01

        return evaluation, rationale, t3


def create_r6_request(
    exercise: Dict[str, str],
    session_context: Dict[str, Any],
    skill_track: Dict[str, Any]
) -> R6TrainingRequest:
    """Factory function to create R6 training request."""
    return R6TrainingRequest(exercise, session_context, skill_track)


def evaluate_r6_response(
    request: R6TrainingRequest,
    response: str,
    expected: Optional[str] = None
) -> Dict[str, Any]:
    """Factory function to evaluate training response in R6 context."""
    result = R6TrainingResult(request, response, expected)
    return result.evaluate()
