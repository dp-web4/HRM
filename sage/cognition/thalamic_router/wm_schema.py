#!/usr/bin/env python3
"""World Model schema — typed slots for game physics.

The codification move: replace prose world models with structured,
populatable, verifiable artifacts. Each slot has a type. The LLM
populates slots (cheaper than generating prose). The system verifies
slots against observed behavior (prediction → outcome → delta).

Design principles:
  1. Slots are typed but the VALUES are natural language — this is
     the prose↔code gradient. The schema is code; the content is
     prose-at-the-structured-end.
  2. predict() and verify() are the calibrated-prediction interface.
     A CausalRule that predicts wrong gets flagged, not silently kept.
  3. The schema is the same for all games. Only the populated content
     differs. This is the base-class pattern.
  4. Serialization is JSON — readable by humans, parseable by code,
     storable in membot cartridges.

Usage:
    from sage.cognition.thalamic_router.wm_schema import GameWorldModel, CausalRule

    # Populate from scratch or from consolidation
    wm = GameWorldModel(
        game="toy_a",
        level=0,
        objects=["basket (movable, 8 positions)", "canvas (10x10 grid)", "target pattern"],
        actions={
            "UP": "move basket counterclockwise on ring",
            "DOWN": "move basket clockwise on ring",
            "SEL": "launch paint from basket position inward",
            "CLICK": "select color from palette",
        },
        win_condition="canvas matches target at all non-diagonal positions",
        causal_rules=[
            CausalRule(
                action="SEL", condition="basket at position 0 (North)",
                predicted_effect="paints top 5 rows with active color",
                confidence=0.9, evidence_count=3,
            ),
        ],
    )

    # Render for LLM context
    prompt_text = wm.render()

    # After observing an action's effect
    wm.observe("SEL", condition="basket at position 3", actual_effect="painted rows 5-9 blue")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class CausalRule:
    """One causal rule: action + condition → predicted effect.

    This is the atom of game physics. Each rule is a discrete
    prediction that reality can verify.
    """
    action: str                    # "SEL", "CLICK at (38,46)", "DOWN"
    condition: str                 # "basket at position 3", "level 0 step > 10"
    predicted_effect: str          # "paints rows 5-9 with active color"
    confidence: float = 0.5        # 0.0 = guess, 1.0 = verified multiple times
    evidence_count: int = 0        # how many times observed
    last_verified: bool = True     # did the last observation match?
    source: str = "discovered"     # "discovered" | "template" | "human" | "consolidated"

    def matches(self, action: str, condition: str) -> bool:
        """Does this rule apply to the given action+condition?"""
        return (self.action.lower() in action.lower()
                and self.condition.lower() in condition.lower())


@dataclass
class GameWorldModel:
    """Typed world model for a game — the codification of game physics.

    Every slot has a defined role. The LLM populates slots through
    discovery probes or consolidation. The system verifies predictions
    against observed outcomes.
    """
    game: str
    level: int = 0

    # What exists in this game
    objects: List[str] = field(default_factory=list)

    # What each action does (action_name → description)
    actions: Dict[str, str] = field(default_factory=dict)

    # How to win
    win_condition: str = ""

    # Causal rules — the core physics
    causal_rules: List[CausalRule] = field(default_factory=list)

    # What has been tried and failed (anti-rules)
    failed_attempts: List[str] = field(default_factory=list)

    # Current hypothesis about what to do next
    current_strategy: str = ""

    # Plan template — the recipe the model fills in
    # Each step has fixed fields (action, phase) and variable fields (target, count, coords)
    # Variable fields marked with "?" are what the model decides
    plan_template: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    discovery_source: str = "empty"  # "empty" | "template" | "phase1" | "consolidated"
    revision_count: int = 0

    # ── Render for LLM context ────────────────────────────────

    def render(self, budget_tokens: int = 500) -> str:
        """Render as structured text for the LLM's MechanicsBlock."""
        sections = []
        sections.append(f"## {self.game} Level {self.level} — World Model")

        if self.objects:
            sections.append("Objects: " + ", ".join(self.objects))

        if self.actions:
            action_lines = [f"  {k}: {v}" for k, v in self.actions.items()]
            sections.append("Actions:\n" + "\n".join(action_lines))

        if self.win_condition:
            sections.append(f"Win: {self.win_condition}")

        if self.causal_rules:
            verified = [r for r in self.causal_rules if r.confidence >= 0.5]
            if verified:
                rule_lines = []
                for r in verified[:8]:  # top 8 by confidence
                    conf_marker = "✓" if r.evidence_count >= 2 else "?"
                    rule_lines.append(f"  {conf_marker} {r.action} when {r.condition} → {r.predicted_effect}")
                sections.append("Physics (discovered):\n" + "\n".join(rule_lines))

        if self.failed_attempts:
            sections.append("Failed: " + "; ".join(self.failed_attempts[-3:]))

        if self.current_strategy:
            sections.append(f"Strategy: {self.current_strategy}")

        if self.plan_template:
            template_lines = ["Plan template (fill in ? fields):"]
            for i, step in enumerate(self.plan_template, 1):
                parts = []
                for k, v in step.items():
                    if k.startswith("_"):
                        continue
                    parts.append(f"{k}={v}")
                template_lines.append(f"  {i}. {', '.join(parts)}")
            sections.append("\n".join(template_lines))

        text = "\n".join(sections)

        # Rough token budget enforcement
        if len(text) > budget_tokens * 4:  # ~4 chars per token
            text = text[:budget_tokens * 4] + "\n[truncated]"

        return text

    # ── Observation interface ─────────────────────────────────

    def observe(self, action: str, condition: str, actual_effect: str,
                frame_delta_pct: float = 0.0) -> Optional[CausalRule]:
        """Record an observation. Updates matching rules or creates new ones.

        Returns the rule that was updated or created, or None if the
        observation was trivial (< 1% frame change, no level change).
        """
        if frame_delta_pct < 1.0 and "level" not in actual_effect.lower():
            return None

        # Find matching rule
        for rule in self.causal_rules:
            if rule.matches(action, condition):
                rule.evidence_count += 1
                if actual_effect.lower() != rule.predicted_effect.lower():
                    rule.last_verified = False
                    rule.confidence = max(0.1, rule.confidence - 0.2)
                    rule.predicted_effect = actual_effect  # update to reality
                else:
                    rule.last_verified = True
                    rule.confidence = min(1.0, rule.confidence + 0.1)
                return rule

        # No matching rule — create new one
        new_rule = CausalRule(
            action=action,
            condition=condition,
            predicted_effect=actual_effect,
            confidence=0.5,
            evidence_count=1,
            source="discovered",
        )
        self.causal_rules.append(new_rule)
        return new_rule

    def add_failure(self, description: str):
        """Record a failed attempt for anti-pattern learning."""
        if description not in self.failed_attempts:
            self.failed_attempts.append(description)

    # ── Serialization ─────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GameWorldModel":
        rules = [CausalRule(**r) for r in d.pop("causal_rules", [])]
        return cls(**d, causal_rules=rules)

    @classmethod
    def from_json(cls, text: str) -> "GameWorldModel":
        return cls.from_dict(json.loads(text))

    # ── Load from existing prose world model ──────────────────

    @classmethod
    def from_prose(cls, game: str, prose: str, level: int = 0) -> "GameWorldModel":
        """Bootstrap from an existing prose world model markdown.

        Parses the semi-structured format (## Objects, ## Rules, etc.)
        into typed slots. Imperfect but gives a starting point.
        """
        wm = cls(game=game, level=level, discovery_source="human")

        current_section = ""
        for line in prose.split("\n"):
            stripped = line.strip()
            if stripped.startswith("## "):
                current_section = stripped[3:].lower()
                continue
            if not stripped or stripped.startswith("#"):
                continue

            if "object" in current_section:
                # Extract object names from "- **Name**: description"
                if stripped.startswith("- "):
                    obj = stripped[2:].split(":")[0].replace("**", "").strip()
                    wm.objects.append(obj)

            elif "rule" in current_section:
                if stripped.startswith("- "):
                    rule_text = stripped[2:]
                    # Try to parse "ACTION does EFFECT" patterns
                    for action in ["UP", "DOWN", "LEFT", "RIGHT", "SEL", "CLICK", "LAUNCH", "ACTION"]:
                        if action in rule_text.upper():
                            wm.causal_rules.append(CausalRule(
                                action=action,
                                condition="any",
                                predicted_effect=rule_text,
                                confidence=0.8,
                                source="human",
                            ))
                            break

            elif "win" in current_section:
                wm.win_condition = stripped

            elif "strateg" in current_section:
                if stripped.startswith("- "):
                    if not wm.current_strategy:
                        wm.current_strategy = stripped[2:]

            elif "action" in current_section:
                if stripped.startswith("- "):
                    parts = stripped[2:].split(":", 1)
                    if len(parts) == 2:
                        wm.actions[parts[0].strip()] = parts[1].strip()

        return wm
