#!/usr/bin/env python3
"""
RouterOutput — normative schema produced by the thalamic router.
================================================================

Every consciousness-loop tick, the router emits one ``RouterOutput``.
The decision space is closed and finite (PRD §1.2): ``invoke`` / ``habit``
/ ``noop``. Validation rules from PRD §3.3 are enforced by
``RouterOutput.validate()``.

Spec: phase2/brain-arch/thalamic-router-prd.md §3.2, §3.3
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

from sage.cognition.router.tiers import PluginTier


# ──────────────────────────────────────────────────────────────────────
# Constants — closed vocabularies
# ──────────────────────────────────────────────────────────────────────

VALID_ACTIONS = {"invoke", "habit", "noop"}

# PRD §3.2: "rationale_code is a small closed vocabulary like ...".
# We seed the vocabulary with the examples called out in the PRD.
# Additions are cheap (string literals) but should be coordinated so
# Nomad's metacog can aggregate them without drift.
VALID_RATIONALE_CODES = {
    "high_novelty",
    "habit_match",
    "low_atp_rest",
    "metacog_blocked",
    "goal_driven",
    "reflex",
    "escalate_frontal",
    "federate_peer",
    # Fallbacks used when the programmatic baseline (Track 3) cannot
    # classify more specifically, and by the validator on coercion:
    "default",
    "coerced_noop",
}


# ──────────────────────────────────────────────────────────────────────
# RouterOutput
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RouterOutput:
    """Structured routing decision.

    Fields, order, and names track PRD §3.2 exactly. See
    ``validate()`` for the rule enforcement path and ``coerce_to_noop``
    for the PRD §3.3 rule-5 escape hatch.
    """

    action: str                             # 'invoke' | 'habit' | 'noop'

    # Per-action fields (None if not applicable)
    plugin: Optional[str]
    plugin_tier: Optional[str]              # PluginTier value or None
    payload_hint: Optional[str]             # short canonical cue, NOT NL
    habit_id: Optional[str]

    # Metadata
    confidence: float                       # 0-1
    energy_estimate: float                  # expected ATP cost, ≥ 0
    rationale_code: str                     # closed vocabulary

    # ──────────────────────────────────────────────────────────────
    # Lightweight type guards (not a substitute for validate())
    # ──────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if not isinstance(self.action, str):
            raise TypeError("action must be str")
        if not isinstance(self.confidence, (int, float)):
            raise TypeError("confidence must be numeric")
        if not isinstance(self.energy_estimate, (int, float)):
            raise TypeError("energy_estimate must be numeric")
        if not isinstance(self.rationale_code, str):
            raise TypeError("rationale_code must be str")
        # Optional fields — if set, must be str
        for f_name in ("plugin", "plugin_tier", "payload_hint", "habit_id"):
            v = getattr(self, f_name)
            if v is not None and not isinstance(v, str):
                raise TypeError(f"{f_name} must be str or None, got {type(v).__name__}")

    # ──────────────────────────────────────────────────────────────
    # PRD §3.3 validation
    # ──────────────────────────────────────────────────────────────

    def validate(self, *, known_plugins: Optional[set] = None,
                 known_habits: Optional[set] = None) -> Tuple[bool, Optional[str]]:
        """Enforce PRD §3.3 rules.

        Returns ``(ok, reason)``:
            - ``ok`` is True if the output is valid and may be dispatched.
            - ``reason`` is a short machine-readable failure code when
              ``ok`` is False. The caller (consciousness loop integration
              in Track 5) uses it to emit a ``warn`` / ``coerce_noop``
              event and substitute a noop.

        ``known_plugins`` / ``known_habits`` are optional registries.
        When provided, rule 1 / rule 2 check membership; when omitted,
        the membership portion is skipped (useful for unit tests that
        don't have a live registry).

        PRD §3.3 rules:
          1. If action == 'invoke': plugin must be set and in the registry.
          2. If action == 'habit': habit_id must be set and exist in cerebellum.
          3. If action == 'noop': all optional fields None.
          4. confidence ∈ [0, 1], energy_estimate ≥ 0.
          5. Invalid outputs are coerced to noop + warn event.  (Rule 5
             is implemented by callers via ``coerce_to_noop``; validate()
             only reports.)
        """
        # Rule 4 first — cheapest, catches NaN/out-of-range early.
        if not 0.0 <= float(self.confidence) <= 1.0:
            return False, "confidence_out_of_range"
        if float(self.energy_estimate) < 0.0:
            return False, "energy_estimate_negative"

        if self.action not in VALID_ACTIONS:
            return False, "unknown_action"

        if self.rationale_code not in VALID_RATIONALE_CODES:
            return False, "unknown_rationale_code"

        # plugin_tier, if set, must be a valid PluginTier value.
        if self.plugin_tier is not None and not PluginTier.is_valid(self.plugin_tier):
            return False, "unknown_plugin_tier"

        if self.action == "invoke":
            # Rule 1
            if not self.plugin:
                return False, "invoke_missing_plugin"
            if known_plugins is not None and self.plugin not in known_plugins:
                return False, "invoke_unknown_plugin"
            # habit_id must NOT be set
            if self.habit_id is not None:
                return False, "invoke_with_habit_id"

        elif self.action == "habit":
            # Rule 2
            if not self.habit_id:
                return False, "habit_missing_habit_id"
            if known_habits is not None and self.habit_id not in known_habits:
                return False, "habit_unknown_habit_id"
            # plugin / plugin_tier / payload_hint must NOT be set
            if self.plugin is not None or self.plugin_tier is not None or self.payload_hint is not None:
                return False, "habit_with_plugin_fields"

        elif self.action == "noop":
            # Rule 3
            if (self.plugin is not None or self.plugin_tier is not None
                    or self.payload_hint is not None or self.habit_id is not None):
                return False, "noop_with_optional_fields"

        return True, None

    def is_valid(self, *, known_plugins: Optional[set] = None,
                 known_habits: Optional[set] = None) -> bool:
        """Boolean shorthand around :meth:`validate`."""
        ok, _ = self.validate(known_plugins=known_plugins, known_habits=known_habits)
        return ok

    def validate_or_raise(self, *, known_plugins: Optional[set] = None,
                          known_habits: Optional[set] = None) -> None:
        """Raise ``ValueError`` with the failure reason if validation fails.

        Convenience for tests and callers that prefer exceptions over
        return codes.
        """
        ok, reason = self.validate(known_plugins=known_plugins, known_habits=known_habits)
        if not ok:
            raise ValueError(f"RouterOutput invalid: {reason}")

    # ──────────────────────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouterOutput":
        known = {
            "action", "plugin", "plugin_tier", "payload_hint", "habit_id",
            "confidence", "energy_estimate", "rationale_code",
        }
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    # ──────────────────────────────────────────────────────────────
    # Factories
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def noop(
        cls,
        *,
        confidence: float = 1.0,
        energy_estimate: float = 0.0,
        rationale_code: str = "default",
    ) -> "RouterOutput":
        """Construct a valid ``action='noop'`` output. Optional fields
        left None."""
        return cls(
            action="noop",
            plugin=None,
            plugin_tier=None,
            payload_hint=None,
            habit_id=None,
            confidence=float(confidence),
            energy_estimate=float(energy_estimate),
            rationale_code=rationale_code,
        )

    @classmethod
    def coerce_to_noop(
        cls, _bad: "RouterOutput", reason: str
    ) -> "RouterOutput":
        """PRD §3.3 rule 5: invalid outputs are coerced to noop.

        The caller is responsible for emitting the corresponding warn
        event — this method returns only the replacement output.
        Rationale code is fixed to ``coerced_noop`` so metacog can
        aggregate coercions as a first-class signal (mass coercion =
        router drift or registry skew).

        The ``reason`` string is captured in the output's rationale only
        via the ``coerced_noop`` code; the detailed reason lives in the
        emitted warn event (see ``events.py``).
        """
        del _bad, reason  # both are used by the caller to build the warn event
        return cls.noop(
            confidence=1.0,
            energy_estimate=0.0,
            rationale_code="coerced_noop",
        )
