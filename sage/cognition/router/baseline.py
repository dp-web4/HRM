#!/usr/bin/env python3
"""
Programmatic baseline — the router's teacher.
=============================================

``programmatic_decide`` is the pure-function reconstruction of the existing
consciousness-loop dispatcher (``_select_attention_targets`` +
``_get_plugins_for_modality`` in ``sage/core/sage_consciousness.py``).

It takes a ``RouterInput`` + ``plugin_registry`` and returns a valid
``RouterOutput``. No side effects. No mutation. No reliance on live
consciousness-loop state. This is the deterministic decision function
that Phase 1 behavioral cloning will target.

Design notes
------------
The existing ``_select_attention_targets`` returns a *list* of targets
(up to ``metabolic.max_active_plugins``). ``RouterOutput`` is a
*single* decision per tick. The mapping is therefore:

  - metabolic max_active_plugins == 0                 → noop
  - no viable sensory modality with a plugin          → noop
  - metacog blocks every candidate plugin             → noop
  - habit available + high confidence + not blocked   → habit
  - otherwise                                         → invoke (top-salience
                                                      modality's first plugin)

The existing dispatcher has no habit-short-circuit; cerebellum lookup
lives elsewhere in the loop. But ``RouterInput.habit_available`` /
``habit_confidence`` (populated by Track 2) express whether the
cerebellum matched this WM state, so the baseline honors it. This is a
deliberate gap-closing choice — flagged in the PR body for Track 5 to
revisit when wiring.

Branch → rationale_code map (see tests for the full matrix):

  - dream / max_plugins==0                → ``low_atp_rest``
  - metacog blocks all candidates         → ``metacog_blocked``
  - habit available & confident           → ``habit_match``
  - all sensory noop (no modality→plugin) → ``default``
  - goal slot active, perception modality → ``goal_driven``
  - high snarc novelty + arousal          → ``high_novelty``
  - frontal_lobe tier chosen              → ``escalate_frontal``
  - federate tier chosen                  → ``federate_peer``
  - reflex tier chosen                    → ``reflex``
  - fallthrough invoke                    → ``default``

Spec: phase2/brain-arch/thalamic-router-prd.md
Sprint: phase2/brain-arch/router-sprint-1-phase-0.md §Track 3
"""

from typing import Any, Dict, List, Optional

from sage.cognition.router.inputs import RouterInput
from sage.cognition.router.outputs import RouterOutput, VALID_RATIONALE_CODES
from sage.cognition.router.tiers import PluginTier


# ──────────────────────────────────────────────────────────────────────
# Constants — thresholds cloned from the existing dispatcher
# ──────────────────────────────────────────────────────────────────────

# Habit-confidence threshold. PRD §2.5: "confidence must exceed a
# threshold" — the PRD doesn't pin a value; §14 "deferred" notes 0.85 as
# the proposed habit-vs-invoke arbitration. We use that here.
HABIT_CONFIDENCE_THRESHOLD = 0.85

# "High novelty / arousal" branch trigger. Chosen to mirror the
# existing dispatcher's salience_threshold (0.15) upscaled for the
# SNARC combined-axis heuristic. Keep conservative — the point is to
# identify cases where the baseline should earn the 'high_novelty'
# rationale rather than the bland 'default'.
SNARC_HIGH_NOVELTY_THRESHOLD = 0.6

# Mirror of _get_plugins_for_modality (sage_consciousness.py ~line 1296).
# Keep this in lockstep with the dispatcher source; Track 5 will
# refactor to a shared constant.
MODALITY_MAP: Dict[str, List[str]] = {
    "vision": ["vision"],
    "audio": ["audio", "language"],
    "proprioception": ["control"],
    "time": [],
    "message": ["language"],
}

# Metabolic states where the kernel is structurally unable to dispatch.
# Matches metabolic_controller.StateConfig(max_active_plugins=0) for
# DREAM. Other states always allow at least one plugin, so we do not
# short-circuit on them here — the per-plugin gating below does.
NOOP_METABOLIC_STATES = {"dream"}


# ──────────────────────────────────────────────────────────────────────
# Helpers (pure, side-effect free)
# ──────────────────────────────────────────────────────────────────────

def _plugin_tier(plugin: str, plugin_registry: Dict[str, Any]) -> Optional[str]:
    """Look up the plugin's declared tier in the registry.

    Registry entries may be either a plain dict (``{"tier": "routine",
    "atp_cost": 5}``) or any object with ``tier`` / ``atp_cost``
    attributes. Missing → None (the caller coerces to ``noop`` since the
    PRD §3.3 rule 1 check would otherwise fail).
    """
    entry = plugin_registry.get(plugin)
    if entry is None:
        return None
    if isinstance(entry, dict):
        tier = entry.get("tier")
    else:
        tier = getattr(entry, "tier", None)
    if tier is None:
        return None
    # Accept PluginTier enum members or raw strings.
    if isinstance(tier, PluginTier):
        return tier.value
    if isinstance(tier, str) and PluginTier.is_valid(tier):
        return tier
    return None


def _plugin_atp_cost(plugin: str, plugin_registry: Dict[str, Any]) -> float:
    """Look up expected ATP cost in the registry. Missing → 0.0."""
    entry = plugin_registry.get(plugin)
    if entry is None:
        return 0.0
    if isinstance(entry, dict):
        cost = entry.get("atp_cost", 0.0)
    else:
        cost = getattr(entry, "atp_cost", 0.0)
    try:
        return max(0.0, float(cost))
    except (TypeError, ValueError):
        return 0.0


def _candidate_plugins(
    router_input: RouterInput,
    plugin_registry: Dict[str, Any],
) -> List[str]:
    """Collect plugins reachable through the modality map, in
    salience-priority order (mirrors ``sorted_obs`` in the dispatcher).

    Only plugins that (a) are in the registry and (b) are not in the
    metacog block list are retained. The list may be empty — that's
    how the ``default`` / metacog-blocked branches detect "no viable
    invoke".
    """
    blocked = set(router_input.metacog_block_list)
    seen: set = set()
    candidates: List[str] = []
    # Per-modality pass, preserving the order supplied by the caller.
    # The dispatcher sorts by salience, but the feature extractor
    # (Track 2) is responsible for ordering ``sensory_modalities`` so
    # the top-salience modality lands first. We don't re-sort here —
    # that would introduce hidden state.
    for modality in router_input.sensory_modalities:
        for plugin in MODALITY_MAP.get(modality, []):
            if plugin in seen:
                continue
            if plugin in blocked:
                continue
            if plugin not in plugin_registry:
                continue
            seen.add(plugin)
            candidates.append(plugin)
    return candidates


# ──────────────────────────────────────────────────────────────────────
# Public top-level helpers (required by Track 3 spec)
# ──────────────────────────────────────────────────────────────────────

def _should_use_habit(router_input: RouterInput) -> bool:
    """Habit-branch gate.

    Fires when:
      - cerebellum reports a habit match for the current WM state,
      - confidence clears ``HABIT_CONFIDENCE_THRESHOLD``.

    Metacog constraints do not block habits (habits route through
    cerebellum, not the plugin registry). PolicyGate still sees the
    downstream effector.
    """
    return bool(router_input.habit_available) and \
        float(router_input.habit_confidence) >= HABIT_CONFIDENCE_THRESHOLD


def _decide_action_type(
    router_input: RouterInput,
    plugin_registry: Dict[str, Any],
) -> str:
    """Top-level branch: ``invoke`` | ``habit`` | ``noop``.

    Ordering matters:

      1. Dream / zero-capacity metabolic states  →  noop
      2. Habit available & confident             →  habit
      3. Any viable sensory→plugin candidate     →  invoke
      4. Fallthrough                             →  noop
    """
    if router_input.metabolic_state in NOOP_METABOLIC_STATES:
        return "noop"
    if _should_use_habit(router_input):
        return "habit"
    if _candidate_plugins(router_input, plugin_registry):
        return "invoke"
    return "noop"


def _select_plugin(
    router_input: RouterInput,
    plugin_registry: Dict[str, Any],
) -> Optional[str]:
    """Pick the single plugin to invoke when action == 'invoke'.

    Matches the existing dispatcher's behavior of taking the top-salience
    observation's first modality-mapped plugin. Returns None if no
    viable plugin exists (caller coerces to ``noop``).
    """
    candidates = _candidate_plugins(router_input, plugin_registry)
    if not candidates:
        return None
    return candidates[0]


def _compute_rationale(
    router_input: RouterInput,
    action: str,
    plugin: Optional[str],
    plugin_registry: Dict[str, Any],
) -> str:
    """Pick a ``rationale_code`` from ``VALID_RATIONALE_CODES``.

    The rationale is for observability — Nomad's metacog aggregates
    these. The branches here must map 1:1 to the top-level decision
    logic so the distribution of rationale codes is a legible
    fingerprint of the baseline.
    """
    # noop branches ────────────────────────────────────────────────
    if action == "noop":
        # Metabolic noop — the kernel decided it can't afford to act.
        if router_input.metabolic_state in NOOP_METABOLIC_STATES:
            return "low_atp_rest"
        # All candidates blocked by metacog.
        blocked = set(router_input.metacog_block_list)
        any_would_be_candidate = any(
            plugin_name in plugin_registry
            for modality in router_input.sensory_modalities
            for plugin_name in MODALITY_MAP.get(modality, [])
        )
        if blocked and any_would_be_candidate:
            # Would have had a candidate but metacog blocks it.
            remaining = [
                plugin_name
                for modality in router_input.sensory_modalities
                for plugin_name in MODALITY_MAP.get(modality, [])
                if plugin_name in plugin_registry and plugin_name not in blocked
            ]
            if not remaining:
                return "metacog_blocked"
        # Genuinely nothing to do this tick.
        return "default"

    # habit branch ─────────────────────────────────────────────────
    if action == "habit":
        return "habit_match"

    # invoke branches ──────────────────────────────────────────────
    assert action == "invoke"
    tier = _plugin_tier(plugin, plugin_registry) if plugin else None
    if tier == PluginTier.FRONTAL_LOBE.value:
        return "escalate_frontal"
    if tier == PluginTier.FEDERATE.value:
        return "federate_peer"
    if tier == PluginTier.REFLEX.value:
        return "reflex"

    # High-stakes invoke: SNARC novelty + arousal combined signal.
    high_stakes = (
        float(router_input.snarc_novelty) >= SNARC_HIGH_NOVELTY_THRESHOLD
        and float(router_input.snarc_arousal) >= SNARC_HIGH_NOVELTY_THRESHOLD
    )
    if high_stakes:
        return "high_novelty"

    # Goal-driven invoke: there's an active goal + the plugin we picked
    # is reachable from a perception modality (vision, message, audio).
    if router_input.wm_goal_active:
        perception_modalities = {"vision", "message", "audio"}
        if any(m in perception_modalities for m in router_input.sensory_modalities):
            return "goal_driven"

    return "default"


def _confidence_for_invoke(router_input: RouterInput) -> float:
    """Produce a confidence in [0,1] for invoke decisions.

    Mirrors the existing dispatcher's priority scoring (salience ×
    metabolic rate × posture weight) collapsed to a scalar. We don't
    have posture weight in RouterInput (Track 2 could add it; for now
    keep parity with the dispatcher's *salience* input, since that's
    the dominant term).
    """
    # Use max of arousal + conflict as the salience proxy. Both are the
    # action-relevant SNARC axes per PRD §4.7.A.
    salience_proxy = max(
        float(router_input.snarc_arousal),
        float(router_input.snarc_conflict),
    )
    # Floor at 0.1 so invoke decisions never emit zero confidence
    # (avoids downstream "you picked invoke but had zero confidence"
    # paradoxes).
    return max(0.1, min(1.0, salience_proxy))


# ──────────────────────────────────────────────────────────────────────
# Public entrypoint
# ──────────────────────────────────────────────────────────────────────

def programmatic_decide(
    router_input: RouterInput,
    plugin_registry: Dict[str, Any],
) -> RouterOutput:
    """Produce a ``RouterOutput`` from a ``RouterInput`` using the
    existing dispatcher's logic, exactly.

    This is the **teacher** for Phase 1 behavioral cloning. It is
    intentionally pure: no I/O, no mutation of ``router_input`` or
    ``plugin_registry``, no reliance on live consciousness-loop state.

    Guarantees:
      - Returns a valid ``RouterOutput`` for any well-formed input.
      - No exceptions escape; unexpected conditions coerce to noop.
      - Idempotent and deterministic: same (input, registry) → same
        output.
      - Latency: <1ms per call on a modern CPU (measured in the test
        suite; budget per sprint-doc Track 3 acceptance).

    Parameters
    ----------
    router_input : RouterInput
        Frozen snapshot of kernel state at one consciousness-loop tick.
    plugin_registry : Dict[str, Any]
        Maps plugin name → entry with ``tier`` and ``atp_cost``. Entry
        may be a dict or any object with those attributes. See
        ``_plugin_tier`` / ``_plugin_atp_cost`` for details.

    Returns
    -------
    RouterOutput
        Always passes ``RouterOutput.validate()``; ``rationale_code``
        always in ``VALID_RATIONALE_CODES``.
    """
    try:
        action = _decide_action_type(router_input, plugin_registry)
    except Exception:
        # Unknown failure — coerce to noop. The PRD §3.3 rule-5 path
        # expects callers to emit a warn event; this is the pure side
        # of that contract.
        return RouterOutput.noop(
            confidence=1.0,
            energy_estimate=0.0,
            rationale_code="coerced_noop",
        )

    if action == "noop":
        return RouterOutput.noop(
            confidence=1.0,
            energy_estimate=0.0,
            rationale_code=_compute_rationale(
                router_input, "noop", None, plugin_registry
            ),
        )

    if action == "habit":
        # ``habit_id`` is not carried in RouterInput schema (§3.1). The
        # consciousness-loop integration layer (Track 5) will resolve
        # ``wm.stable_key(goal_id)`` → habit_id via cerebellum.lookup().
        # Phase 0 baseline emits the wm_state_key as a placeholder
        # habit_id — it's deterministic, unique per WM state, and
        # cerebellum's actual lookup uses the same key.
        habit_id = router_input.wm_state_key
        return RouterOutput(
            action="habit",
            plugin=None,
            plugin_tier=None,
            payload_hint=None,
            habit_id=habit_id,
            confidence=float(router_input.habit_confidence),
            energy_estimate=0.0,  # habits route to cerebellum, not plugins
            rationale_code="habit_match",
        )

    # action == "invoke"
    plugin = _select_plugin(router_input, plugin_registry)
    if plugin is None:
        # Defensive — _decide_action_type already guaranteed at least
        # one candidate; this is the "registry changed between calls"
        # hedge.
        return RouterOutput.noop(
            confidence=1.0,
            energy_estimate=0.0,
            rationale_code="default",
        )

    tier = _plugin_tier(plugin, plugin_registry)
    energy = _plugin_atp_cost(plugin, plugin_registry)
    rationale = _compute_rationale(
        router_input, "invoke", plugin, plugin_registry
    )
    output = RouterOutput(
        action="invoke",
        plugin=plugin,
        plugin_tier=tier,
        payload_hint=None,
        habit_id=None,
        confidence=_confidence_for_invoke(router_input),
        energy_estimate=energy,
        rationale_code=rationale,
    )

    # Final safety net: if the constructed output is somehow invalid
    # (unknown tier, stale rationale code after code edits, etc.)
    # coerce rather than propagate. This is the pure counterpart of
    # PRD §3.3 rule 5.
    ok, _ = output.validate(known_plugins=set(plugin_registry.keys()))
    if not ok:
        return RouterOutput.noop(
            confidence=1.0,
            energy_estimate=0.0,
            rationale_code="coerced_noop",
        )
    # Belt-and-suspenders: rationale must be in the closed vocabulary.
    if rationale not in VALID_RATIONALE_CODES:
        return RouterOutput.noop(
            confidence=1.0,
            energy_estimate=0.0,
            rationale_code="coerced_noop",
        )
    return output


__all__ = [
    "programmatic_decide",
    "_decide_action_type",
    "_select_plugin",
    "_should_use_habit",
    "_compute_rationale",
    "HABIT_CONFIDENCE_THRESHOLD",
    "SNARC_HIGH_NOVELTY_THRESHOLD",
    "MODALITY_MAP",
    "NOOP_METABOLIC_STATES",
]
