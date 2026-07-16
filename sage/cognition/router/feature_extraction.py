#!/usr/bin/env python3
"""
Router feature extraction — Phase 0 Track 2.
=============================================

Turns live kernel state into a normative ``RouterInput`` (PRD §3.1).

Each component of the kernel (WM, SNARC, metabolic, episodic, cerebellum,
RPE, metacog, sensory, plugin registry, cartridge) has its own
sub-extractor — a pure function that takes the component (or ``None``)
and returns a dict of the fields it owns. The top-level
``extract_router_input`` assembles the final ``RouterInput`` from those
dicts.

Design rules (binding, from sprint doc + PRD):

  1. **No torch**.  Router data pipeline must run on edge Sprout with
     no torch overhead (sprint doc "Constraints").
  2. **JSON-serializable**.  Every emitted value is deterministic-
     serializable — ``RouterInput.__post_init__`` enforces this.
  3. **Mock-friendly**.  Every sub-extractor accepts ``None`` and
     returns sensible defaults.  Components that may be legitimately
     absent in Phase 0 (RPE before Phase 4, episodic pre-integration,
     cartridge pending Andy's routing-role wiring) must not raise.
  4. **Feature vector length is fixed** regardless of which optional
     components are ``None``.  This is a binding contract for the
     router model: if the feature layout changes across calls, no
     training corpus survives.
  5. **Failure isolation**.  If a sub-extractor raises, log a warning
     and return defaults — never propagate exceptions up into the
     consciousness loop.  This mirrors the Track 5 integration-layer
     philosophy (sprint doc: "pipeline failures must NEVER break the
     consciousness loop").
  6. **Pure functions**.  Sub-extractors MUST NOT mutate the
     components they read.  Safe to call repeatedly.

Spec:
  phase2/brain-arch/thalamic-router-prd.md §2, §3.1
  phase2/brain-arch/router-sprint-1-phase-0.md (Track 2)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional

from sage.cognition.router.inputs import (
    CARTRIDGE_EMBEDDING_DIM,
    RouterInput,
    VALID_ATP_TRENDS,
    VALID_METABOLIC_STATES,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Module-level defaults (single source of truth)
# ──────────────────────────────────────────────────────────────────────

# Canonical "no WM" state_key.  16 hex chars matches ``wm.stable_key``
# shape so downstream hashing invariants don't accidentally depend on
# shorter strings.  We deliberately choose a constant rather than
# "whatever an empty WM returns" so records from "missing WM" are
# distinguishable from real ones.
_EMPTY_WM_STATE_KEY = "0" * 16

# WM capacity fallback.  Real WM objects always expose ``.capacity``
# (Miller 4±3 upper; default 7), but a duck-typed stub may not.  If
# ``capacity`` is missing or <=0, we treat pressure as 0.
_DEFAULT_WM_CAPACITY = 7

# Metabolic fallbacks when the controller is absent.
_DEFAULT_METABOLIC_STATE = "wake"
_DEFAULT_ATP_LEVEL = 0.0
_DEFAULT_ATP_TREND = "stable"

# SNARC/sensory defaults
_DEFAULT_UNIT = 0.0

# RPE default prior: uniform over the 3 decision classes.  Matches the
# PriorTable default (0.5 center) reasonably — but more importantly,
# sums to 1.0 so downstream consumers that interpret priors as a
# distribution still see a valid distribution.
_UNIFORM_PRIOR = 1.0 / 3.0

# Cap the sensory_modalities list length defensively — plugin registry
# should keep this small, but a bad caller could flood it.  RouterInput
# docstring says "bounded by plugin registry size".
_MAX_SENSORY_MODALITIES = 32
_MAX_METACOG_BLOCK_LIST = 64


# ──────────────────────────────────────────────────────────────────────
# Sub-extractors
#
# Each sub-extractor:
#   - Takes the component (may be None) + any ancillary kwargs it needs
#   - Returns a dict whose keys map 1:1 to RouterInput fields
#   - Never raises on bad/None input (wraps its body in try/except and
#     logs at WARNING if the component was provided but unusable)
#   - Is pure — no mutation of the component
# ──────────────────────────────────────────────────────────────────────

def _extract_wm_features(
    wm: Optional[Any],
    *,
    goal_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract ``wm_*`` fields from a ``WorkingMemory``-like object.

    Fields owned (PRD §3.1):
      - ``wm_state_key``     : ``wm.stable_key(goal_id)``, 16-char hex
      - ``wm_slot_counts``   : per-type counts of active slots
      - ``wm_goal_active``   : any "goal" slot present (optionally filtered
                                to ``goal_id``)
      - ``wm_age_ticks``     : ticks since last WM write (best-effort; 0
                                if the component doesn't expose ticks)
      - ``wm_pressure``      : len(slots) / capacity, clamped to [0, 1]

    ``None`` input → defaults (``_EMPTY_WM_STATE_KEY``, empty counts,
    no active goal, zero age, zero pressure).
    """
    if wm is None:
        return {
            "wm_state_key": _EMPTY_WM_STATE_KEY,
            "wm_slot_counts": {},
            "wm_goal_active": False,
            "wm_age_ticks": 0,
            "wm_pressure": 0.0,
        }

    try:
        # wm.stable_key(goal_id) — deterministic hash.  Missing/raising
        # → sentinel.  We don't re-hash ourselves; the PRD contract is
        # "whatever wm.stable_key returns", 16-char hex.
        try:
            state_key = wm.stable_key(goal_id) if hasattr(wm, "stable_key") else _EMPTY_WM_STATE_KEY
        except Exception as e:
            logger.warning("wm.stable_key raised: %s", e)
            state_key = _EMPTY_WM_STATE_KEY
        if not isinstance(state_key, str) or not state_key:
            state_key = _EMPTY_WM_STATE_KEY

        # Per-type slot counts.  Read from ``.slots`` dict if present.
        slot_counts: Dict[str, int] = {}
        slots = getattr(wm, "slots", None)
        if isinstance(slots, dict):
            for slot in slots.values():
                content_type = getattr(slot, "content_type", None)
                if goal_id is not None:
                    slot_goal = getattr(slot, "goal_id", None)
                    if slot_goal != goal_id:
                        continue
                if isinstance(content_type, str):
                    slot_counts[content_type] = slot_counts.get(content_type, 0) + 1

        # ``wm_goal_active``: true if any "goal" slot exists for this
        # ``goal_id`` (or any, if ``goal_id`` not provided).
        wm_goal_active = slot_counts.get("goal", 0) > 0

        # Capacity for pressure.  Fall back to default if missing.
        capacity = getattr(wm, "capacity", _DEFAULT_WM_CAPACITY)
        if not isinstance(capacity, int) or capacity <= 0:
            capacity = _DEFAULT_WM_CAPACITY

        # Current slot count.  If ``slots`` is None, pressure is 0.
        current = len(slots) if isinstance(slots, dict) else 0
        wm_pressure = min(1.0, max(0.0, current / capacity))

        # Age since last write.  We approximate via
        # ``max(slot.timestamp)`` deltas against ``time.time()`` *only*
        # when the component exposes something tick-shaped.  Real WM
        # tracks ``total_ticks`` but not "ticks since last write";
        # returning 0 is a safe default that Phase 0 records can carry
        # uniformly.  If the WM exposes a ``get_age_ticks`` method,
        # prefer that.
        wm_age_ticks = 0
        if hasattr(wm, "get_age_ticks") and callable(wm.get_age_ticks):
            try:
                candidate = wm.get_age_ticks(goal_id) if goal_id is not None else wm.get_age_ticks()
                if isinstance(candidate, int) and candidate >= 0:
                    wm_age_ticks = candidate
            except Exception as e:
                logger.warning("wm.get_age_ticks raised: %s", e)

        return {
            "wm_state_key": state_key,
            "wm_slot_counts": slot_counts,
            "wm_goal_active": wm_goal_active,
            "wm_age_ticks": wm_age_ticks,
            "wm_pressure": float(wm_pressure),
        }
    except Exception as e:  # noqa: BLE001 — failure isolation is mandatory
        logger.warning("_extract_wm_features raised: %s — falling back to defaults", e)
        return _extract_wm_features(None, goal_id=goal_id)


def _extract_snarc_features(snarc: Optional[Any]) -> Dict[str, Any]:
    """Extract ``snarc_*`` fields from a SNARC-like input.

    Accepts either a mapping ``{'surprise': ..., 'novelty': ..., ...}``
    or an object with those attributes.  Values are clamped:
      - surprise / novelty / arousal / conflict → [0, 1]
      - reward → [-1, 1]

    ``None`` → all zeros.
    """
    if snarc is None:
        return {
            "snarc_surprise": 0.0,
            "snarc_novelty": 0.0,
            "snarc_arousal": 0.0,
            "snarc_reward": 0.0,
            "snarc_conflict": 0.0,
        }

    def _get(key: str, default: float = 0.0) -> float:
        try:
            if isinstance(snarc, dict):
                value = snarc.get(key, default)
            else:
                value = getattr(snarc, key, default)
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    try:
        return {
            "snarc_surprise": _clamp_unit(_get("surprise", 0.0)),
            "snarc_novelty": _clamp_unit(_get("novelty", 0.0)),
            "snarc_arousal": _clamp_unit(_get("arousal", 0.0)),
            "snarc_reward": _clamp_range(_get("reward", 0.0), -1.0, 1.0),
            "snarc_conflict": _clamp_unit(_get("conflict", 0.0)),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("_extract_snarc_features raised: %s — falling back to defaults", e)
        return _extract_snarc_features(None)


def _extract_metabolic_features(metabolic: Optional[Any]) -> Dict[str, Any]:
    """Extract ``metabolic_state`` / ``atp_level`` / ``atp_trend``.

    Accepts a ``MetabolicController``-like object exposing:
      - ``current_state`` (Enum-like; ``.value`` is the str) OR a string
      - ``atp_current`` (0-100)
      - ``atp_trend`` or ``get_atp_trend()`` returning one of
        ``VALID_ATP_TRENDS``

    ``None`` → sensible defaults (wake, 0 ATP, stable).
    """
    if metabolic is None:
        return {
            "metabolic_state": _DEFAULT_METABOLIC_STATE,
            "atp_level": _DEFAULT_ATP_LEVEL,
            "atp_trend": _DEFAULT_ATP_TREND,
        }

    try:
        # current_state — either an Enum or a str.
        raw_state = getattr(metabolic, "current_state", None)
        if raw_state is None and isinstance(metabolic, dict):
            raw_state = metabolic.get("current_state") or metabolic.get("state")
        # Enum → .value, else coerce to str.
        if hasattr(raw_state, "value"):
            state_str = raw_state.value
        else:
            state_str = raw_state
        if not isinstance(state_str, str) or state_str not in VALID_METABOLIC_STATES:
            state_str = _DEFAULT_METABOLIC_STATE

        # ATP level.
        atp = None
        if isinstance(metabolic, dict):
            atp = metabolic.get("atp_current", metabolic.get("atp_level"))
        else:
            atp = getattr(metabolic, "atp_current", getattr(metabolic, "atp_level", None))
        try:
            atp_val = float(atp) if atp is not None else _DEFAULT_ATP_LEVEL
        except (TypeError, ValueError):
            atp_val = _DEFAULT_ATP_LEVEL
        atp_val = _clamp_range(atp_val, 0.0, 100.0)

        # ATP trend.  Prefer a method, then an attribute, then stable.
        trend: Optional[str] = None
        if hasattr(metabolic, "get_atp_trend") and callable(metabolic.get_atp_trend):
            try:
                trend = metabolic.get_atp_trend()
            except Exception as e:
                logger.warning("metabolic.get_atp_trend raised: %s", e)
        if trend is None:
            if isinstance(metabolic, dict):
                trend = metabolic.get("atp_trend")
            else:
                trend = getattr(metabolic, "atp_trend", None)
        if not isinstance(trend, str) or trend not in VALID_ATP_TRENDS:
            trend = _DEFAULT_ATP_TREND

        return {
            "metabolic_state": state_str,
            "atp_level": float(atp_val),
            "atp_trend": trend,
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("_extract_metabolic_features raised: %s — falling back to defaults", e)
        return _extract_metabolic_features(None)


def _extract_episodic_features(
    episodic: Optional[Any],
    *,
    cue: Optional[Any] = None,
    k: int = 5,
) -> Dict[str, Any]:
    """Extract ``recall_*`` fields from an ``EpisodicIndex``-like object.

    Calls ``episodic.recall(cue, k=k)`` if both the component and cue
    are provided; parses the top-k results for:
      - ``recall_count`` : number of results returned (0..k)
      - ``recall_best_similarity`` : top-1 similarity, 0 if empty
      - ``recall_best_outcome`` : reward from best-match episode, None
        if the best match has no reward recorded

    Episodic may not be available before Phase 3 integration.  ``None``
    input OR missing ``cue`` → defaults (0, 0.0, None).
    """
    if episodic is None or cue is None:
        return {
            "recall_count": 0,
            "recall_best_similarity": 0.0,
            "recall_best_outcome": None,
        }

    try:
        if not hasattr(episodic, "recall") or not callable(episodic.recall):
            return _extract_episodic_features(None)

        results = episodic.recall(cue, k=k)
        if not results:
            return {
                "recall_count": 0,
                "recall_best_similarity": 0.0,
                "recall_best_outcome": None,
            }

        count = len(results)
        # Best-match similarity and outcome.  Results are ordered;
        # take the first.  Tolerate both RecallResult and raw dicts.
        best = results[0]
        similarity = getattr(best, "similarity", None)
        if similarity is None and isinstance(best, dict):
            similarity = best.get("similarity")
        try:
            similarity = _clamp_unit(float(similarity)) if similarity is not None else 0.0
        except (TypeError, ValueError):
            similarity = 0.0

        # Outcome is the episode's reward, if any.
        ep = getattr(best, "episode", None)
        if ep is None and isinstance(best, dict):
            ep = best.get("episode")
        outcome: Optional[float] = None
        if ep is not None:
            reward = getattr(ep, "reward", None)
            if reward is None and isinstance(ep, dict):
                reward = ep.get("reward")
            try:
                if reward is not None:
                    outcome = float(reward)
            except (TypeError, ValueError):
                outcome = None

        return {
            "recall_count": int(count),
            "recall_best_similarity": float(similarity),
            "recall_best_outcome": outcome,
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("_extract_episodic_features raised: %s — falling back to defaults", e)
        return _extract_episodic_features(None)


def _extract_cerebellum_features(
    cerebellum: Optional[Any],
    *,
    state: Optional[Any] = None,
) -> Dict[str, Any]:
    """Extract ``habit_available`` / ``habit_confidence`` via cerebellum lookup.

    ``cerebellum.lookup(state)`` returns a ranked list of
    ``HabitMatch`` objects.  The highest-confidence match (sorted by
    the cerebellum itself) supplies ``habit_confidence``.

    ``None`` → no habit available, zero confidence.
    """
    if cerebellum is None or state is None:
        return {
            "habit_available": False,
            "habit_confidence": 0.0,
        }

    try:
        if not hasattr(cerebellum, "lookup") or not callable(cerebellum.lookup):
            return _extract_cerebellum_features(None)

        matches = cerebellum.lookup(state)
        if not matches:
            return {
                "habit_available": False,
                "habit_confidence": 0.0,
            }

        best = matches[0]
        confidence = getattr(best, "confidence", None)
        if confidence is None and isinstance(best, dict):
            confidence = best.get("confidence")
        try:
            confidence = _clamp_unit(float(confidence)) if confidence is not None else 0.0
        except (TypeError, ValueError):
            confidence = 0.0

        return {
            "habit_available": True,
            "habit_confidence": float(confidence),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("_extract_cerebellum_features raised: %s — falling back to defaults", e)
        return _extract_cerebellum_features(None)


def _extract_rpe_features(
    rpe: Optional[Any],
    *,
    state_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract ``prior_invoke`` / ``prior_habit`` / ``prior_noop`` from RPE.

    Per PRD §2.6: the router reads priors via
    ``rpe.get_action_priors(state_key)``, which returns a dict
    ``{action_name: prior_value}``.  The PRD doesn't pin the exact
    action vocabulary — only that 'invoke', 'habit', 'noop' are the
    router's decision classes.  We query those three names and
    normalize.

    RPE is not available before Phase 4 (per sprint doc).  ``None``
    input OR missing ``state_key`` → uniform distribution (1/3 each),
    which is a deliberate "no information" signal.
    """
    if rpe is None or not state_key:
        return {
            "prior_invoke": _UNIFORM_PRIOR,
            "prior_habit": _UNIFORM_PRIOR,
            "prior_noop": _UNIFORM_PRIOR,
        }

    try:
        if not hasattr(rpe, "get_action_priors") or not callable(rpe.get_action_priors):
            return _extract_rpe_features(None)

        priors = rpe.get_action_priors(state_key, ["invoke", "habit", "noop"])
        if not isinstance(priors, dict):
            return _extract_rpe_features(None)

        def _one(name: str) -> float:
            try:
                return _clamp_unit(float(priors.get(name, _UNIFORM_PRIOR)))
            except (TypeError, ValueError):
                return _UNIFORM_PRIOR

        return {
            "prior_invoke": _one("invoke"),
            "prior_habit": _one("habit"),
            "prior_noop": _one("noop"),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("_extract_rpe_features raised: %s — falling back to defaults", e)
        return _extract_rpe_features(None)


def _extract_metacog_features(metacog: Optional[Any]) -> Dict[str, Any]:
    """Extract ``metacog_block_list`` — plugin names currently forbidden.

    Per PRD §2.7: metacog can publish hard constraints (priority >=
    0.9) forbidding the router from invoking specific plugins this
    tick.  We expose them here as a bounded list of plugin names.

    Accepts:
      - A Metacog-like object with ``get_block_list()`` method
      - A Metacog-like object with a ``block_list`` attribute
      - A dict with a ``block_list`` key
      - An iterable of plugin names directly

    ``None`` → empty list.
    """
    if metacog is None:
        return {"metacog_block_list": []}

    try:
        block_list: Iterable[Any]
        if hasattr(metacog, "get_block_list") and callable(metacog.get_block_list):
            block_list = metacog.get_block_list()
        elif hasattr(metacog, "block_list"):
            block_list = metacog.block_list
        elif isinstance(metacog, dict):
            block_list = metacog.get("block_list", [])
        elif isinstance(metacog, (list, tuple, set)):
            block_list = metacog
        else:
            block_list = []

        # Coerce to a list of strings, filter non-strings, cap length.
        cleaned: List[str] = []
        for item in block_list or []:
            if isinstance(item, str):
                cleaned.append(item)
            if len(cleaned) >= _MAX_METACOG_BLOCK_LIST:
                break
        return {"metacog_block_list": cleaned}
    except Exception as e:  # noqa: BLE001
        logger.warning("_extract_metacog_features raised: %s — falling back to defaults", e)
        return {"metacog_block_list": []}


def _extract_sensory_features(sensory: Optional[Any]) -> Dict[str, Any]:
    """Extract ``sensory_modalities`` / ``sensory_novelty`` / ``sensory_urgency``.

    Accepts:
      - A dict with keys {``modalities``, ``novelty``, ``urgency``}
      - An object with those attributes

    ``modalities`` may be ``list[str]`` directly, or ``dict[str, Any]``
    where the keys are modality names (values ignored — presence is
    the signal).  We cap length defensively.

    ``None`` → empty list, zero novelty, zero urgency.
    """
    if sensory is None:
        return {
            "sensory_modalities": [],
            "sensory_novelty": 0.0,
            "sensory_urgency": 0.0,
        }

    try:
        def _get(key: str, default: Any = None) -> Any:
            if isinstance(sensory, dict):
                return sensory.get(key, default)
            return getattr(sensory, key, default)

        raw_modalities = _get("modalities", [])
        modalities: List[str] = []
        if isinstance(raw_modalities, dict):
            for name in raw_modalities.keys():
                if isinstance(name, str):
                    modalities.append(name)
                if len(modalities) >= _MAX_SENSORY_MODALITIES:
                    break
        elif isinstance(raw_modalities, (list, tuple, set)):
            for name in raw_modalities:
                if isinstance(name, str):
                    modalities.append(name)
                if len(modalities) >= _MAX_SENSORY_MODALITIES:
                    break

        novelty = _get("novelty", _DEFAULT_UNIT)
        urgency = _get("urgency", _DEFAULT_UNIT)

        try:
            novelty_f = _clamp_unit(float(novelty)) if novelty is not None else _DEFAULT_UNIT
        except (TypeError, ValueError):
            novelty_f = _DEFAULT_UNIT
        try:
            urgency_f = _clamp_unit(float(urgency)) if urgency is not None else _DEFAULT_UNIT
        except (TypeError, ValueError):
            urgency_f = _DEFAULT_UNIT

        return {
            "sensory_modalities": modalities,
            "sensory_novelty": novelty_f,
            "sensory_urgency": urgency_f,
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("_extract_sensory_features raised: %s — falling back to defaults", e)
        return _extract_sensory_features(None)


def _extract_cartridge_features(cartridge: Optional[Any]) -> Dict[str, Any]:
    """Extract Andy's routing-role cartridge recall stubs.

    Per sprint doc (Track 2): "Cartridge_recall_* fields (PRD §3.1) —
    stubbed to zero for Phase 0; real values arrive once Andy's
    routing-role cartridge is wired."

    This stub always returns the canonical "no recall" triple
    regardless of input — which is correct for Phase 0.  The signature
    still accepts ``cartridge`` so that when Andy's wire lands (next
    track), we can populate this without changing call sites.

    Fields:
      - cartridge_recall_count : 0
      - cartridge_recall_best_similarity : 0.0
      - cartridge_recall_embedding : zeros of length CARTRIDGE_EMBEDDING_DIM
    """
    # Intentionally simple: Phase 0 never populates these from live data.
    # The PluginRegistry/cartridge-integration track will replace this
    # body with a real call to ``router_cartridge.search(...)``.
    del cartridge  # explicitly unused at Phase 0
    return {
        "cartridge_recall_count": 0,
        "cartridge_recall_best_similarity": 0.0,
        "cartridge_recall_embedding": [0.0] * CARTRIDGE_EMBEDDING_DIM,
    }


# ──────────────────────────────────────────────────────────────────────
# Top-level extractor
# ──────────────────────────────────────────────────────────────────────

def extract_router_input(
    wm: Optional[Any] = None,
    snarc: Optional[Any] = None,
    metabolic: Optional[Any] = None,
    episodic: Optional[Any] = None,
    cerebellum: Optional[Any] = None,
    rpe: Optional[Any] = None,
    metacog: Optional[Any] = None,
    plugin_registry: Optional[Any] = None,  # reserved — Phase 0 does not consume
    sensory: Optional[Any] = None,
    *,
    tick: int,
    goal_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    episodic_cue: Optional[Any] = None,
    cerebellum_state: Optional[Any] = None,
    cartridge: Optional[Any] = None,
) -> RouterInput:
    """Build a normative ``RouterInput`` from live kernel state.

    This is the pure-function contract the router will call at every
    consciousness-loop tick.  Every optional component may be
    ``None``; the output is always a valid ``RouterInput`` with the
    same field shape regardless.

    Args:
        wm: WorkingMemory instance (reads ``.slots``, ``.capacity``,
            ``.stable_key(goal_id)``).
        snarc: Mapping or object with keys/attrs ``surprise``,
            ``novelty``, ``arousal``, ``reward``, ``conflict``.
        metabolic: MetabolicController-like with ``current_state``,
            ``atp_current``, optional ``get_atp_trend()``.
        episodic: EpisodicIndex-like with ``recall(cue, k=...)``.
        cerebellum: Cerebellum-like with ``lookup(state)``.
        rpe: RewardPredictionError-like with
            ``get_action_priors(state_key, actions)``.
        metacog: Metacog-like with ``get_block_list()`` or
            ``.block_list``, or an iterable of plugin names.
        plugin_registry: Reserved for Phase 1+ (cartridge wiring,
            plugin-tier lookups).  Not consumed at Phase 0.
        sensory: Mapping/object with ``modalities``, ``novelty``,
            ``urgency``.
        tick: Consciousness-loop cycle counter (required).
        goal_id: Active goal ID from WM.
        timestamp: Wall-clock time.  Defaults to ``time.time()`` if not
            provided — the extractor is the right place to stamp this
            since it's closest to "when we saw the kernel state".
        episodic_cue: An EpisodicCue-like object passed through to
            ``episodic.recall(...)``.  Without a cue, episodic recall
            is a no-op (returns defaults).  Typical construction lives
            in the integration layer (Track 5).
        cerebellum_state: A StateSignature-like object passed to
            ``cerebellum.lookup(...)``.  Without a state, cerebellum
            lookup is a no-op.
        cartridge: Andy's router-role cartridge (Phase 1+ wire).  Not
            consumed at Phase 0 — ``_extract_cartridge_features``
            always returns stubs.

    Returns:
        RouterInput with every field populated from the sub-extractors,
        stamped with ``tick``, ``timestamp``, and ``goal_id``.

    Raises:
        ValueError, TypeError: only if ``RouterInput.__post_init__``
            rejects the assembled payload.  By construction, the
            sub-extractors produce in-range values — so this only
            fires if the caller passed a negative ``tick``, a
            non-numeric ``timestamp``, or similar caller-side bug.
    """
    if plugin_registry is not None:
        # Reserved for future wiring; explicitly no-op at Phase 0.
        # Declared here so Phase 1 can start consuming it without a
        # signature change.  Logging at DEBUG to avoid log noise.
        logger.debug("plugin_registry supplied but not consumed at Phase 0")

    # Stamp context identity first — so if a sub-extractor has a bug
    # and we still want to emit a record, the record at least has a
    # tick/timestamp.
    if timestamp is None:
        timestamp = time.time()

    # Run every sub-extractor.  These are all best-effort; the
    # sub-extractors themselves handle their own failure isolation.
    wm_fields = _extract_wm_features(wm, goal_id=goal_id)
    snarc_fields = _extract_snarc_features(snarc)
    metabolic_fields = _extract_metabolic_features(metabolic)
    episodic_fields = _extract_episodic_features(episodic, cue=episodic_cue)
    cerebellum_fields = _extract_cerebellum_features(cerebellum, state=cerebellum_state)
    rpe_fields = _extract_rpe_features(rpe, state_key=wm_fields["wm_state_key"])
    metacog_fields = _extract_metacog_features(metacog)
    sensory_fields = _extract_sensory_features(sensory)
    cartridge_fields = _extract_cartridge_features(cartridge)

    # Assemble.  Dict order matches RouterInput field order for
    # reader-friendly reading; RouterInput takes kwargs so the order
    # is actually irrelevant, but humans read top-to-bottom.
    return RouterInput(
        tick=int(tick),
        timestamp=float(timestamp),
        goal_id=goal_id,
        **wm_fields,
        **sensory_fields,
        **snarc_fields,
        **metabolic_fields,
        **episodic_fields,
        **cerebellum_fields,
        **rpe_fields,
        **metacog_fields,
        **cartridge_fields,
    )


# ──────────────────────────────────────────────────────────────────────
# Helpers — range clamping (mirrors RouterInput's validators but here
# we SILENTLY clamp rather than raise, because sub-extractors are the
# last line of defense before RouterInput.__post_init__ would reject).
# ──────────────────────────────────────────────────────────────────────

def _clamp_unit(value: float) -> float:
    """Clamp a float to [0, 1].  NaN → 0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN check without math.isnan to avoid an import
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _clamp_range(value: float, lo: float, hi: float) -> float:
    """Clamp a float to [lo, hi].  NaN → midpoint."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return (lo + hi) / 2.0
    if v != v:  # NaN
        return (lo + hi) / 2.0
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


__all__ = [
    "extract_router_input",
    "_extract_wm_features",
    "_extract_snarc_features",
    "_extract_metabolic_features",
    "_extract_episodic_features",
    "_extract_cerebellum_features",
    "_extract_rpe_features",
    "_extract_metacog_features",
    "_extract_sensory_features",
    "_extract_cartridge_features",
]
