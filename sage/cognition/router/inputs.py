#!/usr/bin/env python3
"""
RouterInput — normative schema consumed by the thalamic router.
===============================================================

Every consciousness-loop tick, the router receives a ``RouterInput`` built
from current kernel state (WM, SNARC, metabolic, episodic, cerebellum,
RPE, metacog, sensory, cartridge). The schema is FIXED — PRD §3.1 is the
binding contract; new fields require a coordinated schema-version bump.

Feature extraction (turning live kernel state into a ``RouterInput``)
lives in ``feature_extraction.py`` and is Track 2. This module only
defines the shape + validation + serialization.

Spec: phase2/brain-arch/thalamic-router-prd.md §3.1
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────
# Constants — closed vocabularies referenced by validation
# ──────────────────────────────────────────────────────────────────────

VALID_METABOLIC_STATES = {"wake", "focus", "rest", "dream", "crisis"}
VALID_ATP_TRENDS = {"rising", "stable", "falling"}

# Phase 0 stub — Andy's routing-role cartridge is not yet wired
# (PRD §2.9, §4.9). Fields default to "no recall" so Phase 0 records
# still carry the field and are replay-compatible once the cartridge
# ships. Dimension matches nomic-embed-text (PRD §2.9).
CARTRIDGE_EMBEDDING_DIM = 768


def _zero_embedding() -> List[float]:
    """Default factory for the cartridge_recall_embedding stub."""
    return [0.0] * CARTRIDGE_EMBEDDING_DIM


# ──────────────────────────────────────────────────────────────────────
# RouterInput
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RouterInput:
    """Structured snapshot of kernel state at one consciousness-loop tick.

    Fields, order, and names track PRD §3.1 exactly. The schema is
    deterministic-serializable: every field is JSON-safe, lists are
    bounded by the plugin registry (``sensory_modalities``,
    ``metacog_block_list``) or fixed-dimension (``cartridge_recall_embedding``).

    Validation is intentionally permissive at construction time:
    ``__post_init__`` enforces type + range where doing so is cheap, but
    does NOT reject e.g. a too-high ``recall_count`` (since the real
    bound depends on the caller's ``k``). Strict validation is the
    caller's responsibility — the router itself treats garbage in as
    "coerce to noop + warn" per PRD §3.3 rule 5 (enforced on the output
    side, not here).
    """

    # ── Context identity ─────────────────────────────────────────────
    tick: int
    timestamp: float
    goal_id: Optional[str]

    # ── WM features (derived from wm.get_context) ────────────────────
    wm_state_key: str                       # wm.stable_key(goal_id), 16-char hex
    wm_slot_counts: Dict[str, int]
    wm_goal_active: bool
    wm_age_ticks: int
    wm_pressure: float                      # 0-1: slot count / capacity

    # ── Sensory features (current tick) ──────────────────────────────
    sensory_modalities: List[str]
    sensory_novelty: float                  # 0-1
    sensory_urgency: float                  # 0-1

    # ── SNARC ────────────────────────────────────────────────────────
    snarc_surprise: float                   # 0-1
    snarc_novelty: float                    # 0-1
    snarc_arousal: float                    # 0-1
    snarc_reward: float                     # -1..1
    snarc_conflict: float                   # 0-1

    # ── Metabolic ────────────────────────────────────────────────────
    metabolic_state: str                    # enum (VALID_METABOLIC_STATES)
    atp_level: float                        # 0-100
    atp_trend: str                          # enum (VALID_ATP_TRENDS)

    # ── Episodic recall (top-k similar episodes) ─────────────────────
    recall_count: int
    recall_best_similarity: float           # 0-1
    recall_best_outcome: Optional[float]    # reward from best-match episode

    # ── Habit availability ───────────────────────────────────────────
    habit_available: bool
    habit_confidence: float                 # 0-1

    # ── RPE priors over decision types ───────────────────────────────
    prior_invoke: float                     # 0-1
    prior_habit: float                      # 0-1
    prior_noop: float                       # 0-1

    # ── Metacog constraints ──────────────────────────────────────────
    metacog_block_list: List[str]

    # ── Cartridge recall (Phase 0 stubs per PRD §2.9 / §4.9) ─────────
    # Defaults describe "no recall" — this is the correct value for
    # Phase 0 before Andy's routing-role cartridge is wired. Once the
    # cartridge ships, the feature extractor populates these from
    # ``router_cartridge.search(situation_text, k=3, role='routing')``.
    cartridge_recall_count: int = 0
    cartridge_recall_best_similarity: float = 0.0
    cartridge_recall_embedding: List[float] = field(default_factory=_zero_embedding)

    # ──────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        # Type/shape guards that are cheap and catch the obvious bugs.
        # Range checks on [0,1] floats are enforced because they're
        # binding contracts (PRD §3.1 comments).
        if not isinstance(self.tick, int) or self.tick < 0:
            raise ValueError(f"tick must be non-negative int, got {self.tick!r}")
        if not isinstance(self.timestamp, (int, float)):
            raise TypeError(f"timestamp must be numeric, got {type(self.timestamp).__name__}")

        if not isinstance(self.wm_state_key, str) or not self.wm_state_key:
            raise ValueError("wm_state_key must be a non-empty string")
        if not isinstance(self.wm_slot_counts, dict):
            raise TypeError("wm_slot_counts must be dict[str, int]")
        for k, v in self.wm_slot_counts.items():
            if not isinstance(k, str) or not isinstance(v, int):
                raise TypeError(f"wm_slot_counts entries must be str→int, got {k!r}→{v!r}")

        _check_unit("wm_pressure", self.wm_pressure)
        if not isinstance(self.wm_age_ticks, int) or self.wm_age_ticks < 0:
            raise ValueError(f"wm_age_ticks must be non-negative int, got {self.wm_age_ticks!r}")

        if not isinstance(self.sensory_modalities, list):
            raise TypeError("sensory_modalities must be list[str]")
        for m in self.sensory_modalities:
            if not isinstance(m, str):
                raise TypeError(f"sensory_modalities entries must be str, got {m!r}")
        _check_unit("sensory_novelty", self.sensory_novelty)
        _check_unit("sensory_urgency", self.sensory_urgency)

        _check_unit("snarc_surprise", self.snarc_surprise)
        _check_unit("snarc_novelty", self.snarc_novelty)
        _check_unit("snarc_arousal", self.snarc_arousal)
        _check_range("snarc_reward", self.snarc_reward, -1.0, 1.0)
        _check_unit("snarc_conflict", self.snarc_conflict)

        if self.metabolic_state not in VALID_METABOLIC_STATES:
            raise ValueError(
                f"metabolic_state must be in {sorted(VALID_METABOLIC_STATES)}, "
                f"got {self.metabolic_state!r}"
            )
        _check_range("atp_level", self.atp_level, 0.0, 100.0)
        if self.atp_trend not in VALID_ATP_TRENDS:
            raise ValueError(
                f"atp_trend must be in {sorted(VALID_ATP_TRENDS)}, "
                f"got {self.atp_trend!r}"
            )

        if not isinstance(self.recall_count, int) or self.recall_count < 0:
            raise ValueError(f"recall_count must be non-negative int, got {self.recall_count!r}")
        _check_unit("recall_best_similarity", self.recall_best_similarity)
        if self.recall_best_outcome is not None:
            if not isinstance(self.recall_best_outcome, (int, float)):
                raise TypeError("recall_best_outcome must be numeric or None")

        if not isinstance(self.habit_available, bool):
            raise TypeError("habit_available must be bool")
        _check_unit("habit_confidence", self.habit_confidence)

        _check_unit("prior_invoke", self.prior_invoke)
        _check_unit("prior_habit", self.prior_habit)
        _check_unit("prior_noop", self.prior_noop)

        if not isinstance(self.metacog_block_list, list):
            raise TypeError("metacog_block_list must be list[str]")
        for p in self.metacog_block_list:
            if not isinstance(p, str):
                raise TypeError(f"metacog_block_list entries must be str, got {p!r}")

        if not isinstance(self.cartridge_recall_count, int) or self.cartridge_recall_count < 0:
            raise ValueError(
                f"cartridge_recall_count must be non-negative int, got {self.cartridge_recall_count!r}"
            )
        _check_unit("cartridge_recall_best_similarity", self.cartridge_recall_best_similarity)
        if not isinstance(self.cartridge_recall_embedding, list):
            raise TypeError("cartridge_recall_embedding must be list[float]")
        if len(self.cartridge_recall_embedding) != CARTRIDGE_EMBEDDING_DIM:
            raise ValueError(
                f"cartridge_recall_embedding must have length {CARTRIDGE_EMBEDDING_DIM}, "
                f"got {len(self.cartridge_recall_embedding)}"
            )
        for v in self.cartridge_recall_embedding:
            if not isinstance(v, (int, float)):
                raise TypeError("cartridge_recall_embedding entries must be numeric")

    # ──────────────────────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict representation. Fields preserved in
        declaration order."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouterInput":
        """Reconstruct from a dict produced by ``to_dict``.

        Unknown keys are ignored (forward compatibility). Missing
        optional fields fall back to defaults.
        """
        # Only pass keys we know about — this is the schema-version-aware
        # read path. Older payloads missing cartridge_recall_* get defaults.
        known = {
            "tick", "timestamp", "goal_id",
            "wm_state_key", "wm_slot_counts", "wm_goal_active",
            "wm_age_ticks", "wm_pressure",
            "sensory_modalities", "sensory_novelty", "sensory_urgency",
            "snarc_surprise", "snarc_novelty", "snarc_arousal",
            "snarc_reward", "snarc_conflict",
            "metabolic_state", "atp_level", "atp_trend",
            "recall_count", "recall_best_similarity", "recall_best_outcome",
            "habit_available", "habit_confidence",
            "prior_invoke", "prior_habit", "prior_noop",
            "metacog_block_list",
            "cartridge_recall_count",
            "cartridge_recall_best_similarity",
            "cartridge_recall_embedding",
        }
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _check_unit(name: str, value: Any) -> None:
    """Assert ``value`` is a number in [0, 1]."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _check_range(name: str, value: Any, lo: float, hi: float) -> None:
    """Assert ``value`` is a number in [lo, hi]."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    if not lo <= float(value) <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value!r}")
