"""
Thalamic router — schemas + data pipeline + shadow integration + dashboard.

Phase 0 complete: schemas (Track 1), feature extraction (2), programmatic
baseline (3), data pipeline writer/reader (4), consciousness-loop shadow
integration (5), outcome tracking (6), per-machine deployment (7),
observability dashboard (8), SNARC-driven pruning (9).

Spec: phase2/brain-arch/thalamic-router-prd.md
Sprint: phase2/brain-arch/router-sprint-1-phase-0.md
"""

from sage.cognition.router.events import (
    Event,
    VALID_EVENT_KINDS,
    ROUTER_KINDS,
    WM_COMPATIBLE_KINDS,
)
from sage.cognition.router.inputs import (
    RouterInput,
    CARTRIDGE_EMBEDDING_DIM,
    VALID_ATP_TRENDS,
    VALID_METABOLIC_STATES,
)
from sage.cognition.router.outputs import (
    RouterOutput,
    VALID_ACTIONS,
    VALID_RATIONALE_CODES,
)
from sage.cognition.router.record import (
    RouterRecord,
    ROUTER_SCHEMA_VERSION,
)
from sage.cognition.router.tiers import PluginTier
from sage.cognition.router.shadow import (
    RouterShadowHook,
    SHADOW_ENV_VAR,
    is_shadow_enabled,
)
from sage.cognition.router.outcome import (
    OutcomeTracker,
    OUTCOME_SCHEMA_VERSION,
    TRAJECTORY_TICKS,
    STATUS_COMPLETE,
    STATUS_INCOMPLETE,
    STATUS_FAILED,
)
from sage.cognition.router.dashboard import (
    DashboardBuilder,
    DashboardMetrics,
    DecisionStats,
    MachineMetrics,
    SnarcDimStats,
    render_json,
    render_markdown,
    AGENT_ZERO_MARGIN_PP,
    DASHBOARD_SCHEMA_VERSION,
    SNARC_DIMENSIONS,
)

__all__ = [
    # Dataclasses
    "Event",
    "RouterInput",
    "RouterOutput",
    "RouterRecord",
    # Enum
    "PluginTier",
    # Shadow hook (Track 5)
    "RouterShadowHook",
    "SHADOW_ENV_VAR",
    "is_shadow_enabled",
    # Outcome tracker (Track 6)
    "OutcomeTracker",
    "OUTCOME_SCHEMA_VERSION",
    "TRAJECTORY_TICKS",
    "STATUS_COMPLETE",
    "STATUS_INCOMPLETE",
    "STATUS_FAILED",
    # Dashboard (Track 8)
    "DashboardBuilder",
    "DashboardMetrics",
    "DecisionStats",
    "MachineMetrics",
    "SnarcDimStats",
    "render_json",
    "render_markdown",
    "AGENT_ZERO_MARGIN_PP",
    "DASHBOARD_SCHEMA_VERSION",
    "SNARC_DIMENSIONS",
    # Constants
    "ROUTER_SCHEMA_VERSION",
    "CARTRIDGE_EMBEDDING_DIM",
    "VALID_ACTIONS",
    "VALID_ATP_TRENDS",
    "VALID_EVENT_KINDS",
    "VALID_METABOLIC_STATES",
    "VALID_RATIONALE_CODES",
    "ROUTER_KINDS",
    "WM_COMPATIBLE_KINDS",
]
