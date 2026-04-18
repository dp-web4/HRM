"""
Thalamic router — schemas + data pipeline.

Phase 0 Track 1 ships the schemas only. Feature extraction (Track 2),
programmatic baseline (Track 3), dataset writer/reader (Track 4), and
consciousness-loop integration (Track 5) follow in subsequent PRs.

Spec: shared-context/arc-agi-3/phase2/brain-arch/thalamic-router-prd.md
Sprint: shared-context/arc-agi-3/phase2/brain-arch/router-sprint-1-phase-0.md
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

__all__ = [
    # Dataclasses
    "Event",
    "RouterInput",
    "RouterOutput",
    "RouterRecord",
    # Enum
    "PluginTier",
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
