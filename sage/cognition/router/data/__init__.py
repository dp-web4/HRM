"""
Router Dataset Pipeline — Phase 0 Track 4.

Append-only JSONL writer + replay reader + SNARC-stratified sampler.
All three are torch-free, JSON-serializable, and failure-isolated: a
pipeline crash MUST never propagate to the consciousness loop.

Spec:
  - PRD §3 (schema), §4.7.D (sampling), §5.4 (governance)
  - Sprint doc router-sprint-1-phase-0.md, Track 4

Usage::

    from sage.cognition.router.data import (
        RouterDatasetWriter, RouterDatasetReader, SnarcStratifiedSampler,
        SCHEMA_VERSION,
    )

    sampler = SnarcStratifiedSampler(seed=42)
    writer = RouterDatasetWriter(base_dir="/var/sage/router", machine="sprout")
    snarc = {"arousal": 0.9, "conflict": 0.2, "reward": 0.1,
             "surprise": 0.5, "novelty": 0.4}
    if sampler.should_keep(snarc):
        writer.append({
            "record_id": "...", "schema_version": "0.1.0",
            "timestamp": 1700000000.0, "machine": "sprout",
            "payload": {...},
        })
    writer.close()
"""

from sage.cognition.router.data.sampling import (
    SnarcStratifiedSampler,
    salience_score,
    SamplingStats,
)
from sage.cognition.router.data.writer import (
    RouterDatasetWriter,
    SCHEMA_VERSION,
)
from sage.cognition.router.data.reader import (
    RouterDatasetReader,
    SUPPORTED_SCHEMA_VERSIONS,
)
from sage.cognition.router.data.pruner import (
    RouterDatasetPruner,
    PruneStats,
    AGE_BRACKETS,
    AGENT_ZERO_MARGIN_PP,
    ACTIVE_WRITE_WINDOW_SECONDS,
    RECOGNIZED_PIN_KINDS,
    PRUNER_VERSION,
)

__all__ = [
    "RouterDatasetWriter",
    "RouterDatasetReader",
    "SnarcStratifiedSampler",
    "SamplingStats",
    "salience_score",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "RouterDatasetPruner",
    "PruneStats",
    "AGE_BRACKETS",
    "AGENT_ZERO_MARGIN_PP",
    "ACTIVE_WRITE_WINDOW_SECONDS",
    "RECOGNIZED_PIN_KINDS",
    "PRUNER_VERSION",
]
